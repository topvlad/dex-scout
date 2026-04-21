import os
import socket
import time
import traceback
from typing import Any, Callable, Dict

os.environ["DEX_SCOUT_WORKER_MODE"] = "1"

import app

SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "10"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "10"))
JOB_LOCK_TTL_SEC = max(60, int(os.getenv("JOB_LOCK_TTL_SEC", "900")))
JOB_HEARTBEAT_NAME = "job_dispatch"


def _run_scan_cycle() -> Dict[str, Any]:
    return app.maybe_run_rotating_scanner(
        seeds_raw=SCANNER_SEEDS,
        max_items=SCANNER_MAX_ITEMS,
        use_birdeye_trending=USE_BIRDEYE_TRENDING,
        birdeye_limit=BIRDEYE_LIMIT,
    )


def _run_monitor_cycle() -> Dict[str, Any]:
    monitoring_rows = app.load_monitoring()
    portfolio_rows = [
        r for r in app.load_portfolio()
        if str(r.get("active", "1")).strip() == "1"
    ]
    return app.run_priority_scanner_cycle(
        monitoring_rows=monitoring_rows,
        portfolio_rows=portfolio_rows,
        max_scans=3,
    )


def _run_notify_cycle() -> Dict[str, Any]:
    monitoring_rows = app.load_monitoring()
    portfolio_rows = app.load_portfolio()
    active_monitoring_rows = app.build_active_monitoring_rows(monitoring_rows)
    active_portfolio_rows = [
        r for r in portfolio_rows
        if str(r.get("active", "1")).strip() == "1"
    ]
    scan_state = app.scanner_state_load() or {}
    return app.run_auto_notifications(
        scan_state=scan_state,
        monitoring_rows=active_monitoring_rows,
        portfolio_rows=active_portfolio_rows,
        cycle_context={"candidate_path": "job_dispatch", "fallback_reason": ""},
        trigger_model="job_notify_cycle",
    )


def _run_digest_cycle() -> Dict[str, Any]:
    return app.trigger_digest_notification(trigger_source="job_digest_cycle")


def _run_outcome_cycle() -> Dict[str, Any]:
    return app.evaluate_outcome_journals()


JOB_DISPATCH: Dict[str, Callable[[], Dict[str, Any]]] = {
    "scan_cycle": _run_scan_cycle,
    "monitor_cycle": _run_monitor_cycle,
    "notify_cycle": _run_notify_cycle,
    "digest_cycle": _run_digest_cycle,
    "outcome_cycle": _run_outcome_cycle,
}


def run_job_mode(job_mode: str) -> int:
    mode = str(job_mode or "").strip().lower()
    runner = JOB_DISPATCH.get(mode)
    if runner is None:
        reason = f"unknown_job_mode:{mode or 'empty'}"
        app.update_worker_runtime_state(
            updates={
                "worker_status": "job_mode_invalid",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_job_mode": mode,
                "last_job_status": "invalid_mode",
            }
        )
        app.update_job_heartbeat(
            job_name=JOB_HEARTBEAT_NAME,
            job_mode=mode,
            status="invalid_mode",
            meta={"reason": reason, "allowed_modes": list(JOB_DISPATCH.keys())},
        )
        print(f"[worker] safe-fail: {reason}", flush=True)
        return 2

    owner = f"{socket.gethostname()}:{os.getpid()}"
    lock_key = f"job_mode:{mode}"
    lock_status = app.acquire_lock(lock_key=lock_key, owner=owner, ttl_sec=JOB_LOCK_TTL_SEC)
    if not lock_status.get("ok"):
        code = str(lock_status.get("code") or "lock_failed")
        reason = f"{code}:{lock_status.get('detail') or lock_status.get('message') or ''}".strip(":")
        app.update_worker_runtime_state(
            updates={
                "worker_status": "job_skipped_locked",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_job_mode": mode,
                "last_job_status": "skipped_locked",
            }
        )
        app.update_job_heartbeat(
            job_name=JOB_HEARTBEAT_NAME,
            job_mode=mode,
            status="skipped_locked",
            meta={"lock_key": lock_key, "owner": owner, "lock_status": lock_status},
        )
        print(f"[worker] skipped by lock mode={mode} code={code}", flush=True)
        return 3

    start_epoch = time.time()
    start_ts = app.now_utc_str()
    app.update_worker_runtime_state(
        updates={
            "worker_status": "job_running",
            "worker_id": owner,
            "worker_role": "job_dispatch",
            "last_job_mode": mode,
            "last_job_status": "running",
            "last_job_started_ts": start_ts,
            "last_job_started_epoch": start_epoch,
            "last_error_reason": "",
        }
    )
    app.update_job_heartbeat(
        job_name=JOB_HEARTBEAT_NAME,
        job_mode=mode,
        status="started",
        meta={"lock_key": lock_key, "owner": owner},
    )

    try:
        result = runner()
        end_epoch = time.time()
        duration_sec = round(max(0.0, end_epoch - start_epoch), 3)
        app.update_worker_runtime_state(
            updates={
                "worker_status": "job_finished",
                "last_job_mode": mode,
                "last_job_status": "finished",
                "last_job_finished_ts": app.now_utc_str(),
                "last_job_finished_epoch": end_epoch,
                "last_job_duration_sec": duration_sec,
            }
        )
        app.update_job_heartbeat(
            job_name=JOB_HEARTBEAT_NAME,
            job_mode=mode,
            status="finished",
            meta={"duration_sec": duration_sec, "result": result},
        )
        print(f"[worker] mode={mode} finished result={result}", flush=True)
        return 0
    except Exception as exc:
        reason = f"job_mode_exception:{mode}:{type(exc).__name__}:{exc}"
        app.update_worker_runtime_state(
            updates={
                "worker_status": "job_failed",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_job_mode": mode,
                "last_job_status": "failed",
                "last_job_finished_ts": app.now_utc_str(),
            }
        )
        app.update_job_heartbeat(
            job_name=JOB_HEARTBEAT_NAME,
            job_mode=mode,
            status="failed",
            meta={"reason": reason},
        )
        print(f"[worker] mode={mode} failed reason={reason}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        app.release_lock(lock_key=lock_key, owner=owner)


def main() -> int:
    app.ensure_storage()
    contract = app.check_runtime_contract()
    if not contract.get("ok"):
        reason = f"runtime_contract_error:{contract.get('code')}"
        app.update_worker_runtime_state(
            updates={
                "worker_status": "runtime_contract_error",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
            }
        )
        app.update_job_heartbeat(
            job_name=JOB_HEARTBEAT_NAME,
            job_mode="dispatch_init",
            status="runtime_contract_error",
            meta={"detail": contract.get("failures")},
        )
        print(
            "[worker] runtime contract missing; refusing to start "
            f"code={contract.get('code')} detail={contract.get('failures')}",
            flush=True,
        )
        return 1

    job_mode = os.getenv("JOB_MODE", "").strip().lower()
    return run_job_mode(job_mode=job_mode)


if __name__ == "__main__":
    raise SystemExit(main())
