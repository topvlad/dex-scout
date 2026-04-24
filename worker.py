import os
import socket
import time
import traceback
from importlib import import_module
from typing import Any, Callable, Dict, Optional

os.environ["DEX_SCOUT_WORKER_MODE"] = "1"

REQUIRED_ENV_VARS = (
    "JOB_MODE",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "TG_BOT_TOKEN",
    "TG_CHAT_ID",
)

RUNTIME_REQUIRED_ENTITIES = (
    "runtime_state",
    "locks",
    "tg_state",
    "job_heartbeats",
)

RUNTIME_OPTIONAL_ENTITIES = ("job_runs",)


def _fail_fast(reason: str, exit_code: int) -> int:
    print(f"[worker] fail-fast: {reason}", flush=True)
    return int(exit_code)


def _load_app():
    try:
        return import_module("app")
    except Exception as exc:
        raise RuntimeError(f"import_error:{type(exc).__name__}:{exc}") from exc


def _missing_required_env() -> list[str]:
    missing = []
    for key in REQUIRED_ENV_VARS:
        if not str(os.getenv(key, "")).strip():
            missing.append(key)
    return missing


app: Optional[Any] = None


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", "")
SCANNER_MAX_ITEMS = _env_int("SCANNER_MAX_ITEMS", 10)
USE_BIRDEYE_TRENDING = _env_bool("USE_BIRDEYE_TRENDING", True)
BIRDEYE_LIMIT = _env_int("BIRDEYE_LIMIT", 10)
JOB_LOCK_TTL_SEC = max(60, _env_int("JOB_LOCK_TTL_SEC", 900))
JOB_STALE_RUN_SEC = max(JOB_LOCK_TTL_SEC, _env_int("JOB_STALE_RUN_SEC", JOB_LOCK_TTL_SEC * 2))


def _run_scan_cycle() -> Dict[str, Any]:
    assert app is not None
    seeds_raw = SCANNER_SEEDS or str(app.DEFAULT_SEEDS)
    return app.maybe_run_rotating_scanner(
        seeds_raw=seeds_raw,
        max_items=SCANNER_MAX_ITEMS,
        use_birdeye_trending=USE_BIRDEYE_TRENDING,
        birdeye_limit=BIRDEYE_LIMIT,
    )


def _run_monitor_cycle() -> Dict[str, Any]:
    assert app is not None
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
    assert app is not None
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
    assert app is not None
    return app.trigger_digest_notification(trigger_source="job_digest_cycle")


def _run_outcome_cycle() -> Dict[str, Any]:
    assert app is not None
    return app.evaluate_outcome_journals()


JOB_DISPATCH: Dict[str, Callable[[], Dict[str, Any]]] = {
    "scan_cycle": _run_scan_cycle,
    "monitor_cycle": _run_monitor_cycle,
    "notify_cycle": _run_notify_cycle,
    "digest_cycle": _run_digest_cycle,
    "outcome_cycle": _run_outcome_cycle,
}


def run_job_mode(job_mode: str) -> int:
    assert app is not None
    mode = str(job_mode or "").strip().lower()
    lock_key = None
    runner = JOB_DISPATCH.get(mode)

    if runner is None:
        reason = f"unknown_job_mode:{mode or 'empty'}"
        app.update_worker_runtime_state(
            updates={
                "worker_status": "job_mode_invalid",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_job_mode": mode,
                "last_job_status": "invalid",
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode=mode,
            status="invalid_mode",
            meta={"reason": reason, "allowed_modes": list(JOB_DISPATCH.keys())},
        )
        print(f"[worker] safe-fail: {reason}", flush=True)
        return 2

    owner = f"{socket.gethostname()}:{os.getpid()}"
    runtime = app.get_worker_runtime_state()
    prev_status = str(runtime.get("last_job_status") or "").strip().lower()
    prev_mode = str(runtime.get("last_job_mode") or "").strip().lower()
    prev_started_epoch = float(app.parse_float(runtime.get("last_job_started_epoch", 0.0), 0.0))
    age_sec = max(0.0, time.time() - prev_started_epoch) if prev_started_epoch > 0 else 0.0
    stale_run = bool(prev_status == "running" and prev_mode == mode and age_sec >= JOB_STALE_RUN_SEC)

    if prev_status == "running" and prev_mode == mode and age_sec < JOB_STALE_RUN_SEC:
        reason = f"duplicate_run_guard:mode={mode}:age_sec={int(age_sec)}"
        app.update_worker_runtime_state(
            updates={
                "worker_status": "job_skipped_duplicate_run",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_job_mode": mode,
                "last_job_status": "skipped_duplicate_run",
                "last_job_reason": reason,
                "last_lock_code": "duplicate_run_guard",
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode=mode,
            status="duplicate_run",
            meta={"reason": reason, "age_sec": int(age_sec), "stale_threshold_sec": JOB_STALE_RUN_SEC},
        )
        print(f"[worker] duplicate run blocked mode={mode} age_sec={int(age_sec)}", flush=True)
        return 3

    # FIX #1: initialise lock_key to None before any branching so the
    # finally block never hits NameError on early returns.
    lock_key: Optional[str] = None

    lock_key = f"job_mode:{mode}"

    # FIX #6: only write stale_run heartbeat AFTER we know we will attempt
    # the lock — avoids misleading "stale_run_detected" when lock then fails.
    lock_status = app.acquire_lock(lock_key=lock_key, owner=owner, ttl_sec=JOB_LOCK_TTL_SEC)

    if lock_status.get("ok") and bool(lock_status.get("stale_replaced")):
        app.update_worker_runtime_state(
            updates={
                "last_stale_lock_ts": app.now_utc_str(),
                "last_job_reason": (
                    f"stale_lock_replaced:{lock_key}:age_sec={int(app.parse_float(lock_status.get('stale_age_sec'), 0.0))}"
                ),
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode=mode,
            status="stale_lock_replaced",
            meta={"lock_key": lock_key, "lock_status": lock_status},
        )

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
                "last_job_reason": reason,
                "last_lock_code": code,
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode=mode,
            status="skipped_locked",
            meta={"lock_key": lock_key, "owner": owner, "lock_status": lock_status},
        )
        print(f"[worker] skipped by lock mode={mode} code={code}", flush=True)
        return 3

    if stale_run and lock_status.get("ok"):
        # Write the stale-run note only when lock acquisition succeeded.
        app.update_worker_runtime_state(
            updates={
                "last_stale_run_ts": app.now_utc_str(),
                "last_job_reason": f"stale_run_detected:mode={mode}:age_sec={int(age_sec)}",
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode=mode,
            status="stale_run_detected",
            meta={"age_sec": int(age_sec), "stale_threshold_sec": JOB_STALE_RUN_SEC},
        )

    start_epoch = time.time()
    start_ts = app.now_utc_str()
    app.update_worker_runtime_state(
        updates={
            "worker_status": "job_running",
            "worker_id": owner,
            "worker_role": "job_dispatch",
            "last_job_mode": mode,
            "last_job_status": "running",
            "last_job_reason": "",
            "last_lock_code": "",
            "last_job_started_ts": start_ts,
            "last_job_started_epoch": start_epoch,
            "last_error_reason": "",
        }
    )
    app.update_job_heartbeat(
        job_name="job_dispatch",
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
                "last_job_reason": "ok",
                "last_job_finished_ts": app.now_utc_str(),
                "last_job_finished_epoch": end_epoch,
                "last_job_duration_sec": duration_sec,
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
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
                "last_job_reason": reason,
                "last_job_finished_ts": app.now_utc_str(),
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode=mode,
            status="failed",
            meta={"reason": reason},
        )
        print(f"[worker] mode={mode} failed reason={reason}", flush=True)
        traceback.print_exc()
        return 1

    finally:
        if lock_key is not None:
            app.release_lock(lock_key=lock_key, owner=owner)


def main() -> int:
    global app
    app = _load_app()

    missing_env = _missing_required_env()
    if missing_env:
        return _fail_fast(f"missing_env:{','.join(missing_env)}", 12)

    app.ensure_storage()

    contract = app.check_runtime_contract()
    if not contract.get("ok"):
        failures = contract.get("failures")
        failure_tables = set()
        if isinstance(failures, list):
            for failure in failures:
                if isinstance(failure, dict):
                    failure_tables.add(str(failure.get("table") or "").strip())

        required_entities = set(RUNTIME_REQUIRED_ENTITIES)
        missing_entities = sorted(entity for entity in required_entities if entity in failure_tables)
        missing_entities.extend(
            sorted(
                entity for entity in failure_tables
                if entity and entity not in required_entities and entity not in RUNTIME_OPTIONAL_ENTITIES
            )
        )
        optional_missing = sorted(entity for entity in RUNTIME_OPTIONAL_ENTITIES if entity in failure_tables)
        if optional_missing:
            missing_entities.extend(optional_missing)

        entities_detail = ",".join(missing_entities) or "unknown"
        reason = f"runtime_contract_error:{contract.get('code')}:{entities_detail}"
        app.update_worker_runtime_state(
            updates={
                "worker_status": "runtime_contract_error",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
            }
        )
        app.update_job_heartbeat(
            job_name="job_dispatch",
            job_mode="bootstrap",
            status="runtime_contract_error",
            meta={"detail": contract.get("failures")},
        )
        print(
            "[worker] runtime contract missing; refusing to start "
            f"code={contract.get('code')} missing_entities={missing_entities} detail={contract.get('failures')}",
            flush=True,
        )
        return 14

    job_mode = os.getenv("JOB_MODE", "").strip().lower()
    return run_job_mode(job_mode=job_mode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        msg = str(exc)
        if msg.startswith("import_error:"):
            raise SystemExit(_fail_fast(msg, 11))
        raise
