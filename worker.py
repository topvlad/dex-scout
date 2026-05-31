import os
import signal
import socket
import time
import traceback
import uuid
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Callable, Dict, Optional

# ============================================================
# PRODUCTION ENTRY POINT — one-shot job dispatcher
# Usage: JOB_MODE=scan_cycle python worker.py
# Available JOB_MODE values: scan_cycle, monitor_cycle,
#   notify_cycle, digest_cycle, outcome_cycle
# Locking: acquire_lock() per job mode prevents duplicate runs.
# ============================================================

os.environ["DEX_SCOUT_WORKER_MODE"] = "1"

BASE_REQUIRED_ENV_VARS = (
    "JOB_MODE",
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
    backend = str(os.getenv("STORAGE_BACKEND", "supabase")).strip().lower()
    if backend not in {"supabase", "d1", "local"}:
        backend = "supabase"
    required = list(BASE_REQUIRED_ENV_VARS)
    if backend == "supabase":
        required += ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
    elif backend == "d1":
        required += ["D1_PROXY_URL", "D1_PROXY_TOKEN"]
    missing = []
    for key in required:
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
JOB_STALE_RUN_SEC = max(JOB_LOCK_TTL_SEC + 1, _env_int("JOB_STALE_RUN_SEC", JOB_LOCK_TTL_SEC * 2))
RUNTIME_UPDATE_LOCK_KEY = "worker_runtime_update"
RUNTIME_UPDATE_LOCK_TTL_SEC = 15
RUNTIME_UPDATE_LOCK_RETRIES = 40
RUNTIME_UPDATE_LOCK_SLEEP_SEC = 0.05

if JOB_LOCK_TTL_SEC >= JOB_STALE_RUN_SEC:
    raise RuntimeError(
        f"invalid_runtime_config: JOB_LOCK_TTL_SEC({JOB_LOCK_TTL_SEC}) must be < "
        f"JOB_STALE_RUN_SEC({JOB_STALE_RUN_SEC})"
    )


class _TerminatedSignal(RuntimeError):
    pass


def _heartbeat_job_name(mode: str) -> str:
    return f"job_dispatch:{mode}"


def _runtime_modes(runtime: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = runtime.get("modes")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = dict(value)
    return out


def _runtime_mode_state(runtime: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return dict(_runtime_modes(runtime).get(mode, {}))


def _update_worker_runtime_with_mode_state(
    *,
    mode: str,
    owner: str,
    mode_updates: Optional[Dict[str, Any]] = None,
    global_updates: Optional[Dict[str, Any]] = None,
    increments: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    assert app is not None
    lock_status: Dict[str, Any] = {"ok": False, "code": "not_attempted"}
    for _ in range(RUNTIME_UPDATE_LOCK_RETRIES):
        lock_status = app.acquire_lock(
            lock_key=RUNTIME_UPDATE_LOCK_KEY,
            owner=owner,
            ttl_sec=RUNTIME_UPDATE_LOCK_TTL_SEC,
        )
        if lock_status.get("ok"):
            break
        time.sleep(RUNTIME_UPDATE_LOCK_SLEEP_SEC)
    if not lock_status.get("ok"):
        print(
            "[worker] runtime_update_lock_failed "
            f"mode={mode} owner={owner} code={lock_status.get('code')}",
            flush=True,
        )
        return {}
    try:
        runtime = app.get_worker_runtime_state()
        modes = _runtime_modes(runtime)
        mode_state = dict(modes.get(mode, {}))
        if mode_updates:
            mode_state.update(mode_updates)
        modes[mode] = mode_state
        updates: Dict[str, Any] = {"modes": modes}
        if global_updates:
            updates.update(global_updates)
        return app.update_worker_runtime_state(updates=updates, increments=increments)
    finally:
        app.release_lock(lock_key=RUNTIME_UPDATE_LOCK_KEY, owner=owner)


def _finalize_runtime_if_token_matches(
    *,
    mode: str,
    owner: str,
    run_id: str,
    updates: Dict[str, Any],
    global_updates: Optional[Dict[str, Any]] = None,
    increments: Optional[Dict[str, int]] = None,
) -> bool:
    assert app is not None
    runtime = app.get_worker_runtime_state()
    mode_state = _runtime_mode_state(runtime, mode)
    current_run_id = str(mode_state.get("run_id") or "").strip()
    current_token = str(mode_state.get("job_start_token") or "").strip()
    if current_run_id != run_id or current_token != run_id:
        print(
            f"[worker] finalize token mismatch mode={mode} run_id={run_id} "
            f"current_mode_run_id={current_run_id} current_token={current_token}",
            flush=True,
        )
        return False
    _update_worker_runtime_with_mode_state(
        mode=mode,
        owner=owner,
        mode_updates=updates,
        global_updates={"last_job_mode": mode, **dict(global_updates or {})},
        increments=increments,
    )
    return True


def _run_scan_cycle() -> Dict[str, Any]:
    assert app is not None
    runtime_before = app.get_worker_runtime_state() if hasattr(app, "get_worker_runtime_state") else {}
    scan_request = runtime_before.get("scan_request") if isinstance(runtime_before.get("scan_request"), dict) else {}
    pending = bool(runtime_before.get("scan_request_pending"))
    params = scan_request.get("params") if isinstance(scan_request.get("params"), dict) else {}
    requested_chain = str(params.get("chain") or "").strip().lower()
    chain_reason = ""
    if requested_chain:
        chain_reason = (
            f"ignored_requested_chain:{requested_chain}:scan_cycle_uses_rotation"
        )
    stale_pending = False
    pending_ts = app.parse_ts(runtime_before.get("scan_request_ts")) if hasattr(app, "parse_ts") else None
    if pending and pending_ts:
        stale_pending = (datetime.now(timezone.utc) - pending_ts).total_seconds() > (90 * 60)
    if pending and not stale_pending:
        seeds_raw = str(params.get("seeds_raw") or SCANNER_SEEDS or str(app.DEFAULT_SEEDS))
        max_items = int(app.parse_float(params.get("max_items", SCANNER_MAX_ITEMS), SCANNER_MAX_ITEMS))
        use_birdeye_trending = bool(params.get("use_birdeye_trending", USE_BIRDEYE_TRENDING))
        birdeye_limit = int(app.parse_float(params.get("birdeye_limit", BIRDEYE_LIMIT), BIRDEYE_LIMIT))
    else:
        # Scheduled autonomous scan path (backward compatible behavior).
        seeds_raw = SCANNER_SEEDS or str(app.DEFAULT_SEEDS)
        max_items = SCANNER_MAX_ITEMS
        use_birdeye_trending = USE_BIRDEYE_TRENDING
        birdeye_limit = BIRDEYE_LIMIT
    scan_result: Dict[str, Any] = {}
    updates: Dict[str, Any] = {}
    status = "empty"
    try:
        if pending:
            updates["scan_request_worker_consumed_ts"] = app.now_utc_str()
        scan_result = app.maybe_run_rotating_scanner(
            seeds_raw=seeds_raw,
            max_items=max_items,
            use_birdeye_trending=use_birdeye_trending,
            birdeye_limit=birdeye_limit,
        )
        status = "failed" if scan_result.get("error") else "success"
        stats = scan_result.get("stats", {}) if isinstance(scan_result.get("stats", {}), dict) else {}
        if not scan_result.get("error") and (not stats or int(app.parse_float(stats.get("seen", stats.get("raw_seen", 0)), 0)) <= 0):
            status = "empty"
        updates.update({
            "last_scan_status": status,
            "last_scan_stats": stats,
            "last_scan_ts": app.now_utc_str(),
        })
    except Exception as exc:
        status = "failed"
        updates.update({
            "last_scan_status": "failed",
            "last_scan_error": f"scan_cycle_exception:{type(exc).__name__}:{exc}",
            "last_scan_stats": {},
            "last_scan_ts": app.now_utc_str(),
        })
        scan_result = {"error": f"{type(exc).__name__}: {exc}", "stats": {}}
    finally:
        if chain_reason:
            updates["last_scan_status"] = f"{updates['last_scan_status']}:{chain_reason}"
        if pending:
            updates["scan_request_pending"] = False
            updates["scan_request_processed_ts"] = app.now_utc_str()
            updates["scan_request_status"] = "stale_cleared" if stale_pending else "processed"
            if stale_pending:
                updates["scan_request_last_clear_reason"] = "stale_pending_over_90m"
        if hasattr(app, "update_worker_runtime_state"):
            app.update_worker_runtime_state(updates=updates)
    try:
        monitoring_rows = app.load_monitoring()
        portfolio_rows = app.load_portfolio()
        active_monitoring_rows = app.build_active_monitoring_rows(monitoring_rows)
        active_keys = {app.canonical_token_key(r) for r in active_monitoring_rows if app.canonical_token_key(r)}
        payload = app.build_live_pulse_candidates_payload(
            monitoring_rows=monitoring_rows,
            portfolio_rows=portfolio_rows,
            active_monitoring_keys=active_keys,
            chain_filter="all",
            limit=15,
            scan_result=scan_result,
            status=status,
        )
        app.live_pulse_storage_write(payload)
        app.update_worker_runtime_state(updates={"live_pulse_candidates": payload, "last_empty_reason": payload.get("last_empty_reason", "")})
    except Exception as pulse_exc:
        failed_payload = {
            "ts_utc": app.now_utc_str(),
            "source": "scan_cycle",
            "status": "failed",
            "raw_seen": 0,
            "normalized": 0,
            "clean_candidates": 0,
            "final_count": 0,
            "refill_attempts": 0,
            "last_empty_reason": "worker_failed",
            "sources_tried": [],
            "blocked_reasons": {"worker_failed": 1},
            "final_candidates": [],
            "rejected_candidates": [],
            "debug": {"error": f"{type(pulse_exc).__name__}: {pulse_exc}"},
        }
        try:
            app.live_pulse_storage_write(failed_payload)
        except Exception:
            pass
        app.update_worker_runtime_state(updates={"live_pulse_candidates": failed_payload, "last_empty_reason": "worker_failed"})
    pulse_result = _record_pulse_history_after_cycle_safe()
    return {"scan": scan_result, "pulse_history": pulse_result}


def _run_monitor_cycle() -> Dict[str, Any]:
    assert app is not None
    monitoring_rows = app.load_monitoring()
    portfolio_rows = [
        r for r in app.load_portfolio()
        if str(r.get("active", "1")).strip() == "1"
    ]
    monitor_result = app.run_priority_scanner_cycle(
        monitoring_rows=monitoring_rows,
        portfolio_rows=portfolio_rows,
        max_scans=3,
    )
    history_stats: Dict[str, Any] = {"monitoring": 0, "portfolio": 0}
    ts_now = datetime.now(timezone.utc)
    try:
        append_fn = getattr(app, "append_token_history_snapshot", None)
        if callable(append_fn):
            for row in app.build_active_monitoring_rows(monitoring_rows):
                if append_fn(row, source="monitoring", now_ts=ts_now):
                    history_stats["monitoring"] += 1
            for row in portfolio_rows:
                if append_fn(row, source="portfolio", now_ts=ts_now):
                    history_stats["portfolio"] += 1
        flush_fn = getattr(app, "flush_monitoring_history_buffer", None)
        if callable(flush_fn):
            try:
                flush_fn(force=True)
            except Exception as flush_exc:
                history_stats["history_flush_error"] = f"{type(flush_exc).__name__}:{flush_exc}"
        else:
            history_stats["history_flush_error"] = "missing_flush_monitoring_history_buffer"
    except Exception as history_exc:
        history_stats["history_snapshot_error"] = f"{type(history_exc).__name__}:{history_exc}"
    monitor_result["history_snapshots"] = history_stats
    hard_gate_archived = 0
    hard_gate_reasons: Dict[str, int] = {}
    hard_gate_symbols: List[str] = []
    for row in app.build_active_monitoring_rows(monitoring_rows):
        portfolio_row = app.find_active_portfolio_row(row, portfolio_rows) if hasattr(app, "find_active_portfolio_row") else {}
        gate = app.hard_gate_monitoring_row(row, portfolio_row=portfolio_row) if hasattr(app, "hard_gate_monitoring_row") else {"blocked": False, "reason": "", "flags": []}
        reasons = [str(r) for r in str(gate.get("reason") or "").split("|") if str(r).strip()]
        if not gate.get("blocked"):
            continue
        chain = app.token_chain(row)
        base_addr = app.token_ca(row)
        if chain and base_addr:
            archived = app.archive_monitoring(chain, base_addr, reason=f"hard_gate:{'|'.join(reasons)}", revisit_days=0)
            if archived:
                hard_gate_archived += 1
                for reason in reasons:
                    hard_gate_reasons[reason] = int(hard_gate_reasons.get(reason, 0) or 0) + 1
                hard_gate_symbols.append(str(row.get("base_symbol") or "UNKNOWN").upper())
    monitor_result["hard_gate_archived"] = int(hard_gate_archived)
    monitor_result["hard_gate_reasons"] = hard_gate_reasons
    monitor_result["hard_gate_symbols"] = hard_gate_symbols[:25]
    try:
        app.update_worker_runtime_state(updates={
            "hard_gate_archived": int(hard_gate_archived),
            "hard_gate_reasons": hard_gate_reasons,
            "hard_gate_symbols": hard_gate_symbols[:25],
        })
    except Exception:
        pass
    pulse_result = _record_pulse_history_after_cycle_safe()
    return {"monitor": monitor_result, "pulse_history": pulse_result}


def _record_pulse_history_after_cycle_safe() -> Dict[str, Any]:
    assert app is not None
    try:
        return _record_pulse_history_after_cycle()
    except Exception as exc:
        reason = f"pulse_history_writer_exception:{type(exc).__name__}:{exc}"
        print(f"[worker] {reason}", flush=True)
        try:
            app.update_worker_runtime_state(
                updates={
                    "pulse_history_writer_ok": False,
                    "pulse_history_writer_reason": reason,
                    "pulse_history_writer_error_ts": app.now_utc_str(),
                }
            )
            app.update_job_heartbeat(
                job_name="pulse_history_writer",
                job_mode="pulse_history",
                status="failed_nonblocking",
                meta={"reason": reason, "nonblocking": True},
            )
        except Exception:
            pass
        return {"ok": False, "reason": reason, "nonblocking": True}


def _record_pulse_history_after_cycle() -> Dict[str, Any]:
    assert app is not None
    monitoring_rows = app.load_monitoring()
    active_monitoring_rows = app.build_active_monitoring_rows(monitoring_rows)
    active_keys = {
        app.canonical_token_key(row)
        for row in active_monitoring_rows
        if app.canonical_token_key(row)
    }
    portfolio_rows = app.load_portfolio()
    active_portfolio_rows = [r for r in portfolio_rows if str(r.get("active", "1")).strip() == "1"]
    for row in active_portfolio_rows:
        key = app.canonical_token_key(row)
        if key:
            active_keys.add(key)
    result = app.record_live_pulse_history_from_candidates(
        active_monitoring_keys=active_keys,
        chain_filter="all",
        limit=8,
    )
    print(
        "[worker] pulse_history_writer_ran "
        f"pulse_history_writer_candidates={result.get('candidates_seen', 0)} "
        f"pulse_history_writer_appended={result.get('points_appended', 0)} "
        f"pulse_history_writer_flushed={str(bool(result.get('flushed', False))).lower()} "
        f"pulse_history_writer_reason={result.get('reason', 'unknown')}",
        flush=True,
    )
    return result


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

def _run_maintenance_cycle() -> Dict[str, Any]:
    assert app is not None
    return app.run_storage_maintenance_cycle()


JOB_DISPATCH: Dict[str, Callable[[], Dict[str, Any]]] = {
    "scan_cycle": _run_scan_cycle,
    "monitor_cycle": _run_monitor_cycle,
    "notify_cycle": _run_notify_cycle,
    "digest_cycle": _run_digest_cycle,
    "outcome_cycle": _run_outcome_cycle,
    "maintenance_cycle": _run_maintenance_cycle,
}


def validate_job_dispatch(dispatch: Dict[str, Any] | None = None) -> None:
    dispatch = JOB_DISPATCH if dispatch is None else dispatch
    bad = [str(name) for name, runner in dispatch.items() if not callable(runner)]
    if bad:
        raise RuntimeError(f"invalid_job_dispatch_non_callable:{','.join(sorted(bad))}")


validate_job_dispatch(JOB_DISPATCH)


def run_job_mode(job_mode: str) -> int:
    assert app is not None
    mode = str(job_mode or "").strip().lower()
    lock_key = None
    owner = f"{socket.gethostname()}:{os.getpid()}"
    try:
        validate_job_dispatch(JOB_DISPATCH)
    except RuntimeError as e:
        reason = str(e)
        print(f"[worker] safe-fail: {reason}", flush=True)
        return 2

    runner = JOB_DISPATCH.get(mode)

    if runner is None:
        reason = f"unknown_job_mode:{mode or 'empty'}"
        _update_worker_runtime_with_mode_state(
            mode=mode or "unknown",
            owner=owner,
            mode_updates={
                "last_job_status": "invalid",
                "last_job_reason": reason,
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
            },
            global_updates={
                "worker_status": "job_mode_invalid",
                "last_job_mode": mode,
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode or "unknown"),
            job_mode=mode,
            status="invalid_mode",
            meta={"reason": reason, "allowed_modes": list(JOB_DISPATCH.keys())},
        )
        print(f"[worker] safe-fail: {reason}", flush=True)
        return 2

    runtime = app.get_worker_runtime_state()
    mode_state = _runtime_mode_state(runtime, mode)
    prev_status = str(mode_state.get("last_job_status") or "").strip().lower()
    prev_started_epoch = float(app.parse_float(mode_state.get("last_job_started_epoch", 0.0), 0.0))
    age_sec = max(0.0, time.time() - prev_started_epoch) if prev_started_epoch > 0 else 0.0
    stale_run = bool(prev_status == "running" and age_sec >= JOB_STALE_RUN_SEC)
    if stale_run:
        _update_worker_runtime_with_mode_state(
            mode=mode,
            owner=owner,
            mode_updates={
                "last_job_status": "aborted",
                "last_job_reason": "stale_aborted",
                "last_job_finished_ts": app.now_utc_str(),
                "last_job_finished_epoch": time.time(),
                "last_error_reason": f"stale_recovered:prev_mode={mode}:age_sec={int(age_sec)}",
            },
            global_updates={
                "worker_status": "job_stale_recovered",
                "last_error_reason": (
                    f"stale_recovered:prev_mode={mode or 'unknown'}:age_sec={int(age_sec)}"
                ),
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status="stale_recovered",
            meta={"prev_mode": mode, "age_sec": int(age_sec), "stale_threshold_sec": JOB_STALE_RUN_SEC},
        )
        print(f"[worker] stale_recovered mode={mode or 'unknown'} age_sec={int(age_sec)}", flush=True)
        prev_status = "aborted"

    if prev_status == "running" and age_sec < JOB_STALE_RUN_SEC:
        reason = f"duplicate_run_guard:mode={mode}:age_sec={int(age_sec)}"
        _update_worker_runtime_with_mode_state(
            mode=mode,
            owner=owner,
            mode_updates={
                "last_job_status": "skipped_duplicate_run",
                "last_job_reason": reason,
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_lock_code": "duplicate_run_guard",
            },
            global_updates={
                "worker_status": "job_skipped_duplicate_run",
                "last_job_mode": mode,
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status="duplicate_guard",
            meta={"reason": reason, "age_sec": int(age_sec), "stale_threshold_sec": JOB_STALE_RUN_SEC},
        )
        print(f"[worker] duplicate_guard mode={mode} age_sec={int(age_sec)}", flush=True)
        return 3

    # FIX #1: initialise lock_key to None before any branching so the
    # finally block never hits NameError on early returns.
    lock_key: Optional[str] = None

    lock_key = f"job_mode:{mode}"

    # FIX #6: only write stale_run heartbeat AFTER we know we will attempt
    # the lock — avoids misleading "stale_run_detected" when lock then fails.
    lock_status = app.acquire_lock(lock_key=lock_key, owner=owner, ttl_sec=JOB_LOCK_TTL_SEC)

    if lock_status.get("ok") and bool(lock_status.get("stale_replaced")):
        _update_worker_runtime_with_mode_state(
            mode=mode,
            owner=owner,
            mode_updates={
                "last_stale_lock_ts": app.now_utc_str(),
                "last_job_reason": (
                    f"stale_lock_replaced:{lock_key}:age_sec={int(app.parse_float(lock_status.get('stale_age_sec'), 0.0))}"
                ),
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status="stale_lock_replaced",
            meta={"lock_key": lock_key, "lock_status": lock_status},
        )

    if not lock_status.get("ok"):
        code = str(lock_status.get("code") or "lock_failed")
        reason = f"{code}:{lock_status.get('detail') or lock_status.get('message') or ''}".strip(":")
        heartbeat_status = "lock_held" if code == "lock_held" else "lock_failed"
        _update_worker_runtime_with_mode_state(
            mode=mode,
            owner=owner,
            mode_updates={
                "last_job_status": "skipped_locked",
                "last_job_reason": reason,
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_lock_code": code,
            },
            global_updates={
                "worker_status": "job_skipped_locked",
                "last_job_mode": mode,
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status=heartbeat_status,
            meta={"lock_key": lock_key, "owner": owner, "lock_status": lock_status},
        )
        print(f"[worker] {heartbeat_status} mode={mode} code={code}", flush=True)
        return 3

    start_epoch = time.time()
    start_ts = app.now_utc_str()
    run_id = str(uuid.uuid4())
    job_start_token = run_id
    runtime = app.get_worker_runtime_state()
    overlap_count = 0
    for other_mode, other_state in _runtime_modes(runtime).items():
        if other_mode == mode:
            continue
        if str(other_state.get("last_job_status") or "").strip().lower() == "running":
            overlap_count += 1
    _update_worker_runtime_with_mode_state(
        mode=mode,
        owner=owner,
        mode_updates={
            "run_id": run_id,
            "job_start_token": job_start_token,
            "last_job_status": "running",
            "last_job_reason": "",
            "last_lock_code": "",
            "last_job_started_ts": start_ts,
            "last_job_started_epoch": start_epoch,
            "last_job_finished_ts": "",
            "last_job_finished_epoch": 0.0,
            "last_error_reason": "",
            "concurrent_mode_overlap_last": overlap_count,
        },
        global_updates={
            "worker_status": "job_running",
            "worker_id": owner,
            "worker_role": "job_dispatch",
            "last_job_mode": mode,
            "last_error_reason": "",
        },
        increments={"concurrent_mode_overlap": 1} if overlap_count > 0 else None,
    )
    app.update_job_heartbeat(
        job_name=_heartbeat_job_name(mode),
        job_mode=mode,
        status="started",
        meta={"lock_key": lock_key, "owner": owner, "concurrent_mode_overlap": overlap_count},
    )

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(_signum, _frame):
        raise _TerminatedSignal("terminated_signal")

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        result = runner()
        end_epoch = time.time()
        duration_sec = round(max(0.0, end_epoch - start_epoch), 3)
        _finalize_runtime_if_token_matches(
            mode=mode,
            owner=owner,
            run_id=run_id,
            updates={
                "last_job_status": "finished",
                "last_job_reason": "ok",
                "last_job_finished_ts": app.now_utc_str(),
                "last_job_finished_epoch": end_epoch,
                "last_job_duration_sec": duration_sec,
            },
            global_updates={"worker_status": "job_finished"},
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status="finished",
            meta={"duration_sec": duration_sec, "result": result, "run_id": run_id},
        )
        print(f"[worker] storage_budget_summary {app.get_supabase_storage_budget_snapshot()}", flush=True)
        print(f"[worker] mode={mode} finished result={result}", flush=True)
        return 0

    except _TerminatedSignal:
        end_epoch = time.time()
        _finalize_runtime_if_token_matches(
            mode=mode,
            owner=owner,
            run_id=run_id,
            updates={
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": "terminated_signal",
                "last_job_status": "aborted",
                "last_job_reason": "terminated_signal",
                "last_job_finished_ts": app.now_utc_str(),
                "last_job_finished_epoch": end_epoch,
                "last_job_duration_sec": round(max(0.0, end_epoch - start_epoch), 3),
            },
            global_updates={
                "worker_status": "job_terminated",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": "terminated_signal",
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status="terminated",
            meta={"reason": "terminated_signal", "run_id": run_id},
        )
        print(f"[worker] terminated mode={mode} run_id={run_id}", flush=True)
        return 1

    except Exception as exc:
        reason = f"job_mode_exception:{mode}:{type(exc).__name__}:{exc}"
        _finalize_runtime_if_token_matches(
            mode=mode,
            owner=owner,
            run_id=run_id,
            updates={
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
                "last_job_status": "failed",
                "last_job_reason": reason,
                "last_job_finished_ts": app.now_utc_str(),
            },
            global_updates={
                "worker_status": "job_failed",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": reason,
            },
        )
        app.update_job_heartbeat(
            job_name=_heartbeat_job_name(mode),
            job_mode=mode,
            status="failed",
            meta={"reason": reason, "run_id": run_id},
        )
        print(f"[worker] mode={mode} failed reason={reason}", flush=True)
        traceback.print_exc()
        return 1

    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if lock_key is not None:
            app.release_lock(lock_key=lock_key, owner=owner)


def main() -> int:
    global app
    app = _load_app()

    missing_env = _missing_required_env()
    if missing_env:
        return _fail_fast(f"missing_env:{','.join(missing_env)}", 12)

    app.ensure_storage()

    skip_on_unavailable = _env_bool("RUNTIME_SKIP_ON_SUPABASE_UNAVAILABLE", True)
    contract = app.check_runtime_contract()
    if not contract.get("ok"):
        if skip_on_unavailable and app.is_supabase_unavailable_status(contract):
            print("[worker] skipped: external storage unavailable/quota paused", flush=True)
            return 0
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

    app.reset_supabase_storage_budget_counters()
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
