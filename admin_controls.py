"""Import-safe manual recovery controls for DEX Scout operators.

This module is deliberately pure at import time.  It does not import Streamlit or
``app.py`` and never performs storage/network writes, scanner runs, or Telegram
sends.  Mutating operations are only available through explicit adapter
callbacks supplied to :func:`execute_admin_action`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

DEFAULT_RECOMMENDED_ORDER = [
    "runtime_matrix",
    "read_only",
    "maintenance_cycle",
    "monitor_cycle",
    "notify_cycle",
    "scan_cycle",
]

RUNBOOK_STEPS = {
    "runtime_matrix_runbook": "GitHub Actions → Runtime Smoke → runtime_matrix",
    "read_only_runbook": "GitHub Actions → Runtime Smoke → read_only",
    "maintenance_cycle_runbook": "GitHub Actions → Runtime Jobs → maintenance_cycle",
    "monitor_cycle_runbook": "GitHub Actions → Runtime Jobs → monitor_cycle",
    "notify_cycle_runbook": "GitHub Actions → Runtime Jobs → notify_cycle",
    "scan_cycle_runbook": "GitHub Actions → Runtime Jobs → scan_cycle (last, only after checks are green)",
}

CONFIRMATION_PHRASES = {
    "clear_stale_lock": "CLEAR STALE LOCK",
    "mark_runtime_note": "MARK RUNTIME NOTE",
    "trigger_existing_admin_task": "TRIGGER ADMIN TASK",
}

STALE_LOCK_THRESHOLD_SEC = 300
MAX_ITEMS = 8


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _parse_epoch(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


def _status(value: Any) -> str:
    return _text(value, "unknown").lower()


def _green(value: Any) -> bool:
    return _status(value) in {"ok", "success", "pass", "passed", "green", "verified", "skipped"}


def _bad(value: Any) -> bool:
    return _status(value) in {"failed", "fail", "error", "degraded", "missing", "mismatch", "source_api_failed"}


def _runtime_context(context: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _as_dict(context)
    diag = _as_dict(ctx.get("operator_diagnostics"))
    runtime_diag = _as_dict(diag.get("runtime"))
    runtime_state = _as_dict(ctx.get("runtime_state") or ctx.get("runtime"))
    return {"ctx": ctx, "diag": diag, "runtime_diag": runtime_diag, "runtime_state": runtime_state}


def _candidate_locks(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts = _runtime_context(context)
    ctx = parts["ctx"]
    runtime_state = parts["runtime_state"]
    runtime_diag = parts["runtime_diag"]
    raw: List[Any] = []
    for source in (ctx, runtime_state, runtime_diag):
        for key in ("locks", "active_locks", "stale_locks", "lock", "current_lock"):
            value = source.get(key)
            if isinstance(value, dict):
                raw.append(value)
            elif isinstance(value, list):
                raw.extend(value)
    # Legacy worker breadcrumbs are useful visibility but not enough alone unless
    # they also include an explicit lock key and owner.
    if runtime_state.get("last_stale_lock_ts") or runtime_state.get("last_lock_code"):
        raw.append({
            "lock_key": runtime_state.get("last_lock_key") or runtime_state.get("lock_key"),
            "owner": runtime_state.get("last_lock_owner") or runtime_state.get("lock_owner"),
            "reason": runtime_state.get("last_lock_code") or runtime_state.get("stale_lock_reason"),
            "stale": bool(runtime_state.get("last_stale_lock_ts")),
            "updated_epoch": runtime_state.get("last_stale_lock_epoch"),
            "updated_ts": runtime_state.get("last_stale_lock_ts"),
        })
    return [dict(item) for item in raw if isinstance(item, dict)][:MAX_ITEMS]


def _lock_key(lock: Dict[str, Any]) -> str:
    return _text(lock.get("lock_key") or lock.get("key") or lock.get("id") or lock.get("name"))


def _lock_owner(lock: Dict[str, Any]) -> str:
    return _text(lock.get("owner") or lock.get("holder") or lock.get("job") or lock.get("job_name"))


def _lock_age_sec(lock: Dict[str, Any], *, now_epoch: Optional[float] = None) -> int:
    now = _now_epoch() if now_epoch is None else float(now_epoch)
    if lock.get("age_sec") not in (None, ""):
        try:
            return max(0, int(float(lock.get("age_sec"))))
        except Exception:
            return 0
    expires = _parse_epoch(lock.get("expires_epoch") or lock.get("expires_ts"))
    if expires > 0 and expires <= now:
        return max(0, int(now - expires))
    updated = _parse_epoch(lock.get("updated_epoch") or lock.get("heartbeat_epoch") or lock.get("updated_ts") or lock.get("ts"))
    if updated > 0:
        return max(0, int(now - updated))
    return 0


def _lock_is_stale(lock: Dict[str, Any]) -> bool:
    if lock.get("is_stale") is True or lock.get("stale") is True:
        return True
    expires = _parse_epoch(lock.get("expires_epoch") or lock.get("expires_ts"))
    if expires > 0 and expires <= _now_epoch():
        return True
    return _lock_age_sec(lock) > int(lock.get("stale_threshold_sec") or STALE_LOCK_THRESHOLD_SEC)


def _auditable_stale_lock(context: Dict[str, Any]) -> Dict[str, Any]:
    for lock in _candidate_locks(context):
        if _lock_is_stale(lock) and _lock_key(lock) and _lock_owner(lock):
            out = dict(lock)
            out["lock_key"] = _lock_key(lock)
            out["owner"] = _lock_owner(lock)
            out["age_sec"] = _lock_age_sec(lock)
            return out
    return {}


def _job_heartbeats(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts = _runtime_context(context)
    raw: List[Any] = []
    for source in (parts["ctx"], parts["runtime_state"], parts["runtime_diag"]):
        for key in ("job_heartbeats", "heartbeats", "stale_jobs", "last_jobs"):
            value = source.get(key)
            if isinstance(value, dict):
                raw.extend(dict(v, job=k) if isinstance(v, dict) else {"job": k, "value": v} for k, v in value.items())
            elif isinstance(value, list):
                raw.extend(value)
    return [dict(item) for item in raw if isinstance(item, dict)][:MAX_ITEMS]


def get_admin_action_registry() -> Dict[str, Dict[str, Any]]:
    """Return declarative admin action descriptors."""
    return {
        "runtime_matrix_runbook": {"type": "runbook", "mutating": False, "enabled_by_default": True, "step": RUNBOOK_STEPS["runtime_matrix_runbook"]},
        "read_only_runbook": {"type": "runbook", "mutating": False, "enabled_by_default": True, "step": RUNBOOK_STEPS["read_only_runbook"]},
        "maintenance_cycle_runbook": {"type": "runbook", "mutating": False, "enabled_by_default": True, "step": RUNBOOK_STEPS["maintenance_cycle_runbook"]},
        "monitor_cycle_runbook": {"type": "runbook", "mutating": False, "enabled_by_default": True, "step": RUNBOOK_STEPS["monitor_cycle_runbook"]},
        "notify_cycle_runbook": {"type": "runbook", "mutating": False, "enabled_by_default": True, "step": RUNBOOK_STEPS["notify_cycle_runbook"]},
        "scan_cycle_runbook": {"type": "runbook", "mutating": False, "enabled_by_default": True, "step": RUNBOOK_STEPS["scan_cycle_runbook"]},
        "clear_stale_lock": {"type": "guarded_mutation", "mutating": True, "enabled_by_default": False, "confirmation_phrase": CONFIRMATION_PHRASES["clear_stale_lock"]},
        "mark_runtime_note": {"type": "guarded_mutation", "mutating": True, "enabled_by_default": False, "confirmation_phrase": CONFIRMATION_PHRASES["mark_runtime_note"], "optional": True},
        "trigger_existing_admin_task": {"type": "guarded_mutation", "mutating": True, "enabled_by_default": False, "confirmation_phrase": CONFIRMATION_PHRASES["trigger_existing_admin_task"], "optional": True},
    }


def _statuses(context: Dict[str, Any]) -> Dict[str, str]:
    parts = _runtime_context(context)
    rd = parts["runtime_diag"]
    ctx = parts["ctx"]
    return {
        "runtime_matrix": _status(rd.get("matrix_status") or _as_dict(ctx.get("runtime_matrix")).get("status") or _as_dict(ctx.get("runtime_matrix_summary")).get("status")),
        "read_only": _status(ctx.get("read_only_status") or _as_dict(ctx.get("read_only")).get("status") or _as_dict(ctx.get("runtime_read_only")).get("status") or rd.get("read_only_status")),
        "monitor_cycle": _status(ctx.get("monitor_cycle_status") or rd.get("monitor_cycle_status") or _as_dict(_as_dict(rd.get("last_jobs")).get("monitor_cycle")).get("last_job_status")),
        "notify_cycle": _status(ctx.get("notify_cycle_status") or rd.get("notify_cycle_status") or _as_dict(_as_dict(rd.get("last_jobs")).get("notify_cycle")).get("last_job_status")),
    }


def build_admin_recovery_plan(context: dict) -> dict:
    """Build a compact read-only manual recovery plan from diagnostics context."""
    ctx = _as_dict(context)
    parts = _runtime_context(ctx)
    reasons: Dict[str, Any] = {}
    warnings: List[str] = []
    available: List[str] = []
    blocked: List[str] = []
    registry = get_admin_action_registry()
    statuses = _statuses(ctx)
    stale_lock = _auditable_stale_lock(ctx)
    heartbeats = _job_heartbeats(ctx)
    diag_status = _status(parts["diag"].get("status"))
    scanner = _as_dict(parts["diag"].get("scanner") or ctx.get("scanner_diagnostics"))

    for action_id, descriptor in registry.items():
        validation = validate_admin_action(action_id, ctx, dry_run=True)
        if validation.get("enabled") or not descriptor.get("mutating"):
            available.append(action_id)
        else:
            blocked.append(action_id)

    if stale_lock:
        reasons["stale_lock"] = {"lock_key": stale_lock.get("lock_key"), "owner": stale_lock.get("owner"), "age_sec": stale_lock.get("age_sec")}
        warnings.append("Stale lock detected; clear only after confirming no active worker owns it.")
    elif _candidate_locks(ctx):
        reasons["locks"] = "lock data present but no auditable stale lock"

    stale_jobs = [hb for hb in heartbeats if hb.get("is_stale") is True or hb.get("stale") is True or _status(hb.get("status")) == "stale"]
    if stale_jobs:
        reasons["stale_jobs"] = stale_jobs[:3]
        warnings.append("Stale job heartbeat detected; prefer runbook checks before any scan cycle.")

    source_status = _status(scanner.get("source_status"))
    if source_status in {"source_api_failed", "failed", "error"}:
        reasons["source_api_failed"] = True
        warnings.append("Source API failed; use smoke/read-only checks before runtime jobs.")
    if parts["runtime_state"].get("last_block_reason") or parts["runtime_state"].get("last_duplicate_run_reason"):
        reasons["blocked_cycle"] = parts["runtime_state"].get("last_block_reason") or parts["runtime_state"].get("last_duplicate_run_reason")

    if not ctx:
        status = "unknown"
        warnings.append("Diagnostics context not available; actions remain guarded.")
    elif stale_lock or stale_jobs or source_status in {"source_api_failed", "failed", "error"} or diag_status == "degraded":
        status = "recovery_needed"
    elif warnings or diag_status == "warning":
        status = "attention"
    elif all(_green(v) for v in statuses.values() if v != "unknown") and diag_status in {"ok", "unknown"}:
        status = "ok"
    else:
        status = "attention" if any(v != "unknown" for v in statuses.values()) else "unknown"

    steps = [RUNBOOK_STEPS[action] for action in (
        "runtime_matrix_runbook", "read_only_runbook", "maintenance_cycle_runbook", "monitor_cycle_runbook", "notify_cycle_runbook", "scan_cycle_runbook"
    )]
    return {
        "status": status,
        "recommended_order": list(DEFAULT_RECOMMENDED_ORDER),
        "available_actions": available[:MAX_ITEMS],
        "blocked_actions": blocked[:MAX_ITEMS],
        "warnings": warnings[:MAX_ITEMS],
        "reasons": reasons,
        "manual_workflow_steps": steps,
    }


def validate_admin_action(action_id: str, context: dict, *, confirmation: str = "", dry_run: bool = True) -> dict:
    registry = get_admin_action_registry()
    action = registry.get(str(action_id or ""))
    if not action:
        return {"ok": False, "enabled": False, "dry_run": bool(dry_run), "action_id": str(action_id or ""), "reason": "unknown_action", "requires_confirmation": False, "confirmation_phrase": "", "would_mutate": False, "warnings": []}
    mutating = bool(action.get("mutating"))
    phrase = _text(action.get("confirmation_phrase"))
    warnings: List[str] = []
    enabled = bool(action.get("enabled_by_default")) and not mutating
    reason = "enabled"

    if action_id == "clear_stale_lock":
        stale_lock = _auditable_stale_lock(_as_dict(context))
        enabled = bool(stale_lock)
        reason = "stale_lock_confirmed" if enabled else "stale_lock_not_available_or_not_auditable"
    elif action_id == "mark_runtime_note":
        enabled = bool(_as_dict(context).get("supports_runtime_note") or _as_dict(context).get("runtime_note_supported"))
        reason = "runtime_note_supported" if enabled else "runtime_note_write_not_supported"
    elif action_id == "trigger_existing_admin_task":
        enabled = bool(_as_dict(context).get("supports_admin_task") or _as_dict(context).get("admin_task_supported") or _as_dict(_as_dict(context).get("actions")).get("trigger_admin_task"))
        reason = "existing_admin_task_supported" if enabled else "existing_admin_task_not_supported"
    elif action_id == "scan_cycle_runbook":
        statuses = _statuses(_as_dict(context))
        if not (_green(statuses.get("runtime_matrix")) and _green(statuses.get("read_only")) and _green(statuses.get("monitor_cycle"))):
            warnings.append("scan_cycle should be last and only after runtime_matrix/read_only/monitor are green.")
        reason = "runbook_only"
    elif action.get("type") == "runbook":
        reason = "runbook_only"

    if mutating:
        if not enabled:
            pass
        elif _text(confirmation) != phrase:
            enabled = False
            reason = "confirmation_required"
    return {
        "ok": bool(action),
        "enabled": bool(enabled),
        "dry_run": bool(dry_run),
        "action_id": str(action_id),
        "reason": reason,
        "requires_confirmation": mutating,
        "confirmation_phrase": phrase,
        "would_mutate": mutating,
        "warnings": warnings[:MAX_ITEMS],
    }


def execute_admin_action(action_id: str, context: dict, *, confirmation: str = "", dry_run: bool = True, adapters: dict | None = None) -> dict:
    validation = validate_admin_action(action_id, context, confirmation=confirmation, dry_run=dry_run)
    audit = {"ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "dry_run": bool(dry_run), "would_mutate": bool(validation.get("would_mutate"))}
    if not validation.get("ok") or not validation.get("enabled"):
        return {"ok": False, "status": "blocked", "action_id": str(action_id or ""), "reason": str(validation.get("reason") or "blocked"), "result": {"validation": validation}, "audit": audit}
    if dry_run:
        return {"ok": True, "status": "dry_run", "action_id": str(action_id), "reason": "dry_run_no_mutation", "result": {"validation": validation}, "audit": audit}
    registry = get_admin_action_registry()
    if not registry.get(action_id, {}).get("mutating"):
        return {"ok": True, "status": "executed", "action_id": str(action_id), "reason": "runbook_acknowledged", "result": {"validation": validation}, "audit": audit}

    adapter_key = {
        "clear_stale_lock": "clear_stale_lock_fn",
        "mark_runtime_note": "write_runtime_note_fn",
        "trigger_existing_admin_task": "trigger_admin_task_fn",
    }.get(action_id)
    fn: Optional[Callable[..., Any]] = _as_dict(adapters).get(adapter_key) if adapter_key else None  # type: ignore[assignment]
    if not callable(fn):
        return {"ok": False, "status": "blocked", "action_id": str(action_id), "reason": "adapter_missing", "result": {"validation": validation}, "audit": audit}
    try:
        stale_lock = _auditable_stale_lock(_as_dict(context)) if action_id == "clear_stale_lock" else {}
        result = fn(context=_as_dict(context), action_id=action_id, validation=validation, stale_lock=stale_lock)
        ok = bool(_as_dict(result).get("ok", True)) if isinstance(result, dict) else True
        return {"ok": ok, "status": "executed" if ok else "failed", "action_id": str(action_id), "reason": "adapter_executed" if ok else "adapter_returned_not_ok", "result": result if isinstance(result, dict) else {"value": result}, "audit": audit}
    except Exception as exc:
        return {"ok": False, "status": "failed", "action_id": str(action_id), "reason": f"adapter_exception:{type(exc).__name__}", "result": {}, "audit": audit}
