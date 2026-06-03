"""Import-safe service facade for DEX Scout business summaries.

This module is intentionally pure: it does not import Streamlit, does not import
``app.py``, and does not perform storage/network work at import time.  It exposes
small service-level builders that entrypoints (Streamlit app, worker facade, and
tests) can share while app.py continues to own runtime I/O and UI rendering.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import monitoring_core  # noqa: F401 - service boundary dependency for future monitoring facades
import notification_core
import runtime_core
import storage_core  # noqa: F401 - service boundary dependency for storage/runtime constants
from storage_repository import summarize_storage_result


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_rows(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _int(value: Any, default: int = 0) -> int:
    return int(runtime_core.parse_float(value, float(default)))


def _str(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _top_counts(values: Iterable[Any], limit: int = 5) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        key = _str(value)
        if key:
            counter[key] += 1
    return dict(counter.most_common(max(0, int(limit))))


def build_runtime_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact runtime health summary from a UI/runtime context dict."""
    ctx = _as_dict(context)
    runtime = _as_dict(ctx.get("runtime_state") or ctx.get("runtime"))
    counters = _as_dict(runtime.get("last_notification_counters"))
    return {
        "version": _str(ctx.get("version")),
        "storage_backend": _str(ctx.get("storage_backend") or runtime.get("storage_backend")),
        "selected_page": _str(ctx.get("selected_page"), "Monitoring"),
        "worker_loop_ts": _str(runtime.get("last_loop_ts")),
        "last_notifications_ts": _str(runtime.get("last_notifications_ts")),
        "last_send_success_ts": _str(runtime.get("last_send_success_ts")),
        "last_error_ts": _str(runtime.get("last_error_ts")),
        "last_error_reason": _str(runtime.get("last_error_reason")),
        "last_empty_reason": _str(runtime.get("last_empty_reason")),
        "last_block_reason": _str(runtime.get("last_block_reason")),
        "last_cycle_status": _str(runtime.get("last_cycle_status")),
        "last_diag_summary": _str(runtime.get("last_diag_summary")),
        "loop_iterations": _int(runtime.get("loop_iterations", 0)),
        "notification_runs": _int(runtime.get("notification_runs", 0)),
        "notification_sent": _int(runtime.get("notification_sent", 0)),
        "notification_failed_cycles": _int(runtime.get("notification_failed_cycles", 0)),
        "notification_counters": {
            "before_filter": _int(counters.get("before_filter", 0)),
            "after_filter": _int(counters.get("after_filter", 0)),
            "sent": _int(counters.get("sent", 0)),
            "blocked": _int(counters.get("blocked", 0)),
            "send_fail": _int(counters.get("send_fail", 0)),
            "journal_size": _int(counters.get("journal_size", runtime.get("notification_journal_size", 0))),
        },
    }


def build_storage_summary(storage_result: Dict[str, Any], runtime_state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize storage diagnostics for UI/runtime display."""
    raw_result = _as_dict(storage_result)
    if "verify_status" in raw_result and "verify_attempted" not in raw_result and "verify_skipped" not in raw_result:
        result = {
            "ok": bool(raw_result.get("ok")),
            "write_ok": bool(raw_result.get("write_ok")),
            "verify_status": _str(raw_result.get("verify_status"), "skipped"),
            "verify_source": _str(raw_result.get("verify_source"), "none"),
            "error": _str(raw_result.get("error")),
            **{k: v for k, v in raw_result.items() if k not in {"ok", "write_ok", "verify_status", "verify_source", "error"}},
        }
    else:
        result = summarize_storage_result(raw_result)
    runtime = _as_dict(runtime_state)
    verify_status = _str(result.get("verify_status"), "skipped").lower()
    ok = bool(result.get("ok")) and verify_status not in {"failed", "mismatch", "error"}
    return {
        **result,
        "ok": ok,
        "write_ok": bool(result.get("write_ok")),
        "verify_status": verify_status,
        "verify_ok": verify_status in {"ok", "verified", "skipped"},
        "verify_skipped": verify_status == "skipped",
        "verify_failed": verify_status in {"failed", "mismatch", "error"},
        "backend": _str(result.get("backend") or runtime.get("storage_backend")),
        "last_error_reason": _str(runtime.get("last_error_reason")),
    }


def build_notification_summary(journal: Dict[str, Any]) -> Dict[str, Any]:
    """Build the app-compatible notification journal diagnostic payload."""
    summary = notification_core.summarize_notification_event_journal(_as_dict(journal))
    return {
        "sent": _int(summary.get("sent_count", 0)),
        "skipped_duplicate": _int(summary.get("skipped_count", 0)),
        "failed": _int(summary.get("failed_count", 0)),
        "pending": _int(summary.get("pending_count", 0)),
        "journal_size": _int(summary.get("journal_size", 0)),
        "last_sent_ts": _str(summary.get("last_sent_ts")),
        "last_failed_ts": _str(summary.get("last_failed_ts")),
        "last_failed_reason": _str(summary.get("last_failed_reason")),
        "trimmed": _int(summary.get("trimmed_count", 0)),
    }


def build_monitoring_summary(monitoring_rows: List[Dict[str, Any]], portfolio_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build row-count/status diagnostics without changing portfolio logic."""
    monitoring = _as_rows(monitoring_rows)
    portfolio = _as_rows(portfolio_rows)
    active_statuses = {"", "ACTIVE", "WATCH", "WAIT", "TRACK", "READY", "MONITORING"}
    active_monitoring = [row for row in monitoring if _str(row.get("status")).upper() in active_statuses]
    active_portfolio = [row for row in portfolio if _str(row.get("status"), "ACTIVE").upper() not in {"CLOSED", "ARCHIVED", "SOLD"}]
    return {
        "monitoring_rows": len(monitoring),
        "active_monitoring_rows": len(active_monitoring),
        "portfolio_rows": len(portfolio),
        "active_portfolio_rows": len(active_portfolio),
        "monitoring_status_counts": _top_counts((_str(row.get("status"), "UNKNOWN").upper() for row in monitoring), limit=10),
        "portfolio_status_counts": _top_counts((_str(row.get("status"), "ACTIVE").upper() for row in portfolio), limit=10),
    }


def build_live_pulse_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact, import-safe Live Pulse status payload."""
    data = _as_dict(payload)
    blocked = _as_dict(data.get("blocked_reasons"))
    top_blocked = dict(
        sorted(((str(k), _int(v, 0)) for k, v in blocked.items()), key=lambda item: item[1], reverse=True)[:5]
    )
    final_candidates = data.get("final_candidates") if isinstance(data.get("final_candidates"), list) else []
    status = _str(data.get("status"), "empty").lower()
    return {
        "status": status,
        "ok": status not in {"failed", "error"},
        "empty": _int(data.get("final_count", len(final_candidates))) <= 0,
        "raw_seen": _int(data.get("raw_seen", 0)),
        "normalized": _int(data.get("normalized", 0)),
        "clean_candidates": _int(data.get("clean_candidates", 0)),
        "final_count": _int(data.get("final_count", len(final_candidates))),
        "refill_attempts": _int(data.get("refill_attempts", 0)),
        "last_empty_reason": _str(data.get("last_empty_reason")),
        "sources_tried": list(data.get("sources_tried") or []) if isinstance(data.get("sources_tried"), list) else [],
        "top_blocked_reasons": top_blocked,
    }



_SECRET_KEY_PARTS = ("secret", "token", "password", "passwd", "api_key", "apikey", "private_key", "webhook", "bearer", "authorization")


def _is_secret_key(key: Any) -> bool:
    text = str(key or "").lower()
    return any(part in text for part in _SECRET_KEY_PARTS)


def _public(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded, secret-redacted copy of operator-facing diagnostics."""
    if depth > 3:
        return "…"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(key):
                continue
            out[str(key)] = _public(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_public(item, depth=depth + 1) for item in value[:10]]
    if isinstance(value, tuple):
        return [_public(item, depth=depth + 1) for item in list(value)[:10]]
    if isinstance(value, str):
        text = value.strip()
        if "Traceback (most recent call last)" in text:
            return "traceback redacted"
        return text[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:200]


def _first_dict(ctx: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        value = ctx.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _bounded_dict_list(value: Any, limit: int) -> List[Dict[str, Any]]:
    rows = _as_rows(value)[: max(0, int(limit))]
    return [_public(row) for row in rows if isinstance(_public(row), dict)]


def _status_value(*values: Any, default: str = "unknown") -> str:
    for value in values:
        text = _str(value).lower()
        if text:
            return text
    return default


def _is_good_status(value: Any) -> bool:
    return _status_value(value) in {"ok", "success", "pass", "passed", "green", "verified", "skipped"}


def _is_bad_status(value: Any) -> bool:
    return _status_value(value) in {"failed", "fail", "error", "missing", "degraded", "mismatch"}


def _runtime_diagnostics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    runtime_state = _first_dict(ctx, "runtime_state", "runtime")
    matrix = _first_dict(ctx, "runtime_matrix", "runtime_matrix_summary") or _as_dict(runtime_state.get("runtime_matrix"))
    app_compat = _first_dict(ctx, "app_compat", "app_compat_audit") or _as_dict(runtime_state.get("app_compat"))
    golden = _first_dict(ctx, "golden_fixtures", "golden_fixtures_summary") or _as_dict(runtime_state.get("golden_fixtures"))
    roles = _as_dict(_as_dict(matrix.get("roles")).get("golden_fixtures"))
    jobs = _first_dict(ctx, "last_jobs") or _as_dict(runtime_state.get("last_jobs")) or _as_dict(runtime_state.get("modes"))
    stale_jobs = ctx.get("stale_jobs", runtime_state.get("stale_jobs", []))
    if not isinstance(stale_jobs, list):
        stale_jobs = []
    blocked_cycles = ctx.get("blocked_cycles", runtime_state.get("blocked_cycles", []))
    if not isinstance(blocked_cycles, list):
        blocked_cycles = []
    return {
        "matrix_status": _status_value(matrix.get("status"), "ok" if matrix.get("ok") is True else "failed" if matrix.get("ok") is False else ""),
        "app_compat_status": _status_value(app_compat.get("status"), "ok" if app_compat.get("ok") is True else "failed" if app_compat.get("ok") is False else ""),
        "golden_fixtures_status": _status_value(golden.get("status"), roles.get("status"), "ok" if golden.get("ok") is True or roles.get("ok") is True else "failed" if golden.get("ok") is False or roles.get("ok") is False else ""),
        "last_jobs": _public(jobs),
        "stale_jobs": [_public(item) for item in stale_jobs[:5]],
        "blocked_cycles": [_public(item) for item in blocked_cycles[:5]],
    }


def _scanner_diagnostics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    scanner = _as_dict(ctx.get("scanner_diagnostics"))
    payload = _as_dict(ctx.get("live_pulse_payload") or ctx.get("live_pulse"))
    debug = _as_dict(ctx.get("live_pulse_debug") or payload.get("debug"))
    source_diag = _as_dict(debug.get("source_diagnostics") or scanner.get("source_diagnostics"))
    source_results = debug.get("source_results") if isinstance(debug.get("source_results"), list) else scanner.get("source_results", [])
    if not isinstance(source_results, list):
        source_results = []
    errors = scanner.get("source_errors")
    if not isinstance(errors, list):
        errors = [r.get("error") for r in source_results if isinstance(r, dict) and _str(r.get("error"))]
    tried = scanner.get("sources_tried", payload.get("sources_tried", []))
    if not isinstance(tried, list):
        tried = []
    return {
        "source_status": _status_value(scanner.get("source_status"), source_diag.get("status"), scanner.get("last_scan_status"), payload.get("status")),
        "last_empty_reason": _str(scanner.get("last_empty_reason") or payload.get("last_empty_reason") or debug.get("last_empty_reason")),
        "raw_seen": _int(scanner.get("raw_seen", payload.get("raw_seen", source_diag.get("raw_total", 0)))),
        "normalized": _int(scanner.get("normalized", payload.get("normalized", 0))),
        "clean_candidates": _int(scanner.get("clean_candidates", payload.get("clean_candidates", 0))),
        "final_count": _int(scanner.get("final_count", payload.get("final_count", len(payload.get("final_candidates") if isinstance(payload.get("final_candidates"), list) else [])))),
        "sources_tried": [_str(item) for item in tried[:10] if _str(item)],
        "sources_failed": _int(scanner.get("sources_failed", source_diag.get("sources_failed", 0))),
        "sources_empty": _int(scanner.get("sources_empty", source_diag.get("sources_empty", 0))),
        "sources_disabled": _int(scanner.get("sources_disabled", source_diag.get("sources_disabled", 0))),
        "source_errors": [_str(_public(item)) for item in errors[:3] if _str(_public(item))],
    }


def _priority_diagnostics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    debug = _as_dict(ctx.get("priority_watchlist_debug"))
    rows = ctx.get("priority_watchlist_rows") if isinstance(ctx.get("priority_watchlist_rows"), list) else []
    excluded = _as_dict(debug.get("excluded"))
    wanted = ("no_entry", "archived", "hard_gated", "portfolio_material_conflict", "portfolio_conflict", "missing_score", "below_priority_threshold", "unknown")
    normalized_excluded = {key: _int(excluded.get(key, 0)) for key in wanted}
    if normalized_excluded.get("portfolio_material_conflict", 0) == 0 and normalized_excluded.get("portfolio_conflict", 0):
        normalized_excluded["portfolio_material_conflict"] = normalized_excluded.get("portfolio_conflict", 0)
    return {
        "source_monitoring_rows": _int(debug.get("source_monitoring_rows", debug.get("monitoring_rows", 0))),
        "active_monitoring_rows": _int(debug.get("active_monitoring_rows", 0)),
        "final_priority_rows": _int(debug.get("final_priority_rows", len(rows))),
        "eligible_watch_early_rows": _int(debug.get("eligible_watch_early_rows", 0)),
        "top_excluded_samples": _bounded_dict_list(debug.get("top_excluded_samples"), 3),
        "excluded": normalized_excluded,
    }


def _storage_diagnostics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    storage = _as_dict(ctx.get("storage") or ctx.get("storage_result") or ctx.get("storage_summary"))
    runtime_state = _first_dict(ctx, "runtime_state", "runtime")
    backend = _status_value(storage.get("backend"), ctx.get("storage_backend"), runtime_state.get("storage_backend"), default="unknown")
    if backend not in {"d1", "supabase", "local", "unknown"}:
        backend = "unknown"
    verify = _status_value(storage.get("verify_status"), runtime_state.get("storage_verify_status"), default="unknown")
    return {
        "backend": backend,
        "last_read_ok": storage.get("last_read_ok") if isinstance(storage.get("last_read_ok"), bool) else storage.get("read_ok") if isinstance(storage.get("read_ok"), bool) else storage.get("ok") if isinstance(storage.get("ok"), bool) else None,
        "last_write_ok": storage.get("last_write_ok") if isinstance(storage.get("last_write_ok"), bool) else storage.get("write_ok") if isinstance(storage.get("write_ok"), bool) else None,
        "verify_status": verify if verify in {"ok", "skipped", "failed", "unknown"} else ("ok" if _is_good_status(verify) else "failed" if _is_bad_status(verify) else "unknown"),
        "last_summary": _str(storage.get("summary") or storage.get("last_diag_summary") or runtime_state.get("last_storage_diag_summary")),
    }


def build_operator_diagnostics(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build read-only operator diagnostics from already-loaded UI context.

    This pure helper performs no storage reads/writes, scanner runs, network calls,
    Telegram sends, or Streamlit imports. It tolerates legacy/missing context keys
    and returns bounded, secret-redacted diagnostics for compact UI panels.
    """
    ctx = _as_dict(context)
    runtime = _runtime_diagnostics(ctx)
    scanner = _scanner_diagnostics(ctx)
    priority = _priority_diagnostics(ctx)
    storage = _storage_diagnostics(ctx)

    has_any = any([ctx.get("runtime_state"), ctx.get("scanner_diagnostics"), ctx.get("live_pulse_payload"), ctx.get("priority_watchlist_debug"), ctx.get("storage_result"), ctx.get("storage")])
    degraded_reasons: List[str] = []
    warning_reasons: List[str] = []

    runtime_available = any(key in ctx for key in ("runtime_state", "runtime", "runtime_matrix", "runtime_matrix_summary", "app_compat", "app_compat_audit", "golden_fixtures", "golden_fixtures_summary"))
    if runtime_available and (_is_bad_status(runtime.get("matrix_status")) or runtime.get("matrix_status") in {"unknown", "missing"}):
        degraded_reasons.append("runtime_matrix unavailable")
    if runtime_available and (_is_bad_status(runtime.get("app_compat_status")) or runtime.get("app_compat_status") in {"unknown", "missing"}):
        degraded_reasons.append("app_compat unavailable")
    if runtime_available and _is_bad_status(runtime.get("golden_fixtures_status")):
        degraded_reasons.append("golden_fixtures failed")
    if runtime.get("stale_jobs"):
        degraded_reasons.append("critical jobs stale")

    source_status = str(scanner.get("source_status") or "unknown").lower()
    if source_status in {"source_api_failed", "failed", "error"}:
        degraded_reasons.append("source_api_failed")
    elif scanner.get("sources_failed", 0) and scanner.get("final_count", 0) > 0:
        warning_reasons.append("partial source failure with candidates")
    elif scanner.get("final_count", 0) <= 0 and scanner.get("last_empty_reason"):
        warning_reasons.append(f"Live Pulse empty: {scanner.get('last_empty_reason')}")

    if priority.get("final_priority_rows", 0) <= 0 and (sum(_as_dict(priority.get("excluded")).values()) > 0 or priority.get("eligible_watch_early_rows", 0) > 0):
        warning_reasons.append("Priority empty with known exclusions")

    if storage.get("verify_status") == "failed" or storage.get("last_read_ok") is False:
        degraded_reasons.append("storage verification failed")

    if not has_any:
        status = "unknown"
        summary = "Diagnostics not available yet."
    elif degraded_reasons:
        status = "degraded"
        summary = "; ".join(degraded_reasons[:3])
    elif warning_reasons:
        status = "warning"
        summary = "; ".join(warning_reasons[:3])
    else:
        status = "ok"
        summary = "Runtime, scanner, priority, and storage diagnostics look ok."

    return {
        "status": status,
        "summary": _public(summary),
        "runtime": runtime,
        "scanner": scanner,
        "priority": priority,
        "storage": storage,
    }

def build_operator_status(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build status-card data for operators from already-loaded context."""
    ctx = _as_dict(context)
    runtime_summary = build_runtime_summary(ctx)
    storage_summary = build_storage_summary(_as_dict(ctx.get("storage_result")), _as_dict(ctx.get("runtime_state")))
    monitoring_summary = build_monitoring_summary(_as_rows(ctx.get("monitoring_rows")), _as_rows(ctx.get("portfolio_rows")))
    live_pulse_summary = build_live_pulse_summary(_as_dict(ctx.get("live_pulse")))
    notification_summary = build_notification_summary(_as_dict(ctx.get("notification_journal")))
    return {
        "runtime": runtime_summary,
        "storage": storage_summary,
        "monitoring": monitoring_summary,
        "live_pulse": live_pulse_summary,
        "notifications": notification_summary,
        "ok": bool(storage_summary.get("ok")) and bool(live_pulse_summary.get("ok")),
    }
