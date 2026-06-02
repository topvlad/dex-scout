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
