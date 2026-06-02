"""Import-safe Monitoring archive/priority service helpers.

This module owns pure Monitoring row classification and archive-transition
planning.  It deliberately avoids Streamlit, app.py, storage/network writes, and
Telegram side effects so tests and runtime preflight checks can import it safely.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import monitoring_core
import portfolio_service

CANONICAL_STATUSES = {"WATCH", "EARLY", "WAIT", "NO_ENTRY", "ARCHIVED", "DEAD", "BLOCKED", "UNKNOWN"}
ACTIVE_ACTIONABLE_STATUSES = {"WATCH", "EARLY"}
INACTIVE_STATUSES = {"NO_ENTRY", "ARCHIVED", "DEAD", "BLOCKED"}
_STATUS_FIELDS = (
    "monitoring_status",
    "entry_status",
    "entry_action",
    "final_action",
    "action",
    "state",
    "verdict",
    "status",
    "lifecycle",
)


def _norm_text(value: Any) -> str:
    return "_".join(str(value or "").strip().upper().replace("-", "_").split())


def _canonical_status(value: Any) -> str:
    normalized = _norm_text(value)
    if not normalized:
        return "UNKNOWN"
    aliases = {
        "WATCH": "WATCH",
        "WATCHING": "WATCH",
        "WATCH_PULLBACK": "WATCH",
        "READY": "WATCH",
        "TRACK": "WATCH",
        "MONITORING": "WATCH",
        "EARLY": "EARLY",
        "EARLY_ENTRY": "EARLY",
        "WAIT": "WAIT",
        "HOLD": "WAIT",
        "NO_ENTRY": "NO_ENTRY",
        "NOENTRY": "NO_ENTRY",
        "AVOID": "NO_ENTRY",
        "ARCHIVE": "ARCHIVED",
        "ARCHIVED": "ARCHIVED",
        "CLOSED": "ARCHIVED",
        "INACTIVE": "ARCHIVED",
        "DEAD": "DEAD",
        "UNTRADEABLE": "DEAD",
        "BLOCK": "BLOCKED",
        "BLOCKED": "BLOCKED",
        "HARD_GATED": "BLOCKED",
    }
    return aliases.get(normalized, "UNKNOWN")


def normalize_monitoring_status(value: str | dict) -> str:
    """Return a canonical Monitoring status for a raw string or row dict."""
    if isinstance(value, dict):
        # Prefer explicit inactive states if present so rows like
        # {status: ACTIVE, entry_status: NO_ENTRY} do not look actionable.
        candidates = [value.get(field) for field in _STATUS_FIELDS]
        canonical = [_canonical_status(candidate) for candidate in candidates if str(candidate or "").strip()]
        for status in canonical:
            if status in INACTIVE_STATUSES:
                return status
        for status in canonical:
            if status in {"WATCH", "EARLY", "WAIT"}:
                return status
        return "UNKNOWN"
    return _canonical_status(value)


def _symbol(row: Dict[str, Any]) -> str:
    return str(row.get("base_symbol") or row.get("symbol") or row.get("token") or "").strip()


def _identity(row: Dict[str, Any]) -> Dict[str, str]:
    return monitoring_core.build_token_identity(row or {})


def _copy_row(row: Dict[str, Any], *, score: Optional[float] = None, score_field: str = "") -> Dict[str, Any]:
    copied = dict(row or {})
    if score is not None:
        copied["priority_score"] = score
        copied["priority_score_field"] = score_field
    return copied


def _portfolio_conflict(row: Dict[str, Any], portfolio_row: Optional[Dict[str, Any]]) -> bool:
    if not portfolio_row:
        return False
    try:
        state = portfolio_service.resolve_portfolio_monitoring_conflict(portfolio_row, row)
    except Exception:
        state = monitoring_core.resolve_monitoring_portfolio_state(row, portfolio_row)
    return bool((state.get("is_conflict") or state.get("has_conflict")) and state.get("material_portfolio_action"))


def classify_monitoring_row(
    row: dict,
    portfolio_row: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Classify one Monitoring row without mutating it or performing IO."""
    row = dict(row or {})
    ident = _identity(row)
    status = normalize_monitoring_status(row)
    active_flag = str(row.get("active", row.get("is_active", "1"))).strip().lower()
    lifecycle = _norm_text(row.get("lifecycle"))
    archived_reason = str(row.get("archived_reason") or "").strip()

    is_archived = (
        active_flag in monitoring_core.INACTIVE_FLAGS
        or status == "ARCHIVED"
        or lifecycle in {"ARCHIVED", "CLOSED", "INACTIVE"}
        or bool(archived_reason)
    )
    is_no_entry = status == "NO_ENTRY"
    is_dead = status == "DEAD"

    gate = monitoring_core.hard_gate_monitoring_row(row, portfolio_row=portfolio_row, history=history)
    hard_gated = bool(gate.get("blocked"))
    portfolio_conflict = _portfolio_conflict(row, portfolio_row) or "portfolio_verdict_conflict" in str(gate.get("reason") or "")

    watch_early = status in ACTIVE_ACTIONABLE_STATUSES
    is_active = bool(watch_early and not is_archived and not is_no_entry and not is_dead)
    is_actionable = bool(is_active and not hard_gated and not portfolio_conflict)
    score, score_field = monitoring_core.priority_score_value(row)
    priority_eligible = bool(is_actionable and score is not None and score > 0)

    archive_action = "KEEP"
    if is_no_entry:
        archive_action = "NO_ENTRY"
    if is_archived or is_dead or str(gate.get("action") or "").upper() == "ARCHIVE":
        archive_action = "ARCHIVE"
    elif hard_gated and str(gate.get("action") or "").upper() == "NO_ENTRY":
        archive_action = "NO_ENTRY"

    flags = list(gate.get("flags") or [])
    if is_archived:
        flags.append("archived")
    if is_no_entry:
        flags.append("no_entry")
    if portfolio_conflict:
        flags.append("portfolio_conflict")
    if score is None:
        flags.append("missing_score")

    reason = str(gate.get("reason") or "").strip()
    if not reason:
        if is_archived:
            reason = archived_reason or "archived"
        elif is_no_entry:
            reason = "no_entry"
        elif portfolio_conflict:
            reason = "portfolio_conflict"
        elif not watch_early:
            reason = "not_watch_early"
        elif score is None:
            reason = "missing_score"
        elif score <= 0:
            reason = "below_priority_threshold"
        else:
            reason = "eligible"

    return {
        "symbol": ident.get("symbol") or _symbol(row),
        "chain": ident.get("chain") or str(row.get("chain") or "").strip().lower(),
        "token_addr": ident.get("token_addr") or str(row.get("base_addr") or row.get("token_addr") or "").strip(),
        "status": status,
        "is_active": is_active,
        "is_actionable": is_actionable,
        "is_archived": is_archived,
        "is_no_entry": is_no_entry,
        "hard_gated": hard_gated,
        "portfolio_conflict": portfolio_conflict,
        "priority_eligible": priority_eligible,
        "archive_action": archive_action,
        "reason": reason,
        "diagnostic_flags": sorted(set(str(flag) for flag in flags if str(flag).strip())),
        "priority_score": score,
        "priority_score_field": score_field,
    }


def _portfolio_index(portfolio_rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in portfolio_rows or []:
        if not monitoring_core.is_active_portfolio_row(row):
            continue
        ident = _identity(row)
        if ident.get("token_key"):
            index[ident["token_key"]] = row
    return index


def _find_portfolio_row(row: Dict[str, Any], portfolio_rows: List[Dict[str, Any]], index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    key = _identity(row).get("token_key")
    if key and key in index:
        return index[key]
    return monitoring_core.find_active_portfolio_row(row, portfolio_rows)


def _empty_priority_debug(source_count: int = 0) -> Dict[str, Any]:
    return monitoring_core.empty_priority_watchlist_debug(source_count)


def plan_monitoring_archive_transitions(
    monitoring_rows: list[dict],
    portfolio_rows: list[dict] | None = None,
    history_by_token: dict | None = None,
    *,
    auto_archive_enabled: bool = True,
) -> dict:
    """Plan Monitoring keep/archive/no-entry/priority groups without writes."""
    rows = list(monitoring_rows or [])
    portfolios = list(portfolio_rows or [])
    history_by_token = history_by_token or {}
    portfolio_index = _portfolio_index(portfolios)

    keep_rows: List[Dict[str, Any]] = []
    archive_candidates: List[Dict[str, Any]] = []
    no_entry_rows: List[Dict[str, Any]] = []
    priority_rows: List[Dict[str, Any]] = []
    diagnostics = {
        "active": 0,
        "watch_early": 0,
        "no_entry": 0,
        "archived": 0,
        "hard_gated": 0,
        "portfolio_conflict": 0,
        "priority_eligible": 0,
        "archive_candidates": 0,
    }
    reasons: Counter[str] = Counter()

    for original in rows:
        row = dict(original or {})
        ident = _identity(row)
        portfolio_row = _find_portfolio_row(row, portfolios, portfolio_index)
        history = history_by_token.get(ident.get("token_key")) if isinstance(history_by_token, dict) else None
        classified = classify_monitoring_row(row, portfolio_row=portfolio_row, history=history if isinstance(history, list) else None)
        reason = str(classified.get("reason") or "unknown")
        reasons[reason] += 1

        if classified["is_active"]:
            diagnostics["active"] += 1
        if classified["status"] in ACTIVE_ACTIONABLE_STATUSES:
            diagnostics["watch_early"] += 1
        if classified["is_no_entry"]:
            diagnostics["no_entry"] += 1
        if classified["is_archived"] or classified["status"] == "DEAD":
            diagnostics["archived"] += 1
        if classified["hard_gated"]:
            diagnostics["hard_gated"] += 1
        if classified["portfolio_conflict"]:
            diagnostics["portfolio_conflict"] += 1
        if classified["priority_eligible"]:
            diagnostics["priority_eligible"] += 1
            copied = _copy_row(row, score=classified.get("priority_score"), score_field=str(classified.get("priority_score_field") or ""))
            if portfolio_row:
                copied["is_portfolio_active"] = True
            priority_rows.append(copied)

        action = str(classified.get("archive_action") or "KEEP")
        annotated = _copy_row(row)
        annotated["monitoring_status"] = classified["status"]
        annotated["archive_action"] = action
        annotated["archive_reason"] = reason
        annotated["diagnostic_flags"] = list(classified.get("diagnostic_flags") or [])
        if action == "NO_ENTRY":
            no_entry_rows.append(annotated)
        elif action == "ARCHIVE":
            archive_candidates.append(annotated)
        else:
            keep_rows.append(annotated)

    priority_rows.sort(key=lambda item: float(item.get("priority_score") or 0.0), reverse=True)
    diagnostics["archive_candidates"] = len(archive_candidates)
    return {
        "rows_seen": len(rows),
        "keep_rows": keep_rows,
        "archive_candidates": archive_candidates if auto_archive_enabled else archive_candidates,
        "no_entry_rows": no_entry_rows,
        "priority_rows": priority_rows,
        "diagnostics": diagnostics,
        "top_reasons": dict(reasons.most_common(10)),
    }


def build_priority_watchlist_rows(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: Optional[List[Dict[str, Any]]] = None,
    *,
    max_samples: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compatibility helper returning Priority rows plus legacy debug shape."""
    plan = plan_monitoring_archive_transitions(monitoring_rows, portfolio_rows)
    debug = _empty_priority_debug(int(plan.get("rows_seen") or 0))
    diagnostics = plan.get("diagnostics") if isinstance(plan.get("diagnostics"), dict) else {}
    debug["active_monitoring_rows"] = int(diagnostics.get("active", 0) or 0)
    debug["eligible_watch_early_rows"] = int(diagnostics.get("watch_early", 0) or 0)
    debug["final_priority_rows"] = len(plan.get("priority_rows") or [])
    debug["monitoring_archive_diagnostics"] = {"rows_seen": int(plan.get("rows_seen") or 0), **diagnostics}

    excluded = debug["excluded"]
    for reason, count in (plan.get("top_reasons") or {}).items():
        mapped = reason
        if "portfolio" in reason:
            mapped = "portfolio_material_conflict"
        elif reason in {"no_entry"}:
            mapped = "no_entry"
        elif reason in {"archived", "dead"}:
            mapped = "archived"
        elif reason in {"missing_score", "below_priority_threshold", "eligible"}:
            mapped = reason
        elif reason != "eligible":
            mapped = "hard_gated" if any(x in reason for x in ("liquidity", "scam", "untradeable", "collapse")) else "unknown"
        if mapped in excluded and mapped != "eligible":
            excluded[mapped] = int(excluded.get(mapped, 0) or 0) + int(count or 0)

    samples = debug["top_excluded_samples"]
    priority_keys = {monitoring_core.token_key(row.get("chain"), row.get("base_addr") or row.get("base_token_address") or row.get("token_addr")) for row in plan.get("priority_rows") or []}
    for row in monitoring_rows or []:
        if len(samples) >= max_samples:
            break
        key = monitoring_core.token_key(row.get("chain"), row.get("base_addr") or row.get("base_token_address") or row.get("token_addr"))
        if key and key in priority_keys:
            continue
        cls = classify_monitoring_row(row, portfolio_row=monitoring_core.find_active_portfolio_row(row, portfolio_rows or []))
        if cls.get("reason") == "eligible":
            continue
        score, _ = monitoring_core.priority_score_value(row)
        samples.append({
            "symbol": _symbol(row) or "UNKNOWN",
            "status": cls.get("status") or "UNKNOWN",
            "score": "" if score is None else f"{score:.2f}",
            "reason": cls.get("reason") or "unknown",
        })
    return list(plan.get("priority_rows") or []), debug
