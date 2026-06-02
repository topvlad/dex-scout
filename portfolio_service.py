"""Import-safe Portfolio action/state helpers.

This module contains pure Portfolio business logic only. It deliberately avoids
Streamlit, app.py, storage, network, and Telegram side effects so runtime jobs,
webhooks, tests, and UI adapters can import it safely.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import monitoring_core
import notification_core

CANONICAL_MATERIAL_PORTFOLIO_ACTIONS = {
    "EXIT",
    "REDUCE",
    "TAKE PROFIT",
    "PARTIAL_TP",
    "PROTECT",
    "CLOSE",
    "TRIM",
}
NON_MATERIAL_PORTFOLIO_ACTIONS = {"", "HOLD", "WAIT", "WATCH", "EARLY", "NO_ENTRY", "NO ENTRY"}
ACTION_FIELD_CANDIDATES = (
    "final_action",
    "recommended_action",
    "position_action",
    "entry_action",
    "action",
    "bucket",
    "verdict",
)
PRICE_FIELD_CANDIDATES = (
    "current_price",
    "price_usd",
    "priceUsd",
    "price",
    "last_price",
    "pair_price",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _action_value(value: str | Dict[str, Any] | Any) -> Any:
    if isinstance(value, dict):
        for key in ACTION_FIELD_CANDIDATES:
            candidate = value.get(key)
            if _text(candidate):
                return candidate
        row = value.get("row") if isinstance(value.get("row"), dict) else {}
        for key in ACTION_FIELD_CANDIDATES:
            candidate = row.get(key)
            if _text(candidate):
                return candidate
        return ""
    return value


def normalize_portfolio_action(value: str | Dict[str, Any] | Any) -> str:
    """Normalize raw Portfolio action input while preserving display labels."""
    raw = _text(_action_value(value))
    if not raw:
        return ""
    token = re.sub(r"[\s\-]+", "_", raw.upper().strip())
    token = re.sub(r"_+", "_", token).strip("_")
    aliases = {
        "TAKE_PROFIT": "TAKE PROFIT",
        "TAKEPROFIT": "TAKE PROFIT",
        "PARTIAL_TP": "PARTIAL_TP",
        "PARTIAL_TAKE_PROFIT": "PARTIAL_TP",
        "PARTIAL_TAKEPROFIT": "PARTIAL_TP",
        "PARTIAL_PROFIT": "PARTIAL_TP",
        "NO_ENTRY": "NO_ENTRY",
        "NOENTRY": "NO_ENTRY",
    }
    if token in aliases:
        return aliases[token]
    if token in CANONICAL_MATERIAL_PORTFOLIO_ACTIONS:
        return token
    if token in {"HOLD", "WAIT", "WATCH", "EARLY"}:
        return token
    return token.replace("_", " ")


def is_material_portfolio_action(row_or_action: str | Dict[str, Any] | Any) -> bool:
    """Return True for canonical material Portfolio actions, independent of price."""
    return normalize_portfolio_action(row_or_action) in CANONICAL_MATERIAL_PORTFOLIO_ACTIONS


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().replace("$", "").replace(",", "")
            if not value:
                return None
        return float(value)
    except Exception:
        return None


def _has_price(row: Dict[str, Any], market_snapshot: Optional[Dict[str, Any]] = None) -> bool:
    for source in (row or {}, market_snapshot or {}):
        for key in PRICE_FIELD_CANDIDATES:
            parsed = _parse_float(source.get(key))
            if parsed is not None and parsed > 0:
                return True
    return False


def _identity(row: Dict[str, Any]) -> Dict[str, str]:
    try:
        return monitoring_core.build_token_identity(row or {})
    except Exception:
        row = row or {}
        return {
            "symbol": _text(row.get("symbol") or row.get("base_symbol") or row.get("token")),
            "chain": _text(row.get("chain") or row.get("chainId")).lower(),
            "token_addr": _text(row.get("token_addr") or row.get("base_token_address") or row.get("base_addr") or row.get("ca") or row.get("address")),
            "pair_addr": _text(row.get("pair_address") or row.get("pairAddress") or row.get("pair")),
            "token_key": "",
        }


def classify_portfolio_row(row: Dict[str, Any], market_snapshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Classify a Portfolio row without recomputing recommendation math."""
    row = row or {}
    identity = _identity(row)
    action = normalize_portfolio_action(row)
    material = is_material_portfolio_action(action)
    missing_price = not _has_price(row, market_snapshot)
    flags: List[str] = []
    if missing_price:
        flags.append("missing_price")
    if material:
        urgency = "action_now"
        reason = "material_portfolio_action"
        if missing_price:
            reason = "material_portfolio_action; numeric levels unavailable because price is missing"
    elif action in {"HOLD", "WAIT", "WATCH", "EARLY"}:
        urgency = "hold" if action == "HOLD" else "watch"
        reason = "non_material_portfolio_action"
    elif not action:
        urgency = "unknown"
        reason = "missing_portfolio_action"
        flags.append("missing_action")
    else:
        urgency = "unknown"
        reason = "non_material_portfolio_action"
    return {
        "symbol": identity.get("symbol") or _text(row.get("symbol") or row.get("base_symbol") or row.get("token")),
        "chain": identity.get("chain") or _text(row.get("chain") or row.get("chainId")).lower(),
        "token_addr": identity.get("token_addr") or _text(row.get("token_addr") or row.get("base_token_address") or row.get("base_addr") or row.get("ca") or row.get("address")),
        "action": action,
        "is_material": bool(material),
        "urgency": urgency,
        "reason": reason,
        "missing_price": bool(missing_price),
        "display_label": action or "UNKNOWN",
        "diagnostic_flags": flags,
    }


def _monitoring_action(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    try:
        return monitoring_core.monitoring_display_action(row or {})
    except Exception:
        return _text((row or {}).get("final_action") or (row or {}).get("entry_action") or (row or {}).get("entry_status") or (row or {}).get("status")).upper()


def resolve_portfolio_monitoring_conflict(
    portfolio_row: Dict[str, Any] | None,
    monitoring_row: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Resolve Portfolio/Monitoring display state with Portfolio material actions primary."""
    pf_class = classify_portfolio_row(portfolio_row or {}) if portfolio_row else {}
    pf_action = str(pf_class.get("action") or "")
    material = bool(pf_class.get("is_material"))
    mon_action = _monitoring_action(monitoring_row) or ""
    mon_status = _text((monitoring_row or {}).get("entry_status") or (monitoring_row or {}).get("status") or mon_action).upper() if monitoring_row else ""
    mon_watch = mon_action in monitoring_core.MONITORING_WATCH_ACTIONS or mon_status in monitoring_core.MONITORING_WATCH_ACTIONS

    if portfolio_row and material:
        return {
            "has_conflict": bool(monitoring_row and mon_watch),
            "display_action": pf_action,
            "display_status": pf_action,
            "source": "portfolio",
            "reason": "material_portfolio_action_overrides_monitoring_watch" if monitoring_row and mon_watch else "material_portfolio_action",
            "material_portfolio_action": True,
        }
    if monitoring_row:
        return {
            "has_conflict": False,
            "display_action": mon_action or mon_status or "WATCH",
            "display_status": mon_status or mon_action or "WATCH",
            "source": "monitoring",
            "reason": "monitoring_state",
            "material_portfolio_action": False,
        }
    return {
        "has_conflict": False,
        "display_action": pf_action or "",
        "display_status": pf_action or "",
        "source": "portfolio" if portfolio_row else "none",
        "reason": "portfolio_state" if portfolio_row else "no_state",
        "material_portfolio_action": False,
    }


def _find_monitoring_match(row: Dict[str, Any], monitoring_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for monitoring_row in monitoring_rows or []:
        try:
            if monitoring_core.row_matches_token(row or {}, monitoring_row or {}):
                return monitoring_row
        except Exception:
            continue
    return {}


def _candidate_key(row: Dict[str, Any], action: str) -> str:
    ident = _identity(row or {})
    base = ident.get("token_key") or f"{ident.get('chain')}:{ident.get('token_addr') or ident.get('pair_addr') or ident.get('symbol')}"
    return f"{base}:{action}"


def build_portfolio_notification_candidates(
    portfolio_rows: list[dict],
    monitoring_rows: list[dict] | None = None,
    now_ts: str | None = None,
) -> dict:
    """Build import-safe material Portfolio notification candidates and diagnostics."""
    rows = list(portfolio_rows or [])
    monitoring = list(monitoring_rows or [])
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    diagnostics = {
        "rows_seen": len(rows),
        "missing_price_material": 0,
        "non_material": 0,
        "duplicates": 0,
    }
    for row in rows:
        classification = classify_portfolio_row(row or {})
        if not classification.get("is_material"):
            diagnostics["non_material"] += 1
            continue
        if classification.get("missing_price"):
            diagnostics["missing_price_material"] += 1
        key = _candidate_key(row or {}, str(classification.get("action") or ""))
        if key in seen:
            diagnostics["duplicates"] += 1
            continue
        seen.add(key)
        monitoring_row = _find_monitoring_match(row or {}, monitoring)
        conflict = resolve_portfolio_monitoring_conflict(row or {}, monitoring_row or None)
        candidates.append(
            {
                "source": "portfolio",
                "row": row,
                "action": classification.get("action") or "",
                "classification": classification,
                "conflict": conflict,
                "event_ts": now_ts or "",
                "candidate_key": key,
                "reason": classification.get("reason") or "material_portfolio_action",
            }
        )
    material_count = len(candidates)
    blocked = material_count > 0
    return {
        "material_count": material_count,
        "candidates": candidates,
        "blocked_no_urgent_heartbeat": blocked,
        "blocked_reason": "material_portfolio_actions_present" if blocked else "",
        "diagnostics": diagnostics,
    }


# Explicit aliases for callers that need to compare with existing core helpers.
NOTIFICATION_MATERIAL_PORTFOLIO_ACTIONS = notification_core.MATERIAL_PORTFOLIO_ACTIONS
