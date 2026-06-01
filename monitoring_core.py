"""Import-safe Monitoring/Portfolio row state helpers.

This module contains pure row/state logic only. It deliberately avoids Streamlit,
app.py, storage, network, and environment side effects so tests and workers can
import it safely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

EVM_CHAINS = {"bsc", "eth", "ethereum", "polygon", "arbitrum", "optimism", "base", "avalanche"}
TOKEN_ADDR_ALIASES = ("base_token_address", "base_addr", "token_addr", "token_address", "ca", "address")
PAIR_ADDR_ALIASES = ("pair_address", "pairAddress", "pair")
MATERIAL_PORTFOLIO_ACTIONS = {"EXIT", "REDUCE", "TAKE PROFIT", "TAKE_PROFIT", "PARTIAL_TP", "PROTECT", "CLOSE", "TRIM"}
NON_MATERIAL_PORTFOLIO_ACTIONS = {"", "HOLD", "WAIT", "WATCH", "WATCH CLOSELY", "HOLD / WATCH CAREFULLY"}
MONITORING_WATCH_ACTIONS = {"WATCH", "EARLY", "WATCH_PULLBACK", "READY", "ACTIVE", "TRACK", "WAIT"}
INACTIVE_FLAGS = {"0", "false", "no", "n", "inactive", "archived", "closed"}
ARCHIVED_STATUSES = {"ARCHIVED", "DEAD", "CLOSED", "INACTIVE", "NO_ENTRY", "NO ENTRY", "AVOID", "UNTRADEABLE"}
ACTIVE_MONITORING_STATUSES = {"", "ACTIVE", "WATCH", "WAIT", "TRACK", "READY", "MONITORING", "EARLY", "WATCH_PULLBACK"}


def _first(row: Dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None and key == "base_addr":
            base_token = row.get("baseToken")
            if isinstance(base_token, dict):
                value = base_token.get("address")
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace("$", "").replace(",", "")
            if not value:
                return default
        return float(value)
    except Exception:
        return default


def normalize_chain_name(raw_chain: Any) -> str:
    chain = str(raw_chain or "").strip().lower()
    mapping = {"binance": "bsc", "binance-smart-chain": "bsc", "bnb": "bsc", "ethereum": "eth"}
    return mapping.get(chain, chain)


def normalize_token_address_for_chain(chain: Any, addr: Any) -> str:
    chain_norm = normalize_chain_name(chain)
    text = str(addr or "").strip()
    if not text:
        return ""
    if chain_norm in EVM_CHAINS and chain_norm != "solana":
        return text.lower()
    return text


def token_key(chain: Any, addr: Any) -> str:
    chain_norm = normalize_chain_name(chain)
    addr_norm = normalize_token_address_for_chain(chain_norm, addr)
    return f"{chain_norm}:{addr_norm}" if chain_norm and addr_norm else ""


def build_token_identity(row: Dict[str, Any]) -> Dict[str, str]:
    row = row or {}
    chain = normalize_chain_name(row.get("chain") or row.get("chainId"))
    token_addr_raw = _first(row, TOKEN_ADDR_ALIASES)
    pair_addr_raw = _first(row, PAIR_ADDR_ALIASES)
    token_addr = normalize_token_address_for_chain(chain, token_addr_raw)
    pair_addr = normalize_token_address_for_chain(chain, pair_addr_raw)
    symbol = str(row.get("symbol") or row.get("base_symbol") or row.get("token") or "").strip()
    key_addr = token_addr or pair_addr
    return {
        "chain": chain,
        "token_addr": token_addr,
        "pair_addr": pair_addr,
        "symbol": symbol,
        "token_key": token_key(chain, key_addr),
    }


def row_matches_token(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_id = build_token_identity(left or {})
    right_id = build_token_identity(right or {})
    if left_id["token_key"] and right_id["token_key"]:
        return left_id["token_key"] == right_id["token_key"]
    if left_id["chain"] and right_id["chain"] and left_id["chain"] != right_id["chain"]:
        return False
    return bool(left_id["symbol"] and right_id["symbol"] and left_id["symbol"].lower() == right_id["symbol"].lower())


def normalize_material_portfolio_action(value: Any) -> str:
    action = str(value or "").strip().upper().replace("-", "_")
    action_us = "_".join(action.split())
    if action_us == "TAKE_PROFIT":
        return "TAKE PROFIT"
    if action_us in MATERIAL_PORTFOLIO_ACTIONS:
        return action_us
    return action.replace("_", " ") if action_us in {"HOLD", "WAIT"} else action


def is_material_portfolio_action(row_or_action: Any) -> bool:
    if isinstance(row_or_action, dict):
        value = row_or_action.get("final_action") or row_or_action.get("recommended_action") or row_or_action.get("position_action") or row_or_action.get("action")
    else:
        value = row_or_action
    return normalize_material_portfolio_action(value) in {"EXIT", "REDUCE", "TAKE PROFIT", "PARTIAL_TP", "PROTECT", "CLOSE", "TRIM"}


def monitoring_display_action(row: Dict[str, Any]) -> str:
    return str(row.get("final_action") or row.get("entry_action") or row.get("entry_status") or row.get("status") or "").strip().upper()


def resolve_monitoring_portfolio_state(monitoring_row: Dict[str, Any], portfolio_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mon_action = monitoring_display_action(monitoring_row or {}) or "WATCH"
    display_status = str((monitoring_row or {}).get("entry_status") or (monitoring_row or {}).get("status") or mon_action).strip().upper()
    pf_action = normalize_material_portfolio_action(
        (portfolio_row or {}).get("final_action") or (portfolio_row or {}).get("recommended_action") or (portfolio_row or {}).get("position_action") or (portfolio_row or {}).get("action")
    ) if portfolio_row else ""
    material = is_material_portfolio_action(pf_action)
    mon_watch = mon_action in MONITORING_WATCH_ACTIONS or display_status in MONITORING_WATCH_ACTIONS
    if portfolio_row and material and mon_watch:
        return {
            "display_status": pf_action,
            "display_action": pf_action,
            "source": "conflict_override",
            "reason": "material_portfolio_action_overrides_monitoring_watch",
            "is_conflict": True,
            "material_portfolio_action": True,
        }
    if portfolio_row and material:
        return {
            "display_status": pf_action,
            "display_action": pf_action,
            "source": "portfolio",
            "reason": "material_portfolio_action",
            "is_conflict": False,
            "material_portfolio_action": True,
        }
    return {
        "display_status": display_status,
        "display_action": mon_action,
        "source": "monitoring",
        "reason": "monitoring_state",
        "is_conflict": False,
        "material_portfolio_action": False,
    }


def is_active_portfolio_row(row: Dict[str, Any]) -> bool:
    row = row or {}
    active = str(row.get("active", row.get("is_active", "1"))).strip().lower()
    if active in INACTIVE_FLAGS:
        return False
    status = str(row.get("status") or row.get("lifecycle") or "").strip().upper()
    if status in {"ARCHIVED", "CLOSED", "INACTIVE"}:
        return False
    return bool(build_token_identity(row)["token_key"] or row.get("base_symbol") or row.get("symbol"))


def is_active_monitoring_row(row: Dict[str, Any]) -> bool:
    row = row or {}
    active = str(row.get("active", row.get("is_active", "1"))).strip().lower()
    if active in INACTIVE_FLAGS:
        return False
    status = str(row.get("entry_status") or row.get("status") or "").strip().upper()
    lifecycle = str(row.get("lifecycle") or "").strip().upper()
    action = str(row.get("final_action") or row.get("entry_action") or row.get("entry") or "").strip().upper()
    reason = str(row.get("archived_reason") or "").strip()
    if reason or lifecycle in ARCHIVED_STATUSES or status in ARCHIVED_STATUSES or action in {"NO_ENTRY", "NO ENTRY", "AVOID"}:
        return False
    if status and status not in ACTIVE_MONITORING_STATUSES:
        return False
    if hard_gate_monitoring_row(row).get("blocked"):
        return False
    return True


def hard_gate_monitoring_row(row: Dict[str, Any], portfolio_row: Optional[Dict[str, Any]] = None, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    _ = history
    row = row or {}
    reasons: List[str] = []
    flags: List[str] = []
    health = str(row.get("health_label") or "").upper()
    status = str(row.get("entry_status") or row.get("status") or "").upper()
    action = str(row.get("final_action") or row.get("entry_action") or "").upper()
    liq = _parse_float(row.get("liq_usd", row.get("liquidity", row.get("liquidity_usd", row.get("liq_init", 0)))), 0.0)
    vol24 = _parse_float(row.get("vol24_usd", row.get("vol24_init", 0)), 0.0)
    pc24 = _parse_float(row.get("pc24h", row.get("price_change_h24", row.get("pc24", 0))), 0.0)
    pc1h = _parse_float(row.get("pc1h", row.get("price_change_h1", 0)), 0.0)
    reason_text = " ".join([
        health, status, action, str(row.get("entry_reason") or ""), str(row.get("archived_reason") or ""),
        str(row.get("lifecycle") or ""), str(row.get("risk_flags") or ""), str(row.get("toxic_flags") or ""), str(row.get("why") or ""),
    ]).upper()
    if "POST_PUMP_COLLAPSE" in reason_text or "EXTREME_PUMP_THEN_DUMP" in reason_text:
        reasons.append("post_pump_collapse")
    if "UNTRADEABLE" in reason_text or status == "UNTRADEABLE" or action == "UNTRADEABLE":
        reasons.append("untradeable")
    if any(x in reason_text for x in ("DEAD", "SCAM", "TOXIC", "HONEYPOT", "RUG")):
        reasons.append("dead_or_scam")
    if any(x in reason_text for x in ("NO LIQUIDITY", "NO_LIQUIDITY")) or liq <= 0:
        reasons.append("no_liquidity")
    if any(x in reason_text for x in ("NO RECENT FLOW", "NO_RECENT_FLOW")) or vol24 <= 0:
        reasons.append("no_recent_flow")
    if liq > 0 and vol24 > 0 and (vol24 / max(liq, 1.0)) < 0.08:
        reasons.append("liquidity_volume_decay")
    if (pc24 <= -24.0 or pc1h <= -20.0) and liq < 15_000:
        reasons.append("24h_collapse_low_liq")
    if status in {"NO_ENTRY", "AVOID"} or action == "NO_ENTRY":
        reasons.append("no_entry")
    if portfolio_row:
        pf_action = normalize_material_portfolio_action((portfolio_row or {}).get("final_action") or (portfolio_row or {}).get("recommended_action") or (portfolio_row or {}).get("position_action"))
        mon_label = str(row.get("entry_status") or row.get("status") or row.get("timing_label") or "").upper()
        if pf_action in {"EXIT", "REDUCE"} and mon_label in {"WATCH", "EARLY", "READY", "ACTIVE"}:
            reasons.append("portfolio_verdict_conflict")
            flags.append(f"portfolio_{pf_action.lower().replace(' ', '_')}")
    reasons = sorted(set(reasons))
    action_out = "KEEP"
    if any(r in reasons for r in ("dead_or_scam", "no_liquidity", "post_pump_collapse", "untradeable", "portfolio_verdict_conflict")):
        action_out = "ARCHIVE"
    elif reasons:
        action_out = "NO_ENTRY"
    return {"blocked": bool(reasons), "action": action_out, "reason": "|".join(reasons), "flags": flags + reasons}


def should_surface_in_priority(row: Dict[str, Any], portfolio_row: Optional[Dict[str, Any]] = None) -> bool:
    if not is_active_monitoring_row(row):
        return False
    if hard_gate_monitoring_row(row, portfolio_row=portfolio_row).get("blocked"):
        return False
    state = resolve_monitoring_portfolio_state(row or {}, portfolio_row)
    if state.get("is_conflict") and state.get("material_portfolio_action"):
        return False
    return True


def find_active_portfolio_row(row: Dict[str, Any], portfolio_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for portfolio_row in portfolio_rows or []:
        if is_active_portfolio_row(portfolio_row) and row_matches_token(row or {}, portfolio_row):
            return portfolio_row
    return {}
