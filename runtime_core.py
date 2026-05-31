"""Import-safe runtime helpers for DEX Scout.

This module intentionally has no Streamlit dependency and performs no I/O at
import time. Keep helpers here small, deterministic, and safe for worker/webhook
bootstrap imports.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

CHAIN_MAP = {
    "ethereum": "eth",
    "eth": "eth",
    "bsc": "bsc",
    "binance": "bsc",
    "solana": "solana",
}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
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


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def parse_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def normalize_chain_name(raw_chain: Any) -> str:
    value = str(raw_chain or "").strip().lower()
    return CHAIN_MAP.get(value, value)


def addr_store(chain: str, addr: str) -> str:
    c = (chain or "").lower().strip()
    a = (addr or "").strip()
    if not a:
        return ""
    if c == "solana":
        return a
    return a.lower()


def canonical_entity_key(chain: str, ca: str) -> str:
    norm_chain = normalize_chain_name(chain)
    norm_ca = addr_store(norm_chain, ca)
    if not norm_chain or not norm_ca:
        return ""
    return f"{norm_chain}|{norm_ca}"


def safe_get(d: Dict[str, Any], *path: Any, default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur
