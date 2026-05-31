# dex.py
import time

import requests
from typing import List, Dict, Any
from urllib.parse import quote

DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search/?q={q}"

def fetch_pairs_by_query(q: str, timeout: int = 15) -> List[Dict[str, Any]]:
    query = str(q or "").strip()
    if not query:
        return []
    encoded_q = quote(query, safe="")
    r = requests.get(DEXSCREENER_SEARCH.format(q=encoded_q), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("pairs", []) or []

def _pair_age_hours(created_ms, now_ts=None) -> float | None:
    try:
        if created_ms is None:
            return None
        created_sec = float(created_ms) / 1000.0
    except (TypeError, ValueError):
        return None
    now = time.time() if now_ts is None else float(now_ts)
    return max(0.0, (now - created_sec) / 3600.0)


def normalize_pair_core(p: Dict[str, Any]) -> Dict[str, Any]:
    chain = p.get("chainId")
    dex = p.get("dexId")
    pair_address = p.get("pairAddress")

    base = (p.get("baseToken") or {})
    quote = (p.get("quoteToken") or {})

    symbol = base.get("symbol")
    name = base.get("name")
    token_addr = base.get("address")

    price_usd = _to_float(p.get("priceUsd"))
    liq_usd = _to_float((p.get("liquidity") or {}).get("usd"))

    price_change = p.get("priceChange") or {}
    pc_m5 = _to_float(price_change.get("m5"))
    pc_h1 = _to_float(price_change.get("h1"))

    volume = p.get("volume") or {}
    vol_m5 = _to_float(volume.get("m5"))

    txns = p.get("txns") or {}
    tx_m5 = txns.get("m5") or {}
    buys_m5 = _to_int(tx_m5.get("buys"))
    sells_m5 = _to_int(tx_m5.get("sells"))

    created = p.get("pairCreatedAt")  # raw ms timestamp, may be None

    return {
        "chain": chain,
        "dex": dex,
        "pairAddress": pair_address,
        "tokenSymbol": symbol,
        "tokenName": name,
        "tokenAddress": token_addr,
        "quoteSymbol": quote.get("symbol"),
        "priceUsd": price_usd,
        "liqUsd": liq_usd,
        "pc_m5": pc_m5,
        "pc_h1": pc_h1,
        "vol_m5": vol_m5,
        "buys_m5": buys_m5,
        "sells_m5": sells_m5,
        "txns_m5": buys_m5 + sells_m5,
        "pairCreatedAt": created,
        "pair_age_hours": _pair_age_hours(created),
        "url": p.get("url"),
    }


def normalize_pair_full(p: Dict[str, Any]) -> Dict[str, Any]:
    out = normalize_pair_core(p)
    volume = p.get("volume") or {}
    out.update({
        "fdv": _to_float(p.get("fdv")),
        "mcap": _to_float(p.get("marketCap")),
        "vol_h1": _to_float(volume.get("h1")),
        "vol_h24": _to_float(volume.get("h24")),
    })
    return out


def normalize_pair(p: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_pair_full(p)

def _to_float(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None

def _to_int(x):
    try:
        return int(x) if x is not None else 0
    except (TypeError, ValueError):
        return 0

