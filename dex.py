# dex.py
import requests
from typing import List, Dict, Any

DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search/?q={q}"

def fetch_pairs_by_query(q: str, timeout: int = 15) -> List[Dict[str, Any]]:
    r = requests.get(DEXSCREENER_SEARCH.format(q=q), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("pairs", []) or []

def normalize_pair(p: Dict[str, Any]) -> Dict[str, Any]:
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
    fdv = _to_float(p.get("fdv"))
    mcap = _to_float(p.get("marketCap"))

    price_change = p.get("priceChange") or {}
    pc_m5 = _to_float(price_change.get("m5"))
    pc_h1 = _to_float(price_change.get("h1"))

    volume = p.get("volume") or {}
    vol_m5 = _to_float(volume.get("m5"))
    vol_h1 = _to_float(volume.get("h1"))
    vol_h24 = _to_float(volume.get("h24"))

    txns = p.get("txns") or {}
    tx_m5 = txns.get("m5") or {}
    buys_m5 = _to_int(tx_m5.get("buys"))
    sells_m5 = _to_int(tx_m5.get("sells"))

    created = p.get("pairCreatedAt")  # ms timestamp, may be None

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
        "fdv": fdv,
        "mcap": mcap,
        "pc_m5": pc_m5,
        "pc_h1": pc_h1,
        "vol_m5": vol_m5,
        "vol_h1": vol_h1,
        "vol_h24": vol_h24,
        "buys_m5": buys_m5,
        "sells_m5": sells_m5,
        "txns_m5": buys_m5 + sells_m5,
        "pairCreatedAt": created,
        "url": p.get("url"),
    }

def _to_float(x):
    try:
        return float(x) if x is not None else None
    except:
        return None

def _to_int(x):
    try:
        return int(x) if x is not None else 0
    except:
        return 0

