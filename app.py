# app.py
# Streamlit MVP: DexScreener Option B (SEARCH-based) scanner
# SAFE VERSION (env-checked)

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# ---- SAFE IMPORT ----
try:
    import requests
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: requests\n"
        "➡️ Add `requests` to requirements.txt and redeploy."
    ) from e

import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEXSCREENER_BASE = "https://api.dexscreener.com"
DEFAULT_TIMEOUT_SEC = 12
MAX_PAIRS_PER_QUERY = 80
SLEEP_BETWEEN_CALLS_SEC = 0.15
DEFAULT_TOP_N = 25

DEFAULT_MIN_LIQ_USD = 10_000
DEFAULT_MIN_VOL24_USD = 50_000

DEFAULT_QUERIES = {
    "solana": ["SOL", "USDC", "JUP", "RAY", "BONK", "WIF", "pump"],
    "bsc": ["WBNB", "USDT", "CAKE", "BNB", "meme"],
}


# -----------------------------
# Helpers
# -----------------------------
def http_get_json(url: str, params: Optional[dict] = None) -> Any:
    r = requests.get(
        url,
        params=params,
        headers={"Accept": "application/json"},
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    r.raise_for_status()
    return r.json()


def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pick_txns(obj: dict) -> Tuple[int, int]:
    txns = obj.get("txns") or {}
    for tf in ("m5", "m15", "m30", "h1"):
        if tf in txns:
            buys = int(txns[tf].get("buys", 0))
            sells = int(txns[tf].get("sells", 0))
            return buys, sells
    return 0, 0


@dataclass
class PairRow:
    chain: str
    dex: str
    pair: str
    url: str
    base: str
    quote: str
    price: float
    liquidity: float
    vol24: float
    vol5: float
    pc5: float
    pc1: float
    pc24: float
    buys: int
    sells: int
    score: float


def parse_pair(p: dict) -> PairRow:
    base = p.get("baseToken", {}).get("symbol", "")
    quote = p.get("quoteToken", {}).get("symbol", "")

    liquidity = safe_num(p.get("liquidity", {}).get("usd"))
    volume = p.get("volume", {})
    pc = p.get("priceChange", {})

    buys, sells = pick_txns(p)

    score = (
        safe_num(pc.get("m5")) * 1.2
        + safe_num(pc.get("h1")) * 0.3
        + (buys + 1) / (sells + 1) * 5
        + min(liquidity / 50_000, 2) * 6
    )

    return PairRow(
        chain=p.get("chainId", ""),
        dex=p.get("dexId", ""),
        pair=p.get("pairAddress", ""),
        url=p.get("url", ""),
        base=base,
        quote=quote,
        price=safe_num(p.get("priceUsd")),
        liquidity=liquidity,
        vol24=safe_num(volume.get("h24")),
        vol5=safe_num(volume.get("m5")),
        pc5=safe_num(pc.get("m5")),
        pc1=safe_num(pc.get("h1")),
        pc24=safe_num(pc.get("h24")),
        buys=buys,
        sells=sells,
        score=score,
    )


@st.cache_data(ttl=30)
def search_pairs(query: str) -> List[dict]:
    return http_get_json(
        f"{DEXSCREENER_BASE}/latest/dex/search",
        params={"q": query},
    ).get("pairs", [])


# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("DEX Scout — Option B (DexScreener Search)")

with st.sidebar:
    chain = st.selectbox("Chain", ["bsc", "solana"])
    top_n = st.slider("Top N", 5, 50, DEFAULT_TOP_N)
    min_liq = st.number_input("Min Liquidity ($)", value=DEFAULT_MIN_LIQ_USD)
    min_vol = st.number_input("Min 24h Volume ($)", value=DEFAULT_MIN_VOL24_USD)
    queries = st.text_area(
        "Search queries",
        ", ".join(DEFAULT_QUERIES[chain]),
    )
    refresh = st.button("Refresh")

if refresh:
    st.cache_data.clear()

queries = [q.strip() for q in queries.split(",") if q.strip()]

pairs_raw = []
for q in queries:
    try:
        pairs_raw += search_pairs(q)
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
    except Exception:
        pass

pairs = []
seen = set()

for p in pairs_raw:
    key = (p.get("chainId"), p.get("pairAddress"))
    if key in seen:
        continue
    seen.add(key)

    if p.get("chainId") != chain:
        continue

    r = parse_pair(p)
    if r.liquidity < min_liq or r.vol24 < min_vol:
        continue

    pairs.append(r)

pairs.sort(key=lambda x: x.score, reverse=True)
pairs = pairs[:top_n]

st.subheader(f"Results ({len(pairs)})")

for r in pairs:
    st.markdown(f"### [{r.base}/{r.quote}]({r.url})")
    st.write(
        f"Price: ${r.price:.8f} | "
        f"Liq: ${r.liquidity:,.0f} | "
        f"Vol24: ${r.vol24:,.0f}"
    )
    st.write(
        f"Δ5m: {r.pc5:+.2f}% | "
        f"Δ1h: {r.pc1:+.2f}% | "
        f"Buys/Sells: {r.buys}/{r.sells} | "
        f"Score: {r.score:.2f}"
    )
    st.divider()
