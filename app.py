# app.py
# Streamlit MVP: DexScreener Option B (SEARCH-based) scanner
# - No Binance API, no "monitoring" section
# - Uses ONLY: GET https://api.dexscreener.com/latest/dex/search?q=...
# - Filters + ranks pairs (BSC + Solana friendly)
#
# Run:
#   streamlit run app.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEXSCREENER_BASE = "https://api.dexscreener.com"
DEFAULT_TIMEOUT_SEC = 12
MAX_PAIRS_PER_QUERY = 80          # keep it light
SLEEP_BETWEEN_CALLS_SEC = 0.15    # polite pacing (avoid bursts)
DEFAULT_TOP_N = 25

# Safe-ish defaults for microcaps:
DEFAULT_MIN_LIQ_USD = 10_000
DEFAULT_MIN_VOL24_USD = 50_000

DEFAULT_QUERIES = {
    # Option B = searches; these are just "seed" queries
    # You can edit in the sidebar anytime.
    "solana": ["SOL", "USDC", "JUP", "RAY", "BONK", "WIF", "pump"],
    "bsc": ["WBNB", "USDT", "CAKE", "BUSD", "BNB", "PancakeSwap", "meme"],
}


# -----------------------------
# Helpers
# -----------------------------
def http_get_json(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT_SEC) -> Any:
    headers = {"Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        # strings like "0.0123"
        return float(str(x))
    except Exception:
        return default


def pick_timeframe_field(obj: dict, key: str, preferred: Tuple[str, ...] = ("h1", "m30", "m15", "h6", "h24")) -> float:
    """
    DexScreener returns dicts keyed by timeframes in some fields:
      - txns: {"m5":{"buys":..,"sells":..}, "h1":...}
      - volume: {"m5":..., "h1":..., "h24":...}
      - priceChange: {"m5":..., "h1":..., "h24":...}
    We pick best available by preference order.
    """
    d = obj.get(key) or {}
    if not isinstance(d, dict):
        return 0.0
    for tf in preferred:
        if tf in d:
            return safe_num(d.get(tf), 0.0)
    # fallback to any numeric
    for v in d.values():
        if isinstance(v, (int, float, str)):
            return safe_num(v, 0.0)
    return 0.0


def pick_txns(obj: dict, tf_order: Tuple[str, ...] = ("m5", "m15", "m30", "h1")) -> Tuple[int, int]:
    txns = obj.get("txns") or {}
    if not isinstance(txns, dict):
        return (0, 0)
    for tf in tf_order:
        if tf in txns and isinstance(txns[tf], dict):
            buys = int(safe_num(txns[tf].get("buys"), 0))
            sells = int(safe_num(txns[tf].get("sells"), 0))
            return (buys, sells)
    return (0, 0)


@dataclass
class PairRow:
    chainId: str
    dexId: str
    pairAddress: str
    url: str

    base_symbol: str
    base_address: str
    quote_symbol: str
    quote_address: str

    price_usd: float
    liquidity_usd: float
    volume_h24: float
    vol_m5: float

    price_change_m5: float
    price_change_h1: float
    price_change_h24: float

    buys_m5: int
    sells_m5: int

    fdv: float
    market_cap: float
    created_at: int  # ms
    score: float


def parse_pair(p: dict) -> PairRow:
    base = p.get("baseToken") or {}
    quote = p.get("quoteToken") or {}
    liquidity = p.get("liquidity") or {}

    price_usd = safe_num(p.get("priceUsd"), 0.0)
    liq_usd = safe_num(liquidity.get("usd"), 0.0)

    # volumes
    volume = p.get("volume") or {}
    vol_h24 = safe_num(volume.get("h24"), 0.0) if isinstance(volume, dict) else 0.0
    vol_m5 = safe_num(volume.get("m5"), 0.0) if isinstance(volume, dict) else 0.0

    # price changes
    pc = p.get("priceChange") or {}
    pc_m5 = safe_num(pc.get("m5"), 0.0) if isinstance(pc, dict) else 0.0
    pc_h1 = safe_num(pc.get("h1"), 0.0) if isinstance(pc, dict) else 0.0
    pc_h24 = safe_num(pc.get("h24"), 0.0) if isinstance(pc, dict) else 0.0

    buys_m5, sells_m5 = pick_txns(p, tf_order=("m5", "m15", "m30", "h1"))
    created_at = int(safe_num(p.get("pairCreatedAt"), 0.0))

    fdv = safe_num(p.get("fdv"), 0.0)
    mcap = safe_num(p.get("marketCap"), 0.0)

    # Score: (momentum + activity + liquidity sanity) — simple + stable
    # - prefer rising volume + buys> sells
    # - avoid pure vertical blow-off by penalizing extreme m5 change a bit
    buy_sell = (buys_m5 + 1) / (sells_m5 + 1)
    activity = (vol_m5 / 1_000.0)  # normalize a bit
    liq_factor = min(2.0, (liq_usd / 50_000.0))  # remember: small caps
    momentum = (pc_m5 * 0.6) + (pc_h1 * 0.3) + (pc_h24 * 0.1)

    # penalty if pc_m5 is insane (common rugs/pumps)
    blowoff_penalty = max(0.0, (abs(pc_m5) - 35.0) / 35.0)  # starts after 35% m5
    blowoff_penalty = min(2.0, blowoff_penalty)

    score = (momentum * 1.2) + (activity * 1.5) + (buy_sell * 8.0) + (liq_factor * 6.0) - (blowoff_penalty * 10.0)

    return PairRow(
        chainId=str(p.get("chainId") or ""),
        dexId=str(p.get("dexId") or ""),
        pairAddress=str(p.get("pairAddress") or ""),
        url=str(p.get("url") or ""),

        base_symbol=str(base.get("symbol") or ""),
        base_address=str(base.get("address") or ""),
        quote_symbol=str(quote.get("symbol") or ""),
        quote_address=str(quote.get("address") or ""),

        price_usd=price_usd,
        liquidity_usd=liq_usd,
        volume_h24=vol_h24,
        vol_m5=vol_m5,

        price_change_m5=pc_m5,
        price_change_h1=pc_h1,
        price_change_h24=pc_h24,

        buys_m5=buys_m5,
        sells_m5=sells_m5,

        fdv=fdv,
        market_cap=mcap,
        created_at=created_at,
        score=score,
    )


@st.cache_data(ttl=25, show_spinner=False)
def search_pairs(query: str) -> List[dict]:
    url = f"{DEXSCREENER_BASE}/latest/dex/search"
    data = http_get_json(url, params={"q": query}, timeout=DEFAULT_TIMEOUT_SEC)
    pairs = data.get("pairs") or []
    if not isinstance(pairs, list):
        return []
    return pairs[:MAX_PAIRS_PER_QUERY]


def uniq_pairs(pairs: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for p in pairs:
        key = (p.get("chainId"), p.get("pairAddress"))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def age_minutes(created_at_ms: int) -> float:
    if created_at_ms <= 0:
        return 9e9
    now_ms = int(time.time() * 1000)
    return max(0.0, (now_ms - created_at_ms) / 60000.0)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DEX Screener — Option B Scanner", layout="wide")

st.title("DEX Screener Scanner (Option B: Search-based)")
st.caption("Candidates come from DexScreener search queries only. Then we filter + rank. Light + fast.")

with st.sidebar:
    st.header("Settings")
    chain = st.selectbox("Chain", ["bsc", "solana"], index=0)

    top_n = st.slider("Top N", min_value=5, max_value=80, value=DEFAULT_TOP_N, step=5)

    min_liq = st.number_input("Min Liquidity (USD)", min_value=0, value=DEFAULT_MIN_LIQ_USD, step=1000)
    min_vol24 = st.number_input("Min Volume 24h (USD)", min_value=0, value=DEFAULT_MIN_VOL24_USD, step=5000)

    max_age_min = st.number_input("Max Age (minutes) (0 = ignore)", min_value=0, value=0, step=60)

    st.divider()
    st.subheader("Search seeds (comma-separated)")
    default_list = ", ".join(DEFAULT_QUERIES[chain])
    queries_text = st.text_area("Queries", value=default_list, height=90)

    strict_quotes = st.checkbox("Strict: keep only pairs where query appears in name/symbol", value=False)
    refresh = st.button("Refresh now", type="primary")

if refresh:
    st.cache_data.clear()

queries = [q.strip() for q in queries_text.split(",") if q.strip()]
if not queries:
    st.warning("Add at least one query in the sidebar.")
    st.stop()

# -----------------------------
# Fetch
# -----------------------------
with st.spinner("Fetching from DexScreener search…"):
    all_pairs: List[dict] = []
    errors: List[str] = []

    for q in queries:
        try:
            pairs = search_pairs(q)
            # optional strict filtering by query presence
            if strict_quotes:
                q_low = q.lower()
                def ok(p: dict) -> bool:
                    base = (p.get("baseToken") or {}).get("symbol", "") or ""
                    quote = (p.get("quoteToken") or {}).get("symbol", "") or ""
                    name = f"{base}/{quote}".lower()
                    return q_low in name
                pairs = [p for p in pairs if ok(p)]
            all_pairs.extend(pairs)
        except Exception as e:
            errors.append(f"{q}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    all_pairs = uniq_pairs(all_pairs)

if errors:
    with st.expander("Fetch errors"):
        for e in errors:
            st.write(e)

if not all_pairs:
    st.info("No pairs returned from search. Try different queries.")
    st.stop()

# Filter by chain
pairs_chain = [p for p in all_pairs if str(p.get("chainId") or "").lower() == chain]

if not pairs_chain:
    st.info(f"No pairs matched chain '{chain}'. Try other queries or switch chain.")
    st.stop()

# Parse + filter
rows: List[PairRow] = []
for p in pairs_chain:
    try:
        r = parse_pair(p)
        # Basic filters
        if r.liquidity_usd < float(min_liq):
            continue
        if r.volume_h24 < float(min_vol24):
            continue
        if max_age_min and age_minutes(r.created_at) > float(max_age_min):
            continue
        rows.append(r)
    except Exception:
        continue

if not rows:
    st.warning("Nothing passed filters. Lower min liquidity / min volume, or change queries.")
    st.stop()

# Rank
rows.sort(key=lambda x: x.score, reverse=True)
rows = rows[: int(top_n)]

# -----------------------------
# Output
# -----------------------------
colA, colB, colC, colD = st.columns([2.2, 1.2, 1.2, 1.2])

with colA:
    st.metric("Candidates (unique pairs)", value=len(all_pairs))
with colB:
    st.metric("Chain-matched", value=len(pairs_chain))
with colC:
    st.metric("Passed filters", value=len(rows))
with colD:
    st.metric("Queries used", value=len(queries))

st.divider()

# Table-like display (fast)
for r in rows:
    left, mid, right = st.columns([2.6, 1.7, 1.7])

    with left:
        st.markdown(f"### [{r.base_symbol}/{r.quote_symbol}]({r.url})")
        st.caption(f"{r.chainId} • {r.dexId} • pair: `{r.pairAddress}`")
        st.caption(f"base: `{r.base_address}` • quote: `{r.quote_address}`")

    with mid:
        st.write(f"**Price (USD):** {r.price_usd:,.10f}".rstrip("0").rstrip("."))
        st.write(f"**Liquidity:** ${r.liquidity_usd:,.0f}")
        st.write(f"**Vol 24h:** ${r.volume_h24:,.0f}")
        st.write(f"**Vol m5:** ${r.vol_m5:,.0f}")

    with right:
        st.write(f"**Δ m5:** {r.price_change_m5:+.2f}%")
        st.write(f"**Δ h1:** {r.price_change_h1:+.2f}%")
        st.write(f"**Δ h24:** {r.price_change_h24:+.2f}%")
        st.write(f"**m5 buys/sells:** {r.buys_m5}/{r.sells_m5}")
        st.write(f"**Score:** {r.score:,.2f}")

    st.divider()

st.caption(
    "Note: This is a SEARCH-based candidate stream (Option B). It can miss pairs not discoverable via your queries. "
    "If you want a stronger feed later, we can switch candidates to boosts/top or an on-chain mempool/event feed."
)
