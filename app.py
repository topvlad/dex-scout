# app.py
# Streamlit app: On-chain "Monitoring" scanner (BSC + optional Solana)
# No while True loops. Uses st_autorefresh for auto mode.

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================
# Config
# =========================

APP_TITLE = "On-chain Monitor (BSC + Solana)"
DEFAULT_CHAIN = "bsc"  # "bsc" or "sol"
DEFAULT_REFRESH_SEC = 10
HTTP_TIMEOUT = 15

# Optional: set your keys in Streamlit Secrets or env vars
# Streamlit Cloud: Settings -> Secrets
# Example:
#   BSC_API_KEY="..."
#   SOL_API_KEY="..."
BSC_API_KEY = st.secrets.get("BSC_API_KEY", os.getenv("BSC_API_KEY", ""))
SOL_API_KEY = st.secrets.get("SOL_API_KEY", os.getenv("SOL_API_KEY", ""))

USER_AGENT = "Mozilla/5.0 (compatible; OnchainMonitor/1.0; +https://streamlit.io)"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# =========================
# Helpers
# =========================

@dataclass
class Candidate:
    chain: str
    symbol: str
    name: str
    contract: str
    price_usd: Optional[float]
    mcap_usd: Optional[float]
    liq_usd: Optional[float]
    vol_24h_usd: Optional[float]
    holders: Optional[int]
    top10_pct: Optional[float]
    change_24h_pct: Optional[float]
    age_min: Optional[int]
    source: str

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None

def now_ts() -> int:
    return int(time.time())

def http_get(url: str, params: Optional[dict] = None) -> dict:
    r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def to_df(items: List[Candidate]) -> pd.DataFrame:
    rows = []
    for c in items:
        rows.append({
            "chain": c.chain,
            "symbol": c.symbol,
            "name": c.name,
            "contract": c.contract,
            "price_usd": c.price_usd,
            "mcap_usd": c.mcap_usd,
            "liq_usd": c.liq_usd,
            "vol_24h_usd": c.vol_24h_usd,
            "holders": c.holders,
            "top10_pct": c.top10_pct,
            "change_24h_pct": c.change_24h_pct,
            "age_min": c.age_min,
            "source": c.source,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # nice ordering
    order = ["chain", "symbol", "name", "contract", "mcap_usd", "liq_usd", "vol_24h_usd", "holders", "top10_pct", "change_24h_pct", "age_min", "source", "price_usd"]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    return df[cols]


# =========================
# Data sources (Simple)
# =========================
# NOTE:
# Найпростіший варіант для старту (і реально працює) — брати "нові/трендові" токени з DexScreener.
# Потім ДОЗАПОВНЮВАТИ holders/top10 через окремі onchain API (BscScan / Moralis / Covalent / Bitquery).
#
# Тут я зробив:
# - DexScreener: основний стрім кандидатів (BSC + Sol)
# - Holders/Top10: як опційні поля (поки None), бо провайдер залежить від того, що ти вибереш.
#
# Коли скажеш який саме on-chain API береш для holders/top10 — я додам 20 рядків коду і все оживе.


def dexscreener_latest_pairs(chain: str, limit: int = 50) -> List[dict]:
    """
    DexScreener public endpoint gives trending/boosted pairs.
    We'll use a stable endpoint: /latest/dex/pairs/{chain}
    """
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}"
    data = http_get(url)
    pairs = data.get("pairs", []) or []
    return pairs[:limit]


def parse_dexscreener_pair(pair: dict, chain: str) -> Candidate:
    base = pair.get("baseToken", {}) or {}
    symbol = base.get("symbol") or "?"
    name = base.get("name") or ""
    contract = base.get("address") or pair.get("pairAddress") or "?"
    price_usd = safe_float(pair.get("priceUsd"))
    liq = pair.get("liquidity", {}) or {}
    liq_usd = safe_float(liq.get("usd"))
    fdv = safe_float(pair.get("fdv"))  # Dexscreener often uses fdv as a proxy for mcap-like
    mcap_usd = safe_float(pair.get("marketCap")) if pair.get("marketCap") is not None else fdv
    vol = pair.get("volume", {}) or {}
    vol_24h = safe_float(vol.get("h24"))
    chg = pair.get("priceChange", {}) or {}
    change_24h_pct = safe_float(chg.get("h24"))

    # age minutes (if pairCreatedAt exists)
    age_min = None
    created_at = pair.get("pairCreatedAt")
    if created_at:
        try:
            # DexScreener returns ms timestamp often
            created_sec = int(created_at) // 1000 if int(created_at) > 10_000_000_000 else int(created_at)
            age_min = max(0, int((now_ts() - created_sec) / 60))
        except Exception:
            age_min = None

    return Candidate(
        chain=chain,
        symbol=str(symbol),
        name=str(name),
        contract=str(contract),
        price_usd=price_usd,
        mcap_usd=mcap_usd,
        liq_usd=liq_usd,
        vol_24h_usd=vol_24h,
        holders=None,     # TODO: add via on-chain API
        top10_pct=None,   # TODO: add via on-chain API
        change_24h_pct=change_24h_pct,
        age_min=age_min,
        source="DexScreener",
    )


# =========================
# Filtering / scoring
# =========================

def apply_filters(items: List[Candidate], cfg: dict) -> List[Candidate]:
    out = []
    for c in items:
        # required numeric fields
        if c.mcap_usd is None or c.liq_usd is None or c.vol_24h_usd is None:
            continue

        if not (cfg["mcap_min"] <= c.mcap_usd <= cfg["mcap_max"]):
            continue
        if c.liq_usd < cfg["liq_min"]:
            continue
        if c.vol_24h_usd < cfg["vol24_min"]:
            continue

        # optional % change filter
        if cfg["use_change24"]:
            if c.change_24h_pct is None:
                continue
            if not (cfg["change24_min"] <= c.change_24h_pct <= cfg["change24_max"]):
                continue

        # age filter in minutes
        if cfg["use_age"]:
            if c.age_min is None:
                continue
            if not (cfg["age_min"] <= c.age_min <= cfg["age_max"]):
                continue

        out.append(c)
    return out


def score_candidate(c: Candidate, cfg: dict) -> float:
    """
    Simple score:
      - prefer higher vol/liquidity ratio (activity)
      - prefer not-too-high mcap (more upside)
      - optionally prefer positive change
    """
    score = 0.0
    if c.vol_24h_usd and c.liq_usd and c.liq_usd > 0:
        score += min(10.0, (c.vol_24h_usd / c.liq_usd))  # activity ratio
    if c.mcap_usd and c.mcap_usd > 0:
        # lower mcap -> higher score
        score += max(0.0, 10.0 - (c.mcap_usd / max(1.0, cfg["mcap_target"])) * 10.0)
    if c.change_24h_pct is not None:
        score += (c.change_24h_pct / 10.0)  # small boost
    return score


def rank(items: List[Candidate], cfg: dict, top_n: int) -> List[Tuple[Candidate, float]]:
    scored = [(c, score_candidate(c, cfg)) for c in items]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# =========================
# Scan
# =========================

@st.cache_data(ttl=20, show_spinner=False)
def do_scan(chain: str, cfg: dict) -> Tuple[List[Candidate], List[Tuple[Candidate, float]]]:
    # 1) Fetch from DexScreener
    pairs = dexscreener_latest_pairs(chain=chain, limit=cfg["fetch_limit"])
    items = [parse_dexscreener_pair(p, chain=chain) for p in pairs]

    # 2) Filter
    filtered = apply_filters(items, cfg)

    # 3) Rank
    top = rank(filtered, cfg, top_n=cfg["top_n"])
    return filtered, top


# =========================
# UI
# =========================

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Scan settings")

    chain = st.selectbox("Chain", ["bsc", "solana"], index=0 if DEFAULT_CHAIN == "bsc" else 1)
    auto = st.toggle("Auto mode", value=False, help="Auto-refresh without infinite loops")
    interval = st.number_input("Refresh interval (sec)", min_value=3, max_value=120, value=DEFAULT_REFRESH_SEC, step=1)
    refresh_now = st.button("Refresh now")

    st.divider()
    st.subheader("Filters (simple)")

    fetch_limit = st.slider("Fetch pairs limit", 10, 200, 80, 10)

    mcap_min = st.number_input("MCap min ($)", min_value=0, value=10_000, step=1_000)
    mcap_max = st.number_input("MCap max ($)", min_value=0, value=500_000, step=10_000)
    mcap_target = st.number_input("MCap target ($) for scoring", min_value=1_000, value=200_000, step=10_000)

    liq_min = st.number_input("Liquidity min ($)", min_value=0, value=20_000, step=1_000)
    vol24_min = st.number_input("Volume 24h min ($)", min_value=0, value=100_000, step=10_000)

    use_change24 = st.checkbox("Use 24h % change filter", value=False)
    change24_min = st.number_input("24h % min", value=-50.0, step=5.0)
    change24_max = st.number_input("24h % max", value=5_000.0, step=50.0)

    use_age = st.checkbox("Use age filter (minutes)", value=False)
    age_min = st.number_input("Age min (min)", min_value=0, value=1, step=1)
    age_max = st.number_input("Age max (min)", min_value=0, value=180, step=10)

    top_n = st.slider("Top N", 1, 50, 10, 1)

    cfg = {
        "fetch_limit": int(fetch_limit),
        "mcap_min": float(mcap_min),
        "mcap_max": float(mcap_max),
        "mcap_target": float(mcap_target),
        "liq_min": float(liq_min),
        "vol24_min": float(vol24_min),
        "use_change24": bool(use_change24),
        "change24_min": float(change24_min),
        "change24_max": float(change24_max),
        "use_age": bool(use_age),
        "age_min": int(age_min),
        "age_max": int(age_max),
        "top_n": int(top_n),
    }

# Auto refresh without loops
if auto:
    st_autorefresh(interval=int(interval) * 1000, key="auto_refresh")

# Fetch + render
with st.spinner("Scanning..."):
    candidates, top = do_scan(chain=chain, cfg=cfg)

col1, col2 = st.columns([1.4, 1.0], gap="large")

with col1:
    st.subheader("Candidates (filtered)")
    df = to_df(candidates)
    if df.empty:
        st.info("No matches. Relax filters (mcap/liquidity/volume) or increase Fetch limit.")
    else:
        st.dataframe(df, use_container_width=True, height=520)

with col2:
    st.subheader(f"Top {cfg['top_n']} (ranked)")
    if not top:
        st.info("Top list is empty (no filtered candidates).")
    else:
        top_rows = []
        for c, s in top:
            top_rows.append({
                "score": round(s, 3),
                "symbol": c.symbol,
                "mcap": c.mcap_usd,
                "liq": c.liq_usd,
                "vol24": c.vol_24h_usd,
                "change24%": c.change_24h_pct,
                "age_min": c.age_min,
                "contract": c.contract,
            })
        top_df = pd.DataFrame(top_rows)
        st.dataframe(top_df, use_container_width=True, height=520)

st.divider()
st.caption(
    "NOTE: Holders/Top10% fields are placeholders (None) until you pick an on-chain provider. "
    "Say which API you want (BscScan/Moralis/Covalent/Bitquery) and I’ll add those fields."
)
