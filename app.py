# app.py — DEX Scout v0.3.6 (stable core: BSC + Solana)
# Key fixes vs your last state:
# 1) FIX: "NO ENTRY" badge is now RED (not green).
# 2) FIX: Unique Streamlit keys (no duplicate / TypeError surprises).
# 3) Core locked to BSC + Solana (simple, stable).
# 4) Swap enabled on BOTH:
#    - BSC: PancakeSwap
#    - Solana: Jupiter (jup.ag)
# 5) More output dynamics WITHOUT loosening safety too much:
#    - Seed sampler (random K seeds per run)
#    - Top tokens mode (dedupe by base token)
#    - Pair age window (still early, but anti-rug-ish)
#    - Majors/stables filter ON by default
#    - Optional "Hide if Solana token looks UNVERIFIED" (heuristic) ON by default
#
# NOTE (important): We do NOT have on-chain holder concentration / mint auth / freeze auth checks here.
# DexScreener doesn’t provide them reliably. For Solana, ALWAYS glance at Jupiter warnings before swapping.

import os
import re
import csv
import time
import random
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import requests
import streamlit as st

VERSION = "0.3.6"

DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
TRADES_CSV = os.path.join(DATA_DIR, "portfolio.csv")

# Stable core only
CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap", "babyswap", "mdex", "woofi"],
    "solana": ["raydium", "orca", "meteora", "pumpfun", "pumpswap"],
}

# Softer ticker validation:
# allow unicode letters/digits + common ticker chars ($ _ - .) + spaces
NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

# Major/stable symbols we don't want as "early gems"
MAJORS_STABLES = {
    # majors
    "BTC", "WBTC", "ETH", "WETH", "BNB", "WBNB", "SOL", "WSOL",
    # stables
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDE", "USDS",
    # common wrappers
    "WBNB", "WETH", "WBTC", "WAVAX", "WMATIC",
}


# -----------------------------
# Helpers
# -----------------------------
def safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_usd(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "n/a"


def fmt_pct(x: float) -> str:
    try:
        return f"{x:+.2f}%"
    except Exception:
        return "n/a"


def clamp_int(x: float, lo: int = 0, hi: int = 10**9) -> int:
    try:
        return int(max(lo, min(hi, int(x))))
    except Exception:
        return lo


def hkey(*parts: str, n: int = 10) -> str:
    """Short stable hash for Streamlit keys."""
    raw = "|".join([p or "" for p in parts])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:n]


def safe_key(s: str, limit: int = 80) -> str:
    """Keep Streamlit key safe + bounded."""
    if s is None:
        s = ""
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s)
    return s[:limit] if len(s) > limit else s


# -----------------------------
# HTTP with retry/backoff
# -----------------------------
def _http_get_json(url: str, params: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> Any:
    """
    Basic retry/backoff wrapper for DexScreener calls.
    Retries on 429 / 5xx / common network errors with exponential backoff.
    """
    last_err = None
    backoff = 0.7

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": f"dex-scout/{VERSION}"},
            )
            if r.status_code == 429 or (500 <= r.status_code <= 599):
                last_err = requests.HTTPError(f"HTTP {r.status_code} (attempt {attempt}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"Request failed after {max_retries} tries: {last_err}")


# -----------------------------
# API
# -----------------------------
@st.cache_data(ttl=25, show_spinner=False)
def fetch_latest_pairs_for_query(q: str) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/latest/dex/search"
    data = _http_get_json(url, params={"q": q.strip()}, timeout=20, max_retries=3)
    return data.get("pairs", []) or []


@st.cache_data(ttl=25, show_spinner=False)
def fetch_token_pairs(chain: str, token_address: str) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/token-pairs/v1/{chain}/{token_address}"
    data = _http_get_json(url, params=None, timeout=20, max_retries=3)
    return data or []


def best_pair_for_token(chain: str, token_address: str) -> Optional[Dict[str, Any]]:
    try:
        pools = fetch_token_pairs(chain, token_address)
    except Exception:
        return None
    if not pools:
        return None

    def key(p):
        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
        return (liq, vol24)

    pools.sort(key=key, reverse=True)
    return pools[0]


# -----------------------------
# Safety / heuristics
# -----------------------------
def is_name_suspicious(p: Dict[str, Any]) -> bool:
    sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip()
    if not sym:
        return True
    if len(sym) > 40:
        return True
    return not bool(NAME_OK_RE.match(sym))


def is_major_or_stable(p: Dict[str, Any]) -> bool:
    base = (safe_get(p, "baseToken", "symbol", default="") or "").strip().upper()
    quote = (safe_get(p, "quoteToken", "symbol", default="") or "").strip().upper()
    if base in MAJORS_STABLES:
        return True
    # if quote is a major/stable we still allow (because most pairs quote in USDT/USDC),
    # but we DON'T want "ETH/USDT", "SOL/USDC" etc (base filter already catches that).
    return False


def pair_age_minutes(p: Dict[str, Any]) -> Optional[float]:
    """
    DexScreener usually provides pairCreatedAt (ms).
    If missing, return None (we then treat as unknown age).
    """
    ts = p.get("pairCreatedAt")
    if ts is None:
        return None
    try:
        ts = float(ts)
        # most likely ms
        if ts > 10_000_000_000:
            created = ts / 1000.0
        else:
            created = ts
        age_sec = max(0.0, time.time() - created)
        return age_sec / 60.0
    except Exception:
        return None


def solana_unverified_heuristic(p: Dict[str, Any]) -> bool:
    """
    We can't truly check "verified" on-chain here.
    So we use a conservative heuristic to HIDE likely-degen tokens by default.
    Mark as "unverified-like" if:
      - very new (< 60 min) OR age unknown
      - AND liquidity is low-ish
      - AND 24h volume is extreme relative to liquidity OR 1h pump is extreme
    You can toggle this OFF to see everything.
    """
    if (p.get("chainId") or "").lower() != "solana":
        return False

    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)

    age = pair_age_minutes(p)
    very_new_or_unknown = (age is None) or (age < 60)

    # "too hot" ratios/pumps — typical of farm/stealth launches
    vol_liq_ratio = (vol24 / max(1.0, liq)) if liq else 999.0

    if very_new_or_unknown and liq < 50_000:
        return True
    if very_new_or_unknown and vol_liq_ratio > 25:
        return True
    if very_new_or_unknown and pc1h > 120:
        return True
    return False


# -----------------------------
# Scoring / decision
# -----------------------------
def score_pair(p: Dict[str, Any]) -> float:
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)
    pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)
    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
    trades5 = buys5 + sells5

    s = 0.0
    s += min(liq / 1000.0, 380.0)
    s += min(vol24 / 10000.0, 280.0)
    s += min(vol5 / 2000.0, 220.0)
    s += min(trades5 * 2.0, 140.0)
    s += max(min(pc1h, 90.0), -90.0) * 0.25
    s += max(min(pc5, 40.0), -40.0) * 0.15
    return round(s, 2)


def action_badge(action: str) -> str:
    """
    Color rules:
      - ENTRY / BUY => green
      - WATCH / WAIT => yellow
      - NO ENTRY / CUT / RISK => red
    """
    a = (action or "").strip().upper()

    is_green = ("ENTRY" in a) or a.startswith("BUY")
    is_yellow = ("WATCH" in a) or ("WAIT" in a)
    is_red = ("NO ENTRY" in a) or ("CUT" in a) or ("RISK" in a) or ("AVOID" in a)

    if is_green:
        color = "#1f9d55"
        bg = "rgba(31,157,85,0.15)"
    elif is_yellow:
        color = "#d69e2e"
        bg = "rgba(214,158,46,0.15)"
    elif is_red:
        color = "#e53e3e"
        bg = "rgba(229,62,62,0.12)"
    else:
        # default to yellow-ish (neutral)
        color = "#d69e2e"
        bg = "rgba(214,158,46,0.10)"

    return f"""
    <span style="
      display:inline-block;
      padding:6px 12px;
      border-radius:999px;
      border:1px solid {color};
      background:{bg};
      color:{color};
      font-weight:900;
      font-size:12px;
      letter-spacing:0.35px;
    ">{action}</span>
    """


def build_trade_hint(p: Dict[str, Any]) -> Tuple[str, List[str]]:
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)
    pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)
    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
    trades5 = buys5 + sells5

    if liq >= 250_000:
        liq_tag = "Liquidity: High"
    elif liq >= 60_000:
        liq_tag = "Liquidity: Medium"
    else:
        liq_tag = "Liquidity: Low"

    if trades5 >= 60 and sells5 >= 10:
        flow_tag = "Flow: Hot"
    elif trades5 >= 25 and sells5 >= 6:
        flow_tag = "Flow: Active"
    else:
        flow_tag = "Flow: Thin"

    tags = [
        liq_tag,
        flow_tag,
        f"Trend(1h): {fmt_pct(pc1h)}",
        f"Micro(5m): {fmt_pct(pc5)}",
        f"Vol(5m): {fmt_usd(vol5)}",
    ]

    # Conservative decisioning
    decision = "NO ENTRY"
    if liq >= 30_000 and vol24 >= 20_000 and trades5 >= 12 and sells5 >= 3:
        if pc1h > 6 and vol5 > 2_500 and pc5 >= -3:
            decision = "ENTRY (scalp)"
        else:
            decision = "WATCH / WAIT"

    return decision, tags


# -----------------------------
# Filtering / Debug
# -----------------------------
def filter_pairs_with_debug(
    pairs: List[Dict[str, Any]],
    chain: str,
    any_dex: bool,
    allowed_dexes: set,
    min_liq: float,
    min_vol24: float,
    min_trades_m5: int,
    min_sells_m5: int,
    max_buy_sell_imbalance: int,
    block_suspicious_names: bool,
    block_majors: bool,
    min_age_min: int,
    max_age_min: int,
    enforce_age: bool,
    hide_solana_unverified: bool,
):
    stats = Counter()
    reasons = Counter()

    stats["total_in"] = len(pairs)
    out = []

    for p in pairs:
        # chain
        if (p.get("chainId") or "").lower() != chain.lower():
            reasons["chain_mismatch"] += 1
            continue
        stats["after_chain"] += 1

        # dex filter
        if not any_dex and allowed_dexes:
            if (p.get("dexId") or "").lower() not in allowed_dexes:
                reasons["dex_not_allowed"] += 1
                continue
        stats["after_dex"] += 1

        # majors/stables
        if block_majors and is_major_or_stable(p):
            reasons["major_or_stable"] += 1
            continue
        stats["after_major_filter"] += 1

        # age window
        if enforce_age:
            age = pair_age_minutes(p)
            if age is None:
                # treat unknown as "too old" to avoid random legacy pools
                reasons["age_unknown"] += 1
                continue
            if age < float(min_age_min):
                reasons["age_too_new"] += 1
                continue
            if age > float(max_age_min):
                reasons["age_too_old"] += 1
                continue
        stats["after_age"] += 1

        # solana "unverified-like" hide (heuristic)
        if hide_solana_unverified and solana_unverified_heuristic(p):
            reasons["sol_unverified_like"] += 1
            continue
        stats["after_sol_check"] += 1

        # numeric filters
        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
        buys = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
        sells = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
        trades = buys + sells

        if liq < float(min_liq):
            reasons["liq<min"] += 1
            continue
        stats["after_liq"] += 1

        if vol24 < float(min_vol24):
            reasons["vol24<min"] += 1
            continue
        stats["after_vol24"] += 1

        if trades < int(min_trades_m5):
            reasons["trades<min"] += 1
            continue
        stats["after_trades"] += 1

        if sells < int(min_sells_m5):
            reasons["sells<min"] += 1
            continue
        stats["after_sells"] += 1

        # imbalance
        if sells > 0 and buys > 0:
            imbalance = max(buys, sells) / max(1, min(buys, sells))
            if imbalance > int(max_buy_sell_imbalance):
                reasons["imbalance>max"] += 1
                continue
        else:
            reasons["buys_or_sells_zero"] += 1
            continue
        stats["after_imbalance"] += 1

        # name filter
        if block_suspicious_names and is_name_suspicious(p):
            reasons["suspicious_name"] += 1
            continue
        stats["after_name"] += 1

        out.append(p)

    return out, stats, reasons


# -----------------------------
# Output dynamics
# -----------------------------
def dedupe_mode(pairs: List[Dict[str, Any]], by_base_token: bool) -> List[Dict[str, Any]]:
    """
    If by_base_token=True => keep best pool per base token.
    Else => keep best per pairAddress/url.
    """
    if not pairs:
        return []

    if not by_base_token:
        uniq = {}
        for p in pairs:
            k = p.get("pairAddress") or p.get("url")
            if not k:
                continue
            uniq[k] = p
        return list(uniq.values())

    best = {}
    for p in pairs:
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
        if not base_addr:
            continue
        cur = best.get(base_addr)
        if cur is None:
            best[base_addr] = p
        else:
            # choose by (liq, vol24)
            liq1 = float(safe_get(p, "liquidity", "usd", default=0) or 0)
            vol1 = float(safe_get(p, "volume", "h24", default=0) or 0)
            liq0 = float(safe_get(cur, "liquidity", "usd", default=0) or 0)
            vol0 = float(safe_get(cur, "volume", "h24", default=0) or 0)
            if (liq1, vol1) > (liq0, vol0):
                best[base_addr] = p
    return list(best.values())


def sample_seeds(seeds: List[str], k: int, refresh: bool) -> List[str]:
    seeds = [s.strip() for s in seeds if s.strip()]
    if not seeds:
        return []

    k = max(1, min(int(k), len(seeds)))

    if "seed_sample" not in st.session_state or refresh:
        st.session_state["seed_sample"] = random.sample(seeds, k)

    return st.session_state["seed_sample"]


# -----------------------------
# Storage (portfolio)
# -----------------------------
def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ts_utc",
                    "chain",
                    "dex",
                    "base_symbol",
                    "quote_symbol",
                    "base_token_address",
                    "pair_address",
                    "dexscreener_url",
                    "swap_url",
                    "score",
                    "action",
                    "tags",
                    "entry_price_usd",
                    "note",
                    "active",
                ],
            )
            w.writeheader()


def load_portfolio() -> List[Dict[str, Any]]:
    ensure_storage()
    rows = []
    with open(TRADES_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def save_portfolio(rows: List[Dict[str, Any]]):
    ensure_storage()
    with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ts_utc",
                "chain",
                "dex",
                "base_symbol",
                "quote_symbol",
                "base_token_address",
                "pair_address",
                "dexscreener_url",
                "swap_url",
                "score",
                "action",
                "tags",
                "entry_price_usd",
                "note",
                "active",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


def log_swap_intent(p: Dict[str, Any], score: float, action: str, tags: List[str], swap_url: str):
    rows = load_portfolio()

    chain = (p.get("chainId") or "").lower()
    dex = p.get("dexId") or ""
    base_sym = safe_get(p, "baseToken", "symbol", default="") or ""
    quote_sym = safe_get(p, "quoteToken", "symbol", default="") or ""
    base_addr = safe_get(p, "baseToken", "address", default="") or ""
    pair_addr = p.get("pairAddress", "") or ""
    url = p.get("url", "") or ""
    price = safe_get(p, "priceUsd", default="") or ""

    # avoid duplicates (pair-level preferred)
    for r in rows:
        if r.get("active") != "1":
            continue
        if pair_addr and (r.get("pair_address", "").lower() == pair_addr.lower()):
            return "ALREADY_EXISTS"
        if not pair_addr and base_addr and (r.get("base_token_address", "").lower() == base_addr.lower()):
            return "ALREADY_EXISTS"

    rows.append(
        {
            "ts_utc": now_utc_str(),
            "chain": chain,
            "dex": dex,
            "base_symbol": base_sym,
            "quote_symbol": quote_sym,
            "base_token_address": base_addr,
            "pair_address": pair_addr,
            "dexscreener_url": url,
            "swap_url": swap_url,
            "score": str(score),
            "action": action,
            "tags": " | ".join(tags),
            "entry_price_usd": str(price),
            "note": "",
            "active": "1",
        }
    )
    save_portfolio(rows)
    return "OK"


def portfolio_reco(entry: float, current: float, pc1h: float, pc5: float) -> str:
    if entry <= 0 or current <= 0:
        return "HOLD / WAIT"
    change = (current - entry) / entry * 100.0

    if change >= 35 and pc5 < 0 and pc1h < 0:
        return "SELL (take profit)"
    if change >= 20 and pc5 < -2:
        return "TRIM / TP"
    if change <= -20 and pc1h < -5:
        return "CUT / RISK"
    if pc1h > 5 and pc5 >= 0:
        return "HOLD (momentum)"
    return "HOLD / WAIT"


# -----------------------------
# Presets (more dynamic but still guarded)
# -----------------------------
PRESETS = {
    "Ultra Early (safer)": {
        "top_n": 20,
        "min_liq": 2000,
        "min_vol24": 5000,
        "min_trades_m5": 6,
        "min_sells_m5": 2,
        "max_imbalance": 18,
        "block_suspicious_names": True,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 20,
        "max_age_min": 14400,  # 10 days
        "hide_solana_unverified": True,
        "seed_k": 12,
        "dedupe_by_base": True,
        "seeds": "new, launch, fairlaunch, airdrop, points, claim, v2, v3, inu, pepe, ai, agent, bot, pump, stealth, community, trend",
    },
    "Early Momentum": {
        "top_n": 20,
        "min_liq": 4000,
        "min_vol24": 12000,
        "min_trades_m5": 10,
        "min_sells_m5": 3,
        "max_imbalance": 16,
        "block_suspicious_names": True,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 30,
        "max_age_min": 28800,  # 20 days
        "hide_solana_unverified": True,
        "seed_k": 14,
        "dedupe_by_base": True,
        "seeds": "launch, new, pump, fairlaunch, presale, stealth, inu, pepe, ai agent, agent, bot, points, claim, v2, v3, community, trend",
    },
    "Balanced (default)": {
        "top_n": 15,
        "min_liq": 15000,
        "min_vol24": 60000,
        "min_trades_m5": 18,
        "min_sells_m5": 5,
        "max_imbalance": 10,
        "block_suspicious_names": True,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 60,
        "max_age_min": 43200,  # 30 days
        "hide_solana_unverified": True,
        "seed_k": 10,
        "dedupe_by_base": True,
        "seeds": "launch, new, trend, community, pump, inu, pepe, ai, ai agent, bot, points, claim, v2, v3",
    },
    "Wide Net (explore)": {
        "top_n": 25,
        "min_liq": 8000,
        "min_vol24": 25000,
        "min_trades_m5": 10,
        "min_sells_m5": 3,
        "max_imbalance": 14,
        "block_suspicious_names": False,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 20,
        "max_age_min": 86400,  # 60 days
        "hide_solana_unverified": True,
        "seed_k": 14,
        "dedupe_by_base": True,
        "seeds": "new, launch, trend, community, pump, fairlaunch, presale, stealth, airdrop, claim, points, v2, v3, inu, pepe, ai, agent, bot",
    },
}


# -----------------------------
# Swap URL builders (core)
# -----------------------------
def build_swap_url(chain: str, base_addr: str) -> str:
    chain = (chain or "").lower().strip()
    base_addr = (base_addr or "").strip()

    if not base_addr:
        return ""

    if chain == "bsc":
        # outputCurrency = token contract
        return f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"

    if chain == "solana":
        # Jupiter: SOL -> token mint
        # Common working format:
        #   https://jup.ag/swap/SOL-<MINT>
        return f"https://jup.ag/swap/SOL-{base_addr}"

    return ""


# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="DEX Scout", layout="wide")

    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")

        page = st.radio("Page", ["Scout", "Portfolio"], index=0)

        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=list(PRESETS.keys()).index("Balanced (default)"))
        preset = PRESETS[preset_name]

        st.divider()

        chain = st.selectbox("Chain", options=["bsc", "solana"], index=0)

        st.caption("DEX filter")
        any_dex = st.checkbox("Any DEX (no filter)", value=True)

        dex_options = CHAIN_DEX_PRESETS.get(chain, [])
        selected_dexes = []
        if not any_dex:
            selected_dexes = st.multiselect(
                "Allowed DEXes",
                options=dex_options,
                default=dex_options[:3] if len(dex_options) >= 3 else dex_options,
            )

        top_n = st.slider("Top N", 5, 50, int(preset["top_n"]), step=1)
        min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=int(preset["min_liq"]), step=500)
        min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=int(preset["min_vol24"]), step=1000)

        st.divider()
        st.caption("Real trading filters")
        min_trades_m5 = st.slider("Min trades (5m)", 0, 200, int(preset["min_trades_m5"]), step=1)
        min_sells_m5 = st.slider("Min sells (5m)", 0, 80, int(preset["min_sells_m5"]), step=1)
        max_buy_sell_imbalance = st.slider("Max buy/sell imbalance", 1, 30, int(preset["max_imbalance"]), step=1)

        st.divider()
        st.caption("Safety guards")
        block_majors = st.checkbox("Filter majors/stables (base)", value=bool(preset["block_majors"]))
        block_suspicious_names = st.checkbox("Filter suspicious tickers", value=bool(preset["block_suspicious_names"]))

        enforce_age = st.checkbox("Pair age window (anti-rug + still early)", value=bool(preset["enforce_age"]))
        min_age_min = st.number_input("Min age (minutes)", min_value=0, value=int(preset["min_age_min"]), step=10)
        max_age_min = st.number_input("Max age (minutes)", min_value=30, value=int(preset["max_age_min"]), step=60)

        hide_solana_unverified = st.checkbox(
            "Hide Solana 'unverified-like' (heuristic)",
            value=bool(preset["hide_solana_unverified"]),
            help="Conservative: hides very-new/low-liq/overpumped Solana pairs. Turn OFF to see everything.",
        )

        st.divider()
        st.caption("Output dynamics")
        dedupe_by_base = st.checkbox("Top tokens mode (dedupe by base token)", value=bool(preset["dedupe_by_base"]))
        seed_k = st.slider("Seed sampler K", 3, 20, int(preset["seed_k"]), step=1)

        st.divider()
        st.caption("Queries (comma-separated)")
        seeds_raw = st.text_area("Search seeds", value=str(preset["seeds"]), height=120)

        colA, colB = st.columns([1, 1])
        with colA:
            refresh = st.button("Refresh now", use_container_width=True)
        with colB:
            clear_cache = st.button("Clear cache", use_container_width=True)

        if clear_cache:
            st.cache_data.clear()

    # -------------------- SCOUT PAGE --------------------
    if page == "Scout":
        st.title("DEX Scout — early candidates (DEXScreener API)")
        st.caption("Фокус: дрібні/ранні монети. Majors/stables відсікаються. Stable core: BSC + Solana.")

        seeds = [x.strip() for x in seeds_raw.split(",") if x.strip()]
        if not seeds:
            st.info("Add at least 1 seed query in the sidebar.")
            return

        sampled = sample_seeds(seeds, seed_k, refresh)
        st.caption(f"Seeds sampled ({len(sampled)}/{len(seeds)}): " + ", ".join(sampled))

        all_pairs: List[Dict[str, Any]] = []
        query_failures = 0

        for q in sampled:
            try:
                all_pairs.extend(fetch_latest_pairs_for_query(q))
                time.sleep(0.10)  # gentle pacing
            except Exception as e:
                query_failures += 1
                st.warning(f"Query failed: {q} — {e}")

        if query_failures and not all_pairs:
            st.error("All sampled queries failed. Try Refresh now, reduce seeds, or wait a bit.")
            return

        # Dedupe first (pair/url)
        pairs = dedupe_mode(all_pairs, by_base_token=False)
        # Optional: dedupe by base token (for "Top tokens mode")
        pairs = dedupe_mode(pairs, by_base_token=dedupe_by_base)

        allowed = set([d.lower() for d in selected_dexes]) if selected_dexes else set()

        filtered, fstats, freasons = filter_pairs_with_debug(
            pairs=pairs,
            chain=chain,
            any_dex=any_dex,
            allowed_dexes=allowed,
            min_liq=float(min_liq),
            min_vol24=float(min_vol24),
            min_trades_m5=int(min_trades_m5),
            min_sells_m5=int(min_sells_m5),
            max_buy_sell_imbalance=int(max_buy_sell_imbalance),
            block_suspicious_names=bool(block_suspicious_names),
            block_majors=bool(block_majors),
            min_age_min=int(min_age_min),
            max_age_min=int(max_age_min),
            enforce_age=bool(enforce_age),
            hide_solana_unverified=bool(hide_solana_unverified),
        )

        ranked = []
        for p in filtered:
            s = score_pair(p)
            decision, tags = build_trade_hint(p)
            ranked.append((s, decision, tags, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:top_n]

        st.metric("Passed filters", len(ranked))

        with st.expander("Why 0 results? (Filter Debug)", expanded=(len(ranked) == 0)):
            st.write("**Fetched:**", len(all_pairs), " • **Deduped:**", len(pairs))
            st.write("**Counts (pipeline):**")
            st.json(dict(fstats))
            st.write("**Top reject reasons:**")
            top = freasons.most_common(12)
            st.write({k: v for k, v in top} if top else "No rejects counted (unexpected).")

        if not ranked:
            st.info("0 results. Try: lower Min Liquidity/Vol24 a bit, widen age window, or hit Refresh to resample seeds.")
            return

        # Render cards
        for idx, (s, decision, tags, p) in enumerate(ranked, start=1):
            base = safe_get(p, "baseToken", "symbol", default="???") or "???"
            quote = safe_get(p, "quoteToken", "symbol", default="???") or "???"
            dex = p.get("dexId", "?")
            url = p.get("url", "")
            price = safe_get(p, "priceUsd", default=None)

            liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
            vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
            vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)
            pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
            pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)
            buys = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
            sells = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)

            chain_id = (p.get("chainId") or "").lower()
            base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
            pair_addr = (p.get("pairAddress") or "").strip()

            swap_url = build_swap_url(chain_id, base_addr)

            st.markdown("---")
            left, mid, right = st.columns([3, 2, 2])

            with left:
                st.markdown(f"## {base}/{quote}")
                st.caption(f"{chain_id} • {dex}")
                if url:
                    st.link_button("Open DexScreener", url, use_container_width=True, key=f"ds_{idx}_{hkey(url)}")

                if base_addr:
                    st.caption("Token contract (baseToken.address)")
                    st.code(base_addr, language="text")
                if pair_addr:
                    st.caption("Pair / pool address")
                    st.code(pair_addr, language="text")

                age = pair_age_minutes(p)
                if age is not None:
                    st.caption(f"Pair age: ~{int(age)} min")

            with mid:
                st.write(f"**Price:** ${float(price):.8f}" if price else "**Price:** n/a")
                st.write(f"**Liq:** {fmt_usd(liq)}")
                st.write(f"**Vol24:** {fmt_usd(vol24)}")
                st.write(f"**Vol m5:** {fmt_usd(vol5)}")
                st.write(f"**Δ m5:** {fmt_pct(pc5)}")
                st.write(f"**Δ h1:** {fmt_pct(pc1h)}")
                st.write(f"**Buys/Sells (m5):** {buys}/{sells}")
                st.write(f"**Score:** {s}")

            with right:
                st.write("**Action:**")
                st.markdown(action_badge(decision), unsafe_allow_html=True)

                st.write("**Tags:**")
                for t in tags:
                    st.write(f"• {t}")

                # Unique key base: use URL+pair+idx
                keybase = hkey(url or "", pair_addr or "", base_addr or "", str(idx))

                # Log (works for both chains)
                if base_addr:
                    if st.button("Log → Portfolio (I swapped)", key=f"log_{keybase}", use_container_width=True):
                        if not swap_url:
                            st.info("Swap URL missing for this chain/token (should not happen for BSC/Solana).")
                        else:
                            res = log_swap_intent(p, s, decision, tags, swap_url)
                            if res == "OK":
                                st.success("Logged to Portfolio (entry snapshot saved).")
                            else:
                                st.info("Already in Portfolio (active).")

                # Swap button
                if swap_url:
                    swap_label = "Open Swap (PancakeSwap)" if chain_id == "bsc" else "Open Swap (Jupiter)"
                    st.link_button(
                        swap_label,
                        swap_url,
                        use_container_width=True,
                        key=f"swap_{keybase}",
                    )
                else:
                    st.caption("Swap disabled (unexpected).")

                # Solana safety note
                if chain_id == "solana":
                    st.caption("Solana: перед свопом глянь Jupiter warnings (JupShield).")

    # -------------------- PORTFOLIO PAGE --------------------
    else:
        st.title("Portfolio / Watchlist")
        st.caption("Entries appear here after clicking **Log → Portfolio (I swapped)** in Scout.")

        rows = load_portfolio()
        active_rows = [r for r in rows if r.get("active") == "1"]
        closed_rows = [r for r in rows if r.get("active") != "1"]

        topbar = st.columns([2, 2, 2])
        with topbar[0]:
            st.metric("Active", len(active_rows))
        with topbar[1]:
            st.metric("Closed", len(closed_rows))
        with topbar[2]:
            with open(TRADES_CSV, "rb") as f:
                st.download_button("Download portfolio.csv", f, file_name="portfolio.csv", use_container_width=True)

        st.markdown("---")

        if not active_rows:
            st.info("Portfolio пустий. Йди в Scout → натисни **Log → Portfolio (I swapped)**.")
            return

        st.subheader("Active positions")
        for idx, r in enumerate(active_rows):
            base_addr = (r.get("base_token_address") or "").strip()
            chain = (r.get("chain") or "").strip().lower()
            base_sym = r.get("base_symbol") or "???"
            quote_sym = r.get("quote_symbol") or "???"
            entry_price_str = r.get("entry_price_usd") or "0"

            try:
                entry_price = float(entry_price_str)
            except Exception:
                entry_price = 0.0

            best = best_pair_for_token(chain, base_addr) if (chain and base_addr) else None
            cur_price = 0.0
            pc1h = 0.0
            pc5 = 0.0
            liq = 0.0

            if best:
                try:
                    cur_price = float(best.get("priceUsd") or 0)
                except Exception:
                    cur_price = 0.0
                pc1h = float(safe_get(best, "priceChange", "h1", default=0) or 0)
                pc5 = float(safe_get(best, "priceChange", "m5", default=0) or 0)
                liq = float(safe_get(best, "liquidity", "usd", default=0) or 0)

            reco = portfolio_reco(entry_price, cur_price, pc1h, pc5)

            pnl = 0.0
            if entry_price > 0 and cur_price > 0:
                pnl = (cur_price - entry_price) / entry_price * 100.0

            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])

            with c1:
                st.markdown(f"### {base_sym}/{quote_sym}")
                st.caption(f"Chain: {chain} • DEX: {r.get('dex','')}")
                st.code(base_addr, language="text")
                if r.get("dexscreener_url"):
                    st.link_button("DexScreener", r["dexscreener_url"], use_container_width=True, key=f"p_ds_{idx}_{hkey(r['dexscreener_url'])}")
                if r.get("swap_url"):
                    st.link_button("Swap", r["swap_url"], use_container_width=True, key=f"p_sw_{idx}_{hkey(r['swap_url'])}")

            with c2:
                st.write(f"**Entry:** ${entry_price_str}")
                st.write(f"**Now:** ${cur_price:.8f}" if cur_price else "**Now:** n/a")
                st.write(f"**PnL:** {pnl:+.2f}%" if entry_price and cur_price else "**PnL:** n/a")
                st.write(f"**Liq:** {fmt_usd(liq)}" if liq else "**Liq:** n/a")

            with c3:
                st.write(f"**Δ m5:** {fmt_pct(pc5)}")
                st.write(f"**Δ h1:** {fmt_pct(pc1h)}")
                st.write(f"**Reco:** `{reco}`")

                note_key = f"note_{idx}_{hkey(base_addr, chain)}"
                note_val = st.text_input("Note", value=r.get("note", ""), key=note_key)

            with c4:
                close_key = f"close_{idx}_{hkey(base_addr, chain)}"
                delete_key = f"delete_{idx}_{hkey(base_addr, chain)}"

                close = st.checkbox("Close (archive)", value=False, key=close_key)
                delete = st.checkbox("Delete row", value=False, key=delete_key)

                if st.button("Apply", key=f"apply_{idx}_{hkey(base_addr, chain)}", use_container_width=True):
                    all_rows = load_portfolio()

                    # Update note/close by base token address (active only)
                    for rr in all_rows:
                        if rr.get("active") == "1" and (rr.get("base_token_address") or "").lower() == base_addr.lower():
                            rr["note"] = note_val
                            if close:
                                rr["active"] = "0"
                            break

                    if delete:
                        all_rows = [
                            x for x in all_rows
                            if not (x.get("active") == "1" and (x.get("base_token_address") or "").lower() == base_addr.lower())
                        ]

                    save_portfolio(all_rows)
                    st.success("Saved.")
                    st.rerun()

            st.markdown("---")

        if closed_rows:
            with st.expander("Closed / Archived"):
                for r in closed_rows[-50:]:
                    st.write(
                        f"{r.get('ts_utc','')} — {r.get('base_symbol','')}/{r.get('quote_symbol','')} "
                        f"— entry ${r.get('entry_price_usd','')} — {r.get('action','')}"
                    )


if __name__ == "__main__":
    main()
