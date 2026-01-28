# app.py — DEX Scout v0.3.4
# Core goals:
# - Stable core: BSC + Solana only
# - Early/small candidates focus (majors filtered)
# - Fix: "NO ENTRY" badge must be RED (was green due to substring match)
# - More dynamic output WITHOUT lowering safety:
#   * Seed sampling (rotating subset of seeds per run) -> variety
#   * Pair age filter (avoid ultra-fresh rugs, still early)
#   * Diversity: dedupe by base token (toggle) to avoid same token spamming
#   * Quality guards (liq/vol sanity, activity checks remain)
#
# Notes:
# - Swap URL: PancakeSwap only for BSC (for now)
# - Solana swap button disabled (later map chain -> swap)

import os
import re
import csv
import time
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import requests
import streamlit as st

VERSION = "0.3.4"

DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
TRADES_CSV = os.path.join(DATA_DIR, "portfolio.csv")

# Stable core: only BSC + Solana
CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap", "woofi"],
    "solana": ["raydium", "orca", "meteora"],
}

# Softer, more realistic ticker validation:
# allow unicode letters/digits + common ticker chars ($ _ - .) + spaces
NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

# Base token majors/stables (filter OUT as base; quotes are allowed)
MAJOR_BASE_SYMBOLS = {
    "BTC", "WBTC", "ETH", "WETH", "BNB", "WBNB", "SOL", "WSOL",
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDD",
    "XRP", "ADA", "DOGE", "TRX", "DOT", "MATIC", "POL", "AVAX",
}

# Common stable quote symbols (useful for labeling only)
STABLE_QUOTES = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDD"}

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


def clamp_float(x: float, lo: float = 0.0, hi: float = 1e18) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return lo


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def pair_age_minutes(p: Dict[str, Any]) -> Optional[int]:
    """
    DexScreener sometimes provides pairCreatedAt (ms since epoch).
    Return age in minutes if available.
    """
    ts = safe_get(p, "pairCreatedAt", default=None)
    if ts is None:
        return None
    try:
        ts = int(ts)
        age_ms = max(0, utc_now_ms() - ts)
        return int(age_ms / 60000)
    except Exception:
        return None


# -----------------------------
# HTTP with retry/backoff
# -----------------------------
def _http_get_json(url: str, params: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> Any:
    """
    Basic retry/backoff wrapper for DexScreener calls.
    - Retries on 429 / 5xx / common network errors
    - Exponential backoff (0.6s, 1.2s, 2.4s)
    """
    last_err = None
    backoff = 0.6

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


@st.cache_data(ttl=40, show_spinner=False)
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
# Filtering / Scoring
# -----------------------------
def is_name_suspicious(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip()
    if not base_sym:
        return True
    if len(base_sym) > 40:
        return True
    return not bool(NAME_OK_RE.match(base_sym))


def is_major_base(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip().upper()
    if not base_sym:
        return False
    # If base is a known major or stable => not our target
    return base_sym in MAJOR_BASE_SYMBOLS


def score_pair(p: Dict[str, Any]) -> float:
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)
    pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)
    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
    trades5 = buys5 + sells5

    age_min = pair_age_minutes(p)
    # age bonus: prefer not-too-fresh, not-too-old (light touch)
    age_bonus = 0.0
    if age_min is not None:
        if 30 <= age_min <= 60 * 24 * 7:   # 30 min .. 7 days
            age_bonus = 30.0
        elif age_min < 30:
            age_bonus = -15.0

    s = 0.0
    s += min(liq / 1000.0, 420.0)
    s += min(vol24 / 10000.0, 320.0)
    s += min(vol5 / 2000.0, 220.0)
    s += min(trades5 * 2.0, 120.0)
    s += max(min(pc1h, 80.0), -80.0) * 0.25
    s += max(min(pc5, 30.0), -30.0) * 0.20
    s += age_bonus
    return round(s, 2)


def action_badge(action: str) -> str:
    """
    Fix: "NO ENTRY" contains "ENTRY" substring, so it was wrongly green.
    """
    a = (action or "").upper().strip()

    if a.startswith("NO") or a == "NO ENTRY":
        color = "#e53e3e"      # red
        bg = "rgba(229,62,62,0.12)"
    elif a.startswith("ENTRY") or a.startswith("BUY") or "ENTRY (" in a:
        color = "#1f9d55"      # green
        bg = "rgba(31,157,85,0.15)"
    elif "WATCH" in a or "WAIT" in a:
        color = "#d69e2e"      # yellow
        bg = "rgba(214,158,46,0.15)"
    else:
        color = "#e53e3e"
        bg = "rgba(229,62,62,0.12)"

    return f"""
    <span style="
      display:inline-block;
      padding:6px 12px;
      border-radius:999px;
      border:1px solid {color};
      background:{bg};
      color:{color};
      font-weight:800;
      font-size:12px;
      letter-spacing:0.4px;
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

    age_min = pair_age_minutes(p)
    if age_min is None:
        age_tag = "Age: n/a"
    elif age_min < 60:
        age_tag = f"Age: {age_min}m"
    elif age_min < 60 * 24:
        age_tag = f"Age: {age_min // 60}h"
    else:
        age_tag = f"Age: {age_min // (60*24)}d"

    # liquidity band
    if liq >= 250_000:
        liq_tag = "Liquidity: High"
    elif liq >= 60_000:
        liq_tag = "Liquidity: Medium"
    else:
        liq_tag = "Liquidity: Low"

    # flow band
    if trades5 >= 60 and sells5 >= 10:
        flow_tag = "Flow: Hot"
    elif trades5 >= 20 and sells5 >= 4:
        flow_tag = "Flow: Active"
    else:
        flow_tag = "Flow: Thin"

    trend_tag = f"Trend(1h): {fmt_pct(pc1h)}"
    micro_tag = f"Micro(5m): {fmt_pct(pc5)}"
    vol_tag = f"Vol(5m): {fmt_usd(vol5)}"

    tags = [age_tag, liq_tag, flow_tag, trend_tag, micro_tag, vol_tag]

    # decision
    decision = "NO ENTRY"
    # entry conditions (conservative scalp)
    if liq >= 25_000 and vol24 >= 25_000 and trades5 >= 15 and sells5 >= 3:
        # avoid falling knife
        if pc1h > 2 and vol5 > 800 and pc5 >= -2:
            decision = "ENTRY (scalp)"
        else:
            decision = "WATCH / WAIT"

    return decision, tags


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
    # age safety window (minutes)
    min_age_min: int,
    max_age_min: int,
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

        # dex
        if not any_dex and allowed_dexes:
            if (p.get("dexId") or "").lower() not in allowed_dexes:
                reasons["dex_not_allowed"] += 1
                continue
        stats["after_dex"] += 1

        # majors/stables base filter
        if block_majors and is_major_base(p):
            reasons["major_or_stable_base"] += 1
            continue
        stats["after_major_filter"] += 1

        # age filter (anti rug: avoid ultra-fresh, and avoid very old)
        age_min = pair_age_minutes(p)
        if age_min is not None:
            if age_min < min_age_min:
                reasons["age_too_fresh"] += 1
                continue
            if age_min > max_age_min:
                reasons["age_too_old"] += 1
                continue
        stats["after_age"] += 1

        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
        buys = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
        sells = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
        trades = buys + sells

        # quality sanity: avoid super-low liq pairs even if vol spikes
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
            if imbalance > max_buy_sell_imbalance:
                reasons["imbalance>max"] += 1
                continue
        else:
            reasons["buys_or_sells_zero"] += 1
            continue
        stats["after_imbalance"] += 1

        if block_suspicious_names and is_name_suspicious(p):
            reasons["suspicious_name"] += 1
            continue
        stats["after_name"] += 1

        out.append(p)

    return out, stats, reasons


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

    # avoid duplicates (pair-level)
    for r in rows:
        if r.get("active") != "1":
            continue
        if pair_addr and (r.get("pair_address", "").lower() == pair_addr.lower()):
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
# Presets (more dynamic without more risk)
# -----------------------------
PRESETS = {
    # SAFER early: avoid ultra-fresh (min age), decent liq/vol
    "Early Safe (default)": {
        "top_n": 15,
        "min_liq": 8000,
        "min_vol24": 25000,
        "min_trades_m5": 8,
        "min_sells_m5": 2,
        "max_imbalance": 12,
        "block_suspicious_names": True,
        "block_majors": True,
        "min_age_min": 30,          # avoid insta-rugs
        "max_age_min": 60 * 24 * 14,  # 14 days
        "seed_sample_k": 10,
        "seeds": "launch, fairlaunch, stealth, presale, airdrop, claim, points, v2, v3, ai agent, agent, bot, pepe, inu, meme, pump",
    },
    # More variety but still guarded by age and majors filter
    "Wide Net (explore)": {
        "top_n": 25,
        "min_liq": 3000,
        "min_vol24": 10000,
        "min_trades_m5": 3,
        "min_sells_m5": 1,
        "max_imbalance": 15,
        "block_suspicious_names": True,
        "block_majors": True,
        "min_age_min": 20,
        "max_age_min": 60 * 24 * 10,
        "seed_sample_k": 14,
        "seeds": "new, launch, trend, community, pump, fairlaunch, presale, stealth, airdrop, claim, points, v2, v3, inu, pepe, ai, agent, bot",
    },
    # Hot flow: more active pairs, less noise
    "Flow Hot (strict)": {
        "top_n": 15,
        "min_liq": 15000,
        "min_vol24": 60000,
        "min_trades_m5": 15,
        "min_sells_m5": 4,
        "max_imbalance": 10,
        "block_suspicious_names": True,
        "block_majors": True,
        "min_age_min": 45,
        "max_age_min": 60 * 24 * 7,
        "seed_sample_k": 10,
        "seeds": "launch, fairlaunch, stealth, airdrop, ai agent, bot, meme, pepe, inu, pump",
    },
    # Solana-specific scan (still only via chain selector, but seeds tuned)
    "Solana Early": {
        "top_n": 20,
        "min_liq": 5000,
        "min_vol24": 15000,
        "min_trades_m5": 4,
        "min_sells_m5": 1,
        "max_imbalance": 15,
        "block_suspicious_names": True,
        "block_majors": True,
        "min_age_min": 20,
        "max_age_min": 60 * 24 * 10,
        "seed_sample_k": 14,
        "seeds": "launch, fairlaunch, stealth, airdrop, claim, points, v2, v3, meme, pepe, inu, ai agent, agent, bot",
    },
}


def sample_seeds(all_seeds: List[str], k: int, seed_key: str) -> List[str]:
    """
    Rotating subset of seeds for more variety without reducing safety filters.
    We keep the sample stable until user hits Refresh.
    """
    all_seeds = [s.strip() for s in all_seeds if s.strip()]
    if not all_seeds:
        return []

    k = max(1, min(int(k), len(all_seeds)))

    # stable random for this session sample unless refresh
    if "seed_sampler" not in st.session_state:
        st.session_state.seed_sampler = {}

    if seed_key in st.session_state.seed_sampler:
        return st.session_state.seed_sampler[seed_key]

    rnd = random.Random()
    rnd.seed(f"{seed_key}:{int(time.time())}")  # fresh enough per first run
    picked = rnd.sample(all_seeds, k=k)
    st.session_state.seed_sampler[seed_key] = picked
    return picked


def main():
    st.set_page_config(page_title="DEX Scout", layout="wide")

    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")

        page = st.radio("Page", ["Scout", "Portfolio"], index=0)

        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
        preset = PRESETS[preset_name]

        st.divider()

        chain = st.selectbox("Chain", options=list(CHAIN_DEX_PRESETS.keys()), index=0)

        st.caption("DEX filter")
        any_dex = st.checkbox("Any DEX (no filter)", value=True)

        dex_options = CHAIN_DEX_PRESETS.get(chain, [])
        selected_dexes = []
        if not any_dex:
            selected_dexes = st.multiselect(
                "Allowed DEXes",
                options=dex_options,
                default=dex_options[:2] if len(dex_options) >= 2 else dex_options,
            )

        top_n = st.slider("Top N", 5, 50, int(preset["top_n"]), step=1)

        min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=int(preset["min_liq"]), step=1000)
        min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=int(preset["min_vol24"]), step=5000)

        st.divider()
        st.caption("Real trading filters")
        min_trades_m5 = st.slider("Min trades (5m)", 0, 200, int(preset["min_trades_m5"]), step=1)
        min_sells_m5 = st.slider("Min sells (5m)", 0, 80, int(preset["min_sells_m5"]), step=1)
        max_buy_sell_imbalance = st.slider("Max buy/sell imbalance", 1, 30, int(preset["max_imbalance"]), step=1)

        st.divider()
        st.caption("Safety guards")
        block_majors = st.checkbox("Filter majors/stables (base)", value=bool(preset["block_majors"]))
        block_suspicious_names = st.checkbox("Filter suspicious tickers", value=bool(preset["block_suspicious_names"]))

        st.caption("Pair age window (anti-rug + still early)")
        min_age_min = st.number_input("Min age (minutes)", min_value=0, value=int(preset["min_age_min"]), step=5)
        max_age_min = st.number_input("Max age (minutes)", min_value=60, value=int(preset["max_age_min"]), step=60)

        st.divider()
        st.caption("Output dynamics")
        top_tokens_mode = st.checkbox("Top tokens mode (dedupe by base token)", value=True)
        seed_sample_k = st.slider("Seed sampler K", 3, 20, int(preset["seed_sample_k"]), step=1)

        st.divider()
        st.caption("Queries (comma-separated)")
        seeds_text = st.text_area("Search seeds", value=str(preset["seeds"]), height=120)

        refresh = st.button("Refresh now")
        if refresh:
            st.cache_data.clear()
            # reset sampler so we get a fresh seed subset
            st.session_state.seed_sampler = {}

    # -------------------- SCOUT PAGE --------------------
    if page == "Scout":
        st.title("DEX Scout — early candidates (DEXScreener API)")
        st.caption("Фокус: дрібні/ранні монети. Majors/stables відсікаються. Stable core: BSC + Solana.")

        seeds_all = [s.strip() for s in seeds_text.split(",") if s.strip()]
        if not seeds_all:
            st.info("Add at least 1 seed query in the sidebar.")
            return

        # Seed sampling -> more variety without loosening safety
        seed_key = f"{chain}:{preset_name}:{seed_sample_k}:{seeds_text}"
        queries = sample_seeds(seeds_all, k=seed_sample_k, seed_key=seed_key)

        st.caption(f"Seeds sampled ({len(queries)}/{len(seeds_all)}): " + ", ".join(queries))

        all_pairs: List[Dict[str, Any]] = []
        query_failures = 0

        for q in queries:
            try:
                all_pairs.extend(fetch_latest_pairs_for_query(q))
                time.sleep(0.12)
            except Exception as e:
                query_failures += 1
                st.warning(f"Query failed: {q} — {e}")

        # Deduplicate by pairAddress/url
        uniq = {}
        for p in all_pairs:
            pa = p.get("pairAddress") or p.get("url")
            if not pa:
                continue
            uniq[pa] = p
        pairs = list(uniq.values())

        if query_failures and not pairs:
            st.error("All sampled queries failed or returned no data. Try Refresh or wait a bit.")
            return

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
        )

        ranked = []
        for p in filtered:
            s = score_pair(p)
            decision, tags = build_trade_hint(p)
            ranked.append((s, decision, tags, p))
        ranked.sort(key=lambda x: x[0], reverse=True)

        # Optional: top tokens mode (avoid same token spamming)
        if top_tokens_mode:
            seen = set()
            deduped_ranked = []
            for item in ranked:
                p = item[3]
                base_addr = (safe_get(p, "baseToken", "address", default="") or "").lower()
                base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").upper().strip()
                key = base_addr or base_sym
                if not key:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                deduped_ranked.append(item)
            ranked = deduped_ranked

        ranked = ranked[:top_n]

        st.metric("Passed filters", len(ranked))

        with st.expander("Why 0 results? (Filter Debug)", expanded=False):
            st.write("**Fetched:**", len(all_pairs), " • **Deduped:**", len(pairs))
            st.write("**Counts (pipeline):**")
            st.json(dict(fstats))
            st.write("**Top reject reasons:**")
            top = freasons.most_common(10)
            if top:
                st.write({k: v for k, v in top})
            else:
                st.write("No rejects counted (unexpected).")

        if not ranked:
            st.info("0 results. Try: lower Min Liquidity/Vol24 немного, or hit Refresh to resample seeds.")
            return

        # Render cards
        for s, decision, tags, p in ranked:
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

            base_addr = safe_get(p, "baseToken", "address", default="") or ""
            pair_addr = p.get("pairAddress", "") or ""
            chain_id = (p.get("chainId") or "").lower()

            # Swap URL: BSC PancakeSwap only
            swap_url = ""
            if base_addr and chain_id == "bsc":
                swap_url = f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"

            st.markdown("---")
            left, mid, right = st.columns([3, 2, 2])

            with left:
                st.markdown(f"### {base}/{quote}")
                st.caption(f"{chain_id} • {dex}")

                if url:
                    # unique label => avoids element collisions without using key
                    st.link_button(f"Open DexScreener • {base}/{quote}", url, use_container_width=True)

                if base_addr:
                    st.caption("Token contract (baseToken.address)")
                    st.code(base_addr, language="text")
                if pair_addr:
                    st.caption("Pair / pool address")
                    st.code(pair_addr, language="text")

            with mid:
                st.write(f"**Price:** ${price}" if price else "**Price:** n/a")
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
                    st.caption(f"- {t}")

                # Stable & safe: log only when swap exists (BSC). Solana swap later.
                if base_addr and swap_url:
                    # button keys must be unique; use pair + base
                    key_log = f"log_{chain_id}_{base_addr}_{pair_addr}"
                    if st.button("Log → Portfolio (I swapped)", key=key_log, use_container_width=True):
                        res = log_swap_intent(p, s, decision, tags, swap_url)
                        if res == "OK":
                            st.success("Logged to Portfolio (entry snapshot saved).")
                        else:
                            st.info("Already in Portfolio (active).")

                    st.link_button(f"Open Swap (PancakeSwap) • {base}/{quote}", swap_url, use_container_width=True)
                else:
                    st.caption("Swap disabled (BSC-only пока).")

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
                st.download_button("Download portfolio.csv", f, file_name="portfolio.csv")

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
                    st.link_button(f"DexScreener • {base_sym}/{quote_sym}", r["dexscreener_url"], use_container_width=True)
                if r.get("swap_url"):
                    st.link_button(f"Swap • {base_sym}/{quote_sym}", r["swap_url"], use_container_width=True)

            with c2:
                st.write(f"**Entry:** ${entry_price_str}")
                st.write(f"**Now:** ${cur_price:.8f}" if cur_price else "**Now:** n/a")
                st.write(f"**PnL:** {pnl:+.2f}%" if entry_price and cur_price else "**PnL:** n/a")
                st.write(f"**Liq:** {fmt_usd(liq)}" if liq else "**Liq:** n/a")

            with c3:
                st.write(f"**Δ m5:** {fmt_pct(pc5)}")
                st.write(f"**Δ h1:** {fmt_pct(pc1h)}")
                st.write(f"**Reco:** `{reco}`")

                note_key = f"note_{idx}_{base_addr}"
                note_val = st.text_input("Note", value=r.get("note", ""), key=note_key)

            with c4:
                close_key = f"close_{idx}_{base_addr}"
                delete_key = f"delete_{idx}_{base_addr}"

                close = st.checkbox("Close (archive)", value=False, key=close_key)
                delete = st.checkbox("Delete row", value=False, key=delete_key)

                if st.button("Apply", key=f"apply_{idx}_{base_addr}", use_container_width=True):
                    all_rows = load_portfolio()
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
                for r in closed_rows[-30:]:
                    st.write(
                        f"{r.get('ts_utc','')} — {r.get('base_symbol','')}/{r.get('quote_symbol','')} "
                        f"— entry ${r.get('entry_price_usd','')} — {r.get('action','')}"
                    )


if __name__ == "__main__":
    main()
