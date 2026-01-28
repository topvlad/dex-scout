# app.py — DEX Scout v0.3.5 (stable core: BSC + Solana)
# Focus: early/small candidates. Majors/stables filtered out. Safety guards kept stable.
# Core fixes/features:
# 1) FIX: "NO ENTRY" badge is RED-ish (not green)
# 2) Stable core: only BSC + Solana (no extra chains to keep mechanics simple & stable)
# 3) Stronger Filter Debug: shows ALL pipeline stages + bottleneck reason
# 4) Output dynamics (without extra rug risk):
#    - Seed sampler (K) with resample on Refresh (session_state)
#    - Auto-Relax only on market thresholds (liq/vol24/trades/sells/imbalance) — NOT on safety guards
# 5) Safer Streamlit: no `key=` passed to st.link_button (prevents TypeError in some Streamlit versions)
# 6) Top tokens mode: optional dedupe by base token (reduces duplicates); kept deterministic

import os
import re
import csv
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter

import requests
import streamlit as st

VERSION = "0.3.5"

DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
TRADES_CSV = os.path.join(DATA_DIR, "portfolio.csv")

# Stable core: only BSC + Solana
CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap", "babyswap", "mdex", "woofi"],
    "solana": ["raydium", "orca", "meteora"],
}

# Softer ticker validation (unicode + $, _, -, ., spaces)
NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

# Majors/stables (base token filter)
MAJOR_OR_STABLE_SYMBOLS = {
    # majors
    "BTC", "WBTC", "ETH", "WETH", "BNB", "WBNB", "SOL", "WSOL",
    # stables
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FRAX", "LUSD",
    # common wrappers
    "WAVAX", "WMATIC",
}

PIPE_STAGES = [
    "total_in",
    "after_chain",
    "after_dex",
    "after_major_filter",
    "after_age",
    "after_liq",
    "after_vol24",
    "after_trades",
    "after_sells",
    "after_imbalance",
    "after_name",
    "passed",
]


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


def safe_key(x: Any, max_len: int = 40) -> str:
    s = str(x or "")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s or "x"


# -----------------------------
# HTTP with retry/backoff
# -----------------------------
def _http_get_json(url: str, params: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> Any:
    """
    Basic retry/backoff wrapper for DexScreener calls.
    - Retries on 429 / 5xx / common network errors
    - Exponential backoff (0.6s, 1.2s, 2.4s ...)
    """
    last_err = None
    backoff = 0.6

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": f"dex-scout/{VERSION}"})
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
# Filtering / Scoring
# -----------------------------
def is_name_suspicious(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip()
    if not base_sym:
        return True
    if len(base_sym) > 40:
        return True
    return not bool(NAME_OK_RE.match(base_sym))


def is_major_or_stable(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip().upper()
    if not base_sym:
        return False
    if base_sym in MAJOR_OR_STABLE_SYMBOLS:
        return True
    # also filter obvious stable-ish patterns
    if base_sym.endswith("USD") or base_sym in {"USD", "USDE", "USDB", "USDD"}:
        return True
    return False


def parse_age_minutes(pair_created_at: Any) -> Optional[int]:
    """
    DexScreener often provides pairCreatedAt in ms epoch.
    Return age minutes, or None if cannot parse.
    """
    if pair_created_at is None or pair_created_at == "":
        return None
    try:
        created_ms = int(pair_created_at)
        now_ms = int(time.time() * 1000)
        age_min = int((now_ms - created_ms) / 60000)
        if age_min < 0:
            return None
        return age_min
    except Exception:
        return None


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
    s += min(liq / 1000.0, 400.0)
    s += min(vol24 / 10000.0, 300.0)
    s += min(vol5 / 2000.0, 200.0)
    s += min(trades5 * 2.0, 120.0)
    s += max(min(pc1h, 80.0), -80.0) * 0.3
    s += max(min(pc5, 30.0), -30.0) * 0.2
    return round(s, 2)


def action_badge(action: str) -> str:
    """
    FIX: "NO ENTRY" must not be green. It is RED-ish.
    """
    a = (action or "").upper().strip()

    if "ENTRY" in a or a.startswith("BUY"):
        color = "#1f9d55"      # green
        bg = "rgba(31,157,85,0.14)"
        border = color
    elif "WATCH" in a or "WAIT" in a:
        color = "#d69e2e"      # yellow
        bg = "rgba(214,158,46,0.14)"
        border = color
    else:
        # NO ENTRY / RISK / etc
        color = "#e53e3e"      # red
        bg = "rgba(229,62,62,0.10)"
        border = color

    return f"""
    <span style="
      display:inline-block;
      padding:6px 12px;
      border-radius:999px;
      border:1px solid {border};
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

    trend_tag = f"Trend(1h): {fmt_pct(pc1h)}"
    micro_tag = f"Micro(5m): {fmt_pct(pc5)}"
    vol_tag = f"Vol(5m): {fmt_usd(vol5)}"
    tags = [liq_tag, flow_tag, trend_tag, micro_tag, vol_tag]

    # conservative decision rules (don’t “force buys”)
    decision = "NO ENTRY"
    if liq >= 25_000 and vol24 >= 10_000 and trades5 >= 8 and sells5 >= 2:
        if pc1h > 3 and vol5 > 1_000 and pc5 >= -2:
            decision = "WATCH / WAIT"
            # scalp-entry only if flow supports it
            if trades5 >= 25 and sells5 >= 6 and vol5 >= 5_000 and pc5 >= -1:
                decision = "ENTRY (scalp)"
        else:
            decision = "NO ENTRY"

    return decision, tags


def _init_stats(n: int) -> Counter:
    return Counter({k: 0 for k in PIPE_STAGES}) | Counter({"total_in": n})


def filter_pairs_with_debug(
    pairs: List[Dict[str, Any]],
    chain: str,
    any_dex: bool,
    allowed_dexes: Set[str],
    min_liq: float,
    min_vol24: float,
    min_trades_m5: int,
    min_sells_m5: int,
    max_buy_sell_imbalance: int,
    block_suspicious_names: bool,
    block_majors: bool,
    min_age_min: int,
    max_age_min: int,
) -> Tuple[List[Dict[str, Any]], Counter, Counter, str]:
    stats = _init_stats(len(pairs))
    reasons = Counter()

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

        # majors/stables filter (base only)
        if block_majors and is_major_or_stable(p):
            reasons["major_or_stable"] += 1
            continue
        stats["after_major_filter"] += 1

        # age window (anti-rug + still early)
        age_minutes = parse_age_minutes(safe_get(p, "pairCreatedAt", default=None))
        if age_minutes is not None:
            if age_minutes < int(min_age_min):
                reasons["age_too_fresh"] += 1
                continue
            if age_minutes > int(max_age_min):
                reasons["age_too_old"] += 1
                continue
        else:
            # If API didn't provide creation time — do not kill output.
            reasons["age_unknown"] += 1
        stats["after_age"] += 1

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

        if buys > 0 and sells > 0:
            imbalance = max(buys, sells) / max(1, min(buys, sells))
            if imbalance > int(max_buy_sell_imbalance):
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

    stats["passed"] = len(out)

    # bottleneck reason (excluding chain mismatch noise)
    bottleneck = "none"
    ranked_reasons = [(k, v) for k, v in reasons.items() if k != "chain_mismatch"]
    if ranked_reasons:
        bottleneck = max(ranked_reasons, key=lambda x: x[1])[0]

    return out, stats, reasons, bottleneck


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


def log_swap_intent(p: Dict[str, Any], score: float, action: str, tags: List[str], swap_url: str) -> str:
    rows = load_portfolio()

    chain = (p.get("chainId") or "").lower()
    dex = p.get("dexId") or ""
    base_sym = safe_get(p, "baseToken", "symbol", default="") or ""
    quote_sym = safe_get(p, "quoteToken", "symbol", default="") or ""
    base_addr = safe_get(p, "baseToken", "address", default="") or ""
    pair_addr = p.get("pairAddress", "") or ""
    url = p.get("url", "") or ""
    price = safe_get(p, "priceUsd", default="") or ""

    # avoid duplicates (pair-level uniqueness preferred)
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
# Presets (more dynamic, still safe)
# -----------------------------
PRESETS = {
    "Scalp Hot (strict)": {
        "top_n": 12,
        "min_liq": 25000,
        "min_vol24": 50000,
        "min_trades_m5": 20,
        "min_sells_m5": 6,
        "max_imbalance": 6,
        "block_majors": True,
        "block_suspicious_names": True,
        "min_age_min": 20,
        "max_age_min": 14400,
        "seed_k": 12,
        "seeds": "new, launch, fairlaunch, stealth, airdrop, v2, v3, inu, pepe, ai, agent, bot, pump, points, claim",
    },
    "Balanced (default)": {
        "top_n": 15,
        "min_liq": 8000,
        "min_vol24": 20000,
        "min_trades_m5": 8,
        "min_sells_m5": 2,
        "max_imbalance": 10,
        "block_majors": True,
        "block_suspicious_names": True,
        "min_age_min": 20,
        "max_age_min": 14400,
        "seed_k": 14,
        "seeds": "WBNB, USDT, BNB, meme, ai, ai agent, gaming, cat, dog, launch, pump, sol, eth, usdc, pepe, trump, new, launch, trend, community, fairlaunch, stealth, airdrop, v2, v3, inu, pepe, ai, agent, bot, pump",
    },
    "Wide Net (explore)": {
        "top_n": 25,
        "min_liq": 3000,
        "min_vol24": 10000,
        "min_trades_m5": 3,
        "min_sells_m5": 1,
        "max_imbalance": 15,
        "block_majors": True,
        "block_suspicious_names": True,
        "min_age_min": 20,
        "max_age_min": 14400,
        "seed_k": 16,
        "seeds": "new, launch, trend, community, pump, fairlaunch, stealth, airdrop, claim, points, v2, v3, inu, pepe, ai, agent, bot",
    },
    "Ultra Early (safer)": {
        "top_n": 20,
        "min_liq": 2000,
        "min_vol24": 5000,
        "min_trades_m5": 0,
        "min_sells_m5": 0,
        "max_imbalance": 20,
        "block_majors": True,
        "block_suspicious_names": True,
        "min_age_min": 15,
        "max_age_min": 4320,   # 3 days
        "seed_k": 16,
        "seeds": "new, fairlaunch, stealth, airdrop, claim, points, launch, v2, v3, inu, pepe, ai, agent, bot, pump",
    },
}


# -----------------------------
# Seed sampling / session state
# -----------------------------
def sample_seeds(seeds: List[str], k: int) -> List[str]:
    seeds = [s.strip() for s in seeds if s.strip()]
    if not seeds:
        return []
    k = max(1, min(int(k), len(seeds)))
    return random.sample(seeds, k)


def get_sampled_seeds(all_seeds: List[str], k: int, force_resample: bool) -> List[str]:
    """
    Keeps sampled seeds stable until Refresh (or first load).
    """
    ss_key = "sampled_seeds"
    base_key = "sampled_seeds_base"

    base_sig = "|".join(sorted(set([s.strip().lower() for s in all_seeds if s.strip()])))
    prev_sig = st.session_state.get(base_key)

    if force_resample or (ss_key not in st.session_state) or (prev_sig != base_sig):
        st.session_state[ss_key] = sample_seeds(all_seeds, k)
        st.session_state[base_key] = base_sig

    return st.session_state[ss_key]


# -----------------------------
# Rendering helpers
# -----------------------------
def render_tags(tags: List[str]):
    for t in tags:
        st.write(f"- {t}")


def build_swap_url(chain: str, base_addr: str) -> str:
    # BSC only for now (stable core)
    if chain.lower() == "bsc" and base_addr:
        return f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"
    return ""


def dedupe_pairs(pairs: List[Dict[str, Any]], mode_top_tokens: bool) -> List[Dict[str, Any]]:
    """
    If mode_top_tokens: dedupe by base token address (per chain), keeps the "best" pair by liq+vol24.
    Else: dedupe by pairAddress/url.
    """
    if not pairs:
        return []

    if not mode_top_tokens:
        uniq = {}
        for p in pairs:
            pa = p.get("pairAddress") or p.get("url")
            if not pa:
                continue
            uniq[str(pa)] = p
        return list(uniq.values())

    # top tokens mode: group by base token
    buckets: Dict[str, Dict[str, Any]] = {}
    for p in pairs:
        chain = (p.get("chainId") or "").lower()
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").lower().strip()
        base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").lower().strip()

        key = f"{chain}:{base_addr}" if base_addr else f"{chain}:sym:{base_sym}"
        if key.strip() == ":":
            continue

        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
        score = liq * 1.0 + vol24 * 0.2

        if key not in buckets:
            buckets[key] = {"p": p, "k": score}
        else:
            if score > buckets[key]["k"]:
                buckets[key] = {"p": p, "k": score}

    return [v["p"] for v in buckets.values()]


# -----------------------------
# Auto-Relax (market thresholds only)
# -----------------------------
def auto_relax_search(
    pairs: List[Dict[str, Any]],
    base_params: Dict[str, Any],
    allowed_dexes: Set[str],
    top_n: int,
) -> Tuple[List[Tuple[float, str, List[str], Dict[str, Any]]], Optional[str]]:
    """
    Returns ranked list and a note describing the relaxation applied.
    Safety guards are not relaxed.
    """
    # keep safety stable
    chain = base_params["chain"]
    any_dex = base_params["any_dex"]
    block_majors = base_params["block_majors"]
    block_suspicious_names = base_params["block_suspicious_names"]
    min_age_min = base_params["min_age_min"]
    max_age_min = base_params["max_age_min"]

    # initial + relax levels (ONLY market thresholds)
    relax_levels = [
        {"label": None, "k": 1.0, "imb_add": 0},
        {"label": "Auto-Relax L1", "k": 0.75, "imb_add": 3},
        {"label": "Auto-Relax L2", "k": 0.55, "imb_add": 6},
        {"label": "Auto-Relax L3", "k": 0.40, "imb_add": 10},
    ]

    for level in relax_levels:
        k = level["k"]
        r_min_liq = clamp_int(float(base_params["min_liq"]) * k, 0)
        r_min_vol24 = clamp_int(float(base_params["min_vol24"]) * k, 0)
        r_min_trades = clamp_int(int(base_params["min_trades_m5"]) * k, 0)
        r_min_sells = clamp_int(int(base_params["min_sells_m5"]) * k, 0)
        r_imb = clamp_int(int(base_params["max_imbalance"]) + level["imb_add"], 1, 50)

        filtered, _, _, _ = filter_pairs_with_debug(
            pairs=pairs,
            chain=chain,
            any_dex=any_dex,
            allowed_dexes=allowed_dexes,
            min_liq=r_min_liq,
            min_vol24=r_min_vol24,
            min_trades_m5=r_min_trades,
            min_sells_m5=r_min_sells,
            max_buy_sell_imbalance=r_imb,
            block_suspicious_names=block_suspicious_names,
            block_majors=block_majors,
            min_age_min=min_age_min,
            max_age_min=max_age_min,
        )

        ranked = []
        for p in filtered:
            s = score_pair(p)
            decision, tags = build_trade_hint(p)
            ranked.append((s, decision, tags, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:top_n]

        if ranked:
            if level["label"] is None:
                return ranked, None
            note = (
                f"{level['label']} applied: "
                f"min_liq={r_min_liq}, min_vol24={r_min_vol24}, "
                f"min_trades_m5={r_min_trades}, min_sells_m5={r_min_sells}, "
                f"max_imbalance={r_imb}"
            )
            return ranked, note

    return [], None


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title="DEX Scout", layout="wide")

    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")

        page = st.radio("Page", ["Scout", "Portfolio"], index=0)

        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1)
        preset = PRESETS[preset_name]

        st.divider()

        # core chains only
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
        max_imbalance = st.slider("Max buy/sell imbalance", 1, 30, int(preset["max_imbalance"]), step=1)

        st.divider()
        st.caption("Safety guards")
        block_majors = st.checkbox("Filter majors/stables (base)", value=bool(preset["block_majors"]))
        block_suspicious_names = st.checkbox("Filter suspicious tickers", value=bool(preset["block_suspicious_names"]))

        st.caption("Pair age window (anti-rug + still early)")
        min_age_min = st.number_input("Min age (minutes)", min_value=0, value=int(preset["min_age_min"]), step=5)
        max_age_min = st.number_input("Max age (minutes)", min_value=0, value=int(preset["max_age_min"]), step=60)

        st.divider()
        st.caption("Output dynamics")
        top_tokens_mode = st.checkbox("Top tokens mode (dedupe by base token)", value=True)
        seed_k = st.slider("Seed sampler K", 3, 20, int(preset["seed_k"]), step=1)

        st.divider()
        st.caption("Queries (comma-separated)")
        seeds_text = st.text_area("Search seeds", value=str(preset["seeds"]), height=120)

        refresh = st.button("Refresh now")
        if refresh:
            st.cache_data.clear()
            # force resample by resetting session key
            st.session_state.pop("sampled_seeds", None)
            st.session_state.pop("sampled_seeds_base", None)

    # -------------------- SCOUT PAGE --------------------
    if page == "Scout":
        st.title("DEX Scout — early candidates (DEXScreener API)")
        st.caption("Фокус: дрібні/ранні монети. Majors/stables відсікаються. Stable core: BSC + Solana.")

        all_seeds = [s.strip() for s in seeds_text.split(",") if s.strip()]
        if not all_seeds:
            st.info("Додай хоча б 1 seed у sidebar.")
            return

        sampled = get_sampled_seeds(all_seeds, seed_k, force_resample=refresh)
        st.caption(f"Seeds sampled ({len(sampled)}/{len(all_seeds)}): {', '.join(sampled)}")

        allowed = set([d.lower() for d in selected_dexes]) if selected_dexes else set()

        all_pairs: List[Dict[str, Any]] = []
        query_failures = 0

        for q in sampled:
            try:
                all_pairs.extend(fetch_latest_pairs_for_query(q))
                time.sleep(0.10)
            except Exception as e:
                query_failures += 1
                st.warning(f"Query failed: {q} — {e}")

        # Deduplicate
        pairs = dedupe_pairs(all_pairs, mode_top_tokens=bool(top_tokens_mode))

        if query_failures and not pairs:
            st.error("All sampled queries failed or returned no data. Try Refresh, reduce seeds, or wait a bit.")
            return

        base_params = {
            "chain": chain,
            "any_dex": any_dex,
            "min_liq": float(min_liq),
            "min_vol24": float(min_vol24),
            "min_trades_m5": int(min_trades_m5),
            "min_sells_m5": int(min_sells_m5),
            "max_imbalance": int(max_imbalance),
            "block_suspicious_names": bool(block_suspicious_names),
            "block_majors": bool(block_majors),
            "min_age_min": int(min_age_min),
            "max_age_min": int(max_age_min),
        }

        # First pass: also keep debug stats
        filtered, fstats, freasons, bottleneck = filter_pairs_with_debug(
            pairs=pairs,
            chain=chain,
            any_dex=any_dex,
            allowed_dexes=allowed,
            min_liq=base_params["min_liq"],
            min_vol24=base_params["min_vol24"],
            min_trades_m5=base_params["min_trades_m5"],
            min_sells_m5=base_params["min_sells_m5"],
            max_buy_sell_imbalance=base_params["max_imbalance"],
            block_suspicious_names=base_params["block_suspicious_names"],
            block_majors=base_params["block_majors"],
            min_age_min=base_params["min_age_min"],
            max_age_min=base_params["max_age_min"],
        )

        ranked = []
        for p in filtered:
            s = score_pair(p)
            decision, tags = build_trade_hint(p)
            ranked.append((s, decision, tags, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:top_n]

        # If nothing — try auto-relax (market thresholds only)
        relax_note = None
        if not ranked:
            ranked, relax_note = auto_relax_search(
                pairs=pairs,
                base_params=base_params,
                allowed_dexes=allowed,
                top_n=int(top_n),
            )

        st.metric("Passed filters", len(ranked))

        with st.expander("Why 0 results? (Filter Debug)", expanded=False):
            st.write("**Fetched:**", len(all_pairs), " • **Deduped:**", len(pairs))
            st.write("**Counts (pipeline):**")
            # ensure all keys appear
            st.json({k: int(fstats.get(k, 0)) for k in PIPE_STAGES})
            st.write("**Top reject reasons:**")
            top = freasons.most_common(12)
            st.write({k: int(v) for k, v in top} if top else "No rejects counted (unexpected).")
            st.write("**Bottleneck:**", bottleneck)

        if relax_note:
            st.info(relax_note)

        if not ranked:
            st.warning("0 results. Hit Refresh (resample seeds) або трохи ослаб vol24 / trades / sells.")
            return

        # Render cards
        for idx, (s, decision, tags, p) in enumerate(ranked, start=1):
            base = safe_get(p, "baseToken", "symbol", default="???") or "???"
            quote = safe_get(p, "quoteToken", "symbol", default="???") or "???"
            dex = p.get("dexId", "?")
            url = p.get("url", "")
            chain_id = (p.get("chainId") or "").lower()
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

            swap_url = build_swap_url(chain_id, base_addr)

            st.markdown("---")
            left, mid, right = st.columns([3, 2, 2])

            with left:
                st.markdown(f"### {base}/{quote}")
                st.caption(f"{chain_id} • {dex}")

                if url:
                    # IMPORTANT: do NOT pass key= to link_button (prevents TypeError on some versions)
                    st.link_button("Open DexScreener", url, use_container_width=True)

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
                render_tags(tags)

                # Log button must have unique key
                k_log = f"log_{safe_key(chain_id)}_{safe_key(base_addr)}_{safe_key(pair_addr)}_{idx}"
                if st.button("Log → Portfolio (I swapped)", key=k_log, use_container_width=True):
                    # we log swap_url even if empty (e.g. solana), but it’s fine; it’s a "log intent"
                    res = log_swap_intent(p, s, decision, tags, swap_url)
                    if res == "OK":
                        st.success("Logged to Portfolio (entry snapshot saved).")
                    else:
                        st.info("Already in Portfolio (active).")

                if swap_url:
                    st.link_button("Open Swap (PancakeSwap)", swap_url, use_container_width=True)
                else:
                    st.caption("Swap disabled for this chain (BSC-only).")

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
                if base_addr:
                    st.code(base_addr, language="text")
                if r.get("dexscreener_url"):
                    st.link_button("DexScreener", r["dexscreener_url"], use_container_width=True)
                if r.get("swap_url"):
                    st.link_button("Swap", r["swap_url"], use_container_width=True)

            with c2:
                st.write(f"**Entry:** ${entry_price_str}")
                st.write(f"**Now:** ${cur_price:.8f}" if cur_price else "**Now:** n/a")
                st.write(f"**PnL:** {pnl:+.2f}%" if entry_price and cur_price else "**PnL:** n/a")
                st.write(f"**Liq:** {fmt_usd(liq)}" if liq else "**Liq:** n/a")

            with c3:
                st.write(f"**Δ m5:** {fmt_pct(pc5)}")
                st.write(f"**Δ h1:** {fmt_pct(pc1h)}")
                st.write(f"**Reco:** `{reco}`")

                note_key = f"note_{idx}_{safe_key(base_addr)}"
                note_val = st.text_input("Note", value=r.get("note", ""), key=note_key)

            with c4:
                close_key = f"close_{idx}_{safe_key(base_addr)}"
                delete_key = f"delete_{idx}_{safe_key(base_addr)}"

                close = st.checkbox("Close (archive)", value=False, key=close_key)
                delete = st.checkbox("Delete row", value=False, key=delete_key)

                if st.button("Apply", key=f"apply_{idx}_{safe_key(base_addr)}", use_container_width=True):
                    all_rows = load_portfolio()
                    # update by base token address (simple & stable)
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
