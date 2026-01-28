# app.py — DEX Scout v0.3.1
# Changes from v0.3.0:
# - Filter Debug: pipeline counters + top reject reasons
# - Auto-Relax: if 0 results, progressively relax thresholds automatically
# - Softer name filter (unicode + $, _)
# - Seeds presets expanded (sol, eth, usdc, pepe, trump, ai agent, pump, launch)
# - Basic retry/backoff for DexScreener requests to reduce rate-limit pain
# Notes:
# - Swap URL still PancakeSwap-only (BSC); chain->swap map later.

import os
import re
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import requests
import streamlit as st

VERSION = "0.3.1"

DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
TRADES_CSV = os.path.join(DATA_DIR, "portfolio.csv")

CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap", "babyswap", "mdex", "woofi"],
    "ethereum": ["uniswap", "sushiswap"],
    "base": ["aerodrome", "uniswap"],
    "polygon": ["quickswap", "uniswap", "sushiswap"],
    "arbitrum": ["uniswap", "sushiswap", "camelot"],
    "optimism": ["uniswap", "velodrome"],
    "solana": ["raydium", "orca", "meteora"],
}

# Softer, more realistic ticker validation:
# allow unicode letters/digits + common ticker chars ($ _ - .) + spaces
NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

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


def clamp_int(x: int, lo: int = 0, hi: int = 10**9) -> int:
    try:
        return int(max(lo, min(hi, int(x))))
    except Exception:
        return lo


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
            # Unknown error: do not spin too long
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
    # common garbage patterns often seen on DEX: too many spaces or very long
    if len(base_sym) > 40:
        return True
    return not bool(NAME_OK_RE.match(base_sym))


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
    a = (action or "").upper()
    if "ENTRY" in a or a.startswith("BUY"):
        color = "#1f9d55"      # green
        bg = "rgba(31,157,85,0.15)"
    elif "WATCH" in a or "WAIT" in a:
        color = "#d69e2e"      # yellow
        bg = "rgba(214,158,46,0.15)"
    else:
        color = "#e53e3e"      # red
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

    if liq >= 250_000:
        liq_tag = "Liquidity: High"
    elif liq >= 80_000:
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

    decision = "NO ENTRY"
    if liq >= 80_000 and vol24 >= 80_000 and trades5 >= 25 and sells5 >= 6:
        if pc1h > 5 and vol5 > 5_000 and pc5 >= -2:
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

    # avoid duplicates
    for r in rows:
        if r.get("active") == "1" and r.get("base_token_address", "").lower() == base_addr.lower():
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
# Presets
# -----------------------------
PRESETS = {
    "Scalp Hot (strict)": {
        "top_n": 12,
        "min_liq": 80000,
        "min_vol24": 250000,
        "min_trades_m5": 35,
        "min_sells_m5": 10,
        "max_imbalance": 6,
        "block_suspicious_names": True,
        "seeds": "WBNB, USDT, meme, ai, ai agent, launch, trending, pump, sol, eth, usdc, pepe, trump",
    },
    "Balanced (default)": {
        "top_n": 15,
        "min_liq": 25000,
        "min_vol24": 150000,
        "min_trades_m5": 25,
        "min_sells_m5": 6,
        "max_imbalance": 8,
        "block_suspicious_names": True,
        "seeds": "WBNB, USDT, BNB, meme, ai, ai agent, gaming, cat, dog, launch, pump, sol, eth, usdc, pepe, trump",
    },
    "Wide Net (explore)": {
        "top_n": 25,
        "min_liq": 15000,
        "min_vol24": 60000,
        "min_trades_m5": 15,
        "min_sells_m5": 4,
        "max_imbalance": 10,
        "block_suspicious_names": False,
        "seeds": "WBNB, USDT, new, launch, trend, community, pump, sol, eth, usdc, pepe, trump, ai agent",
    },
}


def main():
    st.set_page_config(page_title="DEX Scout", layout="wide")

    # Sidebar navigation (mobile-friendly)
    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")

        page = st.radio("Page", ["Scout", "Portfolio"], index=0)

        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1)
        preset = PRESETS[preset_name]

        st.divider()

        chain = st.selectbox(
            "Chain",
            options=sorted(list(CHAIN_DEX_PRESETS.keys())),
            index=sorted(list(CHAIN_DEX_PRESETS.keys())).index("bsc") if "bsc" in CHAIN_DEX_PRESETS else 0,
        )

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
        min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=int(preset["min_liq"]), step=5000)
        min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=int(preset["min_vol24"]), step=25000)

        st.divider()
        st.caption("Real trading filters")
        min_trades_m5 = st.slider("Min trades (5m)", 0, 200, int(preset["min_trades_m5"]), step=1)
        min_sells_m5 = st.slider("Min sells (5m)", 0, 80, int(preset["min_sells_m5"]), step=1)
        max_buy_sell_imbalance = st.slider("Max buy/sell imbalance", 1, 20, int(preset["max_imbalance"]), step=1)

        st.divider()
        block_suspicious_names = st.checkbox("Filter suspicious names", value=bool(preset["block_suspicious_names"]))

        st.divider()
        st.caption("Queries (comma-separated)")
        seeds = st.text_area("Search seeds", value=str(preset["seeds"]), height=110)

        refresh = st.button("Refresh now")
        if refresh:
            st.cache_data.clear()

    # -------------------- SCOUT PAGE --------------------
    if page == "Scout":
        st.title("DEX Scout — actionable candidates (DEXScreener API)")
        st.caption("Кнопки, теги без емодзі, кольори тільки по дії, лог в портфель. Mobile-friendly.")

        queries = [q.strip() for q in seeds.split(",") if q.strip()]
        if not queries:
            st.info("Add at least 1 seed query in the sidebar.")
            return

        all_pairs: List[Dict[str, Any]] = []
        query_failures = 0

        for q in queries:
            try:
                all_pairs.extend(fetch_latest_pairs_for_query(q))
                time.sleep(0.12)  # gentle pacing to reduce rate-limits
            except Exception as e:
                query_failures += 1
                st.warning(f"Query failed: {q} — {e}")

        # Deduplicate
        uniq = {}
        for p in all_pairs:
            pa = p.get("pairAddress") or p.get("url")
            if not pa:
                continue
            uniq[pa] = p
        pairs = list(uniq.values())

        if query_failures and not pairs:
            st.error("All queries failed or returned no data. Try Refresh, reduce seeds, or wait a bit.")
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
        )

        ranked = []
        for p in filtered:
            s = score_pair(p)
            decision, tags = build_trade_hint(p)
            ranked.append((s, decision, tags, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
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

        # Auto-relax if no results
        if not ranked:
            st.info("No pairs passed filters. Trying Auto-Relax…")

            relax_steps = [
                {"k": 0.70, "imb_add": 2},
                {"k": 0.55, "imb_add": 4},
                {"k": 0.40, "imb_add": 6},
            ]

            for i, step in enumerate(relax_steps, start=1):
                r_min_liq = clamp_int(float(min_liq) * step["k"], 0)
                r_min_vol24 = clamp_int(float(min_vol24) * step["k"], 0)
                r_min_trades = clamp_int(int(min_trades_m5) * step["k"], 0)
                r_min_sells = clamp_int(int(min_sells_m5) * step["k"], 0)
                r_imb = clamp_int(int(max_buy_sell_imbalance) + step["imb_add"], 1, 50)

                f2, _, _ = filter_pairs_with_debug(
                    pairs=pairs,
                    chain=chain,
                    any_dex=any_dex,
                    allowed_dexes=allowed,
                    min_liq=r_min_liq,
                    min_vol24=r_min_vol24,
                    min_trades_m5=r_min_trades,
                    min_sells_m5=r_min_sells,
                    max_buy_sell_imbalance=r_imb,
                    block_suspicious_names=bool(block_suspicious_names),
                )

                ranked2 = []
                for p in f2:
                    s = score_pair(p)
                    decision, tags = build_trade_hint(p)
                    ranked2.append((s, decision, tags, p))
                ranked2.sort(key=lambda x: x[0], reverse=True)
                ranked2 = ranked2[:top_n]

                if ranked2:
                    st.success(
                        f"Auto-Relax level {i} applied: "
                        f"min_liq={r_min_liq}, min_vol24={r_min_vol24}, "
                        f"min_trades_m5={r_min_trades}, min_sells_m5={r_min_sells}, "
                        f"max_imbalance={r_imb}"
                    )
                    ranked = ranked2
                    break

            if not ranked:
                st.info("Auto-Relax still found 0. Change seeds or disable some filters (especially sells/imbalance/name).")
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

            # Swap URL: BSC PancakeSwap only for now
            swap_url = ""
            if base_addr and (p.get("chainId") or "").lower() == "bsc":
                swap_url = f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"

            st.markdown("---")
            left, mid, right = st.columns([3, 2, 2])

            with left:
                st.markdown(f"### {base}/{quote}")
                st.caption(f"{p.get('chainId','')} • {dex}")
                if url:
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
                for t in tags:
                    st.caption(f"- {t}")

                if base_addr:
                    if swap_url:
                        key_log = f"log_{base_addr}"
                        if st.button("Log → Portfolio (I swapped)", key=key_log, use_container_width=True):
                            res = log_swap_intent(p, s, decision, tags, swap_url)
                            if res == "OK":
                                st.success("Logged to Portfolio (entry snapshot saved).")
                            else:
                                st.info("Already in Portfolio (active).")

                        st.link_button("Open Swap (PancakeSwap)", swap_url, use_container_width=True)
                    else:
                        st.caption("Swap button disabled for this chain (BSC-only in v0.3.1).")
                else:
                    st.caption("Swap unavailable (missing base token address).")

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
