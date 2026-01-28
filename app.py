# app.py — DEX Scout v0.3.6
# Core goals (stable): BSC + Solana only
# Fixes:
# 1) Swap routing:
#    - BSC -> PancakeSwap
#    - Solana -> Jupiter (USDC -> token mint)
# 2) Action color fix:
#    - NO ENTRY is RED (no substring bug with "ENTRY")
# Stability fixes:
# - Avoid StreamlitDuplicateElementKey by using unique keys for buttons
# - Avoid st.link_button(key=...) TypeError by not passing "key" to link_button at all
# - Use HTML-styled links (safe, deterministic) for external links

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

VERSION = "0.3.6"

DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
TRADES_CSV = os.path.join(DATA_DIR, "portfolio.csv")

# Stable core: only these two for now (as requested)
CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap", "babyswap", "mdex", "woofi"],
    "solana": ["raydium", "orca", "meteora", "pumpswap"],
}

# allow unicode letters/digits + common ticker chars ($ _ - .) + spaces
NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

# Majors / stables to exclude (base token symbol)
MAJORS = {
    "BTC", "WBTC", "ETH", "WETH", "BNB", "WBNB", "SOL", "WSOL",
    "MATIC", "AVAX", "OP", "ARB", "LINK", "DOT", "ATOM", "TON",
}
STABLES = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDD", "FRAX", "USDP",
    "LUSD", "SUSD", "EURC",
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
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_usd(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "n/a"


def fmt_price(x: Any) -> str:
    try:
        v = float(x)
        if v >= 1:
            return f"${v:,.4f}"
        if v >= 0.01:
            return f"${v:,.6f}"
        return f"${v:.10f}"
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


def safe_key(x: str, max_len: int = 60) -> str:
    # for Streamlit widget keys
    s = str(x or "")
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def html_link_button(label: str, url: str) -> str:
    # Stable alternative to st.link_button (no key issues)
    if not url:
        return ""
    return f"""
    <a href="{url}" target="_blank" style="
      display:inline-block;
      width:100%;
      text-align:center;
      padding:10px 12px;
      border:1px solid rgba(0,0,0,0.15);
      border-radius:10px;
      text-decoration:none;
      font-weight:600;
      color:inherit;
      background:rgba(0,0,0,0.02);
      margin:4px 0;
    ">{label}</a>
    """


def parse_seeds(seeds_raw: str) -> List[str]:
    items = [x.strip() for x in (seeds_raw or "").split(",")]
    items = [x for x in items if x]
    # dedupe preserving order
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


# -----------------------------
# HTTP with retry/backoff
# -----------------------------
def _http_get_json(url: str, params: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> Any:
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
# Swap routing (core)
# -----------------------------
def build_swap_url(chain: str, base_addr: str) -> str:
    if not base_addr:
        return ""
    chain = (chain or "").lower()

    if chain == "bsc":
        return f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"

    if chain == "solana":
        # Jupiter: USDC -> token mint
        # (works well as a default; you can change to SOL later if you prefer)
        return f"https://jup.ag/swap/USDC-{base_addr}"

    return ""


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


def is_major_or_stable_base(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip().upper()
    if not base_sym:
        return True
    return (base_sym in MAJORS) or (base_sym in STABLES)


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
    s += min(liq / 1000.0, 420.0)
    s += min(vol24 / 10000.0, 320.0)
    s += min(vol5 / 2000.0, 220.0)
    s += min(trades5 * 2.0, 140.0)
    s += max(min(pc1h, 80.0), -80.0) * 0.25
    s += max(min(pc5, 30.0), -30.0) * 0.20
    return round(s, 2)


def action_badge(action: str) -> str:
    # FIXED: NO ENTRY must be red (no substring checks)
    a = (action or "").upper().strip()

    if a.startswith("ENTRY") or a.startswith("BUY"):
        color = "#1f9d55"      # green
        bg = "rgba(31,157,85,0.14)"
    elif a.startswith("WATCH") or a.startswith("WAIT"):
        color = "#d69e2e"      # yellow
        bg = "rgba(214,158,46,0.14)"
    else:
        color = "#e53e3e"      # red
        bg = "rgba(229,62,62,0.10)"

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
      letter-spacing:0.2px;
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

    # Conservative scalp entry idea: still "early-ish", not pure degen.
    # (You can tighten later in presets; this is used after filtering already.)
    if liq >= 20_000 and vol24 >= 20_000 and trades5 >= 10 and sells5 >= 2:
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
    filter_majors_stables: bool,
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

        # majors / stables filter (base token)
        if filter_majors_stables and is_major_or_stable_base(p):
            reasons["major_or_stable"] += 1
            continue
        stats["after_major_filter"] += 1

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

    # avoid duplicates by pair
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
# Presets (more dynamic but still safe-ish)
# -----------------------------
PRESETS = {
    "Ultra Early (safer)": {
        "top_n": 20,
        "min_liq": 2000,
        "min_vol24": 5000,
        "min_trades_m5": 4,
        "min_sells_m5": 1,
        "max_imbalance": 20,
        "filter_majors_stables": True,
        "block_suspicious_names": True,
        "seed_k": 12,
        "seeds": "new, launch, fairlaunch, stealth, airdrop, points, claim, v2, v3, inu, pepe, ai, agent, bot",
    },
    "Balanced (default)": {
        "top_n": 15,
        "min_liq": 15000,
        "min_vol24": 60000,
        "min_trades_m5": 10,
        "min_sells_m5": 2,
        "max_imbalance": 12,
        "filter_majors_stables": True,
        "block_suspicious_names": True,
        "seed_k": 10,
        "seeds": "meme, ai, ai agent, launch, pump, new, community, bot, inu, pepe",
    },
    "Scalp Hot (strict)": {
        "top_n": 12,
        "min_liq": 30000,
        "min_vol24": 120000,
        "min_trades_m5": 25,
        "min_sells_m5": 6,
        "max_imbalance": 8,
        "filter_majors_stables": True,
        "block_suspicious_names": True,
        "seed_k": 8,
        "seeds": "launch, pump, trending, ai, bot, meme, new, community, inu, pepe",
    },
    "Wide Net (explore)": {
        "top_n": 25,
        "min_liq": 3000,
        "min_vol24": 10000,
        "min_trades_m5": 4,
        "min_sells_m5": 1,
        "max_imbalance": 20,
        "filter_majors_stables": True,
        "block_suspicious_names": False,
        "seed_k": 14,
        "seeds": "new, launch, trend, community, pump, fairlaunch, presale, stealth, airdrop, claim, points, v2, v3, inu, pepe, ai, agent, bot",
    },
}


def main():
    st.set_page_config(page_title="DEX Scout", layout="wide")

    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")

        page = st.radio("Page", ["Scout", "Portfolio"], index=0)

        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1)
        preset = PRESETS[preset_name]

        st.divider()

        chain = st.selectbox(
            "Chain",
            options=list(CHAIN_DEX_PRESETS.keys()),
            index=0,
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
        min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=int(preset["min_liq"]), step=1000)
        min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=int(preset["min_vol24"]), step=5000)

        st.divider()
        st.caption("Real trading filters")
        min_trades_m5 = st.slider("Min trades (5m)", 0, 200, int(preset["min_trades_m5"]), step=1)
        min_sells_m5 = st.slider("Min sells (5m)", 0, 80, int(preset["min_sells_m5"]), step=1)
        max_buy_sell_imbalance = st.slider("Max buy/sell imbalance", 1, 30, int(preset["max_imbalance"]), step=1)

        st.divider()
        st.caption("Safety guards")
        filter_majors_stables = st.checkbox("Filter majors/stables (base)", value=bool(preset["filter_majors_stables"]))
        block_suspicious_names = st.checkbox("Filter suspicious tickers", value=bool(preset["block_suspicious_names"]))

        st.divider()
        st.caption("Output dynamics")
        seed_k = st.slider("Seed sampler K", 3, 20, int(preset["seed_k"]), step=1)

        st.divider()
        st.caption("Queries (comma-separated)")
        seeds_raw = st.text_area("Search seeds", value=str(preset["seeds"]), height=110)

        refresh = st.button("Refresh now")
        if refresh:
            st.cache_data.clear()
            # also rotate seed sampling without needing extra input
            st.session_state["seed_shuffle_nonce"] = random.randint(1, 10**9)

    # -------------------- SCOUT --------------------
    if page == "Scout":
        st.title("DEX Scout — early candidates (DEXScreener API)")
        st.caption("Фокус: дрібні/ранні монети. Majors/stables відсікаються. Stable core: BSC + Solana.")

        seeds = parse_seeds(seeds_raw)
        if not seeds:
            st.info("Додай хоча б 1 seed у sidebar.")
            return

        # Seed sampling for dynamic output (still deterministic enough per refresh)
        nonce = st.session_state.get("seed_shuffle_nonce", 0)
        rnd = random.Random(str(nonce) + "|" + preset_name + "|" + chain)
        sampled = seeds[:]
        rnd.shuffle(sampled)
        sampled = sampled[: min(seed_k, len(sampled))]

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

        # Deduplicate
        uniq = {}
        for p in all_pairs:
            pa = p.get("pairAddress") or p.get("url")
            if not pa:
                continue
            uniq[pa] = p
        pairs = list(uniq.values())

        if query_failures and not pairs:
            st.error("Усі запити впали або повернули 0. Натисни Refresh або зменш seeds.")
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
            filter_majors_stables=bool(filter_majors_stables),
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
            st.write({k: v for k, v in top} if top else "No rejects counted (unexpected).")

        if not ranked:
            st.info("0 results. Спробуй: трохи знизити Min Liquidity/Vol24 або натисни Refresh (пересемплити seeds).")
            return

        # Render cards
        for i, (s, decision, tags, p) in enumerate(ranked, start=1):
            base = safe_get(p, "baseToken", "symbol", default="???") or "???"
            quote = safe_get(p, "quoteToken", "symbol", default="???") or "???"
            dex = p.get("dexId", "?")
            chain_id = (p.get("chainId", "") or "").lower()
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

            swap_url = build_swap_url(chain_id, base_addr)

            st.markdown("---")
            left, mid, right = st.columns([3, 2, 2])

            with left:
                st.markdown(f"### {base}/{quote}")
                st.caption(f"{chain_id} • {dex}")

                if url:
                    st.markdown(html_link_button(f"Open DexScreener · {base}/{quote}", url), unsafe_allow_html=True)

                if base_addr:
                    st.caption("Token contract (baseToken.address)")
                    st.code(base_addr, language="text")
                if pair_addr:
                    st.caption("Pair / pool address")
                    st.code(pair_addr, language="text")

            with mid:
                st.write(f"**Price:** {fmt_price(price)}" if price else "**Price:** n/a")
                st.write(f"**Liq:** {fmt_usd(liq)}")
                st.write(f"**Vol24:** {fmt_usd(vol24)}")
                st.write(f"**Vol m5:** {fmt_usd(vol5)}")
                st.write(f"**Δ m5:** {fmt_pct(pc5)}")
                st.write(f"**Δ h1:** {fmt_pct(pc1h)}")
                st.write(f"**Buys/Sells (m5):** {buys}/{sells}")
                st.write(f"**Score:** {s}")

            with right:
                st.markdown("**Action:**")
                st.markdown(action_badge(decision), unsafe_allow_html=True)

                st.markdown("**Tags:**")
                for t in tags:
                    st.write(f"• {t}")

                # unique keys (no duplicates)
                uniq_id = f"{chain_id}_{safe_key(pair_addr or base_addr)}_{i}"

                if swap_url:
                    key_log = f"log_{uniq_id}"
                    if st.button("Log → Portfolio (I swapped)", key=key_log, use_container_width=True):
                        res = log_swap_intent(p, s, decision, tags, swap_url)
                        if res == "OK":
                            st.success("Logged to Portfolio (entry snapshot saved).")
                        else:
                            st.info("Already in Portfolio (active).")

                    st.markdown(html_link_button(f"Open Swap · {base}/{quote}", swap_url), unsafe_allow_html=True)
                else:
                    st.caption("Swap unavailable (missing token address).")

    # -------------------- PORTFOLIO --------------------
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
                    st.markdown(html_link_button("DexScreener", r["dexscreener_url"]), unsafe_allow_html=True)

                if r.get("swap_url"):
                    st.markdown(html_link_button("Swap", r["swap_url"]), unsafe_allow_html=True)

            with c2:
                st.write(f"**Entry:** {fmt_price(entry_price_str)}")
                st.write(f"**Now:** {fmt_price(cur_price)}" if cur_price else "**Now:** n/a")
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

                    # Update by pair address if possible (safer)
                    target_pair = (r.get("pair_address") or "").lower()
                    target_base = (r.get("base_token_address") or "").lower()

                    kept = []
                    for rr in all_rows:
                        is_target = False
                        if rr.get("active") == "1":
                            if target_pair and (rr.get("pair_address") or "").lower() == target_pair:
                                is_target = True
                            elif (rr.get("base_token_address") or "").lower() == target_base:
                                is_target = True

                        if is_target:
                            if delete:
                                continue
                            rr["note"] = note_val
                            if close:
                                rr["active"] = "0"

                        kept.append(rr)

                    save_portfolio(kept)
                    st.success("Saved.")
                    st.rerun()

            st.markdown("---")

        if closed_rows:
            with st.expander("Closed / Archived"):
                for r in closed_rows[-50:]:
                    st.write(
                        f"{r.get('ts_utc','')} — {r.get('base_symbol','')}/{r.get('quote_symbol','')} "
                        f"— entry {r.get('entry_price_usd','')} — {r.get('action','')}"
                    )


if __name__ == "__main__":
    main()
