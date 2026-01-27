# app.py
import re
import time
from typing import Dict, Any, List, Tuple

import requests
import streamlit as st

DEX_BASE = "https://api.dexscreener.com"

# ---- UI defaults / presets ----
CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap"],
    "ethereum": ["uniswap", "sushiswap", "pancakeswap"],
    "base": ["uniswap", "aerodrome"],
    "polygon": ["quickswap", "uniswap", "sushiswap"],
    "arbitrum": ["uniswap", "sushiswap"],
    "optimism": ["uniswap", "velodrome"],
    "solana": ["raydium", "orca", "meteora"],
}

# “ієрогліфи/сміття”: лишаємо латиницю/цифри/частину символів
NAME_OK_RE = re.compile(r"^[A-Za-z0-9\-\._\s]{2,30}$")


def safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


@st.cache_data(ttl=25, show_spinner=False)
def fetch_latest_pairs_for_query(q: str) -> List[Dict[str, Any]]:
    # /latest/dex/search?q=...
    url = f"{DEX_BASE}/latest/dex/search"
    r = requests.get(url, params={"q": q.strip()}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("pairs", []) or []


def score_pair(p: Dict[str, Any]) -> float:
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)

    pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)

    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)

    # Нормальний “екшен”: є і покупки і продажі, є ліквідність і обʼєм
    trades5 = buys5 + sells5

    # Простий скор: ліквідність + короткий обʼєм + активність, з легким бонусом за імпульс
    s = 0.0
    s += min(liq / 1000.0, 400.0)              # кап до 400
    s += min(vol24 / 10000.0, 300.0)           # кап до 300
    s += min(vol5 / 2000.0, 200.0)             # кап до 200
    s += min(trades5 * 2.0, 120.0)             # кап до 120
    s += max(min(pc1h, 80.0), -80.0) * 0.3     # +/- імпульс
    s += max(min(pc5, 30.0), -30.0) * 0.2
    return round(s, 2)


def build_trade_hint(p: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Повертає (рішення, теги)"""
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)

    pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)

    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
    trades5 = buys5 + sells5

    tags = []

    if liq >= 250_000:
        tags.append("Liquid ✅")
    elif liq >= 80_000:
        tags.append("Liq ok")
    else:
        tags.append("Low liq ⚠️")

    if trades5 >= 60:
        tags.append("Hot flow 🔥")
    elif trades5 >= 25:
        tags.append("Active")
    else:
        tags.append("Thin")

    if abs(pc1h) >= 30:
        tags.append("Volatile")
    if pc1h >= 80 or pc5 >= 15:
        tags.append("Pumpish")
    if pc1h <= -40:
        tags.append("Dump risk")

    # Рішення (дуже грубо, але екшенабл)
    decision = "NO ENTRY"
    if liq >= 80_000 and vol24 >= 80_000 and trades5 >= 25 and sells5 >= 5:
        # якщо імпульс позитивний і є короткий обʼєм
        if pc1h > 5 and vol5 > 5_000:
            decision = "BUY (scalp)"
        elif pc1h > -5:
            decision = "WATCH / WAIT"
        else:
            decision = "NO ENTRY"
    return decision, tags


def is_name_suspicious(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip()
    # якщо пуста назва або купа “дивних” символів — підозріло
    if not base_sym:
        return True
    if not NAME_OK_RE.match(base_sym):
        return True
    return False


def main():
    st.set_page_config(page_title="DEX Scout — actionable candidates", layout="wide")
    st.title("DEX Scout — actionable candidates (DEXScreener API)")

    with st.sidebar:
        chain = st.selectbox(
            "Chain",
            options=sorted(list(CHAIN_DEX_PRESETS.keys())),
            index=sorted(list(CHAIN_DEX_PRESETS.keys())).index("bsc") if "bsc" in CHAIN_DEX_PRESETS else 0
        )

        st.caption("DEX filter")
        any_dex = st.checkbox("Any DEX (do not filter by DEX)", value=True)

        dex_options = CHAIN_DEX_PRESETS.get(chain, [])
        selected_dexes = []
        if not any_dex:
            selected_dexes = st.multiselect(
                "Allowed DEXes",
                options=dex_options,
                default=dex_options[:2] if len(dex_options) >= 2 else dex_options
            )

        top_n = st.slider("Top N", 5, 50, 15, step=5)

        min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=10000, step=5000)
        min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=50000, step=25000)

        st.divider()
        st.caption("“Real trading” filters")
        min_trades_m5 = st.slider("Min trades in last 5m (buys+sells)", 0, 200, 20, step=5)
        min_sells_m5 = st.slider("Min sells in last 5m", 0, 80, 5, step=1)
        max_buy_sell_imbalance = st.slider("Max buy/sell imbalance (buys vs sells)", 1, 20, 8, step=1)

        st.divider()
        block_suspicious_names = st.checkbox("Filter out suspicious token names (hieroglyphs / garbage)", value=True)

        st.divider()
        st.caption("Queries (comma-separated)")
        seeds = st.text_area(
            "Search seeds",
            value="WBNB, USDT, CAKE, BNB, PancakeSwap, meme",
            height=110
        )

        refresh = st.button("Refresh now")

    if refresh:
        st.cache_data.clear()

    queries = [q.strip() for q in seeds.split(",") if q.strip()]
    if not queries:
        st.info("Add at least 1 seed query in the sidebar.")
        return

    # Fetch pairs for each query and merge
    all_pairs = []
    for q in queries:
        try:
            all_pairs.extend(fetch_latest_pairs_for_query(q))
            time.sleep(0.12)  # tiny spacing, friendly
        except Exception as e:
            st.warning(f"Query failed: {q} — {e}")

    # Deduplicate by pairAddress
    uniq = {}
    for p in all_pairs:
        pa = p.get("pairAddress") or p.get("url")
        if not pa:
            continue
        uniq[pa] = p
    pairs = list(uniq.values())

    # Filters
    filtered = []
    for p in pairs:
        if (p.get("chainId") or "").lower() != chain.lower():
            continue

        if not any_dex and selected_dexes:
            if (p.get("dexId") or "").lower() not in set([d.lower() for d in selected_dexes]):
                continue

        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
        buys = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
        sells = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
        trades = buys + sells

        # core thresholds
        if liq < float(min_liq):
            continue
        if vol24 < float(min_vol24):
            continue

        # “real trading” thresholds
        if trades < int(min_trades_m5):
            continue
        if sells < int(min_sells_m5):
            continue

        # anti-one-sided flow (часто “накрутка” лише покупками)
        if sells > 0:
            imbalance = max(buys, sells) / max(1, min(buys, sells))
            if imbalance > max_buy_sell_imbalance:
                continue
        else:
            continue

        if block_suspicious_names and is_name_suspicious(p):
            continue

        filtered.append(p)

    # Score + sort
    ranked = []
    for p in filtered:
        s = score_pair(p)
        decision, tags = build_trade_hint(p)
        ranked.append((s, decision, tags, p))
    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[:top_n]

    st.metric("Passed filters", len(ranked))

    if not ranked:
        st.info("No pairs passed filters. Try lowering thresholds, changing seeds, or enabling Any DEX.")
        return

    # Render
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

        st.markdown("---")
        cols = st.columns([3, 2, 2, 2])

        with cols[0]:
            st.markdown(f"### [{base}/{quote}]({url})")
            st.caption(f"{chain} • {dex}")
            st.code(base_addr if base_addr else "no baseToken.address", language="text")
            st.caption("↑ Token contract (baseToken.address)")
            if pair_addr:
                st.caption("Pair / pool address")
                st.code(pair_addr, language="text")

        with cols[1]:
            st.write(f"**Price:** ${price}" if price else "**Price:** n/a")
            st.write(f"**Liq:** ${liq:,.0f}")
            st.write(f"**Vol24:** ${vol24:,.0f}")
            st.write(f"**Vol m5:** ${vol5:,.0f}")

        with cols[2]:
            st.write(f"**Δ m5:** {pc5:+.2f}%")
            st.write(f"**Δ h1:** {pc1h:+.2f}%")
            st.write(f"**Buys/Sells (m5):** {buys}/{sells}")
            st.write(f"**Score:** {s}")

        with cols[3]:
            st.write(f"**Action:** `{decision}`")
            st.write("**Tags:** " + ", ".join(tags))

            # Quick links
            if base_addr:
                # PancakeSwap link (works only if PCS supports token listing; not always)
                pcs_link = f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"
                st.markdown(f"- PancakeSwap swap link: {pcs_link}")

            if url:
                st.markdown(f"- DexScreener: {url}")


if __name__ == "__main__":
    main()
