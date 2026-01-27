# app.py
import os
import re
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import requests
import streamlit as st

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

NAME_OK_RE = re.compile(r"^[A-Za-z0-9\-\._\s]{2,30}$")


def safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


@st.cache_data(ttl=25, show_spinner=False)
def fetch_latest_pairs_for_query(q: str) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/latest/dex/search"
    r = requests.get(url, params={"q": q.strip()}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("pairs", []) or []


@st.cache_data(ttl=25, show_spinner=False)
def fetch_token_pairs(chain: str, token_address: str) -> List[Dict[str, Any]]:
    # returns list of pools for a token address
    url = f"{DEX_BASE}/token-pairs/v1/{chain}/{token_address}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json() or []


def best_pair_for_token(chain: str, token_address: str) -> Optional[Dict[str, Any]]:
    try:
        pools = fetch_token_pairs(chain, token_address)
    except Exception:
        return None
    if not pools:
        return None
    # choose best by liquidity usd, then by vol24
    def key(p):
        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
        return (liq, vol24)
    pools.sort(key=key, reverse=True)
    return pools[0]


def is_name_suspicious(p: Dict[str, Any]) -> bool:
    base_sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip()
    if not base_sym:
        return True
    if not NAME_OK_RE.match(base_sym):
        return True
    return False


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


def build_trade_hint(p: Dict[str, Any]) -> Tuple[str, List[str]]:
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

    # decision (actionable but conservative)
    decision = "NO ENTRY"
    if liq >= 80_000 and vol24 >= 80_000 and trades5 >= 25 and sells5 >= 5:
        if pc1h > 5 and vol5 > 5_000 and pc5 >= -2:
            decision = "ENTRY (scalp)"
        elif pc1h > -5:
            decision = "WATCH / WAIT"
        else:
            decision = "NO ENTRY"

    return decision, tags


def action_badge(action: str) -> str:
    # color via HTML
    a = (action or "").upper()
    if "ENTRY" in a or a.startswith("BUY"):
        color = "#1f9d55"  # green
        bg = "rgba(31,157,85,0.15)"
    elif "WATCH" in a or "WAIT" in a:
        color = "#d69e2e"  # yellow
        bg = "rgba(214,158,46,0.15)"
    else:
        color = "#e53e3e"  # red
        bg = "rgba(229,62,62,0.12)"
    return f"""
    <span style="
      display:inline-block;
      padding:4px 10px;
      border-radius:999px;
      border:1px solid {color};
      background:{bg};
      color:{color};
      font-weight:700;
      font-size:12px;
      letter-spacing:0.4px;
    ">{action}</span>
    """


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

    # avoid duplicates: if same base_addr active exists, don't spam
    for r in rows:
        if r.get("active") == "1" and r.get("base_token_address", "").lower() == base_addr.lower():
            # update note and last seen price if you want — keep simple:
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

    # very simple rules
    if change >= 35 and pc5 < 0 and pc1h < 0:
        return "SELL (take profit)"
    if change >= 20 and pc5 < -2:
        return "TRIM / TP"
    if change <= -20 and pc1h < -5:
        return "CUT / RISK"
    if pc1h > 5 and pc5 >= 0:
        return "HOLD (momentum)"
    return "HOLD / WAIT"


def main():
    st.set_page_config(page_title="DEX Scout", layout="wide")

    tabs = st.tabs(["Scout", "Portfolio"])

    # -------------------- SCOUT TAB --------------------
    with tabs[0]:
        st.title("DEX Scout — actionable candidates (DEXScreener API)")
        st.caption("Кнопки + логіка “Swap натиснув → запис у Portfolio”")

        with st.sidebar:
            chain = st.selectbox(
                "Chain",
                options=sorted(list(CHAIN_DEX_PRESETS.keys())),
                index=sorted(list(CHAIN_DEX_PRESETS.keys())).index("bsc") if "bsc" in CHAIN_DEX_PRESETS else 0,
            )

            st.caption("DEX filter")
            any_dex = st.checkbox("Any DEX (do not filter by DEX)", value=True)

            dex_options = CHAIN_DEX_PRESETS.get(chain, [])
            selected_dexes = []
            if not any_dex:
                selected_dexes = st.multiselect(
                    "Allowed DEXes",
                    options=dex_options,
                    default=dex_options[:3] if len(dex_options) >= 3 else dex_options,
                )

            top_n = st.slider("Top N", 5, 50, 15, step=5)

            min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=25000, step=5000)
            min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=150000, step=25000)

            st.divider()
            st.caption("“Real trading” filters")
            min_trades_m5 = st.slider("Min trades in last 5m (buys+sells)", 0, 200, 25, step=5)
            min_sells_m5 = st.slider("Min sells in last 5m", 0, 80, 6, step=1)
            max_buy_sell_imbalance = st.slider("Max buy/sell imbalance (buys vs sells)", 1, 20, 8, step=1)

            st.divider()
            block_suspicious_names = st.checkbox("Filter out suspicious token names", value=True)

            st.divider()
            st.caption("Queries (comma-separated)")
            seeds = st.text_area(
                "Search seeds",
                value="WBNB, USDT, BNB, meme, ai, gaming, cat, dog",
                height=110,
            )

            refresh = st.button("Refresh now")

        if refresh:
            st.cache_data.clear()

        queries = [q.strip() for q in seeds.split(",") if q.strip()]
        if not queries:
            st.info("Add at least 1 seed query in the sidebar.")
            return

        # Fetch pairs
        all_pairs = []
        for q in queries:
            try:
                all_pairs.extend(fetch_latest_pairs_for_query(q))
                time.sleep(0.12)
            except Exception as e:
                st.warning(f"Query failed: {q} — {e}")

        # Deduplicate
        uniq = {}
        for p in all_pairs:
            pa = p.get("pairAddress") or p.get("url")
            if not pa:
                continue
            uniq[pa] = p
        pairs = list(uniq.values())

        # Filters
        filtered = []
        allowed = set([d.lower() for d in selected_dexes]) if selected_dexes else set()

        for p in pairs:
            if (p.get("chainId") or "").lower() != chain.lower():
                continue

            if not any_dex and allowed:
                if (p.get("dexId") or "").lower() not in allowed:
                    continue

            liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
            vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
            buys = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
            sells = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
            trades = buys + sells

            if liq < float(min_liq):
                continue
            if vol24 < float(min_vol24):
                continue

            if trades < int(min_trades_m5):
                continue
            if sells < int(min_sells_m5):
                continue

            if sells > 0:
                imbalance = max(buys, sells) / max(1, min(buys, sells))
                if imbalance > max_buy_sell_imbalance:
                    continue
            else:
                continue

            if block_suspicious_names and is_name_suspicious(p):
                continue

            filtered.append(p)

        # Rank
        ranked = []
        for p in filtered:
            s = score_pair(p)
            decision, tags = build_trade_hint(p)
            ranked.append((s, decision, tags, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:top_n]

        st.metric("Passed filters", len(ranked))

        if not ranked:
            st.info("No pairs passed filters. Lower thresholds / change seeds / enable Any DEX.")
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

            swap_url = f"https://pancakeswap.finance/swap?outputCurrency={base_addr}" if base_addr else ""

            st.markdown("---")
            left, mid, right = st.columns([3, 2, 2])

            with left:
                st.markdown(f"### {base}/{quote}")
                st.caption(f"{p.get('chainId','')} • {dex}")
                if url:
                    st.link_button("Open DexScreener", url)
                if base_addr:
                    st.caption("Token contract (baseToken.address)")
                    st.code(base_addr, language="text")
                if pair_addr:
                    st.caption("Pair / pool address")
                    st.code(pair_addr, language="text")

            with mid:
                st.write(f"**Price:** ${price}" if price else "**Price:** n/a")
                st.write(f"**Liq:** ${liq:,.0f}")
                st.write(f"**Vol24:** ${vol24:,.0f}")
                st.write(f"**Vol m5:** ${vol5:,.0f}")
                st.write(f"**Δ m5:** {pc5:+.2f}%")
                st.write(f"**Δ h1:** {pc1h:+.2f}%")
                st.write(f"**Buys/Sells (m5):** {buys}/{sells}")
                st.write(f"**Score:** {s}")

            with right:
                st.markdown(action_badge(decision), unsafe_allow_html=True)
                st.write("**Tags:** " + ", ".join(tags))

                # Swap logging button
                if base_addr and swap_url:
                    key_log = f"log_{base_addr}"
                    if st.button("Swap (log)", key=key_log, use_container_width=True):
                        res = log_swap_intent(p, s, decision, tags, swap_url)
                        if res == "OK":
                            st.success("Logged to Portfolio (entry snapshot saved).")
                        else:
                            st.info("Already in Portfolio (active).")

                    # real open swap (separate click, but reliable)
                    st.link_button("Open Swap", swap_url, use_container_width=True)
                else:
                    st.caption("Swap link unavailable (missing base token address).")

    # -------------------- PORTFOLIO TAB --------------------
    with tabs[1]:
        st.title("Portfolio / Watchlist")
        st.caption("Тут зʼявляються позиції після натискання **Swap (log)** у Scout.")

        rows = load_portfolio()
        active_rows = [r for r in rows if r.get("active") == "1"]
        closed_rows = [r for r in rows if r.get("active") != "1"]

        topbar = st.columns([2, 2, 2])
        with topbar[0]:
            st.metric("Active", len(active_rows))
        with topbar[1]:
            st.metric("Closed", len(closed_rows))
        with topbar[2]:
            if st.button("Export CSV"):
                # Offer download
                with open(TRADES_CSV, "rb") as f:
                    st.download_button("Download portfolio.csv", f, file_name="portfolio.csv")

        st.markdown("---")

        if not active_rows:
            st.info("Portfolio пустий. Йди в Scout → натисни **Swap (log)** на потрібному токені.")
            return

        # Update current prices
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

            # PnL
            pnl = 0.0
            if entry_price > 0 and cur_price > 0:
                pnl = (cur_price - entry_price) / entry_price * 100.0

            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                st.markdown(f"### {base_sym}/{quote_sym}")
                st.caption(f"Chain: {chain} • DEX: {r.get('dex','')}")
                st.code(base_addr, language="text")
                if r.get("dexscreener_url"):
                    st.link_button("DexScreener", r["dexscreener_url"])
                if r.get("swap_url"):
                    st.link_button("Swap", r["swap_url"])

            with c2:
                st.write(f"**Entry:** ${entry_price_str}")
                st.write(f"**Now:** ${cur_price:.8f}" if cur_price else "**Now:** n/a")
                st.write(f"**PnL:** {pnl:+.2f}%" if entry_price and cur_price else "**PnL:** n/a")
                st.write(f"**Liq:** ${liq:,.0f}" if liq else "**Liq:** n/a")

            with c3:
                st.write(f"**Δ m5:** {pc5:+.2f}%")
                st.write(f"**Δ h1:** {pc1h:+.2f}%")
                st.write(f"**Reco:** `{reco}`")

                note_key = f"note_{idx}_{base_addr}"
                note_val = st.text_input("Note", value=r.get("note", ""), key=note_key)

            with c4:
                # delete/close controls
                close_key = f"close_{idx}_{base_addr}"
                delete_key = f"delete_{idx}_{base_addr}"

                close = st.checkbox("Close (remove from active)", value=False, key=close_key)
                delete = st.checkbox("Delete row", value=False, key=delete_key)

                if st.button("Apply", key=f"apply_{idx}_{base_addr}", use_container_width=True):
                    # apply changes into full rows list
                    all_rows = load_portfolio()
                    # find matching active row by base_token_address + active
                    for rr in all_rows:
                        if rr.get("active") == "1" and (rr.get("base_token_address") or "").lower() == base_addr.lower():
                            rr["note"] = note_val
                            if close:
                                rr["active"] = "0"
                            break

                    if delete:
                        all_rows = [x for x in all_rows if not ((x.get("active") == "1") and ((x.get("base_token_address") or "").lower() == base_addr.lower()))]

                    save_portfolio(all_rows)
                    st.success("Saved.")
                    st.rerun()

            st.markdown("---")

        if closed_rows:
            with st.expander("Closed / Archived"):
                for r in closed_rows[-20:]:
                    st.write(f"{r.get('ts_utc','')} — {r.get('base_symbol','')}/{r.get('quote_symbol','')} — entry ${r.get('entry_price_usd','')} — {r.get('action','')}")


if __name__ == "__main__":
    main()
