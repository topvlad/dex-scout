import streamlit as st
import requests
import time
import re

API_BASE = "https://api.dexscreener.com"

st.set_page_config(
    page_title="Dex Scout",
    layout="wide",
)

st.title("🔥 Dex Scout — Action Scanner")

REFRESH_SEC = st.sidebar.slider("Refresh interval (sec)", 10, 120, 30)

# --- Helpers -------------------------------------------------

def has_cjk(text):
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

def phase_and_action(p):
    d5 = p["priceChange"].get("m5", 0)
    d1 = p["priceChange"].get("h1", 0)
    buys = p["txns"]["m5"]["buys"]
    sells = p["txns"]["m5"]["sells"]
    vol5 = p["volume"].get("m5", 0)

    if d5 > 20 or d1 > 200:
        return "OVEREXTENDED", "🔴 NO ENTRY"
    if sells >= buys:
        return "DISTRIBUTION", "⚫ AVOID"
    if vol5 > 3000 and d5 < 5:
        return "EARLY", "🟢 BUY"
    if vol5 > 3000 and d5 < 15:
        return "HEATING", "🟡 WATCH"
    return "UNKNOWN", "⚫ AVOID"

def swap_url(chain, address):
    if chain == "bsc":
        return f"https://pancakeswap.finance/swap?outputCurrency={address}"
    if chain == "solana":
        return f"https://jup.ag/swap/SOL-{address}"
    return None

# --- Load data ------------------------------------------------

def load_pairs():
    r = requests.get(f"{API_BASE}/token-boosts/latest/v1", timeout=10)
    return r.json() if r.status_code == 200 else []

# --- UI -------------------------------------------------------

placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("Passed filters")

        pairs = load_pairs()
        results = []

        for t in pairs:
            chain = t["chainId"]
            addr = t["tokenAddress"]

            r = requests.get(f"{API_BASE}/token-pairs/v1/{chain}/{addr}", timeout=10)
            if r.status_code != 200:
                continue

            for p in r.json():
                name = p["baseToken"]["name"]
                symbol = p["baseToken"]["symbol"]

                if has_cjk(name + symbol):
                    continue

                buys = p["txns"]["m5"]["buys"]
                sells = p["txns"]["m5"]["sells"]
                liq = p["liquidity"]["usd"]
                vol5 = p["volume"].get("m5", 0)

                if buys + sells < 15:
                    continue
                if sells < 5:
                    continue
                if buys / max(sells,1) > 5:
                    continue
                if liq < 50000:
                    continue
                if vol5 < 3000:
                    continue

                phase, action = phase_and_action(p)

                results.append({
                    "pair": f"{symbol}/{p['quoteToken']['symbol']}",
                    "chain": chain,
                    "price": p["priceUsd"],
                    "liq": liq,
                    "vol5": vol5,
                    "buys": buys,
                    "sells": sells,
                    "phase": phase,
                    "action": action,
                    "addr": addr,
                    "swap": swap_url(chain, addr)
                })

        st.metric("Tokens", len(results))

        for r in results:
            with st.expander(f"{r['pair']} — {r['action']}"):
                st.write(f"**Chain:** {r['chain']}")
                st.write(f"**Price:** ${r['price']}")
                st.write(f"**Liquidity:** ${int(r['liq'])}")
                st.write(f"**Vol m5:** ${int(r['vol5'])}")
                st.write(f"**Buys / Sells:** {r['buys']} / {r['sells']}")
                st.write(f"**Phase:** `{r['phase']}`")

                cols = st.columns(3)
                cols[0].button("📋 Copy contract", on_click=st.write, args=(r["addr"],))
                if r["swap"]:
                    cols[1].markdown(f"[🔗 Open Swap]({r['swap']})")
                if r["action"] == "🟢 BUY":
                    cols[2].markdown("🔔 **HOT NOW**")
                    st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

    time.sleep(REFRESH_SEC)
