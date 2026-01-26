import re
import time
import math
import base64
from io import BytesIO
from datetime import datetime, timezone

import requests
import streamlit as st


DEX_BASE = "https://api.dexscreener.com"

# --------- Helpers ---------
def safe_get(d, path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def is_ascii_like(s: str) -> bool:
    """Rejects most non-latin / hieroglyphic names. Allows common tickers."""
    if not s:
        return False
    # allow: letters, digits, space, _ - . / + #
    return re.fullmatch(r"[A-Za-z0-9 _\-\.\+/#]{2,}", s) is not None


def now_ts():
    return int(time.time())


def age_hours(pair_created_at_ms: int) -> float:
    if not pair_created_at_ms:
        return 99999.0
    return (now_ts() * 1000 - int(pair_created_at_ms)) / 1000 / 3600


def make_beep_wav_base64(freq=880, duration=0.18, sr=22050):
    """Generate a short beep WAV (base64) without extra deps."""
    import wave
    import struct

    n = int(sr * duration)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            # simple sine
            val = int(16000 * math.sin(2 * math.pi * freq * (i / sr)))
            wf.writeframes(struct.pack("<h", val))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


BEEP_B64 = make_beep_wav_base64()


def chain_swap_link(chain_id: str, token_addr: str):
    chain_id = (chain_id or "").lower()
    if chain_id in ["bsc", "bnb", "bnbchain", "bep20"]:
        # PancakeSwap BSC
        return f"https://pancakeswap.finance/swap?outputCurrency={token_addr}"
    if chain_id == "solana":
        # Jupiter
        return f"https://jup.ag/swap/SOL-{token_addr}"
    # fallback to dexscreener token page (not perfect but better than nothing)
    return f"https://dexscreener.com/{chain_id}/{token_addr}"


def dexscreener_pair_link(chain_id: str, pair_addr: str):
    return f"https://dexscreener.com/{chain_id}/{pair_addr}"


# --------- API ---------
@st.cache_data(ttl=25)
def ds_search(query: str):
    url = f"{DEX_BASE}/latest/dex/search"
    r = requests.get(url, params={"q": query}, timeout=20)
    r.raise_for_status()
    return r.json()


def extract_timeframe_metrics(pair: dict):
    """
    Dexscreener fields:
    - txns: { m5: {buys,sells}, h1: {...}, h24: {...} } (timeframes vary)
    - volume: { m5: x, h1: x, h24: x } (varies)
    - priceChange: { m5: %, h1: %, h24:% } (varies)
    """
    txns = pair.get("txns", {}) or {}
    volume = pair.get("volume", {}) or {}
    pchg = pair.get("priceChange", {}) or {}

    # prefer m5 & h1 if present
    buys_m5 = safe_get(txns, ["m5", "buys"], 0) or 0
    sells_m5 = safe_get(txns, ["m5", "sells"], 0) or 0
    vol_m5 = volume.get("m5", 0) or 0
    d_m5 = pchg.get("m5", 0) or 0

    d_h1 = pchg.get("h1", 0) or 0
    vol_h24 = volume.get("h24", 0) or 0
    liq_usd = safe_get(pair, ["liquidity", "usd"], 0) or 0

    return {
        "buys_m5": int(buys_m5),
        "sells_m5": int(sells_m5),
        "vol_m5": float(vol_m5),
        "d_m5": float(d_m5),
        "d_h1": float(d_h1),
        "vol_h24": float(vol_h24),
        "liq_usd": float(liq_usd),
    }


def tag_and_decide(pair: dict, m: dict):
    """
    Simple actionable rules (not MA-based).
    """
    tags = []

    created = pair.get("pairCreatedAt")
    age_h = age_hours(created)
    if age_h < 6:
        tags.append("EARLY<6H")
    elif age_h < 24:
        tags.append("EARLY<24H")

    if m["liq_usd"] >= 150_000:
        tags.append("LIQ_STRONG")
    elif m["liq_usd"] >= 50_000:
        tags.append("LIQ_OK")
    else:
        tags.append("LIQ_LOW")

    if m["vol_m5"] >= 20_000:
        tags.append("VOL_SPIKE")
    elif m["vol_m5"] >= 5_000:
        tags.append("VOL_OK")
    else:
        tags.append("VOL_LOW")

    total_tx = m["buys_m5"] + m["sells_m5"]
    if total_tx >= 80:
        tags.append("TXNS_HOT")
    elif total_tx >= 25:
        tags.append("TXNS_OK")
    else:
        tags.append("TXNS_THIN")

    if m["sells_m5"] == 0 and m["buys_m5"] > 0:
        tags.append("NO_SELLS(FAKE?)")
    elif m["buys_m5"] == 0 and m["sells_m5"] > 0:
        tags.append("NO_BUYS")

    # pressure
    if m["buys_m5"] >= max(1, int(m["sells_m5"] * 1.6)):
        tags.append("BUY_PRESSURE")
    elif m["sells_m5"] >= max(1, int(m["buys_m5"] * 1.6)):
        tags.append("SELL_PRESSURE")

    # decision
    # Enter if: liquidity ok, vol ok, txns ok, has sells, no extreme red flags
    red_flags = ("NO_SELLS(FAKE?)" in tags) or ("VOL_LOW" in tags) or ("TXNS_THIN" in tags) or ("LIQ_LOW" in tags)
    good = ("VOL_OK" in tags or "VOL_SPIKE" in tags) and ("TXNS_OK" in tags or "TXNS_HOT" in tags) and ("LIQ_OK" in tags or "LIQ_STRONG" in tags) and (m["sells_m5"] >= 3)

    if good and not red_flags:
        decision = "ENTER"
    elif red_flags:
        decision = "AVOID"
    else:
        decision = "WAIT"

    return tags, decision


def score_pair(pair: dict, m: dict):
    # heuristic score used for ranking
    base = 0.0
    base += min(m["liq_usd"] / 10_000, 40)         # up to 40
    base += min(m["vol_m5"] / 1_000, 40)           # up to 40
    base += min((m["buys_m5"] + m["sells_m5"]), 60) / 2  # up to 30
    base += min(max(m["d_h1"], 0), 120) / 6        # up to 20
    # penalty for no sells
    if m["sells_m5"] == 0:
        base *= 0.45
    return round(base, 2)


# --------- UI ---------
st.set_page_config(page_title="DEX Scout", layout="wide")

st.title("DEX Scout — actionable candidates (DEXScreener API)")

with st.sidebar:
    st.subheader("Filters")

    chain = st.selectbox("Chain", ["bsc", "solana"], index=0)

    top_n = st.slider("Top N", min_value=5, max_value=30, value=12, step=1)

    min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=50000, step=5000)
    min_vol24 = st.number_input("Min 24h Volume ($)", min_value=0, value=100000, step=10000)

    # stricter activity filters
    st.markdown("---")
    st.caption("Activity (to avoid fake/owner-only prints)")
    min_m5_vol = st.number_input("Min vol m5 ($)", min_value=0, value=3000, step=500)
    min_m5_tx = st.number_input("Min txns m5 (buys+sells)", min_value=0, value=20, step=5)
    min_m5_sells = st.number_input("Min sells m5", min_value=0, value=3, step=1)

    ascii_only = st.checkbox("Filter out non-latin names/symbols", value=True)

    st.markdown("---")
    seeds = st.text_area(
        "Queries (comma-separated)",
        value="WBNB, USDT, BNB, PancakeSwap, meme, AI, dog, cat, pepe",
        height=90
    )

    refresh = st.button("Refresh now")
    sound_alert = st.checkbox("Sound alert on HOT ENTER", value=False)

# refresh trick
if refresh:
    st.cache_data.clear()

seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
if not seed_list:
    st.warning("Add at least 1 query seed.")
    st.stop()

# fetch
pairs_all = []
errors = 0

for q in seed_list[:20]:
    try:
        data = ds_search(q)
        pairs = data.get("pairs", []) or []
        pairs_all.extend(pairs)
    except Exception:
        errors += 1

# dedupe by pairAddress
seen = set()
dedup = []
for p in pairs_all:
    pa = p.get("pairAddress")
    if not pa:
        continue
    key = (p.get("chainId"), pa)
    if key in seen:
        continue
    seen.add(key)
    dedup.append(p)

# filter chain + minimums
filtered = []
for p in dedup:
    if (p.get("chainId") or "").lower() != chain:
        continue

    base_token = p.get("baseToken", {}) or {}
    quote_token = p.get("quoteToken", {}) or {}

    name = (base_token.get("name") or "")[:80]
    sym = (base_token.get("symbol") or "")[:32]

    if ascii_only and (not is_ascii_like(sym) or not is_ascii_like(name)):
        continue

    m = extract_timeframe_metrics(p)

    if m["liq_usd"] < float(min_liq):
        continue
    if m["vol_h24"] < float(min_vol24):
        continue
    if m["vol_m5"] < float(min_m5_vol):
        continue
    if (m["buys_m5"] + m["sells_m5"]) < int(min_m5_tx):
        continue
    if m["sells_m5"] < int(min_m5_sells):
        continue

    tags, decision = tag_and_decide(p, m)
    score = score_pair(p, m)

    filtered.append({
        "pair": p,
        "m": m,
        "tags": tags,
        "decision": decision,
        "score": score
    })

filtered.sort(key=lambda x: x["score"], reverse=True)
filtered = filtered[:top_n]

# header
colA, colB = st.columns([1, 1])
with colA:
    st.metric("Passed filters", value=len(filtered))
with colB:
    if errors:
        st.warning(f"Some queries failed: {errors}")

st.markdown("---")

if not filtered:
    st.info("No pairs passed filters. Try lowering thresholds or changing seeds.")
    st.stop()

# HOT alert
hot_enters = [x for x in filtered if x["decision"] == "ENTER" and ("TXNS_HOT" in x["tags"] or "VOL_SPIKE" in x["tags"])]
if hot_enters:
    st.toast(f"HOT ENTER candidates: {len(hot_enters)}", icon="🔥")
    if sound_alert:
        st.audio(base64.b64decode(BEEP_B64), format="audio/wav")

# render list
for item in filtered:
    p = item["pair"]
    m = item["m"]
    tags = item["tags"]
    decision = item["decision"]
    score = item["score"]

    base_token = p.get("baseToken", {}) or {}
    quote_token = p.get("quoteToken", {}) or {}

    chain_id = (p.get("chainId") or "").lower()
    dex_id = p.get("dexId") or "dex"
    pair_addr = p.get("pairAddress") or ""

    token_addr = base_token.get("address") or ""  # THIS is what PancakeSwap needs
    token_sym = base_token.get("symbol") or "TOKEN"
    quote_sym = quote_token.get("symbol") or "QUOTE"

    price = p.get("priceUsd") or "?"
    url_pair = p.get("url") or dexscreener_pair_link(chain_id, pair_addr)

    swap_url = chain_swap_link(chain_id, token_addr)

    created_at = p.get("pairCreatedAt") or None
    age_h = age_hours(created_at)

    left, right = st.columns([2.2, 1.3])

    with left:
        st.subheader(f"{token_sym}/{quote_sym}")
        st.caption(f"{chain_id} • {dex_id}")

        # IMPORTANT: show both addresses clearly
        st.code(f"TOKEN (buy/import): {token_addr}", language="text")
        st.code(f"PAIR/LP (Dexscreener): {pair_addr}", language="text")

        btn1, btn2, btn3 = st.columns([1, 1, 1])
        with btn1:
            st.link_button("Open Dexscreener", url_pair)
        with btn2:
            st.link_button("Open Swap", swap_url)
        with btn3:
            st.link_button("Copy token addr (manual)", f"https://dexscreener.com/{chain_id}/{pair_addr}")

        st.write("Tags:", ", ".join(tags))
        st.write("Decision:", decision)

    with right:
        st.write(f"**Price:** ${price}")
        st.write(f"**Liq:** ${m['liq_usd']:,.0f}")
        st.write(f"**Vol24:** ${m['vol_h24']:,.0f}")
        st.write(f"**Vol m5:** ${m['vol_m5']:,.0f}")
        st.write(f"**Δ m5:** {m['d_m5']:+.2f}%")
        st.write(f"**Δ h1:** {m['d_h1']:+.2f}%")
        st.write(f"**Buys/Sells m5:** {m['buys_m5']}/{m['sells_m5']}")
        st.write(f"**Age:** {age_h:.1f}h")
        st.write(f"**Score:** {score}")

    st.markdown("---")
