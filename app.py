# app.py
# DexScreener Option B (SEARCH-based) scanner
# Streamlit-safe: retries, rate-limit handling, request budget, no infinite loops.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import streamlit as st

# requests may be missing if requirements not installed correctly
try:
    import requests
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: requests\n"
        "➡️ Ensure requirements.txt includes: requests>=2.31.0\n"
        "➡️ Then redeploy/reboot the app."
    ) from e


# -----------------------------
# Config
# -----------------------------
DEXSCREENER_BASE = "https://api.dexscreener.com"

DEFAULT_TIMEOUT_SEC = 10
MAX_PAIRS_PER_QUERY = 60            # keep it lighter than 80
SLEEP_BETWEEN_CALLS_SEC = 0.12      # polite pacing
DEFAULT_TOP_N = 25

DEFAULT_MIN_LIQ_USD = 10_000
DEFAULT_MIN_VOL24_USD = 50_000

# Hard budget to avoid killing the Streamlit Cloud instance
MAX_HTTP_CALLS_PER_RUN = 20

DEFAULT_QUERIES = {
    "solana": ["SOL", "USDC", "JUP", "RAY", "BONK", "WIF", "pump"],
    "bsc": ["WBNB", "USDT", "CAKE", "BNB", "PancakeSwap", "meme"],
}

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "dex-scout/1.0 (+streamlit)",
    }
)


# -----------------------------
# Helpers
# -----------------------------
def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def pick_txns(obj: dict) -> Tuple[int, int]:
    txns = obj.get("txns") or {}
    if not isinstance(txns, dict):
        return 0, 0
    for tf in ("m5", "m15", "m30", "h1"):
        block = txns.get(tf)
        if isinstance(block, dict):
            buys = int(safe_num(block.get("buys"), 0))
            sells = int(safe_num(block.get("sells"), 0))
            return buys, sells
    return 0, 0


@dataclass
class PairRow:
    chain: str
    dex: str
    pair: str
    url: str
    base: str
    quote: str
    price: float
    liquidity: float
    vol24: float
    vol5: float
    pc5: float
    pc1: float
    pc24: float
    buys: int
    sells: int
    score: float


def parse_pair(p: dict) -> PairRow:
    base = (p.get("baseToken") or {}).get("symbol", "") or ""
    quote = (p.get("quoteToken") or {}).get("symbol", "") or ""

    liquidity = safe_num((p.get("liquidity") or {}).get("usd"), 0.0)
    volume = p.get("volume") or {}
    pc = p.get("priceChange") or {}

    vol24 = safe_num(volume.get("h24"), 0.0) if isinstance(volume, dict) else 0.0
    vol5 = safe_num(volume.get("m5"), 0.0) if isinstance(volume, dict) else 0.0

    pc5 = safe_num(pc.get("m5"), 0.0) if isinstance(pc, dict) else 0.0
    pc1 = safe_num(pc.get("h1"), 0.0) if isinstance(pc, dict) else 0.0
    pc24 = safe_num(pc.get("h24"), 0.0) if isinstance(pc, dict) else 0.0

    buys, sells = pick_txns(p)
    buy_sell = (buys + 1) / (sells + 1)

    # simple score tuned for "early activity without instant blow-off"
    blowoff_penalty = max(0.0, (abs(pc5) - 35.0) / 35.0)
    blowoff_penalty = min(2.0, blowoff_penalty)

    liq_factor = min(2.0, liquidity / 50_000.0)
    activity = vol5 / 1000.0
    momentum = pc5 * 0.6 + pc1 * 0.3 + pc24 * 0.1

    score = (momentum * 1.2) + (activity * 1.4) + (buy_sell * 7.0) + (liq_factor * 6.0) - (blowoff_penalty * 10.0)

    return PairRow(
        chain=str(p.get("chainId") or ""),
        dex=str(p.get("dexId") or ""),
        pair=str(p.get("pairAddress") or ""),
        url=str(p.get("url") or ""),
        base=str(base),
        quote=str(quote),
        price=safe_num(p.get("priceUsd"), 0.0),
        liquidity=liquidity,
        vol24=vol24,
        vol5=vol5,
        pc5=pc5,
        pc1=pc1,
        pc24=pc24,
        buys=buys,
        sells=sells,
        score=float(score),
    )


class HttpBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self, n: int = 1):
        self.used += n
        if self.used > self.limit:
            raise RuntimeError(f"HTTP budget exceeded: {self.used}/{self.limit}. Reduce queries or refresh less often.")


def http_get_json(url: str, params: Optional[dict], budget: HttpBudget) -> Any:
    """
    Resilient GET:
      - retries on 429 / 5xx
      - short timeouts
      - backoff
    """
    budget.spend(1)
    last_err = None

    for attempt in range(1, 4):  # 3 tries
        try:
            r = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT_SEC)
            if r.status_code == 429:
                # rate limit: backoff
                time.sleep(0.6 * attempt)
                last_err = RuntimeError("DexScreener rate-limited (429).")
                continue
            if 500 <= r.status_code < 600:
                time.sleep(0.4 * attempt)
                last_err = RuntimeError(f"DexScreener server error ({r.status_code}).")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.25 * attempt)

    raise RuntimeError(f"HTTP failed after retries: {last_err}")


@st.cache_data(ttl=25, show_spinner=False)
def search_pairs(query: str) -> List[dict]:
    # NOTE: we do not keep budget inside cache function (cache may rerun unpredictably),
    # budget is enforced by caller via query loop count.
    url = f"{DEXSCREENER_BASE}/latest/dex/search"
    r = SESSION.get(url, params={"q": query}, timeout=DEFAULT_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    pairs = data.get("pairs", []) or []
    if not isinstance(pairs, list):
        return []
    return pairs[:MAX_PAIRS_PER_QUERY]


def uniq_pairs(pairs: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for p in pairs:
        key = (p.get("chainId"), p.get("pairAddress"))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DEX Scout — Option B", layout="wide")
st.title("DEX Scout — Option B (DexScreener Search)")
st.caption("Search-based candidate stream → filter → rank. Safe retries + no hard crashes.")

with st.sidebar:
    chain = st.selectbox("Chain", ["bsc", "solana"])
    top_n = st.slider("Top N", 5, 80, DEFAULT_TOP_N, 5)

    min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=DEFAULT_MIN_LIQ_USD, step=1000)
    min_vol = st.number_input("Min 24h Volume ($)", min_value=0, value=DEFAULT_MIN_VOL24_USD, step=5000)

    st.divider()
    st.subheader("Queries (comma-separated)")
    queries_text = st.text_area("Search seeds", value=", ".join(DEFAULT_QUERIES[chain]), height=90)

    strict = st.checkbox("Strict match (query in symbol pair)", value=False)
    refresh = st.button("Refresh now", type="primary")
    clear_cache = st.button("Clear cache (if stuck)")

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared. Now hit Refresh.")
    st.stop()

if refresh:
    # force fresh fetch
    st.cache_data.clear()

queries = [q.strip() for q in queries_text.split(",") if q.strip()]
if not queries:
    st.warning("Add at least one query.")
    st.stop()

# Request budget: approximate count = len(queries) (since cached function will still fetch once per query on fresh run)
if len(queries) > MAX_HTTP_CALLS_PER_RUN:
    st.warning(f"Too many queries ({len(queries)}). Reduce to <= {MAX_HTTP_CALLS_PER_RUN}.")
    st.stop()

pairs_raw: List[dict] = []
errors: List[str] = []

with st.spinner("Fetching from DexScreener…"):
    for q in queries:
        try:
            pairs = search_pairs(q)
            if strict:
                q_low = q.lower()
                def ok(p: dict) -> bool:
                    b = ((p.get("baseToken") or {}).get("symbol") or "").lower()
                    qt = ((p.get("quoteToken") or {}).get("symbol") or "").lower()
                    return (q_low in f"{b}/{qt}")
                pairs = [p for p in pairs if ok(p)]
            pairs_raw.extend(pairs)
        except Exception as e:
            errors.append(f"{q}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

pairs_raw = uniq_pairs(pairs_raw)

if errors:
    with st.expander("Fetch errors"):
        for e in errors:
            st.write(e)

if not pairs_raw:
    st.info("No pairs returned. Change queries.")
    st.stop()

# filter by chain
pairs_chain = [p for p in pairs_raw if str(p.get("chainId") or "").lower() == chain]
if not pairs_chain:
    st.info(f"No pairs matched chain '{chain}'. Try other queries or switch chain.")
    st.stop()

rows: List[PairRow] = []
for p in pairs_chain:
    try:
        r = parse_pair(p)
        if r.liquidity < float(min_liq):
            continue
        if r.vol24 < float(min_vol):
            continue
        rows.append(r)
    except Exception:
        continue

if not rows:
    st.warning("Nothing passed filters. Lower min liquidity/volume or widen queries.")
    st.stop()

rows.sort(key=lambda x: x.score, reverse=True)
rows = rows[: int(top_n)]

m1, m2, m3 = st.columns(3)
m1.metric("Unique pairs", len(pairs_raw))
m2.metric("Chain matched", len(pairs_chain))
m3.metric("Passed filters", len(rows))

st.divider()

# Render list (fast + readable)
for r in rows:
    a, b, c = st.columns([2.2, 1.3, 1.5])
    with a:
        st.markdown(f"### [{r.base}/{r.quote}]({r.url})")
        st.caption(f"{r.chain} • {r.dex} • `{r.pair}`")
    with b:
        st.write(f"**Price:** ${r.price:.10f}".rstrip("0").rstrip("."))
        st.write(f"**Liq:** ${r.liquidity:,.0f}")
        st.write(f"**Vol24:** ${r.vol24:,.0f}")
        st.write(f"**Vol m5:** ${r.vol5:,.0f}")
    with c:
        st.write(f"**Δ m5:** {r.pc5:+.2f}%")
        st.write(f"**Δ h1:** {r.pc1:+.2f}%")
        st.write(f"**Buys/Sells:** {r.buys}/{r.sells}")
        st.write(f"**Score:** {r.score:,.2f}")
    st.divider()

st.caption("If you get 429/rate limits, reduce queries or refresh less often.")
