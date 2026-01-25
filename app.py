# app.py
import time
import pandas as pd
import streamlit as st

from config import CHAINS_DEFAULT, TOP_N, ALERT_MIN_SCORE
from dex import fetch_pairs_by_query, normalize_pair
from scoring import passes_safe_filters, score_row
from alerts import purge_expired, upsert_watchlist, build_alerts, snapshot_watchlist

st.set_page_config(page_title="Dex Scout (SOL+BSC)", layout="wide")
st.title("Dex Scout — Safe Micro-Edge (DexScreener-only)")
st.caption("Логіка: активність/обʼєм ростуть, ціна ще НЕ втекла (m5). Alerts/Watchlist тримають сигнал 15 хв.")

# ----------------------------
# Session state init
# ----------------------------
if "watch" not in st.session_state:
    st.session_state.watch = {}  # key -> WatchItem (dataclass) stored in alerts.py layer
if "last_scan_ts" not in st.session_state:
    st.session_state.last_scan_ts = 0.0

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.subheader("Скан")
    chains = st.multiselect("Мережі", ["solana", "bsc", "base", "ethereum"], default=CHAINS_DEFAULT)
    auto = st.toggle("Авто-оновлення", value=False)
    interval = st.slider("Інтервал (сек)", 10, 120, 30)

    st.subheader("Alerts")
    min_score = st.slider("Мін. score для alert", 0.0, 50.0, float(ALERT_MIN_SCORE), 0.5)

    colA, colB = st.columns(2)
    refresh_now = colA.button("Оновити зараз")
    clear_watch = colB.button("Очистити watchlist")

    st.divider()
    st.write("🧠 Порада: якщо alerts порожні — це нормально. Це означає, що ринок не дає +EV сетапів зараз.")

if clear_watch:
    st.session_state.watch = {}

# ----------------------------
# Data scan
# ----------------------------
@st.cache_data(ttl=20, show_spinner=False)
def scan(chains_list):
    all_pairs = []
    # простий спосіб: q = chain name
    for c in chains_list:
        pairs = fetch_pairs_by_query(c)
        all_pairs.extend(pairs)

    rows = [normalize_pair(p) for p in all_pairs]
    rows = [r for r in rows if r.get("chain") in set(chains_list)]

    # uniq by (chain, pair)
    uniq = {}
    for r in rows:
        k = (r.get("chain"), r.get("pairAddress"))
        if k not in uniq:
            uniq[k] = r
    return list(uniq.values())

def build_candidates(rows):
    filtered = [r for r in rows if passes_safe_filters(r)]
    for r in filtered:
        r["score"] = score_row(r)
    filtered.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    return filtered

def to_df(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [
        "score","chain","dex","tokenSymbol","priceUsd","pc_m5","pc_h1","vol_m5",
        "txns_m5","buys_m5","sells_m5","liqUsd","fdv","mcap","url"
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df

def do_scan():
    rows = scan(tuple(chains))
    candidates = build_candidates(rows)

    # update watchlist: беремо не тільки TOP_N, а всі, що проходять safe filters
    purge_expired(st.session_state.watch)
    upsert_watchlist(st.session_state.watch, candidates)

    st.session_state.last_scan_ts = time.time()

    # table of top candidates
    top = candidates[:TOP_N]
    return candidates, top

# ----------------------------
# UI layout
# ----------------------------
left, right = st.columns([2, 1], gap="large")

# Manual run at least once
if refresh_now or (st.session_state.last_scan_ts == 0):
    candidates, top = do_scan()
else:
    candidates = []
    top = []

# Auto mode loop (simple)
if auto:
    placeholder = st.empty()
    while True:
        candidates, top = do_scan()

        # build alerts NOW based on watchlist snapshot + cooldown
        alerts_now = build_alerts(st.session_state.watch, min_score=min_score)
        watch_rows = snapshot_watchlist(st.session_state.watch)

        with placeholder.container():
            l, r = st.columns([2, 1], gap="large")

            with l:
                st.subheader("Топ кандидати (SAFE)")
                df_top = to_df(top)
                if df_top.empty:
                    st.info("Немає кандидатів під правила прямо зараз.")
                else:
                    st.dataframe(df_top, use_container_width=True, height=520)

            with r:
                st.subheader("ALERTS (живі сигнали)")
                if len(alerts_now) == 0:
                    st.write("Поки тиша.")
                else:
                    st.warning(f"🚨 Нових сигналів: {len(alerts_now)}")
                    df_a = to_df(alerts_now)
                    st.dataframe(df_a, use_container_width=True, height=260)

                st.subheader("WATCHLIST (15 хв)")
                df_w = to_df(watch_rows[:30])
                if df_w.empty:
                    st.write("Порожньо.")
                else:
                    st.dataframe(df_w, use_container_width=True, height=260)

                st.caption(f"Останній скан: {time.strftime('%H:%M:%S')}")

        time.sleep(interval)
else:
    # Non-auto view
    with left:
        st.subheader("Топ кандидати (SAFE)")
        df_top = to_df(top)
        if df_top.empty:
            st.info("Немає кандидатів під правила прямо зараз.")
        else:
            st.dataframe(df_top, use_container_width=True, height=520)

    with right:
        # show alerts from current watchlist (will respect cooldown)
        alerts_now = build_alerts(st.session_state.watch, min_score=min_score)
        watch_rows = snapshot_watchlist(st.session_state.watch)

        st.subheader("ALERTS (живі сигнали)")
        if len(alerts_now) == 0:
            st.write("Поки тиша.")
        else:
            st.warning(f"🚨 Нових сигналів: {len(alerts_now)}")
            st.dataframe(to_df(alerts_now), use_container_width=True, height=260)

        st.subheader("WATCHLIST (15 хв)")
        if len(watch_rows) == 0:
            st.write("Порожньо.")
        else:
            st.dataframe(to_df(watch_rows[:30]), use_container_width=True, height=260)

        if st.session_state.last_scan_ts:
            st.caption(f"Останній скан: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_scan_ts))}")
