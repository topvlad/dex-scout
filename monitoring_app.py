# monitoring_app.py — DEX Scout Monitoring (AUTO)
# Compatible with app.py v0.3.7
# Purpose:
# - Track WATCH candidates over time
# - Auto-drop dying tokens
# - Rank which ones are WORTH monitoring further
#
# This app does NOT search new tokens.
# It ONLY works with data/monitoring.csv

import os
import csv
import time
from datetime import datetime
from typing import Dict, Any, List

import requests
import streamlit as st

VERSION = "0.1"
DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
MONITORING_CSV = os.path.join(DATA_DIR, "monitoring.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "monitoring_history.csv")


# -----------------------------
# Utils
# -----------------------------
def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


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


# -----------------------------
# Storage
# -----------------------------
def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(MONITORING_CSV):
        with open(MONITORING_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ts_added",
                    "chain",
                    "base_symbol",
                    "base_addr",
                    "pair_addr",
                    "score_init",
                    "liq_init",
                    "vol24_init",
                    "vol5_init",
                    "active",
                ],
            )
            w.writeheader()

    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ts",
                    "base_addr",
                    "liq",
                    "vol24",
                    "vol5",
                    "pc1h",
                    "pc5",
                ],
            )
            w.writeheader()


def load_monitoring() -> List[Dict[str, Any]]:
    ensure_storage()
    rows = []
    with open(MONITORING_CSV, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def save_monitoring(rows: List[Dict[str, Any]]):
    with open(MONITORING_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_history(row: Dict[str, Any]):
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writerow(row)


# -----------------------------
# API
# -----------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_best_pair(chain: str, base_addr: str) -> Dict[str, Any]:
    url = f"{DEX_BASE}/token-pairs/v1/{chain}/{base_addr}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    pairs = r.json() or []

    if not pairs:
        return {}

    def key(p):
        liq = safe_float(p.get("liquidity", {}).get("usd"))
        vol = safe_float(p.get("volume", {}).get("h24"))
        return (liq, vol)

    pairs.sort(key=key, reverse=True)
    return pairs[0]


# -----------------------------
# Core logic
# -----------------------------
def monitoring_score(liq, vol24, vol5, pc1h, pc5) -> float:
    """
    This score answers:
    'Is this token worth surviving WAIT?'
    """
    s = 0.0
    s += min(liq / 2000, 200)
    s += min(vol24 / 8000, 200)
    s += min(vol5 / 1500, 200)
    s += max(min(pc1h, 30), -30) * 2
    s += max(min(pc5, 20), -20) * 1.5
    return round(s, 2)


def should_drop(liq, vol24, vol5, pc1h) -> bool:
    """
    Hard auto-drop rules (NO MERCY)
    """
    if liq < 3000:
        return True
    if vol24 < 4000:
        return True
    if vol5 < 150:
        return True
    if pc1h < -12:
        return True
    return False


def status_label(score: float) -> str:
    if score >= 520:
        return "🔥 HOT (close to entry)"
    if score >= 380:
        return "🟡 WORTH WATCHING"
    return "🪫 WEAK (likely fade)"


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(page_title="DEX Scout — Monitoring", layout="wide")

    st.title("DEX Scout — Monitoring")
    st.caption(f"Version {VERSION} • AUTO refresh")

    rows = load_monitoring()
    active = [r for r in rows if r.get("active") == "1"]

    if not active:
        st.info("Monitoring list is empty. Add WATCH tokens from Scout.")
        return

    ranked = []

    for r in active:
        chain = r["chain"]
        base_addr = r["base_addr"]
        base_symbol = r["base_symbol"]

        try:
            p = fetch_best_pair(chain, base_addr)
        except Exception:
            continue

        liq = safe_float(p.get("liquidity", {}).get("usd"))
        vol24 = safe_float(p.get("volume", {}).get("h24"))
        vol5 = safe_float(p.get("volume", {}).get("m5"))
        pc1h = safe_float(p.get("priceChange", {}).get("h1"))
        pc5 = safe_float(p.get("priceChange", {}).get("m5"))

        score = monitoring_score(liq, vol24, vol5, pc1h, pc5)

        append_history(
            {
                "ts": now_utc(),
                "base_addr": base_addr,
                "liq": liq,
                "vol24": vol24,
                "vol5": vol5,
                "pc1h": pc1h,
                "pc5": pc5,
            }
        )

        drop = should_drop(liq, vol24, vol5, pc1h)

        ranked.append(
            {
                "symbol": base_symbol,
                "chain": chain,
                "base_addr": base_addr,
                "liq": liq,
                "vol24": vol24,
                "vol5": vol5,
                "pc1h": pc1h,
                "pc5": pc5,
                "score": score,
                "status": status_label(score),
                "drop": drop,
            }
        )

    # Auto-drop dead ones
    changed = False
    for r in rows:
        for x in ranked:
            if r.get("base_addr") == x["base_addr"] and x["drop"]:
                r["active"] = "0"
                changed = True

    if changed:
        save_monitoring(rows)

    ranked = [x for x in ranked if not x["drop"]]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    st.metric("Active monitoring", len(ranked))

    st.markdown("---")

    for x in ranked:
        st.subheader(f"{x['symbol']} ({x['chain']})")
        st.write(f"**Status:** {x['status']}")
        st.write(f"**Score:** {x['score']}")
        st.write(f"**Liquidity:** {fmt_usd(x['liq'])}")
        st.write(f"**Vol24:** {fmt_usd(x['vol24'])}")
        st.write(f"**Vol5:** {fmt_usd(x['vol5'])}")
        st.write(f"**Δ 1h:** {fmt_pct(x['pc1h'])}")
        st.write(f"**Δ 5m:** {fmt_pct(x['pc5'])}")
        st.markdown("---")

    st.caption("AUTO mode: dead tokens are removed automatically.")


if __name__ == "__main__":
    main()
