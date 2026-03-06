# app.py — DEX Scout
# base: v0.4.11 restored logic
# fixes: supabase persistence, monitoring stability, autoscan engine

import os
import re
import csv
import io
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from contextlib import contextmanager

import requests
import streamlit as st
import pandas as pd

VERSION = "0.5.5"

DEX_BASE = "https://api.dexscreener.com"

DATA_DIR = "data"

PORTFOLIO_CSV = os.path.join(DATA_DIR, "portfolio.csv")
MONITORING_CSV = os.path.join(DATA_DIR, "monitoring.csv")
MON_HISTORY_CSV = os.path.join(DATA_DIR, "monitoring_history.csv")


# =============================
# secrets helper
# =============================

def _get_secret(name: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name) or default)
    except Exception:
        pass

    return str(os.environ.get(name, default) or default)


SUPABASE_URL = _get_secret("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = _get_secret("SUPABASE_SERVICE_ROLE_KEY", "").strip()

SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


# =============================
# Supabase storage layer
# =============================

def _sb_headers() -> Dict[str, str]:

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _sb_table_url(table: str) -> str:

    return f"{SUPABASE_URL}/rest/v1/{table}"


def sb_get_storage(key: str) -> Optional[str]:

    if not SUPABASE_ENABLED:
        return None

    try:

        url = _sb_table_url("app_storage")

        params = {
            "select": "content",
            "key": f"eq.{key}"
        }

        r = requests.get(url, headers=_sb_headers(), params=params, timeout=12)

        if r.status_code == 404:
            return None

        r.raise_for_status()

        data = r.json() or []

        if not data:
            return None

        return data[0].get("content")

    except Exception:
        return None


def sb_put_storage(key: str, content: str) -> bool:

    if not SUPABASE_ENABLED:
        return False

    try:

        url = _sb_table_url("app_storage")

        payload = [{"key": key, "content": content}]

        headers = _sb_headers().copy()
        headers["Prefer"] = "resolution=merge-duplicates"

        r = requests.post(url, headers=headers, json=payload, timeout=15)

        r.raise_for_status()

        return True

    except Exception:

        return False


def storage_key_for_path(path: str) -> str:

    return os.path.basename(path)


# =============================
# file lock
# =============================

@contextmanager
def file_lock(lock_path: str, timeout_sec: int = 8):

    start = time.time()

    while True:

        try:

            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break

        except FileExistsError:

            if time.time() - start >= timeout_sec:
                raise RuntimeError("Lock timeout")

            time.sleep(0.08)

    try:
        yield
    finally:

        try:
            os.remove(lock_path)
        except Exception:
            pass


# =============================
# CSV helpers
# =============================

def _csv_from_string(content: str) -> List[Dict[str, Any]]:

    if not content.strip():
        return []

    sio = io.StringIO(content)

    rdr = csv.DictReader(sio)

    return [row for row in rdr]


def _csv_to_string(rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:

    sio = io.StringIO()

    w = csv.DictWriter(sio, fieldnames=fieldnames)

    w.writeheader()

    for r in rows:

        out = {k: r.get(k, "") for k in fieldnames}

        w.writerow(out)

    return sio.getvalue()


# =============================
# storage
# =============================

def load_csv(path: str) -> List[Dict[str, Any]]:

    if SUPABASE_ENABLED:

        key = storage_key_for_path(path)

        content = sb_get_storage(key)

        if content is not None:

            try:
                return _csv_from_string(content)
            except Exception:
                pass

    if not os.path.exists(path):
        return []

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for row in csv.DictReader(f):
            rows.append(row)

    return rows


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):

    os.makedirs(DATA_DIR, exist_ok=True)

    lockp = path + ".lock"

    with file_lock(lockp):

        with open(path, "w", encoding="utf-8", newline="") as f:

            w = csv.DictWriter(f, fieldnames=fieldnames)

            w.writeheader()

            for r in rows:

                out = {k: r.get(k, "") for k in fieldnames}

                w.writerow(out)

    if SUPABASE_ENABLED:

        key = storage_key_for_path(path)

        content = _csv_to_string(rows, fieldnames)

        sb_put_storage(key, content)


def append_csv(path: str, row: Dict[str, Any], fieldnames: List[str]):

    rows = load_csv(path)

    rows.append({k: row.get(k, "") for k in fieldnames})

    save_csv(path, rows, fieldnames)


# =============================
# utils
# =============================

def safe_get(d: Dict[str, Any], *path, default=None):

    cur = d

    for p in path:

        if not isinstance(cur, dict) or p not in cur:
            return default

        cur = cur[p]

    return cur


def parse_float(x, default=0.0):

    try:
        return float(x)
    except Exception:
        return default


def now_utc_str():

    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    # =============================
# scoring
# =============================

def score_pair(p):

    if not p:
        return 0.0

    liq = parse_float(safe_get(p, "liquidity", "usd", default=0))
    vol24 = parse_float(safe_get(p, "volume", "h24", default=0))
    vol5 = parse_float(safe_get(p, "volume", "m5", default=0))

    pc5 = parse_float(safe_get(p, "priceChange", "m5", default=0))
    pc1h = parse_float(safe_get(p, "priceChange", "h1", default=0))

    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0))
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0))

    trades5 = buys5 + sells5

    s = 0

    s += min(liq / 1000, 380)
    s += min(vol24 / 10000, 280)
    s += min(vol5 / 2000, 220)

    s += min(trades5 * 2, 140)

    s += max(min(pc1h, 90), -90) * 0.25
    s += max(min(pc5, 40), -40) * 0.15

    return round(s, 2)


def build_trade_hint(p):

    if not p:
        return "NO DATA", []

    liq = parse_float(safe_get(p, "liquidity", "usd", default=0))
    vol24 = parse_float(safe_get(p, "volume", "h24", default=0))
    vol5 = parse_float(safe_get(p, "volume", "m5", default=0))

    pc5 = parse_float(safe_get(p, "priceChange", "m5", default=0))
    pc1h = parse_float(safe_get(p, "priceChange", "h1", default=0))

    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0))
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0))

    trades5 = buys5 + sells5

    tags = []

    if liq >= 250000:
        tags.append("Liquidity: High")
    elif liq >= 60000:
        tags.append("Liquidity: Medium")
    else:
        tags.append("Liquidity: Low")

    if trades5 >= 60:
        tags.append("Flow: Hot")
    elif trades5 >= 25:
        tags.append("Flow: Active")
    else:
        tags.append("Flow: Thin")

    decision = "NO ENTRY"

    if liq >= 30000 and vol24 >= 20000 and trades5 >= 12 and sells5 >= 3:

        if pc1h > 6 and vol5 > 2500 and pc5 >= -3:
            decision = "ENTRY (scalp)"
        else:
            decision = "WATCH / WAIT"

    return decision, tags


# =============================
# dex api
# =============================

@st.cache_data(ttl=25)
def fetch_latest_pairs_for_query(q):

    url = f"{DEX_BASE}/latest/dex/search"

    r = requests.get(url, params={"q": q}, timeout=20)

    r.raise_for_status()

    data = r.json()

    return data.get("pairs", [])


# =============================
# monitoring logic
# =============================

MON_FIELDS = [
    "ts_added",
    "chain",
    "base_symbol",
    "base_addr",
    "pair_addr",
    "score_init",
    "active",
]


def load_monitoring():

    rows = load_csv(MONITORING_CSV)

    for r in rows:

        for k in MON_FIELDS:

            if k not in r:
                r[k] = ""

    return rows


def save_monitoring(rows):

    save_csv(MONITORING_CSV, rows, MON_FIELDS)


def add_to_monitoring(p, score):

    rows = load_monitoring()

    rows.append({

        "ts_added": now_utc_str(),
        "chain": p.get("chainId"),
        "base_symbol": safe_get(p, "baseToken", "symbol", default=""),
        "base_addr": safe_get(p, "baseToken", "address", default=""),
        "pair_addr": p.get("pairAddress"),
        "score_init": score,
        "active": "1",

    })

    save_monitoring(rows)


# =============================
# monitoring page
# =============================

def page_monitoring():

    st.title("Monitoring")

    rows = load_monitoring()

    active = [r for r in rows if r.get("active") == "1"]

    if not active:

        st.info("Monitoring empty")

        return

    for r in active:

        st.markdown("---")

        st.subheader(r["base_symbol"])

        st.write("Added:", r["ts_added"])

        st.code(r["base_addr"])


# =============================
# scout page
# =============================

def page_scout():

    st.title("Scout")

    seeds = ["meme", "ai", "launch"]

    pairs = []

    for s in seeds:

        try:

            pairs.extend(fetch_latest_pairs_for_query(s))

        except Exception:

            pass

    ranked = []

    for p in pairs:

        s = score_pair(p)

        decision, tags = build_trade_hint(p)

        if decision == "NO ENTRY":
            continue

        ranked.append((s, decision, tags, p))

    ranked.sort(reverse=True)

    for s, decision, tags, p in ranked[:20]:

        st.markdown("---")

        base = safe_get(p, "baseToken", "symbol")

        st.subheader(base)

        st.write("Score:", s)

        st.write("Decision:", decision)

        if st.button("Add to monitoring", key=base):

            add_to_monitoring(p, s)

            st.success("Added")


# =============================
# main
# =============================

def main():

    st.set_page_config(page_title="DEX Scout", layout="wide")

    page = st.sidebar.radio(

        "Page",
        ["Scout", "Monitoring"]

    )

    if page == "Scout":

        page_scout()

    else:

        page_monitoring()


if __name__ == "__main__":

    main()
