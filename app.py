# app.py — DEX Scout v0.4.12
# ядро: Scout → Monitoring → Archive (+ Portfolio)
# - Scout: показує тільки re-eligible токени (не в Monitoring active, не в Portfolio active), і НЕ показує "NO ENTRY"
# - Monitoring: тільки WATCH/WAIT. Сортування: priority → momentum → time since added
# - Archive: автоматична архівація (опційно) по мін. score та/або по "NO ENTRY"
# - BSC + Solana, swap routing: PancakeSwap (BSC) + Jupiter (Solana)
# - Address handling: Solana mint addresses are case-sensitive – NEVER lower() them
#
# ✅ Persistence fix:
# This version supports two storage backends:
# 1) Supabase (recommended for Streamlit Cloud persistence) if SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set
# 2) Local CSV fallback (works locally, but Streamlit Cloud can lose files on redeploy)
#
# Supabase tables expected (create once in Supabase SQL editor):
# - portfolio
# - monitoring
# - monitoring_history
#
# Columns must match PORTFOLIO_FIELDS / MON_FIELDS / HIST_FIELDS below (all text is OK).
#
# ⚠️ Safety note: Це НЕ ончейн-аудит. Для Solana – завжди дивись JupShield warnings перед swap.

import os
import re
import csv
import io
import time
import json
import random
import hashlib
import inspect
from urllib.parse import quote_plus
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter
from contextlib import contextmanager

import requests
import streamlit as st
import pandas as pd


st.set_page_config(page_title="DEX Scout", layout="wide")


def _maybe_autorefresh(interval_ms: int, key: str):
    """Best-effort autorefresh without extra deps."""
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
        st_autorefresh(interval=interval_ms, key=key)
        return
    except Exception:
        pass
    try:
        st.autorefresh(interval=interval_ms, key=key)  # type: ignore
        return
    except Exception:
        pass


def maybe_safe_auto_refresh(enabled: bool, interval_sec: int = 60) -> None:
    """
    Session-safe auto-refresh.
    Only reruns if user enabled refresh and interval elapsed.
    """
    if not enabled:
        return

    interval_sec = max(60, int(interval_sec))
    _maybe_autorefresh(interval_ms=interval_sec * 1000, key="dex_scout_autorefresh")

    now = time.time()
    last_refresh = st.session_state.get("last_refresh_ts")
    if last_refresh is None:
        st.session_state["last_refresh_ts"] = now
        return

    if (now - float(last_refresh)) >= interval_sec:
        st.session_state["last_refresh_ts"] = now
        st.rerun()


VERSION = "v0.5.6-entry-engine-v1"
DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
SMART_WALLET_FILE = os.path.join(DATA_DIR, "smart_wallets.json")
ENTRY_MODE = "aggressive"
TEMP_DISABLE_BEST_PAIR = False
WORKER_FAST_MODE = os.getenv("DEX_SCOUT_WORKER_MODE", "0") == "1"
USE_HEAVY_MIGRATION_CHECK = not WORKER_FAST_MODE

SCAN_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300

MEME_KEYWORDS = [
    "dog",
    "cat",
    "pepe",
    "inu",
    "elon",
    "moon",
    "pump",
    "bonk",
    "frog",
]

PORTFOLIO_CSV = os.path.join(DATA_DIR, "portfolio.csv")
MONITORING_CSV = os.path.join(DATA_DIR, "monitoring.csv")
MON_HISTORY_CSV = os.path.join(DATA_DIR, "monitoring_history.csv")
TG_STATE_FILE = os.path.join(DATA_DIR, "tg_state.json")
TG_STATE_KEY = "tg_state.json"
SUPPRESSED_KEY = "suppressed_tokens.json"


def request_rerun() -> None:
    st.session_state["_rerun_flag"] = True


def safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json.dumps({"error": "serialization_failed"})


def log_error(err: Exception):
    try:
        print(f"[scanner-error] {type(err).__name__}: {err}")
    except Exception:
        pass


def debug_log(msg: str):
    line = f"{now_utc_str()} | {msg}"
    print(line, flush=True)

    if WORKER_FAST_MODE:
        return

    try:
        if "debug_log" not in st.session_state:
            st.session_state["debug_log"] = []
        st.session_state["debug_log"].append(line)
        st.session_state["debug_log"] = st.session_state["debug_log"][-50:]
    except Exception:
        return


def load_smart_wallets() -> Dict[str, Any]:
    if os.path.exists(SMART_WALLET_FILE):
        try:
            return json.loads(Path(SMART_WALLET_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_smart_wallets(data: Dict[str, Any]):
    try:
        Path(SMART_WALLET_FILE).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


# =============================
# Supabase (optional persistence) – v0.4.11 compatible
# =============================
# Якщо у Streamlit Cloud є secrets:
#   SUPABASE_URL
#   SUPABASE_SERVICE_ROLE_KEY
# (optional) SUPABASE_ANON_KEY
#
# то CSV зберігаються у Supabase в таблиці public.app_storage як текст:
#
#   create table if not exists public.app_storage (
#     key text primary key,
#     content text not null,
#     updated_at timestamptz not null default now()
#   );
#
# Важливо: ми НЕ переносимо логіку в Postgres-таблиці – лише зберігаємо CSV як текст.
# Так дані й архіви не зникають між деплоями Streamlit Cloud.

def _get_secret(name: str, default: str = "") -> str:
    env_val = os.environ.get(name)
    if env_val:
        return str(env_val)

    if WORKER_FAST_MODE:
        return str(default)

    try:
        return str(st.secrets.get(name) or default)
    except Exception:
        return str(default)


def get_tg_token() -> str:
    return _get_secret("TG_BOT_TOKEN", "").strip() or _get_secret("TELEGRAM_BOT_TOKEN", "").strip()

def get_tg_chat_id() -> str:
    return _get_secret("TG_CHAT_ID", "").strip() or _get_secret("TELEGRAM_CHAT_ID", "").strip()

SUPABASE_URL = _get_secret("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = _get_secret("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_ANON_KEY = _get_secret("SUPABASE_ANON_KEY", "").strip()

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

def _sb_ok() -> bool:
    return bool(USE_SUPABASE)

def _sb_headers() -> Dict[str, str]:
    # Use service role key for server-side storage
    if not SUPABASE_SERVICE_ROLE_KEY:
        return {}
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _sb_table_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"

def sb_get_storage(key: str) -> Optional[str]:
    """
    Returns content for key from public.app_storage, or None if not found.
    """
    if not USE_SUPABASE:
        return None
    try:
        url = _sb_table_url("app_storage")
        params = {"select": "content", "key": f"eq.{key}"}
        r = requests.get(url, headers=_sb_headers(), params=params, timeout=12)
        if r.status_code == 404:
            debug_log(f"supabase_read_404 key={key}")
            return None
        r.raise_for_status()
        data = r.json() or []
        if not data:
            debug_log(f"supabase_read_empty key={key}")
            return None
        content = (data[0].get("content") or "")
        debug_log(f"supabase_read_ok key={key} bytes={len(content)}")
        return content
    except Exception as e:
        debug_log(f"supabase_read_exception key={key} err={type(e).__name__}:{e}")
        return None

def sb_put_storage(key: str, content: str) -> bool:
    try:
        if not _sb_ok():
            return False

        url = f"{SUPABASE_URL}/rest/v1/app_storage"
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        payload = {
            "key": key,
            "content": content,
        }

        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code not in (200, 201):
            err_preview = (res.text or "").strip().replace("\n", " ")[:200]
            debug_log(f"supabase_write_failed key={key} status={res.status_code} body={err_preview}")
            return False

        debug_log(f"supabase_write_ok key={key} bytes={len(content)}")
        return True
    except Exception as e:
        debug_log(f"supabase_write_exception key={key} err={type(e).__name__}:{e}")
        return False

def storage_key_for_path(path: str) -> str:
    # keep it stable and short: data/monitoring.csv -> monitoring.csv
    base = os.path.basename(path)
    return base or path


def local_fallback_path(path: str) -> str:
    ensure_storage()
    return path


# Storage + lock (CSV fallback)
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
            if (time.time() - start) >= timeout_sec:
                raise RuntimeError(f"Lock timeout: {lock_path}")
            time.sleep(0.08)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass


def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)

    # portfolio
    if not os.path.exists(PORTFOLIO_CSV):
        with open(PORTFOLIO_CSV, "w", newline="", encoding="utf-8") as f:
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

    # monitoring
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
                    "ts_archived",
                    "archived_reason",
                    "last_score",
                    "last_decision",
                    "priority_score",
                    "last_decay_ts",
                    "decay_hits",
                    "entry_state",
                    "revisit_count",
                    "revisit_after_ts",
                    "last_revisit_ts",
                ],
            )
            w.writeheader()

    # monitoring history snapshots
    if not os.path.exists(MON_HISTORY_CSV):
        with open(MON_HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ts_utc",
                    "chain",
                    "base_symbol",
                    "base_addr",
                    "pair_addr",
                    "dex",
                    "quote_symbol",
                    "price_usd",
                    "liq_usd",
                    "vol24_usd",
                    "vol5_usd",
                    "pc1h",
                    "pc5",
                    "score_live",
                    "decision",
                ],
            )
            w.writeheader()


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_ts(ts: Any) -> Optional[datetime]:
    raw = str(ts or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


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


def load_csv(path: str, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    ensure_storage()
    key = storage_key_for_path(path)
    del fields  # compatibility

    if _sb_ok():
        content = sb_get_storage(key)
        if content:
            try:
                st.session_state[f"_storage_source_{key}"] = "supabase"
                return _csv_from_string(content)
            except Exception:
                debug_log(f"corrupt_supabase_csv key={key}")

    fallback = local_fallback_path(path)
    if os.path.exists(fallback):
        try:
            with open(fallback, "r", newline="", encoding="utf-8") as f:
                debug_log(f"using_local_fallback key={key}")
                st.session_state[f"_storage_source_{key}"] = "local_fallback"
                return list(csv.DictReader(f))
        except Exception as e:
            debug_log(f"local_fallback_read_failed key={key} err={type(e).__name__}:{e}")

    st.session_state[f"_storage_source_{key}"] = "empty"
    return []


def storage_read_text(key: str, default: str = "") -> str:
    ensure_storage()
    key = storage_key_for_path(key)

    if _sb_ok():
        content = sb_get_storage(key)
        if content is not None:
            return str(content)

    fallback = local_fallback_path(os.path.join(DATA_DIR, key))
    if os.path.exists(fallback):
        try:
            return Path(fallback).read_text(encoding="utf-8")
        except Exception:
            return str(default)
    return str(default)


def storage_write_text(key: str, text: str) -> None:
    ensure_storage()
    key = storage_key_for_path(key)
    content = str(text or "")

    fallback = local_fallback_path(os.path.join(DATA_DIR, key))
    try:
        lockp = fallback + ".lock"
        with file_lock(lockp):
            Path(fallback).write_text(content, encoding="utf-8")
    except Exception as e:
        debug_log(f"local_fallback_write_failed key={key} err={type(e).__name__}:{e}")

    if _sb_ok():
        sb_put_storage(key, content)


@st.cache_data(ttl=60, show_spinner=False)
def load_monitoring_rows_cached() -> List[Dict[str, Any]]:
    return load_csv(MONITORING_CSV, MON_FIELDS)


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_storage()
    key = storage_key_for_path(path)
    content = _csv_to_string(rows, fieldnames)

    fallback = local_fallback_path(path)
    try:
        lockp = fallback + ".lock"
        with file_lock(lockp):
            with open(fallback, "w", newline="", encoding="utf-8") as f:
                f.write(content)
    except Exception as e:
        debug_log(f"local_fallback_write_failed key={key} err={type(e).__name__}:{e}")

    if _sb_ok():
        ok = sb_put_storage(key, content)
        if ok:
            check = sb_get_storage(key)
            if check:
                debug_log(f"supabase_store_verified key={key}")
                st.session_state["_save_badge"] = "💾 saved (supabase verified)"
            else:
                debug_log(f"supabase_store_unverified key={key}")
                st.session_state["_save_badge"] = "⚠️ saved local, supabase unverified"
        else:
            debug_log(f"supabase_store_failed_local_kept key={key}")
            st.session_state["_save_badge"] = "⚠️ saved local, supabase write failed"
    else:
        st.session_state["_save_badge"] = "💾 saved (local)"
    load_monitoring_rows_cached.clear()


def backup_csv_snapshot(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    if not _sb_ok():
        return
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        key = storage_key_for_path(path)
        backup_key = f"backup/{ts}_{key}"
        content = _csv_to_string(rows, fieldnames)
        sb_put_storage(backup_key, content)
    except Exception:
        pass


def append_csv(path: str, row: Dict[str, Any], fieldnames: List[str]):
    # For Supabase we do read-modify-write (safe for low concurrency).
    rows = load_csv(path)
    rows.append({k: row.get(k, "") for k in fieldnames})
    save_csv(path, rows, fieldnames)



# =============================
# Streamlit compatibility helpers
# =============================
def hkey(*parts: str, n: int = 10) -> str:
    raw = "|".join([p or "" for p in parts])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:n]


def link_button(label: str, url: str, use_container_width: bool = True, key: Optional[str] = None):
    """
    Streamlit's st.link_button signature differs across versions.
    This wrapper avoids TypeError by:
    - trying without key
    - then with key
    - falling back to HTML if needed
    """
    if not url:
        return
    if hasattr(st, "link_button"):
        try:
            st.link_button(label, url, use_container_width=use_container_width)
            return
        except TypeError:
            try:
                if key is not None:
                    st.link_button(label, url, use_container_width=use_container_width, key=key)
                else:
                    st.link_button(label, url, use_container_width=use_container_width)
                return
            except Exception:
                pass
        except Exception:
            pass

    st.markdown(
        f"""
        <a href="{url}" target="_blank" style="text-decoration:none;">
          <div style="
            display:inline-block;
            padding:10px 14px;
            border-radius:10px;
            border:1px solid rgba(0,0,0,0.22);
            font-weight:800;
            width:100%;
            text-align:center;
          ">{label}</div>
        </a>
        """,
        unsafe_allow_html=True,
    )


# =============================
# Constants / presets
# =============================
CHAIN_DEX_PRESETS = {
    "bsc": ["pancakeswap", "thena", "biswap", "apeswap", "babyswap", "mdex", "woofi"],
    "solana": ["raydium", "orca", "meteora", "pumpfun", "pumpswap"],
}

DEX_RANK = {
    "solana": {
        "pumpfun": 1,
        "pumpswap": 1,
        "meteora": 2,
        "raydium": 3,
        "orca": 4,
    },
    "bsc": {
        "babyswap": 1,
        "biswap": 2,
        "apeswap": 2,
        "thena": 3,
        "pancakeswap": 4,
        "woofi": 4,
    },
}

ALLOWED_CHAINS = ("solana", "bsc")
CHAIN_MAP = {
    "ethereum": "eth",
    "eth": "eth",
    "bsc": "bsc",
    "binance": "bsc",
    "solana": "solana",
}


def normalize_chain_name(raw_chain: Any) -> str:
    value = str(raw_chain or "").strip().lower()
    return CHAIN_MAP.get(value, value)


NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

MAJORS_STABLES = {
    "BTC", "WBTC", "ETH", "WETH", "BNB", "WBNB", "SOL", "WSOL",
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDE", "USDS",
    "WAVAX", "WMATIC",
}
HARD_BLOCK_SYMBOLS = {"USDT", "USDC", "ETH", "BTC", "BNB"}
MIN_LIQ_USD = 10000
MIN_TXNS_5M = 5
MIN_VOL_5M = 1500
MAX_KEEP = int(os.getenv("SCANNER_MAX_KEEP", "80"))
MAX_DEX_SEARCH_PER_TERM = 20
INGEST_TARGET_KEEP = 200
DEX_SEARCH_TERMS = [
    "meme", "memecoin", "ai", "ai agent", "agentic", "launch", "new", "listing",
    "trending", "hot", "hype", "pump", "pepe", "trump", "inu", "dog", "cat",
    "frog", "shiba", "community", "cto", "community takeover", "takeover",
    "revival", "fairlaunch", "stealth", "presale", "airdrop", "claim", "points",
    "quest", "rewards", "wl", "whitelist", "relaunch", "rebrand", "migration",
    "upgrade", "depin", "rwa", "gaming", "gamefi", "esports", "defi", "dex",
    "perp", "perps", "leverage", "staking", "apr", "farming", "bridge",
    "crosschain", "multichain", "ecosystem", "partnership", "burn", "buyback",
    "revenue share", "pumpfun", "pumpswap", "launchpad",
]

DEFAULT_SEEDS = (
    "meme, memecoin, ai, ai agent, agentic, launch, new, listing, trending, hot, hype, pump, "
    "pepe, trump, inu, dog, cat, frog, shiba, "
    "community, cto, community takeover, takeover, revival, "
    "fairlaunch, stealth, presale, airdrop, claim, points, quest, rewards, wl, whitelist, "
    "v2, v3, v4, relaunch, rebrand, migration, upgrade, "
    "depin, rwa, gaming, gamefi, esports, "
    "defi, dex, perp, perps, leverage, staking, apr, farming, "
    "bridge, crosschain, multichain, ecosystem, partnership, "
    "burn, buyback, revenue share, "
    "pumpfun, pumpswap, launchpad"
)

PRESETS = {
    "Ultra Early (safer)": {
        "top_n": 25,
        "min_liq": 2000,
        "min_vol24": 5000,
        "min_trades_m5": 6,
        "min_sells_m5": 2,
        "max_imbalance": 18,
        "block_suspicious_names": True,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 20,
        "max_age_min": 14400,
        "hide_solana_unverified": True,
        "seed_k": 16,
        "dedupe_by_base": True,
        "seeds": DEFAULT_SEEDS,
    },
    "Balanced (default)": {
        "top_n": 20,
        "min_liq": 8000,
        "min_vol24": 25000,
        "min_trades_m5": 12,
        "min_sells_m5": 3,
        "max_imbalance": 14,
        "block_suspicious_names": True,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 30,
        "max_age_min": 43200,
        "hide_solana_unverified": True,
        "seed_k": 14,
        "dedupe_by_base": True,
        "seeds": DEFAULT_SEEDS,
    },
    "Wide Net (explore)": {
        "top_n": 35,
        "min_liq": 1000,
        "min_vol24": 5000,
        "min_trades_m5": 5,
        "min_sells_m5": 1,
        "max_imbalance": 22,
        "block_suspicious_names": False,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 10,
        "max_age_min": 86400,
        "hide_solana_unverified": True,
        "seed_k": 18,
        "dedupe_by_base": True,
        "seeds": DEFAULT_SEEDS,
    },
    "Momentum (hot)": {
        "top_n": 25,
        "min_liq": 3000,
        "min_vol24": 12000,
        "min_trades_m5": 18,
        "min_sells_m5": 4,
        "max_imbalance": 12,
        "block_suspicious_names": True,
        "block_majors": True,
        "enforce_age": True,
        "min_age_min": 20,
        "max_age_min": 28800,
        "hide_solana_unverified": True,
        "seed_k": 14,
        "dedupe_by_base": True,
        "seeds": DEFAULT_SEEDS,
    },
}

MONITORING_SORT_PENALTY_FLOOR = 120.0
MONITORING_REVIEW_THRESHOLD = 120.0
MONITORING_DEAD_THRESHOLD = 50.0
REASON_LABELS = {
    # "fake_pump": "Suspicious pump",  # disabled: old anti-rug kill-switch label
    "freezable": "Token can be frozen",
    "fake_activity": "Fake trading activity",
    "no_momentum": "No momentum",
    "low_liquidity": "Low liquidity",
    "early_momentum": "Early momentum",
    "breakout": "Breakout",
    "fallback_signal": "Weak signal",
}


# =============================
# Address helpers
# =============================
def addr_key(chain: str, addr: str) -> str:
    c = (chain or "").lower().strip()
    a = (addr or "").strip()
    if not a:
        return ""
    if c == "solana":
        return f"solana:{a}"
    return f"{c}:{a.lower()}"


def addr_store(chain: str, addr: str) -> str:
    c = (chain or "").lower().strip()
    a = (addr or "").strip()
    if not a:
        return ""
    if c == "solana":
        return a
    return a.lower()


def safe_get(d: Dict[str, Any], *path, default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def parse_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def pct_change(a: float, b: float) -> float:
    if not b:
        return 0.0
    return ((a - b) / b) * 100.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def is_freezable(p: Dict[str, Any]) -> bool:
    txt = str(p).lower()
    return "freezable" in txt or "freeze" in txt


def is_fake_activity(p: Dict[str, Any]) -> bool:
    buys = parse_float(safe_get(p, "txns", "h24", "buys", default=p.get("buys_h24", 0)), 0)
    sells = parse_float(safe_get(p, "txns", "h24", "sells", default=p.get("sells_h24", 0)), 0)
    makers = parse_float(safe_get(p, "txns", "h24", "makers", default=p.get("makers_h24", 0)), 0)

    # disabled: old fake activity blocker was too aggressive and marked almost all tokens as toxic
    # if buys <= 1 and sells <= 1 and makers <= 2:
    #     return True
    return False


def is_flat_chart(p: Dict[str, Any]) -> bool:
    pc1h = parse_float(safe_get(p, "priceChange", "h1", default=p.get("price_change_1h", 0)), 0)
    pc5m = parse_float(safe_get(p, "priceChange", "m5", default=p.get("price_change_5m", 0)), 0)

    if abs(pc1h) < 0.5 and abs(pc5m) < 0.2:
        return True
    return False


def is_toxic_token(p: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if is_freezable(p):
        reasons.append("freezable")
    # disabled: fake_activity hard-blocker (old anti-rug kill-switch)
    # if is_fake_activity(p):
    #     reasons.append("fake_activity")
    # flat chart is a soft penalty in scoring, not a toxic hard-flag
    return (len(reasons) > 0), reasons


def is_symbol_major_like(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    if not s:
        return False
    if s in MAJORS_STABLES:
        return True
    if s.startswith("W") and s[1:] in MAJORS_STABLES:
        return True
    if s in {"SOL", "ETH", "BTC", "BNB", "USDT", "USDC", "BUSD", "DAI"}:
        return True
    return False


def looks_like_quote_or_lp(p: Dict[str, Any]) -> bool:
    base_sym = str(safe_get(p, "baseToken", "symbol", default="") or "").strip().upper()
    quote_sym = str(safe_get(p, "quoteToken", "symbol", default="") or "").strip().upper()
    base_name = str(safe_get(p, "baseToken", "name", default="") or "").strip().lower()

    if is_symbol_major_like(base_sym):
        return True

    if "lp" in base_name or "liquidity" in base_name:
        return True

    if base_sym == quote_sym and base_sym:
        return True

    return False


def dedupe_by_symbol_family(rows: List[Dict[str, Any]], per_symbol_limit: int = 1) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        sym = str(safe_get(r, "baseToken", "symbol", default="") or r.get("base_symbol") or "").strip().upper()
        if not sym:
            continue
        buckets.setdefault(sym, []).append(r)

    out: List[Dict[str, Any]] = []
    for _sym, items in buckets.items():
        items.sort(
            key=lambda p: (
                parse_float(safe_get(p, "liquidity", "usd", default=0), 0.0),
                parse_float(safe_get(p, "volume", "h24", default=0), 0.0),
                score_pair(p),
            ),
            reverse=True,
        )
        out.extend(items[:per_symbol_limit])

    return out


def norm_addr(chain: str, addr: str) -> str:
    chain = (chain or "").lower().strip()
    addr = (addr or "").strip()
    if not addr:
        return ""
    if chain in {"bsc", "ethereum", "polygon", "arbitrum", "optimism", "base", "avalanche"}:
        return addr.lower()
    return addr


def token_key(chain: str, addr: str) -> str:
    c = (chain or "").lower().strip()
    a = norm_addr(c, addr)
    return f"{c}|{a}" if (c and a) else ""




def dedupe_tokens_by_address(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the highest-priority row per (chain, base address)."""
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        chain = (row.get("chainId") or row.get("chain") or "").strip().lower()
        base_addr = (
            safe_get(row, "baseToken", "address", default="")
            or row.get("base_addr")
            or row.get("address")
            or ""
        )
        key = token_key(chain, str(base_addr).strip())
        if not key:
            continue

        score = parse_float(row.get("priority_score", row.get("visible_score", 0.0)), 0.0)
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue

        prev_score = parse_float(prev.get("priority_score", prev.get("visible_score", 0.0)), 0.0)
        if score > prev_score:
            best[key] = row
    return list(best.values())


def dedupe_unified_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate unified token feed by normalized address key."""
    if not tokens:
        return []

    best: Dict[str, Dict[str, Any]] = {}
    for token in tokens:
        chain = (token.get("chain") or "").strip().lower()
        address = (token.get("address") or "").strip()
        key = token_key(chain, address)
        if not key:
            continue

        current = best.get(key)
        if current is None:
            best[key] = token
            continue

        liq = parse_float(token.get("liquidity", 0.0), 0.0)
        vol = parse_float(token.get("volume", 0.0), 0.0)
        cur_liq = parse_float(current.get("liquidity", 0.0), 0.0)
        cur_vol = parse_float(current.get("volume", 0.0), 0.0)
        if (liq, vol) > (cur_liq, cur_vol):
            best[key] = token

    return list(best.values())
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


def fmt_usd_delta(x: float) -> str:
    try:
        sign = "+" if x > 0 else ""
        return f"{sign}${x:,.0f}"
    except Exception:
        return "n/a"


# =============================
# HTTP with retry/backoff
# =============================
def _http_get_json(url: str, params: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> Any:
    last_err = None
    backoff = 0.7
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


# =============================
# API
# =============================
def fetch_dexscreener_latest(chain: str, limit: int = 40):
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        pairs = data.get("pairs") or []
        return pairs[:limit]
    except Exception:
        return []

def fetch_tokens_unified(chain: str, limit: int = 50) -> List[Dict[str, Any]]:
    chain = (chain or "").strip().lower()

    def _from_birdeye_solana() -> List[Dict[str, Any]]:
        try:
            url = "https://public-api.birdeye.so/defi/tokenlist"
            params = {
                "sort_by": "v24hUSD",
                "sort_type": "desc",
                "offset": 0,
                "limit": limit,
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return []
            data = r.json() or {}
            raw = safe_get(data, "data", "tokens", default=[]) or []
            out: List[Dict[str, Any]] = []
            for t in raw:
                addr = str(t.get("address") or "").strip()
                sym = str(t.get("symbol") or "NA").strip()
                if not addr:
                    continue
                out.append(
                    {
                        "symbol": sym,
                        "address": addr,
                        "price": parse_float(t.get("price", 0), 0.0),
                        "volume": parse_float(t.get("v24hUSD", 0), 0.0),
                        "liquidity": parse_float(t.get("liquidity", 0), 0.0),
                        "source": "birdeye",
                        "chain": "solana",
                    }
                )
            return out
        except Exception:
            return []

    def _from_dexscreener(chain_id: str) -> List[Dict[str, Any]]:
        try:
            pairs = fetch_dexscreener_trending(chain_id, limit=limit)
            out: List[Dict[str, Any]] = []
            for p in pairs:
                base = p.get("baseToken", {}) or {}
                addr = str(base.get("address") or "").strip()
                sym = str(base.get("symbol") or "NA").strip()
                if not addr:
                    continue
                out.append(
                    {
                        "symbol": sym,
                        "address": addr,
                        "price": parse_float(p.get("priceUsd"), 0.0),
                        "volume": parse_float(safe_get(p, "volume", "h24", default=0), 0.0),
                        "liquidity": parse_float(safe_get(p, "liquidity", "usd", default=0), 0.0),
                        "source": "dexscreener",
                        "chain": chain_id,
                    }
                )
            return out
        except Exception:
            return []

    if chain == "solana":
        sources = ["birdeye", "dexscreener"]
    else:
        sources = ["dexscreener"]

    merged: List[Dict[str, Any]] = []
    for source in sources:
        if source == "birdeye" and chain == "solana":
            merged.extend(_from_birdeye_solana())
        if source == "dexscreener":
            merged.extend(_from_dexscreener(chain))
    if merged:
        return dedupe_unified_tokens(merged)

    return []

@st.cache_data(ttl=30, show_spinner=False)
def fetch_dexscreener_trending(chain: str, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Fetch trending pairs from Dexscreener.
    """
    try:
        url = f"{DEX_BASE}/latest/dex/pairs/{chain}"
        data = _http_get_json(url, timeout=20)

        pairs = data.get("pairs") or []

        pairs.sort(
            key=lambda p: float(safe_get(p, "volume", "h24", default=0) or 0),
            reverse=True,
        )

        return pairs[:limit]

    except Exception:
        return []


def fetch_birdeye_pairs(limit: int = 50) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not BIRDEYE_ENABLED:
        return out
    for mint in birdeye_trending_solana(limit=limit):
        try:
            bp = best_pair_for_token_cached("solana", mint)
            if bp:
                out.append(bp)
        except Exception:
            continue
    return out

@st.cache_data(ttl=60, max_entries=500, show_spinner=False)
def fetch_latest_pairs_for_query(q: str) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/latest/dex/search"
    data = _http_get_json(url, params={"q": q.strip()}, timeout=20, max_retries=3)
    return data.get("pairs", []) or []


@st.cache_data(ttl=60, max_entries=500, show_spinner=False)
def fetch_dexscreener_search(term: str, limit: int = 20) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/latest/dex/search"
    data = _http_get_json(url, params={"q": term.strip()}, timeout=20, max_retries=3)
    pairs = data.get("pairs", []) or []
    return pairs[:limit]


@st.cache_data(ttl=60, max_entries=500, show_spinner=False)
def fetch_token_pairs(chain: str, token_address: str) -> List[Dict[str, Any]]:
    if not token_address:
        return []
    try:
        url = f"{DEX_BASE}/token-pairs/v1/{chain}/{token_address}"
        data = _http_get_json(url, params=None, timeout=8, max_retries=1)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        debug_log(f"fetch_token_pairs_fail chain={chain} token={token_address} err={type(e).__name__}:{e}")
        return []

@st.cache_data(ttl=60, max_entries=500)
def best_pair_for_token(chain: str, token_address: str) -> Optional[Dict[str, Any]]:
    if TEMP_DISABLE_BEST_PAIR:
        return None
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


@st.cache_data(ttl=45, max_entries=1000, show_spinner=False)
def best_pair_for_token_cached(chain: str, token_address: str) -> Optional[Dict[str, Any]]:
    return best_pair_for_token(chain, token_address)


def detect_narrative(pair: Dict[str, Any]) -> str:
    name = (pair.get("baseToken", {}).get("name") or "").lower()
    symbol = (pair.get("baseToken", {}).get("symbol") or "").lower()
    txt = f"{name} {symbol}"

    if any(x in txt for x in ["pepe", "frog", "bonk"]):
        return "frog_meme"
    if any(x in txt for x in ["dog", "inu", "shib"]):
        return "dog_meme"
    if any(x in txt for x in ["ai", "agent", "gpt"]):
        return "ai"
    if any(x in txt for x in ["pump", "moon", "100x"]):
        return "pump"
    return "other"


# =============================
# Safety / heuristics
# =============================
def is_name_suspicious(p: Dict[str, Any]) -> bool:
    sym = (safe_get(p, "baseToken", "symbol", default="") or "").strip()
    if not sym:
        return True
    if len(sym) > 40:
        return True
    return not bool(NAME_OK_RE.match(sym))


def is_major_or_stable(p: Dict[str, Any]) -> bool:
    base = (safe_get(p, "baseToken", "symbol", default="") or "").strip().upper()
    return base in MAJORS_STABLES


def pair_age_minutes(p: Dict[str, Any]) -> Optional[float]:
    ts = p.get("pairCreatedAt")
    if ts is None:
        return None
    try:
        ts = float(ts)
        created = ts / 1000.0 if ts > 10_000_000_000 else ts
        age_sec = max(0.0, time.time() - created)
        return age_sec / 60.0
    except Exception:
        return None


def meme_token_score(token: Dict[str, Any]) -> int:
    score = 0

    name = (safe_get(token, "baseToken", "name", default="") or token.get("name") or "").lower()
    symbol = (safe_get(token, "baseToken", "symbol", default="") or token.get("symbol") or "").lower()

    meme_keywords = [
        "dog", "inu", "cat", "pepe", "frog",
        "elon", "ai", "meme", "moon", "baby"
    ]

    for k in meme_keywords:
        if k in name or k in symbol:
            score += 10

    liq = parse_float(safe_get(token, "liquidity", "usd", default=token.get("liquidity", 0)), 0.0)
    if liq > 50000:
        score += 20
    elif liq > 10000:
        score += 10

    vol = parse_float(safe_get(token, "volume", "h24", default=token.get("volume", 0)), 0.0)
    if vol > 100000:
        score += 20
    elif vol > 20000:
        score += 10

    age = token.get("pairAge")
    if age is None:
        age_min = pair_age_minutes(token)
        age = (age_min / 60.0) if age_min is not None else 0
    age = parse_float(age, 0.0)
    if 0 < age < 24:
        score += 10

    return min(score, 100)


def meme_score(name: str) -> int:
    nm = (name or "").lower()
    for word in MEME_KEYWORDS:
        if word in nm:
            return 1
    return 0


def scout_priority(token: Dict[str, Any]) -> int:
    score = 0
    liq = parse_float(safe_get(token, "liquidity", "usd", default=token.get("liquidity", 0)), 0.0)
    vol5 = parse_float(safe_get(token, "volume", "m5", default=token.get("volume_5m", 0)), 0.0)
    buys_5m = int(safe_get(token, "txns", "m5", "buys", default=token.get("buys_5m", 0)) or 0)
    sells_5m = int(safe_get(token, "txns", "m5", "sells", default=token.get("sells_5m", 0)) or 0)
    age_minutes = pair_age_minutes(token)
    age_minutes = parse_float(age_minutes if age_minutes is not None else token.get("age_minutes", 999999), 999999.0)
    name = str(safe_get(token, "baseToken", "name", default=token.get("name", "")) or "")

    if liq > 50000:
        score += 2
    if vol5 > liq * 0.4:
        score += 2
    if age_minutes < 120:
        score += 2
    if buys_5m > sells_5m:
        score += 1
    score += meme_score(name)
    return score


def detect_smart_money(pair: Optional[Dict[str, Any]]) -> bool:
    if not pair:
        return False

    txns = safe_get(pair, "txns", "h1", default={})

    buys = parse_float(txns.get("buys", 0))
    sells = parse_float(txns.get("sells", 0))

    vol = parse_float(
        safe_get(pair, "volume", "h24", default=0)
    )

    liq = parse_float(
        safe_get(pair, "liquidity", "usd", default=0)
    )

    if buys > sells * 2 and vol > liq:
        return True

    return False


def detect_smart_wallet_activity(pair: Dict[str, Any]) -> int:
    return 1 if detect_smart_money(pair) else 0


def dex_rank(chain: str, dex_id: str) -> int:
    chain = (chain or "").lower().strip()
    dex_id = (dex_id or "").lower().strip()
    return int((DEX_RANK.get(chain) or {}).get(dex_id, 0))


def detect_liquidity_migration(pair: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detects whether token likely migrated to a stronger / deeper DEX pool.
    Returns:
    {
        "migration": 0/1,
        "migration_label": "...",
        "migration_score": int
    }
    """
    if not pair:
        return {
            "migration": 0,
            "migration_label": "",
            "migration_score": 0,
        }

    raw_chain = (pair.get("chainId") or pair.get("chain") or "").lower().strip()
    chain = normalize_chain_name(raw_chain)
    base_addr = (safe_get(pair, "baseToken", "address", default="") or "").strip()
    cur_dex = (pair.get("dexId") or "").lower().strip()

    if not chain or not base_addr or not cur_dex:
        return {
            "migration": 0,
            "migration_label": "",
            "migration_score": 0,
        }

    try:
        pools = fetch_token_pairs(chain, base_addr)
    except Exception:
        pools = []

    if not pools or len(pools) < 2:
        return {
            "migration": 0,
            "migration_label": "",
            "migration_score": 0,
        }

    norm = []
    for p in pools:
        dex_id = (p.get("dexId") or "").lower().strip()
        liq = parse_float(safe_get(p, "liquidity", "usd", default=0), 0.0)
        vol24 = parse_float(safe_get(p, "volume", "h24", default=0), 0.0)
        age_min = pair_age_minutes(p)
        norm.append({
            "dex": dex_id,
            "liq": liq,
            "vol24": vol24,
            "age_min": age_min if age_min is not None else 999999,
            "pair": p,
        })

    norm.sort(key=lambda x: (x["liq"], x["vol24"]), reverse=True)

    best = norm[0]
    if best["dex"] != cur_dex:
        return {
            "migration": 0,
            "migration_label": "",
            "migration_score": 0,
        }

    label = ""
    score = 0

    for prev in norm[1:]:
        old_rank = dex_rank(chain, prev["dex"])
        new_rank = dex_rank(chain, cur_dex)

        if old_rank <= 0 or new_rank <= 0:
            continue

        liq_jump = best["liq"] / max(prev["liq"], 1.0)
        newer_pool = best["age_min"] < prev["age_min"]

        if new_rank > old_rank and liq_jump >= 1.8 and newer_pool:
            label = f"{prev['dex']} → {cur_dex}"
            score = min(10, int((new_rank - old_rank) * 2 + min(liq_jump, 4)))
            break

    if label:
        trap_label_value, trap_level = liquidity_trap(pair)
        if trap_level == "CRITICAL":
            score = max(0, score - 3)

    return {
        "migration": 1 if label else 0,
        "migration_label": label,
        "migration_score": score,
    }


def detect_cex_listing_probability(pair: Dict[str, Any]) -> Dict[str, Any]:
    if not pair:
        return {
            "cex_prob": 0,
            "cex_label": "",
        }

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0))
    vol24 = parse_float(safe_get(pair, "volume", "h24", default=0))
    vol5 = parse_float(safe_get(pair, "volume", "m5", default=0))
    pc1h = parse_float(safe_get(pair, "priceChange", "h1", default=0))

    base_addr = safe_get(pair, "baseToken", "address", default="")
    raw_chain = (pair.get("chainId") or pair.get("chain") or "").lower().strip()
    chain = normalize_chain_name(raw_chain)

    try:
        pools = fetch_token_pairs(chain, base_addr)
    except Exception:
        pools = []

    pool_count = len(pools)

    score = 0

    if liq > 100000:
        score += 1

    if vol24 > 200000:
        score += 1

    if vol5 > 5000:
        score += 1

    if pc1h > 15:
        score += 1

    if pool_count >= 3:
        score += 1

    label = ""

    if score >= 4:
        label = "HIGH"
    elif score >= 3:
        label = "MEDIUM"

    return {
        "cex_prob": score,
        "cex_label": label,
    }


def passes_monitoring_gate(row: Dict[str, Any], score: float) -> Tuple[bool, str]:
    """
    Final noise-reduction gate before token is added to Monitoring.
    Returns: (allowed, reason)
    """

    decision = str(build_trade_hint(row)[0] or "").upper()

    liq = parse_float(safe_get(row, "liquidity", "usd", default=0), 0.0)
    vol5 = parse_float(safe_get(row, "volume", "m5", default=0), 0.0)
    pc1h = parse_float(safe_get(row, "priceChange", "h1", default=0), 0.0)

    buys5 = int(safe_get(row, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(row, "txns", "m5", "sells", default=0) or 0)

    meme_score = int(row.get("meme_score", 0) or 0)
    smart_money = int(row.get("smart_money", 0) or 0)
    cex_prob = int(row.get("cex_prob", 0) or 0)

    trap = str(row.get("trap_signal", "") or "").upper()
    migration = str(row.get("migration_signal", "") or "").upper()

    strength_flags = 0
    if smart_money:
        strength_flags += 1
    if meme_score >= 60:
        strength_flags += 1
    if cex_prob >= 3:
        strength_flags += 1
    if migration:
        strength_flags += 1
    if pc1h >= 12:
        strength_flags += 1
    if vol5 >= 8000:
        strength_flags += 1

    if trap == "FAKE_VOLUME":
        if not (smart_money or cex_prob >= 4):
            return False, "gate: fake_volume"

    if liq < 7000:
        return False, "gate: low_liq"

    if buys5 < 4 and vol5 < 1500:
        return False, "gate: dead_flow"

    if score < 120 and strength_flags < 1:
        return False, "gate: weak_score"

    if score < 170 and vol5 < 1200 and pc1h < 2 and strength_flags < 2:
        return False, "gate: low_conviction"

    if sells5 > buys5 * 1.5 and strength_flags < 3:
        return False, "gate: sell_pressure"

    return True, "ok"


def detect_auto_signal(token: Dict[str, Any]) -> bool:
    liq = parse_float(safe_get(token, "liquidity", "usd", default=token.get("liquidity", 0)), 0.0)
    vol = parse_float(safe_get(token, "volume", "h24", default=token.get("volume", 0)), 0.0)
    txns = token.get("txns") or {}
    buys = int(safe_get(txns, "m5", "buys", default=txns.get("buys", 0)) or 0)
    if liq > 20000 and vol > 50000 and buys > 30:
        return True
    return False


def dex_url_for_token(chain: str, ca: str) -> str:
    chain = str(chain or "").strip().lower()
    ca = str(ca or "").strip()
    if not chain or not ca:
        return ""

    chain_map = {
        "solana": "solana",
        "bsc": "bsc",
        "binance-smart-chain": "bsc",
    }
    dex_chain = chain_map.get(chain, chain)
    return f"https://dexscreener.com/{dex_chain}/{ca}"


def token_ca(row: Dict[str, Any]) -> str:
    return str(
        row.get("base_addr")
        or row.get("pair_address")
        or row.get("pairAddress")
        or row.get("base_token_address")
        or row.get("ca")
        or row.get("address")
        or ""
    ).strip()


def token_chain(row: Dict[str, Any]) -> str:
    return str(row.get("chain") or "").strip().lower()


def normalize_timing_label(value: str) -> str:
    t = str(value or "").strip().upper()

    mapping = {
        "DUMP": "LATE",
        "GOOD": "GOOD",
        "EARLY": "EARLY",
        "LATE": "LATE",
        "NEUTRAL": "NEUTRAL",
        "SKIP": "SKIP",
        "WAIT": "WAIT",
        "READY": "GOOD",
    }

    return mapping.get(t, "NEUTRAL")


def suggested_position_size(row: Dict[str, Any], unified: Dict[str, Any]) -> str:
    action = str(unified.get("final_action") or "").upper()
    risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()

    if action in ("ENTER NOW", "HOLD") and risk in ("LOW", "MEDIUM"):
        return "NORMAL"
    if action in ("WATCH SMALL", "WAIT FOR PULLBACK"):
        return "SMALL" if risk == "LOW" else "WATCH ONLY"
    if action in ("WAIT", "WATCH CLOSELY"):
        return "WATCH ONLY"
    if action in ("REDUCE", "TAKE PROFIT", "EXIT", "NO ENTRY"):
        return "SKIP"
    return "WATCH ONLY"


def compute_unified_recommendation(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    source = str(source or "monitoring").strip().lower()
    entry_action = str(row.get("entry_action") or row.get("entry") or "").upper()
    entry_score = parse_float(row.get("entry_score", row.get("score", 0)), 0.0)
    timing = normalize_timing_label(str(row.get("timing_label") or "NEUTRAL"))
    risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()
    pnl = parse_float(row.get("pnl_pct", row.get("pnl", 0)), 0.0)
    reco_model = str(row.get("reco_model") or "").upper()
    exit_signal = str(row.get("exit_signal") or "").upper()
    trap_score = parse_float(row.get("trap_score", row.get("trap", 0)), 0.0)
    anti_rug = str(row.get("anti_rug") or row.get("safe") or "").upper()
    liquidity_health = str(row.get("liquidity_health") or "").upper()
    final_action = "WAIT"
    final_reason = "watch structure"

    if source == "portfolio":
        if reco_model.startswith("CLOSE") or exit_signal in ("EXIT", "CLOSE", "SELL"):
            final_action, final_reason = "EXIT", "exit/cut signal triggered"
        elif reco_model.startswith("TAKE PROFIT") or pnl >= 25:
            final_action, final_reason = "TAKE PROFIT", "target profit zone reached"
        elif reco_model.startswith("CUT") or (
            risk == "HIGH" and (trap_score > 0 or anti_rug in ("WARNING", "CRITICAL"))
        ):
            final_action, final_reason = "REDUCE", "risk elevated vs position quality"
        elif trap_score > 0:
            final_action, final_reason = "WATCH CLOSELY", "trap pressure detected"
        else:
            final_action, final_reason = "HOLD", "position still within hold conditions"
    else:
        weak = str(row.get("weak_reason") or "").strip().lower()
        score = parse_float(row.get("score", entry_score), 0.0)
        if score <= 0 or "gate_blocked" in weak or entry_action in ("AVOID", "INVALID"):
            final_action, final_reason = "NO ENTRY", "invalid/weak setup"
        elif entry_action in ("ENTRY_NOW", "READY"):
            final_action, final_reason = "ENTER NOW", "setup is ready with momentum"
        elif entry_action in ("WATCH_PULLBACK", "EARLY"):
            if risk == "LOW":
                final_action, final_reason = "WATCH SMALL", "pullback entry preferred"
            else:
                final_action, final_reason = "WAIT FOR PULLBACK", "risk requires deeper pullback"
        elif entry_action in ("TRACK", "WAIT", "WATCH"):
            final_action, final_reason = "WAIT", "monitor without entry confirmation"
        else:
            final_action, final_reason = "NO ENTRY", "no actionable edge"

    severity = "low"
    if final_action in ("WATCH CLOSELY", "REDUCE", "TAKE PROFIT", "WAIT FOR PULLBACK"):
        severity = "medium"
    if final_action in ("EXIT", "NO ENTRY"):
        severity = "high"

    confidence = "LOW"
    if entry_score >= 220:
        confidence = "HIGH"
    elif entry_score >= 120:
        confidence = "MEDIUM"

    unified = {
        "final_action": final_action,
        "final_reason": final_reason,
        "severity": severity,
        "timing": timing,
        "size_hint": "WATCH ONLY",
        "regime": source,
        "confidence": confidence,
    }
    unified["size_hint"] = suggested_position_size(row, unified)

    if liquidity_health in ("DEAD", "CRITICAL"):
        unified["severity"] = "high"
        if source == "portfolio":
            unified["final_action"] = "EXIT"
            unified["final_reason"] = "liquidity is critical/dead"
            unified["size_hint"] = "SKIP"

    return unified


def is_in_portfolio_active(row: Dict[str, Any], portfolio_rows: List[Dict[str, Any]]) -> bool:
    chain = str(row.get("chain") or "").strip().lower()
    addr = str(row.get("base_addr") or row.get("pair_address") or row.get("pairAddress") or "").strip()
    if not chain or not addr:
        return False

    key = f"{chain}:{addr}"
    for p in portfolio_rows:
        if str(p.get("active", "1")).strip() != "1":
            continue
        p_chain = str(p.get("chain") or "").strip().lower()
        p_addr = str(
            p.get("base_token_address")
            or p.get("base_addr")
            or p.get("pair_address")
            or p.get("pairAddress")
            or ""
        ).strip()
        if f"{p_chain}:{p_addr}" == key:
            return True
    return False


def build_history_series(hist: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    return {
        "price": [parse_float(h.get("price_usd"), 0.0) for h in hist if str(h.get("price_usd", "")).strip()],
        "score": [parse_float(h.get("last_score"), 0.0) for h in hist if str(h.get("last_score", "")).strip()],
        "entry_score": [parse_float(h.get("entry_score"), 0.0) for h in hist if str(h.get("entry_score", "")).strip()],
        "liquidity": [parse_float(h.get("liq_usd"), 0.0) for h in hist if str(h.get("liq_usd", "")).strip()],
        "vol5": [parse_float(h.get("vol5"), 0.0) for h in hist if str(h.get("vol5", "")).strip()],
        "vol24": [parse_float(h.get("vol24"), 0.0) for h in hist if str(h.get("vol24", "")).strip()],
    }


def render_monitoring_sparklines(hist: List[Dict[str, Any]]) -> None:
    series = build_history_series(hist)
    valid = {k: [x for x in v if x > 0] for k, v in series.items()}

    enough = sum(1 for v in valid.values() if len(v) >= 3)
    if enough == 0:
        st.caption("Not enough history yet.")
        return

    st.caption("Dynamics (sparklines)")
    cols = st.columns(3)
    items = [
        ("Price", valid["price"]),
        ("Score", valid["score"] or valid["entry_score"]),
        ("Liquidity", valid["liquidity"]),
        ("Vol24", valid["vol24"]),
        ("Vol5", valid["vol5"]),
    ]

    idx = 0
    for label, values in items:
        if len(values) < 3:
            continue
        with cols[idx % 3]:
            st.caption(label)
            st.line_chart(pd.DataFrame({"value": values}), height=100)
        idx += 1


def has_actionable_levels(row: Dict[str, Any]) -> bool:
    return any(
        [
            parse_float(row.get("entry_now"), 0.0) > 0,
            parse_float(row.get("pullback_1"), 0.0) > 0,
            parse_float(row.get("pullback_2"), 0.0) > 0,
            parse_float(row.get("invalidation"), 0.0) > 0,
            parse_float(row.get("tp1"), 0.0) > 0,
            parse_float(row.get("tp2"), 0.0) > 0,
        ]
    )


def load_suppressed_tokens() -> Dict[str, Dict[str, Any]]:
    raw = storage_read_text(SUPPRESSED_KEY, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_suppressed_tokens(data: Dict[str, Dict[str, Any]]) -> None:
    storage_write_text(SUPPRESSED_KEY, json.dumps(data, ensure_ascii=False, indent=2))


def suppress_token(chain: str, ca: str, reason: str = "manual_remove", days: int = 30) -> None:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    data = load_suppressed_tokens()
    until_ts = (datetime.utcnow() + timedelta(days=days)).isoformat()
    data[f"{chain}:{ca}"] = {
        "chain": chain,
        "ca": ca,
        "reason": reason,
        "suppressed_until": until_ts,
        "updated_at": now_utc_str(),
    }
    save_suppressed_tokens(data)


def is_token_suppressed(chain: str, ca: str) -> bool:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    data = load_suppressed_tokens()
    item = data.get(f"{chain}:{ca}")
    if not item:
        return False
    until_ts = str(item.get("suppressed_until") or "").strip()
    if not until_ts:
        return True
    try:
        return datetime.utcnow() < datetime.fromisoformat(until_ts)
    except Exception:
        return True


def build_entry_engine_v2(row: Dict[str, Any]) -> Dict[str, Any]:
    price = parse_float(row.get("priceUsd") or row.get("price_usd"), 0.0)
    liq = parse_float(safe_get(row, "liquidity", "usd", default=row.get("liquidity", 0)), 0.0)
    vol5 = parse_float(safe_get(row, "volume", "m5", default=row.get("volume_5m", 0)), 0.0)
    pc5 = parse_float(safe_get(row, "priceChange", "m5", default=row.get("price_change_5m", 0)), 0.0)
    pc1h = parse_float(safe_get(row, "priceChange", "h1", default=row.get("price_change_1h", 0)), 0.0)
    buys5 = parse_int(safe_get(row, "txns", "m5", "buys", default=0), 0)
    sells5 = parse_int(safe_get(row, "txns", "m5", "sells", default=0), 0)

    txns5 = buys5 + sells5
    buy_ratio = buys5 / txns5 if txns5 else 0.0
    vol_to_liq = vol5 / liq if liq else 0.0

    score = 100.0

    if liq >= 100000:
        score += 90
    elif liq >= 50000:
        score += 60
    elif liq >= 15000:
        score += 30
    else:
        score -= 25

    if vol5 >= 50000:
        score += 80
    elif vol5 >= 10000:
        score += 45
    elif vol5 >= 2000:
        score += 20
    else:
        score -= 15

    if txns5 >= 50:
        score += 70
    elif txns5 >= 20:
        score += 40
    elif txns5 >= 8:
        score += 15
    else:
        score -= 20

    if 2 <= pc5 <= 12:
        score += 35
    elif 12 < pc5 <= 22:
        score += 10
    elif pc5 > 22:
        score -= 30
    elif pc5 < -8:
        score -= 25

    if pc1h > 20:
        score += 15
    elif pc1h < -20:
        score -= 20

    if buy_ratio >= 0.60:
        score += 25
    elif buy_ratio >= 0.52:
        score += 10
    elif buy_ratio < 0.40:
        score -= 20

    if vol_to_liq >= 0.30:
        score += 20
    elif vol_to_liq >= 0.12:
        score += 10
    elif vol_to_liq < 0.03:
        score -= 15

    score = max(score, 1.0)

    risk_flags = []
    if liq < 12000:
        risk_flags.append("low_liq")
    if txns5 < 5:
        risk_flags.append("thin_flow")
    if pc5 > 25:
        risk_flags.append("overextended")

    action = "TRACK"
    timing = "NEUTRAL"
    horizon = "2-12h"
    reason = "watch candidate"

    if score >= 260 and 2 <= pc5 <= 12 and buy_ratio >= 0.55:
        action = "ENTRY_NOW"
        timing = "GOOD"
        horizon = "0-30m"
        reason = "active flow"
    elif score >= 190 and txns5 >= 8 and liq >= 12000:
        action = "WATCH_PULLBACK"
        timing = "EARLY"
        horizon = "0-2h"
        reason = "building flow"
    elif score >= 110:
        action = "TRACK"
        timing = "WAIT"
        horizon = "2-12h"
        reason = "early structure"
    else:
        action = "AVOID"
        timing = "SKIP"
        horizon = "ignore"
        reason = "weak structure"

    entry_now = 0.0
    pullback_1 = 0.0
    pullback_2 = 0.0
    invalidation = 0.0
    tp1 = 0.0
    tp2 = 0.0

    if price > 0:
        if action == "ENTRY_NOW":
            entry_now = price
            pullback_1 = price * 0.97
            pullback_2 = price * 0.93
            invalidation = price * 0.89
            tp1 = price * 1.12
            tp2 = price * 1.25
        elif action == "WATCH_PULLBACK":
            pullback_1 = price * 0.97
            pullback_2 = price * 0.93
            invalidation = price * 0.89
            tp1 = price * 1.10
            tp2 = price * 1.20
        elif action == "TRACK":
            pullback_1 = price * 0.96
            pullback_2 = price * 0.92
            invalidation = price * 0.88
            tp1 = price * 1.08
            tp2 = price * 1.16

    risk_level = "LOW"
    if liq < 15000 or buy_ratio < 0.45 or txns5 < 8:
        risk_level = "MEDIUM"
    if liq < 8000 or txns5 < 4:
        risk_level = "HIGH"

    return {
        "entry_score": round(score, 2),
        "entry_action": action,
        "entry_horizon": horizon,
        "timing_label": normalize_timing_label(timing),
        "entry_reason": reason,
        "entry_now": round(entry_now, 12) if entry_now else 0.0,
        "pullback_1": round(pullback_1, 12) if pullback_1 else 0.0,
        "pullback_2": round(pullback_2, 12) if pullback_2 else 0.0,
        "invalidation": round(invalidation, 12) if invalidation else 0.0,
        "tp1": round(tp1, 12) if tp1 else 0.0,
        "tp2": round(tp2, 12) if tp2 else 0.0,
        "risk_flags": ",".join(risk_flags),
        "risk_level": risk_level,
    }

def normalize_pair_row(pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    row = dict(pair)
    norm_chain = normalize_chain_name(row.get("chainId") or row.get("chain"))
    symbol = str(safe_get(row, "baseToken", "symbol", default=row.get("base_symbol", ""))).upper().strip()
    if norm_chain not in ALLOWED_CHAINS:
        return None
    if symbol in HARD_BLOCK_SYMBOLS:
        return None

    row["chain"] = norm_chain
    row["chainId"] = norm_chain
    row["meme_score"] = meme_token_score(row)
    row["smart_money"] = detect_smart_wallet_activity(row)
    age_minutes = pair_age_minutes(row)
    row["price_change_5m"] = parse_float(safe_get(row, "priceChange", "m5", default=row.get("price_change_5m", 0)), 0.0)
    row["price_change_15m"] = parse_float(safe_get(row, "priceChange", "m15", default=row.get("price_change_15m", 0)), 0.0)
    row["volume_5m"] = parse_float(safe_get(row, "volume", "m5", default=row.get("volume_5m", 0)), 0.0)
    row["liquidity"] = parse_float(safe_get(row, "liquidity", "usd", default=row.get("liquidity", 0)), 0.0)
    row["fdv"] = parse_float(row.get("fdv", 0), 0.0)
    row["age_minutes"] = parse_float(age_minutes if age_minutes is not None else row.get("age_minutes", 0), 0.0)
    liq_usd = parse_float(
        safe_get(row, "liquidity", "usd", default=row.get("liq_usd", 0)),
        0.0,
    )
    vol_5m = parse_float(
        safe_get(row, "volume", "m5", default=row.get("vol_5m", 0)),
        0.0,
    )
    txns_5m = (
        parse_int(safe_get(row, "txns", "m5", "buys", default=0), 0) +
        parse_int(safe_get(row, "txns", "m5", "sells", default=0), 0)
    )
    row["weak_liq"] = liq_usd < MIN_LIQ_USD
    row["weak_activity"] = (vol_5m < MIN_VOL_5M and txns_5m < MIN_TXNS_5M)

    if USE_HEAVY_MIGRATION_CHECK:
        mig = detect_liquidity_migration(row)
    else:
        mig = {
            "lp_migrated": False,
            "migration_flag": "",
            "migration_note": "",
            "migration": 0,
            "migration_label": "",
            "migration_score": 0,
        }
    row.update(mig)
    row["migration"] = parse_int(row.get("migration", 0), 0)
    row["migration_label"] = str(row.get("migration_label", "") or "")
    row["migration_score"] = parse_float(row.get("migration_score", 0), 0.0)

    cex = detect_cex_listing_probability(row)
    row["cex_prob"] = cex.get("cex_prob", 0)
    row["cex_label"] = cex.get("cex_label", "")

    trap_signal, _trap_level = liquidity_trap(row)
    row["trap_signal"] = trap_signal or ""
    row["migration_signal"] = row.get("migration_label", "") if int(row.get("migration", 0) or 0) == 1 else ""

    row["fresh_lp"] = fresh_lp(row)
    row["dev_risk"] = dev_wallet_risk(row)
    row["whale"] = whale_accumulation(row)
    row["signal"] = classify_signal(row)
    entry_data = build_entry_engine_v2(row)
    row.update(entry_data)
    action = str(row.get("entry_action") or "").upper()

    if action == "ENTRY_NOW":
        row["entry_status"] = "READY"
        row["entry"] = "READY"
    elif action == "WATCH_PULLBACK":
        row["entry_status"] = "EARLY"
        row["entry"] = "EARLY"
    elif action == "TRACK":
        row["entry_status"] = "WAIT"
        row["entry"] = "WAIT"
    else:
        row["entry_status"] = "NO_ENTRY"
        row["entry"] = "NO_ENTRY"

    row["status"] = "ACTIVE"
    row["signal_reason"] = row.get("entry_reason") or row.get("signal_reason") or ""
    row["risk_level"] = str(row.get("risk_level") or "MEDIUM").upper()
    toxic, reasons = is_toxic_token(pair)
    if toxic:
        row["risk"] = "HIGH"
        row["decision_reason"] = reasons[0]
        row["toxic_flags"] = ",".join(reasons)

    row["priority"] = parse_float(row.get("entry_score", 0), 0.0)
    row["priority_score"] = parse_float(row.get("priority_score", row.get("entry_score", 0)), 0.0)
    return row


def solana_unverified_heuristic(p: Dict[str, Any]) -> bool:
    if (p.get("chainId") or "").lower() != "solana":
        return False

    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)
    age = pair_age_minutes(p)

    very_new_or_unknown = (age is None) or (age < 60)
    vol_liq_ratio = (vol24 / max(1.0, liq)) if liq else 999.0

    if very_new_or_unknown and liq < 50_000:
        return True
    if very_new_or_unknown and vol_liq_ratio > 25:
        return True
    if very_new_or_unknown and pc1h > 120:
        return True
    return False


# =============================
# Scoring / decision / colors
# =============================
def score_pair(p: Optional[Dict[str, Any]]) -> float:
    if not p:
        return 0.0
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    vol5 = float(safe_get(p, "volume", "m5", default=0) or 0)
    pc5 = float(safe_get(p, "priceChange", "m5", default=0) or 0)
    pc1h = float(safe_get(p, "priceChange", "h1", default=0) or 0)
    buys5 = int(safe_get(p, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(p, "txns", "m5", "sells", default=0) or 0)
    trades5 = buys5 + sells5

    s = 0.0
    s += min(liq / 1000.0, 380.0)
    s += min(vol24 / 10000.0, 280.0)
    s += min(vol5 / 2000.0, 220.0)
    s += min(trades5 * 2.0, 120.0)
    s += max(min(pc1h, 90.0), -90.0) * 0.25
    s += max(min(pc5, 40.0), -40.0) * 0.15
    s += parse_float(p.get("migration_score", 0), 0.0) * 4.0

    cex_prob = parse_float(p.get("cex_prob", 0), 0.0)
    if cex_prob >= 4:
        s += 20
    elif cex_prob >= 3:
        s += 10

    toxic, _reasons = is_toxic_token(p)
    if toxic:
        s -= 200

    return round(s, 2)


def detect_snipers(pair: Optional[Dict[str, Any]]) -> bool:
    if not pair:
        return False

    txns = safe_get(pair, "txns", "h1", default={})
    buys = parse_float(txns.get("buys", 0))
    sells = parse_float(txns.get("sells", 0))

    vol = parse_float(safe_get(pair, "volume", "h24", default=0))
    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0))

    if buys > 40 and sells < buys * 0.25 and vol > liq * 0.5:
        return True

    return False


def pump_probability(pair: Optional[Dict[str, Any]]) -> int:
    if not pair:
        return 0

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0))
    vol = parse_float(safe_get(pair, "volume", "h24", default=0))
    pc1h = parse_float(safe_get(pair, "priceChange", "h1", default=0))
    pc5m = parse_float(safe_get(pair, "priceChange", "m5", default=0))
    txns = safe_get(pair, "txns", "h1", default={})

    buys = parse_float(txns.get("buys", 0))
    sells = parse_float(txns.get("sells", 0))

    score = 0

    if liq > 20000:
        score += 1

    if vol > liq * 1.5:
        score += 2

    if pc1h > 15:
        score += 2

    if pc5m > 5:
        score += 1

    if buys > sells * 1.8:
        score += 2

    return score




def classify_signal(pair: Optional[Dict[str, Any]]) -> str:

    pump = pump_probability(pair)
    trap, level = liquidity_trap(pair)

    if level == "CRITICAL":
        return "RED"

    if pump >= 6:
        return "GREEN"

    if pump >= 3:
        return "YELLOW"

    return "RED"


def liquidity_trap_detector(pair: Optional[Dict[str, Any]], hist: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if not pair:
        return {"trap_score": 0, "trap_level": "SAFE", "trap_flags": []}

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)
    vol24 = parse_float(safe_get(pair, "volume", "h24", default=0), 0.0)
    vol5 = parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)
    pc1h = parse_float(safe_get(pair, "priceChange", "h1", default=0), 0.0)
    pc5 = parse_float(safe_get(pair, "priceChange", "m5", default=0), 0.0)

    buys5 = int(safe_get(pair, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(pair, "txns", "m5", "sells", default=0) or 0)
    age = pair_age_minutes(pair)

    trap_score = 0
    flags: List[str] = []

    if liq > 0:
        vliq24 = vol24 / max(liq, 1.0)
        vliq5 = vol5 / max(liq, 1.0)

        if vliq24 >= 12:
            trap_score += 2
            flags.append("vol/liquidity anomaly 24h")
        if vliq5 >= 1.5:
            trap_score += 2
            flags.append("vol/liquidity anomaly 5m")

    if buys5 >= 12 and sells5 <= 1:
        trap_score += 2
        flags.append("buy-only flow")

    if liq < 12000 and (pc1h >= 35 or pc5 >= 12):
        trap_score += 2
        flags.append("pump on tiny liquidity")

    if age is not None and age < 90 and liq < 20000 and vol24 > 50000:
        trap_score += 2
        flags.append("fresh LP overheating")

    if hist and len(hist) >= 4:
        try:
            liqs = [parse_float(x.get("liq_usd", 0), 0.0) for x in hist[-4:]]
            vols = [parse_float(x.get("vol5_usd", 0), 0.0) for x in hist[-4:]]
            prices = [parse_float(x.get("price_usd", 0), 0.0) for x in hist[-4:]]

            if liqs[-1] < max(liqs[-3], 1.0) * 0.55:
                trap_score += 3
                flags.append("liquidity collapse")

            if vols[-1] < max(vols[-3], 1.0) * 0.25 and max(vols[:-1]) > 0:
                trap_score += 2
                flags.append("volume collapse")

            peak = max(prices[:-1]) if len(prices) >= 2 else 0
            if peak > 0 and prices[-1] < peak * 0.72:
                trap_score += 3
                flags.append("post-pump dump")
        except Exception:
            pass

    if trap_score >= 6:
        level = "CRITICAL"
    elif trap_score >= 3:
        level = "WARNING"
    else:
        level = "SAFE"

    return {"trap_score": trap_score, "trap_level": level, "trap_flags": flags}


def liquidity_trap(pair: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not pair:
        return None, None

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0))
    vol = parse_float(safe_get(pair, "volume", "h24", default=0))
    pc1h = parse_float(safe_get(pair, "priceChange", "h1", default=0))
    txns = safe_get(pair, "txns", "h1", default={})

    buys = parse_float(txns.get("buys", 0))
    sells = parse_float(txns.get("sells", 0))

    if liq < 8000 and vol > liq * 5:
        return "LOW_LIQ_TRAP", "CRITICAL"

    if sells > buys * 2 and pc1h < -10:
        return "DUMP_PATTERN", "CRITICAL"

    if vol > liq * 10:
        return "FAKE_VOLUME", "WARNING"

    return None, None


def fresh_lp(pair: Optional[Dict[str, Any]]) -> Optional[str]:
    if not pair:
        return None

    created = pair.get("pairCreatedAt")
    if not created:
        return None

    age_sec = (time.time() * 1000 - parse_float(created)) / 1000.0
    if age_sec < 3600:
        return "VERY_NEW"
    if age_sec < 86400:
        return "NEW"
    return None


def dev_wallet_risk(pair: Optional[Dict[str, Any]]) -> bool:
    if not pair:
        return False

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0))
    pc1h = parse_float(safe_get(pair, "priceChange", "h1", default=0))
    if liq < 15000 and pc1h > 60:
        return True
    return False


def whale_accumulation(pair: Optional[Dict[str, Any]]) -> bool:
    if not pair:
        return False

    txns = safe_get(pair, "txns", "h1", default={})
    buys = parse_float(txns.get("buys", 0))
    sells = parse_float(txns.get("sells", 0))
    vol = parse_float(safe_get(pair, "volume", "h24", default=0))
    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0))

    if buys > sells * 2 and vol > liq:
        return True
    return False


def trap_label(trap_signal: Optional[str]) -> str:
    if trap_signal == "FAKE_VOLUME":
        return "volume anomaly"
    return trap_signal or "—"


def entry_timing_signal(best: Optional[Dict[str, Any]], hist: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not best:
        return {"timing": "SKIP", "reason": ["no_data"]}

    try:
        m5 = parse_float(safe_get(best, "priceChange", "m5", default=0), 0.0)
        h1 = parse_float(safe_get(best, "priceChange", "h1", default=0), 0.0)

        # keep for future tuning; intentionally not used as hard gates in safe version
        _vol5 = parse_float(safe_get(best, "volume", "m5", default=0), 0.0)
        _liq = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)

        if m5 > 20:
            return {"timing": "SKIP", "reason": ["overextended_m5"]}
        if h1 > 100:
            return {"timing": "SKIP", "reason": ["parabolic_h1"]}
        if m5 > 12 and h1 > 25:
            return {"timing": "SKIP", "reason": ["late_pump_entry"]}
        if h1 > 60:
            return {"timing": "SKIP", "reason": ["overextended_h1"]}
        if m5 < -12:
            return {"timing": "SKIP", "reason": ["dumping"]}

        if 5 < m5 < 20 and h1 > 0:
            return {"timing": "GOOD", "reason": ["controlled_breakout"]}
        if 0 < m5 < 5 and h1 > 0:
            return {"timing": "GOOD", "reason": ["early_trend"]}

        if hist and len(hist) >= 5:
            prices = [parse_float(x.get("price_usd"), 0.0) for x in hist[-5:]]
            prices = [p for p in prices if p > 0]
            if prices:
                local_max = max(prices)
                last = prices[-1]
                drop = (local_max - last) / max(local_max, 1e-9)
                if 0.05 < drop < 0.25 and m5 > 0:
                    return {"timing": "GOOD", "reason": ["pullback_entry"]}

        return {"timing": "NEUTRAL", "reason": ["no_clear_edge"]}
    except Exception:
        return {"timing": "NEUTRAL", "reason": ["error"]}


def evaluate_entry_safe(token: Dict[str, Any]) -> Tuple[str, float, List[str], str]:
    """
    Returns:
        status: "READY" | "WAIT" | "NO_ENTRY"
        score: float (0-100)
        reasons: list[str]
    """
    price_change_5m = parse_float(token.get("price_change_5m", safe_get(token, "priceChange", "m5", default=0)), 0.0)
    price_change_15m = parse_float(token.get("price_change_15m", safe_get(token, "priceChange", "m15", default=0)), 0.0)
    volume_5m = parse_float(token.get("volume_5m", safe_get(token, "volume", "m5", default=0)), 0.0)
    liquidity = parse_float(token.get("liquidity", safe_get(token, "liquidity", "usd", default=0)), 0.0)
    fdv = parse_float(token.get("fdv", 0), 0.0)
    age_minutes = parse_float(token.get("age_minutes", pair_age_minutes(token) or 0), 0.0)

    score = 0.0
    reasons: List[str] = []

    if price_change_5m > 5:
        score += 25
        reasons.append("strong 5m momentum")
    elif price_change_5m > 2:
        score += 15
        reasons.append("moderate 5m momentum")

    if volume_5m > 50000:
        score += 20
        reasons.append("high volume")
    elif volume_5m > 20000:
        score += 10

    if liquidity > 30000:
        score += 15
    elif age_minutes > 30:
        score -= 20
        reasons.append("low liquidity risk")

    if 50000 < fdv < 5000000:
        score += 10
    else:
        score -= 10

    if age_minutes < 10:
        score += 10
        reasons.append("early token")
    elif age_minutes > 120:
        score -= 10

    # soft bonus: confirms short-term continuation when m15 is also green
    if price_change_15m > 0 and price_change_5m > 0:
        score += 5

    score = max(0.0, min(100.0, round(score, 2)))
    if score >= 60:
        return "READY", score, reasons, "LOW"
    if score >= 35:
        return "WAIT", score, reasons, "MEDIUM"
    return "NO_ENTRY", score, reasons, "HIGH"


def evaluate_entry_aggressive(token: Dict[str, Any]) -> Tuple[str, float, List[str], str]:
    """
    X10 mode – high risk / high reward.
    Returns:
        status: READY / WAIT / NO_ENTRY
        score: float
        reasons: list[str]
        risk_level: LOW / MEDIUM / HIGH
    """
    price_5m = parse_float(token.get("price_change_5m", safe_get(token, "priceChange", "m5", default=token.get("priceChange_m5", 0))), 0.0)
    price_1h = parse_float(token.get("price_change_1h", safe_get(token, "priceChange", "h1", default=token.get("priceChange_h1", 0))), 0.0)
    vol_5m = parse_float(token.get("volume_5m", safe_get(token, "volume", "m5", default=token.get("volume_m5", 0))), 0.0)
    vol_24h = parse_float(token.get("volume_24h", safe_get(token, "volume", "h24", default=token.get("volume_h24", 0))), 0.0)
    liq = parse_float(token.get("liquidity", safe_get(token, "liquidity", "usd", default=token.get("liquidity_usd", 0))), 0.0)
    fdv = parse_float(token.get("fdv", 0), 0.0)
    age = parse_float(token.get("age_minutes", pair_age_minutes(token) or 0), 0.0)
    buys = int(safe_get(token, "txns", "m5", "buys", default=token.get("txns_buys_m5", 0)) or 0)
    sells = int(safe_get(token, "txns", "m5", "sells", default=token.get("txns_sells_m5", 0)) or 0)

    score = 0.0
    reasons: List[str] = []
    risk = "MEDIUM"

    if price_5m > 10:
        score += 35
        reasons.append("strong breakout")
    elif price_5m > 5:
        score += 25
    elif price_5m > 2:
        score += 15

    if vol_5m > liq * 0.5:
        score += 30
        reasons.append("volume spike")
    elif vol_5m > liq * 0.2:
        score += 15

    if buys > sells * 2:
        score += 20
        reasons.append("buy pressure")
    elif buys > sells:
        score += 10

    if age < 30:
        score += 20
        reasons.append("very early")
    elif age < 120:
        score += 10

    if liq < 20000:
        score += 15
        risk = "HIGH"
        reasons.append("low liq gem")
    elif liq < 50000:
        score += 5

    if fdv < 2000000:
        score += 10

    # extra small stabilizer to avoid dead spikes when only one metric is elevated
    if vol_24h > 0 and vol_5m > (vol_24h * 0.04):
        score += 5
    if 0 < price_1h < 80:
        score += 2

    if price_5m > 30 and vol_5m < 10000:
        return "NO_ENTRY", round(score, 2), ["fake pump"], "HIGH"
    if sells > buys * 2:
        return "NO_ENTRY", round(score, 2), ["dump pressure"], "HIGH"

    score = round(score, 2)
    if score >= 70:
        return "READY", score, reasons, risk
    if score >= 40:
        return "WAIT", score, reasons, risk
    return "NO_ENTRY", score, reasons, risk


def evaluate_entry(token: Dict[str, Any], mode: str = "safe") -> Tuple[str, float, List[str], str]:
    if mode == "aggressive":
        return evaluate_entry_aggressive(token)
    return evaluate_entry_safe(token)


def classify_entry_status(
    best: Optional[Dict[str, Any]],
    decision: str,
    timing: Dict[str, Any],
    liq_health: Dict[str, Any],
    anti_rug: Dict[str, Any],
    size_info: Dict[str, Any],
) -> Dict[str, Any]:
    if not best:
        return {
            "status": "NO_ENTRY",
            "reason": ["no_live_pair"],
            "rank": 2,
            "can_promote": False,
        }

    decision_u = str(decision or "").upper()
    timing_state = str((timing or {}).get("timing", "SKIP") or "SKIP").upper()
    liq_level = str((liq_health or {}).get("level", "UNKNOWN") or "UNKNOWN").upper()
    anti_level = str((anti_rug or {}).get("level", "SAFE") or "SAFE").upper()
    final_size = float(size_info.get("usd_size_adj", size_info.get("usd_size", 0.0)) or 0.0)
    entry_status, entry_score, entry_reasons, _risk_level = evaluate_entry(best, mode=ENTRY_MODE)

    hard_blockers: List[str] = []
    if liq_level in ("DEAD", "CRITICAL"):
        hard_blockers.append(f"liq={liq_level}")
    if anti_level == "CRITICAL":
        hard_blockers.append("anti_rug_critical")
    if timing_state == "SKIP":
        hard_blockers.append("timing_skip")
    if final_size <= 0:
        hard_blockers.append("size_zero")
    if hard_blockers or entry_status == "NO_ENTRY":
        return {
            "status": "NO_ENTRY",
            "reason": hard_blockers + entry_reasons,
            "rank": 2,
            "can_promote": False,
        }

    if entry_status == "READY" and decision_u.startswith("ENTRY") and timing_state == "GOOD" and liq_level == "OK" and anti_level == "SAFE":
        return {
            "status": "READY",
            "reason": ["entry_ok", f"entry_score={entry_score}"] + entry_reasons,
            "rank": 0,
            "can_promote": True,
        }

    wait_reasons: List[str] = []
    if not decision_u.startswith("ENTRY"):
        wait_reasons.append("decision_wait")
    if timing_state != "GOOD":
        wait_reasons.append(f"timing={timing_state}")
    if liq_level == "WEAK":
        wait_reasons.append("liq_weak")
    if anti_level == "WARNING":
        wait_reasons.append("anti_rug_warning")

    return {
        "status": "WAIT",
        "reason": (wait_reasons or ["mixed_signals"]) + [f"entry_score={entry_score}"] + entry_reasons,
        "rank": 1,
        "can_promote": True,
    }


def build_final_decision(
    base_decision: str,
    entry_status: str,
    timing: Dict[str, Any],
    anti_rug: Dict[str, Any],
    live_score: float = 0.0,
    risk_flags: Optional[List[str]] = None,
) -> str:
    timing_state = str((timing or {}).get("timing", "UNKNOWN") or "UNKNOWN").upper()
    anti_level = str((anti_rug or {}).get("level", "SAFE") or "SAFE").upper()
    entry_status_u = str(entry_status or "UNKNOWN").upper()
    base_decision_u = str(base_decision or "").upper()

    if anti_level == "CRITICAL":
        return "NO ENTRY"
    if timing_state == "SKIP":
        return "NO ENTRY"
    if entry_status_u in ("NO_ENTRY", "NO ENTRY"):
        return "NO ENTRY"
    if float(live_score or 0.0) >= 400 and entry_status_u not in ("NO_ENTRY", "NO ENTRY"):
        return "TRADEABLE"
    flags_u = {str(x).upper() for x in (risk_flags or [])}
    if "CRITICAL_TRAP" in flags_u or "DEAD_LIQ" in flags_u:
        return "NO ENTRY"

    if timing_state == "NEUTRAL":
        return "WATCH / WAIT"
    if entry_status_u == "WAIT":
        return "WATCH / WAIT"
    if entry_status_u == "EARLY":
        return "WATCH / WAIT"
    if entry_status_u in ("ENTER", "READY", "TRADEABLE"):
        return "TRADEABLE"

    if "ENTRY" in base_decision_u:
        return "TRADEABLE"
    if "WATCH" in base_decision_u:
        return "WATCH / WAIT"
    return "NO ENTRY"


def detect_breakout(item: Dict[str, Any]) -> bool:
    pc5 = parse_float(item.get("pc5"), 0.0)
    pc1h = parse_float(item.get("pc1h"), 0.0)
    vol5 = parse_float(item.get("vol5_usd"), 0.0)
    return pc5 > 6 and pc1h < 15 and vol5 > 10000


def is_fake_pump(item: Dict[str, Any]) -> bool:
    buys = int(parse_float(item.get("buys5"), 0))
    sells = int(parse_float(item.get("sells5"), 0))
    vol5 = parse_float(item.get("vol5_usd"), 0.0)
    liq = parse_float(item.get("liq_usd"), 0.0)
    # disabled: old fake pump logic was too broad and killed valid entries
    # if vol5 < 3000:
    #     return True
    # if liq < 2000:
    #     return True
    # if sells > buys * 1.5:
    #     return True
    return False


def is_liquidity_trap(item: Dict[str, Any]) -> bool:
    liq = parse_float(item.get("liq_usd"), 0.0)
    vol5 = parse_float(item.get("vol5_usd"), 0.0)
    return liq < 3000 and vol5 > 15000


def rug_probability(item: Dict[str, Any]) -> int:
    liq = parse_float(item.get("liq_usd"), 0.0)
    pc1h = parse_float(item.get("pc1h"), 0.0)
    buys = int(parse_float(item.get("buys5"), 0))
    sells = int(parse_float(item.get("sells5"), 0))
    score = 0
    if liq < 2000:
        score += 2
    if sells > buys * 2 and buys > 0:
        score += 1
    if pc1h < -40:
        score += 1
    return score


def entry_engine_v2(item: Dict[str, Any]) -> Tuple[str, str, str]:
    pc1h = parse_float(item.get("pc1h"), 0.0)
    pc5m = parse_float(item.get("pc5"), 0.0)
    vol5 = parse_float(item.get("vol5_usd"), 0.0)
    liq = parse_float(item.get("liq_usd"), 0.0)
    buys = parse_float(item.get("buys5"), 0.0)
    sells = parse_float(item.get("sells5"), 0.0)
    txns = int(buys + sells)
    top_holder_pct = parse_float(item.get("top_holder_pct", 0), 0.0)

    risk_score = 0
    flags: List[str] = []
    if liq < 2000:
        risk_score += 2
        flags.append("low_liquidity")
    if top_holder_pct > 25:
        risk_score += 2
        flags.append("whale")
    if sells > buys * 2:
        risk_score += 1
        flags.append("sell_pressure")
    if vol5 < 100 and txns < 5:
        risk_score += 2
        flags.append("dead")

    score = 0.0
    if vol5 > 12000:
        score += 120
    elif vol5 > 6000:
        score += 85
    elif vol5 > 2500:
        score += 55
    elif vol5 > 800:
        score += 30

    if pc1h > 12:
        score += 120
    elif pc1h > 5:
        score += 75
    elif pc1h > 2:
        score += 45

    if pc5m > 6:
        score += 65
    elif pc5m > 3:
        score += 35
    elif pc5m > 1:
        score += 20

    if buys > sells * 1.5 and buys >= 4:
        score += 40
    elif buys > sells:
        score += 20

    if liq > 15000:
        score += 30

    momentum_flag = bool((pc1h > 2 or pc5m > 1) and vol5 > 800)
    if detect_breakout(item):
        momentum_flag = True

    if risk_score >= 5:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Anti-rug is annotation-only, it does NOT block entry.
    if score >= 300 and momentum_flag:
        return "READY", "breakout", risk_level
    if score >= 180:
        return "WATCH", "early_momentum", risk_level
    return "WAIT", "no_momentum", risk_level


def build_trade_hint(p: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not p:
        return "NO DATA", []

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
    elif liq >= 60_000:
        liq_tag = "Liquidity: Medium"
    else:
        liq_tag = "Liquidity: Low"

    if trades5 >= 60 and sells5 >= 10:
        flow_tag = "Flow: Hot"
    elif trades5 >= 25 and sells5 >= 6:
        flow_tag = "Flow: Active"
    else:
        flow_tag = "Flow: Thin"

    tags = [
        liq_tag,
        flow_tag,
        f"Trend(1h): {fmt_pct(pc1h)}",
        f"Micro(5m): {fmt_pct(pc5)}",
        f"Vol(5m): {fmt_usd(vol5)}",
    ]

    if int(p.get("migration", 0) or 0) == 1:
        tags.append(f"Migration: {p.get('migration_label', '')}")

    migration_bonus = int(p.get("migration", 0) or 0) == 1

    decision = "NO ENTRY"
    liq_gate = 25_000 if migration_bonus else 30_000
    vol_gate = 16_000 if migration_bonus else 20_000
    if liq >= liq_gate and vol24 >= vol_gate and trades5 >= 12 and sells5 >= 3:
        if pc1h > 6 and vol5 > 2_500 and pc5 >= -3:
            decision = "ENTRY (scalp)"
        else:
            decision = "WATCH / WAIT"

    trap = liquidity_trap_detector(p, None)

    if trap["trap_level"] == "WARNING" and decision.startswith("ENTRY"):
        decision = "WATCH / WAIT"

    if trap["trap_level"] == "CRITICAL":
        decision = "NO ENTRY"
        tags.append("Trap risk: CRITICAL")
    elif trap["trap_level"] == "WARNING":
        tags.append("Trap risk: WARNING")

    return decision, tags


def action_badge(action: str) -> str:
    a = (action or "").strip().upper()

    if a.startswith("NO ENTRY") or a == "NO ENTRY":
        color = "#e53e3e"
        bg = "rgba(229,62,62,0.12)"
    elif a.startswith("ENTRY") or a.startswith("BUY"):
        color = "#1f9d55"
        bg = "rgba(31,157,85,0.15)"
    elif "WATCH" in a or "WAIT" in a:
        color = "#d69e2e"
        bg = "rgba(214,158,46,0.15)"
    else:
        color = "#e53e3e"
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


def portfolio_action_badge(action: str) -> str:
    action = str(action or "").upper()
    mapping = {
        "EXIT": "🔴 EXIT",
        "TAKE PROFIT": "🟠 TAKE PROFIT",
        "REDUCE": "🟡 REDUCE",
        "WATCH CLOSELY": "🟠 WATCH CLOSELY",
        "HOLD": "🟢 HOLD",
    }
    return mapping.get(action, f"⚪ {action or 'HOLD'}")


def get_portfolio_entry_price(row: Dict[str, Any]) -> float:
    for key in ("avg_entry_price", "entry_price", "entry_price_usd", "entry", "price_at_add"):
        val = parse_float(row.get(key), 0.0)
        if val > 0:
            return val
    return 0.0


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
    block_majors: bool,
    min_age_min: int,
    max_age_min: int,
    enforce_age: bool,
    hide_solana_unverified: bool,
):
    stats = Counter()
    reasons = Counter()
    stats["total_in"] = len(pairs)
    out: List[Dict[str, Any]] = []

    for p in pairs:
        if (p.get("chainId") or "").lower() != chain.lower():
            reasons["chain_mismatch"] += 1
            continue
        stats["after_chain"] += 1

        if not any_dex and allowed_dexes:
            if (p.get("dexId") or "").lower() not in allowed_dexes:
                reasons["dex_not_allowed"] += 1
                continue
        stats["after_dex"] += 1

        if block_majors and is_major_or_stable(p):
            reasons["major_or_stable"] += 1
            continue
        stats["after_major_filter"] += 1

        if enforce_age:
            age = pair_age_minutes(p)
            if age is None:
                reasons["age_unknown"] += 1
                continue
            if age < float(min_age_min):
                reasons["age_too_new"] += 1
                continue
            if age > float(max_age_min):
                reasons["age_too_old"] += 1
                continue
        stats["after_age"] += 1

        if hide_solana_unverified and solana_unverified_heuristic(p):
            reasons["sol_unverified_like"] += 1
            continue
        stats["after_sol_check"] += 1

        liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
        vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)

        txns_m5 = safe_get(p, "txns", "m5", default=None)
        buys = sells = None
        if isinstance(txns_m5, dict):
            buys = float(txns_m5.get("buys", 0) or 0)
            sells = float(txns_m5.get("sells", 0) or 0)
        else:
            txns_h1 = safe_get(p, "txns", "h1", default=None)
            if isinstance(txns_h1, dict):
                buys = float(txns_h1.get("buys", 0) or 0) / 12.0
                sells = float(txns_h1.get("sells", 0) or 0) / 12.0
                reasons["txns_used_h1_scaled"] += 1
            else:
                reasons["txns_missing"] += 1
                buys = sells = None

        trades = (buys + sells) if (buys is not None and sells is not None) else None

        if liq < float(min_liq):
            reasons["liq<min"] += 1
            continue
        stats["after_liq"] += 1

        if vol24 < float(min_vol24):
            reasons["vol24<min"] += 1
            continue
        stats["after_vol24"] += 1

        if trades is None:
            reasons["txns_filters_skipped"] += 1
        else:
            if trades < float(min_trades_m5):
                reasons["trades<min"] += 1
                continue
            stats["after_trades"] += 1

            if sells < float(min_sells_m5):
                reasons["sells<min"] += 1
                continue
            stats["after_sells"] += 1

            if sells > 0 and buys > 0:
                imbalance = max(buys, sells) / max(1.0, min(buys, sells))
                if imbalance > float(max_buy_sell_imbalance):
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


# =============================
# Output dynamics / seed sampling
# =============================
def dedupe_mode(pairs: List[Dict[str, Any]], by_base_token: bool) -> List[Dict[str, Any]]:
    if not pairs:
        return []
    if not by_base_token:
        uniq = {}
        for p in pairs:
            k = p.get("pairAddress") or p.get("url")
            if not k:
                continue
            uniq[k] = p
        return list(uniq.values())

    best = {}
    for p in pairs:
        chain = (p.get("chainId") or "").lower().strip()
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
        k = addr_key(chain, base_addr)
        if not k:
            continue
        cur = best.get(k)
        if cur is None:
            best[k] = p
        else:
            liq1 = float(safe_get(p, "liquidity", "usd", default=0) or 0)
            vol1 = float(safe_get(p, "volume", "h24", default=0) or 0)
            liq0 = float(safe_get(cur, "liquidity", "usd", default=0) or 0)
            vol0 = float(safe_get(cur, "volume", "h24", default=0) or 0)
            if (liq1, vol1) > (liq0, vol0):
                best[k] = p
    return list(best.values())


def sample_seeds(seed_list: List[str], k: int, refresh: bool = False) -> List[str]:
    cleaned = [x.strip() for x in seed_list if str(x).strip()]
    if not cleaned:
        return []

    if WORKER_FAST_MODE:
        return random.sample(cleaned, min(k, len(cleaned)))

    if "seed_sample" not in st.session_state or refresh:
        st.session_state["seed_sample"] = random.sample(cleaned, min(k, len(cleaned)))

    return st.session_state["seed_sample"]


# =============================
# Swap URL builders
# =============================
def build_swap_url(chain: str, base_addr: str) -> str:
    chain = (chain or "").lower().strip()
    base_addr = (base_addr or "").strip()
    if not base_addr:
        return ""
    if chain == "bsc":
        return f"https://pancakeswap.finance/swap?outputCurrency={base_addr}"
    if chain == "solana":
        return f"https://jup.ag/swap/SOL-{base_addr}"
    return ""


# =============================
# Portfolio / Monitoring / Archive logic
# =============================
PORTFOLIO_FIELDS = [
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
]

MON_FIELDS = [
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
    "ts_archived",
    "archived_reason",
    "last_score",
    "last_decision",
    "priority_score",
    "last_decay_ts",
    "decay_hits",
    "source_window",
    "source_preset",
    "risk",
    "tp_target_pct",
    "entry_suggest_usd",
    "ts_last_seen",
    "signal",
    "smart_money",
    "entry_status",
    "entry_score",
    "risk_level",
    "entry_state",
    "revisit_count",
    "revisit_after_ts",
    "last_revisit_ts",
    "portfolio_linked",
    "note",
    "entry",
    "decision_reason",
    "signal_reason",
    "timing_label",
    "weak_reason",
    "in_portfolio",
    "toxic_flags",
    "alert_sent",
    "status",
    "risk_score",
    "risk_flags",
    "why",
]

HIST_FIELDS = [
    "ts_utc",
    "chain",
    "base_symbol",
    "base_addr",
    "pair_addr",
    "dex",
    "quote_symbol",
    "price_usd",
    "liq_usd",
    "vol24_usd",
    "vol5_usd",
    "pc1h",
    "pc5",
    "score_live",
    "decision",
    "entry_score",
    "entry_action",
    "entry_reason",
    "timing_label",
    "risk_level",
]


def load_portfolio() -> List[Dict[str, Any]]:
    rows = load_csv(PORTFOLIO_CSV)
    for r in rows:
        for k in PORTFOLIO_FIELDS:
            if k not in r:
                r[k] = ""
    return rows


def save_portfolio(rows: List[Dict[str, Any]]):
    save_csv(PORTFOLIO_CSV, rows, PORTFOLIO_FIELDS)
    backup_csv_snapshot(PORTFOLIO_CSV, rows, PORTFOLIO_FIELDS)


def active_portfolio_addresses() -> Set[str]:
    rows = load_portfolio()
    out: Set[str] = set()

    for r in rows:
        if r.get("active") != "1":
            continue

        chain = str(r.get("chain") or "").strip().casefold()
        raw = (r.get("base_token_address") or "").strip()
        stored = addr_store(chain, raw)
        if not stored:
            continue

        out.add(addr_key(chain, stored))

    return out


def build_live_portfolio_pairs() -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for r in load_portfolio():
        if r.get("active") != "1":
            continue
        chain = (r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, (r.get("base_token_address") or "").strip())
        if not chain or not base_addr:
            continue
        best = best_pair_for_token_cached(chain, base_addr)
        if best:
            pairs.append(best)
    return pairs


def portfolio_allocation_governor(size_info: Dict[str, Any]) -> Dict[str, Any]:
    size_label = str(size_info.get("size_label") or "").upper()
    if size_label == "SKIP":
        return {"status": "BLOCK", "reason": ["allocation_skip"]}
    return {"status": "ALLOW", "reason": ["allocation_ok"]}


def correlation_governor(candidate: Dict[str, Any]) -> Dict[str, Any]:
    current = build_live_portfolio_pairs()
    narrative_counts: Dict[str, int] = {}

    for p in current:
        n = detect_narrative(p)
        narrative_counts[n] = narrative_counts.get(n, 0) + 1

    cand_narr = detect_narrative(candidate)
    count = narrative_counts.get(cand_narr, 0)

    if count >= 3:
        return {
            "status": "BLOCK",
            "reason": [f"too_many_{cand_narr}"]
        }

    if count == 2:
        return {
            "status": "ALLOW_SMALL_ONLY",
            "reason": [f"{cand_narr}_overloaded"]
        }

    return {
        "status": "ALLOW",
        "reason": ["ok"]
    }


def correlation_size_modifier(corr: Dict[str, Any]) -> float:
    corr = corr or {"status": "ALLOW", "reason": []}
    status = str(corr.get("status", "ALLOW"))

    if status == "BLOCK":
        return 0.25

    if status == "ALLOW_SMALL_ONLY":
        return 0.5

    return 1.0


def risk_adjusted_size(
    base_size: float,
    corr: Dict[str, Any],
    liq_health: Optional[Dict[str, Any]] = None,
    anti_rug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    factor = correlation_size_modifier(corr)
    reasons: List[str] = []
    liq_level = str((liq_health or {}).get("level", "") or "").upper()
    anti_level = str((anti_rug or {}).get("level", "") or "").upper()

    if liq_level == "WEAK":
        factor *= 0.5
        reasons.append("liquidity_weak")
    elif liq_level in ("CRITICAL", "DEAD"):
        factor = 0.0
        reasons.append("liquidity_untradable")

    if anti_level == "CRITICAL":
        factor = 0.0
        reasons.append("anti_rug_critical")
    elif anti_level == "WARNING":
        factor *= 0.5
        reasons.append("anti_rug_warning")

    final_size = max(0.0, base_size * factor)
    if final_size <= 0:
        size_label = "SKIP"
    elif final_size < 10:
        size_label = "MICRO"
    elif final_size < 25:
        size_label = "SMALL"
    elif final_size < 60:
        size_label = "MEDIUM"
    else:
        size_label = "LARGE"

    return {
        "base_size": base_size,
        "final_size": round(final_size, 2),
        "factor": round(factor, 4),
        "size_label": size_label,
        "reason": reasons,
    }


def load_monitoring() -> List[Dict[str, Any]]:
    rows = load_csv(MONITORING_CSV)
    for r in rows:
        for k in MON_FIELDS:
            if k not in r:
                r[k] = ""
    return rows


def save_monitoring(rows: List[Dict[str, Any]]):
    save_csv(MONITORING_CSV, rows, MON_FIELDS)
    backup_csv_snapshot(MONITORING_CSV, rows, MON_FIELDS)


def migrate_reason_fields() -> int:
    rows = load_monitoring()
    changed = 0
    for r in rows:
        legacy = str(r.get("entry_reason") or "").strip()
        if legacy and not str(r.get("decision_reason") or "").strip():
            r["decision_reason"] = legacy
            changed += 1
    if changed:
        save_monitoring(rows)
    return changed


def load_monitoring_history(limit_rows: int = 8000) -> List[Dict[str, Any]]:
    # Source of truth: load_csv (Supabase app_storage if enabled, else local)
    rows = load_csv(MON_HISTORY_CSV)
    if not rows:
        return []
    return rows[-limit_rows:]


def append_monitoring_history(row: Dict[str, Any]):
    hist_row = dict(row)
    hist_row["entry_score"] = str(row.get("entry_score", ""))
    hist_row["entry_action"] = str(row.get("entry_action", ""))
    hist_row["entry_reason"] = str(row.get("entry_reason", ""))
    hist_row["timing_label"] = str(row.get("timing_label", ""))
    hist_row["risk_level"] = str(row.get("risk_level", ""))
    payload = {k: hist_row.get(k, "") for k in HIST_FIELDS}
    append_csv(MON_HISTORY_CSV, payload, HIST_FIELDS)


def token_exists(contract: str) -> bool:
    contract = (contract or "").strip()
    if not contract:
        return False
    rows = load_monitoring()
    for r in rows:
        if r.get("active") == "1" and (r.get("base_addr", "").strip() == contract):
            return True
    return False


def update_existing_token(
    p: Dict[str, Any],
    score: float,
    window_name: str = "",
    preset_key: str = "",
    entry_status: str = "",
    entry_score: float = 0.0,
    risk_level: str = "",
) -> str:
    return add_to_monitoring(
        p,
        score,
        window_name=window_name,
        preset_key=preset_key,
        entry_status=entry_status,
        entry_score=entry_score,
        risk_level=risk_level,
    )


def add_to_monitoring(
    p: Dict[str, Any],
    score: float,
    window_name: str = "",
    preset_key: str = "",
    entry_status: str = "",
    entry_score: float = 0.0,
    risk_level: str = "",
) -> str:
    chain = (p.get("chainId") or "").lower().strip()
    base_addr_raw = (safe_get(p, "baseToken", "address", default="") or "").strip()
    if not base_addr_raw:
        return "NO_ADDR"

    base_addr = addr_store(chain, base_addr_raw)
    key = addr_key(chain, base_addr)
    rows = load_monitoring()

    if not base_addr:
        return "NO_ADDR"

    for r in rows:
        if r.get("active") == "1" and addr_key(r.get("chain", ""), r.get("base_addr", "")) == key:
            # refresh lightweight fields
            r["last_score"] = str(score)
            r["ts_last_seen"] = now_utc_str()
            if entry_status:
                r["entry_status"] = entry_status
                r["entry_score"] = str(entry_score)
            if risk_level:
                r["risk_level"] = str(risk_level).upper()
            if p.get("weak_reason") is not None:
                r["weak_reason"] = str(p.get("weak_reason") or "")
            if p.get("decision_reason") is not None:
                r["decision_reason"] = str(p.get("decision_reason") or "")
            if p.get("signal_reason") is not None:
                r["signal_reason"] = str(p.get("signal_reason") or "")
            if p.get("toxic_flags") is not None:
                r["toxic_flags"] = str(p.get("toxic_flags") or "")
            if p.get("alert_sent") is not None:
                r["alert_sent"] = str(p.get("alert_sent") or "0")
            entry, signal_reason = compute_entry_signal(p)
            r["entry"] = entry
            if not r.get("signal_reason"):
                r["signal_reason"] = signal_reason
            r["timing_label"] = compute_timing(p)
            if window_name:
                r["source_window"] = _merge_csv_values(r.get("source_window", ""), window_name)
            if preset_key:
                r["source_preset"] = _merge_csv_values(r.get("source_preset", ""), preset_key)
            save_monitoring(rows)
            return "EXISTS_ACTIVE"

    for r in rows:
        if r.get("active") != "1" and addr_key(r.get("chain", ""), r.get("base_addr", "")) == key:
            r["ts_last_seen"] = now_utc_str()
            if window_name:
                r["source_window"] = _merge_csv_values(r.get("source_window", ""), window_name)
            if preset_key:
                r["source_preset"] = _merge_csv_values(r.get("source_preset", ""), preset_key)
            save_monitoring(rows)
            return "EXISTS_ARCHIVED"

    risk = "EARLY" if preset_key in {"ultra", "momentum"} else ""
    entry_s, tp_s = suggest_entry_and_tp_usd(p, risk=risk)
    decision = "watch"
    computed_entry_status, computed_entry_score, _, computed_risk = evaluate_entry(p, mode=ENTRY_MODE)
    computed_entry, computed_signal_reason = compute_entry_signal(p)
    final_entry_status = entry_status or computed_entry_status
    final_entry_score = float(entry_score if entry_status else computed_entry_score)
    final_risk_level = str((risk_level or p.get("risk_level") or computed_risk or "MEDIUM")).upper()

    rows.insert(0,
        {
            "ts_added": now_utc_str(),
            "chain": chain,
            "base_symbol": safe_get(p, "baseToken", "symbol", default="") or "",
            "base_addr": base_addr,
            "pair_addr": p.get("pairAddress", "") or "",
            "score_init": str(score),
            "liq_init": str(safe_get(p, "liquidity", "usd", default=0) or 0),
            "vol24_init": str(safe_get(p, "volume", "h24", default=0) or 0),
            "vol5_init": str(safe_get(p, "volume", "m5", default=0) or 0),
            "active": "1",
            "ts_archived": "",
            "archived_reason": "",
            "last_score": str(score),
            "last_decision": decision,
            "priority_score": str(score),
            "last_decay_ts": now_utc_str(),
            "decay_hits": "0",
            "source_window": window_name or "",
            "source_preset": preset_key or "",
            "risk": risk,
            "tp_target_pct": tp_s,
            "entry_suggest_usd": entry_s,
            "ts_last_seen": now_utc_str(),
            "signal": classify_signal(p),
            "smart_money": "1" if detect_smart_money(p) else "0",
            "entry_status": final_entry_status,
            "entry_score": str(final_entry_score),
            "risk_level": final_risk_level,
            "entry_state": "WAIT",
            "revisit_count": "0",
            "revisit_after_ts": "",
            "last_revisit_ts": "",
            "portfolio_linked": "0",
            "note": str(p.get("note", "") or ""),
            "entry": computed_entry,
            "decision_reason": str(p.get("decision_reason") or ""),
            "signal_reason": str(p.get("signal_reason") or computed_signal_reason or ""),
            "timing_label": compute_timing(p),
            "weak_reason": str(p.get("weak_reason") or ""),
            "in_portfolio": "0",
            "toxic_flags": str(p.get("toxic_flags") or ""),
            "alert_sent": str(p.get("alert_sent") or "0"),
        }
    )
    save_monitoring(rows)
    return "OK"


def archive_monitoring(
    chain: str,
    base_addr: str,
    reason: str,
    last_score: float = 0.0,
    last_decision: str = "",
    revisit_days: int = 0,
    portfolio_linked: bool = False,
    note: str = "",
) -> bool:
    if not base_addr:
        return False
    key = addr_key(chain, base_addr)
    rows = load_monitoring()
    changed = False
    for r in rows:
        if r.get("active") == "1" and addr_key(r.get("chain", ""), r.get("base_addr", "")) == key:
            r["active"] = "0"
            r["ts_archived"] = now_utc_str()
            r["archived_reason"] = reason
            r["last_score"] = f"{last_score:.2f}" if last_score else str(last_score)
            r["last_decision"] = last_decision or ""
            if portfolio_linked:
                r["portfolio_linked"] = "1"
            if note:
                r["note"] = note
            if revisit_days > 0:
                future_dt = datetime.utcnow() + timedelta(days=int(revisit_days))
                r["revisit_after_ts"] = future_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                r["revisit_after_ts"] = ""
            changed = True
    if changed:
        save_monitoring(rows)
    return changed


def update_monitoring_status(
    contract: str,
    chain: str,
    status: str,
    reason: str = "",
    last_score: float = 0.0,
    last_decision: str = "",
    revisit_days: int = 0,
    portfolio_linked: bool = False,
    note: str = "",
) -> bool:
    status_u = str(status or "").strip().lower()
    if status_u == "archived":
        return archive_monitoring(
            chain=chain,
            base_addr=contract,
            reason=reason or "manual",
            last_score=last_score,
            last_decision=last_decision,
            revisit_days=revisit_days,
            portfolio_linked=portfolio_linked,
            note=note,
        )
    if status_u == "active":
        return reactivate_monitoring(chain=chain, base_addr=contract)
    return False




def add_to_archive(chain: str, base_addr: str, reason: str) -> bool:
    return archive_monitoring(chain, base_addr, reason=reason)


def reactivate_monitoring(chain: str, base_addr: str) -> bool:
    if not base_addr:
        return False
    key = addr_key(chain, base_addr)
    rows = load_monitoring()
    changed = False
    for r in rows:
        if (r.get("active") != "1") and addr_key(r.get("chain", ""), r.get("base_addr", "")) == key:
            r["active"] = "1"
            r["ts_added"] = now_utc_str()
            r["ts_archived"] = ""
            r["archived_reason"] = ""
            r["last_score"] = ""
            r["last_decision"] = ""
            changed = True
            break
    if changed:
        save_monitoring(rows)
    return changed


def log_to_portfolio(p: Dict[str, Any], score: float, action: str, tags: List[str], swap_url: str) -> str:
    rows = load_portfolio()

    chain = (p.get("chainId") or "").lower().strip()
    dex = p.get("dexId") or ""
    base_sym = safe_get(p, "baseToken", "symbol", default="") or ""
    quote_sym = safe_get(p, "quoteToken", "symbol", default="") or ""
    base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
    pair_addr = (p.get("pairAddress", "") or "").strip()
    url = p.get("url", "") or ""
    price = safe_get(p, "priceUsd", default="") or ""

    key = addr_key(chain, base_addr)

    for r in rows:
        if r.get("active") != "1":
            continue
        if pair_addr and (r.get("pair_address", "").lower() == pair_addr.lower()):
            return "ALREADY_EXISTS"
        if addr_key(r.get("chain", ""), r.get("base_token_address", "")) == key:
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
    debug_log("portfolio_saved_trigger")
    return "OK"


def mark_monitoring_portfolio_linked(chain: str, base_addr: str, reason: str = "promoted_to_portfolio") -> bool:
    return update_monitoring_status(
        contract=base_addr,
        chain=chain,
        status="archived",
        reason=reason,
        portfolio_linked=True,
    )


def promote_to_portfolio(item: Dict[str, Any]) -> str:
    best = item.get("best")
    row = item.get("row") or {}
    if not best:
        return "NO_LIVE_PAIR"

    chain = (row.get("chain") or "").strip().lower()
    base_addr = addr_store(chain, (row.get("base_addr") or "").strip())
    decision = str(item.get("decision") or "WATCH / WAIT")
    res = log_to_portfolio(best, float(item.get("live_score", 0.0) or 0.0), decision, item.get("tags") or [], build_swap_url(chain, base_addr))
    if res == "ALREADY_EXISTS":
        update_monitoring_status(
            contract=base_addr,
            chain=chain,
            status="archived",
            reason="duplicate_portfolio",
            portfolio_linked=True,
        )
        return res
    if res != "OK":
        return res

    update_monitoring_status(
        contract=base_addr,
        chain=chain,
        status="archived",
        reason="promoted",
        portfolio_linked=True,
    )
    return "OK"


def active_base_sets() -> Tuple[set, set]:
    mon = load_monitoring()
    port = load_portfolio()
    mon_set = {addr_key(r.get("chain", ""), r.get("base_addr", "")) for r in mon if r.get("active") == "1" and r.get("base_addr")}
    port_set = {addr_key(r.get("chain", ""), r.get("base_token_address", "")) for r in port if r.get("active") == "1" and r.get("base_token_address")}
    return mon_set, port_set


# =============================
# Monitoring history snapshots
# =============================
def should_snapshot(chain: str, base_addr: str, min_interval_sec: int = 60) -> bool:
    if WORKER_FAST_MODE:
        return True

    key_id = addr_key(chain, base_addr)
    if not key_id:
        return False
    key = f"last_snap_{hashlib.md5(key_id.encode('utf-8')).hexdigest()[:10]}"
    now = time.time()
    last = float(st.session_state.get(key, 0.0))
    jitter = (int(hashlib.md5(key_id.encode("utf-8")).hexdigest(), 16) % 7)
    if (now - last) >= float(min_interval_sec + jitter):
        st.session_state[key] = now
        return True
    return False


def snapshot_live_to_history(chain: str, base_sym: str, base_addr: str, best: Optional[Dict[str, Any]]):
    if not chain or not base_addr or not best:
        return
    if not should_snapshot(chain, base_addr, min_interval_sec=120):
        return

    s_live = score_pair(best)
    decision, _tags = build_trade_hint(best)
    hist_rows = token_history_rows(chain, base_addr, limit=20)
    exit_signal = exit_before_dump_detector(best, hist_rows, 0.0)
    exit_level = str(exit_signal.get("exit_level", "WATCH"))

    row = {
        "ts_utc": now_utc_str(),
        "chain": (chain or "").lower().strip(),
        "base_symbol": base_sym or "",
        "base_addr": addr_store(chain, base_addr),
        "pair_addr": best.get("pairAddress", "") or "",
        "dex": best.get("dexId", "") or "",
        "quote_symbol": safe_get(best, "quoteToken", "symbol", default="") or "",
        "price_usd": str(parse_float(best.get("priceUsd"), 0.0)),
        "liq_usd": str(parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)),
        "vol24_usd": str(parse_float(safe_get(best, "volume", "h24", default=0), 0.0)),
        "vol5_usd": str(parse_float(safe_get(best, "volume", "m5", default=0), 0.0)),
        "pc1h": str(parse_float(safe_get(best, "priceChange", "h1", default=0), 0.0)),
        "pc5": str(parse_float(safe_get(best, "priceChange", "m5", default=0), 0.0)),
        "score_live": str(s_live),
        "decision": f"{decision} | {exit_level}",
    }
    append_monitoring_history(row)


def token_history_rows(chain: str, base_addr: str, limit: int = 180) -> List[Dict[str, Any]]:
    base_addr = addr_store(chain, base_addr)
    if not base_addr:
        return []
    key = addr_key(chain, base_addr)
    if not key:
        return []
    rows = load_monitoring_history(limit_rows=15000)
    filt = [r for r in rows if addr_key(r.get("chain", ""), r.get("base_addr", "")) == key]
    return filt[-limit:]


# =============================
# Monitoring ranking helpers
# =============================
def apply_priority_decay(row: Dict[str, Any], now_ts: datetime, decay_rate: float = 0.98, step_minutes: int = 5) -> Dict[str, Any]:
    last = parse_ts(row.get("last_decay_ts"))
    if not last:
        row["last_decay_ts"] = now_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        if not str(row.get("priority_score", "")).strip():
            row["priority_score"] = str(parse_float(row.get("score_init", 0), 0.0))
        if not str(row.get("decay_hits", "")).strip():
            row["decay_hits"] = "0"
        return row

    delta_min = (now_ts - last).total_seconds() / 60.0
    steps = int(delta_min // max(1, int(step_minutes)))
    if steps <= 0:
        return row

    score = parse_float(row.get("priority_score", row.get("score_init", 0)), 0.0)
    score *= decay_rate ** steps
    row["priority_score"] = str(round(score, 4))
    row["last_decay_ts"] = now_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    row["decay_hits"] = str(int(parse_float(row.get("decay_hits", 0), 0.0)) + steps)
    return row


def recheck_token(chain: str, base_addr: str, hist_limit: int = 30) -> Dict[str, Any]:
    best = best_pair_for_token_cached(chain, base_addr)
    hist = token_history_rows(chain, base_addr, limit=hist_limit)
    decision, _tags = build_trade_hint(best) if best else ("NO DATA", [])
    timing = entry_timing_signal(best, hist)
    timing_label = compute_timing(best if best else row)
    score_live = score_pair(best) if best else 0.0
    return {
        "best": best,
        "hist": hist,
        "decision": decision,
        "timing": timing,
        "score": score_live,
    }


def momentum_positive(hist: List[Dict[str, Any]]) -> bool:
    if len(hist) < 3:
        return False
    scores = [parse_float(x.get("score_live", 0), 0.0) for x in hist[-3:]]
    return scores[-1] > scores[0]


def anti_rug_check(best: Optional[Dict[str, Any]], hist: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not best:
        return ["NO_DATA"]

    flags: List[str] = []

    liq = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)
    vol = parse_float(safe_get(best, "volume", "h24", default=0), 0.0)

    if liq > 0 and vol / liq > 5:
        flags.append("VOL_LIQ_IMBALANCE")

    if hist and len(hist) >= 2:
        liqs = [
            parse_float(x.get("liq_usd", x.get("liquidity_usd", 0)), 0.0)
            for x in hist[-3:]
        ]
        if len(liqs) >= 2 and liqs[-1] < liqs[0] * 0.6:
            flags.append("LIQ_DROP")

    if hist and len(hist) >= 2:
        prices = [
            parse_float(x.get("price", x.get("price_usd", 0)), 0.0)
            for x in hist[-3:]
        ]
        if len(prices) >= 2 and prices[-1] > prices[0] * 2.5:
            flags.append("PARABOLIC")

    buys = parse_float(safe_get(best, "txns", "m5", "buys", default=0), 0.0)
    sells = parse_float(safe_get(best, "txns", "m5", "sells", default=0), 0.0)
    if buys > 0 and sells == 0:
        flags.append("NO_SELLS")

    if vol < 500 and liq < 2000:
        flags.append("DEAD")

    return flags


def dev_wallet_pattern(best: Optional[Dict[str, Any]]) -> Optional[str]:
    if not best:
        return None

    buys = parse_float(safe_get(best, "txns", "m5", "buys", default=0), 0.0)
    sells = parse_float(safe_get(best, "txns", "m5", "sells", default=0), 0.0)

    if buys > 20 and sells < 2:
        return "DEV_ACCUMULATION"

    if sells > buys * 2:
        return "DEV_DUMP"

    return None


def smart_money_signal(hist: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not hist or len(hist) < 5:
        return None

    recent = hist[-5:]
    volumes = [
        parse_float(x.get("vol_5m", x.get("vol5_usd", x.get("volume_5m", 0))), 0.0)
        for x in recent
    ]
    prices = [
        parse_float(x.get("price", x.get("price_usd", 0)), 0.0)
        for x in recent
    ]

    if volumes[-1] > volumes[0] * 2 and prices[-1] <= prices[0] * 1.1:
        return "SMART_ACCUMULATION"

    if volumes[-1] > volumes[0] * 2 and prices[-1] > prices[0] * 1.5:
        return "SMART_BREAKOUT"

    if volumes[-2] > volumes[-1] * 2:
        return "SMART_EXIT"

    return None


def confidence_score(score: float, sm_signal: Optional[str], rug_flags: List[str]) -> float:
    conf = parse_float(score, 0.0) / 500.0

    if sm_signal:
        conf += 0.2

    if rug_flags:
        conf -= 0.3

    return max(0.0, min(conf, 1.0))


def evaluate_entry_state(decision: str, timing: Dict[str, Any], score: float, hist: List[Dict[str, Any]], sm_signal: Optional[str]) -> str:
    if sm_signal == "SMART_ACCUMULATION" and score > 250:
        return "READY"

    if sm_signal == "SMART_BREAKOUT":
        return "READY"

    if sm_signal == "SMART_EXIT":
        return "NO_ENTRY"

    timing_state = str((timing or {}).get("timing", "SKIP") or "SKIP").upper()
    if str(decision or "").upper().startswith("ENTRY") and timing_state == "GOOD":
        return "READY"
    if score >= 400 and momentum_positive(hist):
        return "READY"
    return "WAIT"


def kill_logic(best: Optional[Dict[str, Any]], hist: List[Dict[str, Any]], decision: str, score: float, sm_signal: Optional[str]) -> Optional[str]:
    if not best:
        return "KILL_NO_DATA"

    rug_flags = anti_rug_check(best, hist)
    dev_flag = dev_wallet_pattern(best)
    if dev_flag:
        rug_flags.append(dev_flag)

    if sm_signal == "SMART_EXIT":
        return "KILL_SMART_EXIT"

    if "DEV_DUMP" in rug_flags:
        return "KILL_DEV_DUMP"

    if rug_flags:
        return f"KILL_RUG_{'_'.join(rug_flags[:2])}"

    if rug_like(best, hist):
        return "KILL_RUG"
    if str(liquidity_health(best).get("level", "")).upper() == "CRITICAL":
        return "KILL_LIQ"
    if score < 120:
        return "KILL_LOW_SCORE"
    recent = hist[-3:] if hist else []
    if recent and all(str(x.get("decision", "")).upper().startswith("NO ENTRY") for x in recent):
        return "KILL_NO_FLOW"
    return None


def monitoring_priority(
    score_live: float,
    pc1h: float,
    pc5: float,
    vol5: float,
    liq: float,
    meme_score: float = 0.0,
    migration_score: float = 0.0,
    cex_prob: float = 0.0,
) -> float:
    s = 0.0
    s += score_live * 1.2
    s += max(min(pc1h, 25.0), -25.0) * 3.0
    s += max(min(pc5, 12.0), -12.0) * 2.0
    s += min(vol5 / 2000.0, 35.0) * 4.0
    if liq < 20_000:
        s -= 60.0
    if meme_score > 60:
        s += 10.0
    if migration_score > 0:
        s += migration_score * 8.0
    if cex_prob >= 4:
        s += 30
    elif cex_prob >= 3:
        s += 15
    return round(s, 2)


def compute_priority(row: Dict[str, Any]) -> float:
    score = float(parse_float(row.get("last_score", 0), 0.0))
    momentum = float(parse_float(row.get("vol5_init", 0), 0.0))
    age_factor = 1.0
    if row.get("ts_added"):
        age_factor = 0.8
    return score + momentum * 0.05 * age_factor


# =============================
# Portfolio recommendation (fixed)
# =============================
def portfolio_reco(entry: float, current: float, liq: float, vol24: float, pc1h: float, pc5: float, decision: str, score_live: float, pair: Optional[Dict[str, Any]] = None) -> str:
    # Hard safety exits first
    if current <= 0 or entry <= 0:
        return "REVIEW (no price)"
    pnl = (current - entry) / entry * 100.0

    # dead / rug-like symptoms
    if pnl <= -90:
        return "CLOSE (dead / rug-like)"
    if liq > 0 and liq < 3000 and pnl <= -60:
        return "CLOSE (low liq + heavy loss)"
    if vol24 > 0 and vol24 < 500 and pnl <= -60:
        return "CLOSE (no volume)"
    if str(decision).upper() == "NO ENTRY" and pnl <= -35:
        return "CUT (signal flipped)"

    # take profit / trim
    if pnl >= 120:
        return "TAKE PROFIT (moon)"
    if pnl >= 60 and pc5 < 0 and pc1h < 0:
        return "TAKE PROFIT"
    if pnl >= 30 and pc5 < -2:
        return "TRIM / TP"

    # momentum hold with guardrails
    if pnl >= -10 and pc1h > 5 and pc5 >= -1 and score_live >= 180:
        return "HOLD (momentum)"

    if pair:
        pump = pump_probability(pair)
        trap, trap_level = liquidity_trap(pair)

        if trap_level == "CRITICAL":
            return "⚠️ EXIT (liquidity risk)"

        if pump >= 7:
            return "🚀 HOLD (possible 10x)"

    # default: explicit review (not blind hold)
    if pnl <= -35:
        return "CUT / RISK"
    return "HOLD / WATCH"


def anti_rug_early_detector(
    pair: Optional[Dict[str, Any]],
    hist: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not pair:
        return {
            "level": "SAFE",
            "score": 0,
            "flags": ["no_data"],
            "action": "WATCH",
        }

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)
    vol24 = parse_float(safe_get(pair, "volume", "h24", default=0), 0.0)
    vol5 = parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)

    h1 = parse_float(safe_get(pair, "priceChange", "h1", default=0), 0.0)
    m5 = parse_float(safe_get(pair, "priceChange", "m5", default=0), 0.0)

    buys5 = int(safe_get(pair, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(pair, "txns", "m5", "sells", default=0) or 0)

    score = 0
    flags: List[str] = []

    if liq > 0 and vol24 / max(liq, 1.0) > 8:
        score += 1
        flags.append("vol_liq_stretched")

    if sells5 > max(1, buys5 * 1.8):
        score += 2
        flags.append("sell_pressure")

    if h1 < -12:
        score += 1
        flags.append("weak_h1")

    if m5 < -6:
        score += 1
        flags.append("weak_m5")

    if liq < 15000:
        score += 1
        flags.append("thin_liquidity")
    if liq < 5000:
        score += 3
        flags.append("liq_blackhole")
    if liq < 1000:
        score += 4
        flags.append("near_dead_liquidity")
    if vol24 > 1_000_000 and liq < 10_000:
        score += 2
        flags.append("fake_volume_profile")
    if buys5 > 0 and sells5 > buys5 * 2.2:
        score += 2
        flags.append("aggressive_sell_imbalance")

    if hist and len(hist) >= 4:
        try:
            recent = hist[-4:]

            liqs = [parse_float(x.get("liq_usd", 0), 0.0) for x in recent]
            vols = [parse_float(x.get("vol5_usd", 0), 0.0) for x in recent]
            prices = [parse_float(x.get("price_usd", 0), 0.0) for x in recent]

            if liqs[-1] < liqs[0] * 0.75:
                score += 2
                flags.append("liq_fading")
            if len(liqs) >= 3 and liqs[-1] < max(liqs[:-1]) * 0.5:
                score += 2
                flags.append("liq_halfed")

            if vols[-1] < max(vols[:-1] or [0]) * 0.4:
                score += 1
                flags.append("flow_fading")

            peak = max(prices[:-1]) if prices[:-1] else 0
            if peak > 0 and prices[-1] < peak * 0.85:
                score += 1
                flags.append("failed_reclaim")
            if len(prices) >= 3:
                recent_peak = max(prices[:-1])
                if recent_peak > 0 and prices[-1] < recent_peak * 0.75:
                    score += 2
                    flags.append("deep_failed_reclaim")

        except Exception:
            flags.append("hist_error")

    if score >= 6:
        return {"level": "CRITICAL", "score": score, "flags": flags, "action": "EXIT"}

    if score >= 3:
        return {"level": "WARNING", "score": score, "flags": flags, "action": "REDUCE"}

    return {"level": "SAFE", "score": score, "flags": flags, "action": "WATCH"}


def flow_collapse_detector(hist: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not hist or len(hist) < 4:
        return {"level": "SAFE", "flags": []}

    vols = [parse_float(x.get("vol5_usd", 0), 0.0) for x in hist[-4:]]
    liqs = [parse_float(x.get("liq_usd", 0), 0.0) for x in hist[-4:]]
    flags: List[str] = []

    if vols[-1] < max(vols[:-1] or [0]) * 0.3:
        flags.append("flow_collapse")

    if liqs[-1] < max(liqs[:-1] or [0]) * 0.6:
        flags.append("liq_collapse")

    if len(flags) >= 2:
        return {"level": "WARNING", "flags": flags}

    return {"level": "SAFE", "flags": flags}


def exit_before_dump_detector(
    pair: Optional[Dict[str, Any]],
    hist: Optional[List[Dict[str, Any]]] = None,
    entry_price: float = 0.0,
) -> Dict[str, Any]:
    if not pair or not hist or len(hist) < 5:
        return {
            "exit_level": "WATCH",
            "exit_score": 0,
            "exit_flags": [],
            "suggested_action": "HOLD",
        }

    try:
        h1_change = float(safe_get(pair, "priceChange", "h1", default=0) or 0)
        m5_change = float(safe_get(pair, "priceChange", "m5", default=0) or 0)

        prev = hist[-5:] if len(hist) >= 5 else hist
        liq_vals = [float(x.get("liquidity_usd") or x.get("liq_usd") or 0) for x in prev]
        vol_vals = [float(x.get("volume_5m") or x.get("vol5_usd") or 0) for x in prev]

        liq_trend = (liq_vals[-1] - liq_vals[0]) if len(liq_vals) >= 2 else 0
        avg_vol = (sum(vol_vals) / len(vol_vals)) if vol_vals else 0
        vol_spike = bool(vol_vals) and max(vol_vals) > (avg_vol * 2 if avg_vol > 0 else 1)

        flags: List[str] = []
        score = 0

        if h1_change < -25:
            flags.append("hard_dump_h1")
            score += 3

        if liq_trend < -0.3 * max(liq_vals or [1]):
            flags.append("liquidity_drop")
            score += 2

        if vol_spike and m5_change < -10:
            flags.append("capitulation")
            score += 2

        if h1_change < -15:
            flags.append("weak_h1")
            score += 1

        if m5_change < -7:
            flags.append("weak_m5")
            score += 1

        if score >= 4:
            level = "EXIT"
        elif score >= 2:
            level = "EARLY"
        else:
            level = "WATCH"

        if level == "EXIT":
            suggested_action = "EXIT_100"
        elif level == "EARLY":
            suggested_action = "REDUCE_50" if score >= 3 else "REDUCE_30"
        else:
            suggested_action = "HOLD"

        return {
            "exit_level": level,
            "exit_score": score,
            "exit_flags": flags,
            "suggested_action": suggested_action,
        }
    except Exception:
        return {
            "exit_level": "WATCH",
            "exit_score": 0,
            "exit_flags": ["error"],
            "suggested_action": "HOLD",
        }


def exit_persistence_state(hist: Optional[List[Dict[str, Any]]], min_points: int = 3) -> Dict[str, Any]:
    if not hist:
        return {
            "exit_hits": 0,
            "early_hits": 0,
            "persistent_exit": False,
            "persistent_early": False,
        }

    recent = hist[-max(3, min_points):]
    exit_hits = 0
    early_hits = 0

    for row in recent:
        decision = str(row.get("decision", "") or "").upper()
        if "EXIT" in decision:
            exit_hits += 1
        elif "EARLY" in decision or "TRIM" in decision:
            early_hits += 1

    return {
        "exit_hits": exit_hits,
        "early_hits": early_hits,
        "persistent_exit": exit_hits >= 2,
        "persistent_early": early_hits >= 2,
    }


def liquidity_health(best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not best:
        return {
            "level": "UNKNOWN",
            "liq_usd": 0.0,
            "action": "WATCH",
            "flags": ["no_data"],
        }

    liq = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)
    flags: List[str] = []

    if liq < 100:
        flags.append("liquidity_dead")
        return {
            "level": "DEAD",
            "liq_usd": liq,
            "action": "EXIT_NOW",
            "flags": flags,
        }

    if liq < 1000:
        flags.append("liquidity_critical")
        return {
            "level": "CRITICAL",
            "liq_usd": liq,
            "action": "EXIT",
            "flags": flags,
        }

    if liq < 5000:
        flags.append("liquidity_weak")
        return {
            "level": "WEAK",
            "liq_usd": liq,
            "action": "REDUCE",
            "flags": flags,
        }

    return {
        "level": "OK",
        "liq_usd": liq,
        "action": "HOLD",
        "flags": flags,
    }


def apply_liquidity_exit_override(
    exit_signal: Dict[str, Any],
    liq_health: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    signal = dict(exit_signal or {})
    liq_action = str((liq_health or {}).get("action", "") or "").upper()
    liq_level = str((liq_health or {}).get("level", "") or "").upper()

    if liq_action in ("EXIT_NOW", "EXIT"):
        flags = list(signal.get("exit_flags", []) or [])
        if "liquidity_collapse" not in flags:
            flags.append("liquidity_collapse")
        signal.update(
            {
                "exit_level": "EXIT",
                "exit_flags": flags,
                "suggested_action": "EXIT_100",
            }
        )
        if liq_level == "DEAD":
            signal["exit_score"] = max(int(signal.get("exit_score", 0) or 0), 999)
        else:
            signal["exit_score"] = max(int(signal.get("exit_score", 0) or 0), 5)
        return signal

    if liq_action == "REDUCE" and str(signal.get("exit_level", "WATCH")).upper() == "WATCH":
        flags = list(signal.get("exit_flags", []) or [])
        if "liquidity_weak" not in flags:
            flags.append("liquidity_weak")
        signal.update(
            {
                "exit_level": "EARLY",
                "exit_score": max(int(signal.get("exit_score", 0) or 0), 2),
                "exit_flags": flags,
                "suggested_action": "REDUCE_50",
            }
        )

    return signal


def apply_liquidity_reco_override(reco: str, liq_health: Optional[Dict[str, Any]]) -> str:
    liq_level = str((liq_health or {}).get("level", "") or "").upper()
    reco_text = str(reco or "")

    if liq_level == "DEAD":
        return "FORCE EXIT / DEAD LIQUIDITY"
    if liq_level == "CRITICAL":
        return "EXIT NOW / CRITICAL LIQUIDITY"
    if liq_level == "WEAK" and "HOLD" in reco_text.upper():
        return "REDUCE / WATCH"
    return reco_text


def action_from_exit_signal(exit_signal: Dict[str, Any], persistence: Dict[str, Any]) -> Dict[str, str]:
    level = str(exit_signal.get("exit_level", "WATCH"))
    base_action = str(exit_signal.get("suggested_action", "HOLD"))

    if level == "EXIT":
        if persistence.get("persistent_exit"):
            return {
                "action_code": "EXIT_100",
                "action_label": "Exit fully now",
            }
        return {
            "action_code": "EXIT_CONFIRM",
            "action_label": "Exit likely – confirm next cycle",
        }

    if level == "EARLY":
        if persistence.get("persistent_early"):
            if base_action == "REDUCE_50":
                return {
                    "action_code": "REDUCE_50",
                    "action_label": "Reduce 50%",
                }
            return {
                "action_code": "REDUCE_30",
                "action_label": "Reduce 30%",
            }
        return {
            "action_code": "WATCH_TIGHT",
            "action_label": "Watch closely",
        }

    return {
        "action_code": "HOLD",
        "action_label": "Hold",
    }


def position_strength_score(best: Dict[str, Any], hist: List[Dict[str, Any]], entry_price: float) -> float:
    if not best:
        return 0.0

    price = float(best.get("priceUsd") or 0)
    liq = float(best.get("liquidity", {}).get("usd") or 0)
    vol = float(best.get("volume", {}).get("h24") or 0)

    pnl = 0.0
    if entry_price and entry_price > 0:
        pnl = (price - entry_price) / entry_price

    score = 0.0

    if pnl > 0.5:
        score += 2
    elif pnl > 0.2:
        score += 1
    elif pnl < -0.2:
        score -= 2

    if liq > 200_000:
        score += 1
    elif liq < 30_000:
        score -= 1

    if vol > 2_000_000:
        score += 1
    elif vol < 200_000:
        score -= 1

    return score


def classify_position_strength(score: float) -> str:
    if score >= 3:
        return "STRONG"
    if score <= -1:
        return "WEAK"
    return "NEUTRAL"


def capital_rotation_engine(active_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    strong: List[Dict[str, Any]] = []
    weak: List[Dict[str, Any]] = []
    neutral: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []

    for r in active_rows:
        chain = (r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, (r.get("base_token_address") or "").strip())
        entry_price = get_portfolio_entry_price(r)

        best = best_pair_for_token_cached(chain, base_addr) if (chain and base_addr) else None
        hist = token_history_rows(chain, base_addr, limit=60)
        liq_health = liquidity_health(best)
        anti_rug = anti_rug_early_detector(best, hist)
        if liq_health.get("level") in ("DEAD", "CRITICAL"):
            continue
        if anti_rug.get("level") == "CRITICAL":
            continue

        score = position_strength_score(best or {}, hist, entry_price)
        if anti_rug.get("level") == "WARNING":
            score *= 0.6
        cls = classify_position_strength(score)

        enriched_row = {
            "row": r,
            "class": cls,
            "score": score,
            "best": best,
            "hist": hist,
            "anti_rug": anti_rug,
        }
        enriched.append(enriched_row)

        if cls == "STRONG":
            if liquidity_health(best).get("level") != "OK":
                continue
            strong.append(r)
        elif cls == "WEAK":
            weak.append(r)
        else:
            neutral.append(r)

    return {
        "strong": strong,
        "weak": weak,
        "neutral": neutral,
        "enriched": enriched,
    }


def rotation_actions(engine: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    strong = engine["strong"]
    weak = engine["weak"]

    if not strong or not weak:
        return actions

    for w in weak[:2]:
        for s in strong[:2]:
            actions.append(
                {
                    "type": "ROTATE",
                    "from": w.get("base_symbol") or w.get("symbol") or "?",
                    "to": s.get("base_symbol") or s.get("symbol") or "?",
                    "reason": "capital_shift_weak_to_strong",
                }
            )

    return actions


def rotation_exit_plan(engine: Dict[str, Any]) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []

    strong = engine.get("strong", [])
    weak = engine.get("weak", [])

    if not strong or not weak:
        return plans

    for w in weak:
        plans.append(
            {
                "symbol": w.get("base_symbol") or w.get("symbol") or "?",
                "action": "PARTIAL_EXIT",
                "exit_pct": 0.5,
                "reason": "weak_rotation",
            }
        )

    return plans


def position_sizing_engine(
    pair: Optional[Dict[str, Any]],
    portfolio_value_usd: float = 1000.0,
    hist: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not pair:
        return {
            "size_pct": 0.0,
            "size_label": "SKIP",
            "usd_size": 0.0,
            "risk_score": 10,
            "risk_flags": ["no_data"],
        }

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)
    vol5 = parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)
    pc1h = parse_float(safe_get(pair, "priceChange", "h1", default=0), 0.0)

    score = score_pair(pair)
    signal = classify_signal(pair)
    smart_money = detect_smart_money(pair)
    whale = whale_accumulation(pair)
    dev = dev_wallet_risk(pair)
    fresh = fresh_lp(pair)
    sniper = detect_snipers(pair)

    trap_signal, trap_level = liquidity_trap(pair)

    migration = detect_liquidity_migration(pair)
    migration_score = parse_float(migration.get("migration_score", 0), 0.0)

    cex = detect_cex_listing_probability(pair)
    cex_prob = parse_float(cex.get("cex_prob", 0), 0.0)

    strength = 0.0
    risk = 0.0
    risk_flags: List[str] = []

    if score >= 300:
        strength += 3
    elif score >= 220:
        strength += 2
    elif score >= 180:
        strength += 1

    if signal == "GREEN":
        strength += 2
    elif signal == "YELLOW":
        strength += 1

    if smart_money:
        strength += 2
    if whale:
        strength += 1
    if migration_score > 0:
        strength += 1.5

    if cex_prob >= 4:
        strength += 2
    elif cex_prob >= 3:
        strength += 1

    if pc1h > 10:
        strength += 1
    if vol5 > 5000:
        strength += 1

    if liq < 100:
        risk += 10
        risk_flags.append("dead_liq")
    elif liq < 1000:
        risk += 6
        risk_flags.append("critical_liq")
    elif liq < 5000:
        risk += 3
        risk_flags.append("weak_liq")
    elif liq < 25000:
        risk += 1.5
        risk_flags.append("mid_liq")

    if trap_level == "CRITICAL":
        risk += 4
        risk_flags.append("critical_trap")
    elif trap_signal:
        risk += 2
        risk_flags.append(str(trap_signal).lower())

    if dev:
        risk += 2
        risk_flags.append("dev_risk")
    if sniper:
        risk += 2
        risk_flags.append("sniper_activity")

    if fresh == "VERY_NEW":
        risk += 2
        risk_flags.append("very_new_lp")
    elif fresh == "NEW":
        risk += 1
        risk_flags.append("new_lp")

    if hist and len(hist) >= 3:
        try:
            scores = [parse_float(x.get("score_live", 0), 0.0) for x in hist[-3:]]
            if scores[-1] < scores[0]:
                risk += 1
                risk_flags.append("score_fading")
        except Exception:
            pass

    buys5 = int(safe_get(pair, "txns", "m5", "buys", default=0) or 0)
    sells5 = int(safe_get(pair, "txns", "m5", "sells", default=0) or 0)
    if sells5 > buys5 * 1.2:
        risk += 1.5
        risk_flags.append("sell_pressure")

    total = strength - risk

    if (signal == "RED" and score < 200) or trap_level == "CRITICAL":
        size_pct = 0.0
        label = "SKIP"
    elif total >= 6:
        size_pct = 0.10
        label = "LARGE"
    elif total >= 4:
        size_pct = 0.07
        label = "MEDIUM"
    elif total >= 2:
        size_pct = 0.04
        label = "SMALL"
    elif total >= 1:
        size_pct = 0.02
        label = "PROBE"
    else:
        size_pct = 0.0
        label = "SKIP"

    if liq < 15000 and size_pct > 0.04:
        size_pct = 0.04
        label = "SMALL"

    if fresh == "VERY_NEW" and size_pct > 0.02:
        size_pct = 0.02
        label = "PROBE"

    usd_size = round(portfolio_value_usd * size_pct, 2)

    return {
        "size_pct": round(size_pct * 100, 1),
        "size_label": label,
        "usd_size": usd_size,
        "risk_score": round(risk, 1),
        "risk_flags": risk_flags[:5],
    }


# =============================
# v0.6.0 – automation / scheduler helpers
# =============================

def scanner_acquire_lock(slot: int, ttl_sec: int = 240) -> bool:
    """
    Prevent multiple tabs running the scanner simultaneously.
    Uses Supabase app_storage as a distributed lock.
    """
    if not _sb_ok():
        return True

    key = f"scanner_lock_{slot}"

    try:
        blob = sb_get_storage(key)

        if blob:
            data = json.loads(blob)
            ts = float(data.get("ts", 0))

            if ts and time.time() - ts < ttl_sec:
                return False

        payload = json.dumps({
            "ts": time.time(),
            "host": os.getenv("HOSTNAME", "local"),
            "version": VERSION
        })
        sb_put_storage(key, payload)

        return True

    except Exception:
        return True

SCOUT_WINDOWS = [
    ("Ultra Early (safer)", "ultra"),
    ("Balanced (default)", "balanced"),
    ("Wide Net (explore)", "wide"),
    ("Momentum (hot)", "momentum"),
]

SCAN_ROTATION = [
    ("Ultra Early (safer)", "ultra", "solana"),
    ("Balanced (default)", "balanced", "solana"),
    ("Wide Net (explore)", "wide", "solana"),
    ("Momentum (hot)", "momentum", "solana"),
    ("Ultra Early (safer)", "ultra", "bsc"),
    ("Balanced (default)", "balanced", "bsc"),
    ("Wide Net (explore)", "wide", "bsc"),
    ("Momentum (hot)", "momentum", "bsc"),
]

def _safe_dt_parse(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime((s or "").replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _merge_csv_values(old: str, new: str) -> str:
    vals = [x.strip() for x in (old or "").split(",") if x.strip()]
    if new and new not in vals:
        vals.append(new)
    return ",".join(vals)

def suggest_entry_and_tp_usd(p: Optional[Dict[str, Any]], risk: str = "") -> Tuple[str, str]:
    if not p:
        return ("", "")
    liq = float(safe_get(p, "liquidity", "usd", default=0) or 0)
    vol24 = float(safe_get(p, "volume", "h24", default=0) or 0)
    base = 6.0
    base += min(liq / 50_000.0 * 6.0, 10.0)
    base += min(vol24 / 250_000.0 * 4.0, 6.0)
    if (risk or "").upper() == "EARLY":
        base *= 0.75
    entry = max(3.0, min(base, 20.0))
    tp = 40.0 if (risk or "").upper() == "EARLY" else 25.0
    return (f"{entry:.0f}", f"{tp:.0f}")


TG_NEW_ENGINE_ONLY = True

def send_telegram(text: str, parse_mode: str = "HTML", reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    token = get_tg_token()
    chat_id = get_tg_chat_id()

    if not token or not chat_id:
        debug_log("tg_not_configured")
        print("[TG] no token/chat_id -> skip", flush=True)
        return False

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        print("[TG] sending...", flush=True)
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=10,
        )
        ok = r.status_code == 200
        debug_log(f"tg_send status={r.status_code} ok={ok}")
        print(f"[TG] sent status={r.status_code} ok={ok}", flush=True)
        return ok
    except Exception as e:
        debug_log(f"tg_send_exception {type(e).__name__}:{e}")
        print(f"[TG] exception {type(e).__name__}: {e}", flush=True)
        return False


def load_tg_state() -> Dict[str, Any]:
    print("[TG] loading state", flush=True)

    default = {
        "last_signal_run": 0.0,
        "last_scan_ts_processed": "",
        "sent_events": {},
        "sent_portfolio_events": {},
        "engine_version": "v3",
    }

    try:
        raw = sb_get_storage(TG_STATE_KEY) if USE_SUPABASE else None
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in default.items():
                    data.setdefault(k, v)
                if data.get("engine_version") != "v3":
                    data["sent_events"] = {}
                    data["sent_portfolio_events"] = {}
                    data["last_scan_ts_processed"] = ""
                    data["engine_version"] = "v3"
                return data
    except Exception as e:
        print(f"[TG] load state fallback {type(e).__name__}: {e}", flush=True)

    return default


def save_tg_state(state: Dict[str, Any]) -> None:
    print("[TG] saving state", flush=True)

    try:
        payload = json.dumps(state, ensure_ascii=False)
        if USE_SUPABASE:
            sb_put_storage(TG_STATE_KEY, payload)
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(TG_STATE_FILE, "w", encoding="utf-8") as f:
                f.write(payload)
    except Exception as e:
        print(f"[TG] save state fail {type(e).__name__}: {e}", flush=True)


def reset_tg_state() -> None:
    state = {
        "last_signal_run": 0.0,
        "last_scan_ts_processed": "",
        "sent_events": {},
        "sent_portfolio_events": {},
        "engine_version": "v3",
        "token_state": {},
    }
    save_tg_state(state)


def tg_cooldown_ok(state: Dict[str, Any], seconds: int = 3600) -> bool:
    now = time.time()
    last = float(state.get("last_signal_run", 0.0) or 0.0)
    if now - last < seconds:
        return False
    state["last_signal_run"] = now
    return True


def classify_monitoring_signal(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    score = parse_float(row.get("entry_score", 0), 0.0)
    action = str(row.get("entry_action") or row.get("entry") or "").upper()
    pullback_1 = parse_float(row.get("pullback_1"), 0.0)
    pullback_2 = parse_float(row.get("pullback_2"), 0.0)
    entry_now = parse_float(row.get("entry_now"), 0.0)
    has_levels = any([entry_now, pullback_1, pullback_2])

    if score <= 0:
        return None

    if action in ("ENTRY_NOW", "READY"):
        return {"bucket": "ENTRY_NOW", "horizon": "0-30m", "action": "Entry now"}

    if action in ("WATCH_PULLBACK", "WATCH", "EARLY") and score >= 120 and has_levels:
        return {"bucket": "WATCH", "horizon": "0-2h", "action": "Watch pullback"}

    if action in ("TRACK", "WAIT") and score >= 90:
        return {"bucket": "TRACK", "horizon": "2-12h", "action": "Track"}

    return None


def classify_portfolio_signal(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    score = parse_float(row.get("entry_score", 0), 0.0)
    risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()
    action = str(row.get("entry_action") or row.get("entry") or "").upper()

    if score <= 0:
        return None

    if risk == "HIGH" and action in ("AVOID", "NO_ENTRY"):
        return {"bucket": "EXIT", "horizon": "now", "action": "Consider exit"}

    if action in ("WATCH_PULLBACK", "TRACK", "WAIT", "EARLY") and score >= 100:
        return {"bucket": "HOLD", "horizon": "2-12h", "action": "Hold / monitor"}

    if action in ("ENTRY_NOW", "READY") and score >= 180:
        return {"bucket": "ADD", "horizon": "0-2h", "action": "Consider add"}

    return None


def signal_event_key(source: str, row: Dict[str, Any], signal: Dict[str, str]) -> str:
    chain = token_chain(row)
    addr = str(token_ca(row) or row.get("token_address") or "").strip().lower()
    action = str(row.get("entry_action") or row.get("entry") or "").upper()
    risk = str(row.get("risk_level") or row.get("risk") or "").upper()
    timing = normalize_timing_label(row.get("timing_label") or "")
    score_bucket = int(parse_float(row.get("entry_score"), 0.0) // 25)
    return f"{source}|{chain}|{addr}|{action}|{risk}|{timing}|{score_bucket}"


def build_token_state_key(row: Dict[str, Any]) -> str:
    chain = str(row.get("chain") or "").strip().lower()
    addr = str(
        row.get("base_addr")
        or row.get("pair_address")
        or row.get("pairAddress")
        or ""
    ).strip()
    if not chain or not addr:
        return ""
    return f"{chain}:{addr}"


def is_material_signal_change(prev: Optional[Dict[str, Any]], current: Dict[str, Any]) -> bool:
    if not prev:
        return True

    prev_action = str(prev.get("entry_action") or "").upper()
    curr_action = str(current.get("entry_action") or "").upper()
    if prev_action != curr_action:
        return True

    prev_risk = str(prev.get("risk_level") or "").upper()
    curr_risk = str(current.get("risk_level") or "").upper()
    if prev_risk != curr_risk:
        return True

    prev_score = parse_float(prev.get("entry_score", 0), 0.0)
    curr_score = parse_float(current.get("entry_score", 0), 0.0)
    if abs(curr_score - prev_score) >= 35:
        return True

    prev_timing = normalize_timing_label(prev.get("timing_label") or "")
    curr_timing = normalize_timing_label(current.get("timing_label") or "")
    if prev_timing != curr_timing:
        return True

    return False


def format_signal_message(row: Dict[str, Any], signal: Dict[str, str], source: str) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
    chain = token_chain(row).upper()
    addr = token_ca(row)
    source = str(source or "monitoring").lower()
    unified = compute_unified_recommendation(row, source="monitoring" if source == "monitoring" else "portfolio")

    score = parse_float(row.get("entry_score"), 0.0)
    risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()
    timing = normalize_timing_label(str(row.get("timing_label") or "NEUTRAL"))
    reason = str(unified.get("final_reason") or row.get("entry_reason") or row.get("signal_reason") or "n/a")
    horizon = str(row.get("entry_horizon") or signal.get("horizon") or "n/a")

    entry_now = parse_float(row.get("entry_now"), 0.0)
    pullback_1 = parse_float(row.get("pullback_1"), 0.0)
    pullback_2 = parse_float(row.get("pullback_2"), 0.0)
    invalidation = parse_float(row.get("invalidation"), 0.0)
    tp1 = parse_float(row.get("tp1"), 0.0)
    tp2 = parse_float(row.get("tp2"), 0.0)

    if any([entry_now, pullback_1, pullback_2, invalidation, tp1, tp2]):
        levels_block = (
            f"entry now: {entry_now}\n"
            f"pullback 1: {pullback_1}\n"
            f"pullback 2: {pullback_2}\n"
            f"invalidation: {invalidation}\n"
            f"tp1: {tp1}\n"
            f"tp2: {tp2}"
        )
    else:
        levels_block = "setup: watch only"

    if source == "portfolio":
        return format_portfolio_signal_message(row)

    return (
        f"<b>{unified['final_action']}</b> | <b>{symbol}</b>\n"
        f"source: MONITOR\n"
        f"chain: {chain}\n"
        f"score: <b>{score}</b>\n"
        f"risk: <b>{risk}</b>\n"
        f"timing: <b>{timing}</b>\n"
        f"horizon: {horizon}\n"
        f"reason: {reason}\n"
        f"{levels_block}\n\n"
        f"CA:\n<code>{addr}</code>"
    )


def format_portfolio_signal_message(row: Dict[str, Any]) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
    chain = token_chain(row).upper()
    addr = token_ca(row)
    unified = compute_unified_recommendation(row, source="portfolio")
    return (
        f"<b>{unified['final_action']}</b> | <b>{symbol}</b>\n"
        f"source: PORTFOLIO\n"
        f"chain: {chain}\n"
        f"reason: {unified['final_reason']}\n\n"
        f"CA:\n<code>{addr}</code>"
    )


def tg_buttons(row: Dict[str, Any]) -> Dict[str, Any]:
    chain = token_chain(row)
    ca = token_ca(row)
    dex_url = dex_url_for_token(chain, ca)

    if not chain or not ca:
        return {"inline_keyboard": []}

    buttons = [
        [
            {"text": "➕ Portfolio", "callback_data": f"pf_add|{chain}|{ca}"},
            {"text": "👀 Monitor", "callback_data": f"mon_add|{chain}|{ca}"},
        ],
        [
            {"text": "➖ Remove", "callback_data": f"remove|{chain}|{ca}"}
        ],
    ]

    if dex_url:
        buttons.append([
            {"text": "📈 Dex", "url": dex_url}
        ])

    return {"inline_keyboard": buttons}


def build_notification_candidates(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    limit_monitoring: int = 12,
    limit_portfolio: int = 8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def token_addr(row: Dict[str, Any]) -> str:
        chain = str(row.get("chain") or "").strip().lower()
        ca = str(
            row.get("base_addr")
            or row.get("base_token_address")
            or row.get("ca")
            or row.get("address")
            or ""
        ).strip()
        return f"{chain}:{addr_store(chain, ca)}" if chain and ca else ""

    mon: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for row in build_active_monitoring_rows(monitoring_rows):
        key = token_addr(row)
        if not key or key in seen:
            continue
        seen.add(key)

        if is_token_suppressed(str(row.get("chain") or ""), str(row.get("base_addr") or row.get("ca") or "")):
            continue

        score = parse_float(row.get("entry_score", 0), 0.0)
        if score <= 0:
            continue

        mon.append(row)

    mon.sort(key=lambda r: parse_float(r.get("entry_score", 0), 0.0), reverse=True)
    mon = mon[:limit_monitoring]

    port: List[Dict[str, Any]] = []
    seen_port: Set[str] = set()

    for row in portfolio_rows:
        if str(row.get("active", "1")).strip() != "1":
            continue
        key = token_addr(row)
        if not key or key in seen_port:
            continue
        if is_token_suppressed(str(row.get("chain") or ""), str(token_ca(row) or "")):
            continue
        seen_port.add(key)
        port.append(row)

    port.sort(key=lambda r: parse_float(r.get("entry_score", 0), 0.0), reverse=True)
    port = port[:limit_portfolio]

    return mon, port


def run_auto_notifications(
    scan_state: Dict[str, Any],
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
) -> None:
    state = load_tg_state()

    cooldown_seconds = 900 if WORKER_FAST_MODE else 1800
    if not tg_cooldown_ok(state, seconds=cooldown_seconds):
        print("[TG] cooldown active -> skip", flush=True)
        return

    prev_token_state = state.get("token_state", {})
    if not isinstance(prev_token_state, dict):
        prev_token_state = {}

    sent_events = state.get("sent_events", {})
    if not isinstance(sent_events, dict):
        sent_events = {}

    sent_now = 0
    max_per_run = 3 if WORKER_FAST_MODE else 5
    new_token_state: Dict[str, Dict[str, Any]] = {}

    mon_rows, port_rows = build_notification_candidates(monitoring_rows, portfolio_rows)

    candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for row in mon_rows:
        candidates.append((parse_float(row.get("entry_score", 0), 0.0), "monitoring", row))
    for row in port_rows:
        candidates.append((parse_float(row.get("entry_score", 0), 0.0), "portfolio", row))

    candidates.sort(key=lambda x: x[0], reverse=True)

    stats = {
        "total": len(candidates),
        "no_token_key": 0,
        "no_signal": 0,
        "no_msg": 0,
        "already_sent": 0,
        "sent": 0,
    }

    for _, source, row in candidates:
        chain = token_chain(row)
        ca = token_ca(row)
        if not chain or not ca:
            continue
        if is_token_suppressed(chain, ca):
            continue

        token_key = build_token_state_key(row)
        if not token_key:
            stats["no_token_key"] += 1
            continue

        unified = compute_unified_recommendation(row, source=source)
        current_snapshot = {
            "entry_action": str(unified.get("final_action") or "").upper(),
            "entry_score": parse_float(row.get("entry_score", 0), 0.0),
            "risk_level": str(row.get("risk_level") or row.get("risk") or "").upper(),
            "timing_label": str(unified.get("timing") or "NEUTRAL"),
            "source": source,
        }

        prev = prev_token_state.get(token_key)
        change = is_material_signal_change(prev, current_snapshot)
        signal = classify_monitoring_signal(row) if source == "monitoring" else classify_portfolio_signal(row)

        # always refresh in-memory state snapshot
        new_token_state[token_key] = current_snapshot

        if not signal:
            stats["no_signal"] += 1
            continue

        event_key = signal_event_key(source, row, signal)

        # allow first send even if change is None, unless event already sent
        if not change and event_key in sent_events:
            stats["already_sent"] += 1
            continue

        msg = format_signal_message(row, signal, source)
        if not msg:
            stats["no_msg"] += 1
            continue

        if send_telegram(msg, reply_markup=tg_buttons(row)):
            sent_events[event_key] = now_utc_str()
            sent_now += 1
            stats["sent"] += 1

        if sent_now >= max_per_run:
            break

    print(f"[TG] notification stats: {stats}", flush=True)

    state["token_state"] = new_token_state
    state["sent_events"] = sent_events
    state["last_scan_ts_processed"] = str(scan_state.get("last_run_ts") or now_utc_str())
    save_tg_state(state)


def reset_monitoring():
    st.session_state.setdefault("_last_status", {})


def is_signal_worthy(row: Dict[str, Any]) -> bool:
    score = parse_float(row.get("entry_score", 0), 0.0)
    risk = str(row.get("risk_level") or row.get("risk") or "").upper()
    weak_reason = str(row.get("weak_reason") or "").lower()

    if score <= 0:
        return False
    if "gate_blocked" in weak_reason and score < 60:
        return False
    if risk == "HIGH" and score < 80:
        return False
    return True

def scout_collect_candidates(chain: str, window_name: str, preset: Dict[str, Any], seeds_raw: str, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> List[Dict[str, Any]]:
    chain = normalize_chain_name(chain or "solana")
    cache_key = safe_json({
        "chain": chain,
        "window_name": window_name,
        "preset": preset,
        "seeds_raw": seeds_raw,
        "use_birdeye_trending": bool(use_birdeye_trending),
        "birdeye_limit": int(birdeye_limit),
    })
    cached = SCAN_CACHE.get(cache_key)
    now_ts = time.time()
    if cached and (now_ts - float(cached.get("ts", 0.0))) < CACHE_TTL:
        return list(cached.get("pairs", []))

    seeds = [x.strip() for x in (seeds_raw or "").split(",") if x.strip()]
    if not seeds:
        seeds = [x.strip() for x in str(preset.get("seeds", DEFAULT_SEEDS)).split(",") if x.strip()]
    if not seeds:
        seeds = DEX_SEARCH_TERMS

    sampled = sample_seeds(seeds, int(preset.get("seed_k", 16)), refresh=True)
    all_pairs: List[Dict[str, Any]] = []

    search_terms = list(dict.fromkeys(sampled + DEX_SEARCH_TERMS[:20]))
    for q in search_terms:
        try:
            all_pairs.extend(fetch_dexscreener_search(q, MAX_DEX_SEARCH_PER_TERM))
            time.sleep(0.05)
        except Exception as e:
            debug_log(f"dex_search_fail q={q} err={type(e).__name__}:{e}")

    try:
        all_pairs.extend(fetch_dexscreener_trending(chain, limit=60))
    except Exception:
        pass

    if chain == "bsc":
        try:
            all_pairs.extend(fetch_dexscreener_latest(chain, limit=60))
        except Exception:
            pass

    if use_birdeye_trending and chain == "solana":
        try:
            all_pairs.extend(fetch_birdeye_pairs(limit=int(birdeye_limit)))
        except Exception as e:
            debug_log(f"birdeye_fail {type(e).__name__}:{e}")

    try:
        unified_tokens = fetch_tokens_unified(chain=chain, limit=80)
        for t in unified_tokens:
            symbol = t.get("symbol", "NA")
            address = t.get("address", "")
            price = parse_float(t.get("price", 0), 0.0)
            volume = parse_float(t.get("volume", 0), 0.0)
            liquidity = parse_float(t.get("liquidity", 0), 0.0)
            if not address:
                continue
            all_pairs.append({
                "chainId": chain,
                "dexId": str(t.get("source") or "unified"),
                "pairAddress": address,
                "baseToken": {"symbol": symbol, "address": address},
                "quoteToken": {"symbol": "USDC" if chain == "solana" else "USDT"},
                "priceUsd": price,
                "volume": {"h24": volume, "m5": 0, "h1": 0},
                "liquidity": {"usd": liquidity},
                "txns": {"m5": {"buys": 0, "sells": 0}},
                "priceChange": {"m5": 0, "h1": 0},
                "url": "",
            })
        debug_log(f"unified_tokens_added count={len(unified_tokens)}")
    except Exception as e:
        debug_log(f"unified_tokens_fail {type(e).__name__}:{e}")

    target_chain = str(chain or "").lower().strip()
    all_pairs = [
        p for p in all_pairs
        if str(p.get("chainId") or p.get("chain") or "").lower().strip() == target_chain
    ]

    def is_not_solana_address(addr: str) -> bool:
        if not addr:
            return True
        addr = str(addr).strip()
        # solana addresses ~32-44 chars, no 0x
        if addr.startswith("0x"):
            return True
        if len(addr) < 30:
            return True
        return False

    if target_chain == "solana":
        all_pairs = [
            p for p in all_pairs
            if not is_not_solana_address(safe_get(p, "baseToken", "address", default=""))
        ]

    all_pairs = dedupe_mode(all_pairs, by_base_token=False)
    all_pairs = sorted(
        all_pairs,
        key=lambda p: (
            parse_float(safe_get(p, "txns", "m5", "buys", default=0), 0.0),
            parse_float(safe_get(p, "volume", "m5", default=0), 0.0),
            parse_float(safe_get(p, "liquidity", "usd", default=0), 0.0),
        ),
        reverse=True,
    )
    all_pairs = all_pairs[:INGEST_TARGET_KEEP]
    # SOFT FILTER – do not kill flow
    filtered = []
    for p in all_pairs:
        try:
            liq = parse_float(safe_get(p, "liquidity", "usd", default=0), 0.0)
            vol24 = parse_float(safe_get(p, "volume", "h24", default=0), 0.0)
            buys = parse_int(safe_get(p, "txns", "m5", "buys", default=0), 0)
            sells = parse_int(safe_get(p, "txns", "m5", "sells", default=0), 0)
            txns = buys + sells
            if liq <= 0 and vol24 <= 0:
                continue
            filtered.append(p)
        except Exception:
            continue
    debug_log(f"soft_filter_kept={len(filtered)} from_raw={len(all_pairs)}")

    filtered = [p for p in filtered if not looks_like_quote_or_lp(p)]
    filtered = dedupe_mode(filtered, by_base_token=False)
    if len(filtered) == 0:
        filtered = all_pairs[:20]
    debug_log(f"collect_candidates raw={len(all_pairs)} filtered={len(filtered)}")

    out = []
    for p in filtered:
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
        if not base_addr:
            continue
        row = normalize_pair_row(p)
        if row is None:
            continue
        out.append(row)

    out = dedupe_tokens_by_address(out)
    SCAN_CACHE[cache_key] = {"ts": now_ts, "pairs": out}
    return out

def ingest_window_to_monitoring(chain: str, window_name: str, preset_key: str, seeds_raw: str, max_items: int = 100, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> Dict[str, int]:
    chain = (chain or "solana").strip().lower()
    preset = PRESETS.get(window_name, PRESETS.get("Wide Net (explore)", {}))
    counts = {"added": 0, "updated_active": 0, "skipped_active": 0, "skipped_archived": 0, "skipped_noise": 0, "skipped_major_like": 0, "skipped_symbol_duplicate": 0, "skipped_quote_like": 0, "skipped_wrong_chain": 0, "skipped_major": 0, "skipped_suppressed": 0, "errors": 0, "seen": 0}
    pairs = scout_collect_candidates(chain=chain, window_name=window_name, preset=preset, seeds_raw=seeds_raw, use_birdeye_trending=use_birdeye_trending, birdeye_limit=birdeye_limit)
    debug_log(f"ingest_pairs_raw={len(pairs)}")
    normalized = [r for r in pairs if r is not None]
    normalized = normalized[: max(30, MAX_KEEP)]
    debug_log(f"ingest_pairs_normalized={len(normalized)}")
    debug_log(f"ingest_pairs_signalworthy={len([r for r in normalized if is_signal_worthy(r)])}")
    ranked = []
    smart_wallets = load_smart_wallets()
    for p in normalized:
        smart = detect_smart_money(p)
        signal = classify_signal(p)
        score = score_pair(p)
        if smart:
            score += 4
        ranked.append((score, p, smart, signal))
    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[: max(50, int(max_items))]
    if len(ranked) == 0 and pairs:
        fallback_ranked = []
        for p in pairs[:5]:
            s = score_pair(p)
            fallback_ranked.append((s, p, detect_smart_money(p), classify_signal(p)))
        ranked = fallback_ranked
    seen_token_keys: Set[str] = set()
    existing_rows = load_monitoring()
    active_symbols = {
        str(r.get("base_symbol") or "").strip().upper()
        for r in existing_rows
        if str(r.get("active", "1")).strip() == "1"
    }
    for s, p, smart, signal in ranked:
        counts["seen"] += 1
        row = normalize_pair_row(p)
        if row is None:
            counts["skipped_noise"] += 1
            continue
        toxic, reasons = is_toxic_token(p)
        if toxic:
            row["risk"] = "HIGH"
            row["decision_reason"] = reasons[0]
            row["toxic_flags"] = ",".join(reasons)
        base_sym = str(safe_get(row, "baseToken", "symbol", default="") or row.get("base_symbol") or "").strip().upper()
        row_chain = normalize_chain_name(row.get("chainId") or row.get("chain"))
        row["chainId"] = row_chain
        row["chain"] = row_chain

        if row_chain not in ALLOWED_CHAINS:
            counts["skipped_noise"] += 1
            counts["skipped_wrong_chain"] += 1
            continue

        symbol_lc = base_sym.lower()
        if symbol_lc in {"usdt", "usdc", "eth", "btc", "bnb"}:
            counts["skipped_noise"] += 1
            counts["skipped_major"] += 1
            continue

        if base_sym in MAJORS_STABLES:
            counts["skipped_noise"] += 1
            counts["skipped_major"] += 1
            continue

        if base_sym in {"ETH", "USDT", "USDC", "BTC", "BNB", "SOL"}:
            counts["skipped_noise"] += 1
            counts["skipped_major"] += 1
            continue

        if is_symbol_major_like(base_sym):
            counts["skipped_noise"] += 1
            counts["skipped_major_like"] += 1
            continue

        if looks_like_quote_or_lp(row):
            counts["skipped_noise"] += 1
            counts["skipped_quote_like"] += 1
            continue

        if is_major_or_stable(row) and float(s) < 120.0:
            counts["skipped_noise"] += 1
            counts["skipped_major_like"] += 1
            continue

        if base_sym in active_symbols:
            counts["skipped_symbol_duplicate"] += 1

        chain_id = normalize_chain_name(row.get("chainId") or row.get("chain"))
        base_addr = (safe_get(row, "baseToken", "address", default="") or row.get("base_addr") or "").strip()
        dedupe_key = f"{chain_id}:{base_addr}" if chain_id and base_addr else ""
        if dedupe_key:
            if dedupe_key in seen_token_keys:
                counts["skipped_noise"] += 1
                counts["skipped_symbol_duplicate"] += 1
                continue
            seen_token_keys.add(dedupe_key)
        s = float(s)
        row["entry_score"] = s
        if not is_signal_worthy(row):
            counts["skipped_noise"] += 1
            row["weak_reason"] = "|".join(sorted(set(filter(None, [
                str(row.get("weak_reason") or ""),
                "weak_signal"
            ]))))
        entry_status = str(row.get("entry_status") or "WAIT")
        row["signal_reason"] = str(row.get("entry_reason") or signal or "")
        entry_score = parse_float(row.get("entry_score", 0), 0.0)
        decision = "watch"
        row["smart_money"] = smart
        row["signal"] = signal
        gate_ok, gate_reason = passes_monitoring_gate(row, s)
        row["visible_score"] = s
        row["bucket"] = "watch"
        weak_reasons: List[str] = []

        if not gate_ok:
            gate_tag = str(gate_reason or "gate_blocked").strip()
            row["weak_reason"] = gate_tag if gate_tag.startswith("gate:") else f"gate:{gate_tag}"
        priority = monitoring_priority(
            score_live=s,
            pc1h=parse_float(safe_get(row, "priceChange", "h1", default=0), 0.0),
            pc5=parse_float(safe_get(row, "priceChange", "m5", default=0), 0.0),
            vol5=parse_float(safe_get(row, "volume", "m5", default=0), 0.0),
            liq=parse_float(safe_get(row, "liquidity", "usd", default=0), 0.0),
            meme_score=parse_float(row.get("meme_score", 0), 0.0),
            migration_score=parse_float(row.get("migration_score", 0), 0.0),
            cex_prob=parse_float(row.get("cex_prob", 0), 0.0),
        )
        if priority < 3:
            row["low_priority_flag"] = True
        if weak_reasons:
            row["weak_reason"] = "|".join(sorted(set(weak_reasons)))

        if smart:
            chain = (row.get("chainId") or "").lower()
            base_addr = (safe_get(row, "baseToken", "address", default="") or "").strip()
            key = addr_key(chain, base_addr)
            smart_wallets[key] = {
                "last_seen": now_utc_str(),
                "symbol": row.get("base_symbol", "")
            }
        try:
            contract = addr_store(row.get("chainId", ""), safe_get(row, "baseToken", "address", default=""))
            chain_norm = normalize_chain_name(row.get("chainId", "") or row.get("chain", ""))
            base_addr_norm = addr_store(chain_norm, contract)

            if is_token_suppressed(chain_norm, base_addr_norm):
                counts["skipped_suppressed"] = counts.get("skipped_suppressed", 0) + 1
                continue

            if token_exists(contract):
                update_existing_token(row, s, window_name=window_name, preset_key=preset_key, entry_status=entry_status, entry_score=entry_score, risk_level=str(row.get("risk_level", "MEDIUM")))
                counts["updated_active"] += 1
                continue
            res = add_to_monitoring(
                row,
                s,
                window_name=window_name,
                preset_key=preset_key,
                entry_status=entry_status,
                entry_score=entry_score,
                risk_level=str(row.get("risk_level", "MEDIUM")),
            )

            append_monitoring_history({
                "ts_utc": now_utc_str(),
                "chain": (row.get("chainId") or "").lower().strip(),
                "base_symbol": safe_get(row, "baseToken", "symbol", default="") or "",
                "base_addr": contract,
                "pair_addr": row.get("pairAddress", "") or "",
                "dex": row.get("dexId", "") or "",
                "quote_symbol": safe_get(row, "quoteToken", "symbol", default="") or "",
                "price_usd": str(parse_float(row.get("priceUsd"), 0.0)),
                "liq_usd": str(parse_float(safe_get(row, "liquidity", "usd", default=0), 0.0)),
                "vol24_usd": str(parse_float(safe_get(row, "volume", "h24", default=0), 0.0)),
                "vol5_usd": str(parse_float(safe_get(row, "volume", "m5", default=0), 0.0)),
                "pc1h": str(parse_float(safe_get(row, "priceChange", "h1", default=0), 0.0)),
                "pc5": str(parse_float(safe_get(row, "priceChange", "m5", default=0), 0.0)),
                "score_live": str(s),
                "decision": gate_reason if decision == "watch" else decision,
                "entry_score": str(row.get("entry_score", "")),
                "entry_action": str(row.get("entry_action", "")),
                "entry_reason": str(row.get("entry_reason", "")),
                "timing_label": str(row.get("timing_label", "")),
                "risk_level": str(row.get("risk_level", "")),
            })

            if res == "OK":
                counts["added"] += 1
                if base_sym:
                    active_symbols.add(base_sym)
            elif res == "EXISTS_ACTIVE":
                counts["skipped_active"] += 1
            elif res == "EXISTS_ARCHIVED":
                counts["skipped_archived"] += 1
            if entry_status in {"WATCH", "WAIT"} and detect_auto_signal(row):
                add_to_monitoring(
                    row,
                    s,
                    window_name="AUTO_SIGNAL",
                    preset_key="AUTO_SIGNAL",
                    entry_status=entry_status,
                    entry_score=entry_score,
                    risk_level=str(row.get("risk_level", "MEDIUM")),
                )
        except Exception as e:
            log_error(e)
            counts["errors"] += 1
    if counts["added"] == 0 and ranked:
        fallback = normalize_pair_row(ranked[0][1])
        if fallback is None:
            return counts
        fallback["entry_status"] = "WATCH"
        fallback["status"] = "ACTIVE"
        fallback["signal_reason"] = "fallback_signal"
        add_to_monitoring(
            fallback,
            float(ranked[0][0]),
            window_name=window_name,
            preset_key=preset_key,
            entry_status="WATCH",
            entry_score=parse_float(fallback.get("entry_score", ranked[0][0]), 0.0),
            risk_level=str(fallback.get("risk_level", "MEDIUM")),
        )
        counts["added"] += 1
    save_smart_wallets(smart_wallets)
    debug_log(f"RAW: {len(pairs)}")
    debug_log(f"NORMALIZED: {len(normalized)}")
    debug_log(f"FINAL: {counts['added']}")
    return counts

def auto_reactivate_archived(days: int = 7, max_revisits: int = 3) -> int:
    days = max(1, int(days))
    rows = load_monitoring()
    changed = 0
    now_dt = datetime.utcnow()

    for r in rows:
        if r.get("active") == "1":
            continue

        reason = (r.get("archived_reason") or "").strip().lower()
        if not reason:
            continue

        revisit_count = int(parse_float(r.get("revisit_count", 0), 0))
        if revisit_count >= max_revisits:
            continue

        ts_archived = _safe_dt_parse(r.get("ts_archived", ""))
        revisit_after = _safe_dt_parse(r.get("revisit_after_ts", ""))

        eligible = False
        if revisit_after:
            eligible = now_dt >= revisit_after
        elif ts_archived:
            eligible = ((now_dt - ts_archived).total_seconds() / 86400.0) >= float(days)

        if not eligible:
            continue

        reason_upper = reason.upper()
        if "RUG" in reason_upper or "DEAD" in reason_upper:
            continue

        if str(r.get("status", "")).upper() == "ARCHIVED":
            last = parse_ts(r.get("updated_at")) or parse_ts(r.get("ts_archived"))
            if last and (now_dt - last).days >= int(days):
                r["status"] = "ACTIVE"

        r["active"] = "1"
        r["ts_added"] = now_utc_str()
        r["ts_archived"] = ""
        r["archived_reason"] = ""
        r["ts_last_seen"] = now_utc_str()
        r["last_revisit_ts"] = now_utc_str()
        r["revisit_count"] = str(revisit_count + 1)
        r["revisit_after_ts"] = ""
        changed += 1

    if changed:
        save_monitoring(rows)
    return changed


def revisit_archived(min_days: int = 1, score_threshold: float = 400.0, max_revisits: int = 3) -> int:
    rows = load_monitoring()
    now_dt = datetime.utcnow()
    changed = 0
    for r in rows:
        if r.get("active") == "1":
            continue
        reason = str(r.get("archived_reason", "") or "").lower()
        if any(x in reason for x in ["promoted", "duplicate_portfolio", "dead", "rug", "critical"]):
            continue
        revisit_count = int(parse_float(r.get("revisit_count", 0), 0))
        if revisit_count >= max_revisits:
            continue
        ts_archived = _safe_dt_parse(r.get("ts_archived", ""))
        if not ts_archived:
            continue
        age_days = (now_dt - ts_archived).total_seconds() / 86400.0
        if age_days < float(min_days):
            continue

        chain = (r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, r.get("base_addr", ""))
        if not chain or not base_addr:
            continue
        best = best_pair_for_token_cached(chain, base_addr)
        live_score = score_pair(best) if best else 0.0
        if live_score <= score_threshold:
            continue

        r["active"] = "1"
        r["ts_added"] = now_utc_str()
        r["ts_archived"] = ""
        r["archived_reason"] = ""
        r["last_score"] = str(round(live_score, 2))
        r["last_revisit_ts"] = now_utc_str()
        r["revisit_count"] = str(revisit_count + 1)
        r["revisit_after_ts"] = ""
        r["weak_reason"] = ""
        r["portfolio_linked"] = "0"
        r["in_portfolio"] = "0"
        changed += 1

    if changed:
        save_monitoring(rows)
    return changed


def remove_stale_tokens(ttl_hours: int = 6) -> int:
    rows = load_monitoring()
    now_dt = datetime.utcnow()
    changed = 0
    for r in rows:
        if str(r.get("active", "1")).strip() != "1":
            continue
        ts_added = _safe_dt_parse(r.get("ts_added", "")) or _safe_dt_parse(r.get("ts_last_seen", ""))
        if not ts_added:
            continue
        age_hours = (now_dt - ts_added).total_seconds() / 3600.0
        if age_hours <= float(ttl_hours):
            continue
        r["active"] = "0"
        r["ts_archived"] = now_utc_str()
        r["archived_reason"] = "ttl_expired"
        changed += 1
    if changed:
        save_monitoring(rows)
    return changed


def monitoring_rank_for_trim(r: Dict[str, Any]) -> Tuple[float, float]:
    score = parse_float(r.get("priority_score", 0), 0.0)
    last_seen_dt = _safe_dt_parse(r.get("ts_last_seen", "")) or _safe_dt_parse(r.get("ts_added", ""))
    freshness = last_seen_dt.timestamp() if last_seen_dt else 0.0
    return (score, freshness)


def trim_active_monitoring(max_active: int = 150) -> int:
    rows = load_monitoring()
    active_rows = [r for r in rows if str(r.get("active", "1")).strip() == "1"]
    if len(active_rows) <= max_active:
        return 0
    active_sorted = sorted(active_rows, key=monitoring_rank_for_trim, reverse=True)
    keep_keys = {
        addr_key((r.get("chain") or "").strip().lower(), addr_store((r.get("chain") or "").strip().lower(), r.get("base_addr", "")))
        for r in active_sorted[:max_active]
    }
    changed = 0
    for r in rows:
        if str(r.get("active", "1")).strip() != "1":
            continue
        chain = (r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, r.get("base_addr", ""))
        if addr_key(chain, base_addr) in keep_keys:
            continue
        r["active"] = "0"
        r["ts_archived"] = now_utc_str()
        r["archived_reason"] = "rotation_trim"
        changed += 1
    if changed:
        save_monitoring(rows)
    return changed


def add_new_candidates() -> int:
    return revisit_archived(min_days=1, score_threshold=260.0, max_revisits=3)


def cleanup_monitoring_noise() -> int:
    rows = load_monitoring()
    changed = 0
    for r in rows:
        if str(r.get("active", "1")).strip() != "1":
            continue
        base_sym = str(r.get("base_symbol") or "").strip().upper()
        if is_symbol_major_like(base_sym):
            r["active"] = "0"
            r["ts_archived"] = now_utc_str()
            r["archived_reason"] = "noise_major_like"
            changed += 1
    if changed:
        save_monitoring(rows)
    return changed


def purge_toxic() -> int:
    rows = load_monitoring()
    changed = 0

    for r in rows:
        if str(r.get("active", "1")) != "1":
            continue
        toxic, reasons = is_toxic_token(r)
        if toxic:
            r["active"] = "0"
            r["archived_reason"] = f"toxic_{reasons[0]}"
            r["ts_archived"] = now_utc_str()
            r["decision_reason"] = reasons[0]
            r["toxic_flags"] = ",".join(reasons)
            changed += 1

    if changed:
        save_monitoring(rows)
    return changed


def purge_non_solana_and_majors() -> int:
    rows = load_monitoring()
    changed = 0

    for r in rows:
        if str(r.get("active", "1")) != "1":
            continue

        sym = str(r.get("base_symbol") or "").upper().strip()
        addr = str(r.get("base_addr") or "")

        if sym in {"ETH", "USDT", "USDC", "BTC", "BNB", "SOL"} or addr.startswith("0x"):
            r["active"] = "0"
            r["archived_reason"] = "purged_non_solana_or_major"
            r["ts_archived"] = now_utc_str()
            changed += 1

    if changed:
        save_monitoring(rows)

    return changed

def scanner_state_load() -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, "scanner_state.json")
    ensure_storage()

    # 1. Supabase
    if _sb_ok():
        try:
            blob = sb_get_storage("scanner_state.json")
            if blob:
                data = json.loads(blob)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    # 2. Local fallback
    if os.path.exists(path):
        try:
            txt = Path(path).read_text(encoding="utf-8")
            if txt:
                data = json.loads(txt)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    # 3. Default
    return {
        "last_slot": -1,
        "last_run_ts": "",
        "last_window": "",
        "last_preset": "",
        "last_chain": "",
        "last_stats": {}
    }

def scanner_state_save(state: Dict[str, Any]):
    path = os.path.join(DATA_DIR, "scanner_state.json")
    ensure_storage()
    try:
        Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    if _sb_ok():
        try:
            sb_put_storage("scanner_state.json", json.dumps(state, ensure_ascii=False))
        except Exception:
            pass

def current_scan_slot(now_ts: Optional[float] = None) -> Tuple[str, str, str, int]:
    if now_ts is None:
        now_ts = time.time()
    step = int(now_ts // 300)
    window_name, preset_key, chain = SCAN_ROTATION[step % len(SCAN_ROTATION)]
    return window_name, preset_key, chain, step

def maybe_run_rotating_scanner(seeds_raw: str, max_items: int = 100, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> Dict[str, Any]:
    state = scanner_state_load() or {}
    if not isinstance(state, dict):
        state = {}
    window_name, preset_key, chain, slot = current_scan_slot()
    last_slot = int(state.get("last_slot", -1) or -1)

    if last_slot == slot:
        return {
            "ran": False,
            "slot": slot,
            "window": state.get("last_window", window_name),
            "chain": state.get("last_chain", chain),
            "stats": state.get("last_stats", {}),
        }

    try:
        stats = ingest_window_to_monitoring(chain=chain, window_name=window_name, preset_key=preset_key, seeds_raw=seeds_raw, max_items=max_items, use_birdeye_trending=use_birdeye_trending, birdeye_limit=birdeye_limit)
        stats["stale_removed"] = 0
        stats["revisited"] = add_new_candidates()
        stats["trimmed"] = trim_active_monitoring(max_active=150)
    except Exception as e:
        log_error(e)
        return {"ran": False, "slot": slot, "window": window_name, "chain": chain, "stats": state.get("last_stats", {}), "error": str(e)}
    state.update({
        "last_slot": slot,
        "last_run_ts": now_utc_str(),
        "last_window": window_name,
        "last_preset": preset_key,
        "last_chain": chain,
        "last_stats": stats,
    })
    scanner_state_save(state)
    return {"ran": True, "slot": slot, "window": window_name, "chain": chain, "stats": stats}

def run_scanner_now(seeds_raw: str, max_items: int = 100, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> Dict[str, Any]:
    window_name, preset_key, chain, slot = current_scan_slot()
    try:
        stats = ingest_window_to_monitoring(chain=chain, window_name=window_name, preset_key=preset_key, seeds_raw=seeds_raw, max_items=max_items, use_birdeye_trending=use_birdeye_trending, birdeye_limit=birdeye_limit)
        stats["stale_removed"] = 0
        stats["revisited"] = add_new_candidates()
        stats["trimmed"] = trim_active_monitoring(max_active=150)
    except Exception as e:
        log_error(e)
        return {"ran": False, "slot": slot, "window": window_name, "chain": chain, "stats": {}, "error": str(e)}
    state = scanner_state_load()
    state.update({
        "last_slot": slot,
        "last_run_ts": now_utc_str(),
        "last_window": window_name,
        "last_preset": preset_key,
        "last_chain": chain,
        "last_stats": stats,
    })
    scanner_state_save(state)
    return {"ran": True, "slot": slot, "window": window_name, "chain": chain, "stats": stats}

def run_full_ingestion_now(chain: str, seeds_raw: str, max_items: int = 100, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> Dict[str, Any]:
    state = scanner_state_load() or {}
    window_name = "Wide Net (explore)"
    preset_key = "wide"
    stats = ingest_window_to_monitoring(
        chain=normalize_chain_name(chain or "solana"),
        window_name=window_name,
        preset_key=preset_key,
        seeds_raw=seeds_raw,
        max_items=max_items,
        use_birdeye_trending=use_birdeye_trending,
        birdeye_limit=birdeye_limit,
    )
    stats["stale_removed"] = 0
    stats["revisited"] = add_new_candidates()
    stats["trimmed"] = trim_active_monitoring(max_active=150)
    stats["cleanup_noise"] = 0
    stats["purged_toxic"] = 0
    state["last_window"] = window_name
    state["last_preset"] = preset_key
    state["last_chain"] = normalize_chain_name(chain or "solana")
    state["last_stats"] = stats
    state["last_run_ts"] = now_utc_str()
    scanner_state_save(state)
    return stats


def scan_source_window_name() -> str:
    state = scanner_state_load() or {}
    return str(state.get("last_window") or "manual")


def scan_source_preset_key() -> str:
    state = scanner_state_load() or {}
    return str(state.get("last_preset") or "manual")


def run_scanner_once(limit: int = 50) -> List[Dict[str, Any]]:
    chain = str(st.session_state.get("chain", "solana")).strip().lower() or "solana"
    tokens = fetch_tokens_unified(chain=chain, limit=limit)
    ts_now = now_utc_str()
    new_rows: List[Dict[str, Any]] = []

    def build_row(token: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = str(token.get("symbol") or "NA").strip()
        address = str(token.get("address") or "").strip()
        if not address:
            return None

        best = best_pair_for_token_cached(chain, address)
        pair = best if best else {
            "chainId": chain,
            "dexId": token.get("source", ""),
            "pairAddress": "",
            "baseToken": {"symbol": symbol, "address": address},
            "quoteToken": {"symbol": "USDC" if chain == "solana" else "USDT"},
            "priceUsd": token.get("price", 0),
            "volume": {"h24": token.get("volume", 0), "m5": 0, "h1": 0},
            "liquidity": {"usd": token.get("liquidity", 0)},
            "txns": {"m5": {"buys": 0, "sells": 0}},
            "priceChange": {"m5": 0, "h1": 0},
            "url": "",
        }

        liquidity = parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)
        volume_5m = parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)
        txns_raw = (pair.get("txns") or {}).get("m5", {}) or {}
        txns = int(parse_float(txns_raw.get("buys", 0), 0.0) + parse_float(txns_raw.get("sells", 0), 0.0))
        age_minutes = parse_float(pair_age_minutes(pair) or token.get("age_minutes", 0), 0.0)
        price_change_5m = parse_float(safe_get(pair, "priceChange", "m5", default=0), 0.0)

        score_live = 300.0
        if liquidity < 5000:
            score_live -= 30
        if volume_5m < 1000:
            score_live -= 20
        if txns < 20:
            score_live -= 20
        if age_minutes < 60:
            score_live += 50
        if price_change_5m > 5:
            score_live += 40
        score_live = max(score_live, 50.0)

        if score_live >= 500:
            entry = "READY"
        elif score_live >= 250:
            entry = "WATCH"
        else:
            entry = "WAIT"

        decision, _tags = build_trade_hint(pair)
        entry_status = entry
        entry_score = score_live

        top_holder_pct = parse_float(token.get("top_holder_pct", token.get("top_holder_percent", 0)), 0.0)
        buys = parse_float(txns_raw.get("buys", 0), 0.0)
        sells = parse_float(txns_raw.get("sells", 0), 0.0)
        risk_score = 0
        risk_flags: List[str] = []
        if liquidity < 2000:
            risk_score += 2
            risk_flags.append("low_liquidity")
        if top_holder_pct > 25:
            risk_score += 2
            risk_flags.append("whale")
        if sells > buys * 2:
            risk_score += 1
            risk_flags.append("sell_pressure")
        if volume_5m < 100 and txns < 5:
            risk_score += 2
            risk_flags.append("dead")

        if risk_score >= 5:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        append_monitoring_history({
            "ts_utc": ts_now,
            "chain": chain,
            "base_symbol": symbol,
            "base_addr": address,
            "pair_addr": pair.get("pairAddress", "") or "",
            "dex": pair.get("dexId", "") or "",
            "quote_symbol": safe_get(pair, "quoteToken", "symbol", default="") or "",
            "price_usd": str(parse_float(pair.get("priceUsd"), 0.0)),
            "liq_usd": str(parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)),
            "vol24_usd": str(parse_float(safe_get(pair, "volume", "h24", default=0), 0.0)),
            "vol5_usd": str(parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)),
            "pc1h": str(parse_float(safe_get(pair, "priceChange", "h1", default=0), 0.0)),
            "pc5": str(parse_float(safe_get(pair, "priceChange", "m5", default=0), 0.0)),
            "score_live": str(score_live),
            "decision": decision,
            "entry_score": str(entry_score),
            "entry_action": str(entry_status),
            "entry_reason": "",
            "timing_label": "",
            "risk_level": str(risk_level),
        })

        row = {
            "ts_added": ts_now,
            "chain": chain,
            "base_symbol": symbol,
            "base_addr": address,
            "pair_addr": pair.get("pairAddress", "") or "",
            "score_init": str(score_live),
            "liq_init": str(parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)),
            "vol24_init": str(parse_float(safe_get(pair, "volume", "h24", default=0), 0.0)),
            "vol5_init": str(parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)),
            "active": "1",
            "ts_archived": "",
            "archived_reason": "",
            "last_score": str(score_live),
            "last_decision": decision,
            "priority_score": str(score_live),
            "last_decay_ts": ts_now,
            "decay_hits": "0",
            "source_window": scan_source_window_name(),
            "source_preset": scan_source_preset_key(),
            "risk": risk_level,
            "tp_target_pct": "25",
            "entry_suggest_usd": "10",
            "ts_last_seen": ts_now,
            "signal": classify_signal(pair),
            "smart_money": "1" if detect_smart_money(pair) else "0",
            "entry_status": entry_status,
            "entry_score": str(entry_score),
            "risk_level": risk_level,
            "entry_state": "READY" if entry_status == "READY" else "WAIT",
            "revisit_count": "0",
            "revisit_after_ts": "",
            "last_revisit_ts": "",
            "status": "ACTIVE",
            "entry": entry,
            "risk_score": str(risk_score),
            "risk_flags": ",".join(risk_flags),
            "why": f"score={score_live:.2f} risk={risk_level} flags={risk_flags}",
        }
        return row

    seen_token_keys: Set[Tuple[str, str]] = set()
    deduped_tokens: List[Dict[str, Any]] = []
    for t in tokens:
        t_chain = str(t.get("chain") or chain).strip().lower()
        t_address = addr_store(t_chain, str(t.get("address") or "").strip())
        t_symbol = str(t.get("symbol") or "").strip().lower()
        dedupe_value = t_address or t_symbol
        if not dedupe_value:
            continue
        key = (t_chain, dedupe_value)
        if key in seen_token_keys:
            continue
        seen_token_keys.add(key)
        deduped_tokens.append(t)

    for t in deduped_tokens:
        row = build_row(t)
        if not row:
            continue
        row["status"] = "ACTIVE"
        row["entry"] = row.get("entry", "WAIT")
        new_rows.append(row)

    return new_rows


def persist_scanner_result(scanner_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    existing = load_monitoring()
    existing_keys: Set[Tuple[str, str]] = set()
    for row in existing:
        chain = str(row.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, str(row.get("base_addr") or row.get("address") or "").strip())
        if chain and base_addr:
            existing_keys.add((base_addr, chain))

    new_rows: List[Dict[str, Any]] = []
    inserted = 0
    updated = 0
    for row in scanner_rows:
        chain = str(row.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, str(row.get("base_addr") or row.get("address") or "").strip())
        if not chain or not base_addr:
            continue
        normalized = {k: row.get(k, "") for k in MON_FIELDS}
        normalized["active"] = "1"
        normalized["status"] = "ACTIVE"
        normalized["entry"] = normalized.get("entry") or "WAIT"
        normalized["chain"] = chain
        normalized["base_addr"] = base_addr
        new_rows.append(normalized)
        if (base_addr, chain) in existing_keys:
            updated += 1
        else:
            inserted += 1

    combined = existing + new_rows
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for r in combined:
        chain = str(r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, str(r.get("base_addr") or r.get("address") or "").strip())
        key = (base_addr, chain)
        if not chain or not base_addr:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    save_monitoring(deduped)
    st.write("DEBUG AFTER SAVE:", len(load_monitoring()))
    return {"inserted": inserted, "updated": updated, "total": len(deduped)}

def source_priority(row: Dict[str, Any]) -> int:
    presets = [x.strip() for x in str(row.get("source_preset", "")).split(",") if x.strip()]
    if "momentum" in presets:
        return 4
    if "ultra" in presets:
        return 3
    if "balanced" in presets:
        return 2
    if "wide" in presets:
        return 1
    return 0

def confidence_stars(best: Optional[Dict[str, Any]], hist: List[Dict[str, Any]]) -> str:
    if not best:
        return ""
    liq = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)
    vol24 = parse_float(safe_get(best, "volume", "h24", default=0), 0.0)
    pc1h = parse_float(safe_get(best, "priceChange", "h1", default=0), 0.0)
    score = score_pair(best)
    trap = liquidity_trap_detector(best, hist)

    stars = 0
    if score >= 220:
        stars += 1
    if liq >= 30000 and vol24 >= 20000:
        stars += 1
    if pc1h > 3:
        stars += 1
    if hist and len(hist) >= 3:
        try:
            scores = [parse_float(x.get("score_live", 0), 0.0) for x in hist[-3:]]
            if scores[-1] >= scores[0]:
                stars += 1
        except Exception:
            pass

    if trap["trap_level"] == "WARNING":
        stars -= 1
    elif trap["trap_level"] == "CRITICAL":
        stars -= 2

    stars = max(0, min(3, stars))
    return "★" * stars if stars > 0 else ""

def second_wave_label(hist: List[Dict[str, Any]]) -> str:
    if not hist or len(hist) < 5:
        return ""

    try:
        prices = [parse_float(x.get("price_usd", 0), 0.0) for x in hist]
        if len(prices) < 5:
            return ""

        first = prices[0]
        mid = prices[len(prices)//2]
        last = prices[-1]

        if mid < first * 0.8 and last > mid * 1.2:
            return "SECOND WAVE"

    except Exception:
        pass

    return ""

def rug_like(best: Optional[Dict[str, Any]], hist: List[Dict[str, Any]]) -> bool:
    if not best or not hist or len(hist) < 4:
        return False
    try:
        prices = [parse_float(x.get("price_usd", 0), 0.0) for x in hist if parse_float(x.get("price_usd", 0), 0.0) > 0]
        vols = [parse_float(x.get("vol5_usd", 0), 0.0) for x in hist]
        if len(prices) < 4:
            return False
        peak = max(prices[:-1])
        cur = prices[-1]
        drop = (peak - cur) / max(peak, 1e-9)
        vol_collapse = vols[-1] < max(vols[-3], 1.0) * 0.2
        return drop > 0.70 and vol_collapse
    except Exception:
        return False


def is_token_dead(pair: Optional[Dict[str, Any]]) -> bool:
    if not pair:
        return True

    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)
    vol5 = parse_float(safe_get(pair, "volume", "m5", default=0), 0.0)
    if liq < 1_000 and vol5 <= 0:
        return True
    return False


def compute_entry_signal(item: Dict[str, Any]) -> Tuple[str, str]:
    liq = parse_float(item.get("liquidity", safe_get(item, "liquidity", "usd", default=0)), 0.0)
    vol = parse_float(item.get("volume_5m", safe_get(item, "volume", "m5", default=0)), 0.0)
    price_change = parse_float(item.get("price_change_5m", safe_get(item, "priceChange", "m5", default=0)), 0.0)
    score = parse_float(item.get("priority_score", item.get("score_init", item.get("score", 0))), 0.0)
    momentum_flag = vol > 20_000 and price_change > 5

    if liq < 2000:
        return "NO_ENTRY", "low_liquidity"
    if score > 240 and momentum_flag:
        return "TRADEABLE", "breakout"
    if score > 200:
        return "EARLY", "early_momentum"
    if score > 150:
        return "WATCH", ""
    return "NO_ENTRY", "no_momentum"


def entry_from_score(score: float) -> str:
    if score >= 300:
        return "READY"
    if score >= 180:
        return "WATCH"
    return "WAIT"


def compute_timing(item: Dict[str, Any]) -> str:
    pc = parse_float(item.get("price_change_5m", safe_get(item, "priceChange", "m5", default=0)), 0.0)
    if pc > 10:
        return "HOT"
    if pc > 3:
        return "EARLY"
    if pc < -5:
        return "DUMP"
    return "NEUTRAL"


def classify_bucket(score: float, _item: Optional[Dict[str, Any]] = None) -> str:
    item = _item or {}
    action = str(item.get("entry_action") or item.get("entry") or "").upper()
    if action in ("ENTRY_NOW", "READY"):
        return "signals"
    if action in ("WATCH_ENTRY", "WATCH", "TRACK"):
        return "watchlist"
    if action == "EARLY":
        return "early"
    if score > 700:
        return "signals"
    if score > 400:
        return "watchlist"
    return "noise"


def is_post_rug(pair: Optional[Dict[str, Any]], hist: List[Dict[str, Any]]) -> bool:
    if not pair or not hist:
        return False
    try:
        prices = [
            parse_float(x.get("price_usd", x.get("price", 0)), 0.0)
            for x in hist
            if parse_float(x.get("price_usd", x.get("price", 0)), 0.0) > 0
        ]
        if len(prices) < 5:
            return False
        peak = max(prices)
        cur = prices[-1]
        return peak > 0 and cur < peak * 0.2
    except Exception:
        return False


def is_alive(pair: Optional[Dict[str, Any]]) -> bool:
    if not pair:
        return False
    liq = parse_float(safe_get(pair, "liquidity", "usd", default=0), 0.0)
    vol24 = parse_float(safe_get(pair, "volume", "h24", default=0), 0.0)
    buys = int(safe_get(pair, "txns", "m5", "buys", default=0) or 0)
    sells = int(safe_get(pair, "txns", "m5", "sells", default=0) or 0)
    txns5 = buys + sells
    return liq > 2_000 and (vol24 > 1_000 or txns5 > 5)


def short_addr(addr: str) -> str:
    addr = (addr or "").strip()
    if len(addr) < 10:
        return addr
    return f"{addr[:4]}...{addr[-4:]}"


def extract_name(row: Dict[str, Any]) -> str:
    name = (
        row.get("base_symbol")
        or row.get("symbol")
        or row.get("name")
        or row.get("tokenSymbol")
        or row.get("tokenName")
        or row.get("base_name")
        or (row.get("token") or {}).get("symbol")
        or (row.get("token") or {}).get("name")
        or (row.get("baseToken") or {}).get("symbol")
        or (row.get("baseToken") or {}).get("name")
    )
    if name:
        return str(name).strip()
    addr = (row.get("base_addr") or row.get("address") or row.get("mint") or "").strip()
    return f"{addr[:6]}...{addr[-4:]}" if addr else "UNKNOWN"


def hydrate_monitoring_row_defaults(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    if not row.get("entry_status"):
        row["entry_status"] = item.get("entry_status") or "UNKNOWN"
    if not row.get("risk_level"):
        row["risk_level"] = item.get("risk_level") or "MEDIUM"
    if not row.get("tp_target_pct"):
        row["tp_target_pct"] = "25"
    if not row.get("entry_suggest_usd"):
        row["entry_suggest_usd"] = "10"
    if not row.get("entry"):
        row["entry"] = item.get("entry") or "UNKNOWN"
    if not row.get("decision_reason"):
        row["decision_reason"] = item.get("decision_reason") or ""
    if not row.get("signal_reason"):
        row["signal_reason"] = item.get("signal_reason") or ""
    if not row.get("timing_label"):
        row["timing_label"] = normalize_timing_label(item.get("timing_label") or "NEUTRAL")
    if row.get("portfolio_linked") == "1" and not row.get("note"):
        row["note"] = "IN PORTFOLIO"
    return row


def monitoring_row_to_card(row: Dict[str, Any]) -> Dict[str, Any]:
    chain = (row.get("chain") or "").strip().lower()
    base_addr = addr_store(chain, str(row.get("base_addr") or "").strip())
    best = best_pair_for_token_cached(chain, base_addr) if (chain and base_addr) else None
    hist = token_history_rows(chain, base_addr, limit=30)

    liq_health = liquidity_health(best)
    anti_rug = anti_rug_early_detector(best, hist)
    size_info = position_sizing_engine(best, portfolio_value_usd=1000.0, hist=hist) if best else {
        "size_pct": 0.0,
        "size_label": "NA",
        "usd_size": 0.0,
        "risk_score": 0.0,
        "risk_flags": [],
    }

    entry_score = parse_float(row.get("entry_score"), 0.0)
    entry_action = str(row.get("entry_action") or "").upper()
    entry_status = str(row.get("entry") or row.get("entry_status") or "").upper()
    entry_reason = str(row.get("entry_reason") or row.get("signal_reason") or "")
    decision_reason = str(row.get("decision_reason") or "")
    timing_label = normalize_timing_label(row.get("timing_label") or "NEUTRAL")
    risk_level = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()

    dead = is_token_dead(best)
    post_rug = is_post_rug(best, hist)
    alive = is_alive(best)

    if dead:
        entry_status = "NO_ENTRY"
        entry_action = "AVOID"
        risk_level = "EXTREME"

    if post_rug:
        risk_level = "EXTREME"

    item = {
        "row": row,
        "best": best,
        "hist": hist,
        "raw_live_score": entry_score,
        "adjusted_live_score": entry_score,
        "live_score": entry_score,
        "decision": entry_action,
        "tags": [],
        "timing": {"timing": timing_label},
        "liq_health": liq_health,
        "anti_rug": anti_rug,
        "size_info": size_info,
        "entry_status": entry_status,
        "entry": entry_status,
        "decision_reason": decision_reason,
        "signal_reason": entry_reason,
        "entry_score": entry_score,
        "entry_reasons": [entry_reason] if entry_reason else [],
        "risk_level": risk_level,
        "timing_label": timing_label,
        "is_dead": dead,
        "is_post_rug": post_rug,
        "is_rug": post_rug,
        "is_alive": alive,
    }

    item["row"] = hydrate_monitoring_row_defaults(row, item)
    return item

def monitoring_ui_state(item: Dict[str, Any]) -> Dict[str, Any]:
    row = item.get("row", {}) or {}
    raw_score = parse_float(row.get("entry_score", item.get("entry_score", 0.0)), 0.0)
    visible_score = raw_score
    penalty = 0.0
    flags: List[str] = []

    entry_status = str(row.get("entry") or row.get("entry_status") or "").upper()
    timing_state = normalize_timing_label(row.get("timing_label") or "")
    anti_level = str(item.get("anti_rug", {}).get("level", "SAFE") or "SAFE").upper()
    liq_level = str(item.get("liq_health", {}).get("level", "UNKNOWN") or "UNKNOWN").upper()
    is_dead = bool(item.get("is_dead"))
    is_rug = bool(item.get("is_rug"))

    if entry_status in ("NO_ENTRY", "AVOID", "SKIP"):
        penalty += 25
        flags.append("no_entry")
    if entry_status == "WAIT":
        penalty += 5
        flags.append("wait")
    if timing_state == "SKIP":
        penalty += 20
        flags.append("timing_skip")
    elif timing_state == "NEUTRAL":
        penalty += 5
        flags.append("timing_neutral")
    if anti_level == "WARNING":
        penalty += 10
        flags.append("anti_rug_warning")
    elif anti_level == "CRITICAL":
        penalty += 35
        flags.append("anti_rug_critical")
    if liq_level == "WEAK":
        penalty += 10
        flags.append("liq_weak")
    elif liq_level == "CRITICAL":
        penalty += 25
        flags.append("liq_critical")
    if is_rug:
        penalty += 80
        flags.append("post_rug")
    if is_dead:
        penalty += 200
        flags.append("dead")

    visible_score = max(raw_score - penalty, 0.0)

    if entry_status == "READY":
        bucket = "tradable"
        badge = "TRADEABLE"
    elif entry_status in ("WATCH", "WAIT"):
        bucket = "caution"
        badge = "CAUTION"
    elif visible_score > 0:
        bucket = "review"
        badge = "REVIEW"
    else:
        bucket = "dead"
        badge = "DEAD"

    return {
        "bucket": bucket,
        "visible_score": round(visible_score, 2),
        "penalty": penalty,
        "flags": flags,
        "badge": badge,
    }


# =============================
# Pages
# =============================
def page_scout(cfg: Dict[str, Any]):
    st.title("DEX Scout – early candidates (DexScreener API)")
    st.caption("Фокус: дрібні/ранні монети. Majors/stables відсікаються. Core: BSC + Solana.")
    st.caption(f"DEX Scout {VERSION}")

    with st.expander("Як сортувати у Jupiter Cooking (manual triage)", expanded=False):
        st.write("1) Liquidity / TVL (щоб не застрягти у thin pool)")
        st.write("2) 24h Volume або Volume/TVL (щоб був реальний попит)")
        st.write("3) Trades / Txns (щоб був живий flow, а не один памп)")
        st.caption("У цій апці ми надійно беремо Liquidity/Volume/Txns через DexScreener. Для Solana можна додати трендовий сорс через Birdeye (за ключем).")

    if _sb_ok():
        st.caption("Storage: Supabase (persistent)")
    else:
        st.caption("Storage: Local CSV (Streamlit Cloud може скидати при деплої).")

    seeds = [x.strip() for x in (cfg["seeds_raw"] or "").split(",") if x.strip()]
    if not seeds:
        st.info("Додай seeds у sidebar (хоча б 5–10).")
        return

    sampled = sample_seeds(seeds, int(cfg["seed_k"]), bool(cfg["refresh"]))
    st.caption(f"Seeds sampled ({len(sampled)}/{len(seeds)}): " + ", ".join(sampled))

    all_pairs: List[Dict[str, Any]] = []
    query_failures = 0
    for q in sampled:
        if len(q.strip()) < 2:
            continue
        try:
            all_pairs.extend(fetch_latest_pairs_for_query(q))
            time.sleep(0.10)
        except Exception as e:
            query_failures += 1
            st.warning(f"Query failed: {q} – {e}")

    if query_failures and not all_pairs:
        st.error("All sampled queries failed. Try Refresh / Clear cache / wait a bit.")
        return


    # Optional: extra Solana source – trending mints from Birdeye ("Cooking"-like feed)
    if cfg.get("use_birdeye_trending") and (cfg.get("chain") == "solana") and BIRDEYE_ENABLED:
        mints = birdeye_trending_solana(limit=int(cfg.get("birdeye_limit", 50)))
        if mints:
            st.caption(f"Birdeye trending added: {len(mints)} mints")
            for mint in mints:
                try:
                    bp = best_pair_for_token_cached("solana", mint)
                    if bp:
                        all_pairs.append(bp)
                except Exception:
                    pass
    pairs = dedupe_mode(all_pairs, by_base_token=False)
    pairs = dedupe_mode(pairs, by_base_token=bool(cfg["dedupe_by_base"]))

    allowed = set([d.lower() for d in cfg["selected_dexes"]]) if cfg["selected_dexes"] else set()

    filtered, fstats, freasons = filter_pairs_with_debug(
        pairs=pairs,
        chain=cfg["chain"],
        any_dex=bool(cfg["any_dex"]),
        allowed_dexes=allowed,
        min_liq=float(cfg["min_liq"]),
        min_vol24=float(cfg["min_vol24"]),
        min_trades_m5=int(cfg["min_trades_m5"]),
        min_sells_m5=int(cfg["min_sells_m5"]),
        max_buy_sell_imbalance=int(cfg["max_buy_sell_imbalance"]),
        block_suspicious_names=bool(cfg["block_suspicious_names"]),
        block_majors=bool(cfg["block_majors"]),
        min_age_min=int(cfg["min_age_min"]),
        max_age_min=int(cfg["max_age_min"]),
        enforce_age=bool(cfg["enforce_age"]),
        hide_solana_unverified=bool(cfg["hide_solana_unverified"]),
    )

    mon_set, port_set = active_base_sets()

    ranked = []
    for p in filtered:
        raw_chain = (p.get("chainId") or p.get("chain") or "").lower().strip()
        chain_id = normalize_chain_name(raw_chain)
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
        if not base_addr:
            continue
        key = addr_key(chain_id, base_addr)

        if key in mon_set or key in port_set:
            continue

        row = normalize_pair_row(p)
        if row is None:
            continue
        whale = whale_accumulation(row)
        fresh = fresh_lp(row)

        s = score_pair(p)
        if whale:
            s += 3
        if fresh == "VERY_NEW":
            s += 2
        s += pump_probability(p) * 0.5
        decision, tags = build_trade_hint(row)
        timing = entry_timing_signal(row, None)
        liq_health = liquidity_health(row)
        anti_rug = anti_rug_early_detector(row, None)
        size_info = position_sizing_engine(
            row,
            portfolio_value_usd=float(cfg.get("portfolio_value_usd", 1000.0)),
            hist=None,
        )
        corr = correlation_governor(row)
        adj = risk_adjusted_size(
            float(size_info.get("usd_size", 0.0) or 0.0),
            corr,
            liq_health,
            anti_rug=anti_rug,
        )
        size_info["usd_size_adj"] = round(adj["final_size"], 2)
        # Debug mode: keep all rows, even NO_ENTRY

        ranked.append((s, decision, tags, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[: int(cfg["top_n"])]

    st.metric("Passed filters", len(ranked))

    with st.expander("Why 0 results? (Filter Debug)", expanded=(len(ranked) == 0)):
        st.write("Fetched:", len(all_pairs), " • Deduped:", len(pairs))
        st.write("Counts (pipeline):")
        st.json(dict(fstats))
        st.write("Top reject reasons:")
        top = freasons.most_common(15)
        st.write({k: v for k, v in top} if top else "No rejects counted.")
        st.caption("Note: Scout також відсікає tokens, що вже є у Monitoring/Portfolio, і відсікає 'NO ENTRY'.")

    if not ranked:
        st.info("0 results. Спробуй: lower Min Liquidity/Vol24, widen age window, або Refresh (resample seeds).")
        return

    st.session_state.setdefault("scout_hidden", set())

    def render_scout_card(pobj: Dict[str, Any], idx: int) -> None:
        base = safe_get(pobj, "baseToken", "symbol", default="???") or "???"
        quote = safe_get(pobj, "quoteToken", "symbol", default="???") or "???"
        chain_id = (pobj.get("chainId") or "").lower().strip()
        base_addr = (safe_get(pobj, "baseToken", "address", default="") or "").strip()
        pair_addr = (pobj.get("pairAddress") or "").strip()
        url = pobj.get("url", "")

        if not base_addr:
            return
        if base_addr in st.session_state["scout_hidden"]:
            return

        swap_url = build_swap_url(chain_id, base_addr)
        fresh = fresh_lp(pobj)
        dev = dev_wallet_risk(pobj)
        whale = whale_accumulation(pobj)

        score = score_pair(pobj)
        if pobj.get("smart_money"):
            score += 4
        if whale:
            score += 3
        if fresh == "VERY_NEW":
            score += 2
        pump_score = pump_probability(pobj)
        score += pump_score * 0.5
        decision, tag_list = build_trade_hint(pobj)
        entry_status, entry_score, entry_reasons, risk_level = evaluate_entry(pobj, mode=ENTRY_MODE)
        pobj["entry_status"] = entry_status
        pobj["entry_score"] = entry_score
        pobj["entry_reasons"] = entry_reasons
        pobj["risk_level"] = risk_level
        pobj["priority"] = entry_score
        timing = entry_timing_signal(pobj, None)
        liq_health = liquidity_health(pobj)
        anti_rug = anti_rug_early_detector(pobj, None)
        size_info = position_sizing_engine(
            pobj,
            portfolio_value_usd=float(cfg.get("portfolio_value_usd", 1000.0)),
            hist=None,
        )
        corr = correlation_governor(pobj)
        adj = risk_adjusted_size(
            float(size_info.get("usd_size", 0.0) or 0.0),
            corr,
            liq_health,
            anti_rug=anti_rug,
        )
        size_info["usd_size_adj"] = round(adj["final_size"], 2)
        entry_state = classify_entry_status(pobj, decision, timing, liq_health, anti_rug, size_info)
        sniper_flag = detect_snipers(pobj)
        trap_signal, trap_level = liquidity_trap(pobj)

        st.markdown("---")
        st.subheader(f"{base} / {quote}")
        st.caption(f"{chain_id or '?'} • {pobj.get('dexId','?')}")

        btn1, btn2 = st.columns(2)
        with btn1:
            link_button("Open DexScreener", url or "", use_container_width=True, key=f"s_ds_{idx}_{hkey(pair_addr)}")
        with btn2:
            if swap_url:
                swap_label = "Open Swap (Jupiter)" if chain_id == "solana" else "Open Swap"
                link_button(swap_label, swap_url, use_container_width=True, key=f"s_sw_{idx}_{hkey(base_addr, chain_id)}")
            else:
                st.button("Open Swap", disabled=True, use_container_width=True)

        btn3, btn4 = st.columns(2)
        with btn3:
            if st.button("Add to Monitoring", key=f"add_mon_{pair_addr}", use_container_width=True):
                add_to_monitoring(
                    pobj,
                    float(score),
                    entry_status=entry_status,
                    entry_score=float(entry_score),
                    risk_level=risk_level,
                )
                st.session_state["scout_hidden"].add(base_addr)
                st.toast("Added to monitoring")
                request_rerun()
        with btn4:
            if st.button("Log → Portfolio (I swapped)", key=f"log_pf_{pair_addr}", use_container_width=True):
                res = log_to_portfolio(pobj, float(score), decision, tag_list, swap_url)
                if res == "OK":
                    st.session_state["scout_hidden"].add(base_addr)
                    st.toast("Logged to portfolio")
                    request_rerun()
                else:
                    st.error(f"Portfolio log failed: {res}")

        price = parse_float(pobj.get("priceUsd"), 0.0)
        liq = parse_float(safe_get(pobj, "liquidity", "usd", default=0), 0.0)
        vol24 = parse_float(safe_get(pobj, "volume", "h24", default=0), 0.0)
        volm5 = parse_float(safe_get(pobj, "volume", "m5", default=0), 0.0)
        chg_m5 = parse_float(safe_get(pobj, "priceChange", "m5", default=0), 0.0)
        chg_h1 = parse_float(safe_get(pobj, "priceChange", "h1", default=0), 0.0)
        buys = (pobj.get("txns") or {}).get("m5", {}).get("buys")
        sells = (pobj.get("txns") or {}).get("m5", {}).get("sells")

        st.write(f"Score: {score:,.2f}")
        c_entry_1, c_entry_2 = st.columns(2)
        with c_entry_1:
            if entry_status in ("READY", "TRADEABLE"):
                st.success(f"Entry Status: {entry_status}")
            elif entry_status in ("WAIT", "EARLY"):
                st.warning(f"Entry Status: {entry_status}")
            else:
                st.error("Entry Status: NO_ENTRY")
        with c_entry_2:
            st.metric("Entry Score", f"{entry_score:.2f}")
        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(str(risk_level).upper(), "⚪")
        st.caption(f"Risk Level: {risk_color} {str(risk_level).upper()}")
        st.caption(f"score={score:.2f} vol5={volm5:.2f} pc1h={chg_h1:.2f} risk={risk_level}")
        if entry_reasons:
            st.caption("Entry reasons: " + " • ".join(entry_reasons))
        st.write(f"Meme Score: {int(pobj.get('meme_score', 0) or 0)}")
        smart_money_label = "YES" if int(pobj.get("smart_money", 0) or 0) == 1 else "NO"
        st.write(f"Smart Money: {smart_money_label}")
        if int(pobj.get("migration", 0) or 0) == 1:
            st.write(f"Migration: {pobj.get('migration_label', '')}")
        st.write(f"Sniper: {'YES' if sniper_flag else 'NO'}")
        st.write(f"Pump Score: {pump_score}")
        st.write(f"Trap: {trap_label(trap_signal)}")
        if detect_auto_signal(pobj):
            st.warning("AUTO SIGNAL DETECTED")
        if sniper_flag:
            st.markdown("🧠 sniper activity")
        if pump_score >= 6:
            st.markdown("🚀 pump probability HIGH")
        if trap_signal:
            st.markdown(f"⚠️ {trap_label(trap_signal)}")
        if fresh == "VERY_NEW":
            st.markdown("🆕 fresh LP")
        elif fresh == "NEW":
            st.markdown("🌱 new token")
        if whale:
            st.markdown("🐋 whale accumulation")
        if dev:
            st.markdown("⚠ dev risk")
        st.write(f"Price: ${price:,.8f}" if price else "Price: n/a")
        st.write(f"Liq: {fmt_usd(liq)}")
        st.write(f"Vol24: {fmt_usd(vol24)}")
        st.write(f"Vol m5: {fmt_usd(volm5)}")
        st.write(f"Δ m5: {chg_m5:+.2f}%")
        st.write(f"Δ h1: {chg_h1:+.2f}%")
        if buys is not None and sells is not None:
            st.write(f"Buys/Sells (m5): {buys}/{sells}")

        st.caption("Action")
        st.markdown(action_badge(decision), unsafe_allow_html=True)

        if tag_list:
            st.caption("Tags")
            for t in tag_list:
                st.write(f"• {t}")

        with st.expander("Addresses", expanded=False):
            st.caption("Token contract (baseToken.address)")
            st.code(str(base_addr or ""), language="text")
            st.caption("Pair / pool address")
            st.code(str(pair_addr or ""), language="text")

        if chain_id == "solana":
            st.caption("Solana: check Jupiter/JupShield warnings before swapping.")

    for i, (_s, _decision, _tags, p) in enumerate(ranked, start=1):
        render_scout_card(p, i)


def page_monitoring(auto_cfg: Dict[str, Any]):
    st.title("Monitoring")
    st.caption("Manual pipeline: click scanner → save snapshot → render Monitoring.")
    st.caption("Telegram callbacks update storage directly. Use Refresh live data or reload the page to see external changes.")

    if "selected_token" not in st.session_state:
        st.session_state.selected_token = None

    if bool(auto_cfg.get("auto_revisit_enabled", False)):
        try:
            n = auto_reactivate_archived(days=int(auto_cfg.get("auto_revisit_days", 7)))
            if n:
                st.caption(f"Auto-revisited from Archive: {n}")
        except Exception:
            pass
    try:
        revisited_now = revisit_archived(min_days=1, score_threshold=260.0, max_revisits=3)
        if revisited_now:
            st.caption(f"Revisited high-score candidates: {revisited_now}")
    except Exception:
        pass
    scan_state = scanner_state_load()
    top = st.columns([2,2,3,3])

    # Source of truth for Monitoring page UI: monitoring.csv only.
    monitoring_rows = load_monitoring_rows_cached()
    clean_rows: List[Dict[str, Any]] = []
    for r in monitoring_rows:
        chain = str(r.get("chain") or "").lower().strip()
        symbol = str(r.get("base_symbol") or "").upper().strip()
        if chain not in ALLOWED_CHAINS:
            continue
        if symbol in HARD_BLOCK_SYMBOLS:
            continue
        clean_rows.append(r)
    monitoring_rows = clean_rows
    mon_source = st.session_state.get("_storage_source_monitoring.csv", "unknown")
    st.caption(f"MONITORING FROM DB: {len(monitoring_rows)} (source: {mon_source})")
    rows: List[Dict[str, Any]] = []
    for raw in monitoring_rows:
        row = dict(raw)
        for k in MON_FIELDS:
            if k not in row:
                row[k] = ""
        if "symbol" not in row:
            row["symbol"] = row.get("base_symbol", "NA")
        if "address" not in row:
            row["address"] = row.get("base_addr", "")
        if "price" not in row:
            row["price"] = ""
        if "status" not in row:
            row["status"] = row.get("entry_status", "watch")
        rows.append(row)

    active_rows = []
    _mon_set, active_set = active_base_sets()
    for r in rows:
        chain = (r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, (r.get("base_addr") or "").strip())
        r["in_portfolio"] = "1" if addr_key(chain, base_addr) in active_set else "0"
        if r["in_portfolio"] == "1":
            r["status"] = "PORTFOLIO"
        status_raw = r.get("status", "")
        status = str(status_raw).strip().upper() if status_raw is not None else ""
        if status in ("ACTIVE", "WATCH", "WAIT", "PORTFOLIO", ""):
            active_rows.append(r)
    if not active_rows:
        active_rows = rows[:50]
    active_rows = sorted(active_rows, key=lambda row: parse_float(row.get("priority_score", 0), 0.0), reverse=True)[:50]
    active = list(active_rows)
    archived = [r for r in rows if str(r.get("active", "1")).strip() != "1"]
    top[0].metric("Active", len(active))
    top[1].metric("Archived", len(archived))
    top[2].caption(f"Last scan: {scan_state.get('last_run_ts','—')} • {scan_state.get('last_window','—')} • {scan_state.get('last_chain','—')}")
    top[3].caption(f"Last stats: {scan_state.get('last_stats', {})}")
    stats = scan_state.get("last_stats", {}) or {}
    st.caption(
        f"Added: {stats.get('added', 0)} • "
        f"Skipped active: {stats.get('skipped_active', 0)} • "
        f"Skipped archived: {stats.get('skipped_archived', 0)} • "
        f"Skipped noise: {stats.get('skipped_noise', 0)} • "
        f"Skipped wrong chain: {stats.get('skipped_wrong_chain', 0)} • "
        f"Skipped major: {stats.get('skipped_major', 0)} • "
        f"Revisited: {stats.get('revisited', 0)} • "
        f"Trimmed: {stats.get('trimmed', 0)} • "
        f"Stale removed: {stats.get('stale_removed', 0)}"
    )

    cbtn1, cbtn2, cbtn3 = st.columns([2,2,6])
    with cbtn1:
        if st.button("Run full scout ingestion", use_container_width=True, key="run_scanner_now"):
            stats = run_full_ingestion_now(
                chain=str(st.session_state.get("chain", "solana")),
                seeds_raw=str(auto_cfg.get("scanner_seeds_raw", "")),
                max_items=int(auto_cfg.get("scanner_max_items", 50)),
                use_birdeye_trending=bool(auto_cfg.get("use_birdeye_trending", True)),
                birdeye_limit=int(auto_cfg.get("birdeye_limit", 50)),
            )
            st.success(f"Scanner ran: {stats}")
            load_monitoring_rows_cached.clear()
            request_rerun()
    with cbtn2:
        if st.button("Refresh live data", use_container_width=True):
            st.cache_data.clear()
            load_monitoring_rows_cached.clear()
            _ = load_monitoring()
            _ = load_portfolio()
            request_rerun()
    with cbtn3:
        if st.button("🔥 WIPE MONITORING (destructive)", use_container_width=True):
            save_csv(MONITORING_CSV, [], MON_FIELDS)
            reset_monitoring()
            st.rerun()
        st.caption("Clears monitoring.csv. Does not restore history automatically.")

    if not active_rows:
        st.info("Monitoring is empty. Run scanner now to fetch and save tokens.")
        return

    chain_filter = st.selectbox("Chain filter", ["all", "bsc", "solana"], index=0)
    if chain_filter != "all":
        active = [r for r in active_rows if (r.get("chain") or "").strip().lower() == chain_filter]
    else:
        active = list(active_rows)

    # --- DEDUP BY SYMBOL WITH BEST SCORE ---
    best_by_symbol: Dict[str, Dict[str, Any]] = {}
    for t in active:
        symbol = str(t.get("symbol") or t.get("base_symbol") or "").upper().strip()
        score = parse_float(t.get("score", t.get("priority_score", 0)), 0.0)

        if symbol not in best_by_symbol:
            best_by_symbol[symbol] = t
        else:
            prev_score = parse_float(best_by_symbol[symbol].get("score", best_by_symbol[symbol].get("priority_score", 0)), 0.0)
            if score > prev_score:
                best_by_symbol[symbol] = t

    active = list(best_by_symbol.values())

    cards_raw = [monitoring_row_to_card(r) for r in active]
    cards_all = []
    portfolio_rows = load_portfolio()
    for c in cards_raw:
        ui = monitoring_ui_state(c)
        c["ui_bucket"] = ui["bucket"]
        c["ui_visible_score"] = float(ui["visible_score"])
        c["ui_penalty"] = float(ui["penalty"])
        c["ui_flags"] = list(ui["flags"])
        c["ui_badge"] = ui["badge"]
        cards_all.append(c)

    priority_cards: List[Dict[str, Any]] = []
    portfolio_linked_cards: List[Dict[str, Any]] = []
    review_cards: List[Dict[str, Any]] = []
    dead_cards = [c for c in cards_all if c["ui_bucket"] == "dead"]

    for c in cards_all:
        if c.get("ui_bucket") == "dead":
            continue
        row = c.get("row", {})
        in_pf = is_in_portfolio_active(row, portfolio_rows)
        c["is_portfolio_active"] = in_pf
        weak_score = float(c.get("ui_visible_score", 0.0)) < 90
        is_review = str(c.get("ui_badge", "")).upper() == "REVIEW" or c.get("ui_bucket") == "review" or weak_score
        if in_pf:
            portfolio_linked_cards.append(c)
        elif is_review:
            review_cards.append(c)
        else:
            priority_cards.append(c)

    def render_dedupe(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_symbol: Dict[str, Dict[str, Any]] = {}
        for c in cards:
            r = c.get("row", {})
            sym = str(r.get("base_symbol") or extract_name(r) or "").strip().upper()
            if not sym:
                continue
            prev = best_by_symbol.get(sym)
            if prev is None or float(c.get("ui_visible_score", 0)) > float(prev.get("ui_visible_score", 0)):
                best_by_symbol[sym] = c
        return list(best_by_symbol.values())

    priority_cards = render_dedupe(priority_cards)
    portfolio_linked_cards = render_dedupe(portfolio_linked_cards)
    review_cards = render_dedupe(review_cards)
    priority_cards.sort(key=lambda x: x["ui_visible_score"], reverse=True)
    portfolio_linked_cards.sort(key=lambda x: x["ui_visible_score"], reverse=True)
    review_cards.sort(key=lambda x: x["ui_visible_score"], reverse=True)

    def render_monitoring_card(item: Dict[str, Any]) -> None:
        r = item["row"]
        best = item["best"]
        chain = token_chain(r)
        base_addr = token_ca(r)
        sym = extract_name(r).upper()
        name = sym if sym not in {"", "UNKNOWN"} else "UNKNOWN"
        score = round(float(item.get("ui_visible_score", 0.0)), 2)
        unified = compute_unified_recommendation(r, source="monitoring")
        primary = str(unified.get("final_action") or "WAIT")
        secondary = item.get("ui_badge", "INFO")
        header = f"{name} | {primary} | {score:.2f}"

        with st.expander(header, expanded=False):
            timing_label = unified.get("timing") or normalize_timing_label(item.get("timing_label") or r.get("timing_label") or "NEUTRAL")
            risk_label = str(item.get("risk_level", "UNKNOWN"))
            st.caption(
                f"{chain.upper()} • timing {timing_label} • risk {risk_label} • status {secondary}"
            )
            st.markdown(f"**Recommendation: {primary}**")
            st.caption(f"Reason: {unified.get('final_reason', 'n/a')}")
            st.caption(f"Suggested size: {unified.get('size_hint', 'WATCH ONLY')}")
            st.caption(f"score={score} risk={item.get('risk_level', 'UNKNOWN')}")
            if item.get("is_portfolio_active"):
                st.caption("In portfolio • still monitored")
            if r.get("toxic_flags"):
                st.caption(f"Toxic: {r.get('toxic_flags')}")
            flags = item.get("ui_flags") or []
            st.caption("UI flags: " + (" • ".join(flags) if flags else "none"))
            st.caption(
                f"Raw score: {float(item.get('raw_live_score', 0.0)):.2f} • "
                f"Penalty: -{float(item.get('ui_penalty', 0.0)):.2f} • "
                f"Visible: {float(item.get('ui_visible_score', 0.0)):.2f}"
            )
            reason = str(r.get("entry_reason") or r.get("signal_reason") or "")
            if reason:
                st.caption(f"Reason: {reason}")

            blocker = str(r.get("decision_reason") or "")
            if blocker:
                st.caption(f"Blockers: {blocker}")

            weak = str(r.get("weak_reason") or "")
            if weak:
                st.caption(f"Weak: {weak}")

            c1, c2, c3 = st.columns(3)
            with c1:
                dex_url = dex_url_for_token(token_chain(r), token_ca(r))
                if dex_url:
                    link_button("Dex", dex_url, use_container_width=True)
                elif best and best.get("url"):
                    link_button("Dex", best.get("url"), use_container_width=True)
            with c2:
                swap_url = build_swap_url(chain, base_addr)
                if swap_url:
                    link_button("Swap", swap_url, use_container_width=True)
            with c3:
                if not item.get("is_portfolio_active"):
                    if st.button("Promote → Portfolio", key=f"promote_{chain}_{base_addr}", use_container_width=True):
                        res = promote_to_portfolio(item)
                        if res == "OK":
                            st.success("Promoted.")
                            request_rerun()
                        elif res == "NO_LIVE_PAIR":
                            st.error("No live pool found.")
                        else:
                            request_rerun()

            st.code(token_ca(r), language="text")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Entry", item["entry"])
            with m2:
                st.metric("Timing", normalize_timing_label(item.get("timing_label", "NA")))
            with m3:
                st.metric("Risk", item["risk_level"])
            with m4:
                st.metric("Safe", item["anti_rug"].get("level", "UNKNOWN"))

            p1, p2, p3 = st.columns(3)
            with p1:
                st.write(f"Size: {item['size_info'].get('size_label', 'NA')}")
            with p2:
                st.write(f"USD: ${float(item['size_info'].get('usd_size', 0) or 0):.2f}")
            with p3:
                st.write(f"TP: {r.get('tp_target_pct') or '25'}%")

            entry_now = parse_float(r.get("entry_now"), 0.0)
            pullback_1 = parse_float(r.get("pullback_1"), 0.0)
            pullback_2 = parse_float(r.get("pullback_2"), 0.0)
            invalidation = parse_float(r.get("invalidation"), 0.0)
            tp1 = parse_float(r.get("tp1"), 0.0)
            tp2 = parse_float(r.get("tp2"), 0.0)
            horizon = str(r.get("entry_horizon") or "n/a")

            if has_actionable_levels(r):
                st.caption(
                    f"Horizon: {horizon} • "
                    f"Entry now: {entry_now} • "
                    f"PB1: {pullback_1} • "
                    f"PB2: {pullback_2} • "
                    f"Invalidation: {invalidation} • "
                    f"TP1: {tp1} • TP2: {tp2}"
                )
            else:
                st.caption(f"Horizon: {horizon} • setup: watch only")

            if item["entry_reasons"]:
                st.info(" • ".join(item["entry_reasons"][:3]))

            hist = item.get("hist", []) or []
            render_monitoring_sparklines(hist)

    st.subheader("Priority watchlist")
    if not priority_cards:
        st.info("No items in priority watchlist yet.")
    for item in priority_cards:
        render_monitoring_card(item)

    st.subheader("Portfolio-linked monitoring")
    if not portfolio_linked_cards:
        st.caption("No portfolio-linked monitoring tokens.")
    for item in portfolio_linked_cards:
        render_monitoring_card(item)

    st.subheader("Needs review")
    if not review_cards:
        st.caption("No review candidates.")
    for item in review_cards:
        render_monitoring_card(item)

    with st.expander("Auto-archived / Dead candidates", expanded=False):
        if not dead_cards:
            st.caption("No dead candidates.")
        for item in dead_cards:
            r = item["row"]
            name = extract_name(r).upper()
            chain = (r.get("chain") or "").strip().lower()
            flags = item.get("ui_flags") or []
            st.write(f"{name} ({chain}) • {item.get('decision', 'DEAD')}")
            st.caption("Flags: " + (" • ".join(flags) if flags else "none"))

def page_archive():
    st.title("Archive")
    st.caption("Архівовані токени з Monitoring. Можуть повернутися тільки вручну (re-activate).")

    rows = load_monitoring()
    archived = [r for r in rows if r.get("active") != "1"]

    if not archived:
        st.info("Archive is empty.")
        return

    archived.sort(key=lambda x: x.get("ts_archived", "") or x.get("ts_added", ""), reverse=True)

    for idx, r in enumerate(archived[:200], start=1):
        chain = (r.get("chain") or "").strip().lower()
        base_sym = r.get("base_symbol") or "???"
        base_addr = (r.get("base_addr") or "").strip()
        ts_a = r.get("ts_archived") or ""
        reason = r.get("archived_reason") or ""
        last_score = r.get("last_score") or ""
        last_dec = r.get("last_decision") or ""

        st.markdown("---")
        cols = st.columns([3, 2, 2, 2])
        with cols[0]:
            st.subheader(base_sym)
            st.caption(f"Chain: {chain}")
            st.code(str(base_addr or ""), language="text")
        with cols[1]:
            st.write(f"Archived: {ts_a or 'n/a'}")
            st.write(f"Reason: {reason or 'n/a'}")
        with cols[2]:
            st.write(f"Last score: {last_score or 'n/a'}")
            st.write(f"Last decision: {last_dec or 'n/a'}")
        with cols[3]:
            swap_url = build_swap_url(chain, base_addr)
            if swap_url:
                swap_label = "Swap (PancakeSwap)" if chain == "bsc" else "Swap (Jupiter)"
                link_button(swap_label, swap_url, use_container_width=True, key=f"a_sw_{idx}_{hkey(base_addr, chain)}")

            if st.button("Re-activate → Monitoring", key=f"rea_{idx}_{hkey(base_addr)}", use_container_width=True):
                ok = reactivate_monitoring(chain, base_addr)
                if ok:
                    st.success("Re-activated.")
                    request_rerun()
                else:
                    st.info("Nothing changed (not found).")


def portfolio_alert_count() -> int:
    rows = load_portfolio()
    active_rows = [r for r in rows if r.get("active") == "1"]

    alerts = 0
    for r in active_rows:
        chain = (r.get("chain") or "").lower()
        base_addr = r.get("base_token_address")

        best = best_pair_for_token_cached(chain, base_addr)
        if not best:
            continue

        entry = get_portfolio_entry_price(r)
        hist = token_history_rows(chain, base_addr, limit=60)

        liq_health = liquidity_health(best)
        if liq_health["level"] in ("DEAD", "CRITICAL"):
            alerts += 1
            continue

        exit_signal = exit_before_dump_detector(best, hist, entry)
        exit_signal = apply_liquidity_exit_override(exit_signal, liq_health)
        persistence = exit_persistence_state(hist, min_points=3)
        if liq_health["action"] in ("EXIT_NOW", "EXIT"):
            persistence = {**persistence, "persistent_exit": True}

        if exit_signal.get("exit_level") == "EXIT" and persistence.get("persistent_exit"):
            alerts += 1

    return alerts


def core_safety_self_check():
    issues: List[str] = []

    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        st.error("CORE SAFETY BROKEN")
        st.write("– cannot read source")
        return

    required = {
        "anti_rug_early_detector": "anti-rug missing",
        "liquidity_health": "liquidity health missing",
        "apply_liquidity_exit_override": "liquidity exit override missing",
        "apply_liquidity_reco_override": "liquidity reco override missing",
        "risk_adjusted_size": "risk-adjusted sizing missing",
        "flow_collapse_detector": "flow collapse detector missing",
    }

    for needle, msg in required.items():
        if needle not in src:
            issues.append(msg)

    if ".lower()" in inspect.getsource(active_portfolio_addresses):
        issues.append("address lower() regression in active_portfolio_addresses")

    if issues:
        st.error("CORE SAFETY BROKEN")
        for i in issues[:8]:
            st.write(f"– {i}")
    else:
        st.success("Core safety OK")


def render_debug_panel():
    with st.sidebar.expander("🛠 Debug panel", expanded=False):
        st.write("### Supabase")
        st.write(f"USE_SUPABASE: {USE_SUPABASE}")
        st.write(f"URL present: {bool(SUPABASE_URL)}")
        st.write(f"KEY present: {bool(SUPABASE_SERVICE_ROLE_KEY)}")

        for key in ["portfolio.csv", "monitoring.csv", "monitoring_history.csv"]:
            try:
                content = sb_get_storage(key) if _sb_ok() else None
                if content:
                    st.success(f"{key}: FOUND ({len(content)} chars)")
                else:
                    st.warning(f"{key}: EMPTY / NOT FOUND")
            except Exception as e:
                st.error(f"{key}: ERROR {e}")

        st.write("---")
        st.write("### Local fallback")
        for key in ["portfolio.csv", "monitoring.csv", "monitoring_history.csv"]:
            fpath = os.path.join(DATA_DIR, key)
            if os.path.exists(fpath):
                st.success(f"{key}: local OK")
            else:
                st.warning(f"{key}: no local file")

        st.write("---")
        st.write("### Runtime")
        st.write(f"Session keys: {len(st.session_state.keys())}")
        st.write(list(st.session_state.keys())[:10])

        st.write("---")
        st.write("### Core safety")
        core_safety_self_check()

        st.write("---")
        st.write("### Events")
        for line in st.session_state.get("debug_log", []):
            st.caption(line)


def explain_reco(
    reco: str,
    liq_health: Dict[str, Any],
    anti_rug: Dict[str, Any],
    exit_signal: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    if liq_health.get("level") == "OK":
        reasons.append("liquidity stable")
    else:
        reasons.append(f"liquidity={liq_health.get('level')}")

    if anti_rug.get("level") == "SAFE":
        reasons.append("no rug pattern")
    else:
        reasons.append(f"anti_rug={anti_rug.get('level')}")

    reasons.extend([str(x) for x in exit_signal.get("exit_flags", [])[:3]])
    return reasons[:4]


def page_portfolio():
    st.title("Portfolio / Watchlist")
    st.caption("Entries appear here after clicking Log → Portfolio (I swapped) in Scout or Promote in Monitoring.")

    rows = load_portfolio()
    active_rows = [r for r in rows if r.get("active") == "1"]
    closed_rows = [r for r in rows if r.get("active") != "1"]

    topbar = st.columns([2, 2, 3])
    topbar[0].metric("Active", len(active_rows))
    topbar[1].metric("Closed", len(closed_rows))
    with topbar[2]:
        if not _sb_ok():
            with open(PORTFOLIO_CSV, "rb") as f:
                st.download_button("Download portfolio.csv", f, file_name="portfolio.csv", use_container_width=True)
        else:
            st.caption("Supabase storage – export via Supabase dashboard.")

    st.markdown("---")

    if not active_rows:
        st.info("Portfolio is empty. Use Scout → Log → Portfolio or Monitoring → Promote → Portfolio.")
        return

    rotation_rows = [
        r for r in active_rows
        if liquidity_health(
            best_pair_for_token_cached(
                (r.get("chain") or "").strip().lower(),
                addr_store((r.get("chain") or "").strip().lower(), (r.get("base_token_address") or "").strip()),
            )
        )["level"] not in ("DEAD",)
    ]

    rot = capital_rotation_engine(rotation_rows)
    actions = rotation_actions(rot)
    rotation_plan = rotation_exit_plan(rot)
    weak_rotation_symbols = {str(p.get("symbol") or "") for p in rotation_plan}
    best_strong = rot["strong"][0] if rot.get("strong") else None

    if actions:
        st.markdown("## 🔁 Capital Rotation Signals")
        for a in actions:
            st.warning(f"Rotate: {a['from']} → {a['to']}")
            st.caption(a["reason"])

    st.subheader("Active positions")
    for idx, r in enumerate(active_rows):
        chain = (r.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, (r.get("base_token_address") or "").strip())
        base_sym = r.get("base_symbol") or "???"
        quote_sym = r.get("quote_symbol") or "???"
        entry_price = get_portfolio_entry_price(r)
        entry_price_str = f"{entry_price:.12f}".rstrip("0").rstrip(".") if entry_price > 0 else "0"

        best = best_pair_for_token_cached(chain, base_addr) if (chain and base_addr) else None
        liq_health = liquidity_health(best)

        cur_price = parse_float(best.get("priceUsd"), 0.0) if best else 0.0
        pc1h = parse_float(safe_get(best, "priceChange", "h1", default=0), 0.0) if best else 0.0
        pc5 = parse_float(safe_get(best, "priceChange", "m5", default=0), 0.0) if best else 0.0
        liq = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0) if best else 0.0
        vol24 = parse_float(safe_get(best, "volume", "h24", default=0), 0.0) if best else 0.0
        vol5 = parse_float(safe_get(best, "volume", "m5", default=0), 0.0) if best else 0.0
        s_live = score_pair(best) if best else 0.0
        if whale_accumulation(best):
            s_live += 3
        if fresh_lp(best) == "VERY_NEW":
            s_live += 2
        s_live += pump_probability(best) * 0.5 if best else 0.0
        decision, tags = build_trade_hint(best) if best else ("NO DATA", [])

        # ensure history also captures portfolio tokens (same history table)
        if best:
            snapshot_live_to_history(chain, base_sym, base_addr, best)

        hist = token_history_rows(chain, base_addr, limit=60)
        anti_rug = anti_rug_early_detector(best, hist)
        flow_collapse = flow_collapse_detector(hist)
        if flow_collapse["level"] == "WARNING" and anti_rug["level"] == "SAFE":
            anti_rug = {
                "level": "WARNING",
                "score": max(int(anti_rug.get("score", 0)), 2),
                "flags": list(set((anti_rug.get("flags", []) or []) + flow_collapse["flags"])),
                "action": "REDUCE",
            }
        strength_score = position_strength_score(best or {}, hist, entry_price)
        strength_cls = classify_position_strength(strength_score)
        trap = liquidity_trap_detector(best, hist) if best else {"trap_score": 0, "trap_level": "SAFE", "trap_flags": []}
        exit_signal = exit_before_dump_detector(best, hist, entry_price)
        exit_signal = apply_liquidity_exit_override(exit_signal, liq_health)

        if anti_rug["level"] == "CRITICAL":
            flags = list(exit_signal.get("exit_flags", []) or [])
            for f in anti_rug.get("flags", []):
                if f not in flags:
                    flags.append(f)
            exit_signal.update({
                "exit_level": "EXIT",
                "exit_score": max(int(exit_signal.get("exit_score", 0)), 4),
                "exit_flags": flags,
                "suggested_action": "EXIT_100",
            })
        elif anti_rug["level"] == "WARNING" and str(exit_signal.get("exit_level", "WATCH")).upper() == "WATCH":
            flags = list(exit_signal.get("exit_flags", []) or [])
            for f in anti_rug.get("flags", []):
                if f not in flags:
                    flags.append(f)
            exit_signal.update({
                "exit_level": "EARLY",
                "exit_score": max(int(exit_signal.get("exit_score", 0)), 2),
                "exit_flags": flags,
                "suggested_action": "REDUCE_50",
            })
        persistence = exit_persistence_state(hist, min_points=3)
        if liq_health["action"] in ("EXIT_NOW", "EXIT"):
            persistence = {**persistence, "persistent_exit": True}
        action_plan = action_from_exit_signal(exit_signal, persistence)
        level = str(exit_signal.get("exit_level", "WATCH"))

        if level.startswith("EXIT"):
            exit_pct = 1.0
        elif level.startswith("EARLY"):
            exit_pct = 0.7
        else:
            exit_pct = 0.0

        if strength_cls == "WEAK" and exit_pct < 1.0:
            exit_pct = max(exit_pct, 0.5)

        reco = portfolio_reco(entry_price, cur_price, liq, vol24, pc1h, pc5, decision, s_live, best)
        if trap["trap_level"] == "CRITICAL":
            reco = "EXIT / RISK"
        elif trap["trap_level"] == "WARNING" and str(reco).startswith("HOLD"):
            reco = "HOLD / WATCH CAREFULLY"

        if action_plan["action_code"] == "EXIT_100":
            reco = "EXIT NOW"
        elif action_plan["action_code"] == "EXIT_CONFIRM":
            reco = "EXIT LIKELY"
        elif action_plan["action_code"] == "REDUCE_50":
            reco = "REDUCE 50%"
        elif action_plan["action_code"] == "REDUCE_30":
            reco = "REDUCE 30%"
        elif action_plan["action_code"] == "WATCH_TIGHT" and str(reco).startswith("HOLD"):
            reco = "WATCH CLOSELY"

        if anti_rug["level"] == "CRITICAL":
            reco = "EXIT NOW / EARLY RUG RISK"
        elif anti_rug["level"] == "WARNING" and "HOLD" in str(reco).upper():
            reco = "REDUCE / EARLY RUG RISK"
        reco = apply_liquidity_reco_override(reco, liq_health)

        pnl = 0.0
        if entry_price > 0 and cur_price > 0:
            pnl = (cur_price - entry_price) / entry_price * 100.0

        unified = compute_unified_recommendation(
            {
                **r,
                "pnl_pct": pnl,
                "reco_model": reco,
                "exit_signal": level,
                "trap_score": trap.get("trap_score", 0),
                "anti_rug": anti_rug.get("level", "SAFE"),
                "liquidity_health": liq_health.get("level", "OK"),
                "score": s_live,
            },
            source="portfolio",
        )

        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])

        with c1:
            symbol = f"{base_sym}/{quote_sym}"
            if liq_health["level"] in ("DEAD", "CRITICAL"):
                st.markdown("### 🔴 " + symbol)
            elif level == "EXIT":
                st.markdown("### 🔴 " + symbol)
            elif level == "EARLY" or liq_health["level"] == "WEAK":
                st.markdown("### 🟡 " + symbol)
            else:
                st.markdown("### " + symbol)
            st.caption(f"Chain: {chain} • DEX: {best.get('dexId','') if best else r.get('dex','')}")
            st.code(token_ca(r), language="text")
            dex_url = dex_url_for_token(token_chain(r), token_ca(r))
            if dex_url:
                link_button("DexScreener", dex_url, use_container_width=True, key=f"p_ds_{idx}_{hkey(base_addr)}")
            elif best and best.get("url"):
                link_button("DexScreener", best.get("url", ""), use_container_width=True, key=f"p_ds_{idx}_{hkey(base_addr)}")
            elif r.get("dexscreener_url"):
                link_button("DexScreener", r["dexscreener_url"], use_container_width=True, key=f"p_ds_{idx}_{hkey(base_addr)}")
            swap_url = r.get("swap_url") or build_swap_url(chain, base_addr)
            if swap_url:
                link_button("Swap", swap_url, use_container_width=True, key=f"p_sw_{idx}_{hkey(base_addr, chain)}")

            st.caption(f"Model decision: {decision}")

        with c2:
            st.markdown(f"### {portfolio_action_badge(unified['final_action'])}")
            st.markdown(f"**Recommended action: {unified['final_action']}**")
            st.caption(f"Reason: {unified['final_reason']}")
            with st.expander("Market / position metrics", expanded=False):
                st.write(f"Score: {s_live:.2f}" if best else f"Score: {r.get('score','n/a')}")
                st.write(f"Entry: ${entry_price_str}")
                st.write(f"Now: ${cur_price:.8f}" if cur_price else "Now: n/a")
                st.write(f"PnL: {pnl:+.2f}%" if entry_price and cur_price else "PnL: n/a")
                st.write(f"Liq: {fmt_usd(liq)}" if best else "Liq: n/a")
                st.write(f"Vol24: {fmt_usd(vol24)}" if best else "Vol24: n/a")
                st.write(f"Vol5: {fmt_usd(vol5)}" if best else "Vol5: n/a")
                st.write(f"Δ m5: {fmt_pct(pc5)}")
                st.write(f"Δ h1: {fmt_pct(pc1h)}")
                st.write(f"score: {s_live:.2f}")

        with c3:
            if strength_cls == "WEAK":
                st.caption("⚠ weak position – candidate for rotation")
                if base_sym in weak_rotation_symbols:
                    st.error(f"Sell {int(exit_pct * 100)}% – rotation/exit signal")
                    if best_strong:
                        rotate_to = best_strong.get("base_symbol") or best_strong.get("symbol") or "?"
                        st.caption(f"→ rotate into {rotate_to}")
            with st.expander("Risk diagnostics", expanded=False):
                st.write(f"Reco model: {reco}")
                st.write(f"Exit signal: {level}")
                st.write(f"Liquidity health: {liq_health['level']}")
                st.write(f"Anti-rug: {anti_rug['level']}")
                st.write(f"Trap: {trap.get('trap_level', 'SAFE')} ({trap.get('trap_score', 0)})")
                st.write(f"Strength: {strength_cls} ({strength_score:.1f})")
                notes = explain_reco(reco, liq_health, anti_rug, exit_signal)
                st.write("Diagnostics notes:")
                for rr in notes:
                    st.caption(f"– {rr}")
                for f in anti_rug.get("flags", [])[:4]:
                    st.caption(f"– {f}")
                if anti_rug["level"] == "CRITICAL":
                    st.error("early rug risk – exit")
                elif anti_rug["level"] == "WARNING":
                    st.warning("rug risk building – reduce/watch")
                if liq_health["level"] == "DEAD":
                    st.error("☠ liquidity dead – pool practically unusable")
                elif liq_health["level"] == "CRITICAL":
                    st.error("🔴 liquidity critical – exit immediately")
                elif liq_health["level"] == "WEAK":
                    st.warning("⚠ liquidity weak – reduce risk")
                for fl in liq_health.get("flags", []):
                    st.caption(f"– {fl}")
                if level in ("EXIT", "EARLY"):
                    if level == "EXIT":
                        st.error(f"{level} ({exit_signal.get('exit_score', 0)})")
                    else:
                        st.warning(f"{level} ({exit_signal.get('exit_score', 0)})")
                    for f in exit_signal.get("exit_flags", [])[:3]:
                        st.caption(f"– {f}")
                    st.caption(
                        f"Persistence: exit_hits={persistence.get('exit_hits', 0)}, "
                        f"early_hits={persistence.get('early_hits', 0)}"
                    )
            note_key = f"note_{idx}_{hkey(base_addr, chain)}"
            note_val = st.text_input("Note", value=r.get("note", ""), key=note_key)
            avg_key = f"avg_entry_{idx}_{hkey(base_addr, chain)}"
            avg_entry = st.number_input("Avg entry price", min_value=0.0, value=float(entry_price), format="%.12f", key=avg_key)
            if st.button("Save entry price", key=f"save_avg_{idx}_{hkey(base_addr, chain)}", use_container_width=True):
                all_rows = load_portfolio()
                target_key = addr_key(chain, base_addr)
                for rr in all_rows:
                    if addr_key(rr.get("chain", ""), rr.get("base_token_address", "")) == target_key:
                        rr["avg_entry_price"] = str(avg_entry)
                        rr["updated_at"] = now_utc_str()
                        break
                save_portfolio(all_rows)
                st.success("Average entry price saved.")
                request_rerun()

        with c4:
            close_key = f"close_{idx}_{hkey(base_addr, chain)}"
            delete_key = f"delete_{idx}_{hkey(base_addr, chain)}"

            close = st.checkbox("Close (archive)", value=False, key=close_key)
            delete = st.checkbox("Delete row", value=False, key=delete_key)

            if st.button("Apply", key=f"apply_{idx}_{hkey(base_addr, chain)}", use_container_width=True):
                all_rows = load_portfolio()
                target_key = addr_key(chain, base_addr)
                for rr in all_rows:
                    if rr.get("active") != "1":
                        continue
                    if addr_key(rr.get("chain", ""), rr.get("base_token_address", "")) == target_key:
                        rr["note"] = note_val
                        if close:
                            rr["active"] = "0"
                        break

                if delete:
                    all_rows = [
                        x for x in all_rows
                        if not (
                            (x.get("active") == "1")
                            and (addr_key(x.get("chain", ""), x.get("base_token_address", "")) == target_key)
                        )
                    ]

                save_portfolio(all_rows)
                st.success("Saved.")
                request_rerun()

        # Sparklines for portfolio too (requested)
        with st.expander("Dynamics (sparklines)", expanded=False):
            if not hist:
                st.info("No snapshots yet.")
            else:
                dfh = pd.DataFrame(hist).copy()
                dfh["ts_utc"] = pd.to_datetime(dfh.get("ts_utc", ""), errors="coerce")
                dfh = dfh.sort_values("ts_utc")
                dfh["price_usd"] = pd.to_numeric(dfh.get("price_usd", 0), errors="coerce")
                dfh["score_live"] = pd.to_numeric(dfh.get("score_live", 0), errors="coerce")
                dfh["liq_usd"] = pd.to_numeric(dfh.get("liq_usd", 0), errors="coerce")
                dfh["vol24_usd"] = pd.to_numeric(dfh.get("vol24_usd", 0), errors="coerce")
                dfh["vol5_usd"] = pd.to_numeric(dfh.get("vol5_usd", 0), errors="coerce")
                dfh = dfh.set_index("ts_utc")

                s1, s2, s3 = st.columns(3)
                with s1:
                    st.caption("Price")
                    st.line_chart(dfh[["price_usd"]], height=120, use_container_width=True)
                with s2:
                    st.caption("Score")
                    st.line_chart(dfh[["score_live"]], height=120, use_container_width=True)
                with s3:
                    st.caption("Liquidity")
                    st.line_chart(dfh[["liq_usd"]], height=120, use_container_width=True)

                s4, s5 = st.columns(2)
                with s4:
                    st.caption("Vol24")
                    st.line_chart(dfh[["vol24_usd"]], height=120, use_container_width=True)
                with s5:
                    st.caption("Vol5")
                    st.line_chart(dfh[["vol5_usd"]], height=120, use_container_width=True)

        st.markdown("---")

    if closed_rows:
        with st.expander("Closed / Archived"):
            for r in closed_rows[-50:]:
                st.write(
                    f"{r.get('ts_utc','')} – {r.get('base_symbol','')}/{r.get('quote_symbol','')} "
                    f"– entry ${r.get('entry_price_usd','')} – {r.get('action','')}"
                )


# =============================
# App main
# =============================
def build_active_monitoring_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active_rows: List[Dict[str, Any]] = []
    for r in rows:
        status = str(r.get("status", "")).strip().upper()
        if status in ("ACTIVE", "WATCH", "WAIT", "", "TRACK", "READY"):
            active_rows.append(r)

    active_rows = sorted(
        active_rows,
        key=lambda x: parse_float(x.get("entry_score", x.get("last_score", 0)), 0.0),
        reverse=True,
    )[:150]
    return active_rows


def main():
    with st.expander("Debug / Telegram", expanded=False):
        st.write("TG ENGINE: NEW V3")
        if st.button("TEST TG"):
            ok = send_telegram(f"TEST {time.time()}")
            if ok:
                st.success("TG SENT")
            else:
                st.error("TG FAILED")

    if st.session_state.get("_rerun_flag"):
        st.session_state["_rerun_flag"] = False
        st.rerun()

    ensure_storage()
    if "_last_status" not in st.session_state:
        st.session_state["_last_status"] = {}
    if "_migrated_reason_fields" not in st.session_state:
        migrate_reason_fields()
        st.session_state["_migrated_reason_fields"] = True
    portfolio_rows = load_csv(PORTFOLIO_CSV, PORTFOLIO_FIELDS)
    monitoring_rows = load_csv(MONITORING_CSV, MON_FIELDS)
    monitoring_history_rows = load_csv(MON_HISTORY_CSV, HIST_FIELDS)
    scan_state = scanner_state_load()
    active_monitoring_rows = build_active_monitoring_rows(monitoring_rows)
    active_portfolio_rows = [
        row
        for row in portfolio_rows
        if str(row.get("active", row.get("is_active", "1"))).strip().lower() not in {"0", "false", "no"}
    ]
    if st.session_state.get("last_scan_ts") != scan_state.get("last_run_ts"):
        st.session_state["last_scan_ts"] = scan_state.get("last_run_ts")

        run_auto_notifications(
            scan_state,
            active_monitoring_rows,
            active_portfolio_rows
        )
    st.session_state["_preload_counts"] = {
        "portfolio": len(portfolio_rows),
        "monitoring": len(monitoring_rows),
        "monitoring_history": len(monitoring_history_rows),
    }
    st.sidebar.caption(f"📥 portfolio: {len(portfolio_rows)}")

    scanner_seeds_raw = str(DEFAULT_SEEDS)
    scanner_max_items = 100
    use_birdeye_trending = True
    birdeye_limit = 50
    auto_refresh_enabled = False
    ui_autorefresh_sec = 60

    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")
        chain = st.selectbox(
            "Chain",
            ["solana", "bsc"],
            index=0,
            key="chain_select",
        )
        st.session_state["chain"] = chain

        if "page" not in st.session_state:
            st.session_state["page"] = "Monitoring"
        
        alert_n = portfolio_alert_count()
        portfolio_label = "Portfolio"
        if alert_n > 0:
            portfolio_label = f"Portfolio 🔴 {alert_n}"

        pages = ["Monitoring", "Archive", portfolio_label]

        if str(st.session_state.get("page", "Monitoring")).startswith("Portfolio"):
            st.session_state["page"] = portfolio_label

        if st.session_state.get("page") not in pages:
            st.session_state["page"] = "Monitoring"

        page = st.radio("Page", pages, index=pages.index(st.session_state["page"]))
        if page.startswith("Portfolio"):
            page = "Portfolio"

        st.divider()
        st.caption("Scanner")
        scanner_max_items = st.slider("Max tokens per slot", 10, 100, 100, step=10)
        auto_refresh_enabled = st.checkbox("Auto refresh", value=False)
        ui_autorefresh_sec = st.slider("Auto-refresh interval (sec)", 60, 300, 60, step=10, disabled=(not auto_refresh_enabled))
        use_birdeye_trending = st.checkbox("Use Solana extra stream (Birdeye)", value=True, disabled=(not BIRDEYE_ENABLED))
        if not BIRDEYE_ENABLED:
            st.caption("Birdeye key missing: add BIRDEYE_API_KEY to secrets to enable extra Solana stream.")
        birdeye_limit = st.slider("Birdeye trending size", 10, 100, 50, step=10, disabled=(not use_birdeye_trending or not BIRDEYE_ENABLED))
        scanner_seeds_raw = st.text_area("Scanner seeds", value=str(DEFAULT_SEEDS), height=120)

        st.divider()
        st.caption("Monitoring")
        auto_archive_enabled = st.checkbox("Enable auto-archive", value=True)
        auto_archive_min_score = st.slider("Auto-archive if avg score < …", 0, 900, 150, step=10)
        auto_archive_on_no_entry = st.checkbox("Archive on persistent NO ENTRY", value=False)
        auto_revisit_enabled = st.checkbox("Auto-revisit auto-archived", value=True)
        auto_revisit_days = st.slider("Auto-revisit after (days)", 1, 30, 7, step=1)
        stability_window_n = st.slider("Stability window (snapshots)", 10, 120, 30, step=5)
        hide_red = st.checkbox("Hide red signals", value=False)
        portfolio_value_usd = st.number_input(
            "Portfolio size (USD)",
            min_value=100.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
        )

        st.divider()
        if st.button("Run scanner now", use_container_width=True):
            stats = run_full_ingestion_now(
                chain=str(st.session_state.get("chain", "solana")),
                seeds_raw=str(scanner_seeds_raw or ""),
                max_items=int(scanner_max_items),
                use_birdeye_trending=bool(use_birdeye_trending),
                birdeye_limit=int(birdeye_limit),
            )
            st.session_state["_scan_feedback"] = f"Scanner saved: {stats}"
            request_rerun()

        if st.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            request_rerun()
        st.markdown("### Background alerts limitation")
        st.caption("This Streamlit app does not run alerts in the background when the browser/session is closed.")
        st.caption("For true background alerts, run scanner_worker.py as a separate worker service.")
        st.caption("TG mode: ENV + Streamlit secrets fallback")
        st.divider()
        st.markdown("### Storage status")
        if _sb_ok():
            st.success("Supabase: ON")
        else:
            st.error("Supabase: OFF")
        st.caption(f"URL: {bool(SUPABASE_URL)} | KEY: {bool(SUPABASE_SERVICE_ROLE_KEY)}")
        if _sb_ok():
            test = sb_get_storage("monitoring.csv")
            if test is None:
                st.warning("Supabase reachable but EMPTY / no data")

        if st.session_state.get("_save_badge"):
            st.caption(str(st.session_state.get("_save_badge")))
        DEBUG_MODE = st.checkbox("🛠 Debug mode", value=False)

    if DEBUG_MODE:
        render_debug_panel()

    if st.session_state.get("_scan_feedback"):
        st.info(st.session_state.pop("_scan_feedback"))

    auto_cfg = dict(
        auto_archive_enabled=auto_archive_enabled,
        auto_archive_min_score=auto_archive_min_score,
        auto_archive_on_no_entry=auto_archive_on_no_entry,
        auto_revisit_enabled=auto_revisit_enabled,
        auto_revisit_days=auto_revisit_days,
        stability_window_n=stability_window_n,
        scanner_seeds_raw=scanner_seeds_raw,
        scanner_max_items=scanner_max_items,
        use_birdeye_trending=use_birdeye_trending,
        birdeye_limit=birdeye_limit,
        ui_autorefresh_sec=ui_autorefresh_sec,
        auto_refresh_enabled=auto_refresh_enabled,
        hide_red=hide_red,
        portfolio_value_usd=portfolio_value_usd,
    )

    try:
        if page == "Monitoring":
            page_monitoring(auto_cfg)
        elif page == "Archive":
            page_archive()
        else:
            page_portfolio()
    except Exception as e:
        log_error(e)
        st.error(f"Page render error: {e}")
        debug_log(f"page_render_crash: {type(e).__name__}:{e}")

    with st.expander("Debug log"):
        for line in st.session_state.get("debug_log", []):
            st.text(line)

    maybe_safe_auto_refresh(
        enabled=bool(auto_cfg.get("auto_refresh_enabled", False)),
        interval_sec=int(auto_cfg.get("ui_autorefresh_sec", 60)),
    )


# =============================
# Birdeye (optional, Solana trending + extra stats)
# =============================
BIRDEYE_API_KEY = _get_secret("BIRDEYE_API_KEY", "").strip()
BIRDEYE_ENABLED = bool(BIRDEYE_API_KEY)

def birdeye_get(path: str, params: Optional[dict] = None, timeout: int = 15) -> dict:
    """Minimal Birdeye public API wrapper."""
    if not BIRDEYE_ENABLED:
        return {}
    url = "https://public-api.birdeye.so" + path
    headers = {"X-API-KEY": BIRDEYE_API_KEY, "Accept": "application/json"}
    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json() or {}

@st.cache_data(ttl=45, show_spinner=False)
def birdeye_trending_solana(limit: int = 50, offset: int = 0) -> list[str]:
    """
    Returns a list of Solana mint addresses from Birdeye trending endpoint.
    If Birdeye changes params – we fail soft and return [].
    """
    if not BIRDEYE_ENABLED:
        return []
    try:
        params = {
            "chain": "solana",
            "offset": int(offset),
            "limit": int(limit),
            "sort_by": "rank",
            "sort_type": "asc",
        }
        data = birdeye_get("/defi/token_trending", params=params, timeout=15)
        items = (data.get("data") or {}).get("items") or (data.get("data") or {}).get("tokens") or []
        out = []
        for it in items:
            addr = (it.get("address") or it.get("mint") or it.get("tokenAddress") or "").strip()
            if addr:
                out.append(addr)
        return out[: int(limit)]
    except Exception:
        return []

def run_app():
    try:
        main()
    except Exception as e:
        st.error(f"App crash: {e}")
        debug_log(f"app_crash: {type(e).__name__}:{e}")


if __name__ == "__main__":
    run_app()
