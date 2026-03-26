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
from pathlib import Path
from datetime import datetime
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


VERSION = "0.5.5"
DEX_BASE = "https://api.dexscreener.com"
DATA_DIR = "data"
SMART_WALLET_FILE = os.path.join(DATA_DIR, "smart_wallets.json")

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
    # Streamlit secrets
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name) or default)
    except Exception:
        pass
    # Env fallback
    return str(os.environ.get(name, default) or default)

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
            return None
        r.raise_for_status()
        data = r.json() or []
        if not data:
            return None
        return (data[0].get("content") or "")
    except Exception:
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
            st.sidebar.error(f"❌ Supabase write failed {key}: {res.text}")
            return False

        return True
    except Exception as e:
        st.sidebar.error(f"❌ Supabase EXCEPTION {key}: {e}")
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
                return _csv_from_string(content)
            except Exception:
                st.sidebar.error(f"❌ Corrupt Supabase CSV: {key}")

    fallback = local_fallback_path(path)
    if os.path.exists(fallback):
        try:
            with open(fallback, "r", newline="", encoding="utf-8") as f:
                st.sidebar.warning(f"⚠️ Using local fallback: {key}")
                return list(csv.DictReader(f))
        except Exception as e:
            st.sidebar.error(f"❌ Local fallback read failed {key}: {e}")

    return []


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
        st.sidebar.error(f"❌ Local fallback write failed {key}: {e}")

    if _sb_ok():
        ok = sb_put_storage(key, content)
        if ok:
            check = sb_get_storage(key)
            if check:
                st.sidebar.success(f"✅ Stored in Supabase: {key}")
            else:
                st.sidebar.error(f"❌ Write OK but read FAIL: {key}")
        else:
            st.sidebar.warning(f"⚠️ Supabase write failed, local fallback kept: {key}")


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

NAME_OK_RE = re.compile(r"^[\w\-\.\$\s]{1,40}$", re.UNICODE)

MAJORS_STABLES = {
    "BTC", "WBTC", "ETH", "WETH", "BNB", "WBNB", "SOL", "WSOL",
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDE", "USDS",
    "WAVAX", "WMATIC",
}

DEFAULT_SEEDS = (
    "WBNB, USDT, meme, memecoin, ai, ai agent, agentic, launch, new, listing, trending, hot, hype, pump, "
    "sol, eth, usdc, pepe, trump, inu, dog, cat, frog, shiba, "
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

@st.cache_data(ttl=60, max_entries=500, show_spinner=False)
def fetch_latest_pairs_for_query(q: str) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/latest/dex/search"
    data = _http_get_json(url, params={"q": q.strip()}, timeout=20, max_retries=3)
    return data.get("pairs", []) or []


@st.cache_data(ttl=60, max_entries=500, show_spinner=False)
def fetch_token_pairs(chain: str, token_address: str) -> List[Dict[str, Any]]:
    url = f"{DEX_BASE}/token-pairs/v1/{chain}/{token_address}"
    data = _http_get_json(url, params=None, timeout=20, max_retries=3)
    return data or []

@st.cache_data(ttl=60, max_entries=500)
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

    chain = (pair.get("chainId") or "").lower().strip()
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
    chain = pair.get("chainId")

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

    if decision == "NO ENTRY":
        if score < 260 and strength_flags < 2:
            return False, "gate: no_entry_weak"

    if score < 180 and strength_flags < 2:
        return False, "gate: weak_score"

    if score < 230 and vol5 < 2500 and pc1h < 5 and strength_flags < 3:
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


def normalize_pair_row(pair: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(pair)
    row["meme_score"] = meme_token_score(row)
    row["smart_money"] = detect_smart_wallet_activity(row)

    mig = detect_liquidity_migration(row)
    row["migration"] = mig.get("migration", 0)
    row["migration_label"] = mig.get("migration_label", "")
    row["migration_score"] = mig.get("migration_score", 0)

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


def sample_seeds(seeds: List[str], k: int, refresh: bool) -> List[str]:
    cleaned = []
    for s in seeds:
        s = (s or "").strip()
        if not s:
            continue
        if len(s) < 2:
            continue
        if s.lower() in {"x"}:
            continue
        cleaned.append(s)

    if not cleaned:
        return []

    k = max(1, min(int(k), len(cleaned)))

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
    "source_window",
    "source_preset",
    "risk",
    "tp_target_pct",
    "entry_suggest_usd",
    "ts_last_seen",
    "signal",
    "smart_money",
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
        best = best_pair_for_token(chain, base_addr)
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


def load_monitoring_history(limit_rows: int = 8000) -> List[Dict[str, Any]]:
    # Source of truth: load_csv (Supabase app_storage if enabled, else local)
    rows = load_csv(MON_HISTORY_CSV)
    if not rows:
        return []
    return rows[-limit_rows:]


def append_monitoring_history(row: Dict[str, Any]):
    payload = {k: row.get(k, "") for k in HIST_FIELDS}
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


def add_to_monitoring(p: Dict[str, Any], score: float, window_name: str = "", preset_key: str = "") -> str:
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
    decision, _tags = build_trade_hint(p)

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
            "source_window": window_name or "",
            "source_preset": preset_key or "",
            "risk": risk,
            "tp_target_pct": tp_s,
            "entry_suggest_usd": entry_s,
            "ts_last_seen": now_utc_str(),
            "signal": classify_signal(p),
            "smart_money": "1" if detect_smart_money(p) else "0",
        }
    )
    save_monitoring(rows)
    return "OK"


def archive_monitoring(chain: str, base_addr: str, reason: str, last_score: float = 0.0, last_decision: str = "") -> bool:
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
            changed = True
    if changed:
        save_monitoring(rows)
    return changed




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
    st.sidebar.success("📦 Portfolio saved (trigger)")
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
        entry_price = parse_float(r.get("entry_price_usd") or r.get("entry_price") or 0, 0.0)

        best = best_pair_for_token(chain, base_addr) if (chain and base_addr) else None
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

    if signal == "RED" or trap_level == "CRITICAL":
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
    ("Ultra Early (safer)", "ultra", "bsc"),
    ("Ultra Early (safer)", "ultra", "solana"),
    ("Balanced (default)", "balanced", "bsc"),
    ("Balanced (default)", "balanced", "solana"),
    ("Wide Net (explore)", "wide", "bsc"),
    ("Wide Net (explore)", "wide", "solana"),
    ("Momentum (hot)", "momentum", "bsc"),
    ("Momentum (hot)", "momentum", "solana"),
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

def scout_collect_candidates(chain: str, window_name: str, preset: Dict[str, Any], seeds_raw: str, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> List[Dict[str, Any]]:
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
        return []

    sampled = sample_seeds(seeds, int(preset.get("seed_k", 12)), refresh=False)
    all_pairs: List[Dict[str, Any]] = []
    for q in sampled:
        # Ultra early pairs
        try:
            latest = fetch_dexscreener_latest(chain, limit=40)
            if latest:
                all_pairs.extend(latest)
        except Exception:
            pass
        if len(q.strip()) < 2:
            continue
        try:
            all_pairs.extend(fetch_latest_pairs_for_query(q))
            time.sleep(0.06)
        except Exception:
            continue
        # Dexscreener trending pairs
        try:
            trending = fetch_dexscreener_trending(chain, limit=40)
            if trending:
                all_pairs.extend(trending)
        except Exception:
            pass
            
    # Optional Solana extra stream via Birdeye trending
    if chain == "solana" and use_birdeye_trending and BIRDEYE_ENABLED:
        try:
            for mint in birdeye_trending_solana(limit=int(birdeye_limit)):
                try:
                    pools = fetch_token_pairs("solana", mint)
                    if pools:
                        all_pairs.extend(pools)
                except Exception:
                    continue
        except Exception:
            pass

    pairs = dedupe_mode(all_pairs, by_base_token=False)
    pairs = sorted(
        pairs,
        key=lambda p: float(safe_get(p, "volume", "h24", default=0) or 0),
        reverse=True,
    )
    pairs = dedupe_mode(pairs, by_base_token=bool(preset.get("dedupe_by_base", True)))

    filtered, _fstats, _freasons = filter_pairs_with_debug(
        pairs=pairs,
        chain=chain,
        any_dex=True,
        allowed_dexes=set(),
        min_liq=float(preset.get("min_liq", 1000)),
        min_vol24=float(preset.get("min_vol24", 5000)),
        min_trades_m5=int(preset.get("min_trades_m5", 0)),
        min_sells_m5=int(preset.get("min_sells_m5", 0)),
        max_buy_sell_imbalance=int(preset.get("max_imbalance", 30)),
        block_suspicious_names=bool(preset.get("block_suspicious_names", True)),
        block_majors=bool(preset.get("block_majors", True)),
        min_age_min=int(preset.get("min_age_min", 0)),
        max_age_min=int(preset.get("max_age_min", 999999)),
        enforce_age=bool(preset.get("enforce_age", True)),
        hide_solana_unverified=bool(preset.get("hide_solana_unverified", True)),
    )

    out = []
    for p in filtered:
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
        if not base_addr:
            continue
        decision, _tags = build_trade_hint(p)
        if str(decision).upper() == "NO ENTRY":
            continue
        out.append(p)

    SCAN_CACHE[cache_key] = {"ts": now_ts, "pairs": out}
    return out

def ingest_window_to_monitoring(chain: str, window_name: str, preset_key: str, seeds_raw: str, max_items: int = 100, use_birdeye_trending: bool = True, birdeye_limit: int = 50) -> Dict[str, int]:
    preset = PRESETS.get(window_name, {})
    counts = {"added": 0, "skipped_active": 0, "skipped_archived": 0, "skipped_noise": 0, "errors": 0, "seen": 0}
    pairs = scout_collect_candidates(chain=chain, window_name=window_name, preset=preset, seeds_raw=seeds_raw, use_birdeye_trending=use_birdeye_trending, birdeye_limit=birdeye_limit)
    ranked = []
    smart_wallets = load_smart_wallets()
    for p in pairs:
        smart = detect_smart_money(p)
        signal = classify_signal(p)
        score = score_pair(p)
        if smart:
            score += 4
        ranked.append((score, p, smart, signal))
    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[: max(1, int(max_items))]
    for s, p, smart, signal in ranked:
        counts["seen"] += 1
        row = normalize_pair_row(p)
        row["smart_money"] = smart
        row["signal"] = signal
        if row["signal"] == "RED":
            add_to_archive(
                chain=row.get("chainId"),
                base_addr=safe_get(row, "baseToken", "address", default=""),
                reason="auto_filter"
            )
            continue

        allowed, _gate_reason = passes_monitoring_gate(row, float(s))
        if not allowed:
            counts["skipped_noise"] = counts.get("skipped_noise", 0) + 1
            continue

        if smart:
            chain = (row.get("chainId") or "").lower()
            base_addr = (safe_get(row, "baseToken", "address", default="") or "").strip()
            key = addr_key(chain, base_addr)
            smart_wallets[key] = {
                "last_seen": now_utc_str(),
                "symbol": row.get("base_symbol", "")
            }
        priority = scout_priority(row)
        if priority < 3:
            counts["skipped_noise"] = counts.get("skipped_noise", 0) + 1
            continue

        try:
            contract = addr_store(row.get("chainId", ""), safe_get(row, "baseToken", "address", default=""))
            if token_exists(contract):
                counts["skipped_active"] += 1
                continue

            res = add_to_monitoring(row, float(s), window_name=window_name, preset_key=preset_key)

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
                "score_live": str(float(s)),
                "decision": "SCOUT_SCAN",
            })

            if res == "OK":
                counts["added"] += 1
            elif res == "EXISTS_ACTIVE":
                counts["skipped_active"] += 1
            elif res == "EXISTS_ARCHIVED":
                counts["skipped_archived"] += 1
            if detect_auto_signal(row):
                add_to_monitoring(row, float(s), window_name="AUTO_SIGNAL", preset_key="AUTO_SIGNAL")
        except Exception as e:
            log_error(e)
            counts["errors"] += 1
    save_smart_wallets(smart_wallets)
    return counts

def auto_reactivate_archived(days: int = 7) -> int:
    days = max(1, int(days))
    rows = load_monitoring()
    changed = 0
    now_dt = datetime.utcnow()
    for r in rows:
        if r.get("active") == "1":
            continue
        reason = (r.get("archived_reason") or "").strip().lower()
        if not reason.startswith("auto:"):
            continue
        ts = _safe_dt_parse(r.get("ts_archived", ""))
        if not ts:
            continue
        age_days = (now_dt - ts).total_seconds() / 86400.0
        if age_days >= float(days):
            r["active"] = "1"
            r["ts_added"] = now_utc_str()
            r["ts_archived"] = ""
            r["archived_reason"] = ""
            r["ts_last_seen"] = now_utc_str()
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

    # distributed lock (prevents multi-tab race)
    if not scanner_acquire_lock(slot):
        return {"ran": False, "slot": slot, "window": window_name, "chain": chain, "stats": state.get("last_stats", {})}
    
    if last_slot == slot:
        return {"ran": False, "slot": slot, "window": window_name, "chain": chain, "stats": state.get("last_stats", {})}
    try:
        stats = ingest_window_to_monitoring(chain=chain, window_name=window_name, preset_key=preset_key, seeds_raw=seeds_raw, max_items=max_items, use_birdeye_trending=use_birdeye_trending, birdeye_limit=birdeye_limit)
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
                    bp = best_pair_for_token("solana", mint)
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
        chain_id = (p.get("chainId") or "").lower().strip()
        base_addr = (safe_get(p, "baseToken", "address", default="") or "").strip()
        if not base_addr:
            continue
        key = addr_key(chain_id, base_addr)

        if key in mon_set or key in port_set:
            continue

        whale = whale_accumulation(p)
        fresh = fresh_lp(p)

        s = score_pair(p)
        if whale:
            s += 3
        if fresh == "VERY_NEW":
            s += 2
        s += pump_probability(p) * 0.5
        decision, tags = build_trade_hint(p)

        if decision.strip().upper() == "NO ENTRY":
            continue

        row = normalize_pair_row(p)
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
        sniper_flag = detect_snipers(pobj)
        trap_signal, trap_level = liquidity_trap(pobj)

        st.markdown("---")
        st.subheader(f"{base}/{quote}")
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
                add_to_monitoring(pobj, float(score))
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
            st.code(base_addr)
            st.caption("Pair / pool address")
            st.code(pair_addr)

        if chain_id == "solana":
            st.caption("Solana: check Jupiter/JupShield warnings before swapping.")

    for i, (_s, _decision, _tags, p) in enumerate(ranked, start=1):
        render_scout_card(p, i)


def page_monitoring(auto_cfg: Dict[str, Any]):
    st.title("Monitoring")
    st.caption("Автоматичний pipeline: rotating scans → Monitoring → Signals / Watchlist → Archive.")

    if bool(auto_cfg.get("ui_autorefresh_sec", 0)):
        _maybe_autorefresh(int(auto_cfg.get("ui_autorefresh_sec", 0)) * 1000, key="monitoring_refresh")

    if bool(auto_cfg.get("auto_revisit_enabled", False)):
        try:
            n = auto_reactivate_archived(days=int(auto_cfg.get("auto_revisit_days", 7)))
            if n:
                st.caption(f"Auto-revisited from Archive: {n}")
        except Exception:
            pass

    scan_state = scanner_state_load()
    top = st.columns([2,2,3,3])
    rows = load_monitoring()
    active = [r for r in rows if r.get("active") == "1"]
    archived = [r for r in rows if r.get("active") != "1"]
    top[0].metric("Active", len(active))
    top[1].metric("Archived", len(archived))
    top[2].caption(f"Last scan: {scan_state.get('last_run_ts','—')} • {scan_state.get('last_window','—')} • {scan_state.get('last_chain','—')}")
    top[3].caption(f"Last stats: {scan_state.get('last_stats', {})}")

    cbtn1, cbtn2, cbtn3 = st.columns([2,2,6])
    with cbtn1:
        if st.button("Run scanner now", use_container_width=True, key="run_scanner_now"):
            res = run_scanner_now(
                seeds_raw=str(auto_cfg.get("scanner_seeds_raw", DEFAULT_SEEDS)),
                max_items=int(auto_cfg.get("scanner_max_items", 100)),
                use_birdeye_trending=bool(auto_cfg.get("use_birdeye_trending", True)),
                birdeye_limit=int(auto_cfg.get("birdeye_limit", 50)),
            )
            st.success(f"Scanner ran: {res['window']} / {res['chain']} / {res['stats']}")
            request_rerun()
    with cbtn2:
        if st.button("Refresh live data", use_container_width=True):
            request_rerun()
    with cbtn3:
        st.caption("The app checks scanner slot on every load. One slot = 5 minutes, rotating across 4 presets × 2 chains.")

    if not active:
        st.info("Monitoring is empty. Wait for scanner or run it now.")
        return

    chain_filter = st.selectbox("Chain filter", ["all", "bsc", "solana"], index=0)
    if chain_filter != "all":
        active = [r for r in active if (r.get("chain") or "").strip().lower() == chain_filter]

    active_set = active_portfolio_addresses()

    enriched = []
    for idx, r in enumerate(active, start=1):
        chain = (r.get("chain") or "").strip().lower()
        base_sym = r.get("base_symbol") or "???"
        base_addr = addr_store(chain, (r.get("base_addr") or "").strip())
        if addr_key(chain, base_addr) in active_set:
            continue
        best = best_pair_for_token(chain, base_addr) if (chain and base_addr) else None

        live = {"price": 0.0, "liq": 0.0, "vol24": 0.0, "vol5": 0.0, "pc1h": 0.0, "pc5": 0.0, "dex": "", "pair_addr": "", "url": "", "quote": ""}
        if best:
            live["dex"] = best.get("dexId", "") or ""
            live["pair_addr"] = best.get("pairAddress", "") or ""
            live["url"] = best.get("url", "") or ""
            live["quote"] = safe_get(best, "quoteToken", "symbol", default="") or ""
            live["price"] = parse_float(best.get("priceUsd"), 0.0)
            live["liq"] = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)
            live["vol24"] = parse_float(safe_get(best, "volume", "h24", default=0), 0.0)
            live["vol5"] = parse_float(safe_get(best, "volume", "m5", default=0), 0.0)
            live["pc1h"] = parse_float(safe_get(best, "priceChange", "h1", default=0), 0.0)
            live["pc5"] = parse_float(safe_get(best, "priceChange", "m5", default=0), 0.0)
            snapshot_live_to_history(chain, base_sym, base_addr, best)

        s_live = score_pair(best) if best else 0.0
        smart_money = detect_smart_money(best) if best else False
        if smart_money:
            s_live += 4
        pump_live = pump_probability(best)
        s_live += pump_live * 0.5
        decision, tags = build_trade_hint(best) if best else ("NO DATA", [])
        if decision == "NO ENTRY" and s_live < 200:
            archive_monitoring(chain, base_addr, reason="weak_signal", last_score=s_live, last_decision=decision)
            continue
        hist = token_history_rows(chain, base_addr, limit=int(auto_cfg.get("stability_window_n", 30)))
        anti_rug = anti_rug_early_detector(best, hist)
        flow_collapse = flow_collapse_detector(hist)
        if flow_collapse["level"] == "WARNING" and anti_rug["level"] == "SAFE":
            anti_rug = {
                "level": "WARNING",
                "score": max(int(anti_rug.get("score", 0)), 2),
                "flags": list(set((anti_rug.get("flags", []) or []) + flow_collapse["flags"])),
                "action": "REDUCE",
            }
        timing = entry_timing_signal(best, hist)
        if timing["timing"] == "SKIP":
            decision = "NO ENTRY"
        elif timing["timing"] == "NEUTRAL" and decision == "ENTRY (scalp)":
            decision = "WATCH / WAIT"

        if anti_rug["level"] == "CRITICAL":
            decision = "NO ENTRY"
        elif anti_rug["level"] == "WARNING" and decision == "ENTRY (scalp)":
            decision = "WATCH / WAIT"
        size_info = position_sizing_engine(
            best,
            portfolio_value_usd=float(auto_cfg.get("portfolio_value_usd", 1000.0)),
            hist=hist,
        ) if best else {
            "size_pct": 0.0,
            "size_label": "SKIP",
            "usd_size": 0.0,
            "risk_score": 10,
            "risk_flags": ["no_data"],
        }
        gov = portfolio_allocation_governor(size_info)
        corr = correlation_governor(best) if best else {"status": "BLOCK", "reason": ["no_data"]}
        if not corr:
            corr = {"status": "ALLOW", "reason": []}
        liq_health = liquidity_health(best)
        if liq_health["level"] in ("DEAD", "CRITICAL"):
            decision = "NO ENTRY"
        adj = risk_adjusted_size(
            float(size_info.get("usd_size", 0.0) or 0.0),
            corr,
            liq_health,
            anti_rug=anti_rug,
        )
        size_info["usd_size_adj"] = round(adj["final_size"], 2)
        size_info["size_factor"] = adj["factor"]
        size_info["size_reason"] = adj.get("reason", [])
        size_info["liq_health"] = liq_health["level"]
        if anti_rug.get("level") == "CRITICAL":
            size_info["size_label"] = "SKIP"
            size_info["size_pct"] = 0.0
        elif anti_rug.get("level") == "WARNING" and size_info.get("size_label") == "FULL":
            size_info["size_label"] = "SMALL"
        elif anti_rug.get("level") == "WARNING" and size_info.get("size_label") == "SMALL":
            size_info["size_label"] = "MICRO"
        final_status = gov["status"]  # correlation no longer blocks promote
        trap = liquidity_trap_detector(best, hist) if best else {"trap_score": 0, "trap_level": "SAFE", "trap_flags": []}

        # smart auto-archive
        if best and auto_cfg.get("auto_archive_enabled"):
            min_score = float(auto_cfg.get("auto_archive_min_score", 150.0))
            persistent_no_entry = bool(auto_cfg.get("auto_archive_on_no_entry", False))
            avg_score = 0.0
            if hist:
                scores = [parse_float(x.get("score_live", 0), 0.0) for x in hist[-6:]]
                if scores:
                    avg_score = sum(scores)/len(scores)
            target_score = avg_score if avg_score > 0 else s_live
            if anti_rug["level"] == "CRITICAL":
                archive_monitoring(chain, base_addr, reason="auto: EARLY_RUG", last_score=s_live, last_decision=decision)
                continue
            if rug_like(best, hist):
                archive_monitoring(chain, base_addr, reason="auto: RUG", last_score=s_live, last_decision=decision)
                continue
            if trap["trap_level"] == "CRITICAL":
                archive_monitoring(chain, base_addr, reason="auto: TRAP", last_score=s_live, last_decision=decision)
                continue
            if persistent_no_entry and hist:
                last_decisions = [str(x.get("decision", "")).upper() for x in hist[-4:]]
                if last_decisions and all(d == "NO ENTRY" for d in last_decisions):
                    archive_monitoring(chain, base_addr, reason="auto: NO FLOW", last_score=s_live, last_decision=decision)
                    continue
            if target_score > 0 and target_score < min_score:
                archive_monitoring(chain, base_addr, reason=f"auto: LOW SCORE<{int(min_score)}", last_score=s_live, last_decision=decision)
                continue

        meme_score = meme_token_score(best) if best else 0.0
        migration_score = 0.0
        migration_label = ""
        cex_prob = 0.0

        if best:
            mig = detect_liquidity_migration(best)
            migration_score = parse_float(mig.get("migration_score", 0), 0.0)
            migration_label = mig.get("migration_label", "")
            cex_prob = parse_float(best.get("cex_prob", 0), 0.0)

        pr = source_priority(r) * 10000 + monitoring_priority(
            s_live,
            live["pc1h"],
            live["pc5"],
            live["vol5"],
            live["liq"],
            meme_score=meme_score,
            migration_score=migration_score,
            cex_prob=cex_prob,
        )
        d_score = s_live - parse_float(r.get("score_init"), 0.0)
        d_liq = live["liq"] - parse_float(r.get("liq_init"), 0.0)
        d_v24 = live["vol24"] - parse_float(r.get("vol24_init"), 0.0)
        d_v5 = live["vol5"] - parse_float(r.get("vol5_init"), 0.0)

        stars = confidence_stars(best, hist)
        second_wave = second_wave_label(hist)
        stage = "WATCHLIST"
        if decision.upper().startswith("ENTRY"):
            stage = "SIGNAL"
        elif second_wave:
            stage = "SIGNAL"
        elif (r.get("risk") or "").upper() == "EARLY":
            stage = "SIGNAL" if s_live >= 300 else "WATCHLIST"

        trap_signal, trap_level = liquidity_trap(best)
        fresh = fresh_lp(best)
        dev = dev_wallet_risk(best)
        whale = whale_accumulation(best)
        signal = classify_signal(best) if best else (r.get("signal") or "RED")

        enriched.append((stage, pr, idx, r, best, s_live, decision, timing, tags, hist, live, d_score, d_liq, d_v24, d_v5, stars, second_wave, trap, migration_label, migration_score, detect_snipers(best), pump_live, trap_signal, trap_level, fresh, dev, whale, signal, smart_money, size_info, corr, final_status, liq_health, anti_rug))

    rows_now = load_monitoring()
    if len([r for r in rows_now if r.get("active") == "1"]) != len(active):
        st.info("Archive/revisit changes applied. Refreshing list…")
        request_rerun()

    priority = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    timing_priority = {"GOOD": 0, "NEUTRAL": 1, "SKIP": 2}
    enriched.sort(
        key=lambda x: (
            timing_priority.get(((x[7] or {}).get("timing") or "SKIP"), 2),
            priority.get(x[27], 2),
            -x[5],
        )
    )
    signals = [x for x in enriched if x[0] == "SIGNAL"]
    watchlist = [x for x in enriched if x[0] != "SIGNAL"]

    def render_items(title: str, items: List[tuple]):
        st.subheader(title)
        if not items:
            st.caption("Порожньо")
            return

        for stage, pr, idx, r, best, s_live, decision, timing, tags, hist, live, d_score, d_liq, d_v24, d_v5, stars, second_wave, trap, migration_label, migration_score, sniper_flag, pump_score, trap_signal, trap_level, fresh, dev, whale, signal, smart_money, size_info, corr, final_status, liq_health, anti_rug in items:
            exit_signal = exit_before_dump_detector(best, hist, 0.0)
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
            level = str(exit_signal.get("exit_level", "WATCH"))
            action_plan = action_from_exit_signal(exit_signal, persistence)
            if bool(auto_cfg.get("hide_red", True)) and signal == "RED":
                continue

            chain = (r.get("chain") or "").strip().lower()
            base_sym = r.get("base_symbol") or "???"
            base_addr = addr_store(chain, (r.get("base_addr") or "").strip())
            st.markdown("---")
            c1, c2, c3, c4 = st.columns([3,2,2,2])
            with c1:
                title_bits = [base_sym]
                if stars:
                    title_bits.append(stars)
                if second_wave:
                    title_bits.append(second_wave)
                st.subheader(" ".join(title_bits))
                st.caption(f"{chain} • src: {(r.get('source_window') or '—')} • preset: {(r.get('source_preset') or '—')} • last_seen: {(r.get('ts_last_seen') or '—')}")
                st.code(base_addr, language="text")
                if live["url"]:
                    link_button("DexScreener", live["url"], use_container_width=True, key=f"m_ds_{idx}_{hkey(base_addr)}")
                swap_url = build_swap_url(chain, base_addr)
                if swap_url:
                    label = "Swap (Jupiter)" if chain == "solana" else "Swap (PancakeSwap)"
                    link_button(label, swap_url, use_container_width=True, key=f"m_sw_{idx}_{hkey(base_addr, chain)}")
                if chain == "solana":
                    st.caption("Solana extra stream: Birdeye trending + Jupiter swap link.")
            with c2:
                st.markdown("Live")
                st.write(f"Score: {s_live:.2f}" if best else "Score: n/a")
                st.write(f"Price: ${live['price']:.8f}" if live['price'] else "Price: n/a")
                st.write(f"Liq: {fmt_usd(live['liq'])}")
                st.write(f"Vol24: {fmt_usd(live['vol24'])}")
                st.write(f"Vol5: {fmt_usd(live['vol5'])}")
                st.caption(f"Δ1h {fmt_pct(live['pc1h'])} • Δ5m {fmt_pct(live['pc5'])}")
                st.write(f"SNIPER: {'YES' if sniper_flag else 'NO'}")
                st.write(f"PUMP: {pump_score}")
                st.write(f"Trap: {trap.get('trap_level', 'SAFE')} ({trap.get('trap_score', 0)})")
                if level == "EARLY":
                    st.caption("⚠ early exit forming")
            with c3:
                st.markdown("Plan")
                st.write(f"Decision: {decision}")
                st.write(f"Entry timing: {timing['timing']}")
                for reason in timing.get("reason", []):
                    st.caption(f"– {reason}")
                st.write(f"Risk: {(r.get('risk') or 'standard')}")
                st.write(f"Suggested entry: ${r.get('entry_suggest_usd') or '—'}")
                st.write(f"TP target: {r.get('tp_target_pct') or '—'}%")
                st.write(f"Δscore: {d_score:+.2f}")
                st.write(f"Size: {size_info['size_label']}")
                st.write(f"Suggested size: {size_info['size_pct']}%")
                base_usd_size = float(size_info.get("usd_size", 0.0) or 0.0)
                adjusted_usd_size = float(size_info.get("usd_size_adj", base_usd_size) or base_usd_size)
                size_factor = float(size_info.get("size_factor", 1.0) or 1.0)
                if size_factor < 1.0:
                    st.write(f"USD size: ${adjusted_usd_size:.2f} (reduced from ${base_usd_size:.2f})")
                else:
                    st.write(f"USD size: ${base_usd_size:.2f}")
                st.write(f"Risk score: {size_info['risk_score']}")
                corr_status = str(corr.get("status", "ALLOW"))
                corr_reasons = corr.get("reason", []) or []
                st.write(f"Correlation: {corr_status}")
                for rr in corr_reasons:
                    st.caption(f"– {rr}")
                st.write(f"Liquidity health: {liq_health['level']}")
                st.write(f"Anti-rug: {anti_rug['level']} ({anti_rug['score']})")
                st.write(f"Can enter safely: {'YES' if liq_health['level']=='OK' and anti_rug['level']!='CRITICAL' else 'NO'}")
                st.caption(f"size factor: {float(size_info.get('size_factor', 1.0) or 1.0):.2f}")
                for fl in liq_health.get("flags", []):
                    st.caption(f"– {fl}")
                for fl in anti_rug.get("flags", [])[:4]:
                    st.caption(f"– {fl}")
                if size_info.get("size_reason"):
                    st.caption("Size adjustments:")
                    for rr in size_info["size_reason"]:
                        st.caption(f"– {rr}")
                if size_info["risk_flags"]:
                    st.caption("Risk flags:")
                    for rf in size_info["risk_flags"]:
                        st.caption(f"– {rf}")
                if migration_label:
                    st.info(f"MIGRATION: {migration_label}")
            with c4:
                st.markdown("Status")
                label = f"{decision} {stars}".strip()
                if second_wave and not str(decision).upper().startswith("ENTRY"):
                    label = f"ENTRY (2nd wave) {stars}".strip()
                st.markdown(action_badge(label), unsafe_allow_html=True)
                if signal == "GREEN":
                    st.success("ENTRY signal")
                elif signal == "YELLOW":
                    st.warning("WATCH")
                else:
                    st.error("NO ENTRY")
                if sniper_flag:
                    st.markdown("🧠 sniper activity")
                if pump_score >= 6:
                    st.markdown("🚀 pump probability HIGH")
                cex_label = (best or {}).get("cex_label") or r.get("cex_label")
                if cex_label == "HIGH":
                    st.success("Possible CEX listing")
                elif cex_label == "MEDIUM":
                    st.caption("CEX watch")
                if trap_signal:
                    st.markdown(f"⚠️ {trap_label(trap_signal)}")
                if liq_health["level"] == "DEAD":
                    st.error("☠ liquidity dead – manual promote disabled")
                elif liq_health["level"] == "CRITICAL":
                    st.error("🔴 liquidity critical – avoid entry")
                elif liq_health["level"] == "WEAK":
                    st.warning("⚠ liquidity weak – size reduced")

                if anti_rug["level"] == "CRITICAL":
                    st.error("early rug risk – exit")
                elif anti_rug["level"] == "WARNING":
                    st.warning("rug risk building – reduce/watch")
                if trap.get("trap_flags"):
                    for fl in trap["trap_flags"][:3]:
                        st.caption(f"– {fl}")
                if fresh == "VERY_NEW":
                    st.markdown("🆕 fresh LP")
                elif fresh == "NEW":
                    st.markdown("🌱 new token")
                if whale:
                    st.markdown("🐋 whale accumulation")
                if smart_money:
                    st.markdown("🐋 smart money entry")
                if dev:
                    st.markdown("⚠ dev risk")
                st.write(f"Window: {r.get('source_window') or '—'}")
                st.write(f"Preset tags: {r.get('source_preset') or '—'}")
                st.write(f"Exit action: {action_plan['action_label']}")
                if level == "EXIT":
                    st.warning(f"EXIT ({exit_signal['exit_score']})")
                elif level == "EARLY":
                    st.caption(f"EARLY ({exit_signal['exit_score']})")
                if st.button("Promote → Portfolio", key=f"prom_{idx}_{hkey(base_addr, chain)}", use_container_width=True):
                    adj_size = float(size_info.get("usd_size_adj", size_info.get("usd_size", 0.0)) or 0.0)
                    liq_level = liq_health.get("level", "UNKNOWN")
                    if not best:
                        st.error("No live pool found.")
                    else:
                        if level == "EXIT" and persistence.get("persistent_exit"):
                            st.warning("Persistent exit signal – manual promote is risky.")
                        elif level == "EARLY" and persistence.get("persistent_early"):
                            st.warning("Early warning – use caution / smaller size.")

                        if corr.get("status", "ALLOW") == "BLOCK":
                            st.warning("High correlation risk – size will be reduced")
                        elif corr.get("status", "ALLOW") == "ALLOW_SMALL_ONLY":
                            st.warning("Correlation elevated – use smaller size")

                        if timing["timing"] == "SKIP":
                            st.warning("Bad entry timing – manual promote only.")

                        if anti_rug["level"] == "CRITICAL":
                            st.error("Early rug risk – entry blocked.")
                            return
                        elif anti_rug["level"] == "WARNING":
                            st.warning("Rug risk building – manual entry risky.")

                        if liq_level in ("DEAD", "CRITICAL"):
                            st.error("Liquidity too low for entry")
                            return
                        else:
                            if adj_size < 5:
                                st.warning("Position too small after risk adjustment.")
                            swap_url = build_swap_url(chain, base_addr)
                            res = log_to_portfolio(best, s_live, decision, tags, swap_url)
                            if res == "OK":
                                archive_monitoring(chain, base_addr, reason="manual: promoted", last_score=s_live, last_decision=decision)
                                st.success("Promoted.")
                                request_rerun()
                            else:
                                st.info("Already in portfolio.")
                if st.button("Archive (manual)", key=f"drop_{idx}_{hkey(base_addr, chain)}", use_container_width=True):
                    archive_monitoring(chain, base_addr, reason="manual", last_score=s_live, last_decision=decision)
                    st.success("Archived.")
                    request_rerun()
            with st.expander("Dynamics / sparklines", expanded=False):
                if not hist:
                    st.info("No snapshots yet.")
                else:
                    dfh = pd.DataFrame(hist).copy()
                    dfh["ts_utc"] = pd.to_datetime(dfh.get("ts_utc", ""), errors="coerce")
                    dfh = dfh.sort_values("ts_utc")
                    for col in ["price_usd","score_live","liq_usd","vol24_usd","vol5_usd"]:
                        dfh[col] = pd.to_numeric(dfh.get(col, 0), errors="coerce")
                    dfh = dfh.set_index("ts_utc")
                    s1, s2, s3 = st.columns(3)
                    with s1: st.line_chart(dfh[["price_usd"]], height=120, use_container_width=True)
                    with s2: st.line_chart(dfh[["score_live"]], height=120, use_container_width=True)
                    with s3: st.line_chart(dfh[["liq_usd"]], height=120, use_container_width=True)
                    s4, s5 = st.columns(2)
                    with s4: st.line_chart(dfh[["vol24_usd"]], height=120, use_container_width=True)
                    with s5: st.line_chart(dfh[["vol5_usd"]], height=120, use_container_width=True)

    render_items("Signals", signals)
    st.markdown("---")
    render_items("Watchlist", watchlist)

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
            st.code(base_addr, language="text")
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

        best = best_pair_for_token(chain, base_addr)
        if not best:
            continue

        entry = parse_float(r.get("entry_price_usd"), 0.0)
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
        st.sidebar.error("CORE SAFETY BROKEN")
        st.sidebar.write("– cannot read source")
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
        st.sidebar.error("CORE SAFETY BROKEN")
        for i in issues[:8]:
            st.sidebar.write(f"– {i}")
    else:
        st.sidebar.success("Core safety OK")


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
            best_pair_for_token(
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
        entry_price_str = r.get("entry_price_usd") or "0"

        try:
            entry_price = float(entry_price_str)
        except Exception:
            entry_price = 0.0

        best = best_pair_for_token(chain, base_addr) if (chain and base_addr) else None
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
            st.code(base_addr, language="text")
            if best and best.get("url"):
                link_button("DexScreener", best.get("url", ""), use_container_width=True, key=f"p_ds_{idx}_{hkey(base_addr)}")
            elif r.get("dexscreener_url"):
                link_button("DexScreener", r["dexscreener_url"], use_container_width=True, key=f"p_ds_{idx}_{hkey(base_addr)}")
            swap_url = r.get("swap_url") or build_swap_url(chain, base_addr)
            if swap_url:
                link_button("Swap", swap_url, use_container_width=True, key=f"p_sw_{idx}_{hkey(base_addr, chain)}")

            st.caption("Decision")
            st.markdown(action_badge(decision), unsafe_allow_html=True)

        with c2:
            st.write(f"Score: {s_live:.2f}" if best else f"Score: {r.get('score','n/a')}")
            st.write(f"Entry: ${entry_price_str}")
            st.write(f"Now: ${cur_price:.8f}" if cur_price else "Now: n/a")
            st.write(f"PnL: {pnl:+.2f}%" if entry_price and cur_price else "PnL: n/a")
            st.write(f"Liq: {fmt_usd(liq)}" if best else "Liq: n/a")
            st.write(f"Vol24: {fmt_usd(vol24)}" if best else "Vol24: n/a")

        with c3:
            st.write(f"Vol5: {fmt_usd(vol5)}" if best else "Vol5: n/a")
            st.write(f"Δ m5: {fmt_pct(pc5)}")
            st.write(f"Δ h1: {fmt_pct(pc1h)}")
            st.write(f"Reco: {reco}")
            st.write(f"Strength: {strength_cls} ({strength_score:.1f})")
            if strength_cls == "WEAK":
                st.caption("⚠ weak position – candidate for rotation")
                if base_sym in weak_rotation_symbols:
                    st.error(f"Sell {int(exit_pct * 100)}% – rotation/exit signal")
                    if best_strong:
                        rotate_to = best_strong.get("base_symbol") or best_strong.get("symbol") or "?"
                        st.caption(f"→ rotate into {rotate_to}")
            st.write(f"Trap: {trap.get('trap_level', 'SAFE')} ({trap.get('trap_score', 0)})")
            st.write(f"Exit signal: {level}")
            st.write(f"Recommended action: {action_plan['action_label']}")
            st.write(f"Liquidity health: {liq_health['level']}")
            st.write(f"Anti-rug: {anti_rug['level']} ({anti_rug['score']})")
            st.caption("Why:")
            for rr in explain_reco(reco, liq_health, anti_rug, exit_signal):
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
def main():
    if st.session_state.get("_rerun_flag"):
        st.session_state["_rerun_flag"] = False
        st.rerun()

    ensure_storage()
    portfolio_rows = load_csv(PORTFOLIO_CSV, PORTFOLIO_FIELDS)
    monitoring_rows = load_csv(MONITORING_CSV, MON_FIELDS)
    monitoring_history_rows = load_csv(MON_HISTORY_CSV, HIST_FIELDS)
    st.session_state["_preload_counts"] = {
        "portfolio": len(portfolio_rows),
        "monitoring": len(monitoring_rows),
        "monitoring_history": len(monitoring_history_rows),
    }
    st.sidebar.info(f"📥 Loaded portfolio: {len(portfolio_rows)} rows")

    scanner_seeds_raw = str(DEFAULT_SEEDS)
    scanner_max_items = 100
    use_birdeye_trending = True
    birdeye_limit = 50
    ui_autorefresh_sec = 60

    st.sidebar.markdown("## 🔍 Supabase debug")
    
    st.sidebar.write("USE_SUPABASE:", USE_SUPABASE)
    st.sidebar.write("URL present:", bool(SUPABASE_URL))
    st.sidebar.write("KEY present:", bool(SUPABASE_SERVICE_ROLE_KEY))
    
    if USE_SUPABASE:
        test_keys = ["portfolio.csv", "monitoring.csv", "monitoring_history.csv"]
        for k in test_keys:
            content = sb_get_storage(k)
            if content:
                st.sidebar.success(f"{k}: FOUND ({len(content)} chars)")
            else:
                st.sidebar.error(f"{k}: EMPTY / NOT FOUND")
    if _sb_ok():
        if sb_get_storage("portfolio.csv") is None and sb_get_storage("monitoring.csv") is None:
            st.sidebar.warning("⚠️ Supabase connected but currently empty/unavailable")
        else:
            st.sidebar.success("Supabase data доступні")

    with st.sidebar:
        st.markdown("### DEX Scout")
        st.caption(f"Version: v{VERSION}")

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
        ui_autorefresh_sec = st.slider("UI auto-refresh (sec)", 0, 300, 20, step=10)
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
        hide_red = st.checkbox("Hide red signals", value=True)
        portfolio_value_usd = st.number_input(
            "Portfolio size (USD)",
            min_value=100.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
        )

        st.divider()
        if st.button("Run scanner now", use_container_width=True):
            res = run_scanner_now(scanner_seeds_raw, max_items=int(scanner_max_items), use_birdeye_trending=bool(use_birdeye_trending), birdeye_limit=int(birdeye_limit))
            st.session_state["_scan_feedback"] = f"Scanner ran: {res['window']} / {res['chain']} / {res['stats']}"
            request_rerun()

        if st.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            request_rerun()

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

        core_safety_self_check()

    # rotating scanner – one slot every 5 minutes, runs at most once per slot
    scan_result = maybe_run_rotating_scanner(scanner_seeds_raw, max_items=int(scanner_max_items), use_birdeye_trending=bool(use_birdeye_trending), birdeye_limit=int(birdeye_limit))
    if scan_result.get("ran"):
        st.toast(f"Scanner: {scan_result['window']} / {scan_result['chain']} / {scan_result['stats']}")

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
        hide_red=hide_red,
        portfolio_value_usd=portfolio_value_usd,
    )

    if page == "Monitoring":
        page_monitoring(auto_cfg)
    elif page == "Archive":
        page_archive()
    else:
        page_portfolio()


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

if __name__ == "__main__":
    main()
