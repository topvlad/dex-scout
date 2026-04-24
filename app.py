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
import math
import hashlib
import inspect
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter
from contextlib import contextmanager

import requests
import streamlit as st
import pandas as pd
import config as app_config


st.set_page_config(page_title="DEX Scout", layout="wide")


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


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
PULSE_SPARKLINE_LOG_DEBOUNCE_SEC = 300
PULSE_SPARKLINE_LOG_LAST_TS: Dict[str, float] = {}

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
PORTFOLIO_RECO_LOG_CSV = os.path.join(DATA_DIR, "portfolio_reco_log.csv")
SIGNAL_JOURNAL_CSV = os.path.join(DATA_DIR, "signal_journal.csv")
HEALTH_OVERRIDE_JOURNAL_CSV = os.path.join(DATA_DIR, "health_override_journal.csv")
PORTFOLIO_ACTION_JOURNAL_CSV = os.path.join(DATA_DIR, "portfolio_action_journal.csv")
TG_STATE_FILE = os.path.join(DATA_DIR, "tg_state.json")
TG_STATE_KEY = "tg_state.json"
SUPPRESSED_KEY = "suppressed_tokens.json"
DEFAULT_ALERT_MODE = "normal"
ALERT_MODE_REGISTRY = {"quiet", "normal", "aggressive"}
DIGEST_SOURCE_MODE_DEFAULT = "ui_truth"
DIGEST_SOURCE_MODES = {"ui_truth", "backend_candidates"}
DIGEST_BACKEND_REUSE_TTL_SEC = max(300, int(os.getenv("DIGEST_BACKEND_REUSE_TTL_SEC", "7200")))
DIGEST_UI_HEARTBEAT_HOURS = max(1, _env_int("DIGEST_UI_HEARTBEAT_HOURS", 12))
DIGEST_DISCOVERY_HEARTBEAT_HOURS = max(1, _env_int("DIGEST_DISCOVERY_HEARTBEAT_HOURS", 12))
OUTCOME_HORIZONS_MINUTES: Dict[str, int] = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "24h": 1440,
}

PORTFOLIO_RECO_LOG_FIELDS = [
    "event_id",
    "ts_utc",
    "token",
    "chain",
    "health_label",
    "health_bucket",
    "final_action",
    "final_reason",
    "liq_usd",
    "vol24_usd",
    "snapshot_age_min",
    "outcome_status",
    "outcome_ts_utc",
    "outcome_note",
]

JOURNAL_BASE_FIELDS = [
    "event_id",
    "journal_type",
    "source_event_id",
    "ts_utc",
    "token_key",
    "chain",
    "token_addr",
    "source",
    "event_type",
    "alert_tier",
    "signal_bucket",
    "signal_action",
    "signal_horizon",
    "final_action",
    "final_reason",
    "health_label",
    "health_override_active",
    "health_override_action",
    "health_override_reason",
    "entry_score",
    "timing_label",
    "risk_level",
    "setup_context",
    "reference_price_usd",
    "emission_key",
    "event_key",
    "send_status",
    "send_note",
]

JOURNAL_OUTCOME_FIELDS = [
    "outcome_15m_status",
    "outcome_15m_ts_utc",
    "outcome_15m_return_pct",
    "outcome_15m_price_usd",
    "outcome_1h_status",
    "outcome_1h_ts_utc",
    "outcome_1h_return_pct",
    "outcome_1h_price_usd",
    "outcome_4h_status",
    "outcome_4h_ts_utc",
    "outcome_4h_return_pct",
    "outcome_4h_price_usd",
    "outcome_24h_status",
    "outcome_24h_ts_utc",
    "outcome_24h_return_pct",
    "outcome_24h_price_usd",
]

SIGNAL_JOURNAL_FIELDS = JOURNAL_BASE_FIELDS + JOURNAL_OUTCOME_FIELDS
HEALTH_OVERRIDE_JOURNAL_FIELDS = JOURNAL_BASE_FIELDS + JOURNAL_OUTCOME_FIELDS
PORTFOLIO_ACTION_JOURNAL_FIELDS = JOURNAL_BASE_FIELDS + JOURNAL_OUTCOME_FIELDS

PATTERN_KEY_SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pattern-trust-v1": {
        "required_components": ("setup_class", "health_context", "timing_state_context"),
        "optional_components": ("market_regime_bucket",),
    }
}
PATTERN_KEY_SCHEMA_VERSION = "pattern-trust-v1"
TRUST_MIN_SAMPLE_FLOOR = 5
TRUST_RECENT_WINDOW_DAYS = 3
TRUST_HISTORY_LOOKBACK_ROWS = 10_000
TRUST_SCORE_CAP = 0.75


def _pattern_component_normalize(value: Any, default: str = "na") -> str:
    token = str(value or "").strip().lower()
    if not token:
        token = default
    token = re.sub(r"[^a-z0-9_\\-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or default


def get_pattern_key_registry(schema_version: str = PATTERN_KEY_SCHEMA_VERSION) -> Dict[str, Any]:
    schema = PATTERN_KEY_SCHEMA_REGISTRY.get(schema_version)
    if not schema:
        return {"schema_version": schema_version, "required_components": tuple(), "optional_components": tuple()}
    return {
        "schema_version": schema_version,
        "required_components": tuple(schema.get("required_components") or tuple()),
        "optional_components": tuple(schema.get("optional_components") or tuple()),
    }


def build_pattern_key_components(row: Dict[str, Any]) -> Dict[str, str]:
    setup_class = (
        row.get("setup_class")
        or row.get("setup_context")
        or row.get("signal_bucket")
        or row.get("entry_status")
        or row.get("entry_action")
        or "unknown_setup"
    )
    health_context = row.get("health_label") or row.get("health_bucket") or "unknown_health"
    timing_state_context = (
        row.get("timing_label")
        or row.get("timing")
        or row.get("entry_state")
        or row.get("final_action")
        or "unknown_state"
    )
    market_regime_bucket = row.get("market_regime_bucket") or row.get("market_regime") or ""

    payload = {
        "setup_class": _pattern_component_normalize(setup_class, default="unknown_setup"),
        "health_context": _pattern_component_normalize(health_context, default="unknown_health"),
        "timing_state_context": _pattern_component_normalize(timing_state_context, default="unknown_state"),
    }
    regime_norm = _pattern_component_normalize(market_regime_bucket, default="")
    if regime_norm:
        payload["market_regime_bucket"] = regime_norm
    return payload


def build_pattern_key(row: Dict[str, Any], schema_version: str = PATTERN_KEY_SCHEMA_VERSION) -> str:
    schema = get_pattern_key_registry(schema_version)
    components = build_pattern_key_components(row)
    ordered_bits: List[str] = []
    for req_key in schema.get("required_components", tuple()):
        ordered_bits.append(f"{req_key}={components.get(req_key, 'na')}")
    for opt_key in schema.get("optional_components", tuple()):
        if components.get(opt_key):
            ordered_bits.append(f"{opt_key}={components.get(opt_key)}")
    return f"{schema_version}|{'|'.join(ordered_bits)}"


def _extract_resolved_outcome(row: Dict[str, Any]) -> Optional[Tuple[str, float, datetime]]:
    # Closed/resolved only: explicitly finalized horizons, never PENDING.
    for horizon in ("24h", "4h", "1h", "15m"):
        status = str(row.get(f"outcome_{horizon}_status") or "").strip().upper()
        if status in {"", "PENDING"}:
            continue
        ts = parse_ts(row.get(f"outcome_{horizon}_ts_utc")) or parse_ts(row.get("ts_utc"))
        if ts is None:
            ts = datetime.utcnow()
        ret = parse_float(row.get(f"outcome_{horizon}_return_pct"), 0.0)
        return status, ret, ts
    return None


def _outcome_score(status: str, return_pct: float) -> float:
    status = str(status or "").upper()
    if status == "UP":
        return max(0.1, min(return_pct / 25.0, 1.0))
    if status == "DOWN":
        return min(-0.1, max(return_pct / 25.0, -1.0))
    if status == "FLAT":
        return 0.0
    return 0.0


def build_pattern_trust_index(rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    if rows is None:
        rows = load_csv(SIGNAL_JOURNAL_CSV, SIGNAL_JOURNAL_FIELDS)[-TRUST_HISTORY_LOOKBACK_ROWS:]
    grouped: Dict[str, Dict[str, Any]] = {}
    now_dt = datetime.utcnow()
    for row in rows:
        resolved = _extract_resolved_outcome(row)
        if resolved is None:
            continue
        status, ret_pct, resolved_ts = resolved
        pattern_key = build_pattern_key(row)
        bucket = grouped.setdefault(pattern_key, {"samples": 0, "sum_score": 0.0, "recent_samples": 0})
        bucket["samples"] += 1
        bucket["sum_score"] += _outcome_score(status, ret_pct)
        age_days = max(0.0, (now_dt - resolved_ts).total_seconds() / 86400.0)
        if age_days <= TRUST_RECENT_WINDOW_DAYS:
            bucket["recent_samples"] += 1
    return grouped


@st.cache_data(ttl=120, show_spinner=False)
def load_pattern_trust_index_cached() -> Dict[str, Dict[str, Any]]:
    return build_pattern_trust_index()


def compute_pattern_trust(row: Dict[str, Any], trust_index: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    pattern_key = build_pattern_key(row)
    if trust_index is None:
        trust_index = load_pattern_trust_index_cached()
    bucket = dict(trust_index.get(pattern_key) or {})
    sample_size = int(bucket.get("samples", 0))
    recent_samples = int(bucket.get("recent_samples", 0))
    mean_score = (parse_float(bucket.get("sum_score"), 0.0) / sample_size) if sample_size > 0 else 0.0
    bounded_score = max(-TRUST_SCORE_CAP, min(TRUST_SCORE_CAP, mean_score))

    confidence = 0.15
    reasons: List[str] = []
    if sample_size < TRUST_MIN_SAMPLE_FLOOR:
        bounded_score = 0.0
        reasons.append(f"sample floor: {sample_size}/{TRUST_MIN_SAMPLE_FLOOR}")
    else:
        confidence = min(1.0, sample_size / 20.0)
    if sample_size > 0 and recent_samples >= sample_size:
        confidence *= 0.6
        reasons.append("recent-only sample cluster")
    if sample_size > 0 and abs(bounded_score) >= 0.6:
        confidence *= 0.85
    confidence = max(0.05, min(confidence, 1.0))
    if not reasons:
        reasons.append("resolved journal outcomes")

    return {
        "pattern_key_schema_version": PATTERN_KEY_SCHEMA_VERSION,
        "pattern_key": pattern_key,
        "trust_score": round(bounded_score, 4),
        "trust_confidence": round(confidence, 4),
        "trust_sample_size": sample_size,
        "trust_reason": "; ".join(reasons),
        "advisory_only": True,
    }


def advisory_trust_bias(trust_payload: Dict[str, Any]) -> float:
    score = parse_float(trust_payload.get("trust_score"), 0.0)
    confidence = parse_float(trust_payload.get("trust_confidence"), 0.0)
    sample_size = int(parse_float(trust_payload.get("trust_sample_size"), 0.0))
    if sample_size < TRUST_MIN_SAMPLE_FLOOR:
        return 0.0
    return max(-20.0, min(20.0, score * confidence * 25.0))


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

RUNTIME_STATE_TABLE = "runtime_state"
LOCKS_TABLE = "locks"
TG_STATE_TABLE = "tg_state"
JOB_HEARTBEATS_TABLE = "job_heartbeats"
JOB_RUNS_TABLE = "job_runs"
RUNTIME_REQUIRED_TABLES: Tuple[str, ...] = (
    RUNTIME_STATE_TABLE,
    LOCKS_TABLE,
    TG_STATE_TABLE,
    JOB_HEARTBEATS_TABLE,
)

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


def _runtime_status(ok: bool, code: str, message: str = "", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": bool(ok),
        "code": str(code or "unknown"),
        "message": str(message or ""),
    }
    if extra:
        payload.update(extra)
    return payload


def _sb_select_rows(
    table: str,
    filters: Optional[Dict[str, str]] = None,
    select: str = "*",
    limit: int = 1,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    if not _sb_ok():
        return [], _runtime_status(True, "supabase_disabled")
    try:
        params: Dict[str, str] = {"select": select, "limit": str(max(1, int(limit)))}
        for k, v in (filters or {}).items():
            params[str(k)] = str(v)
        resp = requests.get(_sb_table_url(table), headers=_sb_headers(), params=params, timeout=12)
        if resp.status_code >= 400:
            body = (resp.text or "").replace("\n", " ").strip()[:240]
            status = _runtime_status(False, "supabase_read_failed", f"{table} read failed", table=table, http_status=resp.status_code, detail=body)
            debug_log(f"runtime_select_failed table={table} status={resp.status_code} detail={body}")
            return None, status
        data = resp.json() or []
        if not isinstance(data, list):
            data = []
        return data, _runtime_status(True, "ok", table=table, rows=len(data))
    except Exception as e:
        status = _runtime_status(False, "supabase_read_exception", f"{table} read exception", table=table, detail=f"{type(e).__name__}:{e}")
        debug_log(f"runtime_select_exception table={table} err={type(e).__name__}:{e}")
        return None, status


def _sb_upsert_row(table: str, payload: Dict[str, Any], on_conflict: str) -> Dict[str, Any]:
    if not _sb_ok():
        return _runtime_status(True, "supabase_disabled")
    try:
        headers = _sb_headers()
        headers["Prefer"] = "resolution=merge-duplicates"
        params = {"on_conflict": on_conflict}
        resp = requests.post(_sb_table_url(table), headers=headers, params=params, json=payload, timeout=12)
        if resp.status_code >= 400:
            body = (resp.text or "").replace("\n", " ").strip()[:240]
            status = _runtime_status(False, "supabase_upsert_failed", f"{table} upsert failed", table=table, http_status=resp.status_code, detail=body)
            debug_log(f"runtime_upsert_failed table={table} status={resp.status_code} detail={body}")
            return status
        return _runtime_status(True, "ok", table=table)
    except Exception as e:
        status = _runtime_status(False, "supabase_upsert_exception", f"{table} upsert exception", table=table, detail=f"{type(e).__name__}:{e}")
        debug_log(f"runtime_upsert_exception table={table} err={type(e).__name__}:{e}")
        return status

def sb_get_storage(key: str) -> Optional[str]:
    """
    Returns content for key from public.app_storage, or None if not found/readable.
    Reliability refactor: HTTP 4xx/5xx responses degrade to None instead of bubbling up.
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
        if r.status_code >= 400:
            debug_log(f"supabase_read_http_error key={key} status={r.status_code}")
            return None
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
                    "gem_transition_score",
                    "gem_transition_sufficient",
                    "gem_transition_reason",
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

    # lightweight sidecar for portfolio recommendation outcomes
    if not os.path.exists(PORTFOLIO_RECO_LOG_CSV):
        with open(PORTFOLIO_RECO_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PORTFOLIO_RECO_LOG_FIELDS)
            w.writeheader()

    if not os.path.exists(SIGNAL_JOURNAL_CSV):
        with open(SIGNAL_JOURNAL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SIGNAL_JOURNAL_FIELDS)
            w.writeheader()

    if not os.path.exists(HEALTH_OVERRIDE_JOURNAL_CSV):
        with open(HEALTH_OVERRIDE_JOURNAL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEALTH_OVERRIDE_JOURNAL_FIELDS)
            w.writeheader()

    if not os.path.exists(PORTFOLIO_ACTION_JOURNAL_CSV):
        with open(PORTFOLIO_ACTION_JOURNAL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PORTFOLIO_ACTION_JOURNAL_FIELDS)
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
    """
    Load CSV rows from Supabase storage (if configured) with local fallback.

    Args:
        path: CSV path/key used by storage backends.
        fields: Optional schema projection for returned rows.
            - If provided, every returned row is normalized to exactly these keys
              and in this exact order.
            - Missing columns are filled with "".
            - Extra CSV columns are dropped in the returned payload.
            - If omitted, rows are returned as-is from csv.DictReader.
    """
    ensure_storage()
    key = storage_key_for_path(path)

    def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not fields:
            return rows
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            normalized.append({name: row.get(name, "") for name in fields})
        return normalized

    if _sb_ok():
        content = sb_get_storage(key)
        if content:
            try:
                st.session_state[f"_storage_source_{key}"] = "supabase"
                return _normalize_rows(_csv_from_string(content))
            except Exception:
                debug_log(f"corrupt_supabase_csv key={key}")

    fallback = local_fallback_path(path)
    if os.path.exists(fallback):
        try:
            with open(fallback, "r", newline="", encoding="utf-8") as f:
                debug_log(f"using_local_fallback key={key}")
                st.session_state[f"_storage_source_{key}"] = "local_fallback"
                return _normalize_rows(list(csv.DictReader(f)))
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
            # Keep read-back as diagnostics, but success criterion is exact round-trip.
            check = sb_get_storage(key)
            if check == content:
                debug_log(f"supabase_store_verified key={key} mode=roundtrip")
                st.session_state["_save_badge"] = "💾 saved (supabase round-trip verified)"
            else:
                check_len = len(check) if isinstance(check, str) else 0
                debug_log(
                    f"supabase_store_mismatch key={key} expected_len={len(content)} got_len={check_len}"
                )
                st.session_state["_save_badge"] = "⚠️ saved local, supabase write ok (round-trip mismatch)"
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


def upsert_csv_row(path: str, row: Dict[str, Any], fieldnames: List[str], id_field: str = "event_id") -> None:
    event_id = str(row.get(id_field) or "").strip()
    if not event_id:
        append_csv(path, row, fieldnames)
        return
    rows = load_csv(path, fieldnames)
    replaced = False
    for idx, existing in enumerate(rows):
        if str(existing.get(id_field) or "").strip() == event_id:
            rows[idx] = {k: row.get(k, "") for k in fieldnames}
            replaced = True
            break
    if not replaced:
        rows.append({k: row.get(k, "") for k in fieldnames})
    save_csv(path, rows, fieldnames)


def get_csv_row_by_id(path: str, fieldnames: List[str], event_id: str, id_field: str = "event_id") -> Optional[Dict[str, Any]]:
    key = str(event_id or "").strip()
    if not key:
        return None
    for row in load_csv(path, fieldnames):
        if str(row.get(id_field) or "").strip() == key:
            return dict(row)
    return None


def _journal_outcome_init() -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for horizon in ("15m", "1h", "4h", "24h"):
        payload[f"outcome_{horizon}_status"] = "PENDING"
        payload[f"outcome_{horizon}_ts_utc"] = ""
        payload[f"outcome_{horizon}_return_pct"] = ""
        payload[f"outcome_{horizon}_price_usd"] = ""
    return payload



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
PULSE_MIN_LIQ_USD = float(max(0, _env_int("PULSE_MIN_LIQ_USD", 8000)))
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
    if action in ("WAIT FOR PULLBACK",):
        return "SMALL" if risk == "LOW" else "WATCH ONLY"
    if action in ("TRACK ONLY", "WATCH CLOSELY", "WAIT"):
        return "WATCH ONLY"
    if action in ("REDUCE", "TAKE PROFIT", "EXIT", "NO ENTRY"):
        return "SKIP"
    return "WATCH ONLY"


def _float_from_any(data: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in data and str(data.get(k, "")).strip() != "":
            return parse_float(data.get(k), default)
    return default


def _series_is_flat(values: List[float], rel_eps: float = 0.01, abs_eps: float = 1e-12) -> bool:
    if len(values) < 3:
        return False
    tail = values[-6:]
    vmax = max(tail)
    vmin = min(tail)
    span = vmax - vmin
    baseline = max(abs(vmax), abs(vmin), abs_eps)
    return span <= max(abs_eps, baseline * rel_eps)


DEFAULT_HEALTH_THRESHOLDS: Dict[str, Any] = {
    "liq_low_usd": 1_000.0,
    "vol24_low_usd": 100.0,
    "stale_minutes": 240.0,
    "min_history_points": 4,
    "flat_eps": {
        "price": 0.003,
        "liq": 0.01,
        "vol24": 0.05,
        "vol5": 0.05,
    },
}


def get_health_thresholds() -> Dict[str, Any]:
    cfg = dict(getattr(app_config, "HEALTH_THRESHOLDS", {}) or {})
    base = dict(DEFAULT_HEALTH_THRESHOLDS)
    base_flat = dict(DEFAULT_HEALTH_THRESHOLDS.get("flat_eps", {}) or {})
    base_flat.update(dict(cfg.get("flat_eps", {}) or {}))
    base.update(cfg)
    base["flat_eps"] = base_flat
    return base


def health_thresholds_debug_line() -> str:
    t = get_health_thresholds()
    flat = dict(t.get("flat_eps", {}) or {})
    return (
        "Health thresholds: "
        f"liq_low={parse_float(t.get('liq_low_usd'), 1000.0):.2f}, "
        f"vol24_low={parse_float(t.get('vol24_low_usd'), 100.0):.2f}, "
        f"stale_min={parse_float(t.get('stale_minutes'), 240.0):.1f}, "
        f"min_hist={int(parse_float(t.get('min_history_points'), 4.0))}, "
        f"flat_eps(price={parse_float(flat.get('price'), 0.003):.4f}, "
        f"liq={parse_float(flat.get('liq'), 0.01):.4f}, "
        f"vol24={parse_float(flat.get('vol24'), 0.05):.4f}, "
        f"vol5={parse_float(flat.get('vol5'), 0.05):.4f})"
    )


def is_debug_mode_enabled() -> bool:
    return bool(st.session_state.get("debug_mode", False) or st.session_state.get("show_debug_raw", False))


def detect_position_health(row: Dict[str, Any], hist: List[Dict[str, Any]]) -> Dict[str, Any]:
    health_cfg = get_health_thresholds()
    flat_eps = dict(health_cfg.get("flat_eps", {}) or {})
    grace_age_min = 45.0
    grace_hist_points = int(parse_float(health_cfg.get("min_history_points"), 4.0))
    liq_usd = _float_from_any(row, ["liq_usd", "liquidity", "liquidity_usd"], 0.0)
    vol5 = _float_from_any(row, ["vol5", "vol5_usd", "volume_m5", "volume5"], 0.0)
    vol24 = _float_from_any(row, ["vol24", "vol24_usd", "volume_h24", "volume24"], 0.0)
    price = _float_from_any(row, ["priceUsd", "price_usd", "price"], 0.0)

    if liq_usd <= 0:
        liq_usd = _float_from_any(row, ["liq"], 0.0)
    if vol24 <= 0:
        vol24 = _float_from_any(row, ["volume_h24_usd"], 0.0)

    explicit_dead_flags = [
        str(row.get("dead_flag") or "").strip().lower() in ("1", "true", "yes"),
        str(row.get("is_dead") or "").strip().lower() in ("1", "true", "yes"),
        str(row.get("rug_flag") or "").strip().lower() in ("1", "true", "yes"),
        str(row.get("rugged") or "").strip().lower() in ("1", "true", "yes"),
        str(row.get("liquidity_health") or "").strip().upper() in ("DEAD", "CRITICAL"),
        str(row.get("anti_rug") or "").strip().upper() == "CRITICAL",
    ]
    explicit_dead_rug = any(explicit_dead_flags)
    is_dead = explicit_dead_rug or (price <= 0 and liq_usd <= 0 and vol24 <= 0)

    low_liq_threshold = parse_float(health_cfg.get("liq_low_usd"), 1000.0)
    is_low_liq = liq_usd < low_liq_threshold

    ts_added = (
        parse_ts(row.get("ts_added"))
        or parse_ts(row.get("added_at"))
        or parse_ts(row.get("created_at"))
    )
    latest_ts = parse_ts(row.get("updated_at")) or parse_ts(row.get("ts_utc"))
    hist_times: List[datetime] = []
    for h in hist or []:
        hts = parse_ts(h.get("ts_utc")) or parse_ts(h.get("updated_at"))
        if hts:
            hist_times.append(hts)
    if hist_times:
        latest_hist_ts = max(hist_times)
        if latest_ts is None or latest_hist_ts > latest_ts:
            latest_ts = latest_hist_ts
    first_hist_ts = min(hist_times) if hist_times else None
    position_origin_ts = ts_added or first_hist_ts or latest_ts
    position_age_min = (
        (datetime.utcnow() - position_origin_ts).total_seconds() / 60.0
        if position_origin_ts
        else 10_000.0
    )
    hist_points = len(hist_times)
    insufficient_history = hist_points < grace_hist_points
    young_position = position_age_min < grace_age_min
    health_grace_applied = bool(young_position and insufficient_history and not explicit_dead_rug)
    age_min = (datetime.utcnow() - latest_ts).total_seconds() / 60.0 if latest_ts else 10_000.0
    is_stale = age_min >= parse_float(health_cfg.get("stale_minutes"), 240.0)
    if health_grace_applied:
        is_stale = False

    price_hist = [parse_float(h.get("price_usd", h.get("priceUsd", 0)), 0.0) for h in hist or []]
    liq_hist = [parse_float(h.get("liq_usd", h.get("liquidity", 0)), 0.0) for h in hist or []]
    vol24_hist = [parse_float(h.get("vol24", h.get("vol24_usd", 0)), 0.0) for h in hist or []]
    vol5_hist = [parse_float(h.get("vol5", h.get("vol5_usd", 0)), 0.0) for h in hist or []]

    low_vol24_threshold = parse_float(health_cfg.get("vol24_low_usd"), 100.0)
    near_zero = (vol24 < low_vol24_threshold and vol5 <= 0.0) or (liq_usd <= 0 and vol24 <= 0 and vol5 <= 0)
    flat_price = _series_is_flat(price_hist, rel_eps=parse_float(flat_eps.get("price"), 0.003)) if price_hist else False
    flat_liq = _series_is_flat(liq_hist, rel_eps=parse_float(flat_eps.get("liq"), 0.01)) if liq_hist else False
    flat_vol24 = _series_is_flat(vol24_hist, rel_eps=parse_float(flat_eps.get("vol24"), 0.05)) if vol24_hist else False
    flat_vol5 = _series_is_flat(vol5_hist, rel_eps=parse_float(flat_eps.get("vol5"), 0.05)) if vol5_hist else False
    all_flat_recent = all([flat_price or not price_hist, flat_liq or not liq_hist, flat_vol24 or not vol24_hist, flat_vol5 or not vol5_hist])
    is_cold = bool(near_zero or all_flat_recent)
    if health_grace_applied and is_cold:
        is_cold = False

    no_recent_flow = (liq_usd <= 0 or is_low_liq) and vol24 < 10 and vol5 <= 0
    is_untradeable = bool(is_dead or (is_stale and is_cold) or no_recent_flow)

    health_label = "OK"
    health_reason = "market activity looks tradable"
    if is_dead:
        health_label = "DEAD"
        health_reason = "dead/rug flags or no price-liquidity-volume activity"
    elif is_untradeable:
        health_label = "UNTRADEABLE"
        health_reason = "no liquidity and no recent flow"
    elif is_stale and is_cold:
        health_label = "STALE"
        health_reason = "snapshots stale and history cold"
    elif is_cold:
        health_label = "COLD"
        health_reason = "recent history is flat / inactive"
    elif is_low_liq:
        health_label = "LOW_LIQ"
        health_reason = "liquidity below safety threshold"
    elif is_stale:
        health_label = "STALE"
        health_reason = "latest snapshot is too old"

    return {
        "is_dead": is_dead,
        "explicit_dead_rug": explicit_dead_rug,
        "is_stale": is_stale,
        "is_cold": is_cold,
        "is_low_liq": is_low_liq,
        "is_untradeable": is_untradeable,
        "health_grace_applied": health_grace_applied,
        "health_label": health_label,
        "health_reason": health_reason,
        "liq_usd": liq_usd,
        "vol24": vol24,
        "vol5": vol5,
        "price": price,
        "snapshot_age_min": age_min,
        "position_age_min": position_age_min,
        "history_points": hist_points,
    }


def compute_unified_recommendation(row: Dict[str, Any], source: str, hist: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
    final_action = "TRACK ONLY"
    final_reason = "watch structure"
    health = detect_position_health(row, hist or [])
    trust_payload = compute_pattern_trust({**row, "health_label": health.get("health_label", row.get("health_label"))})
    health_override_active = False
    health_override_action = ""
    health_override_reason = ""

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

        grace_mode = bool(health.get("health_grace_applied"))
        has_explicit_dead_rug = bool(health.get("explicit_dead_rug"))
        health_label = str(health.get("health_label", "OK")).upper()
        if health.get("is_dead"):
            health_override_active = True
            if grace_mode and not has_explicit_dead_rug:
                health_override_action = "REDUCE"
                health_override_reason = "Grace mode: early weak data, reduce risk instead of full exit"
            else:
                health_override_action = "EXIT"
                health_override_reason = "Token appears dead / non-tradeable"
        elif health.get("is_untradeable"):
            health_override_active = True
            if grace_mode and not has_explicit_dead_rug:
                health_override_action = "WATCH CLOSELY"
                health_override_reason = "Grace mode: insufficient history, monitor before forcing exit"
            else:
                health_override_action = "EXIT"
                health_override_reason = str(health.get("health_reason") or "Position is not tradeable")
        elif health.get("is_stale") and health.get("is_cold"):
            health_override_active = True
            if grace_mode and not has_explicit_dead_rug:
                health_override_action = "WATCH CLOSELY"
                health_override_reason = "Grace mode: freshly added position with limited history"
            else:
                health_override_action = "EXIT"
                health_override_reason = "Position is stale and cold"
        elif health.get("is_low_liq") and health.get("is_cold"):
            health_override_active = True
            health_override_action = "REDUCE"
            health_override_reason = "Liquidity collapsed / weak exit conditions"

        if health_override_active:
            final_action = health_override_action
            final_reason = f"Health override ({health_label}): {health_override_reason}"
    else:
        weak = str(row.get("weak_reason") or "").strip().lower()
        score = parse_float(row.get("score", entry_score), 0.0)
        if score <= 0 or "gate_blocked" in weak or entry_action in ("AVOID", "INVALID"):
            final_action, final_reason = "NO ENTRY", "invalid/weak setup"
        elif entry_action in ("ENTRY_NOW", "READY"):
            final_action, final_reason = "ENTER NOW", "setup is ready with momentum"
        elif entry_action in ("WATCH_PULLBACK", "EARLY"):
            final_action, final_reason = "WAIT FOR PULLBACK", "pullback entry preferred"
        elif entry_action in ("TRACK", "WAIT", "WATCH"):
            final_action, final_reason = "TRACK ONLY", "monitor without entry confirmation"
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
        "health": health,
        "health_override_active": health_override_active,
        "health_override_action": health_override_action,
        "health_override_reason": health_override_reason,
        "trust_score": trust_payload.get("trust_score", 0.0),
        "trust_confidence": trust_payload.get("trust_confidence", 0.0),
        "trust_sample_size": trust_payload.get("trust_sample_size", 0),
        "trust_reason": trust_payload.get("trust_reason", ""),
        "trust_pattern_key": trust_payload.get("pattern_key", ""),
        "trust_pattern_key_schema_version": trust_payload.get("pattern_key_schema_version", PATTERN_KEY_SCHEMA_VERSION),
        "trust_advisory_only": True,
    }
    unified["size_hint"] = suggested_position_size(row, unified)

    if liquidity_health in ("DEAD", "CRITICAL") and not health_override_active:
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
    # Single source of truth: sanitized setup payload only.
    # Do not reintroduce raw `> 0` checks here.
    prepared = build_setup_render_inputs(row)
    return bool(prepared.get("has_actionable_levels"))


SETUP_LEVEL_ORDER: List[Tuple[str, str]] = [
    ("entry_now", "entry now"),
    ("pullback_1", "pullback 1"),
    ("pullback_2", "pullback 2"),
    ("invalidation", "invalidation"),
    ("tp1", "tp1"),
    ("tp2", "tp2"),
]
MIN_ACTIONABLE_LEVEL = 1e-12


def _is_actionable_level(value: float) -> bool:
    v = parse_float(value, 0.0)
    return bool(v > MIN_ACTIONABLE_LEVEL and round(v, 12) > 0.0)


def _round_level(value: float) -> float:
    if not _is_actionable_level(value):
        return 0.0
    return round(parse_float(value, 0.0), 12)


def _build_entry_levels_for_action(action: str, price: float) -> Dict[str, float]:
    levels = {k: 0.0 for k, _ in SETUP_LEVEL_ORDER}
    if not _is_actionable_level(price):
        return levels

    action_u = str(action or "").upper()
    if action_u == "ENTRY_NOW":
        levels["entry_now"] = _round_level(price)
        levels["pullback_1"] = _round_level(price * 0.97)
        levels["pullback_2"] = _round_level(price * 0.93)
        levels["invalidation"] = _round_level(price * 0.89)
        levels["tp1"] = _round_level(price * 1.12)
        levels["tp2"] = _round_level(price * 1.25)
    elif action_u == "WATCH_PULLBACK":
        levels["pullback_1"] = _round_level(price * 0.97)
        levels["pullback_2"] = _round_level(price * 0.93)
        levels["invalidation"] = _round_level(price * 0.89)
        levels["tp1"] = _round_level(price * 1.10)
        levels["tp2"] = _round_level(price * 1.20)
    elif action_u == "TRACK":
        levels["pullback_1"] = _round_level(price * 0.96)
        levels["pullback_2"] = _round_level(price * 0.92)
        levels["invalidation"] = _round_level(price * 0.88)
        levels["tp1"] = _round_level(price * 1.08)
        levels["tp2"] = _round_level(price * 1.16)
    return levels


def _levels_pass_setup_guard(action: str, levels: Dict[str, float]) -> bool:
    action_u = str(action or "").upper()
    entry_now = parse_float(levels.get("entry_now"), 0.0)
    pullback_1 = parse_float(levels.get("pullback_1"), 0.0)
    pullback_2 = parse_float(levels.get("pullback_2"), 0.0)
    invalidation = parse_float(levels.get("invalidation"), 0.0)
    tp1 = parse_float(levels.get("tp1"), 0.0)
    tp2 = parse_float(levels.get("tp2"), 0.0)

    if action_u == "ENTRY_NOW":
        if not _is_actionable_level(entry_now):
            return False
        anchor = entry_now
    elif action_u in ("WATCH_PULLBACK", "TRACK"):
        if not _is_actionable_level(pullback_1):
            return False
        anchor = pullback_1
    else:
        return False

    if _is_actionable_level(pullback_1) and _is_actionable_level(pullback_2) and pullback_2 >= pullback_1:
        return False
    if _is_actionable_level(invalidation) and invalidation >= anchor:
        return False
    if _is_actionable_level(tp1) and tp1 <= anchor:
        return False
    if _is_actionable_level(tp2) and _is_actionable_level(tp1) and tp2 <= tp1:
        return False
    return True


def build_setup_render_inputs(row: Dict[str, Any], action_override: str = "") -> Dict[str, Any]:
    normalized_levels = {k: _round_level(parse_float(row.get(k), 0.0)) for k, _ in SETUP_LEVEL_ORDER}
    action = str(action_override or row.get("entry_action") or row.get("entry") or "").upper()
    if not _levels_pass_setup_guard(action, normalized_levels):
        normalized_levels = {k: 0.0 for k, _ in SETUP_LEVEL_ORDER}

    level_rows: List[Dict[str, float]] = []
    for key, label in SETUP_LEVEL_ORDER:
        value = normalized_levels[key]
        if _is_actionable_level(value):
            level_rows.append({"level": label, "value": value})

    if not level_rows:
        return {
            "level_rows": [],
            "has_actionable_levels": False,
            "levels_block": "setup: watch only",
        }

    lines = [f"{r['level']}: {r['value']}" for r in level_rows]
    return {
        "level_rows": level_rows,
        "has_actionable_levels": True,
        "levels_block": "\n".join(lines),
    }


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
    rows = load_monitoring()
    key = addr_key(chain, ca)
    changed = False
    for r in rows:
        if addr_key(r.get("chain", ""), r.get("base_addr", "")) != key:
            continue
        transition = transition_token_state(
            infer_lifecycle_state(r),
            "SUPPRESSED_MANUAL",
            {"reason": reason, "requested_status": "suppressed"},
        )
        r.update(transition.get("updated_fields", {}))
        if transition.get("valid"):
            r["active"] = "0"
            r["archived_reason"] = reason
            r["ts_archived"] = now_utc_str()
        changed = True
        break
    if changed:
        save_monitoring(rows)


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

    levels = _build_entry_levels_for_action(action, price)
    if not _levels_pass_setup_guard(action, levels):
        levels = {k: 0.0 for k, _ in SETUP_LEVEL_ORDER}
    setup_inputs = build_setup_render_inputs(levels, action_override=action)

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
        "entry_now": levels["entry_now"],
        "pullback_1": levels["pullback_1"],
        "pullback_2": levels["pullback_2"],
        "invalidation": levels["invalidation"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "setup_level_rows": setup_inputs["level_rows"],
        "setup_levels_block": setup_inputs["levels_block"],
        "setup_watch_only": "1" if not setup_inputs["has_actionable_levels"] else "0",
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
    def _parse_portfolio_entry_value(raw: Any) -> float:
        if raw is None:
            return 0.0
        if isinstance(raw, (int, float)):
            return parse_float(raw, 0.0)
        text = str(raw).strip().replace("$", "").replace(",", "")
        if not text:
            return 0.0
        return parse_float(text, 0.0)

    avg_entry_price = _parse_portfolio_entry_value(row.get("avg_entry_price"))
    if avg_entry_price > 0:
        return avg_entry_price

    # Controlled fallback for legacy data only when avg_entry_price is missing/invalid.
    for key in ("entry_price_usd", "entry_price", "price_at_add", "entry"):
        legacy_val = _parse_portfolio_entry_value(row.get(key))
        if legacy_val > 0:
            return legacy_val

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
    "avg_entry_price",
    "note",
    "active",
    "updated_at",
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
    "gem_transition_score",
    "gem_transition_sufficient",
    "gem_transition_reason",
    "weak_reason",
    "in_portfolio",
    "toxic_flags",
    "alert_sent",
    "status",
    "lifecycle_state",
    "behavior_state",
    "state_event",
    "transition_valid",
    "invalid_transition",
    "invalid_reason",
    "invalid_from_state",
    "invalid_attempted_to",
    "invalid_event",
    "updated_at",
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


LIFECYCLE_SCOUT = "SCOUT"
LIFECYCLE_MONITORING = "MONITORING"
LIFECYCLE_PORTFOLIO = "PORTFOLIO"
LIFECYCLE_ARCHIVED = "ARCHIVED"
LIFECYCLE_SUPPRESSED = "SUPPRESSED"
LIFECYCLE_REVISIT = "REVISIT"

LIFECYCLE_STATES = {
    LIFECYCLE_SCOUT,
    LIFECYCLE_MONITORING,
    LIFECYCLE_PORTFOLIO,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPPRESSED,
    LIFECYCLE_REVISIT,
}

BEHAVIOR_STATE_DEFAULT = "NEW"
BEHAVIOR_STATE_VOCAB = {
    "NEW", "BUILDING", "PULLBACK_READY", "ENTRY_READY", "TRACKING", "WEAKENING", "COLD", "DEAD",
}

STATE_EVENT_VOCAB = {
    "DISCOVERED",
    "PROMOTED_TO_MONITORING",
    "PROMOTED_TO_PORTFOLIO",
    "ARCHIVED_MANUAL",
    "ARCHIVED_RULE",
    "REACTIVATED",
    "SUPPRESSED_MANUAL",
    "READY_FOR_REVISIT",
}

ALLOWED_LIFECYCLE_TRANSITIONS: Dict[str, Set[str]] = {
    LIFECYCLE_SCOUT: {LIFECYCLE_MONITORING, LIFECYCLE_SUPPRESSED},
    LIFECYCLE_MONITORING: {LIFECYCLE_PORTFOLIO, LIFECYCLE_ARCHIVED, LIFECYCLE_SUPPRESSED, LIFECYCLE_REVISIT},
    LIFECYCLE_PORTFOLIO: {LIFECYCLE_ARCHIVED, LIFECYCLE_SUPPRESSED},
    LIFECYCLE_ARCHIVED: {LIFECYCLE_MONITORING, LIFECYCLE_REVISIT, LIFECYCLE_SUPPRESSED},
    LIFECYCLE_SUPPRESSED: {LIFECYCLE_MONITORING, LIFECYCLE_ARCHIVED, LIFECYCLE_REVISIT},
    LIFECYCLE_REVISIT: {LIFECYCLE_MONITORING, LIFECYCLE_ARCHIVED, LIFECYCLE_SUPPRESSED},
}

EVENT_TO_TARGET_LIFECYCLE: Dict[str, str] = {
    "DISCOVERED": LIFECYCLE_MONITORING,
    "PROMOTED_TO_MONITORING": LIFECYCLE_MONITORING,
    "PROMOTED_TO_PORTFOLIO": LIFECYCLE_PORTFOLIO,
    "ARCHIVED_MANUAL": LIFECYCLE_ARCHIVED,
    "ARCHIVED_RULE": LIFECYCLE_ARCHIVED,
    "REACTIVATED": LIFECYCLE_MONITORING,
    "SUPPRESSED_MANUAL": LIFECYCLE_SUPPRESSED,
    "READY_FOR_REVISIT": LIFECYCLE_REVISIT,
}


def infer_lifecycle_state(row: Dict[str, Any]) -> str:
    explicit = str(row.get("lifecycle_state") or "").strip().upper()
    if explicit in LIFECYCLE_STATES:
        return explicit
    status_u = str(row.get("status") or "").strip().upper()
    if status_u in LIFECYCLE_STATES:
        return status_u
    if str(row.get("portfolio_linked", "0")) == "1":
        return LIFECYCLE_PORTFOLIO
    if str(row.get("active", "1")).strip() == "1":
        return LIFECYCLE_MONITORING
    return LIFECYCLE_ARCHIVED


def derive_state_event(
    current_lifecycle_state: str,
    event: str,
    context_row: Optional[Dict[str, Any]] = None,
) -> str:
    event_u = str(event or "").strip().upper()
    if event_u in STATE_EVENT_VOCAB:
        return event_u
    ctx = context_row or {}
    requested_status = str(ctx.get("requested_status") or ctx.get("status") or "").strip().lower()
    reason = str(ctx.get("reason") or "").strip().lower()
    target = str(ctx.get("target_lifecycle_state") or "").strip().upper()
    if "suppress" in reason or requested_status == "suppressed":
        return "SUPPRESSED_MANUAL"
    if requested_status in {"active", "reactivate"}:
        return "REACTIVATED"
    if requested_status == "revisit" or target == LIFECYCLE_REVISIT:
        return "READY_FOR_REVISIT"
    if requested_status in {"portfolio", "promoted"} or "promoted" in reason or target == LIFECYCLE_PORTFOLIO:
        return "PROMOTED_TO_PORTFOLIO"
    if requested_status == "archived":
        if "manual" in reason:
            return "ARCHIVED_MANUAL"
        return "ARCHIVED_RULE"
    if current_lifecycle_state == LIFECYCLE_SCOUT:
        return "DISCOVERED"
    return "PROMOTED_TO_MONITORING"


def transition_token_state(
    current_lifecycle_state: str,
    event: str,
    context_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = context_row or {}
    current = str(current_lifecycle_state or "").strip().upper() or LIFECYCLE_SCOUT
    if current not in LIFECYCLE_STATES:
        current = infer_lifecycle_state(ctx)
    state_event = derive_state_event(current, event, ctx)
    target_override = str(ctx.get("target_lifecycle_state") or "").strip().upper()
    if target_override in LIFECYCLE_STATES:
        target = target_override
    else:
        target = EVENT_TO_TARGET_LIFECYCLE.get(state_event, current)
    allowed = target == current or target in ALLOWED_LIFECYCLE_TRANSITIONS.get(current, set())
    now_ts = now_utc_str()
    note = str(ctx.get("transition_note") or ctx.get("note") or "").strip()
    if not note:
        note = f"{current} --{state_event}--> {target}"
    if allowed:
        return {
            "next_lifecycle_state": target,
            "valid": True,
            "transition_note": note,
            "state_event": state_event,
            "updated_fields": {
                "lifecycle_state": target,
                "status": target,
                "state_event": state_event,
                "transition_valid": "1",
                "invalid_transition": "0",
                "invalid_reason": "",
                "invalid_from_state": "",
                "invalid_attempted_to": "",
                "invalid_event": "",
                "updated_at": now_ts,
                "note": note,
            },
        }
    invalid_reason = f"transition_not_allowed:{current}->{target} via {state_event}"
    return {
        "next_lifecycle_state": current,
        "valid": False,
        "transition_note": invalid_reason,
        "state_event": state_event,
        "updated_fields": {
            "transition_valid": "0",
            "invalid_transition": "1",
            "invalid_reason": invalid_reason,
            "invalid_from_state": current,
            "invalid_attempted_to": target,
            "invalid_event": state_event,
            "updated_at": now_ts,
        },
    }


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
        if not str(r.get("lifecycle_state") or "").strip():
            r["lifecycle_state"] = infer_lifecycle_state(r)
        if not str(r.get("status") or "").strip():
            r["status"] = r["lifecycle_state"]
        if not str(r.get("behavior_state") or "").strip():
            r["behavior_state"] = BEHAVIOR_STATE_DEFAULT
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


def classify_health_bucket(health: Dict[str, Any]) -> str:
    if bool(health.get("is_dead")):
        return "DEAD"
    if bool(health.get("is_untradeable")):
        return "UNTRADEABLE"
    if bool(health.get("is_low_liq")) and bool(health.get("is_cold")):
        return "LOW_LIQ+COLD"
    return "OTHER"


def log_portfolio_recommendation_snapshot(
    token: str,
    chain: str,
    unified: Dict[str, Any],
) -> None:
    health = dict(unified.get("health") or {})
    event_id = hashlib.md5(f"{chain}|{token}|{now_utc_str()}".encode("utf-8")).hexdigest()[:16]
    row = {
        "event_id": event_id,
        "ts_utc": now_utc_str(),
        "token": token,
        "chain": chain,
        "health_label": str(health.get("health_label", "OK")),
        "health_bucket": classify_health_bucket(health),
        "final_action": str(unified.get("final_action", "")),
        "final_reason": str(unified.get("final_reason", "")),
        "liq_usd": str(parse_float(health.get("liq_usd"), 0.0)),
        "vol24_usd": str(parse_float(health.get("vol24"), 0.0)),
        "snapshot_age_min": str(parse_float(health.get("snapshot_age_min"), 0.0)),
        # Sidecar-only placeholders for next stage (outcome tracking),
        # no changes in primary portfolio/monitoring storage format.
        "outcome_status": "PENDING",
        "outcome_ts_utc": "",
        "outcome_note": "",
    }
    append_csv(PORTFOLIO_RECO_LOG_CSV, row, PORTFOLIO_RECO_LOG_FIELDS)


def portfolio_reco_health_counts(period_hours: int = 24) -> Dict[str, int]:
    rows = load_csv(PORTFOLIO_RECO_LOG_CSV, PORTFOLIO_RECO_LOG_FIELDS)
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=max(1, int(period_hours)))
    counts = {"DEAD": 0, "UNTRADEABLE": 0, "LOW_LIQ+COLD": 0}
    for r in rows:
        ts = parse_ts(r.get("ts_utc"))
        if ts is None or ts < cutoff:
            continue
        bucket = str(r.get("health_bucket", "")).strip().upper()
        if bucket in counts:
            counts[bucket] += 1
    return counts


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

    hist = token_history_rows(chain, base_addr, limit=30)
    gem_data = compute_gem_transition_score(p, hist)
    gem_score = parse_float(gem_data.get("gem_transition_score"), GEM_TRANSITION_NEUTRAL)
    gem_sufficient = bool(gem_data.get("gem_transition_sufficient"))
    health_state = detect_position_health(p, hist)
    health_label = str(health_state.get("health_label", "OK")).upper()
    adjusted_priority = gem_transition_priority_bias(score, gem_score, health_label, gem_sufficient)

    for r in rows:
        if r.get("active") == "1" and addr_key(r.get("chain", ""), r.get("base_addr", "")) == key:
            # refresh lightweight fields
            r["last_score"] = str(score)
            r["priority_score"] = str(adjusted_priority)
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
            r["gem_transition_score"] = str(gem_score)
            r["gem_transition_sufficient"] = "1" if gem_sufficient else "0"
            r["gem_transition_reason"] = str(gem_data.get("gem_transition_reason") or "")
            r["timing_label"] = compute_timing({**p, **gem_data})
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
            "priority_score": str(adjusted_priority),
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
            "timing_label": compute_timing({**p, **gem_data}),
            "gem_transition_score": str(gem_score),
            "gem_transition_sufficient": "1" if gem_sufficient else "0",
            "gem_transition_reason": str(gem_data.get("gem_transition_reason") or ""),
            "weak_reason": str(p.get("weak_reason") or ""),
            "in_portfolio": "0",
            "toxic_flags": str(p.get("toxic_flags") or ""),
            "alert_sent": str(p.get("alert_sent") or "0"),
            "lifecycle_state": LIFECYCLE_MONITORING,
            "behavior_state": BEHAVIOR_STATE_DEFAULT,
            "state_event": "DISCOVERED",
            "transition_valid": "1",
            "invalid_transition": "0",
            "invalid_reason": "",
            "invalid_from_state": "",
            "invalid_attempted_to": "",
            "invalid_event": "",
            "updated_at": now_utc_str(),
            "status": LIFECYCLE_MONITORING,
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
            requested_status = "revisit" if revisit_days > 0 else "archived"
            transition = transition_token_state(
                infer_lifecycle_state(r),
                derive_state_event(
                    infer_lifecycle_state(r),
                    "",
                    {"requested_status": requested_status, "reason": reason},
                ),
                {"requested_status": requested_status, "reason": reason, "note": note},
            )
            if not transition.get("valid"):
                r.update(transition.get("updated_fields", {}))
                continue
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
            r.update(transition.get("updated_fields", {}))
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
    if status_u == "suppressed":
        suppress_token(chain=chain, ca=contract, reason=reason or "manual_suppress")
        return True
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
    if status_u == "revisit":
        return archive_monitoring(
            chain=chain,
            base_addr=contract,
            reason=reason or "revisit_ready",
            last_score=last_score,
            last_decision=last_decision,
            revisit_days=max(1, int(revisit_days or 1)),
            portfolio_linked=portfolio_linked,
            note=note,
        )
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
            transition = transition_token_state(
                infer_lifecycle_state(r),
                "REACTIVATED",
                {"requested_status": "active"},
            )
            if not transition.get("valid"):
                r.update(transition.get("updated_fields", {}))
                continue
            r["active"] = "1"
            r["ts_added"] = now_utc_str()
            r["ts_archived"] = ""
            r["archived_reason"] = ""
            r["last_score"] = ""
            r["last_decision"] = ""
            r.update(transition.get("updated_fields", {}))
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
    if not base_addr:
        return False
    key = addr_key(chain, base_addr)
    rows = load_monitoring()
    changed = False
    for r in rows:
        if addr_key(r.get("chain", ""), r.get("base_addr", "")) != key:
            continue
        transition = transition_token_state(
            infer_lifecycle_state(r),
            "PROMOTED_TO_PORTFOLIO",
            {
                "reason": reason,
                "requested_status": "portfolio",
                "target_lifecycle_state": LIFECYCLE_MONITORING,
                "transition_note": f"portfolio_linked:{reason}",
            },
        )
        if not transition.get("valid"):
            r.update(transition.get("updated_fields", {}))
            continue
        r["portfolio_linked"] = "1"
        r["in_portfolio"] = "1"
        r["active"] = "1"
        r["ts_archived"] = ""
        r["archived_reason"] = ""
        r.update(transition.get("updated_fields", {}))
        changed = True
        break
    if changed:
        save_monitoring(rows)
    return changed


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
        mark_monitoring_portfolio_linked(chain=chain, base_addr=base_addr, reason="duplicate_portfolio")
        return res
    if res != "OK":
        return res

    mark_monitoring_portfolio_linked(chain=chain, base_addr=base_addr, reason="promoted")
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


ROTATION_GAP_FLOOR = 25.0
ROTATION_CONFIDENCE_FLOOR = 0.55


def _rotation_score_confidence(rotation_gap: float, in_score: float, out_score: float) -> float:
    base = 0.5 + (rotation_gap / 120.0)
    quality_adj = max(0.0, (in_score - 120.0) / 400.0) - max(0.0, (out_score - 80.0) / 500.0)
    return max(0.0, min(0.99, base + quality_adj))


def portfolio_position_score(
    row: Dict[str, Any],
    best: Optional[Dict[str, Any]],
    hist: List[Dict[str, Any]],
    unified: Dict[str, Any],
) -> float:
    entry_price = get_portfolio_entry_price(row)
    strength = position_strength_score(best or {}, hist, entry_price)
    entry_component = parse_float(row.get("entry_score", row.get("score", 0)), 0.0) * 0.15
    trust_component = parse_float(unified.get("trust_score"), 0.0) * 40.0
    score = strength * 20.0 + entry_component + trust_component

    health_label = str((unified.get("health") or {}).get("health_label") or "").upper()
    final_action = str(unified.get("final_action") or "").upper()
    if health_label in {"DEAD", "UNTRADEABLE"}:
        score -= 200.0
    elif health_label in {"STALE", "COLD", "LOW_LIQ"}:
        score -= 30.0
    if final_action in {"EXIT", "REDUCE"}:
        score -= 30.0
    return float(round(score, 2))


def monitoring_candidate_score(row: Dict[str, Any], unified: Dict[str, Any]) -> float:
    entry_score = parse_float(row.get("entry_score", row.get("score", 0)), 0.0)
    trust_component = parse_float(unified.get("trust_score"), 0.0) * 40.0
    timing = str(unified.get("timing") or "").upper()
    action = str(unified.get("final_action") or "").upper()

    score = entry_score + trust_component
    if action == "ENTER NOW":
        score += 12.0
    elif action == "WAIT FOR PULLBACK":
        score += 5.0
    if timing in {"GOOD", "EARLY"}:
        score += 5.0
    return float(round(score, 2))


def build_rotation_advisory(
    active_portfolio_rows: List[Dict[str, Any]],
    monitoring_rows: List[Dict[str, Any]],
    gap_floor: float = ROTATION_GAP_FLOOR,
    confidence_floor: float = ROTATION_CONFIDENCE_FLOOR,
) -> Dict[str, Any]:
    portfolio_positions: List[Dict[str, Any]] = []
    for row in active_portfolio_rows:
        if str(row.get("active", "1")).strip() != "1":
            continue
        precomputed_position_score = row.get("portfolio_position_score")
        if precomputed_position_score not in (None, ""):
            unified_pre = {
                "final_action": str(row.get("final_action") or "HOLD"),
                "health": {"health_label": str(row.get("health_label") or "OK")},
            }
            portfolio_positions.append(
                {
                    "symbol": str(row.get("base_symbol") or row.get("symbol") or "").upper() or "?",
                    "chain": str(row.get("chain") or "").strip().lower(),
                    "row": row,
                    "unified": unified_pre,
                    "portfolio_position_score": parse_float(precomputed_position_score, 0.0),
                    "health_label": str(row.get("health_label") or "OK").upper(),
                }
            )
            continue
        chain = (row.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, (row.get("base_token_address") or "").strip())
        if not chain or not base_addr:
            continue
        best = best_pair_for_token_cached(chain, base_addr)
        hist = token_history_rows(chain, base_addr, limit=60)
        unified = compute_unified_recommendation({**row, "liquidity_health": str(liquidity_health(best).get("level", "OK"))}, source="portfolio", hist=hist)
        score = portfolio_position_score(row, best, hist, unified)
        portfolio_positions.append(
            {
                "symbol": str(row.get("base_symbol") or row.get("symbol") or "").upper() or "?",
                "chain": chain,
                "row": row,
                "unified": unified,
                "portfolio_position_score": score,
                "health_label": str((unified.get("health") or {}).get("health_label") or "OK").upper(),
            }
        )

    eligible_monitoring: List[Dict[str, Any]] = []
    for row in monitoring_rows:
        if str(row.get("active", "1")).strip() != "1":
            continue
        if is_in_portfolio_active(row, active_portfolio_rows):
            continue
        precomputed_candidate_score = row.get("monitoring_candidate_score")
        if precomputed_candidate_score not in (None, ""):
            final_action = str(row.get("final_action") or "ENTER NOW").upper()
            health_label = str(row.get("health_label") or "OK").upper()
            if final_action not in {"ENTER NOW", "WAIT FOR PULLBACK"} or health_label in {"DEAD", "UNTRADEABLE"}:
                continue
            eligible_monitoring.append(
                {
                    "symbol": str(row.get("base_symbol") or row.get("symbol") or "").upper() or "?",
                    "row": row,
                    "unified": {"final_action": final_action, "health": {"health_label": health_label}},
                    "monitoring_candidate_score": parse_float(precomputed_candidate_score, 0.0),
                }
            )
            continue
        unified = compute_unified_recommendation(row, source="monitoring")
        final_action = str(unified.get("final_action") or "").upper()
        health_label = str((unified.get("health") or {}).get("health_label") or "OK").upper()
        if final_action not in {"ENTER NOW", "WAIT FOR PULLBACK"}:
            continue
        if health_label in {"DEAD", "UNTRADEABLE"}:
            continue
        score = monitoring_candidate_score(row, unified)
        eligible_monitoring.append(
            {
                "symbol": str(row.get("base_symbol") or row.get("symbol") or "").upper() or "?",
                "row": row,
                "unified": unified,
                "monitoring_candidate_score": score,
            }
        )

    portfolio_sorted = sorted(portfolio_positions, key=lambda x: x.get("portfolio_position_score", 0.0))
    candidates_sorted = sorted(eligible_monitoring, key=lambda x: x.get("monitoring_candidate_score", 0.0), reverse=True)
    best_candidate = candidates_sorted[0] if candidates_sorted else None

    suggestions: List[Dict[str, Any]] = []
    if best_candidate:
        in_symbol = str(best_candidate.get("symbol") or "?")
        in_score = parse_float(best_candidate.get("monitoring_candidate_score"), 0.0)
        in_action = str((best_candidate.get("unified") or {}).get("final_action") or "").upper()
        for out in portfolio_sorted:
            out_symbol = str(out.get("symbol") or "?")
            out_score = parse_float(out.get("portfolio_position_score"), 0.0)
            health_label = str(out.get("health_label") or "OK").upper()
            out_action = str((out.get("unified") or {}).get("final_action") or "").upper()
            weak_or_dead = (out_score < 90.0) or (health_label in {"DEAD", "UNTRADEABLE"}) or (out_action in {"EXIT", "REDUCE"})
            if not weak_or_dead:
                continue
            gap = round(in_score - out_score, 2)
            conf = round(_rotation_score_confidence(gap, in_score, out_score), 2)
            if gap < gap_floor or conf < confidence_floor:
                continue
            reason = (
                f"monitoring candidate {in_symbol} ({in_action}) stronger than "
                f"portfolio position {out_symbol}; gap={gap:.2f}"
            )
            suggestions.append(
                {
                    "rotate_out_symbol": out_symbol,
                    "rotate_in_symbol": in_symbol,
                    "rotation_reason": reason,
                    "rotation_confidence": conf,
                    "rotation_gap": gap,
                    "portfolio_position_score": out_score,
                    "monitoring_candidate_score": in_score,
                    "advisory_only": True,
                }
            )
            break

    return {
        "suggestions": suggestions,
        "portfolio_positions": portfolio_positions,
        "eligible_monitoring_candidates": eligible_monitoring,
        "comparison_layer": {
            "portfolio_position_score": [x.get("portfolio_position_score") for x in portfolio_sorted],
            "monitoring_candidate_score": [x.get("monitoring_candidate_score") for x in candidates_sorted],
            "rotation_gap": [x.get("rotation_gap") for x in suggestions],
            "gap_floor": float(gap_floor),
            "confidence_floor": float(confidence_floor),
        },
    }


CALIBRATION_MIN_JOURNAL_SAMPLES = 12
CALIBRATION_MIN_REVIEW_SAMPLES = 8
CALIBRATION_ALLOWED_SCOPE_TYPES = ("threshold", "weight", "window")
CALIBRATION_SAFETY_CRITICAL_PARAMS = {
    "health.liq_low_usd",
    "health.vol24_low_usd",
    "health.stale_minutes",
    "health.min_history_points",
}


def _calibration_outcome_snapshot(journal_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    resolved_rows: List[Dict[str, Any]] = []
    returns_1h: List[float] = []
    returns_4h: List[float] = []
    positive_1h = 0
    negative_1h = 0

    for row in journal_rows:
        status_1h = str(row.get("outcome_1h_status") or "").upper()
        ret_1h = parse_float(row.get("outcome_1h_return_pct"), 0.0)
        ret_4h = parse_float(row.get("outcome_4h_return_pct"), 0.0)
        has_1h = status_1h in {"UP", "DOWN"}
        has_4h = str(row.get("outcome_4h_status") or "").upper() in {"UP", "DOWN"}
        if not has_1h and not has_4h:
            continue
        resolved_rows.append(row)
        if has_1h:
            returns_1h.append(ret_1h)
            if ret_1h >= 0:
                positive_1h += 1
            else:
                negative_1h += 1
        if has_4h:
            returns_4h.append(ret_4h)

    resolved_count = len(resolved_rows)
    sample_1h = len(returns_1h)
    avg_1h = sum(returns_1h) / sample_1h if sample_1h else 0.0
    avg_4h = sum(returns_4h) / len(returns_4h) if returns_4h else 0.0
    win_rate_1h = (positive_1h / sample_1h) if sample_1h else 0.0
    return {
        "resolved_count": resolved_count,
        "sample_1h": sample_1h,
        "win_rate_1h": round(win_rate_1h, 4),
        "avg_return_1h_pct": round(avg_1h, 4),
        "avg_return_4h_pct": round(avg_4h, 4),
        "negative_1h": negative_1h,
        "journal_slice_used": {
            "last_rows_considered": min(len(journal_rows), 200),
            "resolved_rows_used": resolved_count,
            "horizons": ["1h", "4h"],
        },
    }


def _calibration_trust_snapshot(trust_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not trust_index:
        return {
            "patterns": 0,
            "avg_samples": 0.0,
            "high_confidence_patterns": 0,
            "trust_slice_used": {"top_patterns_used": 0},
        }
    items = list(trust_index.values())
    samples = [int(parse_float(x.get("samples"), 0.0)) for x in items]
    high_conf = [x for x in items if int(parse_float(x.get("samples"), 0.0)) >= 10]
    avg_samples = (sum(samples) / len(samples)) if samples else 0.0
    return {
        "patterns": len(items),
        "avg_samples": round(avg_samples, 2),
        "high_confidence_patterns": len(high_conf),
        "trust_slice_used": {
            "top_patterns_used": min(len(items), 25),
            "min_samples_for_high_confidence": 10,
        },
    }


def _calibration_review_snapshot(review_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    weak_items = 0
    severe_items = 0
    for item in review_findings:
        score = parse_float(item.get("ui_visible_score"), 0.0)
        badge = str(item.get("ui_badge") or "").upper()
        if score < 90.0:
            weak_items += 1
        if badge in {"REVIEW", "WARNING"}:
            severe_items += 1
    total = len(review_findings)
    weak_ratio = (weak_items / total) if total else 0.0
    return {
        "review_count": total,
        "weak_items": weak_items,
        "severe_items": severe_items,
        "weak_ratio": round(weak_ratio, 4),
        "review_finding_used": {
            "weak_score_threshold": 90.0,
            "rows_used": total,
        },
    }


def _is_suggestion_scope_allowed(parameter: str, explicit_whitelist: Optional[List[str]] = None) -> bool:
    if parameter in CALIBRATION_SAFETY_CRITICAL_PARAMS:
        if not explicit_whitelist:
            return False
        return parameter in set(explicit_whitelist)
    return True


def build_manual_calibration_suggestions(
    journal_rows: List[Dict[str, Any]],
    trust_index: Dict[str, Dict[str, Any]],
    review_findings: List[Dict[str, Any]],
    explicit_safety_whitelist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    journal_snapshot = _calibration_outcome_snapshot(journal_rows[-200:])
    trust_snapshot = _calibration_trust_snapshot(trust_index)
    review_snapshot = _calibration_review_snapshot(review_findings)

    if journal_snapshot.get("resolved_count", 0) < CALIBRATION_MIN_JOURNAL_SAMPLES or review_snapshot.get("review_count", 0) < CALIBRATION_MIN_REVIEW_SAMPLES:
        return {
            "status": "not_enough_evidence",
            "message": "not enough evidence",
            "suggestions": [],
            "scope": {
                "allowed_types": list(CALIBRATION_ALLOWED_SCOPE_TYPES),
                "safety_critical_excluded": sorted(CALIBRATION_SAFETY_CRITICAL_PARAMS),
            },
            "evidence_meta": {
                "journal_slice_used": journal_snapshot.get("journal_slice_used", {}),
                "trust_slice_used": trust_snapshot.get("trust_slice_used", {}),
                "review_finding_used": review_snapshot.get("review_finding_used", {}),
            },
        }

    suggestions: List[Dict[str, Any]] = []

    weak_ratio = parse_float(review_snapshot.get("weak_ratio"), 0.0)
    win_rate_1h = parse_float(journal_snapshot.get("win_rate_1h"), 0.0)
    avg_samples = parse_float(trust_snapshot.get("avg_samples"), 0.0)

    suggested_review_threshold = 95.0 if weak_ratio > 0.6 else 85.0
    review_threshold_confidence = min(0.9, max(0.3, 0.45 + weak_ratio * 0.5))
    suggestions.append(
        {
            "proposed_change": {
                "scope_type": "threshold",
                "parameter": "review.score_visibility_cutoff",
                "current_value": 90.0,
                "suggested_value": suggested_review_threshold,
                "change_note": "Adjust review scoring cutoff for manual triage only.",
            },
            "evidence": {
                "journal_slice_used": journal_snapshot.get("journal_slice_used", {}),
                "trust_slice_used": trust_snapshot.get("trust_slice_used", {}),
                "review_finding_used": review_snapshot.get("review_finding_used", {}),
            },
            "confidence": round(review_threshold_confidence, 2),
            "risk_note": "Higher cutoff can increase review backlog; validate on a short dry-run before manual apply.",
            "manual_apply_required": True,
        }
    )

    trust_weight_value = 1.15 if (win_rate_1h > 0.58 and avg_samples >= 10) else 0.9
    trust_weight_confidence = min(0.88, max(0.35, 0.4 + (avg_samples / 40.0)))
    suggestions.append(
        {
            "proposed_change": {
                "scope_type": "weight",
                "parameter": "scoring.trust_priority_bias_multiplier",
                "current_value": 1.0,
                "suggested_value": round(trust_weight_value, 2),
                "change_note": "Tune trust bias contribution without changing trust logic.",
            },
            "evidence": {
                "journal_slice_used": journal_snapshot.get("journal_slice_used", {}),
                "trust_slice_used": trust_snapshot.get("trust_slice_used", {}),
                "review_finding_used": review_snapshot.get("review_finding_used", {}),
            },
            "confidence": round(trust_weight_confidence, 2),
            "risk_note": "Overweighting trust can under-react to fresh signals; keep manual review in loop.",
            "manual_apply_required": True,
        }
    )

    window_days = 5 if avg_samples >= 12 else 3
    suggestions.append(
        {
            "proposed_change": {
                "scope_type": "window",
                "parameter": "trust.recent_window_days",
                "current_value": TRUST_RECENT_WINDOW_DAYS,
                "suggested_value": window_days,
                "change_note": "Window sizing for trust recency smoothing.",
            },
            "evidence": {
                "journal_slice_used": journal_snapshot.get("journal_slice_used", {}),
                "trust_slice_used": trust_snapshot.get("trust_slice_used", {}),
                "review_finding_used": review_snapshot.get("review_finding_used", {}),
            },
            "confidence": round(min(0.85, 0.4 + avg_samples / 30.0), 2),
            "risk_note": "Longer windows reduce noise but may delay adaptation after market regime shifts.",
            "manual_apply_required": True,
        }
    )

    filtered_suggestions: List[Dict[str, Any]] = []
    for suggestion in suggestions:
        proposal = suggestion.get("proposed_change") or {}
        parameter = str(proposal.get("parameter") or "")
        if not _is_suggestion_scope_allowed(parameter, explicit_whitelist=explicit_safety_whitelist):
            continue
        filtered_suggestions.append(suggestion)

    return {
        "status": "ok",
        "message": "manual suggestions generated",
        "suggestions": filtered_suggestions,
        "scope": {
            "allowed_types": list(CALIBRATION_ALLOWED_SCOPE_TYPES),
            "safety_critical_excluded": sorted(CALIBRATION_SAFETY_CRITICAL_PARAMS),
            "explicit_whitelist": list(explicit_safety_whitelist or []),
        },
        "evidence_meta": {
            "journal_slice_used": journal_snapshot.get("journal_slice_used", {}),
            "trust_slice_used": trust_snapshot.get("trust_slice_used", {}),
            "review_finding_used": review_snapshot.get("review_finding_used", {}),
        },
    }


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

def get_runtime_contract_sql() -> str:
    return """
create table if not exists public.runtime_state (
  state_key text primary key,
  state_json jsonb not null default '{}'::jsonb,
  updated_ts text not null default '',
  updated_epoch double precision not null default 0
);

create table if not exists public.locks (
  lock_key text primary key,
  owner text not null default '',
  expires_epoch double precision not null default 0,
  updated_epoch double precision not null default 0
);

create table if not exists public.tg_state (
  state_key text primary key,
  state_json jsonb not null default '{}'::jsonb,
  updated_epoch double precision not null default 0
);

create table if not exists public.job_heartbeats (
  job_name text primary key,
  job_mode text not null default '',
  heartbeat_ts text not null default '',
  heartbeat_epoch double precision not null default 0,
  status text not null default '',
  meta_json jsonb not null default '{}'::jsonb
);

create table if not exists public.job_runs (
  run_id text primary key,
  job_name text not null,
  job_mode text not null default '',
  started_ts text not null default '',
  ended_ts text not null default '',
  status text not null default '',
  note text not null default ''
);
""".strip()


def check_runtime_contract(required_tables: Optional[List[str]] = None) -> Dict[str, Any]:
    tables = required_tables or list(RUNTIME_REQUIRED_TABLES)
    if not _sb_ok():
        return _runtime_status(True, "supabase_disabled", "runtime contract check skipped (supabase disabled)", tables=tables)
    failures: List[Dict[str, Any]] = []
    for table in tables:
        rows, status = _sb_select_rows(table=table, select="*", limit=1)
        if rows is None or not status.get("ok"):
            failures.append({
                "table": table,
                "code": status.get("code"),
                "detail": status.get("detail") or status.get("message"),
                "http_status": status.get("http_status"),
            })
    if failures:
        return _runtime_status(False, "runtime_contract_missing", "required runtime tables are missing or unreadable", failures=failures, sql_contract=get_runtime_contract_sql())
    return _runtime_status(True, "ok", "runtime contract ready", tables=tables)


def read_runtime_state(state_key: str = "worker_runtime") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    key = str(state_key or "worker_runtime").strip()
    default: Dict[str, Any] = {}
    if _sb_ok():
        rows, status = _sb_select_rows(
            table=RUNTIME_STATE_TABLE,
            filters={"state_key": f"eq.{key}"},
            select="state_json,updated_ts,updated_epoch",
            limit=1,
        )
        if rows is None:
            return default, status
        if not rows:
            return default, _runtime_status(True, "not_found", table=RUNTIME_STATE_TABLE, state_key=key)
        payload = rows[0].get("state_json")
        if isinstance(payload, dict):
            return dict(payload), _runtime_status(True, "ok", table=RUNTIME_STATE_TABLE, state_key=key)
        return default, _runtime_status(False, "bad_state_payload", "runtime state payload is not an object", table=RUNTIME_STATE_TABLE, state_key=key)
    path = os.path.join(DATA_DIR, f"{key}.runtime.json")
    if os.path.exists(path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data, _runtime_status(True, "ok_local", state_key=key)
            return default, _runtime_status(False, "bad_local_state_payload", state_key=key)
        except Exception as e:
            return default, _runtime_status(False, "local_read_exception", detail=f"{type(e).__name__}:{e}", state_key=key)
    return default, _runtime_status(True, "not_found_local", state_key=key)


def update_runtime_state(
    updates: Dict[str, Any],
    increments: Optional[Dict[str, int]] = None,
    state_key: str = "worker_runtime",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    current, read_status = read_runtime_state(state_key=state_key)
    payload = dict(current if isinstance(current, dict) else {})
    for k, v in (updates or {}).items():
        payload[k] = v
    for k, delta in (increments or {}).items():
        payload[k] = int(parse_float(payload.get(k, 0), 0.0)) + int(delta or 0)
    now_ts = now_utc_str()
    now_epoch = time.time()
    key = str(state_key or "worker_runtime").strip()
    if _sb_ok():
        status = _sb_upsert_row(
            table=RUNTIME_STATE_TABLE,
            on_conflict="state_key",
            payload={
                "state_key": key,
                "state_json": payload,
                "updated_ts": now_ts,
                "updated_epoch": now_epoch,
            },
        )
        return payload, status
    ensure_storage()
    path = os.path.join(DATA_DIR, f"{key}.runtime.json")
    try:
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload, _runtime_status(True, "ok_local", state_key=key)
    except Exception as e:
        return payload, _runtime_status(False, "local_write_exception", detail=f"{type(e).__name__}:{e}", state_key=key, read_status=read_status)


def acquire_lock(lock_key: str, owner: str, ttl_sec: int = 240) -> Dict[str, Any]:
    lk = str(lock_key or "").strip()
    if not lk:
        return _runtime_status(False, "lock_key_required", "lock_key is required")
    owner = str(owner or "unknown").strip() or "unknown"
    now_epoch = time.time()
    expires_epoch = now_epoch + max(1, int(ttl_sec or 0))
    if _sb_ok():
        rows, status = _sb_select_rows(
            table=LOCKS_TABLE,
            filters={"lock_key": f"eq.{lk}"},
            select="owner,expires_epoch",
            limit=1,
        )
        if rows is None:
            return status
        if rows:
            holder = str(rows[0].get("owner") or "")
            held_until = float(parse_float(rows[0].get("expires_epoch", 0), 0.0))
            if held_until > now_epoch and holder and holder != owner:
                return _runtime_status(False, "lock_held", f"lock already held by {holder}", lock_key=lk, holder=holder, expires_epoch=held_until)
            stale_replaced = bool(held_until > 0 and held_until <= now_epoch and holder and holder != owner)
            stale_age_sec = int(max(0.0, now_epoch - held_until)) if stale_replaced else 0
        else:
            stale_replaced = False
            stale_age_sec = 0
        upsert_status = _sb_upsert_row(
            table=LOCKS_TABLE,
            on_conflict="lock_key",
            payload={"lock_key": lk, "owner": owner, "expires_epoch": expires_epoch, "updated_epoch": now_epoch},
        )
        if not upsert_status.get("ok"):
            return upsert_status
        return _runtime_status(
            True,
            "ok",
            lock_key=lk,
            owner=owner,
            expires_epoch=expires_epoch,
            stale_replaced=stale_replaced,
            stale_age_sec=stale_age_sec,
        )
    ensure_storage()
    lock_file = os.path.join(DATA_DIR, f"{lk}.lock.json")
    try:
        if os.path.exists(lock_file):
            current_raw = Path(lock_file).read_text(encoding="utf-8")
            current = json.loads(current_raw) if current_raw else {}
            if isinstance(current, dict):
                holder = str(current.get("owner") or "")
                held_until = float(parse_float(current.get("expires_epoch", 0), 0.0))
                if held_until > now_epoch and holder and holder != owner:
                    return _runtime_status(
                        False,
                        "lock_held",
                        f"lock already held by {holder}",
                        lock_key=lk,
                        holder=holder,
                        expires_epoch=held_until,
                    )
                stale_replaced = bool(held_until > 0 and held_until <= now_epoch and holder and holder != owner)
                stale_age_sec = int(max(0.0, now_epoch - held_until)) if stale_replaced else 0
            else:
                stale_replaced = False
                stale_age_sec = 0
        else:
            stale_replaced = False
            stale_age_sec = 0
        payload = {"owner": owner, "expires_epoch": expires_epoch, "updated_epoch": now_epoch}
        Path(lock_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return _runtime_status(
            True,
            "ok_local",
            lock_key=lk,
            owner=owner,
            expires_epoch=expires_epoch,
            stale_replaced=stale_replaced,
            stale_age_sec=stale_age_sec,
        )
    except Exception as e:
        return _runtime_status(False, "local_lock_write_exception", detail=f"{type(e).__name__}:{e}", lock_key=lk)


def release_lock(lock_key: str, owner: str) -> Dict[str, Any]:
    lk = str(lock_key or "").strip()
    owner = str(owner or "").strip()
    if not lk:
        return _runtime_status(False, "lock_key_required", "lock_key is required")
    if _sb_ok():
        rows, status = _sb_select_rows(
            table=LOCKS_TABLE,
            filters={"lock_key": f"eq.{lk}"},
            select="owner",
            limit=1,
        )
        if rows is None:
            return status
        if rows and owner and str(rows[0].get("owner") or "") not in ("", owner):
            return _runtime_status(False, "lock_owner_mismatch", "cannot release lock owned by another worker", lock_key=lk, owner=rows[0].get("owner"))
        try:
            resp = requests.delete(
                _sb_table_url(LOCKS_TABLE),
                headers=_sb_headers(),
                params={"lock_key": f"eq.{lk}"},
                timeout=12,
            )
            if resp.status_code >= 400:
                body = (resp.text or "").replace("\n", " ").strip()[:240]
                return _runtime_status(False, "supabase_delete_failed", "lock release failed", lock_key=lk, http_status=resp.status_code, detail=body)
            return _runtime_status(True, "ok", lock_key=lk)
        except Exception as e:
            return _runtime_status(False, "supabase_delete_exception", "lock release exception", lock_key=lk, detail=f"{type(e).__name__}:{e}")
    ensure_storage()
    lock_file = os.path.join(DATA_DIR, f"{lk}.lock.json")
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        return _runtime_status(True, "ok_local", lock_key=lk)
    except Exception as e:
        return _runtime_status(False, "local_lock_delete_exception", detail=f"{type(e).__name__}:{e}", lock_key=lk)


def update_job_heartbeat(job_name: str, job_mode: str, status: str = "alive", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    name = str(job_name or "").strip()
    if not name:
        return _runtime_status(False, "job_name_required", "job_name is required")
    now_ts = now_utc_str()
    now_epoch = time.time()
    payload = {
        "job_name": name,
        "job_mode": str(job_mode or "").strip(),
        "heartbeat_ts": now_ts,
        "heartbeat_epoch": now_epoch,
        "status": str(status or "").strip() or "alive",
        "meta_json": dict(meta or {}),
    }
    if _sb_ok():
        status_payload = _sb_upsert_row(
            table=JOB_HEARTBEATS_TABLE,
            on_conflict="job_name",
            payload=payload,
        )
        if status_payload.get("ok"):
            return _runtime_status(True, "ok", **payload)
        return status_payload
    ensure_storage()
    path = os.path.join(DATA_DIR, "job_heartbeats.json")
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    data[name] = payload
    try:
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return _runtime_status(True, "ok_local", **payload)
    except Exception as e:
        return _runtime_status(False, "local_write_exception", detail=f"{type(e).__name__}:{e}", **payload)


def get_job_heartbeats_snapshot() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if _sb_ok():
        rows, status = _sb_select_rows(table=JOB_HEARTBEATS_TABLE, limit=2000)
        if not status.get("ok"):
            return {}, status
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("job_name") or "").strip()
            if not name:
                continue
            out[name] = {
                "job_name": name,
                "job_mode": str(row.get("job_mode") or "").strip(),
                "status": str(row.get("status") or "").strip(),
                "heartbeat_ts": str(row.get("heartbeat_ts") or "").strip(),
                "heartbeat_epoch": parse_float(row.get("heartbeat_epoch"), 0.0),
                "meta_json": row.get("meta_json") if isinstance(row.get("meta_json"), dict) else {},
            }
        return out, _runtime_status(True, "ok", jobs=len(out))

    ensure_storage()
    path = os.path.join(DATA_DIR, "job_heartbeats.json")
    if not os.path.exists(path):
        return {}, _runtime_status(True, "not_found_local", jobs=0)
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return {}, _runtime_status(False, "bad_local_payload", jobs=0)
        out: Dict[str, Dict[str, Any]] = {}
        for raw_name, raw_payload in loaded.items():
            name = str(raw_name or "").strip()
            payload = raw_payload if isinstance(raw_payload, dict) else {}
            if not name:
                continue
            out[name] = {
                "job_name": name,
                "job_mode": str(payload.get("job_mode") or "").strip(),
                "status": str(payload.get("status") or "").strip(),
                "heartbeat_ts": str(payload.get("heartbeat_ts") or "").strip(),
                "heartbeat_epoch": parse_float(payload.get("heartbeat_epoch"), 0.0),
                "meta_json": payload.get("meta_json") if isinstance(payload.get("meta_json"), dict) else {},
            }
        return out, _runtime_status(True, "ok_local", jobs=len(out))
    except Exception as e:
        return {}, _runtime_status(False, "local_read_exception", detail=f"{type(e).__name__}:{e}", jobs=0)

def scanner_acquire_lock(slot: int, ttl_sec: int = 240) -> bool:
    """
    Prevent multiple tabs running the scanner simultaneously.
    Uses persistent lock storage (Supabase locks table or local fallback).
    """
    key = f"scanner_lock_{slot}"
    owner = f"{os.getenv('HOSTNAME', 'local')}:{os.getpid()}"
    status = acquire_lock(lock_key=key, owner=owner, ttl_sec=ttl_sec)
    if not status.get("ok"):
        debug_log(f"scanner_acquire_lock_failed key={key} code={status.get('code')} detail={status.get('detail') or status.get('message')}")
    return bool(status.get("ok"))

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
        "sent_emissions": {},
        "sent_digest_emissions": {},
        "last_ui_digest_fingerprint": "",
        "last_discovery_digest_fingerprint": "",
        "last_ui_digest_ts": "",
        "last_discovery_digest_ts": "",
        "sent_portfolio_events": {},
        "engine_version": "v3",
        "token_state": {},
        "settings": {
            "alert_mode": DEFAULT_ALERT_MODE,
        },
    }

    try:
        raw = None
        if USE_SUPABASE:
            rows, status = _sb_select_rows(
                table=TG_STATE_TABLE,
                filters={"state_key": "eq.default"},
                select="state_json",
                limit=1,
            )
            if rows is None:
                debug_log(f"tg_state_read_failed code={status.get('code')} detail={status.get('detail') or status.get('message')}")
                return {
                    **default,
                    "_runtime_error": {
                        "code": status.get("code"),
                        "message": status.get("detail") or status.get("message"),
                    },
                }
            if rows and isinstance(rows[0].get("state_json"), dict):
                raw = json.dumps(rows[0].get("state_json") or {}, ensure_ascii=False)
            else:
                raw = sb_get_storage(TG_STATE_KEY)
        if (not raw) and os.path.exists(TG_STATE_FILE):
            with open(TG_STATE_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in default.items():
                    data.setdefault(k, v)
                settings = data.get("settings", {})
                if not isinstance(settings, dict):
                    settings = {}
                legacy_mode = data.get("alert_mode")
                if legacy_mode and not settings.get("alert_mode"):
                    settings["alert_mode"] = legacy_mode
                settings["alert_mode"] = str(settings.get("alert_mode") or DEFAULT_ALERT_MODE).strip().lower()
                if settings["alert_mode"] not in ALERT_MODE_REGISTRY:
                    settings["alert_mode"] = DEFAULT_ALERT_MODE
                data["settings"] = settings
                if data.get("engine_version") != "v3":
                    data["sent_events"] = {}
                    data["sent_portfolio_events"] = {}
                    data["last_scan_ts_processed"] = ""
                    data["engine_version"] = "v3"
                legacy_digest_ts = str(data.get("last_digest_sent_at") or "")
                if legacy_digest_ts:
                    if not str(data.get("last_ui_digest_ts") or "").strip():
                        data["last_ui_digest_ts"] = legacy_digest_ts
                    if not str(data.get("last_discovery_digest_ts") or "").strip():
                        data["last_discovery_digest_ts"] = legacy_digest_ts
                return data
    except Exception as e:
        print(f"[TG] load state fallback {type(e).__name__}: {e}", flush=True)

    return default


def save_tg_state(state: Dict[str, Any]) -> None:
    print("[TG] saving state", flush=True)

    try:
        payload = json.dumps(state, ensure_ascii=False)
        if USE_SUPABASE:
            status = _sb_upsert_row(
                table=TG_STATE_TABLE,
                on_conflict="state_key",
                payload={
                    "state_key": "default",
                    "state_json": state,
                    "updated_epoch": time.time(),
                },
            )
            if not status.get("ok"):
                debug_log(f"tg_state_write_failed code={status.get('code')} detail={status.get('detail') or status.get('message')}")
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
        "sent_emissions": {},
        "sent_digest_emissions": {},
        "last_ui_digest_fingerprint": "",
        "last_discovery_digest_fingerprint": "",
        "last_ui_digest_ts": "",
        "last_discovery_digest_ts": "",
        "sent_portfolio_events": {},
        "engine_version": "v3",
        "token_state": {},
        "settings": {
            "alert_mode": DEFAULT_ALERT_MODE,
        },
    }
    save_tg_state(state)


def _normalize_alert_mode(mode: Any) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in ALERT_MODE_REGISTRY:
        return DEFAULT_ALERT_MODE
    return normalized


def get_alert_mode(state: Optional[Dict[str, Any]] = None) -> str:
    state_obj = state if isinstance(state, dict) else load_tg_state()
    settings = state_obj.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    mode = settings.get("alert_mode", state_obj.get("alert_mode"))
    normalized = _normalize_alert_mode(mode)
    settings["alert_mode"] = normalized
    state_obj["settings"] = settings
    return normalized


def set_alert_mode(mode: str, state: Optional[Dict[str, Any]] = None, autosave: bool = True) -> str:
    state_obj = state if isinstance(state, dict) else load_tg_state()
    settings = state_obj.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    normalized = _normalize_alert_mode(mode)
    settings["alert_mode"] = normalized
    state_obj["settings"] = settings
    if autosave:
        save_tg_state(state_obj)
    return normalized


def tg_cooldown_ok(state: Dict[str, Any], seconds: int = 3600) -> bool:
    now = time.time()
    last = float(state.get("last_signal_run", 0.0) or 0.0)
    if now - last < seconds:
        return False
    state["last_signal_run"] = now
    return True


EVENT_TYPE_REGISTRY: Dict[str, str] = {
    "ENTRY_ALERT": "entry_alert",
    "PORTFOLIO_ACTION": "portfolio_action",
    "STATE_TRANSITION": "state_transition",
    "DIGEST": "digest",
    "HEALTH_WARNING": "health_warning",
}

ALERT_TIER_REGISTRY: Dict[str, str] = {
    "CRITICAL": "critical",
    "HIGH": "high",
    "MEDIUM": "medium",
    "DIGEST_ONLY": "digest_only",
}


def resolve_event_type(name: str, fallback: str = "STATE_TRANSITION") -> str:
    key = str(name or "").strip().upper()
    if key in EVENT_TYPE_REGISTRY:
        return EVENT_TYPE_REGISTRY[key]
    return EVENT_TYPE_REGISTRY.get(str(fallback).strip().upper(), EVENT_TYPE_REGISTRY["STATE_TRANSITION"])


def resolve_alert_tier(name: str, fallback: str = "MEDIUM") -> str:
    key = str(name or "").strip().upper()
    if key in ALERT_TIER_REGISTRY:
        return ALERT_TIER_REGISTRY[key]
    return ALERT_TIER_REGISTRY.get(str(fallback).strip().upper(), ALERT_TIER_REGISTRY["MEDIUM"])


def resolve_tg_alert_classification(
    source: str,
    row: Dict[str, Any],
    signal: Dict[str, str],
    unified: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    source = str(source or "").strip().lower()
    action = str((unified or {}).get("final_action") or row.get("entry_action") or row.get("entry") or "").upper()
    risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()
    reason = str((unified or {}).get("final_reason") or row.get("entry_reason") or row.get("signal_reason") or "").lower()
    signal_bucket = str(signal.get("bucket") or "").upper()

    health_critical_markers = (
        "anti_rug_critical",
        "liquidity critical/dead",
        "critical_trap",
        "liq_critical",
    )
    is_health_critical = any(marker in reason for marker in health_critical_markers) or (
        source == "monitoring" and risk == "HIGH" and action in ("AVOID", "NO_ENTRY", "EXIT")
    )
    if is_health_critical:
        return {
            "event_type": resolve_event_type("HEALTH_WARNING"),
            "alert_tier": resolve_alert_tier("CRITICAL"),
        }

    if source == "portfolio":
        tier_key = "MEDIUM"
        if signal_bucket == "EXIT":
            tier_key = "HIGH"
        elif signal_bucket == "ADD":
            tier_key = "HIGH"
        return {
            "event_type": resolve_event_type("PORTFOLIO_ACTION"),
            "alert_tier": resolve_alert_tier(tier_key),
        }

    tier_key = "MEDIUM"
    if signal_bucket in ("ENTRY_NOW", "WATCH"):
        tier_key = "HIGH"
    return {
        "event_type": resolve_event_type("ENTRY_ALERT"),
        "alert_tier": resolve_alert_tier(tier_key),
    }


def _is_portfolio_exit_reduce_alert(
    source: str,
    row: Dict[str, Any],
    unified: Optional[Dict[str, Any]] = None,
) -> bool:
    if str(source or "").strip().lower() != "portfolio":
        return False
    action = str(
        (unified or {}).get("final_action")
        or row.get("entry_action")
        or row.get("entry")
        or ""
    ).strip().upper()
    return action in ("EXIT", "REDUCE")


def should_emit_for_alert_mode(
    mode: str,
    source: str,
    row: Dict[str, Any],
    signal: Dict[str, str],
    classification: Dict[str, str],
    unified: Optional[Dict[str, Any]] = None,
) -> bool:
    resolved_mode = _normalize_alert_mode(mode)
    event_type = str(classification.get("event_type") or "").strip().lower()
    alert_tier = str(classification.get("alert_tier") or "").strip().lower()

    is_critical = alert_tier == resolve_alert_tier("CRITICAL")
    is_health_warning = event_type == resolve_event_type("HEALTH_WARNING")
    is_portfolio_exit_reduce = _is_portfolio_exit_reduce_alert(source, row, unified=unified)

    if resolved_mode == "quiet":
        # Hard safety overrides: never silence critical / hard health warnings /
        # portfolio risk actions (EXIT/REDUCE).
        if is_critical or is_health_warning or is_portfolio_exit_reduce:
            return True
        if alert_tier in (resolve_alert_tier("MEDIUM"), resolve_alert_tier("DIGEST_ONLY")):
            return False
        return True

    if resolved_mode == "aggressive":
        # Aggressive mode allows extra state/monitoring flow, but idempotency is
        # still applied later in pipeline.
        return True

    # normal (default + safe fallback)
    return True


def build_emission_key_foundation(
    source: str,
    row: Dict[str, Any],
    signal: Dict[str, str],
    unified: Optional[Dict[str, Any]] = None,
) -> str:
    classification = resolve_tg_alert_classification(source, row, signal, unified=unified)
    chain = token_chain(row)
    addr = str(token_ca(row) or row.get("token_address") or "").strip().lower()
    event_type = classification["event_type"]
    alert_tier = classification["alert_tier"]
    return f"{source}|{chain}|{addr}|{event_type}|{alert_tier}"


def normalize_alert_action_bucket(row: Dict[str, Any], unified: Optional[Dict[str, Any]] = None) -> str:
    action_raw = str(
        (unified or {}).get("final_action")
        or row.get("entry_action")
        or row.get("entry")
        or "UNKNOWN"
    ).strip().upper()
    compact = re.sub(r"[^A-Z0-9]+", "_", action_raw).strip("_")
    return compact or "UNKNOWN"


def normalize_alert_reason_bucket(row: Dict[str, Any], unified: Optional[Dict[str, Any]] = None) -> str:
    reason_raw = str(
        (unified or {}).get("final_reason")
        or row.get("entry_reason")
        or row.get("signal_reason")
        or "unknown_reason"
    ).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", reason_raw).strip("_")
    if not normalized:
        return "unknown_reason"
    parts = [p for p in normalized.split("_") if p]
    if not parts:
        return "unknown_reason"
    return "_".join(parts[:3])


def emission_cooldown_bucket(now_ts: float, cooldown_seconds: int) -> int:
    cooldown_seconds = max(60, int(cooldown_seconds or 0))
    return int(now_ts // cooldown_seconds)


def build_emission_key(
    source: str,
    row: Dict[str, Any],
    signal: Dict[str, str],
    cooldown_seconds: int,
    now_ts: Optional[float] = None,
    unified: Optional[Dict[str, Any]] = None,
) -> str:
    classification = resolve_tg_alert_classification(source, row, signal, unified=unified)
    chain = token_chain(row)
    addr = str(token_ca(row) or row.get("token_address") or "").strip().lower()
    event_type = str(classification.get("event_type") or "")
    action_bucket = normalize_alert_action_bucket(row, unified=unified)
    reason_bucket = normalize_alert_reason_bucket(row, unified=unified)
    current_ts = float(now_ts if now_ts is not None else time.time())
    cooldown_bucket = emission_cooldown_bucket(current_ts, cooldown_seconds)
    return f"{source}|{chain}|{addr}|{event_type}|{action_bucket}|{reason_bucket}|{cooldown_bucket}"


def is_duplicate_emission(state: Dict[str, Any], emission_key: str, now_ts: float, cooldown_seconds: int) -> bool:
    sent_emissions = state.get("sent_emissions", {})
    if not isinstance(sent_emissions, dict):
        sent_emissions = {}
    state["sent_emissions"] = sent_emissions

    last_ts = float(sent_emissions.get(emission_key, 0.0) or 0.0)
    if not last_ts:
        return False
    return (now_ts - last_ts) < max(60, int(cooldown_seconds or 0))


def mark_emission_sent(state: Dict[str, Any], emission_key: str, now_ts: float) -> None:
    sent_emissions = state.get("sent_emissions", {})
    if not isinstance(sent_emissions, dict):
        sent_emissions = {}
    sent_emissions[emission_key] = float(now_ts)
    state["sent_emissions"] = sent_emissions


def classify_monitoring_signal(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    score = parse_float(row.get("entry_score", 0), 0.0)
    action = str(row.get("entry_action") or row.get("entry") or "").upper()
    setup_inputs = build_setup_render_inputs(row)
    has_levels = bool(setup_inputs.get("has_actionable_levels"))

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
    foundation = build_emission_key_foundation(source, row, signal)
    action = str(row.get("entry_action") or row.get("entry") or "").upper()
    risk = str(row.get("risk_level") or row.get("risk") or "").upper()
    timing = normalize_timing_label(row.get("timing_label") or "")
    score_bucket = int(parse_float(row.get("entry_score"), 0.0) // 25)
    return f"{foundation}|{action}|{risk}|{timing}|{score_bucket}"


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


def _journal_reference_price(row: Dict[str, Any]) -> float:
    return parse_float(
        row.get("price_usd")
        or row.get("priceUsd")
        or row.get("price")
        or row.get("entry_price_usd")
        or 0.0,
        0.0,
    )


def build_journal_event_id(event_key: str, emission_key: str) -> str:
    raw = f"{event_key}|{emission_key}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:24]


def build_base_journal_row(
    row: Dict[str, Any],
    source: str,
    signal: Dict[str, str],
    alert_classification: Dict[str, str],
    unified: Dict[str, Any],
    event_key: str,
    emission_key: str,
    send_status: str,
    send_note: str = "",
) -> Dict[str, Any]:
    token_key = build_token_state_key(row)
    health = dict(unified.get("health") or {})
    return {
        "event_id": build_journal_event_id(event_key, emission_key),
        "journal_type": "signal",
        "source_event_id": str(event_key or ""),
        "ts_utc": now_utc_str(),
        "token_key": token_key,
        "chain": token_chain(row),
        "token_addr": token_ca(row),
        "source": str(source or "").strip().lower(),
        "event_type": str(alert_classification.get("event_type") or ""),
        "alert_tier": str(alert_classification.get("alert_tier") or ""),
        "signal_bucket": str(signal.get("bucket") or ""),
        "signal_action": str(signal.get("action") or ""),
        "signal_horizon": str(signal.get("horizon") or ""),
        "final_action": str(unified.get("final_action") or row.get("entry_action") or row.get("entry") or ""),
        "final_reason": str(unified.get("final_reason") or row.get("entry_reason") or row.get("signal_reason") or ""),
        "health_label": str(health.get("health_label") or "OK"),
        "health_override_active": "1" if bool(unified.get("health_override_active")) else "0",
        "health_override_action": str(unified.get("health_override_action") or ""),
        "health_override_reason": str(unified.get("health_override_reason") or ""),
        "entry_score": str(parse_float(row.get("entry_score", 0), 0.0)),
        "timing_label": str(unified.get("timing") or row.get("timing_label") or ""),
        "risk_level": str(row.get("risk_level") or row.get("risk") or ""),
        "setup_context": safe_json({
            "entry_action": str(row.get("entry_action") or row.get("entry") or ""),
            "entry_reason": str(row.get("entry_reason") or row.get("signal_reason") or ""),
            "setup_watch_only": str(row.get("setup_watch_only") or ""),
        }),
        "reference_price_usd": str(_journal_reference_price(row)),
        "emission_key": emission_key,
        "event_key": event_key,
        "send_status": str(send_status or "").strip().lower(),
        "send_note": str(send_note or ""),
    }


def _carry_forward_outcomes(base_row: Dict[str, Any], existing_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = dict(base_row)
    row.update(_journal_outcome_init())
    if existing_row:
        for key in JOURNAL_OUTCOME_FIELDS:
            if str(existing_row.get(key) or "").strip() != "":
                row[key] = existing_row.get(key, "")
    return row


def write_signal_journal_row(base_row: Dict[str, Any]) -> None:
    signal_row = dict(base_row)
    signal_row["journal_type"] = "monitoring_signal" if str(base_row.get("source") or "") == "monitoring" else "portfolio_signal"
    existing = get_csv_row_by_id(SIGNAL_JOURNAL_CSV, SIGNAL_JOURNAL_FIELDS, base_row.get("event_id", ""))
    upsert_csv_row(
        SIGNAL_JOURNAL_CSV,
        _carry_forward_outcomes(signal_row, existing),
        SIGNAL_JOURNAL_FIELDS,
    )


def write_portfolio_action_journal_row(base_row: Dict[str, Any]) -> None:
    port_row = dict(base_row)
    port_row["journal_type"] = "portfolio_action"
    existing = get_csv_row_by_id(PORTFOLIO_ACTION_JOURNAL_CSV, PORTFOLIO_ACTION_JOURNAL_FIELDS, base_row.get("event_id", ""))
    upsert_csv_row(
        PORTFOLIO_ACTION_JOURNAL_CSV,
        _carry_forward_outcomes(port_row, existing),
        PORTFOLIO_ACTION_JOURNAL_FIELDS,
    )


def write_health_override_journal_row(base_row: Dict[str, Any]) -> None:
    health_row = dict(base_row)
    health_row["journal_type"] = "health_override"
    existing = get_csv_row_by_id(HEALTH_OVERRIDE_JOURNAL_CSV, HEALTH_OVERRIDE_JOURNAL_FIELDS, base_row.get("event_id", ""))
    upsert_csv_row(
        HEALTH_OVERRIDE_JOURNAL_CSV,
        _carry_forward_outcomes(health_row, existing),
        HEALTH_OVERRIDE_JOURNAL_FIELDS,
    )


def journal_actionable_event(
    row: Dict[str, Any],
    source: str,
    signal: Dict[str, str],
    alert_classification: Dict[str, str],
    unified: Dict[str, Any],
    event_key: str,
    emission_key: str,
    send_status: str,
    send_note: str = "",
) -> None:
    try:
        base = build_base_journal_row(
            row=row,
            source=source,
            signal=signal,
            alert_classification=alert_classification,
            unified=unified,
            event_key=event_key,
            emission_key=emission_key,
            send_status=send_status,
            send_note=send_note,
        )
        write_signal_journal_row(base)

        if str(source or "").strip().lower() == "portfolio":
            write_portfolio_action_journal_row(base)

        if bool(unified.get("health_override_active")):
            write_health_override_journal_row(base)
    except Exception as exc:
        debug_log(f"journal_write_failed:{type(exc).__name__}:{exc}")


def _resolve_horizon_outcome(event_ts: datetime, horizon_minutes: int, token_key: str, reference_price: float, history_by_key: Dict[str, List[Dict[str, Any]]], now_ts: datetime) -> Dict[str, str]:
    payload = {"status": "PENDING", "ts_utc": "", "return_pct": "", "price_usd": ""}
    target_ts = event_ts + timedelta(minutes=max(1, int(horizon_minutes)))
    if now_ts < target_ts:
        return payload
    if reference_price <= 0:
        payload["status"] = "ERROR"
        return payload
    entries = history_by_key.get(token_key, [])
    if not entries:
        return payload

    picked: Optional[Dict[str, Any]] = None
    for item in entries:
        hist_ts = item.get("_ts")
        if not isinstance(hist_ts, datetime):
            continue
        if hist_ts >= target_ts:
            picked = item
            break
    if picked is None:
        return payload

    price = parse_float(picked.get("price_usd"), 0.0)
    if price <= 0:
        payload["status"] = "ERROR"
        payload["ts_utc"] = str(picked.get("ts_utc") or "")
        return payload
    ret_pct = ((price - reference_price) / reference_price) * 100.0
    status = "FLAT"
    if ret_pct >= 1.0:
        status = "UP"
    elif ret_pct <= -1.0:
        status = "DOWN"
    payload["status"] = status
    payload["ts_utc"] = str(picked.get("ts_utc") or "")
    payload["return_pct"] = f"{ret_pct:.4f}"
    payload["price_usd"] = f"{price:.12g}"
    return payload


def evaluate_outcome_journals() -> Dict[str, int]:
    now_dt = datetime.utcnow()
    history_rows = load_monitoring_history(limit_rows=25000)
    history_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for h in history_rows:
        chain = str(h.get("chain") or "").strip().lower()
        addr = str(h.get("base_addr") or "").strip()
        if not chain or not addr:
            continue
        key = f"{chain}:{addr}"
        item = dict(h)
        item["_ts"] = parse_ts(h.get("ts_utc"))
        history_by_key.setdefault(key, []).append(item)
    for arr in history_by_key.values():
        arr.sort(key=lambda x: x.get("_ts") or datetime.min)

    files = [
        (SIGNAL_JOURNAL_CSV, SIGNAL_JOURNAL_FIELDS),
        (HEALTH_OVERRIDE_JOURNAL_CSV, HEALTH_OVERRIDE_JOURNAL_FIELDS),
        (PORTFOLIO_ACTION_JOURNAL_CSV, PORTFOLIO_ACTION_JOURNAL_FIELDS),
    ]
    stats = {"updated": 0, "rows_scanned": 0}
    for path, fields in files:
        partial = update_journal_outcome_snapshots(
            path=path,
            fields=fields,
            history_by_key=history_by_key,
            now_dt=now_dt,
        )
        stats["updated"] += int(partial.get("updated", 0))
        stats["rows_scanned"] += int(partial.get("rows_scanned", 0))
    return stats


def update_journal_outcome_snapshots(
    path: str,
    fields: List[str],
    history_by_key: Dict[str, List[Dict[str, Any]]],
    now_dt: datetime,
) -> Dict[str, int]:
    rows = load_csv(path, fields)
    changed = False
    stats = {"updated": 0, "rows_scanned": 0}
    for row in rows:
        stats["rows_scanned"] += 1
        ts = parse_ts(row.get("ts_utc"))
        token_key = str(row.get("token_key") or "").strip()
        reference_price = parse_float(row.get("reference_price_usd"), 0.0)
        if ts is None or not token_key:
            continue
        for horizon, minutes in OUTCOME_HORIZONS_MINUTES.items():
            status_field = f"outcome_{horizon}_status"
            if str(row.get(status_field) or "").strip().upper() not in {"", "PENDING"}:
                continue
            resolved = _resolve_horizon_outcome(
                event_ts=ts,
                horizon_minutes=minutes,
                token_key=token_key,
                reference_price=reference_price,
                history_by_key=history_by_key,
                now_ts=now_dt,
            )
            row[status_field] = resolved["status"]
            row[f"outcome_{horizon}_ts_utc"] = resolved["ts_utc"]
            row[f"outcome_{horizon}_return_pct"] = resolved["return_pct"]
            row[f"outcome_{horizon}_price_usd"] = resolved["price_usd"]
            changed = True
            stats["updated"] += 1
    if changed:
        save_csv(path, rows, fields)
        try:
            load_pattern_trust_index_cached.clear()
        except Exception:
            pass
    return stats


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




def _parse_numeric_score_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def _format_notification_score_display(score_value: Optional[float], score_available: bool) -> str:
    if not score_available or score_value is None:
        return "n/a"
    return f"{score_value:.2f}"


def build_notification_semantics(row: Dict[str, Any], source_context: Optional[str] = None) -> Dict[str, Any]:
    score_sources: List[Tuple[str, Any]] = [
        ("entry_score", row.get("entry_score")),
        ("composite_score", row.get("composite_score")),
        ("final_score", row.get("final_score")),
        ("priority_score", row.get("priority_score")),
        ("scanner_score", row.get("scanner_score")),
    ]

    score_value: Optional[float] = None
    score_kind = "missing"
    for kind, raw_value in score_sources:
        parsed = _parse_numeric_score_value(raw_value)
        if parsed is None:
            continue
        score_value = parsed
        score_kind = kind
        break

    score_available = score_value is not None
    score_display = _format_notification_score_display(score_value, score_available)

    return {
        "source_context": str(source_context or ""),
        "score_value": score_value,
        "score_available": bool(score_available),
        "score_display": score_display,
        "score_kind": score_kind,
    }


def format_entry_alert_message(row: Dict[str, Any], signal: Dict[str, str], source: str) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
    chain = token_chain(row).upper()
    addr = token_ca(row)
    source = str(source or "monitoring").lower()
    unified = compute_unified_recommendation(row, source="monitoring")

    semantics = build_notification_semantics(row, source_context=source)
    risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()
    timing = normalize_timing_label(str(row.get("timing_label") or "NEUTRAL"))
    reason = str(unified.get("final_reason") or row.get("entry_reason") or row.get("signal_reason") or "n/a")
    horizon = str(row.get("entry_horizon") or signal.get("horizon") or "n/a")

    setup_inputs = build_setup_render_inputs(row)
    levels_block = str(row.get("setup_levels_block") or setup_inputs.get("levels_block") or "setup: watch only")

    return (
        f"<b>{unified['final_action']}</b> | <b>{symbol}</b>\n"
        f"source: MONITOR\n"
        f"chain: {chain}\n"
        f"score: <b>{semantics['score_display']}</b>\n"
        f"risk: <b>{risk}</b>\n"
        f"timing: <b>{timing}</b>\n"
        f"horizon: {horizon}\n"
        f"reason: {reason}\n"
        f"{levels_block}\n\n"
        f"CA:\n<code>{addr}</code>"
    )


def format_portfolio_action_message(row: Dict[str, Any]) -> str:
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


def format_state_transition_message(row: Dict[str, Any], source: str) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
    chain = token_chain(row).upper()
    addr = token_ca(row)
    action = str(row.get("entry_action") or row.get("entry") or "UPDATE").upper()
    return (
        f"<b>STATE TRANSITION</b> | <b>{symbol}</b>\n"
        f"source: {str(source or 'monitoring').upper()}\n"
        f"action: <b>{action}</b>\n"
        f"chain: {chain}\n\n"
        f"CA:\n<code>{addr}</code>"
    )


def format_digest_message(row: Dict[str, Any], source: str) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "MARKET").strip()
    chain = token_chain(row).upper() or "MULTI"
    addr = token_ca(row) or "n/a"
    semantics = build_notification_semantics(row, source_context=source)
    dex_url = str(row.get("dex_url") or row.get("dexscreener_url") or dex_url_for_token(token_chain(row), token_ca(row))).strip()
    dex_line = f"dex: {dex_url}\n" if dex_url else ""
    provenance_block = ""
    if str(source or "").lower() == "discovery":
        origin_label = str(row.get("origin_label") or row.get("source") or "scanner_candidate")
        reason_short = str(row.get("reason_short") or row.get("entry_reason") or "discovery candidate")
        timing = normalize_timing_label(str(row.get("timing") or row.get("timing_label") or "NEUTRAL"))
        provenance_block = (
            f"origin: {origin_label}\n"
            f"reason: {reason_short}\n"
            f"timing: {timing}\n"
        )
    return (
        f"<b>DIGEST</b> | <b>{symbol}</b>\n"
        f"source: {str(source or 'monitoring').upper()}\n"
        f"chain: {chain}\n"
        f"score: <b>{semantics['score_display']}</b>\n\n"
        f"{provenance_block}"
        f"{dex_line}"
        f"CA:\n<code>{addr}</code>"
    )


def _risk_priority_value(row: Dict[str, Any]) -> int:
    risk = str(row.get("risk_level") or row.get("risk") or "").strip().upper()
    if risk == "HIGH":
        return 3
    if risk == "MEDIUM":
        return 2
    if risk == "LOW":
        return 1
    return 0


def digest_source_mode() -> str:
    raw = str(os.getenv("DIGEST_SOURCE_MODE", DIGEST_SOURCE_MODE_DEFAULT) or "").strip().lower()
    if raw in DIGEST_SOURCE_MODES:
        return raw
    return DIGEST_SOURCE_MODE_DEFAULT


def canonical_token_key(row: Dict[str, Any]) -> str:
    chain = token_chain(row)
    ca = token_ca(row)
    if not chain or not ca:
        return ""
    return f"{chain}|{addr_store(chain, ca)}"


def canonical_entity_key(chain: str, ca: str) -> str:
    norm_chain = normalize_chain_name(chain)
    norm_ca = addr_store(norm_chain, ca)
    if not norm_chain or not norm_ca:
        return ""
    return f"{norm_chain}|{norm_ca}"


def _discovery_candidate_eligible(
    row: Dict[str, Any],
    seen_keys: Set[str],
    monitoring_state_keys: Set[str],
    portfolio_state_keys: Set[str],
    recent_emitted_keys: Set[str],
) -> Tuple[bool, str, str]:
    chain = normalize_chain_name(token_chain(row))
    ca = addr_store(chain, token_ca(row))
    entity_key = canonical_entity_key(chain, ca)
    if not chain:
        return False, "invalid_chain", entity_key
    if not ca:
        return False, "invalid_ca", entity_key
    dex_url = str(row.get("dex_url") or row.get("dexscreener_url") or dex_url_for_token(chain, ca)).strip()
    if not dex_url:
        return False, "missing_dex_url", entity_key
    if is_token_suppressed(chain, ca):
        return False, "suppressed", entity_key

    weak_reason = str(row.get("weak_reason") or row.get("entry_reason") or "").strip().lower()
    lifecycle = str(row.get("lifecycle_state") or row.get("status") or "").strip().lower()
    if any(flag in weak_reason for flag in ("dead", "cold", "archived")):
        return False, "dead_or_cold", entity_key
    if any(flag in lifecycle for flag in ("dead", "cold", "archived")):
        return False, "dead_or_cold", entity_key

    if entity_key in seen_keys:
        return False, "dedup", entity_key
    if entity_key in monitoring_state_keys or entity_key in portfolio_state_keys:
        return False, "already_in_ui_truth", entity_key
    if entity_key in recent_emitted_keys:
        return False, "cooldown", entity_key
    return True, "ok", entity_key


def build_discovery_candidate_pool(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    limit: int = 60,
) -> List[Dict[str, Any]]:
    _ = portfolio_rows
    ui_truth_rows = _ui_monitoring_source_rows(monitoring_rows) + _ui_portfolio_source_rows(portfolio_rows)
    ui_truth_keys = {canonical_token_key(r) for r in ui_truth_rows if canonical_token_key(r)}

    key_to_row: Dict[str, Dict[str, Any]] = {}
    for row in monitoring_rows:
        key = canonical_token_key(row)
        if key:
            key_to_row[key] = row

    scan_state = scanner_state_load() or {}
    queue_state = scan_state.get("queue_state", {}) if isinstance(scan_state.get("queue_state"), dict) else {}
    runtime = scan_state.get("worker_runtime", {}) if isinstance(scan_state.get("worker_runtime"), dict) else {}
    ordered_keys: List[str] = []
    for key in list(runtime.get("last_candidate_keys", []) or []):
        key_str = str(key or "").strip()
        if key_str:
            ordered_keys.append(key_str.replace(":", "|"))
    for key in queue_state.keys():
        key_str = str(key or "").strip()
        if key_str:
            ordered_keys.append(key_str.replace(":", "|"))

    # Backfill from backend rows (not UI truth) sorted by score freshness.
    backend_rows = sorted(
        [r for r in monitoring_rows if canonical_token_key(r) and canonical_token_key(r) not in ui_truth_keys],
        key=lambda r: parse_float(r.get("entry_score", r.get("score", 0)), 0.0),
        reverse=True,
    )
    for row in backend_rows:
        ordered_keys.append(canonical_token_key(row))

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for key in ordered_keys:
        if key in seen or key in ui_truth_keys:
            continue
        row = key_to_row.get(key)
        if not row:
            continue
        seen.add(key)
        out.append(dict(row))
        if len(out) >= max(1, int(limit or 60)):
            break
    return out


def _ui_monitoring_source_rows(monitoring_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_rows: List[Dict[str, Any]] = []
    for r in monitoring_rows:
        chain = str(r.get("chain") or "").lower().strip()
        symbol = str(r.get("base_symbol") or "").upper().strip()
        if chain not in ALLOWED_CHAINS:
            continue
        if symbol in HARD_BLOCK_SYMBOLS:
            continue
        clean_rows.append(r)

    normalized_rows: List[Dict[str, Any]] = []
    for raw in clean_rows:
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
        normalized_rows.append(row)

    _mon_set, active_set = active_base_sets()
    active_rows: List[Dict[str, Any]] = []
    for r in normalized_rows:
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
        active_rows = normalized_rows[:50]
    return active_rows


def _ui_monitoring_visible_rows(active_rows: List[Dict[str, Any]], chain_filter: str = "all") -> List[Dict[str, Any]]:
    if chain_filter != "all":
        filtered = [r for r in active_rows if (r.get("chain") or "").strip().lower() == chain_filter]
    else:
        filtered = list(active_rows)

    best_by_symbol: Dict[str, Dict[str, Any]] = {}
    for t in filtered:
        symbol = str(t.get("symbol") or t.get("base_symbol") or "").upper().strip()
        if not symbol:
            continue
        score = parse_float(t.get("score", t.get("priority_score", 0)), 0.0)
        prev = best_by_symbol.get(symbol)
        if prev is None:
            best_by_symbol[symbol] = t
            continue
        prev_score = parse_float(prev.get("score", prev.get("priority_score", 0)), 0.0)
        if score > prev_score:
            best_by_symbol[symbol] = t
    return list(best_by_symbol.values())


def _ui_portfolio_source_rows(portfolio_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in portfolio_rows if str(r.get("active", "1")).strip() == "1"]


def _extract_recent_token_keys_from_tg_state(state: Dict[str, Any], now_ts: float, ttl_seconds: int) -> Set[str]:
    out: Set[str] = set()
    sent_emissions = state.get("sent_emissions", {})
    if not isinstance(sent_emissions, dict):
        return out
    for emission_key, ts_val in sent_emissions.items():
        try:
            ts_float = float(ts_val or 0.0)
        except Exception:
            continue
        if ts_float <= 0 or (now_ts - ts_float) > ttl_seconds:
            continue
        emission_text = str(emission_key or "")
        parts = emission_text.split("|")
        if len(parts) >= 2 and parts[1]:
            out.add(parts[1])
    return out


def build_digest_sources(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    top_monitoring_limit: int,
    digest_path: str,
) -> Dict[str, Any]:
    mode = "ui_truth" if digest_path == "digest_ui" else "backend_candidates"
    ui_monitoring_source_rows = _ui_monitoring_source_rows(monitoring_rows)
    ui_portfolio_source_rows = _ui_portfolio_source_rows(portfolio_rows)
    ui_source_rows = ui_monitoring_source_rows + ui_portfolio_source_rows
    ui_visible_rows = _ui_monitoring_visible_rows(ui_monitoring_source_rows) + ui_portfolio_source_rows
    ui_source_keys = {canonical_token_key(r) for r in ui_source_rows if canonical_token_key(r)}
    ui_visible_keys = {canonical_token_key(r) for r in ui_visible_rows if canonical_token_key(r)}

    now_ts = time.time()
    state = load_tg_state()
    scan_state = scanner_state_load()
    discovery_pool = build_discovery_candidate_pool(
        monitoring_rows=monitoring_rows,
        portfolio_rows=portfolio_rows,
        limit=max(30, int(top_monitoring_limit or 3) * 10),
    )
    backend_candidate_keys = {canonical_token_key(r) for r in discovery_pool if canonical_token_key(r)}
    monitoring_active_keys = {canonical_token_key(r) for r in build_active_monitoring_rows(monitoring_rows) if canonical_token_key(r)}
    portfolio_active_keys = {canonical_token_key(r) for r in ui_portfolio_source_rows if canonical_token_key(r)}
    tg_recent_keys = _extract_recent_token_keys_from_tg_state(state, now_ts=now_ts, ttl_seconds=DIGEST_BACKEND_REUSE_TTL_SEC)
    scan_runtime = scan_state.get("worker_runtime", {}) if isinstance(scan_state.get("worker_runtime"), dict) else {}
    scan_state_keys = set(scan_runtime.get("last_candidate_keys", []) or []) if isinstance(scan_runtime, dict) else set()
    scan_state_keys = {str(k) for k in scan_state_keys if str(k)}

    digest_source_keys: Set[str]
    if mode == "ui_truth":
        digest_source_keys = set(ui_source_keys)
    else:
        digest_source_keys = set(backend_candidate_keys | tg_recent_keys | scan_state_keys)
    origin_map: Dict[str, str] = {}
    for key in digest_source_keys:
        if key in ui_source_keys:
            if key in portfolio_active_keys:
                origin_map[key] = "portfolio_active"
            else:
                origin_map[key] = "monitoring_active"
        elif key in backend_candidate_keys:
            origin_map[key] = "backend_candidate_recent"
        elif key in scan_state_keys:
            origin_map[key] = "scan_state_only"
        elif key in tg_recent_keys:
            origin_map[key] = "tg_state_reused"
        else:
            origin_map[key] = "unknown"

    only_in_digest = sorted(digest_source_keys - ui_source_keys)
    only_in_ui = sorted(ui_source_keys - digest_source_keys)
    debug_payload = {
        "digest_source_mode": mode,
        "digest_source_keys": sorted(digest_source_keys),
        "ui_source_keys": sorted(ui_source_keys),
        "ui_visible_keys": sorted(ui_visible_keys),
        "only_in_digest": only_in_digest,
        "only_in_ui": only_in_ui,
        "digest_token_origins": {k: origin_map.get(k, "unknown") for k in sorted(digest_source_keys)},
        "tg_state_recent_keys": sorted(tg_recent_keys),
        "scan_state_candidate_keys": sorted(scan_state_keys),
        "scan_state_last_run_ts": str(scan_state.get("last_run_ts") or ""),
        "tg_state_last_digest_sent_at": str(state.get("last_digest_sent_at") or ""),
    }
    debug_log(f"digest_source_compare {safe_json(debug_payload)}")
    update_worker_runtime_state(updates={"last_digest_source_compare": debug_payload})
    return {
        "mode": mode,
        "digest_source_keys": digest_source_keys,
        "ui_source_keys": ui_source_keys,
        "ui_visible_keys": ui_visible_keys,
        "only_in_digest": only_in_digest,
        "only_in_ui": only_in_ui,
        "origin_map": origin_map,
        "debug_payload": debug_payload,
    }


def build_digest_summary(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    digest_path: str = "digest_ui",
    top_monitoring_limit: int = 3,
    top_risk_limit: int = 3,
) -> Dict[str, Any]:
    source_snapshot = build_digest_sources(
        monitoring_rows,
        portfolio_rows,
        top_monitoring_limit=top_monitoring_limit,
        digest_path=digest_path,
    )
    active_monitoring = build_active_monitoring_rows(monitoring_rows)
    active_portfolio = _ui_portfolio_source_rows(portfolio_rows)
    if digest_path == "digest_discovery":
        mon_candidates = build_discovery_candidate_pool(
            monitoring_rows=monitoring_rows,
            portfolio_rows=portfolio_rows,
            limit=max(20, int(top_monitoring_limit or 3) * 8),
        )
    else:
        mon_candidates, _ = build_notification_candidates(
            monitoring_rows,
            portfolio_rows,
            limit_monitoring=max(5, int(top_monitoring_limit or 3)),
            limit_portfolio=5,
        )
    allowed_keys = source_snapshot.get("ui_source_keys", set()) if source_snapshot.get("mode") == "ui_truth" else source_snapshot.get("digest_source_keys", set())
    mon_candidates = [row for row in mon_candidates if canonical_token_key(row) in allowed_keys]
    active_monitoring = [row for row in active_monitoring if canonical_token_key(row) in allowed_keys]

    top_monitoring_now: List[str] = []
    for row in mon_candidates[: max(1, int(top_monitoring_limit or 3))]:
        symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
        semantics = build_notification_semantics(row, source_context="digest_summary_monitoring")
        timing = normalize_timing_label(str(row.get("timing_label") or "NEUTRAL"))
        top_monitoring_now.append(f"{symbol}: {semantics['score_display']} / {timing}")

    portfolio_sorted = sorted(
        active_portfolio,
        key=lambda r: (
            _risk_priority_value(r),
            parse_float(r.get("entry_score", 0), 0.0),
        ),
        reverse=True,
    )
    top_portfolio_risks: List[str] = []
    for row in portfolio_sorted[: max(1, int(top_risk_limit or 3))]:
        symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
        risk = str(row.get("risk_level") or row.get("risk") or "UNKNOWN").upper()
        action = str(row.get("entry_action") or row.get("entry") or "HOLD").upper()
        semantics = build_notification_semantics(row, source_context="digest_summary_portfolio")
        top_portfolio_risks.append(f"{symbol}: risk {risk}, action {action}, score {semantics['score_display']}")

    dead_cold_warnings: List[str] = []
    for row in active_portfolio + active_monitoring:
        symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
        weak_reason = str(row.get("weak_reason") or "").lower()
        entry_reason = str(row.get("entry_reason") or row.get("signal_reason") or "").lower()
        risk = str(row.get("risk_level") or row.get("risk") or "").upper()
        if ("dead" in weak_reason) or ("cold" in weak_reason) or ("dead" in entry_reason) or ("cold" in entry_reason):
            dead_cold_warnings.append(f"{symbol}: weak={weak_reason or entry_reason}")
        elif risk == "HIGH" and ("liq" in entry_reason or "rug" in entry_reason):
            dead_cold_warnings.append(f"{symbol}: {entry_reason}")
        if len(dead_cold_warnings) >= 3:
            break

    return {
        "event_type": resolve_event_type("DIGEST"),
        "digest_path": digest_path,
        "alert_tier": resolve_alert_tier("DIGEST_ONLY"),
        "digest_source_mode": source_snapshot.get("mode"),
        "active_portfolio_count": len(active_portfolio),
        "active_monitoring_count": len(active_monitoring),
        "top_monitoring_now": top_monitoring_now,
        "top_portfolio_risks": top_portfolio_risks,
        "dead_cold_warnings": dead_cold_warnings,
        "digest_debug": source_snapshot.get("debug_payload", {}),
        "digest_token_origins": source_snapshot.get("origin_map", {}),
    }


def format_digest_summary_message(digest: Dict[str, Any], trigger_source: str = "manual") -> str:
    top_monitoring = digest.get("top_monitoring_now") or []
    top_risks = digest.get("top_portfolio_risks") or []
    warnings = digest.get("dead_cold_warnings") or []

    monitoring_block = "\n".join([f"• {line}" for line in top_monitoring]) if top_monitoring else "• no active candidates yet"
    risk_block = "\n".join([f"• {line}" for line in top_risks]) if top_risks else "• no active portfolio risks"
    warning_block = "\n".join([f"• {line}" for line in warnings]) if warnings else "• none"
    source_mode = str(digest.get("digest_source_mode") or DIGEST_SOURCE_MODE_DEFAULT)
    debug_payload = digest.get("digest_debug", {}) if isinstance(digest.get("digest_debug"), dict) else {}
    only_in_digest = list(debug_payload.get("only_in_digest", []) or [])
    only_in_digest_block = ""
    if only_in_digest:
        origins = digest.get("digest_token_origins", {}) if isinstance(digest.get("digest_token_origins"), dict) else {}
        lines = []
        for token_key in only_in_digest[:6]:
            lines.append(f"• {token_key} ({origins.get(token_key, 'non_ui_origin')})")
        only_in_digest_block = "\n\n<b>Non-UI digest tokens</b>\n" + "\n".join(lines)
    backend_disclaimer = ""
    if source_mode == "backend_candidates":
        backend_disclaimer = (
            "⚠ backend candidates, not current UI list\n"
        )

    digest_path = str(digest.get("digest_path") or "digest_ui")
    digest_label = "digest_ui" if digest_path == "digest_ui" else "digest_discovery"
    return (
        f"<b>DEX Scout {digest_label}</b>\n"
        f"event_type: <code>{digest.get('event_type', 'digest')}</code>\n"
        f"trigger: {str(trigger_source or 'manual').upper()}\n"
        f"source_mode: {source_mode}\n"
        f"{backend_disclaimer}"
        f"portfolio active: {int(digest.get('active_portfolio_count', 0) or 0)}\n"
        f"monitoring active: {int(digest.get('active_monitoring_count', 0) or 0)}\n\n"
        f"<b>Top monitoring now</b>\n{monitoring_block}\n\n"
        f"<b>Top portfolio risks</b>\n{risk_block}\n\n"
        f"<b>Dead/cold warnings</b>\n{warning_block}"
        f"{only_in_digest_block}"
    )


def _token_key_from_parts(chain: str, ca: str) -> str:
    return canonical_entity_key(chain, ca)


def _build_token_state_sets(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
) -> Tuple[Set[str], Set[str]]:
    monitoring_keys: Set[str] = set()
    portfolio_keys: Set[str] = set()
    for row in build_active_monitoring_rows(monitoring_rows):
        key = _token_key_from_parts(token_chain(row), token_ca(row))
        if key:
            monitoring_keys.add(key)
    for row in portfolio_rows:
        if str(row.get("active", "1")).strip() != "1":
            continue
        key = _token_key_from_parts(token_chain(row), token_ca(row))
        if key:
            portfolio_keys.add(key)
    return monitoring_keys, portfolio_keys


def build_digest_token_blocks(
    digest: Dict[str, Any],
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    digest_path: str = "digest_ui",
    top_monitoring_limit: int = 3,
    top_risk_limit: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    source_snapshot = digest.get("digest_debug", {}) if isinstance(digest.get("digest_debug"), dict) else {}
    source_mode = str(digest.get("digest_source_mode") or DIGEST_SOURCE_MODE_DEFAULT)
    allowed_keys = set(source_snapshot.get("ui_source_keys", []) or []) if source_mode == "ui_truth" else set(source_snapshot.get("digest_source_keys", []) or [])

    mon_candidates, _ = build_notification_candidates(
        monitoring_rows,
        portfolio_rows,
        limit_monitoring=max(5, int(top_monitoring_limit or 3)),
        limit_portfolio=5,
    )
    portfolio_sorted = sorted(
        [r for r in portfolio_rows if str(r.get("active", "1")).strip() == "1"],
        key=lambda r: (
            _risk_priority_value(r),
            parse_float(r.get("entry_score", 0), 0.0),
        ),
        reverse=True,
    )

    monitoring_state_keys, portfolio_state_keys = _build_token_state_sets(monitoring_rows, portfolio_rows)
    tg_state = load_tg_state()
    recent_emitted_keys = _extract_recent_token_keys_from_tg_state(
        tg_state if isinstance(tg_state, dict) else {},
        now_ts=time.time(),
        ttl_seconds=DIGEST_BACKEND_REUSE_TTL_SEC,
    )
    blocks: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    skipped_missing_identity = 0
    skipped_not_allowed = 0

    def append_block(row: Dict[str, Any], source: str) -> None:
        nonlocal skipped_missing_identity, skipped_not_allowed
        chain = normalize_chain_name(token_chain(row))
        ca = addr_store(chain, token_ca(row))
        if not chain or not ca:
            skipped_missing_identity += 1
            return
        token_key = f"{chain}|{ca}"
        if allowed_keys and token_key not in allowed_keys:
            skipped_not_allowed += 1
            return
        if token_key in seen:
            return
        seen.add(token_key)

        row_norm = dict(row)
        row_norm["chain"] = chain
        row_norm["base_addr"] = ca
        row_norm["ca"] = ca
        row_norm["in_portfolio"] = "1" if token_key in portfolio_state_keys else "0"
        row_norm["in_monitoring"] = "1" if token_key in monitoring_state_keys else "0"
        row_norm["digest_source"] = source
        blocks.append(row_norm)

    if digest_path == "digest_discovery":
        eligible_rejects = 0
        discovery_pool = build_discovery_candidate_pool(
            monitoring_rows=monitoring_rows,
            portfolio_rows=portfolio_rows,
            limit=max(20, int(top_monitoring_limit or 3) * 8),
        )
        for row in discovery_pool:
            ok, reason, entity_key = _discovery_candidate_eligible(
                row,
                seen_keys=seen,
                monitoring_state_keys=monitoring_state_keys,
                portfolio_state_keys=portfolio_state_keys,
                recent_emitted_keys=recent_emitted_keys,
            )
            if not ok:
                eligible_rejects += 1
                continue
            candidate = dict(row)
            candidate["entity_key"] = entity_key
            candidate["origin_label"] = str(row.get("origin_label") or row.get("source") or "scanner_candidate")
            candidate["reason_short"] = str(row.get("reason_short") or row.get("entry_reason") or "scanner discovery")
            candidate["timing"] = normalize_timing_label(str(row.get("timing") or row.get("timing_label") or "NEUTRAL"))
            candidate["dex_url"] = str(row.get("dex_url") or row.get("dexscreener_url") or dex_url_for_token(token_chain(row), token_ca(row)))
            candidate["source_marker"] = "digest_discovery"
            append_block(candidate, "discovery")
            if len(blocks) >= max(1, int(top_monitoring_limit or 3)):
                break
        stats = {
            "token_blocks": len(blocks),
            "skipped_missing_identity": skipped_missing_identity,
            "skipped_not_allowed": skipped_not_allowed + eligible_rejects,
        }
        return blocks, stats

    for row in mon_candidates[: max(1, int(top_monitoring_limit or 3))]:
        append_block(row, "monitoring")
    for row in portfolio_sorted[: max(1, int(top_risk_limit or 3))]:
        append_block(row, "portfolio")

    stats = {
        "token_blocks": len(blocks),
        "skipped_missing_identity": skipped_missing_identity,
        "skipped_not_allowed": skipped_not_allowed,
    }
    return blocks, stats


def build_digest_emission_key(cooldown_seconds: int, now_ts: Optional[float] = None) -> str:
    ts = float(now_ts if now_ts is not None else time.time())
    bucket = emission_cooldown_bucket(ts, cooldown_seconds)
    return f"digest|{bucket}"


def is_duplicate_digest_emission(state: Dict[str, Any], emission_key: str, now_ts: float, cooldown_seconds: int) -> bool:
    sent = state.get("sent_digest_emissions", {})
    if not isinstance(sent, dict):
        sent = {}
    state["sent_digest_emissions"] = sent
    last_ts = float(sent.get(emission_key, 0.0) or 0.0)
    if not last_ts:
        return False
    return (now_ts - last_ts) < max(60, int(cooldown_seconds or 0))


def mark_digest_emission_sent(state: Dict[str, Any], emission_key: str, now_ts: float) -> None:
    sent = state.get("sent_digest_emissions", {})
    if not isinstance(sent, dict):
        sent = {}
    sent[emission_key] = float(now_ts)
    state["sent_digest_emissions"] = sent


def _stable_digest_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_digest_value(value[k]) for k in sorted(value.keys(), key=lambda x: str(x))}
    if isinstance(value, list):
        normalized = [_stable_digest_value(v) for v in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    if isinstance(value, tuple):
        return _stable_digest_value(list(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest_fingerprint(payload: Dict[str, Any]) -> str:
    normalized = _stable_digest_value(payload)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:24]


def _resolve_digest_path(trigger_source: str) -> str:
    source = str(trigger_source or "").strip().lower()
    if "summary" in source or source.startswith("ui_") or source.endswith("_ui"):
        return "digest_ui"
    return "digest_discovery"


def _build_ui_digest_fingerprint_payload(digest: Dict[str, Any]) -> Dict[str, Any]:
    debug_payload = digest.get("digest_debug", {}) if isinstance(digest.get("digest_debug"), dict) else {}
    only_in_digest = list(debug_payload.get("only_in_digest", []) or [])
    origins = digest.get("digest_token_origins", {}) if isinstance(digest.get("digest_token_origins"), dict) else {}
    return {
        "event_type": str(digest.get("event_type") or resolve_event_type("DIGEST")),
        "source_mode": str(digest.get("digest_source_mode") or DIGEST_SOURCE_MODE_DEFAULT),
        "active_portfolio_count": int(digest.get("active_portfolio_count", 0) or 0),
        "active_monitoring_count": int(digest.get("active_monitoring_count", 0) or 0),
        "top_monitoring_now": sorted([str(x).strip() for x in list(digest.get("top_monitoring_now") or []) if str(x).strip()]),
        "top_portfolio_risks": sorted([str(x).strip() for x in list(digest.get("top_portfolio_risks") or []) if str(x).strip()]),
        "dead_cold_warnings": sorted([str(x).strip() for x in list(digest.get("dead_cold_warnings") or []) if str(x).strip()]),
        "warning_blocks": sorted([str(x).strip() for x in list(digest.get("dead_cold_warnings") or []) if str(x).strip()]),
        "only_in_digest": sorted([str(x).strip() for x in only_in_digest if str(x).strip()]),
        "only_in_digest_origins": {str(k): str(origins.get(k, "")) for k in sorted(origins.keys(), key=lambda x: str(x)) if str(k) in only_in_digest},
    }


def _build_discovery_digest_fingerprint_payload(
    digest: Dict[str, Any],
    token_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for row in token_blocks:
        chain = normalize_chain_name(token_chain(row))
        ca = addr_store(chain, token_ca(row))
        if not chain or not ca:
            continue
        semantics = build_notification_semantics(row, source_context="digest_fingerprint_discovery")
        items.append(
            {
                "token_key": f"{chain}|{ca}",
                "symbol": str(row.get("base_symbol") or row.get("symbol") or "").strip().upper(),
                "source": str(row.get("digest_source") or "").strip().lower(),
                "score": str(semantics.get("score_display") or "N/A"),
                "risk": str(row.get("risk_level") or row.get("risk") or "UNKNOWN").strip().upper(),
                "action": str(row.get("entry_action") or row.get("entry") or "HOLD").strip().upper(),
                "timing": normalize_timing_label(str(row.get("timing_label") or "NEUTRAL")),
                "in_portfolio": "1" if str(row.get("in_portfolio", "0")).strip() == "1" else "0",
                "in_monitoring": "1" if str(row.get("in_monitoring", "0")).strip() == "1" else "0",
            }
        )

    return {
        "event_type": str(digest.get("event_type") or resolve_event_type("DIGEST")),
        "source_mode": str(digest.get("digest_source_mode") or DIGEST_SOURCE_MODE_DEFAULT),
        "active_portfolio_count": int(digest.get("active_portfolio_count", 0) or 0),
        "active_monitoring_count": int(digest.get("active_monitoring_count", 0) or 0),
        "normalized_top_items": items,
        "warning_blocks": sorted([str(x).strip() for x in list(digest.get("dead_cold_warnings") or []) if str(x).strip()]),
    }


def _heartbeat_due(last_ts_raw: Any, now_ts: float, heartbeat_hours: int) -> bool:
    heartbeat_seconds = max(3600, int(heartbeat_hours or 0) * 3600)
    last_dt = parse_ts(last_ts_raw)
    if not last_dt:
        return False
    return (now_ts - last_dt.timestamp()) >= heartbeat_seconds


def trigger_digest_notification(
    trigger_source: str = "manual",
    cooldown_seconds: int = 3600,
    force: bool = False,
) -> Dict[str, Any]:
    state = load_tg_state()
    monitoring_rows = load_monitoring()
    portfolio_rows = load_portfolio()
    now_ts = time.time()
    emission_key = build_digest_emission_key(cooldown_seconds=cooldown_seconds, now_ts=now_ts)

    if (not force) and is_duplicate_digest_emission(
        state,
        emission_key=emission_key,
        now_ts=now_ts,
        cooldown_seconds=cooldown_seconds,
    ):
        return {
            "ok": True,
            "sent": False,
            "duplicate": True,
            "event_type": resolve_event_type("DIGEST"),
        }

    digest_path = _resolve_digest_path(trigger_source)
    digest = build_digest_summary(monitoring_rows, portfolio_rows, digest_path=digest_path)
    fingerprint_field = "last_ui_digest_fingerprint" if digest_path == "digest_ui" else "last_discovery_digest_fingerprint"
    last_ts_field = "last_ui_digest_ts" if digest_path == "digest_ui" else "last_discovery_digest_ts"
    heartbeat_hours = DIGEST_UI_HEARTBEAT_HOURS if digest_path == "digest_ui" else DIGEST_DISCOVERY_HEARTBEAT_HOURS

    token_blocks, block_stats = build_digest_token_blocks(
        digest=digest,
        monitoring_rows=monitoring_rows,
        portfolio_rows=portfolio_rows,
        digest_path=digest_path,
    )
    fingerprint_payload = (
        _build_ui_digest_fingerprint_payload(digest)
        if digest_path == "digest_ui"
        else _build_discovery_digest_fingerprint_payload(digest, token_blocks)
    )
    new_fingerprint = _digest_fingerprint(fingerprint_payload)
    prev_fingerprint = str(state.get(fingerprint_field) or "")
    heartbeat_due = _heartbeat_due(state.get(last_ts_field), now_ts=now_ts, heartbeat_hours=heartbeat_hours)
    fingerprint_changed = new_fingerprint != prev_fingerprint
    emit_reason = ""
    if fingerprint_changed:
        emit_reason = "fingerprint_changed"
    elif heartbeat_due:
        emit_reason = "forced_heartbeat"
    else:
        debug_log(
            f"digest_{digest_path}_suppressed reason=no_meaningful_change "
            f"path={digest_path} trigger={trigger_source} "
            f"fingerprint={new_fingerprint} prev_fingerprint={prev_fingerprint}"
        )
        return {
            "ok": True,
            "sent": False,
            "duplicate": False,
            "event_type": resolve_event_type("DIGEST"),
            "suppressed_reason": "no_meaningful_change",
            "digest_path": digest_path,
        }

    text = format_digest_summary_message(digest, trigger_source=trigger_source)
    summary_ok = send_telegram(text, parse_mode="HTML")
    block_sent = 0
    for row in token_blocks:
        msg = format_digest_message(row, source=str(row.get("digest_source") or "monitoring"))
        if send_telegram(msg, parse_mode="HTML", reply_markup=tg_buttons(row)):
            block_sent += 1
    ok = bool(summary_ok and (block_sent == len(token_blocks)))
    digest["delivery_debug"] = {
        **block_stats,
        "summary_sent": bool(summary_ok),
        "token_blocks_sent": int(block_sent),
        "chunk_messages": int(block_sent),
        "digest_path": digest_path,
        "emit_reason": emit_reason,
        "fingerprint": new_fingerprint,
        "heartbeat_due": bool(heartbeat_due),
    }
    update_worker_runtime_state(updates={"last_digest_delivery_debug": digest.get("delivery_debug", {})})
    if ok:
        mark_digest_emission_sent(state, emission_key=emission_key, now_ts=now_ts)
        state["last_digest_sent_at"] = now_utc_str()
        state[fingerprint_field] = new_fingerprint
        state[last_ts_field] = now_utc_str()
        save_tg_state(state)
        debug_log(
            f"digest_{digest_path}_emitted reason={emit_reason} "
            f"path={digest_path} trigger={trigger_source} "
            f"fingerprint={new_fingerprint} prev_fingerprint={prev_fingerprint}"
        )

    return {
        "ok": bool(ok),
        "sent": bool(ok),
        "duplicate": False,
        "event_type": digest.get("event_type", resolve_event_type("DIGEST")),
        "digest": digest,
    }


def format_health_warning_message(row: Dict[str, Any], source: str, unified: Optional[Dict[str, Any]] = None) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "TOKEN").strip()
    chain = token_chain(row).upper()
    addr = token_ca(row)
    unified_reco = unified or compute_unified_recommendation(row, source=source)
    reason = str(unified_reco.get("final_reason") or row.get("entry_reason") or "health_warning")
    return (
        f"<b>HEALTH WARNING</b> | <b>{symbol}</b>\n"
        f"source: {str(source or 'monitoring').upper()}\n"
        f"chain: {chain}\n"
        f"reason: {reason}\n\n"
        f"CA:\n<code>{addr}</code>"
    )


def format_signal_message(
    row: Dict[str, Any],
    signal: Dict[str, str],
    source: str,
    event_type: str,
    unified: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    event_type_key = str(event_type or "").strip().lower()
    formatter_map = {
        "entry_alert": lambda: format_entry_alert_message(row, signal, source),
        "portfolio_action": lambda: format_portfolio_action_message(row),
        "state_transition": lambda: format_state_transition_message(row, source),
        "digest": lambda: format_digest_message(row, source),
        "health_warning": lambda: format_health_warning_message(row, source, unified=unified),
    }
    formatter = formatter_map.get(event_type_key)
    if not formatter:
        return None
    return formatter()


def tg_buttons(row: Dict[str, Any]) -> Dict[str, Any]:
    chain = normalize_chain_name(token_chain(row))
    ca = addr_store(chain, token_ca(row))
    dex_url = str(row.get("dex_url") or row.get("dexscreener_url") or dex_url_for_token(chain, ca)).strip()

    if not chain or not ca:
        return {"inline_keyboard": []}

    in_portfolio = str(row.get("in_portfolio", "0")).strip() == "1"
    in_monitoring = str(row.get("in_monitoring", "0")).strip() == "1"
    action_row: List[Dict[str, str]] = []
    if not in_portfolio:
        action_row.append({"text": "➕ Add to Portfolio", "callback_data": f"pf_add|{chain}|{ca}"})
    if not in_monitoring:
        action_row.append({"text": "👀 Monitor", "callback_data": f"mon_add|{chain}|{ca}"})

    buttons: List[List[Dict[str, str]]] = []
    if action_row:
        buttons.append(action_row)
    if str(row.get("digest_source") or "").strip().lower() != "discovery":
        buttons.append([{"text": "➖ Remove", "callback_data": f"remove|{chain}|{ca}"}])

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


def _compact_notification_diagnostics(stats: Dict[str, Any]) -> Dict[str, Any]:
    blocked_map = stats.get("blocked_reasons") if isinstance(stats.get("blocked_reasons"), dict) else {}
    blocked_sorted = sorted(
        [
            (str(k), int(parse_float(v, 0.0)))
            for k, v in blocked_map.items()
            if int(parse_float(v, 0.0)) > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "pre": int(parse_float(stats.get("candidates_total_pre_filter", 0), 0.0)),
        "post": int(parse_float(stats.get("post_filter_candidates", 0), 0.0)),
        "sent": int(parse_float(stats.get("sent", 0), 0.0)),
        "send_fail": int(parse_float(stats.get("send_fail", 0), 0.0)),
        "duplicate": int(parse_float(stats.get("duplicate", 0), 0.0)),
        "top_blocked": [{k: v} for k, v in blocked_sorted[:3]],
    }


def run_auto_notifications(
    scan_state: Dict[str, Any],
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    cycle_context: Optional[Dict[str, Any]] = None,
    trigger_model: str = "ui_scan_sync",
) -> Dict[str, Any]:
    now_ts = time.time()
    cycle_context = cycle_context if isinstance(cycle_context, dict) else {}
    trigger_model = str(trigger_model or "unknown").strip().lower() or "unknown"
    state = load_tg_state()
    alert_mode = get_alert_mode(state)

    cooldown_seconds = 900 if WORKER_FAST_MODE else 1800

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
    eligible_after_candidate_build = len(mon_rows) + len(port_rows)

    candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for row in mon_rows:
        candidates.append((parse_float(row.get("entry_score", 0), 0.0), "monitoring", row))
    for row in port_rows:
        candidates.append((parse_float(row.get("entry_score", 0), 0.0), "portfolio", row))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if candidates and not tg_cooldown_ok(state, seconds=cooldown_seconds):
        block_reason = f"cooldown_active:{cooldown_seconds}s"
        update_worker_runtime_state(
            updates={
                "last_notifications_ts": now_utc_str(),
                "last_notifications_epoch": now_ts,
                "last_notification_trigger_model": trigger_model,
                "last_block_reason": block_reason,
                "last_candidate_path": str(cycle_context.get("candidate_path") or "baseline"),
                "last_cycle_status": "blocked",
                "last_fallback_reason": str(cycle_context.get("fallback_reason") or ""),
                "last_notification_counters": {
                    "before_filter": len(candidates),
                    "after_filter": 0,
                    "sent": 0,
                    "blocked": 1,
                    "send_fail": 0,
                },
                "last_notification_block_reasons": {"cooldown": 1},
            },
            increments={
                "notification_runs": 1,
                "notification_blocked_cycles": 1,
                "notification_runs_background": 1 if trigger_model == "background_worker" else 0,
                "notification_runs_ui": 1 if trigger_model.startswith("ui_") else 0,
                "notification_runs_manual": 1 if trigger_model.startswith("manual") else 0,
            },
        )
        print(
            f"[TG] cooldown active -> skip cooldown_seconds={cooldown_seconds} "
            f"mode={alert_mode} before_filter={len(candidates)}",
            flush=True,
        )
        return {"status": "blocked", "block_reason": block_reason, "stats": {"sent": 0}}


    stats = {
        "candidates_total_pre_filter": len(candidates),
        "candidates_total_post_candidate_build": eligible_after_candidate_build,
        "mode": alert_mode,
        "no_token_key": 0,
        "no_actionable_event": 0,
        "suppressed": 0,
        "mode_filtered": 0,
        "tier_filtered": 0,
        "unknown_event_type": 0,
        "duplicate": 0,
        "cooldown": 0,
        "no_msg": 0,
        "already_sent": 0,
        "sent": 0,
        "sent_by_event_type": {},
        "sent_by_alert_tier": {},
        "post_filter_candidates": 0,
        "send_fail": 0,
        "blocked_reasons": {},
    }

    for _, source, row in candidates:
        chain = token_chain(row)
        ca = token_ca(row)
        if not chain or not ca:
            continue
        if is_token_suppressed(chain, ca):
            stats["suppressed"] += 1
            stats["blocked_reasons"]["suppressed"] = int(
                stats["blocked_reasons"].get("suppressed", 0)
            ) + 1
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
            stats["no_actionable_event"] += 1
            stats["blocked_reasons"]["no_actionable_event"] = int(
                stats["blocked_reasons"].get("no_actionable_event", 0)
            ) + 1
            continue

        alert_classification = resolve_tg_alert_classification(source, row, signal, unified=unified)
        current_snapshot["event_type"] = alert_classification["event_type"]
        current_snapshot["alert_tier"] = alert_classification["alert_tier"]
        current_snapshot["emission_key_foundation"] = build_emission_key_foundation(
            source,
            row,
            signal,
            unified=unified,
        )
        event_key = signal_event_key(source, row, signal)
        emission_key = build_emission_key(
            source,
            row,
            signal,
            cooldown_seconds=cooldown_seconds,
            now_ts=now_ts,
            unified=unified,
        )

        if not should_emit_for_alert_mode(
            alert_mode,
            source,
            row,
            signal,
            classification=alert_classification,
            unified=unified,
        ):
            alert_tier_lower = str(alert_classification.get("alert_tier") or "").strip().lower()
            if (
                _normalize_alert_mode(alert_mode) == "quiet"
                and alert_tier_lower in (resolve_alert_tier("MEDIUM"), resolve_alert_tier("DIGEST_ONLY"))
            ):
                stats["tier_filtered"] += 1
                stats["blocked_reasons"]["tier_filtered"] = int(
                    stats["blocked_reasons"].get("tier_filtered", 0)
                ) + 1
                send_status = "tier_filtered"
            else:
                stats["mode_filtered"] += 1
                stats["blocked_reasons"]["mode_filtered"] = int(
                    stats["blocked_reasons"].get("mode_filtered", 0)
                ) + 1
                send_status = "mode_filtered"
            journal_actionable_event(
                row=row,
                source=source,
                signal=signal,
                alert_classification=alert_classification,
                unified=unified,
                event_key=event_key,
                emission_key=emission_key,
                send_status=send_status,
                send_note=alert_mode,
            )
            continue
        stats["post_filter_candidates"] += 1

        # allow first send even if change is None, unless event already sent
        if not change and event_key in sent_events:
            stats["already_sent"] += 1
            stats["blocked_reasons"]["duplicate"] = int(
                stats["blocked_reasons"].get("duplicate", 0)
            ) + 1
            journal_actionable_event(
                row=row,
                source=source,
                signal=signal,
                alert_classification=alert_classification,
                unified=unified,
                event_key=event_key,
                emission_key=emission_key,
                send_status="already_sent",
            )
            continue

        if is_duplicate_emission(state, emission_key, now_ts=now_ts, cooldown_seconds=cooldown_seconds):
            stats["duplicate"] += 1
            stats["cooldown"] += 1
            stats["blocked_reasons"]["cooldown"] = int(
                stats["blocked_reasons"].get("cooldown", 0)
            ) + 1
            journal_actionable_event(
                row=row,
                source=source,
                signal=signal,
                alert_classification=alert_classification,
                unified=unified,
                event_key=event_key,
                emission_key=emission_key,
                send_status="duplicate",
            )
            continue

        msg = format_signal_message(
            row,
            signal,
            source,
            event_type=alert_classification["event_type"],
            unified=unified,
        )
        if not msg:
            stats["unknown_event_type"] += 1
            stats["no_msg"] += 1
            debug_log(f"tg_skip_unknown_event_type event_type={alert_classification['event_type']}")
            journal_actionable_event(
                row=row,
                source=source,
                signal=signal,
                alert_classification=alert_classification,
                unified=unified,
                event_key=event_key,
                emission_key=emission_key,
                send_status="no_msg",
            )
            continue

        if send_telegram(msg, reply_markup=tg_buttons(row)):
            sent_events[event_key] = now_utc_str()
            mark_emission_sent(state, emission_key, now_ts=now_ts)
            sent_now += 1
            stats["sent"] += 1
            event_type_key = str(alert_classification.get("event_type") or "unknown")
            tier_key = str(alert_classification.get("alert_tier") or "unknown")
            stats["sent_by_event_type"][event_type_key] = int(stats["sent_by_event_type"].get(event_type_key, 0)) + 1
            stats["sent_by_alert_tier"][tier_key] = int(stats["sent_by_alert_tier"].get(tier_key, 0)) + 1
            journal_actionable_event(
                row=row,
                source=source,
                signal=signal,
                alert_classification=alert_classification,
                unified=unified,
                event_key=event_key,
                emission_key=emission_key,
                send_status="sent",
            )
        else:
            stats["send_fail"] += 1
            journal_actionable_event(
                row=row,
                source=source,
                signal=signal,
                alert_classification=alert_classification,
                unified=unified,
                event_key=event_key,
                emission_key=emission_key,
                send_status="send_fail",
            )

        if sent_now >= max_per_run:
            break

    print(f"[TG] notification stats: {stats}", flush=True)

    cycle_status = "sent" if stats["sent"] > 0 else "empty"
    if stats["send_fail"] > 0 and stats["sent"] == 0:
        cycle_status = "failed"
    elif stats["sent"] == 0:
        blocked_total = (
            int(stats["duplicate"])
            + int(stats["already_sent"])
            + int(stats["mode_filtered"])
            + int(stats["tier_filtered"])
            + int(stats["suppressed"])
            + int(stats["cooldown"])
            + int(stats["no_actionable_event"])
        )
        if blocked_total > 0:
            cycle_status = "blocked"

    empty_reason = ""
    block_reason = ""
    error_reason = ""
    if eligible_after_candidate_build <= 0:
        empty_reason = "empty_candidates"
    elif stats["post_filter_candidates"] <= 0:
        empty_reason = "post_filter_empty"
    if cycle_status == "blocked":
        blocked = sorted(
            [
                (k, v)
                for k, v in stats["blocked_reasons"].items()
                if int(v or 0) > 0
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        if blocked:
            block_reason = str(blocked[0][0])
        elif stats["duplicate"] > 0:
            block_reason = "duplicate"
        elif stats["cooldown"] > 0:
            block_reason = "cooldown"
        elif stats["suppressed"] > 0:
            block_reason = "suppressed"
        elif stats["mode_filtered"] > 0:
            block_reason = "mode_filtered"
        elif stats["tier_filtered"] > 0:
            block_reason = "tier_filtered"
        elif stats["no_actionable_event"] > 0:
            block_reason = "no_actionable_event"
    if stats["send_fail"] > 0:
        error_reason = f"send_fail:{stats['send_fail']}"
    diag = _compact_notification_diagnostics(stats)
    diag_reason = (
        f"pre={diag['pre']} post={diag['post']} sent={diag['sent']} "
        f"fail={diag['send_fail']} dup={diag['duplicate']}"
    )

    runtime_snapshot = get_worker_runtime_state()
    update_worker_runtime_state(
        updates={
            "last_notifications_ts": now_utc_str(),
            "last_notifications_epoch": now_ts,
            "last_notification_trigger_model": trigger_model,
            "last_send_success_ts": now_utc_str() if stats["sent"] > 0 else runtime_snapshot.get("last_send_success_ts", ""),
            "last_send_success_epoch": now_ts if stats["sent"] > 0 else runtime_snapshot.get("last_send_success_epoch", 0.0),
            "last_error_ts": now_utc_str() if error_reason else runtime_snapshot.get("last_error_ts", ""),
            "last_error_reason": error_reason or runtime_snapshot.get("last_error_reason", ""),
            "last_empty_reason": empty_reason,
            "last_block_reason": block_reason,
            "last_duplicate_reason": "duplicate_emission" if int(stats.get("duplicate", 0) or 0) > 0 else "",
            "last_send_failure_reason": error_reason if error_reason else "",
            "last_candidate_path": str(cycle_context.get("candidate_path") or "baseline"),
            "last_cycle_status": cycle_status,
            "last_fallback_reason": str(cycle_context.get("fallback_reason") or ""),
            "last_diag_summary": diag_reason,
            "last_notification_counters": {
                "before_filter": int(stats.get("candidates_total_pre_filter", 0) or 0),
                "after_filter": int(stats.get("post_filter_candidates", 0) or 0),
                "sent": int(stats.get("sent", 0) or 0),
                "blocked": int(
                    int(stats.get("duplicate", 0) or 0)
                    + int(stats.get("already_sent", 0) or 0)
                    + int(stats.get("mode_filtered", 0) or 0)
                    + int(stats.get("tier_filtered", 0) or 0)
                    + int(stats.get("suppressed", 0) or 0)
                    + int(stats.get("cooldown", 0) or 0)
                    + int(stats.get("no_actionable_event", 0) or 0)
                ),
                "send_fail": int(stats.get("send_fail", 0) or 0),
            },
            "last_notification_block_reasons": {
                k: int(v or 0)
                for k, v in stats.get("blocked_reasons", {}).items()
                if int(v or 0) > 0
            },
            "last_notification_diag": diag,
        },
        increments={
            "notification_runs": 1,
            "notification_runs_background": 1 if trigger_model == "background_worker" else 0,
            "notification_runs_ui": 1 if trigger_model.startswith("ui_") else 0,
            "notification_runs_manual": 1 if trigger_model.startswith("manual") else 0,
            "notification_sent": int(stats["sent"]),
            "notification_empty_cycles": 1 if cycle_status == "empty" else 0,
            "notification_blocked_cycles": 1 if cycle_status == "blocked" else 0,
            "notification_failed_cycles": 1 if cycle_status == "failed" else 0,
            "notification_send_failures": int(stats["send_fail"]),
        },
    )

    state["token_state"] = new_token_state
    state["sent_events"] = sent_events
    state["last_scan_ts_processed"] = str(scan_state.get("last_run_ts") or now_utc_str())
    save_tg_state(state)
    return {"status": cycle_status, "stats": stats}


def ui_notification_emission_enabled() -> bool:
    raw = os.getenv("DEX_SCOUT_UI_EMIT_NOTIFICATIONS", "0")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def runtime_liveness(runtime: Dict[str, Any]) -> Dict[str, Any]:
    now_ts = time.time()
    last_heartbeat_epoch = float(parse_float(runtime.get("last_heartbeat_epoch", 0.0), 0.0))
    heartbeat_interval_sec = max(1.0, float(parse_float(runtime.get("heartbeat_interval_sec", 0), 0.0)))
    if last_heartbeat_epoch <= 0:
        return {"status": "unknown", "age_sec": None, "stale_threshold_sec": int(heartbeat_interval_sec * 2)}
    age_sec = max(0.0, now_ts - last_heartbeat_epoch)
    stale_threshold_sec = max(120.0, heartbeat_interval_sec * 2.0)
    status = "alive" if age_sec <= stale_threshold_sec else "stale"
    return {
        "status": status,
        "age_sec": int(age_sec),
        "stale_threshold_sec": int(stale_threshold_sec),
    }


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

        transition = transition_token_state(
            infer_lifecycle_state(r),
            "REACTIVATED",
            {"requested_status": "active", "reason": "auto_revisit"},
        )
        r.update(transition.get("updated_fields", {}))
        if not transition.get("valid"):
            continue
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

        transition = transition_token_state(
            infer_lifecycle_state(r),
            "REACTIVATED",
            {"requested_status": "active", "reason": "revisit_threshold"},
        )
        r.update(transition.get("updated_fields", {}))
        if not transition.get("valid"):
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
        "last_stats": {},
        "queue_state": {},
        "queue_next_due_ts": 0.0,
        "queue_last_run_ts": "",
        "worker_runtime": {},
    }


def worker_runtime_defaults() -> Dict[str, Any]:
    return {
        "worker_id": "",
        "worker_role": "",
        "worker_status": "unknown",
        "worker_boot_count": 0,
        "worker_process_started_ts": "",
        "worker_process_started_epoch": 0.0,
        "worker_last_restart_ts": "",
        "worker_last_restart_reason": "",
        "worker_consecutive_crashes": 0,
        "worker_last_crash_ts": "",
        "worker_last_crash_reason": "",
        "last_loop_ts": "",
        "last_loop_epoch": 0.0,
        "last_heartbeat_ts": "",
        "last_heartbeat_epoch": 0.0,
        "last_notifications_ts": "",
        "last_notifications_epoch": 0.0,
        "last_notification_trigger_model": "",
        "last_send_success_ts": "",
        "last_send_success_epoch": 0.0,
        "last_error_ts": "",
        "last_error_reason": "",
        "last_send_failure_reason": "",
        "last_duplicate_reason": "",
        "last_empty_reason": "",
        "last_block_reason": "",
        "last_candidate_path": "",
        "last_cycle_status": "",
        "last_fallback_reason": "",
        "last_job_reason": "",
        "last_lock_code": "",
        "last_stale_lock_ts": "",
        "last_stale_run_ts": "",
        "last_diag_summary": "",
        "last_notification_diag": {},
        "last_notification_counters": {},
        "last_notification_block_reasons": {},
        "heartbeat_interval_sec": 0,
        "loop_iterations": 0,
        "notification_runs": 0,
        "notification_runs_background": 0,
        "notification_runs_ui": 0,
        "notification_runs_manual": 0,
        "notification_sent": 0,
        "notification_empty_cycles": 0,
        "notification_blocked_cycles": 0,
        "notification_failed_cycles": 0,
        "notification_send_failures": 0,
    }


def get_worker_runtime_state(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}
    if state is None:
        persisted, persisted_status = read_runtime_state("worker_runtime")
        if persisted_status.get("ok") and isinstance(persisted, dict) and persisted:
            runtime = persisted
        else:
            src = scanner_state_load() or {}
            runtime = src.get("worker_runtime", {}) if isinstance(src, dict) else {}
            if not persisted_status.get("ok"):
                runtime["_runtime_error"] = {
                    "code": persisted_status.get("code"),
                    "message": persisted_status.get("detail") or persisted_status.get("message"),
                }
    else:
        src = state if isinstance(state, dict) else {}
        runtime = src.get("worker_runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    out = worker_runtime_defaults()
    out.update(runtime)
    return out


def update_worker_runtime_state(
    updates: Dict[str, Any],
    increments: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    runtime, status = update_runtime_state(
        updates=updates,
        increments=increments,
        state_key="worker_runtime",
    )
    state = scanner_state_load() or {}
    if not isinstance(state, dict):
        state = {}
    state["worker_runtime"] = dict(runtime)
    scanner_state_save(state)
    merged = worker_runtime_defaults()
    merged.update(runtime)
    if not status.get("ok"):
        merged["_runtime_error"] = {
            "code": status.get("code"),
            "message": status.get("detail") or status.get("message"),
        }
        debug_log(
            f"worker_runtime_update_failed code={status.get('code')} "
            f"detail={status.get('detail') or status.get('message')}"
        )
    return merged


SCAN_QUEUE_TIERS: List[Tuple[str, int]] = [
    ("portfolio_active", 0),
    ("portfolio_linked_monitoring", 1),
    ("high_priority_monitoring", 2),
    ("review_weak_monitoring", 3),
    ("archive_revisit", 4),
]
SCAN_QUEUE_TIER_RANK = {name: rank for name, rank in SCAN_QUEUE_TIERS}


def _scan_queue_token_key(chain: str, base_addr: str) -> str:
    c = normalize_chain_name(chain or "")
    a = addr_store(c, base_addr or "")
    return f"{c}:{a}" if c and a else ""


def _scanner_queue_state(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = state.get("queue_state", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = dict(value)
    return out


def _scanner_tier_for_row(row: Dict[str, Any], portfolio_keys: Set[str]) -> str:
    chain = token_chain(row)
    ca = token_ca(row)
    key = _scan_queue_token_key(chain, ca)
    if key and key in portfolio_keys:
        return "portfolio_active"
    if str(row.get("portfolio_linked", "0")).strip() == "1":
        return "portfolio_linked_monitoring"
    lifecycle = infer_lifecycle_state(row)
    if lifecycle == LIFECYCLE_REVISIT or str(row.get("active", "1")).strip() != "1":
        return "archive_revisit"
    score = parse_float(row.get("entry_score", row.get("priority_score", 0)), 0.0)
    weak = str(row.get("weak_reason") or "").strip()
    low_flag = str(row.get("low_priority_flag", "")).strip().lower() in {"1", "true", "yes"}
    if score >= 220 and not weak and not low_flag:
        return "high_priority_monitoring"
    if score >= 120 and not low_flag:
        return "high_priority_monitoring"
    return "review_weak_monitoring"


def _scanner_interval_for_tier(tier: str, score: float) -> int:
    base = {
        "portfolio_active": 75,
        "portfolio_linked_monitoring": 120,
        "high_priority_monitoring": 180,
        "review_weak_monitoring": 420,
        "archive_revisit": 900,
    }.get(tier, 420)
    if score >= 300:
        base = int(base * 0.8)
    elif score <= 80:
        base = int(base * 1.3)
    return max(60, base)


def _row_last_freshness_sec(row: Dict[str, Any], now_dt: datetime) -> float:
    ts_candidates = [
        row.get("ts_last_seen"),
        row.get("ts_added"),
        row.get("updated_at"),
        row.get("ts_utc"),
    ]
    best: Optional[datetime] = None
    for raw in ts_candidates:
        ts = parse_ts(raw)
        if ts and (best is None or ts > best):
            best = ts
    if best is None:
        return 10**9
    return max(0.0, (now_dt - best).total_seconds())


def build_priority_scan_queue(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    state: Optional[Dict[str, Any]] = None,
    now_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    state = state or {}
    now_ts = float(now_ts if now_ts is not None else time.time())
    now_dt = datetime.utcfromtimestamp(now_ts)
    queue_state = _scanner_queue_state(state)
    portfolio_keys: Set[str] = set()
    for row in portfolio_rows:
        if str(row.get("active", "1")).strip() != "1":
            continue
        key = _scan_queue_token_key(token_chain(row), token_ca(row))
        if key:
            portfolio_keys.add(key)

    queue: List[Dict[str, Any]] = []
    for row in monitoring_rows:
        chain = token_chain(row)
        ca = token_ca(row)
        key = _scan_queue_token_key(chain, ca)
        if not key:
            continue
        tier = _scanner_tier_for_row(row, portfolio_keys)
        score = parse_float(row.get("entry_score", row.get("priority_score", 0)), 0.0)
        freshness_sec = _row_last_freshness_sec(row, now_dt)
        prev = queue_state.get(key, {})
        next_due_ts = float(prev.get("next_due_ts", 0.0) or 0.0)
        backoff_sec = int(parse_float(prev.get("backoff_sec", 0), 0.0))
        if next_due_ts <= 0:
            next_due_ts = now_ts
        due_in_sec = next_due_ts - now_ts
        queue.append({
            "token_key": key,
            "chain": chain,
            "base_addr": ca,
            "tier": tier,
            "tier_rank": SCAN_QUEUE_TIER_RANK.get(tier, 99),
            "score": score,
            "freshness_sec": freshness_sec,
            "next_due_ts": next_due_ts,
            "due_in_sec": due_in_sec,
            "backoff_sec": backoff_sec,
            "row": row,
        })

    queue.sort(
        key=lambda x: (
            x["tier_rank"],
            0 if x["due_in_sec"] <= 0 else 1,
            x["due_in_sec"],
            -x["freshness_sec"],
            -x["score"],
        )
    )
    return queue


def run_priority_scanner_cycle(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    max_scans: int = 3,
) -> Dict[str, Any]:
    now_ts = time.time()
    now_dt = datetime.utcfromtimestamp(now_ts)
    state = scanner_state_load() or {}
    queue_state = _scanner_queue_state(state)
    queue = build_priority_scan_queue(monitoring_rows, portfolio_rows, state=state, now_ts=now_ts)

    scanned = 0
    errors = 0
    due_items = [item for item in queue if item["due_in_sec"] <= 0]
    picks = due_items[: max(1, int(max_scans))]

    for item in picks:
        token_key = str(item.get("token_key") or "")
        chain = str(item.get("chain") or "")
        base_addr = str(item.get("base_addr") or "")
        tier = str(item.get("tier") or "review_weak_monitoring")
        score = parse_float(item.get("score", 0), 0.0)
        interval_sec = _scanner_interval_for_tier(tier, score)
        row_state = dict(queue_state.get(token_key, {}))

        try:
            pair = best_pair_for_token_cached(chain, base_addr)
            if pair:
                score_live = score_pair(pair)
                normalized = normalize_pair_row(pair)
                if normalized is not None:
                    update_existing_token(
                        normalized,
                        score_live,
                        window_name="adaptive_priority_queue",
                        preset_key=f"adaptive:{tier}",
                        entry_status=str(item["row"].get("entry_status") or ""),
                        entry_score=parse_float(item["row"].get("entry_score", score_live), 0.0),
                        risk_level=str(item["row"].get("risk_level", "MEDIUM")),
                    )
                    row_state["last_score"] = score_live
            row_state["last_error_ts"] = ""
            row_state["backoff_sec"] = 0
        except Exception as exc:
            log_error(exc)
            errors += 1
            current_backoff = int(parse_float(row_state.get("backoff_sec", 0), 0.0))
            row_state["backoff_sec"] = min(1800, max(60, current_backoff * 2 if current_backoff else 90))
            row_state["last_error_ts"] = now_utc_str()

        backoff_sec = int(parse_float(row_state.get("backoff_sec", 0), 0.0))
        row_state["tier"] = tier
        row_state["last_scan_ts"] = now_utc_str()
        row_state["next_due_ts"] = now_ts + interval_sec + backoff_sec
        queue_state[token_key] = row_state
        scanned += 1

    if queue:
        queue_head = min(
            (float(queue_state.get(item["token_key"], {}).get("next_due_ts", now_ts)) for item in queue),
            default=now_ts + 300,
        )
    else:
        queue_head = now_ts + 300

    state["queue_state"] = queue_state
    state["queue_size"] = len(queue)
    state["queue_due_now"] = len(due_items)
    state["queue_next_due_ts"] = float(queue_head)
    state["queue_last_run_ts"] = now_utc_str()
    state["queue_last_tiers"] = [item.get("tier") for item in picks]
    scanner_state_save(state)

    sleep_suggest = max(30, int(queue_head - now_ts))
    return {
        "queue_size": len(queue),
        "due_now": len(due_items),
        "scanned": scanned,
        "errors": errors,
        "sleep_suggest_sec": sleep_suggest,
        "tiers_scanned": [item.get("tier") for item in picks],
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
        "queue_next_due_ts": float(time.time() + 300),
        "queue_last_run_ts": now_utc_str(),
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
        "queue_next_due_ts": float(time.time() + 300),
        "queue_last_run_ts": now_utc_str(),
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


GEM_TRANSITION_NEUTRAL = 50.0
GEM_TRANSITION_MIN_HISTORY = 4
GEM_TRANSITION_MAX_AGE_MIN = 180.0


def _recent_history_for_transition(hist: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = hist or []
    if not rows:
        return []
    now_dt = datetime.utcnow()
    recent: List[Dict[str, Any]] = []
    for row in rows[-12:]:
        ts = parse_ts(row.get("ts_utc"))
        if ts is None:
            continue
        age_min = (now_dt - ts).total_seconds() / 60.0
        if 0 <= age_min <= GEM_TRANSITION_MAX_AGE_MIN:
            recent.append(row)
    return recent


def _series_acceleration(values: List[float]) -> float:
    if len(values) < 4:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    if len(diffs) < 3:
        return 0.0
    head = sum(diffs[: max(1, len(diffs) // 2)]) / max(1, len(diffs) // 2)
    tail = sum(diffs[max(1, len(diffs) // 2):]) / max(1, len(diffs) - max(1, len(diffs) // 2))
    return tail - head


def compute_gem_transition_score(
    best: Optional[Dict[str, Any]],
    hist: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    recent = _recent_history_for_transition(hist)
    if len(recent) < GEM_TRANSITION_MIN_HISTORY:
        return {
            "gem_transition_score": GEM_TRANSITION_NEUTRAL,
            "gem_transition_sufficient": False,
            "gem_transition_reason": "insufficient_recent_history",
            "gem_transition_components": {},
        }

    prices = [parse_float(x.get("price_usd", 0), 0.0) for x in recent]
    liqs = [parse_float(x.get("liq_usd", 0), 0.0) for x in recent]
    vols = [parse_float(x.get("vol5_usd", 0), 0.0) for x in recent]
    pcs = [parse_float(x.get("pc5", 0), 0.0) for x in recent]

    price_acc = _series_acceleration(prices)
    liq_acc = _series_acceleration(liqs)
    vol_acc = _series_acceleration(vols)
    pc_build = (sum(pcs[-3:]) / 3.0) if len(pcs) >= 3 else pcs[-1]

    liq_now = parse_float(safe_get(best or {}, "liquidity", "usd", default=liqs[-1] if liqs else 0.0), 0.0)
    vol_now = parse_float(safe_get(best or {}, "volume", "m5", default=vols[-1] if vols else 0.0), 0.0)
    buys5 = parse_float(safe_get(best or {}, "txns", "m5", "buys", default=0), 0.0)
    sells5 = parse_float(safe_get(best or {}, "txns", "m5", "sells", default=0), 0.0)
    txns5 = buys5 + sells5
    buy_ratio = (buys5 / txns5) if txns5 > 0 else 0.5

    score = GEM_TRANSITION_NEUTRAL
    score += max(min(price_acc * 8000.0, 14.0), -14.0)
    score += max(min((liq_acc / max(liq_now, 1.0)) * 900.0, 10.0), -10.0)
    score += max(min((vol_acc / max(vol_now, 1.0)) * 800.0, 12.0), -12.0)
    score += max(min(pc_build * 1.1, 10.0), -10.0)

    if liq_now >= 15_000 and vol_now >= 2_000:
        score += 4.0
    elif liq_now < 3_000 or vol_now < 200:
        score -= 8.0

    if buy_ratio >= 0.58 and txns5 >= 8:
        score += 4.0
    elif buy_ratio <= 0.40 and txns5 >= 8:
        score -= 5.0

    score = max(0.0, min(100.0, round(score, 2)))
    return {
        "gem_transition_score": score,
        "gem_transition_sufficient": True,
        "gem_transition_reason": "ok",
        "gem_transition_components": {
            "price_acc": round(price_acc, 8),
            "liq_acc": round(liq_acc, 4),
            "vol_acc": round(vol_acc, 4),
            "pc_build": round(pc_build, 4),
            "buy_ratio": round(buy_ratio, 4),
            "txns5": int(txns5),
        },
    }


def gem_transition_priority_bias(
    base_priority: float,
    gem_transition_score: float,
    health_label: str,
    sufficient_history: bool,
) -> float:
    base = float(base_priority)
    if not sufficient_history:
        return round(base, 2)
    if is_strong_negative_health_state(health_label):
        return round(base, 2)
    delta = max(min((gem_transition_score - GEM_TRANSITION_NEUTRAL) * 0.45, 16.0), -12.0)
    return round(max(base + delta, 0.0), 2)


def is_strong_negative_health_state(health_label: str) -> bool:
    return str(health_label or "").upper() in {"DEAD", "UNTRADEABLE", "LOW_LIQ", "COLD", "STALE", "CRITICAL"}


def entry_from_score(score: float) -> str:
    if score >= 300:
        return "READY"
    if score >= 180:
        return "WATCH"
    return "WAIT"


def compute_timing(item: Dict[str, Any]) -> str:
    pc = parse_float(item.get("price_change_5m", safe_get(item, "priceChange", "m5", default=0)), 0.0)
    gem_score = parse_float(item.get("gem_transition_score"), GEM_TRANSITION_NEUTRAL)
    gem_ok = str(item.get("gem_transition_sufficient", "")).strip().lower() in {"1", "true", "yes"}

    if pc > 10:
        return "HOT"
    if gem_ok and abs(gem_score - GEM_TRANSITION_NEUTRAL) < 1e-9:
        gem_ok = False
    if gem_ok and gem_score >= 70 and pc > 1:
        return "EARLY"
    if pc > 3:
        return "EARLY"
    if gem_ok and gem_score <= 35 and pc <= 3:
        return "NEUTRAL"
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
    if not str(row.get("gem_transition_score", "")).strip():
        row["gem_transition_score"] = str(parse_float(item.get("gem_transition_score"), GEM_TRANSITION_NEUTRAL))
    if not str(row.get("gem_transition_sufficient", "")).strip():
        row["gem_transition_sufficient"] = "1" if bool(item.get("gem_transition_sufficient")) else "0"
    if not str(row.get("gem_transition_reason", "")).strip():
        row["gem_transition_reason"] = str(item.get("gem_transition_reason") or "insufficient_recent_history")
    if row.get("portfolio_linked") == "1" and not row.get("note"):
        row["note"] = "IN PORTFOLIO"
    return row


def monitoring_row_to_card(row: Dict[str, Any]) -> Dict[str, Any]:
    chain = (row.get("chain") or "").strip().lower()
    base_addr = addr_store(chain, str(row.get("base_addr") or "").strip())
    best = best_pair_for_token_cached(chain, base_addr) if (chain and base_addr) else None
    hist = token_history_rows(chain, base_addr, limit=30)
    gem_data = compute_gem_transition_score(best, hist)

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

    health = detect_position_health(best or row, hist)
    health_label = str(health.get("health_label", "OK")).upper()
    if str(liq_health.get("level", "")).upper() == "CRITICAL":
        health_label = "CRITICAL"
    gem_score = parse_float(row.get("gem_transition_score", gem_data.get("gem_transition_score", GEM_TRANSITION_NEUTRAL)), GEM_TRANSITION_NEUTRAL)
    gem_sufficient = str(row.get("gem_transition_sufficient", gem_data.get("gem_transition_sufficient", ""))).strip().lower() in {"1", "true", "yes"}

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
        "gem_transition_score": gem_score,
        "gem_transition_sufficient": gem_sufficient,
        "gem_transition_reason": str(row.get("gem_transition_reason") or gem_data.get("gem_transition_reason") or ""),
        "health_label": health_label,
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
    health_label = str(item.get("health_label", "OK") or "OK").upper()
    gem_score = parse_float(item.get("gem_transition_score"), GEM_TRANSITION_NEUTRAL)
    gem_sufficient = bool(item.get("gem_transition_sufficient"))
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

    gem_bonus = 0.0
    if gem_sufficient and not is_strong_negative_health_state(health_label) and not is_dead and not is_rug:
        gem_bonus = max(min((gem_score - GEM_TRANSITION_NEUTRAL) * 0.22, 8.0), -6.0)
        if gem_bonus >= 3.5:
            flags.append("gem_transition")

    visible_score = max(raw_score - penalty + gem_bonus, 0.0)

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
        "gem_bonus": round(gem_bonus, 2),
        "flags": flags,
        "badge": badge,
    }


def _pulse_freshness_minutes(row: Dict[str, Any], queue_row: Optional[Dict[str, Any]] = None) -> float:
    ts_candidates = [
        row.get("ts_last_seen"),
        row.get("updated_at"),
        row.get("last_seen"),
        row.get("ts_added"),
        row.get("last_scan_ts"),
        (queue_row or {}).get("last_scan_ts"),
    ]
    parsed = [parse_ts(ts) for ts in ts_candidates]
    parsed = [x for x in parsed if x is not None]
    if not parsed:
        return 9_999.0
    age_min = (datetime.utcnow() - max(parsed)).total_seconds() / 60.0
    return max(0.0, age_min)


def _pulse_status_marker(row: Dict[str, Any]) -> str:
    risk = str(row.get("risk_level") or row.get("risk") or "").strip().upper()
    timing = normalize_timing_label(row.get("timing_label") or row.get("timing") or row.get("entry_status") or "NEUTRAL")
    score = parse_float(row.get("entry_score", row.get("priority_score", row.get("score", 0))), 0.0)
    if risk in {"HIGH", "EXTREME"}:
        return "high risk"
    if timing in {"EARLY", "VERY_EARLY"}:
        return "early"
    if score >= 220:
        return "watch"
    return "speculative"


def _pulse_card_summary_line(best: Optional[Dict[str, Any]], row: Dict[str, Any]) -> str:
    """Compact one-line snapshot for pulse cards (important metrics only)."""
    if not best:
        liq = parse_float(row.get("liq_init"), 0.0)
        vol24 = parse_float(row.get("vol24_init"), 0.0)
        score = parse_float(row.get("entry_score", row.get("score", 0.0)), 0.0)
        return f"snapshot: score {score:.1f} • liq {fmt_usd(liq)} • vol24 {fmt_usd(vol24)} • live pair pending"

    liq = parse_float(safe_get(best, "liquidity", "usd", default=0), 0.0)
    vol24 = parse_float(safe_get(best, "volume", "h24", default=0), 0.0)
    chg_h1 = parse_float(safe_get(best, "priceChange", "h1", default=0), 0.0)
    buys = int(safe_get(best, "txns", "m5", "buys", default=0) or 0)
    sells = int(safe_get(best, "txns", "m5", "sells", default=0) or 0)
    score = parse_float(row.get("entry_score", row.get("score", 0.0)), 0.0)
    return (
        f"snapshot: score {score:.1f} • liq {fmt_usd(liq)} • vol24 {fmt_usd(vol24)} • "
        f"Δ1h {chg_h1:+.1f}% • m5 txns {buys + sells}"
    )


def _pulse_card_liquidity_usd(best: Optional[Dict[str, Any]], row: Dict[str, Any]) -> float:
    liq = parse_float(safe_get(best or {}, "liquidity", "usd", default=0), 0.0)
    if liq > 0:
        return liq
    for candidate in (row.get("liq_init"), row.get("liq_usd")):
        parsed = parse_float(candidate, 0.0)
        if parsed > 0:
            return parsed
    return 0.0


def _log_pulse_sparkline_missing(reason: str, token_key: str = "") -> None:
    normalized = str(reason or "").strip().lower()
    if normalized in {"no_history", "not_enough_points", "no_score_or_price_series"}:
        debounce_key = f"{str(token_key or '').strip().lower()}|{normalized}"
        if debounce_key != f"|{normalized}":
            now_ts = time.time()
            last_ts = float(PULSE_SPARKLINE_LOG_LAST_TS.get(debounce_key, 0.0) or 0.0)
            if (now_ts - last_ts) < PULSE_SPARKLINE_LOG_DEBOUNCE_SEC:
                return
            PULSE_SPARKLINE_LOG_LAST_TS[debounce_key] = now_ts
        debug_log(f"pulse_sparkline_missing reason={normalized}")


def _pulse_card_mini_sparkline(row: Dict[str, Any], min_points: int = 3) -> Dict[str, Any]:
    chain = token_chain(row)
    base_addr = token_ca(row)
    token_key = canonical_entity_key(chain, base_addr)
    empty_payload = {
        "series": "score",
        "values": [],
        "points": 0,
        "reason": "warming_up",
        "source": "token_history",
    }

    def _coerce_row_series(raw_values: Any) -> List[float]:
        if raw_values is None:
            return []
        values = raw_values
        if isinstance(values, str):
            stripped = values.strip()
            if not stripped:
                return []
            try:
                loaded = json.loads(stripped)
                values = loaded
            except Exception:
                values = [part.strip() for part in stripped.split(",")]
        if isinstance(values, dict):
            values = list(values.values())
        if not isinstance(values, list):
            values = [values]
        return [parse_float(v, 0.0) for v in values if parse_float(v, 0.0) > 0]

    if not chain or not base_addr:
        _log_pulse_sparkline_missing("no_history", token_key=token_key)
        return empty_payload

    hist = token_history_rows(chain, base_addr, limit=40)
    series = build_history_series(hist) if hist else {}

    def _series_values(raw_values: Any) -> List[float]:
        return [parse_float(v, 0.0) for v in list(raw_values or []) if parse_float(v, 0.0) > 0]

    fallback_chain: List[Tuple[str, List[float], str]] = [
        ("score", _series_values(series.get("score")), "history_series.score"),
        ("entry_score", _series_values(series.get("entry_score")), "history_series.entry_score"),
        ("price", _series_values(series.get("price")), "history_series.price"),
        (
            "last_score",
            _series_values([h.get("last_score") for h in hist if str(h.get("last_score", "")).strip()]),
            "history_raw.last_score",
        ),
        (
            "price_usd",
            _series_values([h.get("price_usd") for h in hist if str(h.get("price_usd", "")).strip()]),
            "history_raw.price_usd",
        ),
        ("row.history_series", _coerce_row_series(row.get("history_series")), "row.history_series"),
        ("row.entry_score_trail", _coerce_row_series(row.get("entry_score_trail")), "row.entry_score_trail"),
    ]

    best_partial: Optional[Tuple[str, List[float], str]] = None
    for series_name, values, source in fallback_chain:
        points = len(values)
        if points >= min_points:
            trimmed = values[-24:]
            return {
                "series": series_name,
                "values": trimmed,
                "points": len(trimmed),
                "reason": "",
                "source": source,
            }
        if points > 0 and best_partial is None:
            best_partial = (series_name, values, source)

    if best_partial:
        series_name, values, source = best_partial
        trimmed = values[-24:]
        _log_pulse_sparkline_missing("not_enough_points", token_key=token_key)
        return {
            "series": series_name,
            "values": trimmed,
            "points": len(trimmed),
            "reason": "warming_up",
            "source": source,
        }

    _log_pulse_sparkline_missing("no_score_or_price_series", token_key=token_key)
    return {
        "series": "score",
        "values": [],
        "points": 0,
        "reason": "warming_up",
        "source": "token_history",
    }


def _pulse_pair_payload(row: Dict[str, Any], best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    chain = token_chain(row)
    ca = token_ca(row)
    symbol = str(row.get("base_symbol") or row.get("symbol") or extract_name(row) or "").strip().upper()
    payload: Dict[str, Any] = {
        "chainId": chain,
        "pairAddress": str(row.get("pair_addr") or row.get("pairAddress") or ""),
        "baseToken": {
            "address": ca,
            "symbol": symbol,
            "name": str(row.get("base_name") or row.get("name") or symbol),
        },
    }
    if best:
        payload["url"] = best.get("url") or ""
        payload["liquidity"] = {"usd": safe_get(best, "liquidity", "usd", default=0)}
        payload["volume"] = {
            "h24": safe_get(best, "volume", "h24", default=0),
            "m5": safe_get(best, "volume", "m5", default=0),
        }
        payload["priceChange"] = {
            "h1": safe_get(best, "priceChange", "h1", default=0),
            "m5": safe_get(best, "priceChange", "m5", default=0),
        }
        payload["txns"] = {"m5": safe_get(best, "txns", "m5", default={})}
    return payload


def build_market_pulse_cards(
    monitoring_rows: List[Dict[str, Any]],
    portfolio_rows: List[Dict[str, Any]],
    active_monitoring_keys: Set[str],
    chain_filter: str = "all",
    limit: int = 8,
) -> List[Dict[str, Any]]:
    scan_state = scanner_state_load() or {}
    queue_state = scan_state.get("queue_state", {}) if isinstance(scan_state.get("queue_state"), dict) else {}
    runtime = scan_state.get("worker_runtime", {}) if isinstance(scan_state.get("worker_runtime"), dict) else {}
    runtime_keys = {str(k or "").replace(":", "|") for k in list(runtime.get("last_candidate_keys", []) or []) if str(k or "").strip()}
    queue_keys = {str(k or "").replace(":", "|") for k in queue_state.keys() if str(k or "").strip()}
    discovery_pool = build_discovery_candidate_pool(
        monitoring_rows=monitoring_rows,
        portfolio_rows=portfolio_rows,
        limit=max(30, int(limit) * 8),
    )
    pulse_cards: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    low_liq_skipped = 0
    for row in discovery_pool:
        key = canonical_token_key(row)
        chain = token_chain(row)
        if not key or key in seen:
            continue
        if key in active_monitoring_keys:
            continue
        if chain_filter in {"bsc", "solana"} and chain != chain_filter:
            continue
        seen.add(key)
        queue_row = queue_state.get(key.replace("|", ":"), {}) if isinstance(queue_state, dict) else {}
        freshness_min = _pulse_freshness_minutes(row, queue_row if isinstance(queue_row, dict) else {})
        score = parse_float(row.get("entry_score", row.get("priority_score", row.get("score", 0))), 0.0)
        timing_label = normalize_timing_label(row.get("timing_label") or row.get("entry_status") or row.get("status") or "NEUTRAL")
        timing_bonus = 6.0 if timing_label in {"EARLY", "READY"} else (2.0 if timing_label == "NEUTRAL" else 0.0)
        freshness_bonus = max(0.0, 18.0 - min(freshness_min, 180.0) / 10.0)
        novelty_bonus = (8.0 if key in runtime_keys else 0.0) + (4.0 if key in queue_keys else 0.0)
        pulse_sort = score + freshness_bonus + novelty_bonus + timing_bonus
        best = best_pair_for_token_cached(chain, token_ca(row)) if chain and token_ca(row) else None
        liq_usd = _pulse_card_liquidity_usd(best if isinstance(best, dict) else None, row)
        if liq_usd < PULSE_MIN_LIQ_USD:
            low_liq_skipped += 1
            continue
        pulse_cards.append(
            {
                "row": row,
                "key": key,
                "chain": chain,
                "score": round(score, 2),
                "timing": timing_label,
                "stage": str(row.get("entry_status") or row.get("entry") or row.get("status") or "candidate").upper(),
                "status_marker": _pulse_status_marker(row),
                "freshness_min": round(freshness_min, 1),
                "pulse_sort": round(pulse_sort, 3),
                "best": best,
                "liq_usd": round(liq_usd, 2),
                "pulse_min_liq_usd": float(PULSE_MIN_LIQ_USD),
                "low_liquidity": liq_usd < PULSE_MIN_LIQ_USD,
                "summary_line": _pulse_card_summary_line(best, row),
            }
        )
    pulse_cards.sort(
        key=lambda x: (
            float(x.get("pulse_sort", 0.0)),
            float(x.get("score", 0.0)),
            -float(x.get("freshness_min", 9_999.0)),
        ),
        reverse=True,
    )
    selected_cards = pulse_cards[: max(1, int(limit or 8))]
    for card in selected_cards:
        row = card.get("row", {}) or {}
        card["mini_sparkline"] = _pulse_card_mini_sparkline(row)
    update_worker_runtime_state(
        updates={
            "last_pulse_liq_min_usd": float(PULSE_MIN_LIQ_USD),
            "last_pulse_candidates_total": int(len(discovery_pool)),
            "last_pulse_candidates_low_liq_skipped": int(low_liq_skipped),
            "last_pulse_candidates_after_liq_gate": int(len(pulse_cards)),
        }
    )
    if low_liq_skipped > 0:
        debug_log(
            f"pulse_liquidity_gate skipped={low_liq_skipped} min_liq_usd={float(PULSE_MIN_LIQ_USD):.2f}"
        )
    return selected_cards


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
    if is_debug_mode_enabled():
        st.caption(health_thresholds_debug_line())

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

    # Source of truth for Monitoring page UI: monitoring.csv filtered through _ui_monitoring_source_rows.
    monitoring_rows = load_monitoring_rows_cached()
    mon_source = st.session_state.get("_storage_source_monitoring.csv", "unknown")
    st.caption(f"MONITORING FROM DB: {len(monitoring_rows)} (source: {mon_source})")
    active_rows = _ui_monitoring_source_rows(monitoring_rows)
    rows = list(monitoring_rows)
    trust_index = load_pattern_trust_index_cached()
    for row in active_rows:
        trust_payload = compute_pattern_trust(row, trust_index=trust_index)
        row["trust_score"] = trust_payload.get("trust_score", 0.0)
        row["trust_confidence"] = trust_payload.get("trust_confidence", 0.0)
        row["trust_sample_size"] = trust_payload.get("trust_sample_size", 0)
        row["trust_reason"] = trust_payload.get("trust_reason", "")
        row["trust_pattern_key"] = trust_payload.get("pattern_key", "")
        row["trust_priority_bias"] = advisory_trust_bias(trust_payload)
    active_rows = sorted(
        active_rows,
        key=lambda row: (
            parse_float(row.get("priority_score", 0), 0.0) + parse_float(row.get("trust_priority_bias", 0), 0.0)
        ),
        reverse=True,
    )[:50]
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
    active = _ui_monitoring_visible_rows(active_rows, chain_filter=chain_filter)

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

    active_monitoring_keys = {canonical_token_key(r) for r in active_rows if canonical_token_key(r)}
    pulse_cards = build_market_pulse_cards(
        monitoring_rows=rows,
        portfolio_rows=portfolio_rows,
        active_monitoring_keys=active_monitoring_keys,
        chain_filter=chain_filter,
        limit=8,
    )

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
            final_reason = str(unified.get("final_reason", "n/a"))
            timing_label = unified.get("timing") or normalize_timing_label(item.get("timing_label") or r.get("timing_label") or "NEUTRAL")
            risk_label = str(item.get("risk_level", "UNKNOWN"))
            short_caption = f"{chain.upper()} • timing {timing_label} • risk {risk_label} • status {secondary}"
            st.markdown(f"**Recommendation: {primary}**")
            st.caption(f"Reason: {final_reason}")
            st.caption(f"Suggested size: {unified.get('size_hint', 'WATCH ONLY')}")
            st.caption(short_caption)
            if item.get("is_portfolio_active"):
                st.caption("In portfolio • still monitored")

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
            setup_inputs = build_setup_render_inputs(r)
            setup_tab, diagnostics_tab, dynamics_tab = st.tabs(
                ["Setup / levels", "Diagnostics", "Dynamics"]
            )
            with setup_tab:
                level_rows = setup_inputs.get("level_rows") or []
                if level_rows:
                    st.table(pd.DataFrame(level_rows))
                else:
                    st.caption("setup: watch only")

            with diagnostics_tab:
                st.markdown("**Decision diagnostics**")
                st.write(f"recommendation: {primary}")
                st.write(f"reason: {final_reason}")
                st.write(f"size hint: {unified.get('size_hint', 'WATCH ONLY')}")
                st.write(f"risk: {risk_label}")
                st.write(f"timing: {timing_label}")
                blocker = str(r.get("decision_reason") or r.get("gate_reason") or r.get("gate_blocker") or "").strip()
                if blocker:
                    st.write(f"gate/blocker: {blocker}")
                debug_mode = bool(st.session_state.get("debug_mode", False) or st.session_state.get("show_debug_raw", False))
                if debug_mode:
                    st.markdown("**Raw internals**")
                    st.write(f"raw score: {float(item.get('raw_live_score', 0.0)):.2f}")
                    st.write(f"penalty: {float(item.get('ui_penalty', 0.0)):.2f}")
                    st.write(f"score(raw row): {parse_float(r.get('score', 0.0), 0.0):.2f}")
                    st.write(f"reason: {final_reason}")
                    st.write(f"weak flags: {str(r.get('weak_reason') or 'none')}")
                    flags = item.get("ui_flags") or []
                    st.write("ui flags: " + (" • ".join(flags) if flags else "none"))
                    if str(r.get("invalid_transition", "0")) == "1":
                        st.error("invalid_transition")
                        st.write(
                            {
                                "from_state": r.get("invalid_from_state", ""),
                                "event": r.get("invalid_event", ""),
                                "attempted_to": r.get("invalid_attempted_to", ""),
                                "reason": r.get("invalid_reason", ""),
                            }
                        )
                else:
                    st.caption("Raw internals hidden (debug mode only).")

            hist = item.get("hist", []) or []
            with dynamics_tab:
                render_monitoring_sparklines(hist)

    st.subheader("Live pulse candidates")
    st.caption("Live candidates behind glass (read-only layer). Confirmed Monitoring below remains the settled active layer.")
    if not pulse_cards:
        st.caption("No pulse candidates right now.")
    pulse_symbol_counts: Dict[str, int] = {}
    for pulse in pulse_cards:
        pulse_row = pulse.get("row", {}) or {}
        pulse_symbol = str(
            pulse_row.get("base_symbol")
            or pulse_row.get("symbol")
            or extract_name(pulse_row)
            or "UNKNOWN"
        ).strip().upper()
        pulse_symbol_counts[pulse_symbol] = pulse_symbol_counts.get(pulse_symbol, 0) + 1
    rendered_symbol_group_label: Set[str] = set()
    for idx, pulse in enumerate(pulse_cards, start=1):
        row = pulse.get("row", {}) or {}
        chain = str(pulse.get("chain") or token_chain(row) or "").strip().lower()
        base_addr = token_ca(row)
        pair_addr = str(row.get("pair_addr") or row.get("pair_address") or row.get("pairAddress") or "").strip()
        symbol = str(row.get("base_symbol") or row.get("symbol") or extract_name(row) or "UNKNOWN").strip().upper()
        if not symbol:
            symbol = "UNKNOWN"
        score = float(pulse.get("score", 0.0))
        timing = str(pulse.get("timing") or "NEUTRAL").upper()
        stage = str(pulse.get("stage") or "CANDIDATE").upper()
        status_marker = str(pulse.get("status_marker") or "speculative")
        freshness_min = float(pulse.get("freshness_min", 9_999.0))
        card_key = str(pulse.get("key") or canonical_entity_key(chain, base_addr))
        in_monitoring = card_key in active_monitoring_keys
        dex_url = dex_url_for_token(chain, base_addr)
        best = pulse.get("best")
        liq_usd = parse_float(pulse.get("liq_usd"), 0.0)
        pulse_min_liq_usd = parse_float(pulse.get("pulse_min_liq_usd", PULSE_MIN_LIQ_USD), PULSE_MIN_LIQ_USD)
        low_liq_for_monitor = liq_usd < pulse_min_liq_usd
        if not dex_url and isinstance(best, dict):
            dex_url = str(best.get("url") or "")
        ticker_group_count = int(pulse_symbol_counts.get(symbol, 0))
        if ticker_group_count > 1 and symbol not in rendered_symbol_group_label:
            st.caption(f"same ticker, different contracts: {symbol} ({ticker_group_count})")
            rendered_symbol_group_label.add(symbol)
        header = f"{symbol} • {chain.upper()} • score {score:.1f} • ca: {short_addr(base_addr)}"
        with st.expander(header, expanded=False):
            st.caption(f"token_ca: {base_addr or 'n/a'}")
            st.caption(f"pair_addr: {pair_addr or 'n/a'}")
            st.caption(f"timing: {timing} • stage: {stage} • status: {status_marker} • freshness: {freshness_min:.1f}m")
            st.caption(str(pulse.get("summary_line") or "snapshot: n/a"))
            mini_sparkline = pulse.get("mini_sparkline") if isinstance(pulse.get("mini_sparkline"), dict) else None
            spark_values = list(mini_sparkline.get("values", [])) if mini_sparkline else []
            spark_source = str(mini_sparkline.get("source") or "unavailable") if mini_sparkline else "unavailable"
            spark_points = len(spark_values)
            if len(spark_values) >= 3:
                spark_series = str(mini_sparkline.get("series") or "score")
                st.caption(f"mini-sparkline: {spark_series}")
                st.line_chart(pd.DataFrame({"value": spark_values}), height=80, use_container_width=True)
            else:
                fallback_reason = str(mini_sparkline.get("reason") or "").strip().lower() if mini_sparkline else ""
                if not fallback_reason:
                    fallback_reason = "warming_up"
                if fallback_reason in {"warming_up", "not_enough_points"} or spark_points < 3:
                    _log_pulse_sparkline_missing("not_enough_points", token_key=card_key)
                    st.caption("collecting history…")
                else:
                    _log_pulse_sparkline_missing(fallback_reason, token_key=card_key)
                    st.caption(fallback_reason)
            st.caption(f"sparkline source: {spark_source}")
            st.caption(f"points: {spark_points}")
            c1, c2 = st.columns(2)
            with c1:
                if dex_url:
                    link_button("Dex", dex_url, use_container_width=True, key=f"pulse_dex_{idx}_{hkey(card_key)}")
                else:
                    st.button("Dex", disabled=True, use_container_width=True, key=f"pulse_dex_disabled_{idx}_{hkey(card_key)}")
            with c2:
                if in_monitoring:
                    st.button("Already in Monitoring", disabled=True, use_container_width=True, key=f"pulse_in_mon_{idx}_{hkey(card_key)}")
                elif low_liq_for_monitor:
                    st.button("+ Monitor", disabled=True, use_container_width=True, key=f"pulse_add_disabled_{idx}_{hkey(card_key)}")
                    st.caption(f"Low liquidity ({fmt_usd(liq_usd)} < {fmt_usd(pulse_min_liq_usd)})")
                else:
                    if st.button("+ Monitor", use_container_width=True, key=f"pulse_add_{idx}_{hkey(card_key)}"):
                        payload = _pulse_pair_payload(row, best if isinstance(best, dict) else None)
                        add_res = add_to_monitoring(
                            payload,
                            float(score),
                            entry_status=str(row.get("entry_status") or row.get("entry") or ""),
                            entry_score=float(parse_float(row.get("entry_score", score), score)),
                            risk_level=str(row.get("risk_level") or row.get("risk") or ""),
                        )
                        if add_res in {"OK", "EXISTS_ACTIVE"}:
                            st.success("Added to Monitoring.")
                        elif add_res == "EXISTS_ARCHIVED":
                            st.info("Token already archived. Re-activate from Archive if needed.")
                        else:
                            st.warning(f"Monitor action: {add_res}")
                        load_monitoring_rows_cached.clear()
                        request_rerun()

    st.markdown("---")
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

    st.markdown("---")
    st.subheader("🧭 Manual calibration assistant (suggestions only)")
    journal_rows = load_csv(SIGNAL_JOURNAL_CSV, SIGNAL_JOURNAL_FIELDS)
    trust_index = load_pattern_trust_index_cached()
    calibration_output = build_manual_calibration_suggestions(
        journal_rows=journal_rows,
        trust_index=trust_index,
        review_findings=review_cards,
        explicit_safety_whitelist=[],
    )
    if calibration_output.get("status") != "ok":
        st.info(str(calibration_output.get("message") or "not enough evidence"))
    else:
        st.caption("Manual apply required for every change. No automatic mutation path in this assistant.")
        for idx, suggestion in enumerate(calibration_output.get("suggestions") or [], start=1):
            proposed = suggestion.get("proposed_change") or {}
            evidence = suggestion.get("evidence") or {}
            st.markdown(f"**Suggestion {idx}: {proposed.get('parameter', 'n/a')}**")
            st.json(
                {
                    "proposed_change": proposed,
                    "evidence": evidence,
                    "confidence": suggestion.get("confidence"),
                    "risk_note": suggestion.get("risk_note"),
                    "manual_apply_required": suggestion.get("manual_apply_required", True),
                }
            )
        st.caption(
            "Safety-critical parameters remain excluded unless explicitly whitelisted for manual review."
        )

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
    # Source of truth for Portfolio page UI: load_portfolio() + active flag filter.
    active_rows = _ui_portfolio_source_rows(rows)

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


def _wave_a_build_legacy_portfolio_reco(
    row: Dict[str, Any],
    entry_price: float,
    cur_price: float,
    liq: float,
    vol24: float,
    pc1h: float,
    pc5: float,
    score_live: float,
    best: Dict[str, Any],
    hist: List[Dict[str, Any]],
    liq_health: Dict[str, Any],
) -> str:
    decision = str(build_trade_hint(best)[0] if best else "NO DATA")
    exit_signal = exit_before_dump_detector(best, hist, entry_price)
    trap = liquidity_trap_detector(best, hist) if best else {"trap_level": "SAFE"}
    persistence = exit_persistence_state(hist, min_points=3)
    action_plan = action_from_exit_signal(exit_signal, persistence)
    anti_rug = anti_rug_early_detector(best, hist)
    level = str(exit_signal.get("exit_level", "")).upper()

    reco = portfolio_reco(entry_price, cur_price, liq, vol24, pc1h, pc5, decision, score_live, best)
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

    if level in ("EXIT", "CLOSE", "SELL"):
        return "EXIT"
    return str(reco or "").upper()


def wave_a_invariant_checks(max_rows: int = 25) -> List[str]:
    issues: List[str] = []
    rows = [r for r in load_portfolio() if str(r.get("active", "1")).strip() == "1"][:max_rows]
    for row in rows:
        chain = (row.get("chain") or "").strip().lower()
        base_addr = addr_store(chain, (row.get("base_token_address") or "").strip())
        symbol = str(row.get("base_symbol") or base_addr or "unknown").upper()
        if not chain or not base_addr:
            issues.append(f"[{symbol}] missing chain/base address in active portfolio row")
            continue
        try:
            best = best_pair_for_token_cached(chain, base_addr) or {}
            hist = token_history_rows(chain, base_addr, limit=120)
            entry_price = get_portfolio_entry_price(row)
            cur_price = parse_float((best or {}).get("priceUsd"), 0.0)
            liq = parse_float((best or {}).get("liquidity", {}).get("usd"), 0.0)
            vol24 = parse_float((best or {}).get("volume", {}).get("h24"), 0.0)
            vol5 = parse_float((best or {}).get("volume", {}).get("m5"), 0.0)
            pc1h = parse_float((best or {}).get("priceChange", {}).get("h1"), 0.0)
            pc5 = parse_float((best or {}).get("priceChange", {}).get("m5"), 0.0)
            score_live = score_pair(best) if best else parse_float(row.get("score"), 0.0)
            liq_health = liquidity_health(best) if best else {"level": "UNKNOWN"}
            legacy_reco = _wave_a_build_legacy_portfolio_reco(
                row=row,
                entry_price=entry_price,
                cur_price=cur_price,
                liq=liq,
                vol24=vol24,
                pc1h=pc1h,
                pc5=pc5,
                score_live=score_live,
                best=best,
                hist=hist,
                liq_health=liq_health,
            )

            pnl = 0.0
            if entry_price > 0 and cur_price > 0:
                pnl = (cur_price - entry_price) / entry_price * 100.0
            unified = compute_unified_recommendation(
                {
                    **row,
                    "pnl_pct": pnl,
                    "reco_model": legacy_reco,
                    "exit_signal": "EXIT" if "EXIT" in legacy_reco else "",
                    "liquidity_health": str(liq_health.get("level", "OK")),
                    "score": score_live,
                    "price_usd": cur_price,
                    "liq_usd": liq,
                    "vol24": vol24,
                    "vol5": vol5,
                },
                source="portfolio",
                hist=hist,
            )
        except Exception as e:
            issues.append(f"[{symbol}] invariant check failed: {type(e).__name__}: {e}")
            continue

        final_action = str(unified.get("final_action") or "").upper()
        health = dict(unified.get("health") or {})
        health_label = str(health.get("health_label") or "OK").upper()
        override_active = bool(unified.get("health_override_active"))
        override_action = str(unified.get("health_override_action") or "").upper()

        if final_action == "NO ENTRY":
            issues.append(f"[{symbol}] active portfolio final_action must not be NO ENTRY")
        if health_label in {"DEAD", "UNTRADEABLE"} and final_action == "HOLD":
            issues.append(f"[{symbol}] {health_label} cannot result in HOLD")
        if override_active:
            if override_action and final_action != override_action:
                issues.append(f"[{symbol}] health override mismatch: final={final_action}, override={override_action}")
            if override_action and override_action != "HOLD" and "HOLD" in legacy_reco:
                issues.append(f"[{symbol}] override conflicts with legacy reco ({legacy_reco})")

        avg_entry = parse_float(row.get("avg_entry_price"), 0.0)
        legacy_entries = [
            parse_float(row.get("entry_price_usd"), 0.0),
            parse_float(row.get("entry_price"), 0.0),
            parse_float(row.get("price_at_add"), 0.0),
            parse_float(row.get("entry"), 0.0),
        ]
        if avg_entry > 0 and abs(entry_price - avg_entry) > max(1e-12, avg_entry * 1e-6):
            issues.append(f"[{symbol}] entry source mismatch: avg_entry_price should be authoritative")
        if avg_entry > 0 and any(v > 0 and abs(v - avg_entry) > max(1e-12, avg_entry * 1e-6) for v in legacy_entries):
            if abs(entry_price - avg_entry) > max(1e-12, avg_entry * 1e-6):
                issues.append(f"[{symbol}] legacy entry field overrode avg_entry_price")

    return issues


def wave_b_invariant_checks(max_rows: int = 25) -> List[str]:
    issues: List[str] = []

    invalid_transition = transition_token_state(
        LIFECYCLE_SCOUT,
        "PROMOTED_TO_PORTFOLIO",
        {"target_lifecycle_state": LIFECYCLE_PORTFOLIO},
    )
    invalid_fields = dict(invalid_transition.get("updated_fields") or {})
    if bool(invalid_transition.get("valid")):
        issues.append("state_machine: invalid transition unexpectedly marked valid")
    if any(k in invalid_fields for k in ("lifecycle_state", "status", "state_event")):
        issues.append("state_machine: invalid transition attempted to mutate lifecycle fields")

    promoted_transition = transition_token_state(
        LIFECYCLE_MONITORING,
        "PROMOTED_TO_PORTFOLIO",
        {
            "requested_status": "portfolio",
            "target_lifecycle_state": LIFECYCLE_MONITORING,
            "transition_note": "guardrail_check",
        },
    )
    promoted_fields = dict(promoted_transition.get("updated_fields") or {})
    if not bool(promoted_transition.get("valid")):
        issues.append("state_machine: PROMOTED_TO_PORTFOLIO linking transition became invalid")
    if str(promoted_fields.get("lifecycle_state") or "").upper() != LIFECYCLE_MONITORING:
        issues.append("state_machine: portfolio-linked promotion should keep monitoring lifecycle_state")
    if str(promoted_fields.get("status") or "").upper() != LIFECYCLE_MONITORING:
        issues.append("state_machine: portfolio-linked promotion should keep monitoring status")
    if str(promoted_fields.get("state_event") or "").upper() != "PROMOTED_TO_PORTFOLIO":
        issues.append("state_machine: portfolio-linked promotion must emit PROMOTED_TO_PORTFOLIO")

    equal_score = 200.0
    equal_freshness = 900.0
    equal_due = 0.0
    rank_rows = []
    for tier_name, _rank in SCAN_QUEUE_TIERS:
        rank_rows.append(
            {
                "tier": tier_name,
                "tier_rank": SCAN_QUEUE_TIER_RANK.get(tier_name, 99),
                "score": equal_score,
                "freshness_sec": equal_freshness,
                "due_in_sec": equal_due,
            }
        )
    sorted_rows = sorted(
        rank_rows,
        key=lambda x: (
            x["tier_rank"],
            0 if x["due_in_sec"] <= 0 else 1,
            x["due_in_sec"],
            -x["freshness_sec"],
            -x["score"],
        ),
    )
    observed = [str(x.get("tier") or "") for x in sorted_rows]
    expected = [name for name, _rank in SCAN_QUEUE_TIERS]
    if observed != expected:
        issues.append(
            "queue: tier order must win when all else is equal "
            f"(expected={expected}, observed={observed})"
        )

    default_sleep = max(60, int(os.getenv("SCAN_INTERVAL_SEC", "300")))
    small_queue_sleep = max(30, min(default_sleep, 1))
    if small_queue_sleep != 30:
        issues.append("queue: empty/small queue fallback sleep changed from baseline-safe floor")
    queue_state_example = {"next_due_ts": time.time() + 120.0}
    computed_due = float(queue_state_example.get("next_due_ts", 0.0) or 0.0)
    if computed_due <= 0:
        issues.append("queue: next_due_ts was neutralized unexpectedly")

    monitoring_rows = load_monitoring()
    for row in monitoring_rows[:max_rows]:
        setup_inputs = build_setup_render_inputs(row)
        has_levels = bool(setup_inputs.get("has_actionable_levels"))
        watch_only = str(setup_inputs.get("watch_only_text") or "").strip().lower()
        actionable_via_api = has_actionable_levels(row)
        if actionable_via_api != has_levels:
            issues.append("setup: has_actionable_levels() diverged from sanitized setup path")
            break
        if not has_levels:
            for level_key in ("entry_now", "pullback_1", "pullback_2", "invalidation", "tp1", "tp2"):
                if _is_actionable_level(parse_float(setup_inputs.get(level_key), 0.0)):
                    issues.append(f"setup: non-actionable row exposes actionable level {level_key}")
                    break
            if watch_only != "setup: watch only":
                issues.append("setup: fallback text must be exactly 'setup: watch only'")

    neutral_gem = compute_gem_transition_score(best=None, hist=[])
    if bool(neutral_gem.get("gem_transition_sufficient")):
        issues.append("gem: insufficient history fallback should stay non-sufficient")
    if parse_float(neutral_gem.get("gem_transition_score"), GEM_TRANSITION_NEUTRAL) > GEM_TRANSITION_NEUTRAL:
        issues.append("gem: insufficient history fallback must have no positive priority bias")
    neutral_item = {
        "price_change_5m": 2.0,
        "gem_transition_score": neutral_gem.get("gem_transition_score"),
        "gem_transition_sufficient": "1" if bool(neutral_gem.get("gem_transition_sufficient")) else "0",
    }
    if compute_timing(neutral_item) == "EARLY":
        issues.append("gem: insufficient history fallback must have no positive timing bias")
    neutral_ui = monitoring_ui_state(
        {
            "row": {"entry_status": "WATCH", "entry_score": "150"},
            "health_label": "OK",
            "gem_transition_score": neutral_gem.get("gem_transition_score"),
            "gem_transition_sufficient": neutral_gem.get("gem_transition_sufficient"),
            "is_dead": False,
            "is_rug": False,
            "anti_rug": {"level": "SAFE"},
            "liq_health": {"level": "OK"},
        }
    )
    if "gem_transition" in [str(x) for x in neutral_ui.get("flags", [])]:
        issues.append("gem: insufficient history fallback must have no highlight uplift")

    blocked = gem_transition_priority_bias(120.0, 90.0, "DEAD", sufficient_history=True)
    if blocked != 120.0:
        issues.append("gem: strong negative health must block gem priority overrides")

    return issues


def wave_c_invariant_checks() -> List[str]:
    issues: List[str] = []

    mock_row = {
        "chain": "solana",
        "base_addr": "So11111111111111111111111111111111111111112",
        "base_token_address": "So11111111111111111111111111111111111111112",
        "base_symbol": "WAVEC",
        "entry_action": "ENTRY_NOW",
        "entry_score": "180",
        "risk_level": "MEDIUM",
        "timing_label": "EARLY",
    }
    mock_signal = {"bucket": "ENTRY_NOW", "horizon": "0-30m", "action": "Entry now"}
    now_ts = time.time()
    cooldown_seconds = 1800

    # Unknown event type must be surfaced by missing formatter routing.
    unknown_event_type = "__wave_c_unknown_event_type__"
    unknown_event_msg = format_signal_message(
        mock_row,
        mock_signal,
        source="monitoring",
        event_type=unknown_event_type,
        unified=None,
    )
    if unknown_event_msg is not None:
        issues.append("guardrail: unknown event_type did not surface as missing formatter")

    # Unknown tier must remain detectable (i.e. never normalized into known registry values).
    unknown_tier = "__wave_c_unknown_tier__"
    if unknown_tier in ALERT_TIER_REGISTRY.values():
        issues.append("guardrail: unknown tier unexpectedly normalized into registry")

    # Template routing must remain valid for every known event type.
    formatter_src = inspect.getsource(format_signal_message)
    for event_type in EVENT_TYPE_REGISTRY.values():
        if f"\"{event_type}\"" not in formatter_src:
            issues.append(f"guardrail: missing formatter route for event_type={event_type}")

    # Digest must stay isolated from operational alert classification/emission keys.
    mon_classification = resolve_tg_alert_classification("monitoring", mock_row, mock_signal, unified=None)
    if str(mon_classification.get("event_type") or "").strip().lower() == resolve_event_type("DIGEST"):
        issues.append("guardrail: monitoring operational alert leaked into digest event_type")
    portfolio_row = dict(mock_row)
    portfolio_row["entry_action"] = "ENTRY_NOW"
    portfolio_signal = {"bucket": "ADD", "horizon": "0-2h", "action": "Consider add"}
    port_classification = resolve_tg_alert_classification("portfolio", portfolio_row, portfolio_signal, unified=None)
    if str(port_classification.get("event_type") or "").strip().lower() == resolve_event_type("DIGEST"):
        issues.append("guardrail: portfolio operational alert leaked into digest event_type")

    digest_state = {"sent_emissions": {}, "sent_digest_emissions": {}}
    emission_key = build_emission_key(
        "monitoring",
        mock_row,
        mock_signal,
        cooldown_seconds=cooldown_seconds,
        now_ts=now_ts,
        unified=None,
    )
    mark_emission_sent(digest_state, emission_key=emission_key, now_ts=now_ts)
    digest_emission_key = build_digest_emission_key(cooldown_seconds=cooldown_seconds, now_ts=now_ts)
    if is_duplicate_digest_emission(
        digest_state,
        emission_key=digest_emission_key,
        now_ts=now_ts,
        cooldown_seconds=cooldown_seconds,
    ):
        issues.append("guardrail: digest duplicate state leaked from operational sent_emissions")

    # Mode filtering order in quiet mode: critical/health/portfolio-exit-reduce overrides must win.
    quiet = "quiet"
    critical_classification = {
        "event_type": resolve_event_type("ENTRY_ALERT"),
        "alert_tier": resolve_alert_tier("CRITICAL"),
    }
    if not should_emit_for_alert_mode(quiet, "monitoring", mock_row, mock_signal, critical_classification, unified=None):
        issues.append("guardrail: quiet mode must not suppress critical alerts")

    medium_classification = {
        "event_type": resolve_event_type("ENTRY_ALERT"),
        "alert_tier": resolve_alert_tier("MEDIUM"),
    }
    if should_emit_for_alert_mode(quiet, "monitoring", mock_row, mock_signal, medium_classification, unified=None):
        issues.append("guardrail: quiet mode medium filtering regression")

    health_classification = {
        "event_type": resolve_event_type("HEALTH_WARNING"),
        "alert_tier": resolve_alert_tier("MEDIUM"),
    }
    if not should_emit_for_alert_mode(quiet, "monitoring", mock_row, mock_signal, health_classification, unified=None):
        issues.append("guardrail: quiet mode must not suppress health warnings")

    exit_row = dict(mock_row)
    exit_row["entry_action"] = "EXIT"
    portfolio_medium_classification = {
        "event_type": resolve_event_type("PORTFOLIO_ACTION"),
        "alert_tier": resolve_alert_tier("MEDIUM"),
    }
    if not should_emit_for_alert_mode(
        quiet,
        "portfolio",
        exit_row,
        {"bucket": "EXIT", "horizon": "now", "action": "Consider exit"},
        portfolio_medium_classification,
        unified={"final_action": "EXIT"},
    ):
        issues.append("guardrail: quiet mode must not suppress portfolio EXIT/REDUCE")

    # Duplicate checks: same token + same event/key in cooldown => blocked.
    duplicate_state = {"sent_emissions": {}}
    key_same = build_emission_key(
        "monitoring",
        mock_row,
        mock_signal,
        cooldown_seconds=cooldown_seconds,
        now_ts=now_ts,
        unified=None,
    )
    mark_emission_sent(duplicate_state, emission_key=key_same, now_ts=now_ts)
    if not is_duplicate_emission(
        duplicate_state,
        emission_key=key_same,
        now_ts=now_ts + 10.0,
        cooldown_seconds=cooldown_seconds,
    ):
        issues.append("guardrail: same token + same event/key should be blocked in cooldown")

    # Duplicate checks: same token + different event_type => allowed.
    key_different_event = build_emission_key(
        "portfolio",
        mock_row,
        {"bucket": "ADD", "horizon": "0-2h", "action": "Consider add"},
        cooldown_seconds=cooldown_seconds,
        now_ts=now_ts,
        unified={"final_action": "ENTRY_NOW", "final_reason": "wave_c"},
    )
    if key_different_event == key_same:
        issues.append("guardrail: different event type unexpectedly produced identical emission key")
    elif is_duplicate_emission(
        duplicate_state,
        emission_key=key_different_event,
        now_ts=now_ts + 10.0,
        cooldown_seconds=cooldown_seconds,
    ):
        issues.append("guardrail: same token + different event_type should remain allowed")

    return issues


def wave_d_pattern_trust_invariant_checks() -> List[str]:
    issues: List[str] = []

    row_key = {
        "setup_class": "breakout",
        "health_label": "OK",
        "timing_label": "EARLY",
        "market_regime_bucket": "risk_on",
    }
    key_one = build_pattern_key(row_key)
    key_two = build_pattern_key(dict(row_key))
    if key_one != key_two:
        issues.append("pattern trust: deterministic key generation regression")

    pending_only_rows = [
        {
            "setup_context": "breakout",
            "health_label": "OK",
            "timing_label": "EARLY",
            "outcome_15m_status": "PENDING",
            "outcome_1h_status": "PENDING",
            "outcome_4h_status": "PENDING",
            "outcome_24h_status": "PENDING",
        }
    ]
    pending_index = build_pattern_trust_index(rows=pending_only_rows)
    pending_payload = compute_pattern_trust(row_key, trust_index=pending_index)
    if parse_float(pending_payload.get("trust_score"), 0.0) != 0.0:
        issues.append("pattern trust: pending-only rows should stay neutral")

    low_sample_rows = []
    for _i in range(3):
        low_sample_rows.append(
            {
                "setup_context": "breakout",
                "health_label": "OK",
                "timing_label": "EARLY",
                "outcome_1h_status": "UP",
                "outcome_1h_return_pct": "18.0",
                "outcome_1h_ts_utc": now_utc_str(),
            }
        )
    low_sample_index = build_pattern_trust_index(rows=low_sample_rows)
    low_sample_payload = compute_pattern_trust(row_key, trust_index=low_sample_index)
    if parse_float(low_sample_payload.get("trust_score"), 0.0) != 0.0:
        issues.append("pattern trust: low sample must stay neutral/advisory")

    mixed_rows = []
    for pct in (20.0, -18.0, 12.0, -11.0, 8.0, -7.0, 6.0, -5.0):
        mixed_rows.append(
            {
                "setup_context": "breakout",
                "health_label": "OK",
                "timing_label": "EARLY",
                "outcome_4h_status": "UP" if pct > 0 else "DOWN",
                "outcome_4h_return_pct": str(pct),
                "outcome_4h_ts_utc": "2026-04-01T00:00:00Z",
            }
        )
    mixed_index = build_pattern_trust_index(rows=mixed_rows)
    mixed_payload = compute_pattern_trust(row_key, trust_index=mixed_index)
    mixed_score = abs(parse_float(mixed_payload.get("trust_score"), 0.0))
    if mixed_score >= TRUST_SCORE_CAP:
        issues.append("pattern trust: conflicting outcomes became extreme instead of bounded")

    dead_row = {
        "entry_action": "ENTRY_NOW",
        "entry_score": "260",
        "risk_level": "LOW",
        "timing_label": "GOOD",
        "liquidity_health": "DEAD",
        "is_dead": True,
        "setup_class": "breakout",
        "setup_context": "breakout",
        "health_label": "DEAD",
        "reco_model": "HOLD",
        "score": "260",
        "pnl_pct": "5",
    }
    strong_trust = {
        build_pattern_key(dead_row): {
            "samples": 50,
            "sum_score": 35.0,
            "recent_samples": 5,
        }
    }
    dead_payload = compute_pattern_trust(dead_row, trust_index=strong_trust)
    if parse_float(dead_payload.get("trust_score"), 0.0) <= 0:
        issues.append("pattern trust: synthetic strong trust fixture invalid")
    dead_unified = compute_unified_recommendation(dead_row, source="portfolio")
    if str(dead_unified.get("final_action") or "").upper() != "EXIT":
        issues.append("pattern trust: dead/untradeable candidate was upgraded by trust")

    return issues


def wave_d_plus_rotation_invariant_checks() -> List[str]:
    issues: List[str] = []

    active_rows = [
        {"active": "1", "chain": "solana", "base_symbol": "DEADPOS", "portfolio_position_score": -120, "health_label": "DEAD", "final_action": "EXIT"},
        {"active": "1", "chain": "solana", "base_symbol": "WEAKPOS", "portfolio_position_score": 72, "health_label": "OK", "final_action": "REDUCE"},
    ]
    monitoring_rows = [
        {"active": "1", "chain": "solana", "base_symbol": "STRONGIN", "monitoring_candidate_score": 260, "final_action": "ENTER NOW", "health_label": "OK"},
        {"active": "1", "chain": "solana", "base_symbol": "LOWIN", "monitoring_candidate_score": 120, "final_action": "TRACK ONLY", "health_label": "OK"},
    ]
    active_snapshot = json.dumps(active_rows, sort_keys=True)
    monitoring_snapshot = json.dumps(monitoring_rows, sort_keys=True)
    advisory = build_rotation_advisory(active_rows, monitoring_rows, gap_floor=10.0, confidence_floor=0.20)
    suggestions = list(advisory.get("suggestions") or [])
    if not suggestions:
        issues.append("rotation advisor: weak portfolio vs strong monitoring should produce suggestion")
    else:
        first = suggestions[0]
        required_fields = (
            "rotate_out_symbol",
            "rotate_in_symbol",
            "rotation_reason",
            "rotation_confidence",
            "rotation_gap",
            "portfolio_position_score",
            "monitoring_candidate_score",
        )
        for field in required_fields:
            if field not in first:
                issues.append(f"rotation advisor: missing structured field {field}")
        if parse_float(first.get("rotation_gap"), 0.0) <= 0:
            issues.append("rotation advisor: expected positive gap for strong candidate")

    active_strong_rows = [
        {"active": "1", "chain": "solana", "base_symbol": "HOLDPOS", "portfolio_position_score": 150, "health_label": "OK", "final_action": "HOLD"},
    ]
    no_stronger_rows = [
        {"active": "1", "chain": "solana", "base_symbol": "SAMEIN", "monitoring_candidate_score": 70, "final_action": "ENTER NOW", "health_label": "OK"},
    ]
    no_stronger = build_rotation_advisory(active_strong_rows, no_stronger_rows, gap_floor=20.0, confidence_floor=0.20)
    if no_stronger.get("suggestions"):
        issues.append("rotation advisor: must not force rotation when stronger candidate is absent")

    low_gap_rows = [
        {"active": "1", "chain": "solana", "base_symbol": "TINYIN", "monitoring_candidate_score": 82, "final_action": "ENTER NOW", "health_label": "OK"},
    ]
    low_gap = build_rotation_advisory(active_strong_rows, low_gap_rows, gap_floor=40.0, confidence_floor=0.80)
    if low_gap.get("suggestions"):
        issues.append("rotation advisor: must not force suggestion on low gap/low confidence")

    if active_snapshot != json.dumps(active_rows, sort_keys=True) or monitoring_snapshot != json.dumps(monitoring_rows, sort_keys=True):
        issues.append("rotation advisor: input state mutated (advisory must be read-only)")

    return issues


def wave_d_plus_manual_calibration_invariant_checks() -> List[str]:
    issues: List[str] = []

    journal_rows = []
    for idx in range(16):
        journal_rows.append(
            {
                "outcome_1h_status": "UP" if idx % 3 != 0 else "DOWN",
                "outcome_1h_return_pct": "12.0" if idx % 3 != 0 else "-8.0",
                "outcome_4h_status": "UP" if idx % 4 != 0 else "DOWN",
                "outcome_4h_return_pct": "18.0" if idx % 4 != 0 else "-10.0",
            }
        )
    trust_index = {
        "p1": {"samples": 12, "sum_score": 6.0},
        "p2": {"samples": 14, "sum_score": 7.0},
    }
    review_findings = [{"ui_visible_score": 84.0, "ui_badge": "REVIEW"} for _i in range(10)]

    output = build_manual_calibration_suggestions(
        journal_rows=journal_rows,
        trust_index=trust_index,
        review_findings=review_findings,
    )
    if output.get("status") != "ok":
        issues.append("manual calibration: expected suggestions on sufficient evidence")
    suggestions = list(output.get("suggestions") or [])
    if not suggestions:
        issues.append("manual calibration: no suggestions generated on sufficient evidence")
    else:
        first = suggestions[0]
        required_fields = ("proposed_change", "evidence", "confidence", "risk_note", "manual_apply_required")
        for field in required_fields:
            if field not in first:
                issues.append(f"manual calibration: missing structured field {field}")
        if first.get("manual_apply_required") is not True:
            issues.append("manual calibration: manual_apply_required must stay true")
        evidence = first.get("evidence") or {}
        for evidence_field in ("journal_slice_used", "trust_slice_used", "review_finding_used"):
            if evidence_field not in evidence:
                issues.append(f"manual calibration: missing evidence trail field {evidence_field}")

    low_evidence = build_manual_calibration_suggestions(
        journal_rows=[{"outcome_1h_status": "PENDING"}],
        trust_index={},
        review_findings=[],
    )
    if low_evidence.get("status") != "not_enough_evidence":
        issues.append("manual calibration: sparse input must return not_enough_evidence")

    blocked = build_manual_calibration_suggestions(
        journal_rows=journal_rows,
        trust_index=trust_index,
        review_findings=review_findings,
        explicit_safety_whitelist=[],
    )
    blocked_params = {
        str((x.get("proposed_change") or {}).get("parameter") or "")
        for x in (blocked.get("suggestions") or [])
    }
    if blocked_params.intersection(CALIBRATION_SAFETY_CRITICAL_PARAMS):
        issues.append("manual calibration: safety-critical params leaked without whitelist")

    return issues


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

    if (
        "debug_mode = bool(st.session_state.get(\"debug_mode\", False) or st.session_state.get(\"show_debug_raw\", False))" not in src
        or "Raw internals hidden (debug mode only)." not in src
    ):
        issues.append("monitoring normal mode raw internals guard missing")

    wave_a_issues = wave_a_invariant_checks()
    for wave_issue in wave_a_issues:
        issues.append(f"Wave A invariant: {wave_issue}")
    wave_b_issues = wave_b_invariant_checks()
    for wave_issue in wave_b_issues:
        issues.append(f"Wave B invariant: {wave_issue}")
    wave_c_issues = wave_c_invariant_checks()
    for wave_issue in wave_c_issues:
        issues.append(f"Wave C invariant: {wave_issue}")
    wave_d_issues = wave_d_pattern_trust_invariant_checks()
    for wave_issue in wave_d_issues:
        issues.append(f"Wave D invariant: {wave_issue}")
    wave_d_plus_issues = wave_d_plus_rotation_invariant_checks()
    for wave_issue in wave_d_plus_issues:
        issues.append(f"Wave D+ invariant: {wave_issue}")
    wave_d_plus_manual_calibration_issues = wave_d_plus_manual_calibration_invariant_checks()
    for wave_issue in wave_d_plus_manual_calibration_issues:
        issues.append(f"Wave D+ PR6 invariant: {wave_issue}")
    alert_pipeline_issues = alert_pipeline_reliability_checks()
    for pipeline_issue in alert_pipeline_issues:
        issues.append(f"Alert pipeline reliability: {pipeline_issue}")

    if issues:
        st.error("CORE SAFETY BROKEN")
        st.caption(f"Violations: {len(issues)}")
        for i in issues[:20]:
            st.write(f"– {i}")
    else:
        st.success("Core safety OK")


def alert_pipeline_reliability_checks() -> List[str]:
    issues: List[str] = []
    state = scanner_state_load() or {}
    runtime = get_worker_runtime_state(state=state)
    required_runtime_keys = [
        "worker_status",
        "worker_boot_count",
        "last_loop_ts",
        "last_heartbeat_ts",
        "last_notifications_ts",
        "last_notification_trigger_model",
        "last_send_success_ts",
        "last_error_ts",
        "last_error_reason",
        "last_empty_reason",
    ]
    for key in required_runtime_keys:
        if key not in runtime:
            issues.append(f"runtime key missing: {key}")
    try:
        worker_src = Path(os.path.join(os.path.dirname(__file__), "scanner_worker.py")).read_text(encoding="utf-8")
    except Exception:
        worker_src = ""
    if "run_auto_notifications(" not in worker_src:
        issues.append("worker loop does not call run_auto_notifications")
    return issues


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
        runtime = get_worker_runtime_state()
        st.caption(
            f"Worker loop: {runtime.get('last_loop_ts') or '—'} | "
            f"notifications: {runtime.get('last_notifications_ts') or '—'} | "
            f"last send: {runtime.get('last_send_success_ts') or '—'}"
        )
        st.caption(
            f"last_error: {runtime.get('last_error_ts') or '—'} "
            f"reason={runtime.get('last_error_reason') or '—'}"
        )
        st.caption(
            f"send_failure_reason={runtime.get('last_send_failure_reason') or '—'} | "
            f"duplicate_reason={runtime.get('last_duplicate_reason') or '—'} | "
            f"job_reason={runtime.get('last_job_reason') or '—'} | "
            f"lock_code={runtime.get('last_lock_code') or '—'}"
        )
        st.caption(
            f"last_empty_reason={runtime.get('last_empty_reason') or '—'} | "
            f"last_block_reason={runtime.get('last_block_reason') or '—'} | "
            f"path={runtime.get('last_candidate_path') or '—'} | "
            f"fallback={runtime.get('last_fallback_reason') or '—'} | "
            f"cycle_status={runtime.get('last_cycle_status') or '—'}"
        )
        st.caption(
            f"stale_lock_ts={runtime.get('last_stale_lock_ts') or '—'} | "
            f"stale_run_ts={runtime.get('last_stale_run_ts') or '—'} | "
            f"diag={runtime.get('last_diag_summary') or '—'}"
        )
        counters = runtime.get("last_notification_counters") or {}
        if isinstance(counters, dict):
            st.caption(
                f"last_counters: before={int(parse_float(counters.get('before_filter', 0), 0.0))} | "
                f"after={int(parse_float(counters.get('after_filter', 0), 0.0))} | "
                f"sent={int(parse_float(counters.get('sent', 0), 0.0))} | "
                f"blocked={int(parse_float(counters.get('blocked', 0), 0.0))} | "
                f"send_fail={int(parse_float(counters.get('send_fail', 0), 0.0))}"
            )
        blocked_reasons = runtime.get("last_notification_block_reasons") or {}
        if isinstance(blocked_reasons, dict) and blocked_reasons:
            st.caption(f"last_block_reasons={safe_json(blocked_reasons)}")
        st.caption(
            f"loops={runtime.get('loop_iterations', 0)} | "
            f"notif_runs={runtime.get('notification_runs', 0)} | "
            f"sent={runtime.get('notification_sent', 0)} | "
            f"empty={runtime.get('notification_empty_cycles', 0)} | "
            f"blocked={runtime.get('notification_blocked_cycles', 0)} | "
            f"failed={runtime.get('notification_failed_cycles', 0)}"
        )

        st.write("---")
        st.write("### Portfolio health (recent)")
        period_hours = st.number_input("Window (hours)", min_value=1, max_value=168, value=24, step=1)
        counts = portfolio_reco_health_counts(period_hours=int(period_hours))
        st.caption(
            f"Last {int(period_hours)}h: "
            f"DEAD={counts['DEAD']} • "
            f"UNTRADEABLE={counts['UNTRADEABLE']} • "
            f"LOW_LIQ+COLD={counts['LOW_LIQ+COLD']}"
        )

        st.write("---")
        st.write("### Core safety")
        core_safety_self_check()

        st.write("---")
        st.write("### Alert pipeline reliability")
        pipeline_issues = alert_pipeline_reliability_checks()
        if pipeline_issues:
            st.caption(f"Violations: {len(pipeline_issues)}")
            for issue in pipeline_issues[:10]:
                st.write(f"– {issue}")
        else:
            st.caption("No violations detected.")

        st.write("---")
        st.write("### Wave B guardrails")
        wave_b_issues = wave_b_invariant_checks()
        if wave_b_issues:
            st.caption(f"Violations: {len(wave_b_issues)}")
            for issue in wave_b_issues[:10]:
                st.write(f"– {issue}")
        else:
            st.caption("No violations detected.")

        st.write("---")
        st.write("### Wave C guardrails")
        wave_c_issues = wave_c_invariant_checks()
        if wave_c_issues:
            st.caption(f"Violations: {len(wave_c_issues)}")
            for issue in wave_c_issues[:10]:
                st.write(f"– {issue}")
        else:
            st.caption("No violations detected.")

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
    if is_debug_mode_enabled():
        st.caption(health_thresholds_debug_line())

    rows = load_portfolio()
    # Source of truth for Portfolio page UI: load_portfolio() + active flag filter.
    active_rows = _ui_portfolio_source_rows(rows)
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

    rotation_advisor = build_rotation_advisory(active_rows, load_monitoring())
    rotation_suggestions = list(rotation_advisor.get("suggestions") or [])
    weak_rotation_symbols = {
        str(x.get("rotate_out_symbol") or "").upper()
        for x in rotation_suggestions
        if str(x.get("rotate_out_symbol") or "").strip()
    }

    st.markdown("## 🔁 Rotation advisory (advisory-only)")
    if rotation_suggestions:
        for s in rotation_suggestions:
            st.warning(
                f"Rotate-out: {s.get('rotate_out_symbol', '?')} → "
                f"Rotate-into: {s.get('rotate_in_symbol', '?')}"
            )
            st.caption(
                f"{s.get('rotation_reason', '')} | "
                f"confidence={parse_float(s.get('rotation_confidence'), 0.0):.2f} | "
                f"rotation_gap={parse_float(s.get('rotation_gap'), 0.0):.2f}"
            )
    else:
        st.caption("No forced rotation: stronger eligible candidate absent or gap/confidence below threshold.")

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
                "price_usd": cur_price,
                "liq_usd": liq,
                "vol24": vol24,
                "vol5": vol5,
            },
            source="portfolio",
            hist=hist,
        )
        log_portfolio_recommendation_snapshot(
            token=base_addr,
            chain=chain,
            unified=unified,
        )
        health = dict(unified.get("health") or detect_position_health(
            {
                **r,
                "price_usd": cur_price,
                "liq_usd": liq,
                "vol24": vol24,
                "vol5": vol5,
                "liquidity_health": liq_health.get("level", "OK"),
                "anti_rug": anti_rug.get("level", "SAFE"),
            },
            hist,
        ))

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
            if str(health.get("health_label", "OK")) in ("DEAD", "UNTRADEABLE"):
                st.caption(f"⚠ Market health: {health.get('health_label')} — {health.get('health_reason')}")
            if unified.get("health_override_active"):
                st.caption(
                    f"🛡 Health override active: {health.get('health_label', 'OK')} — {unified.get('health_override_reason', '')}"
                )
            swap_url = r.get("swap_url") or build_swap_url(chain, base_addr)
            if swap_url:
                link_button("Swap", swap_url, use_container_width=True, key=f"p_sw_{idx}_{hkey(base_addr, chain)}")

        with c2:
            st.markdown(f"### {portfolio_action_badge(unified['final_action'])}")
            st.markdown(f"**Recommended action: {unified['final_action']}**")
            st.caption(f"Reason: {unified['final_reason']}")
            if unified.get("health_override_active"):
                health_copy = str(health.get("health_label", "UNKNOWN")).replace("_", " ")
                st.caption(f"⚠ Health override: {health_copy} → {unified.get('health_override_action', unified['final_action'])}")
                st.caption(f"Override reason: {unified.get('health_override_reason', health.get('health_reason', 'n/a'))}")
            elif str(health.get("health_label", "OK")) != "OK":
                health_copy = str(health.get("health_label", "UNKNOWN")).replace("_", " ")
                st.caption(f"⚠ Health: {health_copy}")
            with st.expander("Market / position metrics", expanded=False):
                st.write(f"Score: {s_live:.2f}" if best else f"Score: {r.get('score','n/a')}")
                st.write(f"Entry: ${entry_price_str}")
                st.write(f"Now: ${cur_price:.8f}" if cur_price else "Now: n/a")
                st.write(f"PnL: {pnl:+.2f}%" if entry_price and cur_price else "PnL: n/a")
                if entry_price > 0 and cur_price > 0:
                    st.write(f"Δ vs entry: {cur_price - entry_price:+.10f}")
                else:
                    st.write("Δ vs entry: n/a")
                st.write(f"Liq: {fmt_usd(liq)}" if best else "Liq: n/a")
                st.write(f"Vol24: {fmt_usd(vol24)}" if best else "Vol24: n/a")
                st.write(f"Vol5: {fmt_usd(vol5)}" if best else "Vol5: n/a")
                st.write(f"Δ m5: {fmt_pct(pc5)}")
                st.write(f"Δ h1: {fmt_pct(pc1h)}")
                st.write(f"score: {s_live:.2f}")

        with c3:
            with st.expander("Risk diagnostics", expanded=False):
                if strength_cls == "WEAK":
                    st.caption("⚠ weak position – candidate for rotation")
                    if str(base_sym).upper() in weak_rotation_symbols:
                        st.caption(f"Rotation hint: sell {int(exit_pct * 100)}%")
                        rotate_suggestion = next(
                            (
                                s for s in rotation_suggestions
                                if str(s.get("rotate_out_symbol") or "").upper() == str(base_sym).upper()
                            ),
                            None,
                        )
                        if rotate_suggestion:
                            st.caption(f"→ rotate into {rotate_suggestion.get('rotate_in_symbol', '?')}")
                st.write(f"Reco model: {reco}")
                st.write(f"Exit signal: {level}")
                st.write(f"Health label: {health.get('health_label', 'OK')}")
                st.write(f"Health reason: {health.get('health_reason', 'n/a')}")
                st.write(
                    f"Health flags: dead={bool(health.get('is_dead'))}, "
                    f"stale={bool(health.get('is_stale'))}, cold={bool(health.get('is_cold'))}, "
                    f"low_liq={bool(health.get('is_low_liq'))}, untradeable={bool(health.get('is_untradeable'))}"
                )
                st.write(
                    f"Health grace mode: applied={bool(health.get('health_grace_applied'))}, "
                    f"position_age_min={parse_float(health.get('position_age_min', 0), 0.0):.1f}, "
                    f"history_points={int(parse_float(health.get('history_points', 0), 0.0))}"
                )
                st.write(
                    f"Health checks: liq={fmt_usd(parse_float(health.get('liq_usd', 0), 0.0))}, "
                    f"vol24={fmt_usd(parse_float(health.get('vol24', 0), 0.0))}, "
                    f"vol5={fmt_usd(parse_float(health.get('vol5', 0), 0.0))}, "
                    f"snapshot_age_min={parse_float(health.get('snapshot_age_min', 0), 0.0):.1f}"
                )
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
                avg_entry_str = f"{float(avg_entry):.12f}".rstrip("0").rstrip(".")
                for rr in all_rows:
                    if addr_key(rr.get("chain", ""), rr.get("base_token_address", "")) == target_key:
                        rr["avg_entry_price"] = avg_entry_str
                        # Keep legacy field synchronized for older flows that still read it.
                        rr["entry_price_usd"] = avg_entry_str
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
                closed_entry_price = get_portfolio_entry_price(r)
                st.write(
                    f"{r.get('ts_utc','')} – {r.get('base_symbol','')}/{r.get('quote_symbol','')} "
                    f"– entry ${closed_entry_price:.12f}".rstrip("0").rstrip(".")
                    + f" – {r.get('action','')}"
                )


# =============================
# App main
# =============================
def build_active_monitoring_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active_rows: List[Dict[str, Any]] = []
    for r in rows:
        status = str(r.get("status", "")).strip().upper()
        if status in ("ACTIVE", "WATCH", "WAIT", "", "TRACK", "READY", LIFECYCLE_MONITORING):
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
        tg_state = load_tg_state()
        if not isinstance(tg_state, dict):
            tg_state = {}
            st.warning("Telegram state data unavailable.")
        # Backward-compatible alias for older debug snippets that may still
        # reference `state` directly. Keep this local to the panel scope.
        state = tg_state
        runtime_snapshot = scanner_state_load()
        if not isinstance(runtime_snapshot, dict):
            runtime_snapshot = {}
            st.caption("Runtime state data unavailable.")
        runtime_debug = get_worker_runtime_state(state=runtime_snapshot)
        delivery_debug = runtime_debug.get("last_digest_delivery_debug")
        if isinstance(delivery_debug, dict) and delivery_debug:
            st.caption(
                f"Digest delivery: ok={delivery_debug.get('ok', '—')} | "
                f"trigger={delivery_debug.get('trigger_source', '—')} | "
                f"chunks={delivery_debug.get('chunks_sent', '—')}"
            )
        else:
            st.caption("Digest delivery debug data unavailable.")
        last_digest_sent = str(tg_state.get("last_digest_sent_at") or "")
        if last_digest_sent:
            st.caption(f"Last digest sent at: {last_digest_sent}")
        else:
            st.caption("Last digest sent at: data unavailable.")

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
    scan_state = scanner_state_load() or {}
    if not isinstance(scan_state, dict):
        scan_state = {}
    active_monitoring_rows = build_active_monitoring_rows(monitoring_rows)
    active_portfolio_rows = [
        row
        for row in portfolio_rows
        if str(row.get("active", row.get("is_active", "1"))).strip().lower() not in {"0", "false", "no"}
    ]
    if st.session_state.get("last_scan_ts") != scan_state.get("last_run_ts"):
        st.session_state["last_scan_ts"] = scan_state.get("last_run_ts")
        if ui_notification_emission_enabled():
            run_auto_notifications(
                scan_state,
                active_monitoring_rows,
                active_portfolio_rows,
                trigger_model="ui_scan_sync",
            )
        evaluate_outcome_journals()
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
        st.markdown("### Background runtime")
        st.caption("Alerts/notifications are emitted by scanner_worker.py (background service), not by this UI tab.")
        runtime = get_worker_runtime_state(state=scan_state)
        liveness = runtime_liveness(runtime)
        st.caption(
            f"Worker status: {runtime.get('worker_status') or 'unknown'} | "
            f"boots={runtime.get('worker_boot_count', 0)} | "
            f"consecutive_crashes={runtime.get('worker_consecutive_crashes', 0)}"
        )
        worker_loop_ts = runtime.get("last_loop_ts") or ""
        if worker_loop_ts:
            st.caption(f"Worker heartbeat: active (last loop {worker_loop_ts}).")
        else:
            st.caption("Worker heartbeat: no loop heartbeat recorded yet.")
        st.caption(
            f"Liveness: {liveness.get('status')} | "
            f"age_sec={liveness.get('age_sec') if liveness.get('age_sec') is not None else '—'} | "
            f"stale_threshold_sec={liveness.get('stale_threshold_sec')}"
        )
        st.caption(
            "Last notification trigger model: "
            f"{runtime.get('last_notification_trigger_model') or '—'}"
        )
        st.caption(
            f"Last cycle status: {runtime.get('last_cycle_status') or '—'} | "
            f"path={runtime.get('last_candidate_path') or '—'} | "
            f"fallback={runtime.get('last_fallback_reason') or '—'}"
        )
        st.caption(
            f"Last job reason: {runtime.get('last_job_reason') or '—'} | "
            f"lock_code={runtime.get('last_lock_code') or '—'} | "
            f"stale_lock_ts={runtime.get('last_stale_lock_ts') or '—'} | "
            f"stale_run_ts={runtime.get('last_stale_run_ts') or '—'}"
        )
        counters = runtime.get("last_notification_counters") or {}
        if isinstance(counters, dict):
            st.caption(
                "Last counters: "
                f"before={int(parse_float(counters.get('before_filter', 0), 0.0))} | "
                f"after={int(parse_float(counters.get('after_filter', 0), 0.0))} | "
                f"sent={int(parse_float(counters.get('sent', 0), 0.0))} | "
                f"blocked={int(parse_float(counters.get('blocked', 0), 0.0))} | "
                f"send_fail={int(parse_float(counters.get('send_fail', 0), 0.0))}"
            )
        blocked_reasons = runtime.get("last_notification_block_reasons") or {}
        if isinstance(blocked_reasons, dict) and blocked_reasons:
            st.caption(f"Last block reasons: {safe_json(blocked_reasons)}")
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
