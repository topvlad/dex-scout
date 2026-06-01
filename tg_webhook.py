import hmac
import os
import re
import time
from html import escape
from typing import Any, Callable, Dict, List, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request, Response

from runtime_core import addr_store, canonical_entity_key, normalize_chain_name, now_utc_str
import app_runtime_facade

BOOTSTRAP_ERROR: Dict[str, str] = {}
IMPORT_FAILED = False
IMPORT_ERROR = ""
CALLBACK_ID_TTL_SECONDS = 24 * 60 * 60
CALLBACK_ID_MAX_ITEMS = 5000
TG_STATE: Dict[str, Dict[str, float]] = {"processed_callback_ids": {}}
CA_RE = re.compile(r"^[a-zA-Z0-9]{20,64}$")
ACTION_ALIASES = {
    "pf": "pf",
    "pf_add": "pf",
    "portfolio": "pf",
    "portfolio_add": "pf",
    "discovery_pf_add": "pf",
    "mn": "mn",
    "mon_add": "mn",
    "monitor": "mn",
    "monitor_add": "mn",
    "discovery_mon_add": "mn",
    "rm": "rm",
    "remove": "rm",
    "delete": "rm",
    "archive": "rm",
}
from config import CHAINS_DEFAULT
try:
    from config import TG_CALLBACK_VALID_CHAINS
except ImportError:
    TG_CALLBACK_VALID_CHAINS = CHAINS_DEFAULT
VALID_CHAINS = {str(c or "").strip().lower() for c in TG_CALLBACK_VALID_CHAINS if str(c or "").strip()}


def _env_int_bounded(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
    return max(min_value, min(max_value, value))


TG_META_CACHE_TTL_SEC = _env_int_bounded("TG_META_CACHE_TTL_SEC", 30, 5, 120)
TG_META_CACHE: Dict[str, Any] = {"ts": 0.0, "index": {}}

MON_FIELDS = ["chain", "base_addr", "active", "archived", "status", "updated_at", "note", "archived_reason"]
PORTFOLIO_FIELDS = ["chain", "base_token_address", "active", "archived", "updated_at", "note"]


def _facade_state(force_reload: bool = False) -> Dict[str, Any]:
    state = app_runtime_facade.get_import_state(force_reload=force_reload)
    if IMPORT_FAILED or BOOTSTRAP_ERROR:
        state["ok"] = False
        state["error"] = IMPORT_ERROR or BOOTSTRAP_ERROR.get("exception_text", state.get("error", ""))
        state["exception_type"] = BOOTSTRAP_ERROR.get("exception_type", state.get("exception_type", ""))
    return state


def _app_available() -> bool:
    return bool(_facade_state().get("ok"))


def _facade_call(name: str, *args: Any, **kwargs: Any) -> Any:
    state = _facade_state()
    if not state.get("ok"):
        return {
            "ok": False,
            "status": "app_import_failed",
            "error": "app_import_failed",
            "reason": state.get("error", ""),
            "exception_type": state.get("exception_type", ""),
        }
    module = app_runtime_facade.get_app()
    if module is None:
        return {"ok": False, "status": "app_import_failed", "error": "app_import_failed"}
    return getattr(module, name)(*args, **kwargs)


def load_tg_state() -> Dict[str, Any]:
    result = _facade_call("load_tg_state")
    return result if isinstance(result, dict) and result.get("status") != "app_import_failed" else TG_STATE


def save_tg_state(state: Dict[str, Any]) -> Any:
    result = _facade_call("save_tg_state", state)
    if isinstance(result, dict) and result.get("status") == "app_import_failed":
        TG_STATE.clear()
        TG_STATE.update(state)
    return result


def load_monitoring() -> List[Dict[str, Any]]:
    result = _facade_call("load_monitoring")
    return result if isinstance(result, list) else []


def load_portfolio() -> List[Dict[str, Any]]:
    result = _facade_call("load_portfolio")
    return result if isinstance(result, list) else []


def save_monitoring(rows: List[Dict[str, Any]]) -> Any:
    return _facade_call("save_monitoring", rows)


def save_portfolio(rows: List[Dict[str, Any]]) -> Any:
    return _facade_call("save_portfolio", rows)


def get_worker_runtime_state() -> Dict[str, Any]:
    result = app_runtime_facade.get_worker_runtime_state()
    return result if isinstance(result, dict) else {}


def get_job_heartbeats_snapshot() -> Any:
    return _facade_call("get_job_heartbeats_snapshot")


def manual_add_token_to_monitoring(*args: Any, **kwargs: Any) -> Any:
    return _facade_call("manual_add_token_to_monitoring", *args, **kwargs)


def suppress_token(*args: Any, **kwargs: Any) -> Any:
    return _facade_call("suppress_token", *args, **kwargs)


def trigger_digest_notification(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    result = app_runtime_facade.trigger_digest_notification(*args, **kwargs)
    return result if isinstance(result, dict) else {"ok": False, "error": "bad_digest_result"}


def build_active_monitoring_rows(*args: Any, **kwargs: Any) -> Any:
    return _facade_call("build_active_monitoring_rows", *args, **kwargs)


def build_notification_candidates(*args: Any, **kwargs: Any) -> Any:
    return _facade_call("build_notification_candidates", *args, **kwargs)


def normalize_timing_label(*args: Any, **kwargs: Any) -> Any:
    return _facade_call("normalize_timing_label", *args, **kwargs)


def send_telegram(*args: Any, **kwargs: Any) -> Any:
    return _facade_call("send_telegram", *args, **kwargs)

def _app_import_failed_response() -> Dict[str, Any]:
    state = _facade_state()
    return {"ok": False, "error": "app_import_failed", "detail": state.get("error", "")}


def _require_bootstrap() -> None:
    state = _facade_state()
    if not state.get("ok"):
        raise RuntimeError(
            "bootstrap_unavailable:"
            f"{state.get('exception_type')}:"
            f"{state.get('error')}"
        )


app = FastAPI()


def _raise_bootstrap_http_500() -> None:
    _require_bootstrap()


def tg_token() -> str:
    return os.getenv("TG_BOT_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")


def summary_key_ok(value: str) -> bool:
    expected = os.getenv("TG_SUMMARY_KEY", "").strip()
    supplied = str(value or "")
    return bool(expected) and hmac.compare_digest(supplied, expected)


def tg_api(method: str, payload: dict) -> dict:
    token = tg_token()
    if not token:
        return {"ok": False, "error": "missing_bot_token"}

    r = requests.post(
        f"https://api.telegram.org/bot{token}/{method}",
        json=payload,
        timeout=15,
    )
    try:
        return r.json()
    except Exception:
        return {"ok": False, "status_code": r.status_code, "text": r.text}


def _norm_contract(chain: str, ca: str) -> Tuple[str, str]:
    norm_chain = normalize_chain_name(chain or "")
    norm_ca = addr_store(norm_chain, ca or "")
    return norm_chain, norm_ca


def _touch_active(row: Dict[str, Any], note: str) -> None:
    row["active"] = "1"
    row["archived"] = "0"
    row["updated_at"] = now_utc_str()
    row["note"] = note


ADDR_ALIAS_FIELDS = (
    "base_token_address",
    "base_addr",
    "token_addr",
    "token_address",
    "ca",
    "address",
    "pair_address",
)


def _row_matches_token(row: Dict[str, Any], chain: str, ca: str) -> bool:
    row_chain = normalize_chain_name(row.get("chain", ""))
    norm_chain, norm_ca = _norm_contract(chain, ca)
    if not row_chain or row_chain != norm_chain or not norm_ca:
        return False
    for field in ADDR_ALIAS_FIELDS:
        value = str(row.get(field) or "").strip()
        if value and addr_store(norm_chain, value) == norm_ca:
            return True
    return False


def _save_result_ok(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, dict):
        return bool(result.get("ok", True))
    return bool(result)


def _processed_callback_store(state: Dict[str, Any]) -> Dict[str, float]:
    store = state.setdefault("processed_callback_ids", {})
    if not isinstance(store, dict):
        store = {}
        state["processed_callback_ids"] = store
    return store


def _prune_processed_callback_ids(state: Dict[str, Any], now: float | None = None) -> None:
    now = time.time() if now is None else now
    store = _processed_callback_store(state)
    expired = [cb_id for cb_id, ts in store.items() if now - float(ts or 0.0) > CALLBACK_ID_TTL_SECONDS]
    for cb_id in expired:
        store.pop(cb_id, None)
    overflow = len(store) - CALLBACK_ID_MAX_ITEMS
    if overflow > 0:
        oldest = sorted(store.items(), key=lambda kv: kv[1])[:overflow]
        for cb_id, _ in oldest:
            store.pop(cb_id, None)


def _load_callback_state() -> Dict[str, Any]:
    try:
        state = load_tg_state()
        if isinstance(state, dict):
            return state
    except Exception as e:
        print(f"[TG_WEBHOOK] callback state load fallback {type(e).__name__}: {e}", flush=True)
    return TG_STATE


def _save_callback_state(state: Dict[str, Any]) -> None:
    try:
        save_tg_state(state)
    except Exception as e:
        print(f"[TG_WEBHOOK] callback state save fallback {type(e).__name__}: {e}", flush=True)
        TG_STATE.clear()
        TG_STATE.update(state)


def _mark_callback_processed(cb_id: Any) -> bool:
    cb_key = str(cb_id or "").strip()
    if not cb_key:
        return False
    now = time.time()
    state = _load_callback_state()
    _prune_processed_callback_ids(state, now)
    store = _processed_callback_store(state)
    if cb_key in store:
        _save_callback_state(state)
        return True
    store[cb_key] = now
    _prune_processed_callback_ids(state, now)
    _save_callback_state(state)
    return False

def _canonical_callback_action(action: str) -> str:
    return ACTION_ALIASES.get(str(action or "").strip()) or ""


def _validate_callback_data(action: str, chain: str, ca: str) -> Tuple[bool, str, str, str]:
    canonical_action = _canonical_callback_action(action)
    if not canonical_action:
        return False, "invalid_action", "", ""
    norm_chain = normalize_chain_name(chain)
    if norm_chain not in VALID_CHAINS:
        return False, "invalid_chain", "", ""
    norm_ca = addr_store(norm_chain, ca)
    if not CA_RE.fullmatch(norm_ca):
        return False, "invalid_ca", "", ""
    return True, canonical_action, norm_chain, norm_ca

def _clear_token_meta_cache() -> None:
    TG_META_CACHE["ts"] = 0.0
    TG_META_CACHE["index"] = {}


def _token_meta_index_key(chain: str, ca: str) -> str:
    norm_chain, norm_ca = _norm_contract(chain, ca)
    if not norm_chain or not norm_ca:
        return ""
    return f"{norm_chain}:{norm_ca}"


def _meta_from_row(row: Dict[str, Any], source: str) -> Dict[str, str]:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "").strip()
    name = str(row.get("name") or row.get("base_name") or symbol or "").strip()
    return {"symbol": symbol, "name": name, "source": source}


def _index_meta_rows(index: Dict[str, Dict[str, str]], rows: List[Dict[str, Any]], source: str) -> None:
    for row in rows:
        norm_chain = normalize_chain_name(row.get("chain", ""))
        if not norm_chain:
            continue
        meta = _meta_from_row(row, source)
        for field in ADDR_ALIAS_FIELDS:
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            norm_ca = addr_store(norm_chain, value)
            if not norm_ca:
                continue
            index.setdefault(f"{norm_chain}:{norm_ca}", meta)


def _build_token_meta_index() -> Dict[str, Dict[str, str]]:
    if not _app_available():
        return {}
    index: Dict[str, Dict[str, str]] = {}
    _index_meta_rows(index, load_monitoring(), "monitoring")
    _index_meta_rows(index, load_portfolio(), "portfolio")
    return index


def find_token_meta(chain: str, ca: str) -> Dict[str, str]:
    key = _token_meta_index_key(chain, ca)
    if not key:
        return {"symbol": "", "name": ""}

    now = time.time()
    cache_ts = float(TG_META_CACHE.get("ts") or 0.0)
    index = TG_META_CACHE.get("index")
    if not isinstance(index, dict) or (now - cache_ts) >= TG_META_CACHE_TTL_SEC:
        index = _build_token_meta_index()
        if not (IMPORT_FAILED or BOOTSTRAP_ERROR):
            TG_META_CACHE["ts"] = now
            TG_META_CACHE["index"] = index

    meta = index.get(key) if isinstance(index, dict) else None
    if not isinstance(meta, dict):
        return {"symbol": "", "name": ""}
    return {"symbol": str(meta.get("symbol") or ""), "name": str(meta.get("name") or "")}

def add_contract_to_portfolio(chain: str, ca: str) -> bool:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    if not chain or not ca:
        return False

    rows = load_portfolio()
    for row in rows:
        if _row_matches_token(row, chain, ca):
            _touch_active(row, "added_from_tg_callback")
            return _save_result_ok(save_portfolio(rows))

    new_row: Dict[str, Any] = {k: "" for k in PORTFOLIO_FIELDS}
    new_row.update(
        {
            "chain": chain,
            "base_token_address": ca,
            "active": "1",
            "archived": "0",
            "updated_at": now_utc_str(),
            "note": "added_from_tg_callback",
        }
    )
    rows.append(new_row)
    return _save_result_ok(save_portfolio(rows))

def add_contract_to_monitoring(chain: str, ca: str) -> bool:
    chain = normalize_chain_name(chain)
    res = manual_add_token_to_monitoring(
        raw_input=ca,
        chain=chain,
        note="added_from_tg_callback",
        tags=["manual_watch"],
        source="tg_callback",
    )
    return str(res.get("status") or "") in {"OK", "OK_DEFERRED", "EXISTS_ACTIVE", "REACTIVATED"}


def remove_contract_everywhere_atomicish(chain: str, ca: str) -> dict:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    result = {
        "ok": False,
        "portfolio_changed": False,
        "monitoring_changed": False,
        "portfolio_saved": False,
        "monitoring_saved": False,
        "error": None,
    }
    if not chain or not ca:
        result["error"] = "invalid_contract"
        return result

    try:
        p_rows = load_portfolio()
        m_rows = load_monitoring()
    except Exception as e:
        result["error"] = f"load_failed:{type(e).__name__}:{e}"
        return result

    for row in p_rows:
        if _row_matches_token(row, chain, ca):
            row["active"] = "0"
            row["archived"] = "1"
            row["note"] = "removed_from_tg_callback"
            row["updated_at"] = now_utc_str()
            result["portfolio_changed"] = True

    for row in m_rows:
        if _row_matches_token(row, chain, ca):
            row["active"] = "0"
            row["archived"] = "1"
            row["status"] = "ARCHIVED"
            row["archived_reason"] = "removed_from_tg_callback"
            row["updated_at"] = now_utc_str()
            result["monitoring_changed"] = True

    try:
        result["portfolio_saved"] = True if not result["portfolio_changed"] else _save_result_ok(save_portfolio(p_rows))
    except Exception as e:
        result["portfolio_saved"] = False
        result["error"] = f"portfolio_save_failed:{type(e).__name__}:{e}"
        return result
    if not result["portfolio_saved"]:
        result["error"] = "portfolio_save_failed"
        return result

    if result["portfolio_changed"]:
        try:
            persisted = any(_row_matches_token(row, chain, ca) and str(row.get("active") or "") == "0" and str(row.get("archived") or "") == "1" for row in load_portfolio())
        except Exception as e:
            result["error"] = f"portfolio_verify_failed:{type(e).__name__}:{e}"
            return result
        if not persisted:
            result["portfolio_saved"] = False
            result["error"] = "portfolio_verify_failed"
            return result

    try:
        result["monitoring_saved"] = True if not result["monitoring_changed"] else _save_result_ok(save_monitoring(m_rows))
    except Exception as e:
        result["monitoring_saved"] = False
        result["error"] = f"monitoring_save_failed:{type(e).__name__}:{e}"
        return result
    if not result["monitoring_saved"]:
        result["error"] = "monitoring_save_failed"
        return result

    result["ok"] = True
    return result


def remove_contract_everywhere(chain: str, ca: str) -> bool:
    return bool(remove_contract_everywhere_atomicish(chain, ca).get("ok"))

def _status_text(action: str, chain: str, ca: str) -> str:
    chain_label = normalize_chain_name(chain).upper()
    ca_safe = escape(ca)
    if action == "pf":
        header = "Added to portfolio"
    elif action == "mn":
        header = "Added to monitoring"
    elif action == "rm":
        header = "Removed / archived"
    else:
        header = "Action processed"

    return f"{header}\nchain: {chain_label}\nCA:\n<pre>{ca_safe}</pre>"


def _do_action(action: str) -> Callable[[str, str], bool]:
    return {
        "pf": add_contract_to_portfolio,
        "mn": add_contract_to_monitoring,
        "rm": remove_contract_everywhere,
    }.get(action, lambda _chain, _ca: False)


@app.get("/_import_status")
def import_status():
    state = _facade_state()
    failed = not bool(state.get("ok"))
    return {
        "ok": not failed,
        "import_failed": failed,
        "error": str(state.get("error") or ""),
        "exception_type": str(state.get("exception_type") or ""),
        "worker_fast_mode": bool(state.get("worker_fast_mode")),
        "app_module_loaded": bool(state.get("app_module_loaded")),
        "bootstrap_error": dict(BOOTSTRAP_ERROR),
    }


@app.get("/")
def root():
    if not _app_available():
        return _app_import_failed_response()
    return {"ok": True, "service": "dex-scout-tg-webhook"}


@app.get("/health")
def health():
    if not _app_available():
        return _app_import_failed_response()
    return {"ok": True}


@app.head("/health")
def health_head():
    return Response(status_code=500 if not _app_available() else 200)


@app.get("/runtime")
def runtime():
    if not _app_available():
        return _app_import_failed_response()
    try:
        _raise_bootstrap_http_500()
    except Exception:
        return _app_import_failed_response()
    try:
        runtime_state = get_worker_runtime_state() or {}
        if not isinstance(runtime_state, dict):
            runtime_state = {}
    except Exception as e:
        runtime_state = {"_runtime_error": f"{type(e).__name__}:{e}"}

    try:
        heartbeats, hb_status = get_job_heartbeats_snapshot()
        if not isinstance(heartbeats, dict):
            heartbeats = {}
    except Exception as e:
        heartbeats = {}
        hb_status = {"ok": False, "code": "heartbeat_read_exception", "detail": f"{type(e).__name__}:{e}"}

    return {
        "ok": True,
        "runtime": {
            "last_loop_ts": runtime_state.get("last_loop_ts") or "",
            "last_notifications_ts": runtime_state.get("last_notifications_ts") or "",
            "last_send_success_ts": runtime_state.get("last_send_success_ts") or "",
            "last_error_ts": runtime_state.get("last_error_ts") or "",
            "last_error_reason": runtime_state.get("last_error_reason") or "",
            "last_empty_reason": runtime_state.get("last_empty_reason") or "",
            "last_send_failure_reason": runtime_state.get("last_send_failure_reason") or "",
            "last_portfolio_alert": runtime_state.get("last_portfolio_alert") or {},
            "last_portfolio_alert_ts": runtime_state.get("last_portfolio_alert_ts") or "",
            "last_portfolio_alert_reason": runtime_state.get("last_portfolio_alert_reason") or "",
            "last_suppressed_reason": runtime_state.get("last_suppressed_reason") or "",
            "last_digest_suppressed_ts": runtime_state.get("last_digest_suppressed_ts") or "",
            "last_digest_suppressed_reason": runtime_state.get("last_digest_suppressed_reason") or "",
            "last_portfolio_heartbeat_ts": runtime_state.get("last_portfolio_heartbeat_ts") or "",
            "last_heartbeat_ts": runtime_state.get("last_heartbeat_ts") or "",
            "worker_status": runtime_state.get("worker_status") or "",
        },
        "job_heartbeats": heartbeats,
        "job_heartbeats_status": hb_status if isinstance(hb_status, dict) else {"ok": False, "code": "bad_status_payload"},
    }


@app.post("/tg_webhook")
async def tg_webhook(req: Request):
    if not _app_available():
        return _app_import_failed_response()
    try:
        _raise_bootstrap_http_500()
    except Exception:
        return _app_import_failed_response()
    try:
        data = await req.json()
        cb = data.get("callback_query")
        if not cb:
            return {"ok": True}

        cb_id = cb.get("id")
        msg = cb.get("message", {})
        chat_id = ((msg.get("chat") or {}).get("id"))
        message_id = msg.get("message_id")
        raw = str(cb.get("data") or "").strip()
        parts = raw.split("|", 2)
        if len(parts) != 3:
            tg_api(
                "answerCallbackQuery",
                {
                    "callback_query_id": cb_id,
                    "text": "Ignored",
                    "show_alert": False,
                },
            )
            return {"ok": True, "ignored": True, "reason": "bad_callback_data"}

        action, raw_chain, raw_ca = parts[0].strip(), parts[1].strip(), parts[2].strip()
        valid, action_or_reason, chain, ca = _validate_callback_data(action, raw_chain, raw_ca)
        if not valid:
            tg_api(
                "answerCallbackQuery",
                {
                    "callback_query_id": cb_id,
                    "text": "Rejected",
                    "show_alert": False,
                },
            )
            return {"ok": False, "error": action_or_reason}

        duplicate = _mark_callback_processed(cb_id)
        if duplicate:
            tg_api(
                "answerCallbackQuery",
                {
                    "callback_query_id": cb_id,
                    "text": "Already processed",
                    "show_alert": False,
                },
            )
            return {"ok": True, "duplicate": True}

        entity_key = canonical_entity_key(chain, ca)
        print(f"[TG_WEBHOOK] callback raw={raw} action={action_or_reason} chain={chain} ca={ca} entity_key={entity_key}", flush=True)

        meta = find_token_meta(chain, ca)
        token_label = meta.get("symbol") or meta.get("name") or ca[:8]
        print(f"[TG_WEBHOOK] token_label={token_label}", flush=True)

        result_text = "Ignored"
        action_ok = False
        if action_or_reason == "pf":
            action_ok = add_contract_to_portfolio(chain, ca)
            result_text = "Added to portfolio" if action_ok else "Portfolio add failed"
        elif action_or_reason == "mn":
            action_ok = add_contract_to_monitoring(chain, ca)
            result_text = "Added to monitoring" if action_ok else "Monitoring add failed"
        elif action_or_reason == "rm":
            remove_result = remove_contract_everywhere_atomicish(chain, ca)
            action_ok = bool(remove_result.get("ok"))
            if action_ok:
                suppress_token(chain, ca, reason="manual_remove_from_tg", days=30)
            result_text = "Removed / archived" if action_ok else f"Remove failed: {remove_result.get('error') or 'unknown'}"

        print(f"[TG_WEBHOOK] result={result_text}", flush=True)

        tg_api(
            "answerCallbackQuery",
            {
                "callback_query_id": cb_id,
                "text": result_text[:180],
                "show_alert": False,
            },
        )

        safe_chain = escape(str(chain or "").upper())
        safe_ca = escape(str(ca or ""))
        safe_label = escape(str(token_label or ""))
        result_title = escape(result_text)
        if not safe_label:
            safe_label = safe_ca[:8]

        if chat_id and message_id:
            resp = tg_api(
                "editMessageText",
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": (
                        f"<b>{result_title}</b>\n"
                        f"chain: {safe_chain}\n"
                        f"token: {safe_label}\n"
                        f"CA:\n<code>{safe_ca}</code>"
                    ),
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if not resp.get("ok"):
                tg_api(
                    "editMessageReplyMarkup",
                    {
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "reply_markup": {"inline_keyboard": []},
                    },
                )
        return {"ok": action_ok}

    except Exception as e:
        print(f"[tg_webhook] error: {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "error": "callback_failed", "detail": str(e)}


@app.post("/tg/webhook")
async def tg_webhook_alias(req: Request):
    return await tg_webhook(req)


@app.get("/tg/summary")
def tg_summary(key: str):
    if not _app_available():
        return _app_import_failed_response()
    try:
        _raise_bootstrap_http_500()
    except Exception:
        return _app_import_failed_response()
    if not summary_key_ok(key):
        return {"ok": False, "error": "forbidden"}
    result = trigger_digest_notification(trigger_source="tg_summary", cooldown_seconds=3600, force=False)
    return {
        "ok": bool(result.get("ok")),
        "sent": bool(result.get("sent")),
        "duplicate": bool(result.get("duplicate")),
        "event_type": str(result.get("event_type") or "digest"),
    }


@app.get("/tg/digest")
def tg_digest(key: str, force: int = 0):
    if not _app_available():
        return _app_import_failed_response()
    try:
        _raise_bootstrap_http_500()
    except Exception:
        return _app_import_failed_response()
    if not summary_key_ok(key):
        return {"ok": False, "error": "forbidden"}
    result = trigger_digest_notification(
        trigger_source="tg_digest",
        cooldown_seconds=3600,
        force=bool(int(force or 0)),
    )
    return {
        "ok": bool(result.get("ok")),
        "sent": bool(result.get("sent")),
        "duplicate": bool(result.get("duplicate")),
        "event_type": str(result.get("event_type") or "digest"),
    }
