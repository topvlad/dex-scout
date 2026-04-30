import os
from html import escape
from typing import Any, Callable, Dict, List, Tuple

import requests
from fastapi import FastAPI, Request, Response

BOOTSTRAP_ERROR: Dict[str, str] = {}

try:
    from app import (
        MON_FIELDS,
        PORTFOLIO_FIELDS,
        addr_store,
        canonical_entity_key,
        build_active_monitoring_rows,
        build_notification_candidates,
        get_job_heartbeats_snapshot,
        get_worker_runtime_state,
        load_monitoring,
        load_portfolio,
        normalize_chain_name,
        normalize_timing_label,
        now_utc_str,
        parse_float,
        save_monitoring,
        save_portfolio,
        send_telegram,
        suppress_token,
        trigger_digest_notification,
    )
except Exception as e:
    BOOTSTRAP_ERROR = {
        "module": "app",
        "helper": "bootstrap_import",
        "exception_type": type(e).__name__,
        "exception_text": str(e),
    }
    print(
        "[tg_webhook] bootstrap import failed "
        f"helper={BOOTSTRAP_ERROR['helper']} "
        f"type={BOOTSTRAP_ERROR['exception_type']} "
        f"text={BOOTSTRAP_ERROR['exception_text']}",
        flush=True,
    )
    MON_FIELDS = ["chain", "base_addr", "active", "archived", "status", "updated_at", "note", "archived_reason"]
    PORTFOLIO_FIELDS = ["chain", "base_token_address", "active", "archived", "updated_at", "note"]

    def normalize_chain_name(raw_chain: Any) -> str:
        return str(raw_chain or "").strip().lower()

    def addr_store(chain: str, addr: str) -> str:
        _ = chain
        return str(addr or "").strip()


def _require_bootstrap() -> None:
    if BOOTSTRAP_ERROR:
        raise RuntimeError(
            "bootstrap_unavailable:"
            f"{BOOTSTRAP_ERROR.get('helper')}:"
            f"{BOOTSTRAP_ERROR.get('exception_type')}:"
            f"{BOOTSTRAP_ERROR.get('exception_text')}"
        )


app = FastAPI()


def tg_token() -> str:
    return os.getenv("TG_BOT_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")


def summary_key_ok(value: str) -> bool:
    expected = os.getenv("TG_SUMMARY_KEY", "").strip()
    return bool(expected) and value == expected


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


def find_token_meta(chain: str, ca: str) -> Dict[str, str]:
    chain = normalize_chain_name(chain)
    ca_norm = addr_store(chain, ca)

    def match_addr(row: Dict[str, Any]) -> bool:
        vals = [
            row.get("base_addr"),
            row.get("base_token_address"),
            row.get("ca"),
            row.get("address"),
        ]
        return any(addr_store(chain, str(v or "")) == ca_norm for v in vals if str(v or "").strip())

    for row in load_monitoring():
        if match_addr(row):
            return {
                "symbol": str(row.get("base_symbol") or row.get("symbol") or "").strip(),
                "name": str(row.get("name") or row.get("base_symbol") or row.get("symbol") or "").strip(),
            }

    for row in load_portfolio():
        if match_addr(row):
            return {
                "symbol": str(row.get("base_symbol") or row.get("symbol") or "").strip(),
                "name": str(row.get("name") or row.get("base_symbol") or row.get("symbol") or "").strip(),
            }

    return {"symbol": "", "name": ""}


def add_contract_to_portfolio(chain: str, ca: str) -> bool:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    if not chain or not ca:
        return False

    rows = load_portfolio()
    for row in rows:
        row_chain = normalize_chain_name(row.get("chain", ""))
        vals = [row.get("base_token_address"), row.get("base_addr"), row.get("ca"), row.get("address")]
        if row_chain == chain and any(addr_store(chain, str(v or "")) == ca for v in vals if str(v or "").strip()):
            _touch_active(row, "added_from_tg_callback")
            save_portfolio(rows)
            return True

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
    save_portfolio(rows)
    return True


def add_contract_to_monitoring(chain: str, ca: str) -> bool:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    if not chain or not ca:
        return False

    rows = load_monitoring()
    for row in rows:
        row_chain = normalize_chain_name(row.get("chain", ""))
        vals = [row.get("base_addr"), row.get("base_token_address"), row.get("ca"), row.get("address")]
        if row_chain == chain and any(addr_store(chain, str(v or "")) == ca for v in vals if str(v or "").strip()):
            _touch_active(row, "added_from_tg_callback")
            row["status"] = "ACTIVE"
            row["updated_at"] = now_utc_str()
            save_monitoring(rows)
            return True

    new_row: Dict[str, Any] = {k: "" for k in MON_FIELDS}
    new_row.update(
        {
            "chain": chain,
            "base_addr": ca,
            "active": "1",
            "archived": "0",
            "status": "ACTIVE",
            "updated_at": now_utc_str(),
            "note": "added_from_tg_callback",
        }
    )
    rows.append(new_row)
    save_monitoring(rows)
    return True


def remove_contract_everywhere(chain: str, ca: str) -> bool:
    chain = normalize_chain_name(chain)
    ca = addr_store(chain, ca)
    if not chain or not ca:
        return False

    p_rows = load_portfolio()
    for row in p_rows:
        row_chain = normalize_chain_name(row.get("chain", ""))
        vals = [row.get("base_token_address"), row.get("base_addr"), row.get("ca"), row.get("address")]
        if row_chain == chain and any(addr_store(chain, str(v or "")) == ca for v in vals if str(v or "").strip()):
            row["active"] = "0"
            row["archived"] = "1"
            row["note"] = "removed_from_tg_callback"
            row["updated_at"] = now_utc_str()
    save_portfolio(p_rows)

    m_rows = load_monitoring()
    for row in m_rows:
        row_chain = normalize_chain_name(row.get("chain", ""))
        vals = [row.get("base_addr"), row.get("base_token_address"), row.get("ca"), row.get("address")]
        if row_chain == chain and any(addr_store(chain, str(v or "")) == ca for v in vals if str(v or "").strip()):
            row["active"] = "0"
            row["archived"] = "1"
            row["status"] = "ARCHIVED"
            row["archived_reason"] = "removed_from_tg_callback"
            row["updated_at"] = now_utc_str()
    save_monitoring(m_rows)

    return True


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


@app.get("/")
def root():
    return {"ok": True, "service": "dex-scout-tg-webhook"}


@app.get("/health")
def health():
    return {"ok": True}


@app.head("/health")
def health_head():
    return Response(status_code=200)


@app.get("/runtime")
def runtime():
    try:
        _require_bootstrap()
    except Exception as e:
        return {"ok": False, "error": str(e)}
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
            "worker_status": runtime_state.get("worker_status") or "",
        },
        "job_heartbeats": heartbeats,
        "job_heartbeats_status": hb_status if isinstance(hb_status, dict) else {"ok": False, "code": "bad_status_payload"},
    }


@app.post("/tg_webhook")
async def tg_webhook(req: Request):
    try:
        _require_bootstrap()
    except Exception as e:
        print(f"[tg_webhook] runtime bootstrap error: {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "error": str(e)}
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

        action, chain, ca = parts[0].strip(), parts[1].strip(), parts[2].strip()
        chain = normalize_chain_name(chain)
        ca = addr_store(chain, ca)
        entity_key = canonical_entity_key(chain, ca)
        print(f"[TG_WEBHOOK] callback raw={raw} action={action} chain={chain} ca={ca} entity_key={entity_key}", flush=True)

        meta = find_token_meta(chain, ca)
        token_label = meta.get("symbol") or meta.get("name") or ca[:8]
        print(f"[TG_WEBHOOK] token_label={token_label}", flush=True)

        result_text = "Ignored"
        if action in ("pf", "pf_add", "portfolio", "portfolio_add", "discovery_pf_add"):
            add_contract_to_portfolio(chain, ca)
            result_text = "Added to portfolio"
        elif action in ("mn", "mon_add", "monitor", "monitor_add", "discovery_mon_add"):
            add_contract_to_monitoring(chain, ca)
            result_text = "Added to monitoring"
        elif action in ("rm", "remove", "delete", "archive"):
            remove_contract_everywhere(chain, ca)
            suppress_token(chain, ca, reason="manual_remove_from_tg", days=30)
            result_text = "Removed / archived"

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

    except Exception as e:
        print(f"[tg_webhook] error: {type(e).__name__}: {e}", flush=True)

    return {"ok": True}


@app.post("/tg/webhook")
async def tg_webhook_alias(req: Request):
    return await tg_webhook(req)


@app.get("/tg/summary")
def tg_summary(key: str):
    try:
        _require_bootstrap()
    except Exception as e:
        return {"ok": False, "error": str(e)}
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
    try:
        _require_bootstrap()
    except Exception as e:
        return {"ok": False, "error": str(e)}
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
