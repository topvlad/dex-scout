import os
from html import escape
from typing import Any, Callable, Dict, List, Tuple

import requests
from fastapi import FastAPI, Request

try:
    from app import (
        MON_FIELDS,
        PORTFOLIO_FIELDS,
        addr_store,
        load_monitoring,
        load_portfolio,
        normalize_chain_name,
        now_utc_str,
        save_monitoring,
        save_portfolio,
    )
except Exception as e:
    print(f"[tg_webhook] helper import failed: {e}", flush=True)
    MON_FIELDS = ["chain", "base_addr", "active", "archived", "status", "updated_at", "note", "archived_reason"]
    PORTFOLIO_FIELDS = ["chain", "base_token_address", "active", "archived", "updated_at", "note"]

    def load_monitoring() -> List[Dict[str, Any]]:
        return []

    def load_portfolio() -> List[Dict[str, Any]]:
        return []

    def save_monitoring(rows: List[Dict[str, Any]]) -> None:
        _ = rows

    def save_portfolio(rows: List[Dict[str, Any]]) -> None:
        _ = rows

    def normalize_chain_name(raw_chain: Any) -> str:
        return str(raw_chain or "").strip().lower()

    def addr_store(chain: str, addr: str) -> str:
        _ = chain
        return str(addr or "").strip()

    def now_utc_str() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


app = FastAPI()


def tg_token() -> str:
    return os.getenv("TG_BOT_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")


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


def add_contract_to_portfolio(chain: str, ca: str) -> bool:
    chain, ca = _norm_contract(chain, ca)
    if not chain or not ca:
        return False

    rows = load_portfolio()
    for row in rows:
        row_chain, row_ca = _norm_contract(row.get("chain", ""), row.get("base_token_address", ""))
        if row_chain == chain and row_ca == ca:
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
    chain, ca = _norm_contract(chain, ca)
    if not chain or not ca:
        return False

    rows = load_monitoring()
    for row in rows:
        row_chain, row_ca = _norm_contract(row.get("chain", ""), row.get("base_addr", ""))
        if row_chain == chain and row_ca == ca:
            _touch_active(row, "added_from_tg_callback")
            row["status"] = "ACTIVE"
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
    chain, ca = _norm_contract(chain, ca)
    if not chain or not ca:
        return False

    p_rows = load_portfolio()
    for row in p_rows:
        row_chain, row_ca = _norm_contract(row.get("chain", ""), row.get("base_token_address", ""))
        if row_chain == chain and row_ca == ca:
            row["active"] = "0"
            row["archived"] = "1"
            row["note"] = "removed_from_tg_callback"
            row["updated_at"] = now_utc_str()
    save_portfolio(p_rows)

    m_rows = load_monitoring()
    for row in m_rows:
        row_chain, row_ca = _norm_contract(row.get("chain", ""), row.get("base_addr", ""))
        if row_chain == chain and row_ca == ca:
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

    return f"{header}\nchain: {chain_label}\nCA:\n<code>{ca_safe}</code>"


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


@app.post("/tg/webhook")
async def tg_webhook(req: Request):
    try:
        data = await req.json()
        cb = data.get("callback_query")
        if not cb:
            return {"ok": True}

        cb_id = cb.get("id")
        msg = cb.get("message", {})
        chat_id = ((msg.get("chat") or {}).get("id"))
        message_id = msg.get("message_id")
        raw = cb.get("data", "")

        action = ""
        chain = ""
        ca = ""
        parts = str(raw).split("|", 2)
        if len(parts) == 3:
            action, chain, ca = parts

        ok = False
        if action and chain and ca:
            func = _do_action(action)
            ok = bool(func(chain, ca))
        else:
            print(f"[tg_webhook] malformed callback data: {raw}", flush=True)

        if cb_id:
            tg_api(
                "answerCallbackQuery",
                {
                    "callback_query_id": cb_id,
                    "text": "Done" if ok else "Ignored",
                    "show_alert": False,
                },
            )

        if chat_id is not None and message_id is not None and action in {"pf", "mn", "rm"} and chain and ca:
            text = _status_text(action, chain, ca)
            res = tg_api(
                "editMessageText",
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if not res.get("ok"):
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
