import os
from html import escape
from typing import Any, Callable, Dict, List, Tuple

import requests
from fastapi import FastAPI, Request, Response

try:
    from app import (
        MON_FIELDS,
        PORTFOLIO_FIELDS,
        addr_store,
        build_active_monitoring_rows,
        build_notification_candidates,
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

    def parse_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def normalize_timing_label(raw: Any) -> str:
        return str(raw or "NEUTRAL").upper()

    def build_active_monitoring_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in rows if str(r.get("active", "1")).strip() == "1"]

    def build_notification_candidates(
        monitoring_rows: List[Dict[str, Any]],
        portfolio_rows: List[Dict[str, Any]],
        limit_monitoring: int = 12,
        limit_portfolio: int = 8,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        _ = limit_monitoring, limit_portfolio
        return monitoring_rows, portfolio_rows

    def send_telegram(text: str, parse_mode: str = "HTML", reply_markup: Dict[str, Any] = None) -> bool:
        _ = text, parse_mode, reply_markup
        return False

    def suppress_token(chain: str, ca: str, reason: str = "manual_remove", days: int = 30) -> None:
        _ = chain, ca, reason, days
        return None


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
        print(f"[TG_WEBHOOK] callback raw={raw} action={action} chain={chain} ca={ca}", flush=True)

        meta = find_token_meta(chain, ca)
        token_label = meta.get("symbol") or meta.get("name") or ca[:8]
        print(f"[TG_WEBHOOK] token_label={token_label}", flush=True)

        result_text = "Ignored"
        if action in ("pf", "pf_add", "portfolio", "portfolio_add"):
            add_contract_to_portfolio(chain, ca)
            result_text = f"Added to portfolio | {token_label}"
        elif action in ("mn", "mon_add", "monitor", "monitor_add"):
            add_contract_to_monitoring(chain, ca)
            result_text = f"Added to monitoring | {token_label}"
        elif action in ("rm", "remove", "delete", "archive"):
            remove_contract_everywhere(chain, ca)
            suppress_token(chain, ca, reason="manual_remove_from_tg", days=30)
            result_text = f"Removed / archived | {token_label}"

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


@app.get("/tg/summary")
def tg_summary(key: str):
    if not summary_key_ok(key):
        return {"ok": False, "error": "forbidden"}

    portfolio_rows = load_portfolio()
    monitoring_rows = load_monitoring()

    active_portfolio = [r for r in portfolio_rows if str(r.get("active", "1")).strip() == "1"]
    mon_rows, _ = build_notification_candidates(monitoring_rows, portfolio_rows, limit_monitoring=5, limit_portfolio=5)

    top_mon = mon_rows[:5]
    top_lines = []
    for r in top_mon:
        top_lines.append(
            f"{str(r.get('base_symbol') or r.get('symbol') or 'TOKEN')}: "
            f"{parse_float(r.get('entry_score', 0), 0.0)} / "
            f"{normalize_timing_label(str(r.get('timing_label') or 'NEUTRAL'))}"
        )

    text = (
        f"<b>DEX Scout summary</b>\n"
        f"portfolio active: {len(active_portfolio)}\n"
        f"monitoring active: {len(build_active_monitoring_rows(monitoring_rows))}\n\n"
        f"<b>Top monitoring now:</b>\n"
        + ("\n".join(top_lines) if top_lines else "no active candidates")
    )

    ok = send_telegram(text, parse_mode="HTML")
    return {"ok": ok, "sent": ok}
