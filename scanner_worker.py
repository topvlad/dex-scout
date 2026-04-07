import os
import time
from typing import Any, Dict, List, Optional

import requests

import app

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "600"))
SCAN_CHAIN = os.getenv("SCAN_CHAIN", "solana")
SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "100"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "50"))

TG_MAX_PER_HOUR = int(os.getenv("TG_MAX_PER_HOUR", "5"))
TG_SCAN_COOLDOWN_SEC = int(os.getenv("TG_SCAN_COOLDOWN_SEC", "600"))

LAST_STATUS: Dict[str, str] = {}

tg_state: Dict[str, Any] = app.load_sent_state()

def _parse_score(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("entry_score", 0) or 0)
    except Exception:
        return 0.0


def _risk(row: Dict[str, Any]) -> str:
    return str(row.get("risk") or row.get("risk_level") or "").upper().strip()


def _addr(row: Dict[str, Any]) -> str:
    return str(
        row.get("pair_address")
        or row.get("pairAddress")
        or row.get("base_addr")
        or row.get("base_token_address")
        or ""
    ).strip()


def tg_should_fire() -> bool:
    now = time.time()
    last = float(tg_state.get("last_run", 0.0) or 0.0)
    if now - last < TG_SCAN_COOLDOWN_SEC:
        return False
    tg_state["last_run"] = now
    app.save_sent_state(tg_state)
    return True


def can_send_tg() -> bool:
    sent = tg_state.get("sent_ts", [])
    now = time.time()
    sent = [float(ts) for ts in sent if now - float(ts) < 3600]
    tg_state["sent_ts"] = sent
    return len(sent) < TG_MAX_PER_HOUR


def mark_sent_ts() -> None:
    sent = tg_state.get("sent_ts", [])
    sent.append(time.time())
    tg_state["sent_ts"] = sent[-300:]


def send_telegram(text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
    token = (os.getenv("TG_BOT_TOKEN") or app._get_secret("TG_BOT_TOKEN", "")).strip()
    chat_id = (os.getenv("TG_CHAT_ID") or app._get_secret("TG_CHAT_ID", "")).strip()
    if not token or not chat_id:
        return False
    if not can_send_tg():
        return False
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=10,
        )
        ok = r.status_code == 200
        if ok:
            mark_sent_ts()
        return ok
    except Exception:
        return False


def is_valid(row: Dict[str, Any]) -> bool:
    score = _parse_score(row)
    risk = _risk(row)
    if score <= 0:
        return False
    if not risk or risk == "UNKNOWN":
        return False
    if not _addr(row):
        return False
    return True


def build_signal(row: Dict[str, Any], source: str) -> Optional[Dict[str, str]]:
    score = _parse_score(row)
    if score >= 320:
        return {
            "type": "ENTRY NOW",
            "priority": "HIGH",
            "reason": "strong momentum + structure",
        }
    if 260 <= score < 320:
        return {
            "type": "WATCH ENTRY",
            "priority": "MEDIUM",
            "reason": "early signal forming",
        }
    if source == "portfolio" and score < 140:
        return {
            "type": "EXIT",
            "priority": "HIGH",
            "reason": "structure breakdown",
        }
    return None


def was_sent(key: str) -> bool:
    return key in set(tg_state.get("sent_events", []))


def mark_sent(key: str) -> None:
    sent_events = list(tg_state.get("sent_events", []))
    sent_events.append(key)
    tg_state["sent_events"] = sent_events[-200:]


def fmt(row: Dict[str, Any], signal: Dict[str, str], source: str) -> str:
    ca = _addr(row) or "N/A"
    symbol = str(row.get("base_symbol") or row.get("symbol") or "UNKNOWN").upper()
    timing = str(row.get("timing") or row.get("timing_label") or "UNKNOWN")
    return (
        f"🚀 {signal['type']} | {symbol}\n\n"
        f"source: {source}\n"
        f"score: {round(_parse_score(row), 1)}\n"
        f"risk: {_risk(row)}\n"
        f"timing: {timing}\n\n"
        f"reason:\n{signal['reason']}\n\n"
        f"CA:\n<code>{ca}</code>"
    )


def tg_buttons(row: Dict[str, Any]) -> Dict[str, Any]:
    ca = _addr(row)
    return {
        "inline_keyboard": [
            [
                {"text": "➕ Portfolio", "callback_data": f"add_pf|{ca}"},
                {"text": "👀 Monitor", "callback_data": f"add_mon|{ca}"},
            ],
            [{"text": "❌ Remove", "callback_data": f"remove|{ca}"}],
        ]
    }


def run_signal_engine(rows: List[Dict[str, Any]], portfolio: List[Dict[str, Any]]) -> None:
    if not tg_should_fire():
        return

    sent = 0

    for row in rows:
        if not is_valid(row):
            continue
        signal = build_signal(row, "monitoring")
        if not signal:
            continue
        key = f"{_addr(row)}_{signal['type']}"
        if was_sent(key):
            continue
        if send_telegram(fmt(row, signal, "monitoring"), reply_markup=tg_buttons(row)):
            mark_sent(key)
            sent += 1
            LAST_STATUS[_addr(row)] = signal["type"]
        if sent >= 3:
            break

    for row in portfolio:
        if not is_valid(row):
            continue
        signal = build_signal(row, "portfolio")
        if not signal:
            continue
        key = f"{_addr(row)}_{signal['type']}"
        if was_sent(key):
            continue
        if send_telegram(fmt(row, signal, "portfolio"), reply_markup=tg_buttons(row)):
            mark_sent(key)
            sent += 1
            LAST_STATUS[_addr(row)] = signal["type"]
        if sent >= 5:
            break

    app.save_sent_state(tg_state)


def run_full_scan() -> Dict[str, Any]:
    return app.run_full_ingestion_now(
        chain=SCAN_CHAIN,
        seeds_raw=SCANNER_SEEDS,
        max_items=SCANNER_MAX_ITEMS,
        use_birdeye_trending=USE_BIRDEYE_TRENDING,
        birdeye_limit=BIRDEYE_LIMIT,
    )


def run_worker_loop() -> None:
    app.ensure_storage()
    while True:
        try:
            stats = run_full_scan()
            print(f"[worker] scan done: {stats}", flush=True)

            monitoring = app.load_monitoring()
            portfolio = app.load_portfolio()
            run_signal_engine(monitoring, portfolio)
        except Exception as exc:
            print(f"[worker] error: {type(exc).__name__}: {exc}", flush=True)

        time.sleep(max(60, SCAN_INTERVAL_SEC))


if __name__ == "__main__":
    run_worker_loop()
