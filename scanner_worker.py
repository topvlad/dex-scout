import os
import time
from typing import Any, Dict, List, Optional

import requests

import app

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
SCAN_CHAIN = os.getenv("SCAN_CHAIN", "solana")
SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "100"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "50"))

LAST_STATUS: Dict[str, str] = {}
SENT_TOKENS: set[str] = set()
PORTFOLIO_SENT: set[str] = set()
TG_LAST_SENT: List[float] = []
LAST_RUN_TS: float = 0.0


def send_telegram(text: str) -> bool:
    token = (os.getenv("TG_BOT_TOKEN") or app._get_secret("TG_BOT_TOKEN", "")).strip()
    chat_id = (os.getenv("TG_CHAT_ID") or app._get_secret("TG_CHAT_ID", "")).strip()
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def can_send_tg() -> bool:
    global TG_LAST_SENT
    now = time.time()
    TG_LAST_SENT = [t for t in TG_LAST_SENT if now - float(t) < 3600]
    return len(TG_LAST_SENT) < 5


def mark_sent() -> None:
    TG_LAST_SENT.append(time.time())


def format_token_msg(row: Dict[str, Any], action: str) -> str:
    symbol = str(row.get("base_symbol") or row.get("symbol") or "UNKNOWN").upper()
    score = round(float(row.get("entry_score", 0) or 0), 1)
    risk = str(row.get("risk") or row.get("risk_level") or "UNKNOWN").upper()
    addr = row.get("pair_address") or row.get("pairAddress") or row.get("base_addr") or "N/A"
    return (
        f"{action} | {symbol}\n\n"
        f"score: {score}\n"
        f"risk: {risk}\n\n"
        "CA:\n"
        f"`{addr}`"
    )


def is_top_signal(row: Dict[str, Any]) -> bool:
    score = float(row.get("entry_score", 0) or 0)
    risk = str(row.get("risk") or row.get("risk_level") or "").upper()
    if score < 260:
        return False
    if risk == "HIGH" and score < 320:
        return False
    return True


def process_signals(rows: List[Dict[str, Any]]) -> None:
    global LAST_RUN_TS
    now = time.time()
    if now - LAST_RUN_TS < 300:
        return
    LAST_RUN_TS = now

    rows = rows[:50] if len(rows) > 50 else rows
    top_candidates = sorted(
        [r for r in rows if is_top_signal(r)],
        key=lambda x: float(x.get("entry_score", 0) or 0),
        reverse=True,
    )[:3]

    for row in top_candidates:
        token = row.get("pair_address") or row.get("pairAddress") or row.get("base_addr")
        if not token or token in SENT_TOKENS:
            continue
        if not can_send_tg():
            break
        if send_telegram(format_token_msg(row, "🚀 SIGNAL")):
            mark_sent()
            SENT_TOKENS.add(str(token))
            LAST_STATUS[str(token)] = "SIGNAL_SENT"


def portfolio_signal(row: Dict[str, Any]) -> Optional[str]:
    score = float(row.get("entry_score", 0) or 0)
    if score > 320:
        return "ADD"
    if score < 140:
        return "EXIT"
    return None


def process_portfolio(rows: List[Dict[str, Any]]) -> None:
    rows = rows[:50] if len(rows) > 50 else rows
    for row in rows:
        action = portfolio_signal(row)
        if not action:
            continue
        token = str(row.get("pair_address") or row.get("pairAddress") or "").strip()
        if not token:
            continue
        key = f"{token}_{action}"
        if key in PORTFOLIO_SENT:
            continue
        if not can_send_tg():
            break
        if send_telegram(format_token_msg(row, f"📊 PORTFOLIO {action}")):
            mark_sent()
            PORTFOLIO_SENT.add(key)


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
            process_signals(monitoring)

            portfolio = app.load_portfolio()
            process_portfolio(portfolio)
        except Exception as exc:
            print(f"[worker] error: {type(exc).__name__}: {exc}", flush=True)

        time.sleep(max(60, SCAN_INTERVAL_SEC))


if __name__ == "__main__":
    run_worker_loop()
