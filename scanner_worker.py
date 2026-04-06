import os
import time
from typing import Any, Dict, List

import requests

import app

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
SCAN_CHAIN = os.getenv("SCAN_CHAIN", "solana")
SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "100"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "50"))

LAST_STATUS: Dict[str, str] = {}
LAST_PORTFOLIO: set[str] = set()


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


def classify_entry(score: float) -> str:
    if score >= 320:
        return "READY"
    if score >= 220:
        return "WATCH"
    if score >= 150:
        return "TRACK"
    return "IGNORE"


def is_signal_worthy(row: Dict[str, Any]) -> bool:
    score = float(row.get("entry_score", 0) or 0)
    risk = str(row.get("risk") or row.get("risk_level") or "").upper()
    if score < 200:
        return False
    if risk == "HIGH" and score < 300:
        return False
    return True


def process_signals(rows: List[Dict[str, Any]]) -> None:
    global LAST_STATUS

    for row in rows:
        if not is_signal_worthy(row):
            continue

        token = row.get("pair_address") or row.get("pairAddress") or row.get("base_addr")
        symbol = str(row.get("base_symbol") or token or "UNKNOWN")
        score = float(row.get("entry_score", 0) or 0)
        if not token:
            continue

        if token not in LAST_STATUS:
            send_telegram(f"🆕 NEW TOKEN\n{symbol}\nScore: {round(score,1)}")

        new_status = classify_entry(score)
        old_status = LAST_STATUS.get(token)
        if old_status != new_status and new_status in ("READY", "WATCH"):
            send_telegram(
                "📡 SIGNAL UPDATE\n"
                f"{symbol}\n\n"
                f"Status: {old_status or 'NEW'} → {new_status}\n"
                f"Score: {round(score,1)}"
            )

        LAST_STATUS[token] = new_status


def process_portfolio(rows: List[Dict[str, Any]]) -> None:
    global LAST_PORTFOLIO

    current = {
        str(p.get("pair_address") or "").strip()
        for p in rows
        if str(p.get("active", "1")).strip() == "1" and str(p.get("pair_address") or "").strip()
    }

    for t in current - LAST_PORTFOLIO:
        send_telegram(f"🟢 ADDED TO PORTFOLIO\n{t}")

    for t in LAST_PORTFOLIO - current:
        send_telegram(f"🔴 REMOVED FROM PORTFOLIO\n{t}")

    LAST_PORTFOLIO = current


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
