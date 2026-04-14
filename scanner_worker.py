import os
import time
from typing import Optional

import app

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
SCAN_CHAIN = os.getenv("SCAN_CHAIN", "solana")
SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "80"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "50"))


def run_worker_loop(stop_event: Optional[object] = None) -> None:
    app.ensure_storage()

    while True:
        if stop_event is not None and stop_event.is_set():
            print("[worker] stop requested", flush=True)
            break

        try:
            stats = app.run_full_ingestion_now(
                chain=SCAN_CHAIN,
                seeds_raw=SCANNER_SEEDS,
                max_items=SCANNER_MAX_ITEMS,
                use_birdeye_trending=USE_BIRDEYE_TRENDING,
                birdeye_limit=BIRDEYE_LIMIT,
            )
            print(f"[worker] scan done: {stats}", flush=True)

            monitoring_rows = app.load_monitoring()
            portfolio_rows = app.load_portfolio()
            scan_state = app.scanner_state_load() or {}
            active_monitoring_rows = app.build_active_monitoring_rows(monitoring_rows)
            active_portfolio_rows = [
                r
                for r in portfolio_rows
                if str(r.get("active", "1")).strip() == "1"
            ]

            print(
                f"[worker] notifications called with "
                f"{len(active_monitoring_rows)} monitoring + {len(active_portfolio_rows)} portfolio",
                flush=True,
            )
            app.run_auto_notifications(
                scan_state,
                active_monitoring_rows,
                active_portfolio_rows,
            )
            print("[worker] notifications processed", flush=True)
        except Exception as exc:
            print(f"[worker] error: {type(exc).__name__}: {exc}", flush=True)

        sleep_for = max(300, SCAN_INTERVAL_SEC)
        slept = 0
        while slept < sleep_for:
            if stop_event is not None and stop_event.is_set():
                print("[worker] stop requested during sleep", flush=True)
                return
            time.sleep(1)
            slept += 1


if __name__ == "__main__":
    run_worker_loop()
