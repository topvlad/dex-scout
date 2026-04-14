import os
import time
from typing import Any, Dict, List

import app

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
SCAN_CHAIN = os.getenv("SCAN_CHAIN", "solana")
SCANNER_WINDOW = os.getenv("SCANNER_WINDOW", "Wide Net (explore)")
SCANNER_PRESET = os.getenv("SCANNER_PRESET", "wide")
SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "80"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "50"))


def _parse_score(row: Dict[str, Any]) -> float:
    return app.parse_float(row.get("entry_score", 0), 0.0)


def _active_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r
        for r in rows
        if str(r.get("active", r.get("is_active", "1"))).strip().lower() not in {"0", "false", "no"}
    ]


def run_ingest_cycle() -> Dict[str, Any]:
    stats = app.ingest_window_to_monitoring(
        chain=app.normalize_chain_name(SCAN_CHAIN or "solana"),
        window_name=SCANNER_WINDOW,
        preset_key=SCANNER_PRESET,
        seeds_raw=SCANNER_SEEDS,
        max_items=SCANNER_MAX_ITEMS,
        use_birdeye_trending=USE_BIRDEYE_TRENDING,
        birdeye_limit=BIRDEYE_LIMIT,
    )
    stats["revisited"] = app.add_new_candidates()
    stats["trimmed"] = app.trim_active_monitoring(max_active=120)

    state = app.scanner_state_load()
    state.update(
        {
            "last_run_ts": app.now_utc_str(),
            "last_window": SCANNER_WINDOW,
            "last_preset": SCANNER_PRESET,
            "last_chain": app.normalize_chain_name(SCAN_CHAIN or "solana"),
            "last_stats": stats,
        }
    )
    app.scanner_state_save(state)
    return state


def run_worker_loop() -> None:
    app.ensure_storage()
    while True:
        try:
            scan_state = run_ingest_cycle()
            monitoring = _active_rows(app.load_monitoring())
            portfolio = _active_rows(app.load_portfolio())
            scored = len([r for r in monitoring if _parse_score(r) > 0])

            print(
                f"[worker] monitoring={len(monitoring)} scored={scored} portfolio={len(portfolio)}",
                flush=True,
            )
            app.run_auto_notifications(scan_state, monitoring, portfolio)
        except Exception as exc:
            print(f"[worker] error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(60)
            continue

        time.sleep(max(60, SCAN_INTERVAL_SEC))


if __name__ == "__main__":
    run_worker_loop()
