import os
os.environ["DEX_SCOUT_WORKER_MODE"] = "1"

import time
from typing import Optional

import app

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
SCAN_CHAIN = os.getenv("SCAN_CHAIN", "solana")
SCANNER_SEEDS = os.getenv("SCANNER_SEEDS", str(app.DEFAULT_SEEDS))
SCANNER_MAX_ITEMS = int(os.getenv("SCANNER_MAX_ITEMS", "10"))
USE_BIRDEYE_TRENDING = os.getenv("USE_BIRDEYE_TRENDING", "1") != "0"
BIRDEYE_LIMIT = int(os.getenv("BIRDEYE_LIMIT", "10"))
WORKER_HEARTBEAT_MINUTES = max(1, int(os.getenv("WORKER_HEARTBEAT_MINUTES", "5")))
WORKER_HEARTBEAT_SEC = WORKER_HEARTBEAT_MINUTES * 60


def _queue_invariant_telemetry(queue_stats: dict, sleep_for: int, default_sleep_for: int) -> None:
    """
    Assertion/telemetry-only hooks for Wave B queue invariants.
    No scheduling or scanner behavior changes are allowed here.
    """
    try:
        tiers_scanned = [str(x or "") for x in (queue_stats.get("tiers_scanned") or [])]
        tier_rank = {
            "portfolio_active": 0,
            "portfolio_linked_monitoring": 1,
            "high_priority_monitoring": 2,
            "review_weak_monitoring": 3,
            "archive_revisit": 4,
        }
        if len(tiers_scanned) >= 2:
            ranked = [tier_rank.get(t, 99) for t in tiers_scanned]
            if ranked != sorted(ranked):
                print(
                    f"[worker][wave-b] queue_tier_order_violation "
                    f"(when_all_else_equal_assumption): {tiers_scanned}",
                    flush=True,
                )
    except Exception as exc:
        print(f"[worker][wave-b] queue_tier_order_telemetry_error: {type(exc).__name__}: {exc}", flush=True)

    try:
        due_now = int(queue_stats.get("due_now", 0) or 0)
        queue_size = int(queue_stats.get("queue_size", 0) or 0)
        if queue_size <= 1 and sleep_for < 30:
            print(
                f"[worker][wave-b] small_queue_sleep_floor_violation: sleep_for={sleep_for}, "
                f"default_sleep_for={default_sleep_for}, due_now={due_now}",
                flush=True,
            )
    except Exception as exc:
        print(f"[worker][wave-b] small_queue_telemetry_error: {type(exc).__name__}: {exc}", flush=True)

    try:
        suggested = queue_stats.get("sleep_suggest_sec")
        if suggested is None:
            print("[worker][wave-b] sleep_suggest_sec_missing", flush=True)
    except Exception as exc:
        print(f"[worker][wave-b] sleep_suggest_telemetry_error: {type(exc).__name__}: {exc}", flush=True)


def run_worker_loop(stop_event: Optional[object] = None, one_pass: bool = False) -> None:
    app.ensure_storage()
    last_heartbeat_ts = 0.0

    while True:
        default_sleep_for = max(60, SCAN_INTERVAL_SEC)
        sleep_for = default_sleep_for
        now_ts = time.time()
        now_utc = app.now_utc_str()
        runtime = app.get_worker_runtime_state()
        prev_loop_epoch = float(runtime.get("last_loop_epoch", 0.0) or 0.0)
        stale_threshold_sec = max(WORKER_HEARTBEAT_SEC * 2, SCAN_INTERVAL_SEC * 2, 600)
        if prev_loop_epoch > 0 and (now_ts - prev_loop_epoch) > stale_threshold_sec:
            stale_reason = f"stale_loop_gap_sec>{stale_threshold_sec}"
            print(
                f"[worker] stale loop detected gap_sec={int(now_ts - prev_loop_epoch)} "
                f"threshold_sec={stale_threshold_sec}",
                flush=True,
            )
            app.update_worker_runtime_state(
                updates={"last_error_ts": now_utc, "last_error_reason": stale_reason}
            )
        app.update_worker_runtime_state(
            updates={
                "last_loop_ts": now_utc,
                "last_loop_epoch": now_ts,
                "heartbeat_interval_sec": WORKER_HEARTBEAT_SEC,
            },
            increments={"loop_iterations": 1},
        )
        if (now_ts - last_heartbeat_ts) >= WORKER_HEARTBEAT_SEC:
            print(
                f"[worker] heartbeat alive=1 interval_sec={SCAN_INTERVAL_SEC} "
                f"heartbeat_sec={WORKER_HEARTBEAT_SEC}",
                flush=True,
            )
            last_heartbeat_ts = now_ts
        if stop_event is not None and stop_event.is_set():
            print("[worker] stop requested", flush=True)
            break

        try:
            print("[worker] loading monitoring/portfolio", flush=True)

            rotate_stats = app.maybe_run_rotating_scanner(
                seeds_raw=SCANNER_SEEDS,
                max_items=SCANNER_MAX_ITEMS,
                use_birdeye_trending=USE_BIRDEYE_TRENDING,
                birdeye_limit=BIRDEYE_LIMIT,
            )
            if rotate_stats.get("ran"):
                print(f"[worker] rotating scanner ran: {rotate_stats}", flush=True)

            monitoring_rows = app.load_monitoring()
            portfolio_rows = app.load_portfolio()
            active_monitoring_rows = app.build_active_monitoring_rows(monitoring_rows)
            active_portfolio_rows = [
                r for r in portfolio_rows
                if str(r.get("active", "1")).strip() == "1"
            ]

            print(
                f"[worker] notifications called with "
                f"{len(active_monitoring_rows)} monitoring + {len(active_portfolio_rows)} portfolio",
                flush=True,
            )
            queue_stats = app.run_priority_scanner_cycle(
                monitoring_rows=monitoring_rows,
                portfolio_rows=active_portfolio_rows,
                max_scans=3,
            )
            print(f"[worker] adaptive queue stats: {queue_stats}", flush=True)
            if "sleep_suggest_sec" in queue_stats:
                suggested = int(queue_stats.get("sleep_suggest_sec", default_sleep_for))
                sleep_for = max(30, min(default_sleep_for, suggested))
            else:
                sleep_for = default_sleep_for
            _queue_invariant_telemetry(queue_stats, sleep_for=sleep_for, default_sleep_for=default_sleep_for)
            fallback_reason = ""
            if int(queue_stats.get("scanned", 0) or 0) <= 0:
                fallback_reason = "adaptive_no_scans"
            elif int(queue_stats.get("errors", 0) or 0) >= int(queue_stats.get("scanned", 0) or 0):
                fallback_reason = "adaptive_all_scans_failed"
            candidate_path = "adaptive_queue"
            if fallback_reason:
                candidate_path = "baseline_fallback"
                print(f"[worker] baseline fallback enabled reason={fallback_reason}", flush=True)
                app.update_worker_runtime_state(updates={"last_empty_reason": fallback_reason})

            monitoring_rows = app.load_monitoring()
            portfolio_rows = app.load_portfolio()
            active_monitoring_rows = app.build_active_monitoring_rows(monitoring_rows)
            active_portfolio_rows = [
                r for r in portfolio_rows
                if str(r.get("active", "1")).strip() == "1"
            ]
            scan_state = app.scanner_state_load() or {}

            notif_result = app.run_auto_notifications(
                scan_state,
                active_monitoring_rows,
                active_portfolio_rows,
                cycle_context={
                    "candidate_path": candidate_path,
                    "fallback_reason": fallback_reason,
                },
            )
            print(f"[worker] notifications processed result={notif_result}", flush=True)

            if one_pass:
                print("[worker] one_pass done", flush=True)
                return
        except Exception as exc:
            import traceback
            print(f"[worker] error: {type(exc).__name__}: {exc}", flush=True)
            app.update_worker_runtime_state(
                updates={
                    "last_error_ts": app.now_utc_str(),
                    "last_error_reason": f"worker_exception:{type(exc).__name__}:{exc}",
                },
                increments={"notification_failed_cycles": 1},
            )
            traceback.print_exc()

        slept = 0
        while slept < sleep_for:
            if stop_event is not None and stop_event.is_set():
                print("[worker] stop requested during sleep", flush=True)
                return
            time.sleep(1)
            slept += 1


if __name__ == "__main__":
    run_worker_loop()
