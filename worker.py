import os
import time

import app
from scanner_worker import run_worker_loop


def run_forever() -> None:
    contract = app.check_runtime_contract()
    if not contract.get("ok"):
        print(
            "[worker-bootstrap] runtime contract missing; refusing to start "
            f"code={contract.get('code')} detail={contract.get('failures')}",
            flush=True,
        )
        app.update_worker_runtime_state(
            updates={
                "worker_status": "runtime_contract_error",
                "last_error_ts": app.now_utc_str(),
                "last_error_reason": f"runtime_contract_error:{contract.get('code')}",
            }
        )
        return

    restart_delay_sec = max(5, int(os.getenv("WORKER_RESTART_DELAY_SEC", "15")))
    max_restarts = max(1, int(os.getenv("WORKER_MAX_RESTARTS", "100")))
    restart_count = 0

    while True:
        start_epoch = time.time()
        start_ts = app.now_utc_str()
        app.update_worker_runtime_state(
            updates={
                "worker_status": "booting",
                "worker_process_started_ts": start_ts,
                "worker_process_started_epoch": start_epoch,
                "worker_last_restart_ts": start_ts if restart_count > 0 else "",
                "worker_last_restart_reason": "process_restart" if restart_count > 0 else "",
                "worker_consecutive_crashes": 0,
            },
            increments={"worker_boot_count": 1},
        )
        app.update_job_heartbeat(
            job_name="worker_bootstrap",
            job_mode="bootstrap",
            status="booting",
            meta={"restart_count": restart_count},
        )
        print(
            f"[worker-bootstrap] start boot={restart_count + 1} "
            f"restart_delay_sec={restart_delay_sec}",
            flush=True,
        )
        try:
            run_worker_loop()
            app.update_worker_runtime_state(updates={"worker_status": "stopped"})
            app.update_job_heartbeat(
                job_name="worker_bootstrap",
                job_mode="bootstrap",
                status="stopped",
                meta={"restart_count": restart_count},
            )
            print("[worker-bootstrap] worker loop exited normally", flush=True)
            return
        except Exception as exc:
            restart_count += 1
            crash_ts = app.now_utc_str()
            reason = f"worker_bootstrap_exception:{type(exc).__name__}:{exc}"
            app.update_worker_runtime_state(
                updates={
                    "worker_status": "crashed",
                    "worker_last_crash_ts": crash_ts,
                    "worker_last_crash_reason": reason,
                    "last_error_ts": crash_ts,
                    "last_error_reason": reason,
                },
                increments={"worker_consecutive_crashes": 1},
            )
            app.update_job_heartbeat(
                job_name="worker_bootstrap",
                job_mode="bootstrap",
                status="crashed",
                meta={"restart_count": restart_count, "reason": reason},
            )
            print(
                f"[worker-bootstrap] crash restart={restart_count}/{max_restarts} "
                f"reason={reason}",
                flush=True,
            )
            if restart_count >= max_restarts:
                print("[worker-bootstrap] restart limit reached -> exiting non-zero", flush=True)
                raise
            time.sleep(restart_delay_sec)
            continue


if __name__ == "__main__":
    run_forever()
