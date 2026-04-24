# DEX Scout

## Runtime roles (operational separation)

- **UI/Web** (`streamlit run app.py`): visualization and operator controls.
- **Webhook** (`uvicorn tg_webhook:app ...`): Telegram callback handling only.
- **Background worker** (`python worker.py`): autonomous scanner + notification emission loop.

By default, notification emission is expected from the background worker. UI-triggered notification emission is disabled unless explicitly enabled with:

`DEX_SCOUT_UI_EMIT_NOTIFICATIONS=1`

Worker startup is explicit/reproducible through `python worker.py` (used by deploy worker service). If the loop crashes, bootstrap restarts it with:

- `WORKER_RESTART_DELAY_SEC` (default `15`)
- `WORKER_MAX_RESTARTS` (default `100`)

## Dependencies

### UI (Streamlit)
- `streamlit==1.44.1`
- `pandas==2.2.3`
- `streamlit-autorefresh==1.0.1`
- `requests==2.32.3`
- `supabase==2.15.3`

### Webhook (FastAPI + Telegram callback)
- `fastapi==0.115.12`
- `uvicorn[standard]==0.34.2`
- `requests==2.32.3`

### Worker / scanner
- `requests==2.32.3`
- `supabase==2.15.3`

> Усі залежності зафіксовані (pinning) у `requirements.txt`, щоб мінімізувати дрейф поведінки між деплоями.


## Emergency runtime reset + workflow dispatch

For stuck runtime state/locks incidents, run:

```bash
SUPABASE_URL=https://PROJECT.supabase.co \
SUPABASE_SERVICE_ROLE_KEY=... \
GITHUB_TOKEN=ghp_... \
GITHUB_REPOSITORY=OWNER/REPO \
python scripts/emergency_runtime_reset_and_dispatch.py
```

The helper will:
- detect the effective JSON column in `public.runtime_state` (prefers `state_json`, but adapts if schema differs);
- set `last_job_status` to `finished` for `worker_runtime`;
- remove `job_mode:%` locks;
- dispatch `workflow_dispatch` for `scan_cycle`, then `all`;
- download run logs and fail if `exit code 3` / `duplicate run blocked` / `skipped by lock` are still present.

## Runtime Jobs repo vars (recommended)

For `.github/workflows/runtime-jobs.yml`, define these **Repository Variables**:

- `JOB_LOCK_TTL_SEC` (recommended: `600`)
- `JOB_STALE_RUN_SEC` (recommended: `1200`)

Workflow timeout is `30m` (`1800s`) now, so keep this invariant valid:

`JOB_LOCK_TTL_SEC < JOB_STALE_RUN_SEC < workflow timeout`

Recommended ops values:

- `JOB_LOCK_TTL_SEC=600`
- `JOB_STALE_RUN_SEC=1200`
- workflow `timeout-minutes=30` (or any value above `20m`, so `stale < timeout` remains true)

If repo vars are missing, workflow preflight logs effective defaults and still validates the invariant before executing `worker.py`.
