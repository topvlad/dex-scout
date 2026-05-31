# DEX Scout Dash Dashboard

`dash_app.py` is a separate, read-only Dash entrypoint for fast operational views backed by the existing Cloudflare D1 proxy.

## Run locally

```bash
STORAGE_BACKEND=d1 \
D1_PROXY_URL=https://<worker-host> \
D1_PROXY_TOKEN=<token> \
python dash_app.py
```

The app listens on `PORT` (default `8050`) and exposes the Flask server as `dash_app:server` for process managers such as Gunicorn.

## Pages

- **Overview** — active Monitoring count, Portfolio count, Live Pulse clean candidates, pending scan state, last scan status/stats, runtime health, and hard-gated/dead candidate count.
- **Monitoring** — active Monitoring table plus a token selector that lazily loads monitoring history for the selected token sparkline.
- **Portfolio** — active Portfolio table and recommendation/action timeline from `portfolio_reco_log.csv`.
- **Live Pulse** — final/clean candidates plus hard-gated/dead candidates from the runtime Live Pulse payload.
- **Runtime** — D1/runtime contract, worker runtime JSON, and job heartbeat rows.
- **Archive** — inactive or archived Monitoring rows with reason counts.

## Data and safety contract

Dash only uses existing D1 read helpers from `app.py`:

- `d1_get_storage()` for CSV/blob keys in D1 `app_storage`.
- `d1_select_rows()` for runtime tables exposed by the existing proxy.
- `get_worker_runtime_state()` and `check_runtime_contract()` for read-only runtime snapshots.

It does **not** call scanner functions, storage write/delete helpers, Telegram send helpers, worker dispatch functions, scoring mutation paths, or Streamlit UI actions.

## Performance

D1 reads are process-cached for 30–60 seconds (`DASH_D1_CACHE_TTL_SEC`, default `45`). Full Monitoring and Portfolio CSVs are loaded once per cache window, while token sparkline details are filtered only when a token is selected. Monitoring history is bounded by `DASH_HISTORY_ROWS_LIMIT` (default `5000`).
