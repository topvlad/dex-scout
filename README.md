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
