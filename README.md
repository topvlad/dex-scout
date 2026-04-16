# DEX Scout

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
