import worker


def test_worker_d1_backend_does_not_require_supabase(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "d1")
    monkeypatch.setenv("JOB_MODE", "scan_cycle")
    monkeypatch.setenv("TG_BOT_TOKEN", "x")
    monkeypatch.setenv("TG_CHAT_ID", "y")
    monkeypatch.setenv("D1_PROXY_URL", "https://d1.example")
    monkeypatch.setenv("D1_PROXY_TOKEN", "t")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    missing = worker._missing_required_env()
    assert "SUPABASE_URL" not in missing
    assert "SUPABASE_SERVICE_ROLE_KEY" not in missing
