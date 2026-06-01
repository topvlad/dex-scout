from fastapi.testclient import TestClient

import tg_webhook
import worker


class FailedFacade:
    def load_app_runtime(self):
        return {"ok": False, "error": "boom", "exception_type": "RuntimeError", "worker_fast_mode": True}


def test_worker_app_import_failure_exits_safe_without_fake_success(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_load_app", lambda: FailedFacade())
    monkeypatch.setattr(worker, "_missing_required_env", lambda: [])
    monkeypatch.setattr(worker, "_fail_fast", lambda reason, code: calls.append((reason, code)) or code)
    rc = worker.main()
    assert rc == 2
    assert calls == []


def test_worker_normal_monkeypatched_facade_path_preserves_job_mode(monkeypatch):
    class AppStub:
        def get_worker_runtime_state(self):
            return {"modes": {"digest_cycle": {"last_job_status": "idle", "last_job_started_epoch": 0}}}
        def update_worker_runtime_state(self, **kwargs):
            return {"ok": True}
        def acquire_lock(self, **kwargs):
            return {"ok": True}
        def release_lock(self, **kwargs):
            return {"ok": True}
        def update_job_heartbeat(self, **kwargs):
            return {"ok": True}
        def get_supabase_storage_budget_snapshot(self):
            return {}
    stub = AppStub()
    monkeypatch.setattr(worker, "app", stub)
    monkeypatch.setattr(worker, "JOB_DISPATCH", {"digest_cycle": lambda: {"ok": True, "sent": False}})
    monkeypatch.setattr(worker, "_update_worker_runtime_with_mode_state", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(worker, "_finalize_runtime_if_token_matches", lambda **kwargs: True)
    assert worker.run_job_mode("digest_cycle") == 0


def test_import_status_reports_app_import_failed(monkeypatch):
    client = TestClient(tg_webhook.app)
    monkeypatch.setattr(tg_webhook.app_runtime_facade, "get_import_state", lambda force_reload=False: {"ok": False, "error": "boom", "exception_type": "RuntimeError", "worker_fast_mode": True, "app_module_loaded": False})
    monkeypatch.setattr(tg_webhook, "IMPORT_FAILED", False)
    monkeypatch.setattr(tg_webhook, "BOOTSTRAP_ERROR", {})
    data = client.get("/_import_status").json()
    assert data["ok"] is False
    assert data["import_failed"] is True
    assert data["error"] == "boom"
    assert data["app_module_loaded"] is False


def test_callback_returns_app_import_failed_if_unavailable(monkeypatch):
    client = TestClient(tg_webhook.app)
    monkeypatch.setattr(tg_webhook.app_runtime_facade, "get_import_state", lambda force_reload=False: {"ok": False, "error": "boom", "exception_type": "RuntimeError", "worker_fast_mode": True, "app_module_loaded": False})
    response = client.post("/tg_webhook", json={"callback_query": {"id": "cb1", "data": "pf|bsc|abcdefghijklmnopqrst", "message": {"chat": {"id": 1}, "message_id": 2}}})
    data = response.json()
    assert data["ok"] is False
    assert data["error"] == "app_import_failed"


def test_status_never_contains_token_values(monkeypatch):
    client = TestClient(tg_webhook.app)
    monkeypatch.setenv("TG_BOT_TOKEN", "super-secret-token")
    monkeypatch.setattr(tg_webhook.app_runtime_facade, "get_import_state", lambda force_reload=False: {"ok": False, "error": "boom", "exception_type": "RuntimeError", "worker_fast_mode": True, "app_module_loaded": False})
    text = repr(client.get("/_import_status").json())
    assert "super-secret-token" not in text
