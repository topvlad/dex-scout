import importlib
import sys
import types

import app_runtime_facade as facade
import worker


def _reset_facade():
    facade._APP_MODULE = None
    facade._LAST_RESULT = None


def test_facade_import_does_not_import_app_immediately():
    sys.modules.pop("app", None)
    _reset_facade()
    importlib.reload(facade)
    assert "app" not in sys.modules


def test_load_app_runtime_success_caches_module(monkeypatch):
    _reset_facade()
    module = types.SimpleNamespace(WORKER_FAST_MODE=True)
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        assert name == "app"
        return module

    monkeypatch.setattr(facade.importlib, "import_module", fake_import)
    first = facade.load_app_runtime()
    second = facade.load_app_runtime()
    assert first["ok"] is True
    assert second["app"] is module
    assert calls["n"] == 1


def test_failure_path_returns_structured_error(monkeypatch):
    _reset_facade()

    def fake_import(name):
        raise ValueError("boom")

    monkeypatch.setattr(facade.importlib, "import_module", fake_import)
    state = facade.load_app_runtime()
    assert state["ok"] is False
    assert state["app"] is None
    assert state["error"] == "boom"
    assert state["exception_type"] == "ValueError"


def test_force_reload_retries_after_failure(monkeypatch):
    _reset_facade()
    module = types.SimpleNamespace()
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first")
        return module

    monkeypatch.setattr(facade.importlib, "import_module", fake_import)
    assert facade.load_app_runtime()["ok"] is False
    assert facade.load_app_runtime()["ok"] is False
    assert facade.load_app_runtime(force_reload=True)["ok"] is True
    assert calls["n"] == 2


def test_no_secrets_in_worker_preflight(monkeypatch):
    monkeypatch.setenv("D1_PROXY_URL", "https://private.example")
    monkeypatch.setenv("D1_PROXY_TOKEN", "secret-token")
    monkeypatch.setenv("TG_BOT_TOKEN", "telegram-secret")
    monkeypatch.setenv("TG_CHAT_ID", "123")
    payload = worker._runtime_preflight({"ok": True})
    text = repr(payload)
    assert "secret-token" not in text
    assert "telegram-secret" not in text
    assert "https://private.example" not in text


def _install_fake_app(monkeypatch, module):
    _reset_facade()
    monkeypatch.setattr(facade.importlib, "import_module", lambda name: module)
    return module


def test_update_job_heartbeat_forwards_public_job_mode_kwarg(monkeypatch):
    captured = {}

    def update_job_heartbeat(job_name, job_mode, status="alive", meta=None):
        captured["job_name"] = job_name
        captured["job_mode"] = job_mode
        captured["status"] = status
        captured["meta"] = meta
        return {"ok": True}

    _install_fake_app(monkeypatch, types.SimpleNamespace(update_job_heartbeat=update_job_heartbeat))

    result = facade.update_job_heartbeat(
        job_name="runtime_notify_cycle",
        job_mode="notify_cycle",
        status="started",
        meta={"x": 1},
    )

    assert result == {"ok": True}
    assert captured == {
        "job_name": "runtime_notify_cycle",
        "job_mode": "notify_cycle",
        "status": "started",
        "meta": {"x": 1},
    }


def test_internal_facade_job_mode_does_not_leak_into_app_kwargs(monkeypatch):
    captured = {}

    def app_function(**kwargs):
        assert "_facade_job_mode" not in kwargs
        captured.update(kwargs)
        return {"ok": True}

    _install_fake_app(monkeypatch, types.SimpleNamespace(app_function=app_function))

    result = facade._call_app("app_function", _facade_job_mode="internal_mode", token_addr="0xabc", chain="solana")

    assert result == {"ok": True}
    assert captured == {"token_addr": "0xabc", "chain": "solana"}


def test_run_auto_notifications_uses_facade_job_mode_only_for_failure_payload(monkeypatch):
    captured = {}

    def run_auto_notifications(**kwargs):
        assert "_facade_job_mode" not in kwargs
        captured.update(kwargs)
        return {"ok": True, "status": "called"}

    _install_fake_app(monkeypatch, types.SimpleNamespace(run_auto_notifications=run_auto_notifications))

    result = facade.run_auto_notifications(scan_state={"ok": True})

    assert result == {"ok": True, "status": "called"}
    assert captured == {"scan_state": {"ok": True}}

    _reset_facade()

    def fake_import_failure(name):
        raise RuntimeError("import boom")

    monkeypatch.setattr(facade.importlib, "import_module", fake_import_failure)
    payload = facade.run_auto_notifications()

    assert payload["ok"] is False
    assert payload["status"] == "app_import_failed"
    assert payload["job_mode"] == "notify_cycle"


def test_missing_app_function_returns_structured_payload(monkeypatch):
    _install_fake_app(monkeypatch, types.SimpleNamespace(run_auto_notifications=None))

    payload = facade.run_auto_notifications()

    assert payload == {
        "ok": False,
        "status": "missing_app_function",
        "function": "run_auto_notifications",
        "job_mode": "notify_cycle",
        "reason": "app function is missing or not callable",
    }


def test_required_worker_job_modes_resolve_through_runtime_facade(monkeypatch):
    module = types.SimpleNamespace(
        maybe_run_rotating_scanner=lambda: {"ok": True},
        run_priority_scanner_cycle=lambda: {"ok": True},
        run_auto_notifications=lambda: {"ok": True},
        trigger_digest_notification=lambda: {"ok": True},
        evaluate_outcome_journals=lambda: {"ok": True},
        run_storage_maintenance_cycle=lambda: {"ok": True},
    )
    _install_fake_app(monkeypatch, module)

    for mode in (
        "maintenance_cycle",
        "monitor_cycle",
        "notify_cycle",
        "digest_cycle",
        "outcome_cycle",
        "scan_cycle",
    ):
        payload = facade.resolve_worker_job_mode(mode, dry_run=True)
        assert payload["ok"] is True
        assert payload["status"] == "resolved"
        assert payload["job_mode"] == mode
        assert payload["runner"] == facade.WORKER_JOB_MODE_RUNNERS[mode]


class _WorkerHeartbeatStub:
    def __init__(self):
        self.runtime = {"modes": {}}
        self.heartbeats = []

    def get_worker_runtime_state(self):
        return self.runtime

    def update_worker_runtime_state(self, updates=None, increments=None):
        updates = dict(updates or {})
        if "modes" in updates:
            self.runtime["modes"] = updates["modes"]
        self.runtime.update({k: v for k, v in updates.items() if k != "modes"})
        return {"ok": True}

    def acquire_lock(self, **kwargs):
        return {"ok": True}

    def release_lock(self, **kwargs):
        return {"ok": True}

    def update_job_heartbeat(self, job_name, job_mode, status="alive", meta=None):
        self.heartbeats.append({"job_name": job_name, "job_mode": job_mode, "status": status, "meta": meta})
        return {"ok": True}

    def get_supabase_storage_budget_snapshot(self):
        return {"ok": True}


def test_worker_heartbeat_start_path_forwards_job_mode_and_notify_has_no_typeerror(monkeypatch):
    stub = _WorkerHeartbeatStub()
    monkeypatch.delenv("JOB_DRY_RUN", raising=False)
    monkeypatch.setattr(worker, "app", stub)
    monkeypatch.setattr(worker, "JOB_DISPATCH", {"notify_cycle": lambda: {"ok": True}})

    assert worker.run_job_mode("notify_cycle") == 0

    started = [item for item in stub.heartbeats if item["status"] == "started"]
    assert len(started) == 1
    assert started[0]["job_name"] == "job_dispatch:notify_cycle"
    assert started[0]["job_mode"] == "notify_cycle"
