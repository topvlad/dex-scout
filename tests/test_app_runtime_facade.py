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
