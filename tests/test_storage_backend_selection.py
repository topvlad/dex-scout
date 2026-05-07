import importlib
import os
import sys


def _load_app(monkeypatch, **env):
    for key in ["STORAGE_BACKEND", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "D1_PROXY_URL", "D1_PROXY_TOKEN", "WORKER_FAST_MODE"]:
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_default_storage_backend_is_supabase(monkeypatch):
    app = _load_app(monkeypatch)
    assert app.STORAGE_BACKEND == "supabase"


def test_storage_backend_d1_selects_d1_path(monkeypatch):
    app = _load_app(monkeypatch, STORAGE_BACKEND="d1", D1_PROXY_URL="https://d1.example", D1_PROXY_TOKEN="t")
    assert app.USE_D1 is True
    assert app.USE_SUPABASE is False


def test_invalid_storage_backend_falls_back(monkeypatch):
    app = _load_app(monkeypatch, STORAGE_BACKEND="weird")
    assert app.STORAGE_BACKEND == "supabase"


def test_d1_mode_does_not_require_supabase(monkeypatch):
    app = _load_app(monkeypatch, STORAGE_BACKEND="d1", D1_PROXY_URL="https://d1.example", D1_PROXY_TOKEN="t")
    assert app.USE_SUPABASE is False
    assert app._d1_ok() is True


def test_app_imports_in_worker_mode(monkeypatch):
    app = _load_app(monkeypatch, WORKER_FAST_MODE="1", STORAGE_BACKEND="d1", D1_PROXY_URL="https://d1.example", D1_PROXY_TOKEN="t")
    assert hasattr(app, "check_runtime_contract")


def test_runtime_contract_d1_uses_d1_checks(monkeypatch):
    app = _load_app(monkeypatch, STORAGE_BACKEND="d1", D1_PROXY_URL="https://d1.example", D1_PROXY_TOKEN="t")

    def fake_select_rows(table, filters=None, select="*", limit=1):
        return [{}], {"ok": True, "code": "ok"}

    monkeypatch.setattr(app, "d1_select_rows", fake_select_rows)
    status = app.check_runtime_contract()
    assert status["ok"] is True


def test_d1_timeout_from_env_parses_without_name_error(monkeypatch):
    app = _load_app(monkeypatch, STORAGE_BACKEND="d1", D1_PROXY_URL="https://d1.example", D1_PROXY_TOKEN="t", D1_TIMEOUT_SEC="12")
    assert app.D1_TIMEOUT_SEC == 12


def test_d1_timeout_invalid_falls_back_to_default(monkeypatch):
    app = _load_app(monkeypatch, STORAGE_BACKEND="d1", D1_PROXY_URL="https://d1.example", D1_PROXY_TOKEN="t", D1_TIMEOUT_SEC="bad")
    assert app.D1_TIMEOUT_SEC == 12
