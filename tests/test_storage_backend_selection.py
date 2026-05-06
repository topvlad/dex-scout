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
