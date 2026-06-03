import pathlib

import pytest

import app
import scanner_sources
import ui.pages_runtime as pages_runtime


class _Ctx:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_streamlit(monkeypatch, *, buttons=False, confirm="", dry_run=True):
    calls = []
    st = pages_runtime.st
    for name in ("title", "caption", "markdown", "json", "error", "warning", "code"):
        monkeypatch.setattr(st, name, lambda *a, _name=name, **k: calls.append((_name, a, k)), raising=False)
    monkeypatch.setattr(st, "expander", lambda *a, **k: _Ctx(), raising=False)
    monkeypatch.setattr(st, "text_input", lambda *a, **k: confirm, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: dry_run, raising=False)
    monkeypatch.setattr(st, "button", lambda *a, **k: bool(buttons), raising=False)
    return calls


def _context(execute=None):
    actions = {"render_runtime": lambda: None}
    if execute:
        actions["execute_admin_action"] = execute
    return {
        "operator_diagnostics": {
            "status": "degraded",
            "summary": "stale lock",
            "runtime": {"matrix_status": "ok", "app_compat_status": "ok", "golden_fixtures_status": "ok", "locks": [{"lock_key": "scanner_lock_1", "owner": "worker-a", "expires_epoch": 1}]},
            "storage": {"backend": "d1", "verify_status": "ok"},
        },
        "locks": [{"lock_key": "scanner_lock_1", "owner": "worker-a", "expires_epoch": 1}],
        "actions": actions,
    }


def test_runtime_page_render_does_not_execute_action(monkeypatch):
    _patch_streamlit(monkeypatch, buttons=False)
    called = []
    pages_runtime.render_runtime_page(_context(execute=lambda *a, **k: called.append((a, k))))
    assert called == []


def test_runtime_page_button_dry_run_path_does_not_call_mutation_adapter(monkeypatch):
    _patch_streamlit(monkeypatch, buttons=True, confirm="CLEAR STALE LOCK", dry_run=True)
    called = []
    def execute(*args, **kwargs):
        called.append((args, kwargs))
        return {"ok": True, "status": "dry_run", "reason": "dry_run_no_mutation", "result": {}, "audit": {}}
    pages_runtime.render_runtime_page(_context(execute=execute))
    assert len(called) == 1
    assert called[0][0][0] == "clear_stale_lock"
    assert called[0][1]["dry_run"] is True


def test_runtime_page_no_scanner_telegram_or_storage_on_render(monkeypatch):
    _patch_streamlit(monkeypatch, buttons=False)
    monkeypatch.setattr(scanner_sources, "collect_scanner_sources", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scanner called")))
    monkeypatch.setattr(pathlib.Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("storage write")))
    monkeypatch.setattr(app, "send_telegram_message", lambda *a, **k: (_ for _ in ()).throw(AssertionError("telegram")), raising=False)
    pages_runtime.render_runtime_page(_context())
