import pathlib

import pytest

import app
import scanner_sources
import ui.pages_monitoring as pages_monitoring
import ui.pages_runtime as pages_runtime
import ui.pages_scout as pages_scout


def _capture_streamlit(monkeypatch, module):
    calls = []

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    st = module.st
    for name in ("title", "caption", "markdown", "json", "error"):
        monkeypatch.setattr(st, name, lambda *a, _name=name, **k: calls.append((_name, a, k)), raising=False)
    monkeypatch.setattr(st, "expander", lambda *a, **k: _Ctx(), raising=False)
    return calls


def test_runtime_page_renders_diagnostics_with_fake_context_and_no_scanner_fetch(monkeypatch):
    calls = _capture_streamlit(monkeypatch, pages_runtime)
    monkeypatch.setattr(scanner_sources, "collect_scanner_sources", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scanner fetch called")))
    pages_runtime.render_runtime_page(
        {
            "operator_diagnostics": {
                "status": "ok",
                "summary": "all green",
                "runtime": {"matrix_status": "ok", "app_compat_status": "ok", "golden_fixtures_status": "ok", "last_jobs": {"scan_cycle": {"last_job_status": "success"}}, "stale_jobs": []},
                "storage": {"backend": "d1", "verify_status": "ok", "last_read_ok": True, "last_write_ok": True},
            },
            "actions": {"render_runtime": lambda: None},
        }
    )
    rendered = " ".join(str(args) for _, args, _ in calls)
    assert "runtime_matrix" in rendered
    assert "golden_fixtures" in rendered


def test_scout_page_shows_live_pulse_empty_reason_and_does_not_write(monkeypatch):
    calls = _capture_streamlit(monkeypatch, pages_scout)
    monkeypatch.setattr(pathlib.Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("write_text called")))
    pages_scout.render_scout_page(
        {
            "operator_diagnostics": {
                "status": "warning",
                "scanner": {
                    "source_status": "source_api_empty",
                    "last_empty_reason": "source_api_empty",
                    "raw_seen": 0,
                    "normalized": 0,
                    "clean_candidates": 0,
                    "final_count": 0,
                    "sources_tried": ["dexscreener"],
                    "sources_failed": 0,
                    "sources_empty": 1,
                    "sources_disabled": 0,
                    "source_errors": [],
                },
            },
            "live_pulse_payload": {"status": "empty", "final_count": 0},
            "actions": {"render_scout": lambda cfg: None},
        }
    )
    rendered = " ".join(str(args) for _, args, _ in calls)
    assert "source_api_empty" in rendered
    assert "final_count" in rendered


def test_monitoring_page_shows_priority_empty_reason_and_does_not_save(monkeypatch):
    calls = _capture_streamlit(monkeypatch, pages_monitoring)
    monkeypatch.setattr(app, "save_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("save_csv called")))
    monkeypatch.setattr(app, "save_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("save_json called")))
    pages_monitoring.render_monitoring_page(
        {
            "operator_diagnostics": {
                "priority": {
                    "source_monitoring_rows": 2,
                    "active_monitoring_rows": 2,
                    "eligible_watch_early_rows": 1,
                    "final_priority_rows": 0,
                    "excluded": {"no_entry": 1, "hard_gated": 1, "unknown": 0},
                    "top_excluded_samples": [{"symbol": "AAA", "reason": "no_entry"}],
                }
            },
            "priority_watchlist_rows": [],
            "actions": {"render_monitoring": lambda cfg: None},
        }
    )
    rendered = " ".join(str(args) for _, args, _ in calls)
    assert "Priority watchlist diagnostics" in rendered
    assert "hard_gated" in rendered


def test_ui_pages_tolerate_missing_old_diagnostics(monkeypatch):
    _capture_streamlit(monkeypatch, pages_runtime)
    _capture_streamlit(monkeypatch, pages_scout)
    _capture_streamlit(monkeypatch, pages_monitoring)
    pages_runtime.render_runtime_page({"actions": {"render_runtime": lambda: None}})
    pages_scout.render_scout_page({"live_pulse_payload": {}, "actions": {"render_scout": lambda cfg: None}})
    pages_monitoring.render_monitoring_page({"actions": {"render_monitoring": lambda cfg: None}})


def test_build_ui_context_exposes_canonical_keys_and_does_not_call_scan_runner(monkeypatch):
    monkeypatch.setattr(app, "run_scan_cycle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scan runner called")), raising=False)
    context = app.build_ui_context(selected_page="Scout", actions={"render_scout": lambda *a, **k: None})
    for key in (
        "operator_diagnostics",
        "scanner_diagnostics",
        "live_pulse_payload",
        "live_pulse_debug",
        "priority_watchlist_rows",
        "priority_watchlist_debug",
        "monitoring_archive_diagnostics",
        "portfolio_action_diagnostics",
    ):
        assert key in context
