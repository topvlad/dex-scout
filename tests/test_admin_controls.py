import importlib
import sys


def _ok_context():
    return {
        "operator_diagnostics": {
            "status": "ok",
            "runtime": {
                "matrix_status": "ok",
                "read_only_status": "ok",
                "monitor_cycle_status": "ok",
                "notify_cycle_status": "ok",
                "stale_jobs": [],
            },
            "scanner": {"source_status": "success"},
        },
        "read_only_status": "ok",
    }


def _stale_lock_context():
    return {
        "locks": [{"lock_key": "scanner_lock_1", "owner": "worker-a", "expires_epoch": 1}],
        "operator_diagnostics": {"status": "degraded", "runtime": {"matrix_status": "ok"}},
    }


def test_admin_controls_imports_without_streamlit_or_app(monkeypatch):
    sys.modules.pop("admin_controls", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)
    module = importlib.import_module("admin_controls")
    assert module.__name__ == "admin_controls"
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_build_plan_missing_context_unknown_no_crash():
    import admin_controls
    plan = admin_controls.build_admin_recovery_plan({})
    assert plan["status"] == "unknown"
    assert plan["recommended_order"] == ["runtime_matrix", "read_only", "maintenance_cycle", "monitor_cycle", "notify_cycle", "scan_cycle"]


def test_build_plan_ok_context_status_ok():
    import admin_controls
    assert admin_controls.build_admin_recovery_plan(_ok_context())["status"] == "ok"


def test_build_plan_source_failed_or_stale_job_needs_recovery():
    import admin_controls
    failed = {"operator_diagnostics": {"status": "degraded", "scanner": {"source_status": "source_api_failed"}}}
    stale = {"job_heartbeats": [{"job": "notify_cycle", "stale": True}], "operator_diagnostics": {"status": "warning"}}
    assert admin_controls.build_admin_recovery_plan(failed)["status"] == "recovery_needed"
    assert admin_controls.build_admin_recovery_plan(stale)["status"] in {"attention", "recovery_needed"}


def test_validate_unknown_action_blocked():
    import admin_controls
    res = admin_controls.validate_admin_action("missing", {})
    assert res["ok"] is False
    assert res["reason"] == "unknown_action"


def test_validate_mutating_dry_run_does_not_mutate():
    import admin_controls
    res = admin_controls.validate_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=True)
    assert res["dry_run"] is True
    assert res["would_mutate"] is True
    assert res["enabled"] is True


def test_validate_clear_stale_lock_disabled_when_not_stale():
    import admin_controls
    res = admin_controls.validate_admin_action("clear_stale_lock", {"locks": [{"lock_key": "k", "owner": "o", "age_sec": 1}]}, confirmation="CLEAR STALE LOCK")
    assert res["enabled"] is False


def test_validate_clear_stale_lock_enabled_with_confirmation():
    import admin_controls
    res = admin_controls.validate_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=False)
    assert res["enabled"] is True


def test_scan_cycle_warns_when_checks_not_green():
    import admin_controls
    res = admin_controls.validate_admin_action("scan_cycle_runbook", {"operator_diagnostics": {"runtime": {"matrix_status": "failed"}}})
    assert res["enabled"] is True
    assert res["warnings"]


def test_execute_dry_run_never_calls_adapter():
    import admin_controls
    calls = []
    def adapter(**kwargs):
        calls.append(kwargs)
        return {"ok": True}
    res = admin_controls.execute_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=True, adapters={"clear_stale_lock_fn": adapter})
    assert res["status"] == "dry_run"
    assert calls == []


def test_execute_mutating_without_adapter_blocked():
    import admin_controls
    res = admin_controls.execute_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=False, adapters={})
    assert res["status"] == "blocked"
    assert res["reason"] == "adapter_missing"


def test_execute_mutating_with_adapter_calls_once():
    import admin_controls
    calls = []
    def adapter(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "cleared": True}
    res = admin_controls.execute_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=False, adapters={"clear_stale_lock_fn": adapter})
    assert res["status"] == "executed"
    assert len(calls) == 1


def test_adapter_exception_returns_failed():
    import admin_controls
    def adapter(**kwargs):
        raise RuntimeError("boom")
    res = admin_controls.execute_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=False, adapters={"clear_stale_lock_fn": adapter})
    assert res["status"] == "failed"
    assert "RuntimeError" in res["reason"]


def test_no_telegram_scanner_storage_side_effects(monkeypatch):
    import admin_controls
    monkeypatch.setitem(sys.modules, "telegram", object())
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("side effect")
    plan = admin_controls.build_admin_recovery_plan(_ok_context())
    dry = admin_controls.execute_admin_action("clear_stale_lock", _stale_lock_context(), confirmation="CLEAR STALE LOCK", dry_run=True, adapters={"clear_stale_lock_fn": forbidden})
    assert plan["status"] == "ok"
    assert dry["status"] == "dry_run"
    assert calls == []
