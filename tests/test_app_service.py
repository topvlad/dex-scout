import importlib
import sys


def _fresh_import_app_service():
    for name in ("app_service", "app", "streamlit"):
        sys.modules.pop(name, None)
    return importlib.import_module("app_service")


def test_app_service_imports_without_streamlit_or_app():
    module = _fresh_import_app_service()

    assert module.build_runtime_summary({})["selected_page"] == "Monitoring"
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_app_service_import_has_no_writes_or_network(monkeypatch):
    calls = []

    def blocked(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("import side effect")

    for target in ("requests.post", "requests.put", "requests.patch", "requests.delete"):
        module_name, attr = target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, attr, blocked, raising=False)

    _fresh_import_app_service()

    assert calls == []


def test_build_runtime_summary_handles_missing_keys():
    svc = importlib.import_module("app_service")

    summary = svc.build_runtime_summary({})

    assert summary["selected_page"] == "Monitoring"
    assert summary["loop_iterations"] == 0
    assert summary["notification_counters"]["sent"] == 0


def test_build_storage_summary_handles_verify_states():
    svc = importlib.import_module("app_service")

    skipped = svc.build_storage_summary({"ok": True, "write_ok": True, "verify_status": "skipped"}, {})
    failed = svc.build_storage_summary({"ok": True, "write_ok": True, "verify_status": "failed"}, {})
    ok = svc.build_storage_summary({"ok": True, "write_ok": True, "verify_status": "ok"}, {})

    assert skipped["verify_skipped"] is True
    assert failed["ok"] is False
    assert failed["verify_failed"] is True
    assert ok["ok"] is True
    assert ok["verify_ok"] is True


def test_build_notification_summary_handles_empty_journal():
    svc = importlib.import_module("app_service")

    summary = svc.build_notification_summary({})

    assert summary == {
        "sent": 0,
        "skipped_duplicate": 0,
        "failed": 0,
        "pending": 0,
        "journal_size": 0,
        "last_sent_ts": "",
        "last_failed_ts": "",
        "last_failed_reason": "",
        "trimmed": 0,
    }


def test_build_monitoring_summary_handles_empty_rows():
    svc = importlib.import_module("app_service")

    summary = svc.build_monitoring_summary([], [])

    assert summary["monitoring_rows"] == 0
    assert summary["active_monitoring_rows"] == 0
    assert summary["portfolio_rows"] == 0
    assert summary["active_portfolio_rows"] == 0


def test_build_live_pulse_summary_handles_failed_empty_success():
    svc = importlib.import_module("app_service")

    failed = svc.build_live_pulse_summary({"status": "failed", "last_empty_reason": "worker_failed"})
    empty = svc.build_live_pulse_summary({})
    success = svc.build_live_pulse_summary({"status": "success", "final_count": 2, "blocked_reasons": {"a": 3, "b": 1}})

    assert failed["ok"] is False
    assert failed["empty"] is True
    assert empty["status"] == "empty"
    assert empty["empty"] is True
    assert success["ok"] is True
    assert success["empty"] is False
    assert success["top_blocked_reasons"] == {"a": 3, "b": 1}


def test_app_wrappers_preserve_notification_shape(monkeypatch):
    monkeypatch.setenv("DEX_SCOUT_WORKER_MODE", "1")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")

    summary = app.notification_event_journal_counts({"events": {}})

    assert set(summary) == {
        "sent",
        "skipped_duplicate",
        "failed",
        "pending",
        "journal_size",
        "last_sent_ts",
        "last_failed_ts",
        "last_failed_reason",
        "trimmed",
    }
