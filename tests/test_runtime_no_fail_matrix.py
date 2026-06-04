import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app_runtime_facade

matrix = importlib.import_module("scripts.runtime_no_fail_matrix")


def test_runtime_matrix_script_builds_schema_and_redacts_secret(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", "matrix-secret-token")
    summary = matrix.build_matrix()
    payload = json.dumps(summary, sort_keys=True, default=str)

    assert set(summary) == {"ok", "ts_utc", "commit", "roles", "secrets_redacted"}
    assert set(summary["roles"]) == {"ui_streamlit", "worker", "webhook", "dash_readonly", "app_compat", "core_modules", "golden_fixtures", "admin_controls", "external_audit_claims"}
    assert summary["secrets_redacted"] is True
    assert "matrix-secret-token" not in payload


def test_core_modules_role_succeeds_without_importing_app(monkeypatch):
    sys.modules.pop("app", None)
    result = matrix._core_modules_role()

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert "app" not in sys.modules
    assert "streamlit_imported_by_core_modules" not in result.get("errors", [])
    assert "app_service" in result.get("modules", [])
    assert "portfolio_service" in result.get("modules", [])


def test_worker_role_reports_app_import_failed_on_forced_facade_failure(monkeypatch):
    monkeypatch.setattr(
        app_runtime_facade,
        "load_app_runtime",
        lambda force_reload=False: {"ok": False, "error": "forced-secret-free", "exception_type": "RuntimeError", "app": None},
    )

    result = matrix._worker_role()

    assert result["ok"] is False
    assert result["status"] == "app_import_failed"
    assert result["exception_type"] == "RuntimeError"


def test_webhook_role_reports_degraded_state_on_forced_app_failure():
    result = matrix._webhook_role()

    assert result["ok"] is True
    assert result["forced_degraded"]["ok"] is False
    assert result["forced_degraded"]["status"] == "app_import_failed"


def test_dash_role_skips_when_dash_app_absent(monkeypatch):
    real_exists = Path.exists

    def fake_exists(path):
        if str(path) == "dash_app.py":
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    result = matrix._dash_readonly_role()

    assert result["ok"] is True
    assert result["status"] == "skipped_absent"


def test_worker_role_verifies_facade_heartbeat_forwarding(monkeypatch):
    fake_app = SimpleNamespace(
        maybe_run_rotating_scanner=lambda: None,
        run_priority_scanner_cycle=lambda: None,
        run_auto_notifications=lambda: None,
        trigger_digest_notification=lambda: None,
        evaluate_outcome_journals=lambda: None,
        run_storage_maintenance_cycle=lambda: None,
    )
    monkeypatch.setattr(app_runtime_facade, "load_app_runtime", lambda force_reload=False: {"ok": True, "app": fake_app})

    result = matrix._worker_role()

    assert result["ok"] is True
    assert result["facade_forwarding"]["ok"] is True
    assert result["facade_forwarding"]["heartbeat"]["job_mode"] == "notify_cycle"
    assert result["facade_forwarding"]["missing_function"]["status"] == "missing_app_function"


def test_facade_job_mode_resolver_detects_required_modes(monkeypatch):
    fake_app = SimpleNamespace(
        maybe_run_rotating_scanner=lambda: None,
        run_priority_scanner_cycle=lambda: None,
        run_auto_notifications=lambda: None,
        trigger_digest_notification=lambda: None,
        evaluate_outcome_journals=lambda: None,
        run_storage_maintenance_cycle=lambda: None,
    )
    monkeypatch.setattr(app_runtime_facade, "load_app_runtime", lambda force_reload=False: {"ok": True, "app": fake_app})

    for mode in app_runtime_facade.REQUIRED_WORKER_JOB_MODES:
        result = app_runtime_facade.resolve_worker_job_mode(mode, dry_run=True)
        assert result["ok"] is True
        assert result["status"] == "resolved"


def test_facade_invalid_job_mode_fails_explicitly(monkeypatch):
    result = app_runtime_facade.resolve_worker_job_mode("not_a_mode", dry_run=True)

    assert result["ok"] is False
    assert result["status"] == "invalid_mode"


def test_facade_missing_app_function_reports_missing_runner(monkeypatch):
    fake_app = SimpleNamespace()
    monkeypatch.setattr(app_runtime_facade, "load_app_runtime", lambda force_reload=False: {"ok": True, "app": fake_app})

    result = app_runtime_facade.resolve_worker_job_mode("scan_cycle", dry_run=True)

    assert result["ok"] is False
    assert result["status"] == "missing_runner"


def test_runtime_matrix_json_file_schema(tmp_path):
    out = tmp_path / "runtime-no-fail-summary.json"
    with patch.object(matrix, "build_matrix", lambda: {
        "ok": True,
        "ts_utc": "2026-01-01T00:00:00Z",
        "commit": "abc",
        "roles": {name: {"ok": True, "status": "ok", "errors": []} for name in matrix.ROLE_ORDER},
        "secrets_redacted": True,
    }):
        assert matrix.main(["--json", str(out)]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["roles"]["ui_streamlit"]["errors"] == []


def test_runtime_matrix_includes_portfolio_service_smokes():
    result = matrix._core_modules_role()

    assert result["ok"] is True
    assert "portfolio_service" in result.get("modules", [])
    assert not any("portfolio_material_classifier_smoke_failed" in err for err in result.get("errors", []))
    assert not any("portfolio_conflict_resolver_smoke_failed" in err for err in result.get("errors", []))


def test_runtime_matrix_includes_golden_fixtures_role():
    result = matrix._golden_fixtures_role()

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["fixtures_checked"] >= 8

