import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import app_runtime_facade
from scripts import runtime_no_fail_matrix as matrix

MANIFEST_PATH = Path("tests/fixtures/app_compat_manifest.json")
REQUIRED_JOB_MODES = {
    "scan_cycle",
    "monitor_cycle",
    "notify_cycle",
    "digest_cycle",
    "outcome_cycle",
    "maintenance_cycle",
}
RUNNER_FUNCTIONS = [
    "maybe_run_rotating_scanner",
    "run_priority_scanner_cycle",
    "run_auto_notifications",
    "trigger_digest_notification",
    "evaluate_outcome_journals",
    "run_storage_maintenance_cycle",
]


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_names():
    names = []
    for values in load_manifest().values():
        names.extend(values)
    return sorted(set(names))


@pytest.fixture(scope="module")
def app_module():
    os.environ["DEX_SCOUT_WORKER_MODE"] = "1"
    sys.modules.pop("app", None)
    app_runtime_facade.load_app_runtime(force_reload=True)
    return importlib.import_module("app")


def test_app_compat_manifest_exists_and_documents_required_categories():
    manifest = load_manifest()
    assert MANIFEST_PATH.exists()
    for category in [
        "worker_runners",
        "runtime_state_locks_heartbeats",
        "storage_compatibility",
        "ui_context_diagnostics",
        "telegram_notifications",
    ]:
        assert category in manifest
        assert manifest[category]


def test_manifest_functions_exist_and_are_callable(app_module):
    missing = [name for name in manifest_names() if not callable(getattr(app_module, name, None))]
    assert not missing, f"app.py compatibility wrappers missing/not callable: {missing}"


def test_facade_runner_map_matches_app_callables_and_resolves(app_module):
    assert set(app_runtime_facade.WORKER_JOB_MODE_RUNNERS) == REQUIRED_JOB_MODES
    for mode, runner_name in app_runtime_facade.WORKER_JOB_MODE_RUNNERS.items():
        assert callable(getattr(app_module, runner_name, None)), f"{mode} runner {runner_name} missing"
        resolved = app_runtime_facade.resolve_worker_job_mode(mode, dry_run=True)
        assert resolved["ok"] is True, f"{mode} failed resolution: {resolved}"


def _assert_signature_contains(fn, params):
    sig = inspect.signature(fn)
    missing = [name for name in params if name not in sig.parameters]
    assert not missing, f"{fn.__name__} missing expected params: {missing}; signature={sig}"


def test_facade_app_signature_contracts(app_module):
    _assert_signature_contains(app_module.update_job_heartbeat, ["job_name", "job_mode", "status", "meta"])
    _assert_signature_contains(app_module.acquire_lock, ["lock_key", "owner", "ttl_sec"])
    _assert_signature_contains(app_module.release_lock, ["lock_key", "owner"])
    _assert_signature_contains(app_module.update_worker_runtime_state, ["updates", "increments"])


def test_runner_functions_have_no_required_positional_args(app_module):
    offenders = []
    for name in RUNNER_FUNCTIONS:
        sig = inspect.signature(getattr(app_module, name))
        required = [
            param_name
            for param_name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if required:
            offenders.append(f"{name}:{required}")
    assert not offenders, f"runner functions require positional args: {offenders}"


def test_audit_script_reports_no_duplicate_top_level_defs_and_valid_json(tmp_path):
    out = tmp_path / "app-compat-audit.json"
    result = subprocess.run(
        [sys.executable, "scripts/audit_app_compat.py", "--json", str(out)],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    file_payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == file_payload
    assert payload["ok"] is True
    assert payload["duplicates"] == []
    assert isinstance(payload["stale_markers"], list)


def test_audit_script_redacts_secret_values(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", "audit-secret-token")
    result = subprocess.run(
        [sys.executable, "scripts/audit_app_compat.py"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "audit-secret-token" not in result.stdout


def test_service_wrapper_parity_for_representative_inputs(app_module, monkeypatch):
    portfolio_service = importlib.import_module("portfolio_service")
    monitoring_service = importlib.import_module("monitoring_service")
    app_service = importlib.import_module("app_service")
    scanner_service = importlib.import_module("scanner_service")

    portfolio_row = {"final_action": "Reduce", "current_price": "1.23"}
    assert app_module.normalize_portfolio_action(" reduce ") == portfolio_service.normalize_portfolio_action(" reduce ")
    assert app_module.is_material_portfolio_action(portfolio_row) == portfolio_service.is_material_portfolio_action(portfolio_row)

    monitoring_row = {"entry_status": " WATCH ", "score": "12", "chain": "solana", "base_addr": "abc"}
    assert app_module.normalize_monitoring_status(monitoring_row) == monitoring_service.normalize_monitoring_status(monitoring_row)
    assert app_module.classify_monitoring_row(monitoring_row) == monitoring_service.classify_monitoring_row(monitoring_row)

    journal = {"events": [{"send_status": "sent", "event_key": "k"}], "notification_journal_trimmed": 0}
    assert app_module.build_notification_summary(journal) == app_service.build_notification_summary(journal)

    if hasattr(app_module, "build_live_pulse_payload"):
        kwargs = {
            "raw_candidates": [{"chain": "solana"}],
            "normalized_candidates": [{"chain": "solana"}],
            "final_candidates": [{"chain": "solana", "base_addr": "abc"}],
            "status": "success",
            "source": "test",
            "now_ts": "2026-01-01T00:00:00Z",
        }
        assert app_module.build_live_pulse_payload(**kwargs) == scanner_service.build_live_pulse_payload(**kwargs)


def test_storage_wrappers_preserve_basic_shapes(app_module, monkeypatch):
    assert app_module.storage_key_for_path("data/example.json").endswith("example.json")
    monkeypatch.setattr(app_module, "storage_read_text", lambda key, default="": '{"ok": true}')
    monkeypatch.setattr(app_module, "storage_write_text", lambda key, text: True)
    assert app_module.load_json("example.json", default={}) == {"ok": True}
    assert app_module.save_json("example.json", {"ok": True}) == {"ok": True, "write_ok": True}


def test_runtime_matrix_includes_app_compat_role():
    result = matrix._app_compat_role()
    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["checked_wrappers"] >= len(manifest_names())
    assert result["duplicates"] == []


def test_service_modules_import_without_app_or_streamlit():
    snapshot = {name: sys.modules.get(name) for name in ("app", "streamlit", *matrix.CORE_MODULES)}
    try:
        sys.modules.pop("app", None)
        sys.modules.pop("streamlit", None)
        for name in matrix.CORE_MODULES:
            sys.modules.pop(name, None)
            importlib.import_module(name)
        assert "app" not in sys.modules
        assert "streamlit" not in sys.modules
    finally:
        for name, module in snapshot.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
