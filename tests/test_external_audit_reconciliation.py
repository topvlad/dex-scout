import asyncio
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_scoring_legacy_status_and_non_negative_score_row():
    source = (ROOT / "scoring.py").read_text(encoding="utf-8")
    assert "LEGACY FILE" in source or "not imported by any active codepath" in source
    assert "SCORE_MIN_VALUE" in source

    import scoring

    valid_row = {"buys_m5": 20, "sells_m5": 5, "vol_m5": 50_000, "liqUsd": 25_000, "pc_m5": 4, "pc_h1": 8, "txns_m5": 25}
    assert scoring.score_row(valid_row) >= 0


def test_active_scoring_source_audit_and_floor_regression():
    if "app" in sys.modules:
        sys.modules.pop("app", None)
    app = importlib.import_module("app")
    toxic_high_penalty_row = {
        "liquidity": {"usd": 0},
        "volume": {"h24": 0, "m5": 0},
        "priceChange": {"m5": -40, "h1": -90},
        "txns": {"m5": {"buys": 0, "sells": 0}},
        "token_symbol": "Honeypot Inu",
        "symbol": "HONEYPOT",
        "pairCreatedAt": 0,
    }
    audit = {
        "active_scoring_owner": "app.score_pair / app.evaluate_entry_safe / app.evaluate_entry_aggressive",
        "legacy_scoring_py_imported_by_active_path": "import scoring" in (ROOT / "app.py").read_text(encoding="utf-8"),
        "non_negative_guard_present": app.score_pair(toxic_high_penalty_row) >= 0 and app.evaluate_entry_safe({"age_minutes": 999, "fdv": 1})[1] >= 0,
    }
    assert audit == {
        "active_scoring_owner": "app.score_pair / app.evaluate_entry_safe / app.evaluate_entry_aggressive",
        "legacy_scoring_py_imported_by_active_path": False,
        "non_negative_guard_present": True,
    }
    assert app.score_pair(toxic_high_penalty_row) >= 0


def test_worker_lock_key_source_guard_and_early_return_no_nameerror(monkeypatch):
    import worker

    source = inspect.getsource(worker.run_job_mode)
    assert source.find("lock_key = None") < source.find("app.acquire_lock(lock_key=lock_key")
    assert "if lock_key is not None" in source
    assert "app.release_lock(lock_key=lock_key" in source

    fake = _FakeWorkerApp(runtime={})
    monkeypatch.setattr(worker, "app", fake)
    assert worker.run_job_mode("missing_mode") == 2
    assert fake.job_release_count == 0


class _FakeWorkerApp:
    def __init__(self, runtime=None, lock_ok=True, lock_code="ok"):
        self.runtime = runtime if runtime is not None else {}
        self.lock_ok = lock_ok
        self.lock_code = lock_code
        self.heartbeats = []
        self.releases = []
        self.job_release_count = 0
        self.runtime_updates = []

    def acquire_lock(self, lock_key, owner, ttl_sec):
        if lock_key == "worker_runtime_update":
            return {"ok": True, "code": "ok"}
        return {"ok": self.lock_ok, "code": self.lock_code, "detail": self.lock_code}

    def release_lock(self, lock_key, owner):
        self.releases.append(lock_key)
        if str(lock_key).startswith("job_mode:"):
            self.job_release_count += 1
        return {"ok": True}

    def get_worker_runtime_state(self):
        return self.runtime

    def update_worker_runtime_state(self, updates=None, increments=None):
        self.runtime_updates.append((updates or {}, increments or {}))
        if updates:
            self.runtime.update(updates)
        return {"ok": True}

    def update_job_heartbeat(self, **kwargs):
        self.heartbeats.append(kwargs)
        return {"ok": True}

    def get_supabase_storage_budget_snapshot(self):
        return {"ok": True}


def test_worker_dispatcher_early_return_and_lock_paths(monkeypatch):
    import worker

    monkeypatch.setattr(worker, "JOB_DISPATCH", {"ok_mode": lambda: {"ok": True}})
    monkeypatch.setattr(worker, "validate_job_dispatch", lambda dispatch=None: None)

    duplicate_runtime = {"modes": {"ok_mode": {"last_job_status": "running", "last_job_started_epoch": time.time()}}}
    duplicate_app = _FakeWorkerApp(runtime=duplicate_runtime)
    monkeypatch.setattr(worker, "app", duplicate_app)
    assert worker.run_job_mode("ok_mode") == 3
    assert duplicate_app.job_release_count == 0
    assert duplicate_app.heartbeats[-1]["status"] == "duplicate_guard"

    lock_failed_app = _FakeWorkerApp(runtime={"modes": {}}, lock_ok=False, lock_code="lock_failed")
    monkeypatch.setattr(worker, "app", lock_failed_app)
    assert worker.run_job_mode("ok_mode") == 3
    assert lock_failed_app.job_release_count == 0
    assert lock_failed_app.heartbeats[-1]["status"] == "lock_failed"

    success_app = _FakeWorkerApp(runtime={"modes": {}})
    monkeypatch.setattr(worker, "app", success_app)
    assert worker.run_job_mode("ok_mode") == 0
    assert success_app.job_release_count == 1


def _reload_tg(monkeypatch, ok=True):
    import app_runtime_facade

    monkeypatch.setattr(app_runtime_facade, "get_import_state", lambda force_reload=False: {"ok": ok, "error": "forced", "exception_type": "ForcedError", "app_module_loaded": ok})
    monkeypatch.setattr(app_runtime_facade, "get_app", lambda: SimpleNamespace() if ok else None)
    if "tg_webhook" in sys.modules:
        return importlib.reload(sys.modules["tg_webhook"])
    return importlib.import_module("tg_webhook")


class _Req:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def test_tg_webhook_degraded_state_regression(monkeypatch):
    tg = _reload_tg(monkeypatch, ok=False)
    monkeypatch.setattr(tg.app_runtime_facade, "get_import_state", lambda force_reload=False: {"ok": False, "error": "forced", "exception_type": "ForcedError", "app_module_loaded": False})
    status = tg.import_status()
    assert status["ok"] is False
    assert status["import_failed"] is True
    assert tg.health()["ok"] is False
    assert tg.health_head().status_code == 500
    payload = {"callback_query": {"id": "cb1", "data": "pf|bsc|" + "a" * 40}}
    result = asyncio.run(tg.tg_webhook(_Req(payload)))
    assert result["ok"] is False
    assert result["error"] == "app_import_failed"


def test_tg_webhook_ok_path_constant_time_aliases_and_invalid_callback(monkeypatch):
    tg = _reload_tg(monkeypatch, ok=True)
    monkeypatch.setattr(tg.app_runtime_facade, "get_import_state", lambda force_reload=False: {"ok": True, "error": "", "exception_type": "", "app_module_loaded": True})
    assert tg.import_status()["ok"] is True
    assert tg.health() == {"ok": True}

    called = []
    monkeypatch.setenv("TG_SUMMARY_KEY", "expected-secret")
    monkeypatch.setattr(tg.hmac, "compare_digest", lambda supplied, expected: called.append((supplied, expected)) or supplied == expected)
    assert tg.summary_key_ok("expected-secret") is True
    assert called and called[-1] == ("expected-secret", "expected-secret")

    aliases = ("base_token_address", "base_addr", "token_addr", "token_address", "ca", "address", "pair_address")
    assert hasattr(tg, "_row_matches_token")
    for field in aliases:
        assert tg._row_matches_token({"chain": "bsc", field: "A" * 40}, "bsc", "A" * 40)

    rows = [{"chain": "bsc", "token_addr": "B" * 40, "active": "1", "archived": "0"}]
    monkeypatch.setattr(tg, "load_portfolio", lambda: [dict(row) for row in rows])
    monkeypatch.setattr(tg, "load_monitoring", lambda: [dict(row) for row in rows])
    saved = {}
    monkeypatch.setattr(tg, "save_portfolio", lambda updated: saved.setdefault("portfolio", updated) or {"ok": True})
    monkeypatch.setattr(tg, "save_monitoring", lambda updated: saved.setdefault("monitoring", updated) or {"ok": True})
    assert tg.add_contract_to_portfolio("bsc", "B" * 40) is True
    assert saved["portfolio"][0]["active"] == "1"

    saved.pop("portfolio", None)
    saved.pop("monitoring", None)
    monkeypatch.setattr(tg, "load_portfolio", lambda: [dict(row) for row in saved.get("portfolio", rows)])
    monkeypatch.setattr(tg, "load_monitoring", lambda: [dict(row) for row in saved.get("monitoring", rows)])
    result = tg.remove_contract_everywhere_atomicish("bsc", "B" * 40)
    assert result["ok"] is True

    invalid = asyncio.run(tg.tg_webhook(_Req({"callback_query": {"id": "cb2", "data": "pf|badchain|" + "C" * 40}})))
    assert invalid["ok"] is False
    assert invalid["error"] == "invalid_chain"
    invalid_ca = asyncio.run(tg.tg_webhook(_Req({"callback_query": {"id": "cb3", "data": "pf|bsc|short"}})))
    assert invalid_ca["ok"] is False
    assert invalid_ca["error"] in {"invalid_contract", "invalid_ca"}


def test_streamlit_stub_imports_app_without_real_streamlit():
    if "app" in sys.modules:
        sys.modules.pop("app", None)
    app = importlib.import_module("app")
    import streamlit

    assert app is not None
    for name in ("cache_data", "session_state", "sidebar", "set_page_config", "columns", "expander", "button"):
        assert hasattr(streamlit, name)
    assert getattr(streamlit, "_DEX_SCOUT_TEST_STUB", False) is True or "tests/stubs/streamlit.py" in str(getattr(streamlit, "__file__", ""))


def test_stale_review_text_guard_in_master_guide():
    guide = (ROOT / "docs" / "DEX_SCOUT_INTERVENTION_GUIDE.html").read_text(encoding="utf-8")
    stale_phrases = [
        "Supabase primary",
        "Немає тестів",
        "tg_webhook silent import fail",
        "score_row negative",
        "worker lock_key NameError",
        "Render + Supabase",
    ]
    for phrase in stale_phrases:
        assert phrase not in guide
    assert "External audit reconciliation" in guide
    assert "historical" in guide.lower() or "fixed" in guide.lower()


def test_audit_external_review_claims_script_contract():
    from scripts import audit_external_review_claims

    report = audit_external_review_claims.audit_claims()
    assert set(report["claims"]) == {
        "scoring_negative_active",
        "worker_lock_key_nameerror",
        "tg_webhook_silent_import_fail",
        "summary_key_not_constant_time",
        "duplicated_row_matching",
        "missing_streamlit_test_stub",
    }
    assert report["ok"] is True
    assert report["active_risks"] == []
    assert all(value["status"] == "false" for value in report["claims"].values())
    json.dumps(report)
