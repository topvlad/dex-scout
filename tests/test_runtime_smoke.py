import json

import worker


class SmokeAppStub:
    def __init__(self):
        self.writes = 0

    def check_runtime_contract(self, required_tables=None):
        return {"ok": True, "code": "ok", "tables": required_tables or []}

    def read_runtime_state(self):
        return {"worker_status": "idle"}, {"ok": True, "code": "ok"}

    def load_notification_event_journal(self):
        return {"version": 1, "events": {}}

    def live_pulse_storage_read(self):
        return {"status": "empty", "final_candidates": []}

    def load_monitoring(self):
        return [{"base_symbol": "AAA"}, {"base_symbol": "BBB"}]

    def load_portfolio(self):
        return [{"base_symbol": "AAA"}]

    def load_tg_state(self):
        return {"settings": {}}

    def storage_key_for_path(self, key):
        return key

    def _app_storage_remote_ok(self):
        return True

    def sb_get_storage(self, key):
        return {
            "notification_event_journal.json": "{\"version\": 1, \"events\": {}}",
            "live_pulse_candidates.json": "{\"status\": \"empty\", \"final_candidates\": []}",
            "monitoring.csv": "base_symbol\nAAA\nBBB\n",
            "portfolio.csv": "base_symbol\nAAA\n",
            "tg_state.json": "{\"settings\": {}}",
        }.get(key, "")

    def storage_write_text(self, *args, **kwargs):  # would indicate a read-only violation
        self.writes += 1
        raise AssertionError("read-only smoke must not write")


def test_dry_run_scan_cycle_does_not_call_scan_runner(monkeypatch):
    called = []
    monkeypatch.setenv("JOB_DRY_RUN", "true")
    monkeypatch.setattr(worker, "app", object())
    monkeypatch.setattr(worker, "JOB_DISPATCH", {"scan_cycle": lambda: called.append("scan") or {"ok": True}})

    assert worker.run_job_mode("scan_cycle") == 0
    assert called == []


def test_dry_run_notify_cycle_does_not_call_send_telegram_path(monkeypatch):
    called = []
    monkeypatch.setenv("JOB_DRY_RUN", "true")
    monkeypatch.setattr(worker, "app", object())
    monkeypatch.setattr(worker, "JOB_DISPATCH", {"notify_cycle": lambda: called.append("send_telegram") or {"ok": True}})

    assert worker.run_job_mode("notify_cycle") == 0
    assert called == []


def test_dry_run_invalid_mode_still_fails(monkeypatch):
    monkeypatch.setenv("JOB_DRY_RUN", "true")
    monkeypatch.setattr(worker, "app", object())
    monkeypatch.setattr(worker, "JOB_DISPATCH", {"scan_cycle": lambda: {"ok": True}})

    assert worker.run_job_mode("not_a_mode") == 2


def test_read_only_smoke_helper_does_not_write(monkeypatch):
    stub = SmokeAppStub()
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    summary = worker.run_runtime_smoke_read_only(stub)

    assert summary["ok"] is True
    assert summary["runtime_state_ok"] is True
    assert summary["heartbeats_ok"] is True
    assert summary["notification_journal_ok"] is True
    assert summary["live_pulse_ok"] is True
    assert summary["monitoring_rows"] == 2
    assert summary["portfolio_rows"] == 1
    assert stub.writes == 0


def test_safe_sequence_skips_notify_digest_when_tg_not_allowed(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "run_job_mode", lambda mode: called.append(mode) or 0)

    summary = worker.run_safe_runtime_sequence(dry_run=False, allow_tg_send=False, allow_scan=True)

    skipped = {step["job_mode"]: step for step in summary["steps"] if step["status"] == "skipped"}
    assert skipped["notify_cycle"]["reason"] == "allow_tg_send_false"
    assert skipped["digest_cycle"]["reason"] == "allow_tg_send_false"
    assert "notify_cycle" not in called
    assert "digest_cycle" not in called


def test_safe_sequence_skips_scan_when_scan_not_allowed(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "run_job_mode", lambda mode: called.append(mode) or 0)

    summary = worker.run_safe_runtime_sequence(dry_run=False, allow_tg_send=True, allow_scan=False)

    scan_step = [step for step in summary["steps"] if step["job_mode"] == "scan_cycle"][0]
    assert scan_step["status"] == "skipped"
    assert scan_step["reason"] == "allow_scan_false"
    assert "scan_cycle" not in called


def test_summary_json_does_not_include_secret_env_values(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("TG_BOT_TOKEN", "telegram-secret-value")
    monkeypatch.setenv("TG_CHAT_ID", "123456")
    monkeypatch.setenv("D1_PROXY_URL", "https://example.invalid/secret-url")
    monkeypatch.setenv("D1_PROXY_TOKEN", "d1-secret-token")
    monkeypatch.setenv("STORAGE_BACKEND", "d1")

    summary = worker.build_runtime_smoke_summary(
        mode="preflight_only",
        dry_run=True,
        allow_tg_send=False,
        allow_scan=False,
        steps=[],
    )
    payload = json.dumps(summary, sort_keys=True)

    assert "telegram-secret-value" not in payload
    assert "123456" not in payload
    assert "https://example.invalid/secret-url" not in payload
    assert "d1-secret-token" not in payload
    assert summary["config"]["d1_configured"] is True
    assert summary["config"]["tg_configured"] is True
