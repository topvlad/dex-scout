import worker


class PulseFailAppStub:
    def now_utc_str(self):
        return "2026-01-01T00:00:00Z"

    def update_worker_runtime_state(self, updates=None, increments=None):
        return {"ok": True}

    def update_job_heartbeat(self, **kwargs):
        return {"ok": True}

    def record_live_pulse_history_from_candidates(self, **kwargs):
        raise RuntimeError("pulse fail")

    def load_monitoring(self):
        return []

    def build_active_monitoring_rows(self, rows):
        return []

    def canonical_token_key(self, row):
        return ""


class ScanMonitorStub(PulseFailAppStub):
    DEFAULT_SEEDS = "eth,sol"

    def maybe_run_rotating_scanner(self, **kwargs):
        return {"ok": True, "mode": "scan"}

    def load_portfolio(self):
        return []

    def run_priority_scanner_cycle(self, **kwargs):
        return {"ok": True, "mode": "monitor"}


def test_pulse_history_safe_wrapper_nonblocking(monkeypatch):
    monkeypatch.setattr(worker, "app", PulseFailAppStub())
    result = worker._record_pulse_history_after_cycle_safe()
    assert result["ok"] is False
    assert result["nonblocking"] is True
    assert "pulse_history_writer_exception" in result["reason"]


def test_run_scan_cycle_nonblocking_when_pulse_writer_fails(monkeypatch):
    monkeypatch.setattr(worker, "app", ScanMonitorStub())
    result = worker._run_scan_cycle()
    assert result["scan"]["ok"] is True
    assert result["pulse_history"]["ok"] is False
    assert result["pulse_history"]["nonblocking"] is True


def test_run_monitor_cycle_nonblocking_when_pulse_writer_fails(monkeypatch):
    monkeypatch.setattr(worker, "app", ScanMonitorStub())
    result = worker._run_monitor_cycle()
    assert result["monitor"]["ok"] is True
    assert result["pulse_history"]["ok"] is False
    assert result["pulse_history"]["nonblocking"] is True
