import app
import worker


def test_monitor_cycle_archives_hard_gated_rows(monkeypatch):
    rows = [{"active": "1", "chain": "solana", "base_addr": "milk", "base_symbol": "MILKERS", "status": "UNTRADEABLE", "liq_usd": "0", "vol24_usd": "0"}]
    archived = []
    monkeypatch.setattr(app, "load_monitoring", lambda: rows)
    monkeypatch.setattr(app, "load_portfolio", lambda: [])
    monkeypatch.setattr(app, "run_priority_scanner_cycle", lambda **kwargs: {})
    monkeypatch.setattr(app, "append_token_history_snapshot", lambda *a, **k: False)
    monkeypatch.setattr(app, "flush_monitoring_history_buffer", lambda *a, **k: None)
    monkeypatch.setattr(app, "build_active_monitoring_rows", lambda r: [x for x in r if x.get("active") == "1"])
    monkeypatch.setattr(app, "archive_monitoring", lambda chain, base_addr, reason, revisit_days=0: archived.append((chain, base_addr, reason)) or True)
    monkeypatch.setattr(worker, "_record_pulse_history_after_cycle_safe", lambda: {})
    out = worker._run_monitor_cycle()
    assert out["monitor"]["hard_gate_archived"] == 1
    assert archived and archived[0][1] == "milk"
    assert "untradeable" in archived[0][2]


def test_priority_watchlist_excludes_hard_gated_row():
    row = {"chain": "solana", "base_addr": "milk", "base_symbol": "MILKERS", "status": "UNTRADEABLE", "liq_usd": "0", "vol24_usd": "0"}
    gate = app.hard_gate_monitoring_row(row)
    assert gate["blocked"] is True
    assert gate["action"] == "ARCHIVE"


def test_portfolio_material_action_overrides_monitoring_watch():
    mon = {"chain": "solana", "base_addr": "addr", "base_symbol": "AAA", "entry_status": "WATCH", "status": "ACTIVE", "liq_usd": "10000", "vol24_usd": "10000"}
    pf = {"active": "1", "chain": "solana", "base_token_address": "addr", "base_symbol": "AAA", "final_action": "REDUCE"}
    gate = app.hard_gate_monitoring_row(mon, portfolio_row=pf)
    assert gate["blocked"] is True
    assert "portfolio_verdict_conflict" in gate["reason"]
    assert app.normalize_material_portfolio_action(pf["final_action"]) == "REDUCE"
