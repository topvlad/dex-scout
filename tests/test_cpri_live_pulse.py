import json

import app
import worker


def _row(i, **extra):
    row = {
        "chain": "solana",
        "base_addr": f"addr{i}",
        "base_symbol": f"TK{i}",
        "entry_status": "WATCH",
        "score": "100",
        "liq_usd": "100000",
        "vol24_usd": "100000",
        "active": "1",
        "status": "ACTIVE",
    }
    row.update(extra)
    return row


def _patch_pulse_deps(monkeypatch, rows):
    monkeypatch.setattr(app, "scanner_state_load", lambda: {"queue_state": {}, "worker_runtime": {}})
    monkeypatch.setattr(app, "build_discovery_candidate_pool", lambda **kwargs: list(rows))
    monkeypatch.setattr(app, "best_pair_for_token_cached", lambda chain, ca: {"liquidity": {"usd": 100_000}, "volume": {"h24": 100_000}, "priceUsd": 1, "txns": {"h1": {"buys": 5, "sells": 5}}, "priceChange": {"h24": 5}})
    monkeypatch.setattr(app, "_pulse_card_liquidity_usd", lambda best, row: app.parse_float(row.get("liq_usd", 100000), 0))
    monkeypatch.setattr(app, "_pulse_freshness_minutes", lambda row, queue: 1)
    monkeypatch.setattr(app, "_pulse_status_marker", lambda row: "ok")
    monkeypatch.setattr(app, "_pulse_card_summary_line", lambda best, row: "ok")
    monkeypatch.setattr(app, "_pulse_card_mini_sparkline", lambda row: {})


def test_fake_scan_with_raw_candidates_writes_payload(monkeypatch):
    rows = [_row(i) for i in range(3)]
    _patch_pulse_deps(monkeypatch, rows)
    monkeypatch.setattr(app, "evaluate_live_pulse_scam_gate", lambda row: {"blocked": False, "reasons": []})
    payload = app.build_live_pulse_candidates_payload(rows, [], set(), scan_result={"stats": {"seen": 3}, "raw_candidates": rows}, status="success")
    assert payload["raw_seen"] == 3
    assert payload["final_count"] == 3
    assert payload["status"] == "success"


def test_fake_scan_all_rejected_writes_blocked_reasons(monkeypatch):
    rows = [_row(i) for i in range(3)]
    _patch_pulse_deps(monkeypatch, rows)
    monkeypatch.setattr(app, "evaluate_live_pulse_scam_gate", lambda row: {"blocked": True, "reasons": ["scam"]})
    payload = app.build_live_pulse_candidates_payload(rows, [], set(), scan_result={"stats": {"seen": 3}, "raw_candidates": rows}, status="success")
    assert payload["raw_seen"] == 3
    assert payload["final_count"] == 0
    assert payload["last_empty_reason"] == "all_filtered_scam"
    assert payload["blocked_reasons"].get("all_filtered_scam") == 3


def test_worker_exception_writes_failed_live_pulse_payload(monkeypatch):
    writes = []
    monkeypatch.setattr(app, "get_worker_runtime_state", lambda *a, **k: {})
    monkeypatch.setattr(app, "maybe_run_rotating_scanner", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(app, "load_monitoring", lambda: (_ for _ in ()).throw(RuntimeError("pulse boom")))
    monkeypatch.setattr(app, "update_worker_runtime_state", lambda updates=None, **kwargs: writes.append(updates or {}))
    monkeypatch.setattr(app, "live_pulse_storage_write", lambda payload: writes.append({"storage": payload}) or True)
    monkeypatch.setattr(worker, "_record_pulse_history_after_cycle_safe", lambda: {"ok": True})
    result = worker._run_scan_cycle()
    stored = [w["storage"] for w in writes if "storage" in w][-1]
    assert stored["status"] == "failed"
    assert stored["last_empty_reason"] == "worker_failed"
    assert result["scan"].get("error")
