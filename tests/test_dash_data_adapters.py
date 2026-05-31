import json

import pytest

pytest.importorskip("dash")
pytest.importorskip("plotly")

import dash_app


def clear_dash_caches():
    for fn_name in [
        "read_monitoring_rows",
        "read_portfolio_rows",
        "read_monitoring_history_rows",
        "read_portfolio_reco_rows",
        "read_scanner_state",
        "read_tg_state",
        "read_runtime_snapshot",
    ]:
        clear = getattr(getattr(dash_app, fn_name), "cache_clear", None)
        if clear:
            clear()


def test_read_d1_csv_uses_storage_helper(monkeypatch):
    clear_dash_caches()
    calls = []

    def fake_get_storage(key):
        calls.append(key)
        return "ts_added,chain,base_symbol,base_addr,active,last_score\n2026-01-01 00:00:00 UTC,solana,AAA,AddrA,1,42\n"

    monkeypatch.setattr(dash_app, "d1_get_storage", fake_get_storage)

    rows = dash_app.read_monitoring_rows()

    assert calls == ["monitoring.csv"]
    assert rows[0]["base_symbol"] == "AAA"
    assert rows[0]["last_score"] == "42"


def test_dashboard_snapshot_is_read_only_and_summarizes(monkeypatch):
    clear_dash_caches()
    storage = {
        "monitoring.csv": "ts_added,chain,base_symbol,base_addr,active,last_score,status\n2026-01-01 00:00:00 UTC,solana,AAA,AddrA,1,42,WATCH\n2026-01-01 00:00:00 UTC,solana,OLD,AddrOld,0,1,ARCHIVED\n",
        "portfolio.csv": "ts_utc,chain,base_symbol,base_token_address,active,action\n2026-01-01 00:00:00 UTC,solana,BBB,AddrB,1,HOLD\n",
        "scanner_state.json": json.dumps({"last_run_ts": "2026-01-01 00:05:00 UTC", "last_stats": {"normalized": 3}}),
    }
    methods = []

    def fake_get_storage(key):
        methods.append(("GET_STORAGE", key))
        return storage.get(key)

    def fake_select_rows(table, filters=None, select="*", limit=1):
        methods.append(("SELECT", table, select, limit))
        if table == dash_app.JOB_HEARTBEATS_TABLE:
            return [], {"ok": True, "code": "ok"}
        if table == dash_app.JOB_RUNS_TABLE:
            return [], {"ok": True, "code": "ok"}
        return [{"state_json": {"last_scan_status": "ok", "last_scan_stats": {"seen": 2}, "scan_request_pending": True, "scan_request_ts": "2026-01-01 00:04:00 UTC"}}], {"ok": True, "code": "ok"}

    monkeypatch.setattr(dash_app, "d1_get_storage", fake_get_storage)
    monkeypatch.setattr(dash_app, "d1_select_rows", fake_select_rows)
    monkeypatch.setattr(dash_app, "get_worker_runtime_state", lambda: {"last_scan_status": "ok", "last_scan_stats": {"seen": 2}, "scan_request_pending": True, "scan_request_ts": "2026-01-01 00:04:00 UTC"})
    monkeypatch.setattr(dash_app, "check_runtime_contract", lambda: {"ok": True, "code": "ok"})
    monkeypatch.setattr(dash_app, "_d1_ok", lambda: True)

    snap = dash_app.dashboard_snapshot()

    assert len(snap["active_monitoring"]) == 1
    assert len(snap["archive"]) == 1
    assert len(snap["active_portfolio"]) == 1
    assert snap["pending"] is True
    assert snap["last_scan_status"] == "ok"
    assert snap["last_scan_stats"] == {"seen": 2}
    assert all(method[0] in {"GET_STORAGE", "SELECT"} for method in methods)


def test_monitoring_sparkline_filters_selected_token(monkeypatch):
    clear_dash_caches()
    csv_text = "ts_utc,chain,base_symbol,base_addr,price_usd\n2026-01-01 00:00:00 UTC,solana,AAA,AddrA,1.1\n2026-01-01 00:01:00 UTC,solana,AAA,AddrA,1.2\n2026-01-01 00:00:00 UTC,solana,BBB,AddrB,9.9\n"
    monkeypatch.setattr(dash_app, "d1_get_storage", lambda key: csv_text if key == "monitoring_history.csv" else "")

    fig = dash_app.monitoring_sparkline_figure("solana:AddrA")

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [1.1, 1.2]
