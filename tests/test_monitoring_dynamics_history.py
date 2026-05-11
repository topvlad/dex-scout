import app


def test_load_dynamics_history_matches_by_token_addr(monkeypatch):
    row = {"chain": "solana", "base_symbol": "PIPPIN", "base_addr": "AbCd111"}
    history = [
        {"chain": "solana", "token_addr": "AbCd111", "ts_utc": "2026-01-01 00:00:00 UTC"},
        {"chain": "solana", "token_addr": "AbCd111", "ts_utc": "2026-01-01 00:01:00 UTC"},
    ]
    monkeypatch.setattr(app, "load_csv", lambda *args, **kwargs: history)
    out = app.load_token_dynamics_history(row, limit=120)
    assert len(out) == 2


def test_load_dynamics_history_symbol_only_matches_chain(monkeypatch):
    row = {"chain": "bsc", "symbol": "PIPPIN"}
    history = [
        {"chain": "bsc", "symbol": "pippin", "ts_utc": "2026-01-01 00:00:00 UTC"},
        {"chain": "solana", "symbol": "pippin", "ts_utc": "2026-01-01 00:01:00 UTC"},
    ]
    monkeypatch.setattr(app, "load_csv", lambda *args, **kwargs: history)
    out = app.load_token_dynamics_history(row, limit=120)
    assert len(out) == 1
    assert out[0]["chain"] == "bsc"


def test_solana_alias_matching_is_case_sensitive_for_addresses():
    row_aliases = {"AbCdEfGh1234567890XYZ"}
    hist_aliases = {"abcdefGh1234567890XYZ"}
    assert app._history_alias_overlap("solana", row_aliases, hist_aliases) is False


def test_fast_render_path_does_not_call_history_loader(monkeypatch):
    called = {"count": 0}

    def _boom(*_args, **_kwargs):
        called["count"] += 1
        return []

    monkeypatch.setattr(app, "load_token_dynamics_history", _boom)
    app.render_monitoring_sparklines([])
    assert called["count"] == 0
