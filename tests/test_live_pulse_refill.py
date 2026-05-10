import app


def _mk_row(i, status="watch"):
    return {
        "chain": "solana",
        "base_addr": f"addr{i}",
        "base_symbol": f"TK{i}",
        "entry_status": status,
        "score": 10,
    }


def _setup_common(monkeypatch, rows):
    monkeypatch.setattr(app, "scanner_state_load", lambda: {"queue_state": {}, "worker_runtime": {}})
    monkeypatch.setattr(app, "build_discovery_candidate_pool", lambda **kwargs: rows)
    monkeypatch.setattr(app, "best_pair_for_token_cached", lambda chain, ca: {"liquidity": {"usd": 100_000}, "volume": {"h24": 100_000}, "priceUsd": 1, "txns": {"h1": {"buys": 10, "sells": 10}, "m5": {"buys": 5, "sells": 5}}, "priceChange": {"h1": 1, "m5": 0, "h24": 5}})
    monkeypatch.setattr(app, "_pulse_card_liquidity_usd", lambda best, row: 100_000)
    monkeypatch.setattr(app, "_pulse_freshness_minutes", lambda row, queue: 1)
    monkeypatch.setattr(app, "_pulse_status_marker", lambda row: "ok")
    monkeypatch.setattr(app, "_pulse_card_summary_line", lambda best, row: "ok")
    monkeypatch.setattr(app, "_pulse_card_mini_sparkline", lambda row: "")


def test_live_pulse_refill_returns_10_clean(monkeypatch):
    rows = [_mk_row(i) for i in range(200)]
    _setup_common(monkeypatch, rows)
    monkeypatch.setattr(app, "evaluate_live_pulse_scam_gate", lambda row: {"blocked": int(row["base_addr"].replace("addr", "")) < 190, "reasons": ["scam"]})
    cards = app.build_market_pulse_cards(rows, [], set(), limit=15, record_pulse_history=False)
    assert len(cards) == 10


def test_live_pulse_all_scam(monkeypatch):
    rows = [_mk_row(i) for i in range(30)]
    _setup_common(monkeypatch, rows)
    monkeypatch.setattr(app, "evaluate_live_pulse_scam_gate", lambda row: {"blocked": True, "reasons": ["scam"]})
    cards = app.build_market_pulse_cards(rows, [], set(), limit=15, record_pulse_history=False)
    assert len(cards) == 0
    stats = app.st.session_state.get("last_pulse_ui_stats", {})
    assert stats.get("scam_blocked", 0) == 30


def test_live_pulse_excludes_dupes_archive_portfolio_and_caps_15(monkeypatch):
    rows = [_mk_row(i) for i in range(40)]
    _setup_common(monkeypatch, rows)
    monkeypatch.setattr(app, "evaluate_live_pulse_scam_gate", lambda row: {"blocked": False, "reasons": []})
    active_keys = {"solana|addr1"}
    cards = app.build_market_pulse_cards(rows, [], active_keys, limit=99, record_pulse_history=False)
    assert len(cards) == 15
    keys = {c["key"] for c in cards}
    assert "solana|addr1" not in keys
