import app


def test_final_action_reduce_with_missing_price_forced_action_now():
    row = {"active": "1", "chain": "solana", "base_symbol": "ZEREBRO", "base_token_address": "z1", "final_action": "REDUCE"}
    out = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": []}, now_ts=0)
    assert len(out["action_now"]) == 1
    assert out["action_now"][0]["action"] == "REDUCE"


def test_take_profit_with_missing_price_forced_action_now_and_normalized():
    row = {"active": "1", "chain": "solana", "base_symbol": "BUTTCOIN", "base_token_address": "b1", "final_action": "TAKE PROFIT"}
    out = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": []}, now_ts=0)
    assert len(out["action_now"]) == 1
    assert out["action_now"][0]["action"] == "PARTIAL_TP"


def test_heartbeat_suppressed_when_material_row_exists(monkeypatch):
    sent = []
    monkeypatch.setattr(app, "send_telegram", lambda msg, parse_mode="HTML": sent.append(msg) or True)
    monkeypatch.setattr(app, "load_tg_state", lambda: {"last_position_analysis_dedupe_keys": [], "last_position_analysis_sent_ts": "", "last_position_analysis_heartbeat_ts": ""})
    monkeypatch.setattr(app, "save_tg_state", lambda state: None)
    monkeypatch.setattr(app, "build_notification_candidates", lambda m, p: ([], []))
    monkeypatch.setattr(app, "build_portfolio_meaningful_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(app, "_is_tg_quiet_hours", lambda now_ts: False)
    row = {"active": "1", "chain": "solana", "base_symbol": "ZEREBRO", "base_token_address": "z1", "final_action": "REDUCE"}
    app.run_auto_notifications({}, [], [row], cycle_context={}, trigger_model="ui_scan_sync")
    assert sent
    assert "Portfolio check: no urgent changes" not in sent[0]
    assert "DEX Scout · Position Check" in sent[0]


def test_material_rows_survive_dedupe():
    row = {"active": "1", "chain": "solana", "base_symbol": "ZEREBRO", "base_token_address": "z1", "final_action": "REDUCE"}
    analyzed = app.analyze_position_for_tg(row, None, {"price_context": {}})
    out = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": [analyzed["dedupe_key"]]}, now_ts=0)
    assert len(out["action_now"]) == 1


def test_no_tp_unavailable_spam():
    row = {"active": "1", "chain": "solana", "base_symbol": "BUTTCOIN", "base_token_address": "b1", "final_action": "TAKE PROFIT"}
    out = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": []}, now_ts=0)
    text = app.format_tg_position_analysis(out)
    assert "TP unavailable" not in text
