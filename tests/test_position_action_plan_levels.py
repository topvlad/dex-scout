import app


def test_levels_take_profit_with_trim_and_levels():
    row = {
        "chain": "solana",
        "base_symbol": "BUTTCOIN",
        "base_token_address": "Abc",
        "final_action": "TAKE PROFIT",
        "risk_level": "MEDIUM",
        "price_usd": "0.01",
        "avg_entry_price": "0.006",
    }
    plan = app.build_position_action_plan(row, {})
    assert plan["action"] == "TAKE_PROFIT"
    assert plan["reduce_size"] in {"25-33%", "33-50%"}
    assert "unavailable" not in plan["tp1"]
    assert "unavailable" not in plan["protect"]


def test_no_price_explicit_unavailable_not_blank():
    row = {"base_symbol": "NOPRICE", "final_action": "WATCH CLOSELY"}
    plan = app.build_position_action_plan(row, {})
    assert plan["tp1"] == ""
    assert plan["tp2"] == ""
    assert plan["protect"] == ""
    assert plan["add_zone"] == ""
    assert plan["data_quality"] == "missing"
    assert "missing price snapshot" in plan["reason"]


def test_watch_closely_not_unexplained_hold():
    row = {"base_symbol": "WATCH", "final_action": "WATCH CLOSELY", "price_usd": "1", "avg_entry_price": "0.8"}
    out = app.analyze_position_for_tg(row, None, {})
    assert out["action"] in {"PROTECT", "PARTIAL_TP", "HOLD"}
    assert "trend still alive" not in out["reason"]


def test_tg_format_take_profit_has_tp_protect_trim():
    row = {"active": "1", "base_symbol": "BUTT", "base_token_address": "X", "chain": "solana", "final_action": "TAKE PROFIT", "price_usd": "0.01", "avg_entry_price": "0.007"}
    analysis = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": []}, now_ts=0)
    text = app.format_tg_position_analysis(analysis)
    assert "trim" in text
    assert "TP:" in text
    assert "protect" in text


def test_missing_price_exit_stays_action_now_and_no_unavailable_spam():
    row = {
        "active": "1",
        "chain": "solana",
        "base_symbol": "RUG",
        "base_token_address": "RUG1",
        "final_action": "EXIT",
        "health_label": "UNTRADEABLE",
        "health_reasons": ["UNTRADEABLE – no liquidity and no recent flow"],
    }
    analysis = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": []}, now_ts=0)
    assert analysis["action_now"]
    text = app.format_tg_position_analysis(analysis)
    assert "RUG · EXIT" in text
    assert "No urgent change" not in text
    assert "TP unavailable" not in text
    assert "protect unavailable" not in text


def test_missing_price_hold_is_skipped_and_summarized():
    row = {"active": "1", "chain": "solana", "base_symbol": "WAIT", "base_token_address": "WAIT1", "final_action": "HOLD"}
    analysis = app.build_tg_position_analysis([row], [], {}, {"last_position_analysis_dedupe_keys": []}, now_ts=0)
    assert analysis["action_now"] == []
    assert analysis["watch"] == []
    text = app.format_tg_position_analysis(analysis)
    assert "positions skipped: missing price snapshot" in text
