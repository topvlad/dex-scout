import app


def test_position_analysis_skipped_count_respects_visible_limit():
    portfolio = []
    for i in range(9):
        portfolio.append({"active": "1", "chain": "solana", "base_symbol": f"T{i}", "base_token_address": f"Addr{i}", "final_action": "HOLD", "risk_level": "LOW", "entry_score": "10", "price_usd": "1"})
    tg_state = {"last_position_analysis_dedupe_keys": []}
    out = app.build_tg_position_analysis(portfolio, [], {}, tg_state, now_ts=0)
    # watch list is capped, so others are skipped/unchanged
    assert out["skipped_count"] == 4


def test_build_position_action_plan_take_profit_includes_trim_and_protect():
    row = {"chain": "solana", "base_symbol": "BUTTCOIN", "base_token_address": "Abc", "final_action": "PARTIAL_TP", "risk_level": "MEDIUM", "price_usd": "0.01", "entry_price_usd": "0.005"}
    plan = app.build_position_action_plan(row, {})
    assert plan["action"] == "TAKE_PROFIT"
    assert plan["reduce_size"] in {"25%", "33%", "33-50%"}
    assert plan["protect"]


def test_unchanged_hold_not_material_after_dedupe_seen():
    row = {"active": "1", "chain": "solana", "base_symbol": "HOLD1", "base_token_address": "Addr1", "final_action": "HOLD", "risk_level": "LOW", "entry_score": "11", "price_usd": "1"}
    analyzed = app.analyze_position_for_tg(row, None, {})
    tg_state = {"last_position_analysis_dedupe_keys": [analyzed["dedupe_key"]]}
    out = app.build_tg_position_analysis([row], [], {}, tg_state, now_ts=0)
    assert out["action_now"] == []
    assert out["watch"] == []
