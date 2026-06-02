import importlib
import sys


def test_portfolio_service_import_safe_without_streamlit_or_app():
    sys.modules.pop("portfolio_service", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)

    module = importlib.import_module("portfolio_service")

    assert module.__name__ == "portfolio_service"
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_action_normalization_variants():
    import portfolio_service as ps

    assert ps.normalize_portfolio_action("TAKE_PROFIT") == "TAKE PROFIT"
    assert ps.normalize_portfolio_action("take profit") == "TAKE PROFIT"
    assert ps.normalize_portfolio_action("REDUCE") == "REDUCE"
    assert ps.normalize_portfolio_action("partial-tp") == "PARTIAL_TP"
    assert ps.normalize_portfolio_action("PARTIAL_TAKE_PROFIT") == "PARTIAL_TP"
    assert ps.normalize_portfolio_action("HOLD") == "HOLD"
    assert ps.normalize_portfolio_action("") == ""


def test_material_action_detection():
    import portfolio_service as ps

    assert ps.is_material_portfolio_action({"final_action": "REDUCE", "current_price": ""}) is True
    assert ps.is_material_portfolio_action({"final_action": "TAKE PROFIT", "price_usd": "1"}) is True
    assert ps.is_material_portfolio_action("HOLD") is False
    assert ps.is_material_portfolio_action("WATCH") is False
    assert ps.is_material_portfolio_action("NO_ENTRY") is False


def test_classify_reduce_missing_price_remains_action_now():
    import portfolio_service as ps

    result = ps.classify_portfolio_row({"chain": "solana", "base_symbol": "AAA", "base_token_address": "addr", "final_action": "REDUCE"})

    assert result["action"] == "REDUCE"
    assert result["is_material"] is True
    assert result["urgency"] == "action_now"
    assert result["missing_price"] is True
    assert "numeric levels unavailable" in result["reason"]


def test_classify_hold_non_material():
    import portfolio_service as ps

    result = ps.classify_portfolio_row({"chain": "solana", "base_symbol": "AAA", "final_action": "HOLD", "price_usd": "1"})

    assert result["action"] == "HOLD"
    assert result["is_material"] is False
    assert result["urgency"] == "hold"


def test_conflict_resolver_portfolio_reduce_overrides_monitoring_watch():
    import portfolio_service as ps

    result = ps.resolve_portfolio_monitoring_conflict({"final_action": "REDUCE"}, {"entry_status": "WATCH"})

    assert result["has_conflict"] is True
    assert result["display_action"] == "REDUCE"
    assert result["source"] == "portfolio"
    assert result["material_portfolio_action"] is True


def test_conflict_resolver_take_profit_overrides_monitoring_early():
    import portfolio_service as ps

    result = ps.resolve_portfolio_monitoring_conflict({"final_action": "TAKE_PROFIT"}, {"entry_status": "EARLY"})

    assert result["has_conflict"] is True
    assert result["display_action"] == "TAKE PROFIT"
    assert result["source"] == "portfolio"


def test_conflict_resolver_exit_missing_price_overrides_monitoring_watch():
    import portfolio_service as ps

    result = ps.resolve_portfolio_monitoring_conflict({"final_action": "EXIT", "current_price": ""}, {"entry_status": "WATCH"})

    assert result["has_conflict"] is True
    assert result["display_action"] == "EXIT"
    assert result["source"] == "portfolio"


def test_conflict_resolver_hold_does_not_override_monitoring_watch():
    import portfolio_service as ps

    result = ps.resolve_portfolio_monitoring_conflict({"final_action": "HOLD"}, {"entry_status": "WATCH"})

    assert result["has_conflict"] is False
    assert result["display_action"] == "WATCH"
    assert result["source"] == "monitoring"
    assert result["material_portfolio_action"] is False


def test_conflict_resolver_no_portfolio_uses_monitoring_watch():
    import portfolio_service as ps

    result = ps.resolve_portfolio_monitoring_conflict(None, {"entry_status": "WATCH"})

    assert result["has_conflict"] is False
    assert result["display_action"] == "WATCH"
    assert result["source"] == "monitoring"


def test_notification_candidates_block_no_urgent_heartbeat_when_material_rows_exist():
    import portfolio_service as ps

    result = ps.build_portfolio_notification_candidates([{"chain": "solana", "base_token_address": "addr", "final_action": "REDUCE", "price_usd": "1"}])

    assert result["material_count"] == 1
    assert result["blocked_no_urgent_heartbeat"] is True
    assert result["blocked_reason"] == "material_portfolio_actions_present"
    assert result["diagnostics"]["non_material"] == 0


def test_material_missing_price_row_still_creates_notification_candidate():
    import portfolio_service as ps

    result = ps.build_portfolio_notification_candidates([{"chain": "solana", "base_token_address": "addr", "final_action": "EXIT", "current_price": ""}])

    assert result["material_count"] == 1
    assert len(result["candidates"]) == 1
    assert result["diagnostics"]["missing_price_material"] == 1
    assert result["candidates"][0]["classification"]["missing_price"] is True


def test_app_wrappers_preserve_portfolio_service_shapes():
    import app

    assert app.normalize_material_portfolio_action("TAKE_PROFIT") == "TAKE PROFIT"
    assert app.is_material_portfolio_action({"final_action": "REDUCE", "current_price": ""}) is True
    classified = app.classify_portfolio_row({"final_action": "REDUCE", "current_price": ""})
    assert {"symbol", "chain", "token_addr", "action", "is_material", "urgency", "reason", "missing_price", "display_label", "diagnostic_flags"}.issubset(classified)
    bridge = app.build_portfolio_notification_candidates([{"chain": "solana", "base_token_address": "addr", "final_action": "REDUCE"}])
    assert {"material_count", "candidates", "blocked_no_urgent_heartbeat", "blocked_reason", "diagnostics"}.issubset(bridge)
