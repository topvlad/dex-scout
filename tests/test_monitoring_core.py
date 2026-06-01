import importlib
import sys

import monitoring_core


def test_monitoring_core_import_safe_without_streamlit_or_app():
    sys.modules.pop("monitoring_core", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)
    mod = importlib.import_module("monitoring_core")
    assert mod.build_token_identity({})["token_key"] == ""
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_token_identity_aliases_and_pair_fallback():
    row = {"chain": "bsc", "base_token_address": "0xABC", "base_symbol": "AAA", "pairAddress": "0xPAIR"}
    ident = monitoring_core.build_token_identity(row)
    assert ident["token_addr"] == "0xabc"
    assert ident["pair_addr"] == "0xpair"
    assert ident["symbol"] == "AAA"
    assert ident["token_key"] == "bsc:0xabc"

    fallback = monitoring_core.build_token_identity({"chain": "bsc", "pair_address": "0xPAIR"})
    assert fallback["token_addr"] == ""
    assert fallback["token_key"] == "bsc:0xpair"


def test_missing_chain_or_address_returns_safe_empty_identity():
    assert monitoring_core.build_token_identity({"chain": "bsc"})["token_key"] == ""
    assert monitoring_core.build_token_identity({"base_addr": "0xabc"})["token_key"] == ""


def test_bsc_identity_is_case_insensitive():
    left = monitoring_core.build_token_identity({"chain": "bsc", "base_addr": "0xAbC"})
    right = monitoring_core.build_token_identity({"chain": "bsc", "base_addr": "0xabc"})
    assert left["token_key"] == right["token_key"]


def test_solana_identity_is_case_sensitive():
    left = monitoring_core.build_token_identity({"chain": "solana", "base_addr": "AbCd"})
    right = monitoring_core.build_token_identity({"chain": "solana", "base_addr": "abcd"})
    assert left["token_key"] != right["token_key"]


def test_active_monitoring_row_detection():
    assert monitoring_core.is_active_monitoring_row({"chain": "bsc", "base_addr": "0x1", "status": "WATCH", "liq_usd": 1000, "vol24_usd": 1000})
    assert not monitoring_core.is_active_monitoring_row({"chain": "bsc", "base_addr": "0x1", "active": "0", "status": "WATCH"})
    assert not monitoring_core.is_active_monitoring_row({"chain": "bsc", "base_addr": "0x1", "status": "ARCHIVED"})
    assert not monitoring_core.is_active_monitoring_row({"chain": "solana", "base_addr": "milk", "status": "UNTRADEABLE", "liq_usd": "0", "vol24_usd": "0"})


def test_active_portfolio_row_detection():
    assert monitoring_core.is_active_portfolio_row({"chain": "bsc", "base_token_address": "0x1", "active": "1"})
    assert not monitoring_core.is_active_portfolio_row({"chain": "bsc", "base_token_address": "0x1", "active": "false"})
    assert not monitoring_core.is_active_portfolio_row({"chain": "bsc", "base_token_address": "0x1", "status": "closed"})


def test_priority_surfacing_excludes_archived_dead_and_portfolio_exit_conflict():
    healthy = {"chain": "bsc", "base_addr": "0x1", "status": "WATCH", "liq_usd": 1000, "vol24_usd": 1000}
    assert monitoring_core.should_surface_in_priority(healthy)
    assert not monitoring_core.should_surface_in_priority({**healthy, "status": "ARCHIVED"})
    assert not monitoring_core.should_surface_in_priority({**healthy, "status": "UNTRADEABLE", "liq_usd": 0, "vol24_usd": 0})
    assert not monitoring_core.should_surface_in_priority(healthy, {"chain": "bsc", "base_token_address": "0x1", "final_action": "EXIT"})


def test_portfolio_reduce_overrides_monitoring_watch():
    result = monitoring_core.resolve_monitoring_portfolio_state(
        {"chain": "solana", "base_addr": "addr", "entry_status": "WATCH", "status": "ACTIVE"},
        {"chain": "solana", "base_token_address": "addr", "final_action": "REDUCE"},
    )
    assert result["source"] == "conflict_override"
    assert result["display_action"] == "REDUCE"
    assert result["is_conflict"] is True
    assert result["material_portfolio_action"] is True


def test_portfolio_take_profit_overrides_monitoring_early():
    result = monitoring_core.resolve_monitoring_portfolio_state(
        {"chain": "solana", "base_addr": "addr", "entry_status": "EARLY"},
        {"chain": "solana", "base_token_address": "addr", "final_action": "TAKE PROFIT"},
    )
    assert result["source"] == "conflict_override"
    assert result["display_action"] == "TAKE PROFIT"


def test_hold_does_not_override_as_material():
    result = monitoring_core.resolve_monitoring_portfolio_state(
        {"chain": "bsc", "base_addr": "0x1", "entry_status": "WATCH"},
        {"chain": "bsc", "base_token_address": "0x1", "final_action": "HOLD"},
    )
    assert result["source"] == "monitoring"
    assert result["display_action"] == "WATCH"
    assert result["material_portfolio_action"] is False


def test_missing_price_does_not_remove_material_exit():
    assert monitoring_core.is_material_portfolio_action({"final_action": "EXIT", "current_price": ""})
    result = monitoring_core.resolve_monitoring_portfolio_state({"entry_status": "WATCH"}, {"final_action": "EXIT", "current_price": ""})
    assert result["material_portfolio_action"] is True


def test_hard_gate_classifier_dead_healthy_and_portfolio_conflict():
    dead = monitoring_core.hard_gate_monitoring_row({"chain": "solana", "base_addr": "milk", "status": "UNTRADEABLE", "liq_usd": "0", "vol24_usd": "0"})
    assert dead["blocked"] is True
    assert dead["action"] == "ARCHIVE"
    healthy = monitoring_core.hard_gate_monitoring_row({"chain": "bsc", "base_addr": "0x1", "status": "WATCH", "liq_usd": 1000, "vol24_usd": 1000})
    assert healthy["blocked"] is False
    assert healthy["action"] == "KEEP"
    conflict = monitoring_core.hard_gate_monitoring_row(
        {"chain": "bsc", "base_addr": "0x1", "status": "WATCH", "liq_usd": 1000, "vol24_usd": 1000},
        {"chain": "bsc", "base_token_address": "0x1", "final_action": "EXIT"},
    )
    # Existing app semantics archive REDUCE/EXIT conflicts at hard-gate boundary.
    assert conflict["blocked"] is True
    assert conflict["action"] == "ARCHIVE"


def test_app_wrappers_preserve_core_behavior():
    import app

    row = {"chain": "bsc", "base_addr": "0xAbC", "status": "WATCH", "liq_usd": 1000, "vol24_usd": 1000}
    pf = {"chain": "bsc", "base_token_address": "0xabc", "final_action": "REDUCE"}
    assert app.build_token_identity(row) == monitoring_core.build_token_identity(row)
    assert app.row_matches_token(row, pf) == monitoring_core.row_matches_token(row, pf)
    assert app.normalize_material_portfolio_action("TAKE_PROFIT") == monitoring_core.normalize_material_portfolio_action("TAKE_PROFIT")
    assert app.hard_gate_monitoring_row(row, pf) == monitoring_core.hard_gate_monitoring_row(row, pf)
    assert app.should_surface_in_priority(row, pf) == monitoring_core.should_surface_in_priority(row, pf)
