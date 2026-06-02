import copy
import importlib
import sys

import monitoring_service


def _row(symbol="AAA", status="WATCH", score="100", **extra):
    return {
        "active": "1",
        "chain": "solana",
        "base_addr": symbol.lower(),
        "base_symbol": symbol,
        "entry_status": status,
        "priority_score": str(score),
        **extra,
    }


def test_monitoring_service_import_safe_without_streamlit_or_app():
    sys.modules.pop("monitoring_service", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)
    mod = importlib.import_module("monitoring_service")
    assert mod.normalize_monitoring_status("watch") == "WATCH"
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_status_normalization_variants():
    assert monitoring_service.normalize_monitoring_status("watch") == "WATCH"
    assert monitoring_service.normalize_monitoring_status("early") == "EARLY"
    assert monitoring_service.normalize_monitoring_status("NO ENTRY") == "NO_ENTRY"
    assert monitoring_service.normalize_monitoring_status("no-entry") == "NO_ENTRY"
    assert monitoring_service.normalize_monitoring_status({"status": "archived"}) == "ARCHIVED"
    assert monitoring_service.normalize_monitoring_status({}) == "UNKNOWN"


def test_classify_watch_and_early_rows_active_actionable():
    watch = monitoring_service.classify_monitoring_row(_row("WATCHME", "WATCH"))
    early = monitoring_service.classify_monitoring_row(_row("EARLYME", "EARLY"))
    assert watch["is_active"] is True
    assert watch["is_actionable"] is True
    assert watch["priority_eligible"] is True
    assert early["is_active"] is True
    assert early["is_actionable"] is True
    assert early["priority_eligible"] is True


def test_classify_no_entry_not_actionable_or_priority():
    result = monitoring_service.classify_monitoring_row(_row("SPX", "NO_ENTRY", 45))
    assert result["is_no_entry"] is True
    assert result["is_actionable"] is False
    assert result["priority_eligible"] is False
    assert result["archive_action"] == "NO_ENTRY"


def test_classify_archived_row_not_priority_eligible():
    result = monitoring_service.classify_monitoring_row(_row("OLD", "WATCH", 77, active="0", archived_reason="manual"))
    assert result["is_archived"] is True
    assert result["priority_eligible"] is False
    assert result["archive_action"] == "ARCHIVE"


def test_classify_hard_gated_row_not_priority_eligible():
    result = monitoring_service.classify_monitoring_row(_row("MILK", "WATCH", 88, health_label="UNTRADEABLE", liq_usd="0", vol24_usd="0"))
    assert result["hard_gated"] is True
    assert result["priority_eligible"] is False


def test_portfolio_reduce_watch_excludes_from_priority():
    result = monitoring_service.classify_monitoring_row(
        _row("REDUCE", "WATCH", 101, base_addr="reduce"),
        {"active": "1", "chain": "solana", "base_token_address": "reduce", "base_symbol": "REDUCE", "final_action": "REDUCE"},
    )
    assert result["portfolio_conflict"] is True
    assert result["priority_eligible"] is False


def test_portfolio_take_profit_early_excludes_from_priority():
    result = monitoring_service.classify_monitoring_row(
        _row("TP", "EARLY", 101, base_addr="tp"),
        {"active": "1", "chain": "solana", "base_token_address": "tp", "base_symbol": "TP", "final_action": "TAKE PROFIT"},
    )
    assert result["portfolio_conflict"] is True
    assert result["priority_eligible"] is False


def test_archive_planner_returns_priority_rows_for_active_watch_early():
    plan = monitoring_service.plan_monitoring_archive_transitions([
        _row("OOO", "EARLY", 266.57),
        _row("PUNCH", "WATCH", 73.39),
    ])
    assert [row["base_symbol"] for row in plan["priority_rows"]] == ["OOO", "PUNCH"]
    assert plan["diagnostics"]["priority_eligible"] == 2


def test_archive_planner_excludes_no_entry_rows():
    plan = monitoring_service.plan_monitoring_archive_transitions([
        _row("PUNCH", "WATCH", 73.39),
        _row("SPX", "NO_ENTRY", 45),
    ])
    symbols = {row["base_symbol"] for row in plan["priority_rows"]}
    assert "PUNCH" in symbols
    assert "SPX" not in symbols
    assert len(plan["no_entry_rows"]) == 1


def test_archive_planner_diagnostics_counts_correct():
    plan = monitoring_service.plan_monitoring_archive_transitions([
        _row("A", "WATCH", 1),
        _row("B", "EARLY", 2),
        _row("C", "NO_ENTRY", 3),
        _row("D", "WATCH", 4, health_label="UNTRADEABLE", liq_usd="0", vol24_usd="0"),
    ])
    assert plan["rows_seen"] == 4
    assert plan["diagnostics"]["active"] == 3
    assert plan["diagnostics"]["watch_early"] == 3
    assert plan["diagnostics"]["no_entry"] == 1
    assert plan["diagnostics"]["hard_gated"] == 2  # NO_ENTRY and UNTRADEABLE are both hard-gated by core logic.
    assert plan["diagnostics"]["priority_eligible"] == 2
    assert plan["diagnostics"]["archive_candidates"] >= 1


def test_planner_does_not_mutate_input_rows():
    rows = [_row("PUNCH", "WATCH", 73.39)]
    original = copy.deepcopy(rows)
    monitoring_service.plan_monitoring_archive_transitions(rows)
    assert rows == original


def test_app_py_wrappers_preserve_service_behavior():
    import app

    row = _row("PUNCH", "WATCH", 73.39)
    assert app.normalize_monitoring_status(row) == monitoring_service.normalize_monitoring_status(row)
    assert app.classify_monitoring_row(row) == monitoring_service.classify_monitoring_row(row)
    app_plan = app.plan_monitoring_archive_transitions([row])
    svc_plan = monitoring_service.plan_monitoring_archive_transitions([row])
    assert app_plan["diagnostics"] == svc_plan["diagnostics"]
    priority_rows, debug = app.build_priority_watchlist_rows([row], [])
    assert priority_rows[0]["base_symbol"] == "PUNCH"
    assert debug["monitoring_archive_diagnostics"]["priority_eligible"] == 1


def test_screenshot_regression_priority_rows_and_diagnostics():
    rows = [
        _row("OOO", "EARLY", 266.57, base_addr="ooo"),
        _row("FARTCOIN", "EARLY", 127.04, base_addr="fart"),
        _row("PUNCH", "WATCH", 73.39, base_addr="punch"),
        _row("8X5VQB...O2WN", "NO_ENTRY", 47, base_addr="8x5vqb-o2wn"),
        _row("SPX", "NO_ENTRY", 45, base_addr="spx"),
        _row("BUTTCOIN", "WATCH", 45.02, base_addr="butt"),
    ]
    plan = monitoring_service.plan_monitoring_archive_transitions(rows)
    symbols = {row["base_symbol"] for row in plan["priority_rows"]}
    assert plan["priority_rows"]
    assert "SPX" not in symbols
    assert "8X5VQB...O2WN" not in symbols
    assert plan["diagnostics"]["watch_early"] >= 4
    assert plan["diagnostics"]["priority_eligible"] >= 1


def test_runtime_no_fail_matrix_includes_monitoring_service():
    import scripts.runtime_no_fail_matrix as matrix

    assert "monitoring_service" in matrix.CORE_MODULES
