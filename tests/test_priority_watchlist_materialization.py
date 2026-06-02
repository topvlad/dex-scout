import copy
import importlib
import sys
import types

import monitoring_core


def _row(symbol, status, score, addr=None, **extra):
    addr = addr or symbol.lower()
    return {
        "active": "1",
        "chain": "solana",
        "base_addr": addr,
        "base_symbol": symbol,
        "entry_status": status,
        "status": "ACTIVE",
        "priority_score": str(score),
        **extra,
    }


def test_priority_watchlist_includes_screenshot_watch_early_rows_and_excludes_no_entry():
    rows = [
        _row("OOO", "EARLY", 266.57, "ooo"),
        _row("FARTCOIN", "EARLY", 127.04, "fart"),
        _row("PUNCH", "WATCH", 73.39, "punch"),
        _row("8X5VQB...O2WN", "NO_ENTRY", 47, "8x5vqb-o2wn"),
        _row("SPX", "NO_ENTRY", 45, "spx"),
        _row("BUTTCOIN", "WATCH", 45.02, "butt"),
    ]
    portfolio_rows = [
        {"active": "1", "chain": "solana", "base_token_address": "ooo", "base_symbol": "OOO", "final_action": "HOLD"},
        {"active": "1", "chain": "solana", "base_token_address": "fart", "base_symbol": "FARTCOIN", "final_action": "HOLD"},
    ]

    priority_rows, debug = monitoring_core.build_priority_watchlist_rows(rows, portfolio_rows)
    symbols = {row["base_symbol"] for row in priority_rows}

    assert priority_rows
    assert {"OOO", "FARTCOIN", "PUNCH", "BUTTCOIN"}.issubset(symbols)
    assert "SPX" not in symbols
    assert "8X5VQB...O2WN" not in symbols
    assert debug["final_priority_rows"] > 0
    assert debug["excluded"]["no_entry"] == 2


def test_priority_watchlist_empty_debug_explains_no_entry_and_hard_gated_counts():
    rows = [
        _row("SPX", "NO_ENTRY", 45),
        _row("MILKERS", "WATCH", 88, "milk", health_label="UNTRADEABLE", liq_usd="0", vol24_usd="0"),
    ]

    priority_rows, debug = monitoring_core.build_priority_watchlist_rows(rows, [])

    assert priority_rows == []
    assert debug["final_priority_rows"] == 0
    assert debug["excluded"]["no_entry"] == 1
    assert debug["excluded"]["hard_gated"] == 1
    assert len(debug["top_excluded_samples"]) <= 3


def test_portfolio_exit_monitoring_watch_is_material_conflict_not_priority():
    rows = [_row("EXITME", "WATCH", 101, "exitme")]
    portfolio_rows = [
        {"active": "1", "chain": "solana", "base_token_address": "exitme", "base_symbol": "EXITME", "final_action": "EXIT"}
    ]

    priority_rows, debug = monitoring_core.build_priority_watchlist_rows(rows, portfolio_rows)

    assert priority_rows == []
    assert debug["excluded"]["portfolio_material_conflict"] == 1
    assert debug["final_priority_rows"] == 0


def test_priority_debug_construction_is_pure():
    rows = [_row("PUNCH", "WATCH", 73.39)]
    original = copy.deepcopy(rows)

    priority_rows, debug = monitoring_core.build_priority_watchlist_rows(rows, [])

    assert rows == original
    assert priority_rows[0] is not rows[0]
    assert debug["source_monitoring_rows"] == 1


def test_render_monitoring_page_passes_canonical_priority_context_without_writes(monkeypatch):
    class _StreamlitStub(types.ModuleType):
        def __init__(self):
            super().__init__("streamlit")
            self.errors = []
        def error(self, *args, **kwargs):
            self.errors.append((args, kwargs))

    st = _StreamlitStub()
    monkeypatch.setitem(sys.modules, "streamlit", st)
    sys.modules.pop("ui.pages_monitoring", None)
    monitoring_page = importlib.import_module("ui.pages_monitoring")

    write_calls = []
    save_calls = []
    context_calls = []

    def renderer(cfg, *, context):
        context_calls.append((cfg, context))

    monkeypatch.setattr(monitoring_page, "save_csv", lambda *a, **k: save_calls.append((a, k)), raising=False)
    monkeypatch.setattr(monitoring_page, "write_text", lambda *a, **k: write_calls.append((a, k)), raising=False)

    context = {
        "auto_cfg": {"x": 1},
        "priority_rows": [{"base_symbol": "LEGACY"}],
        "actions": {"render_monitoring": renderer},
    }

    monitoring_page.render_monitoring_page(context)

    assert context["priority_watchlist_rows"] == [{"base_symbol": "LEGACY"}]
    assert context_calls == [({"x": 1}, context)]
    assert save_calls == []
    assert write_calls == []
