import importlib
import inspect
import sys
from copy import deepcopy

import app
import scanner_service


def _candidate(addr="a", **extra):
    row = {"chain": "solana", "base_addr": addr, "base_symbol": addr.upper(), "entry_status": "WATCH", "score": "10"}
    row.update(extra)
    return row


def test_scanner_service_imports_without_streamlit_or_app(monkeypatch):
    snapshot = {name: sys.modules.get(name) for name in ("scanner_service", "streamlit", "app")}
    sys.modules.pop("scanner_service", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)
    try:
        importlib.import_module("scanner_service")
        assert "streamlit" not in sys.modules
        assert "app" not in sys.modules
    finally:
        for name in ("scanner_service", "streamlit", "app"):
            sys.modules.pop(name, None)
            if snapshot.get(name) is not None:
                sys.modules[name] = snapshot[name]


def test_build_live_pulse_payload_counters_are_correct():
    payload = scanner_service.build_live_pulse_payload(
        raw_candidates=[_candidate("a"), _candidate("b"), _candidate("c")],
        normalized_candidates=[_candidate("a"), _candidate("b")],
        final_candidates=[_candidate("a")],
        status="success",
        source="test",
        now_ts="now",
    )
    assert payload["raw_seen"] == 3
    assert payload["normalized"] == 2
    assert payload["final_count"] == 1
    assert payload["status"] == "success"


def test_build_live_pulse_payload_empty_has_reason():
    payload = scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], status="empty", now_ts="now")
    assert payload["final_count"] == 0
    assert payload["last_empty_reason"] == "source_api_empty"


def test_failed_payload_sets_worker_or_scanner_reason():
    worker = scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], status="failed", error="boom", now_ts="now")
    scanner = scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], status="failed", error="scanner boom", now_ts="now")
    assert worker["last_empty_reason"] == "worker_failed"
    assert scanner["last_empty_reason"] == "scanner_failed"


def test_raw_present_all_rejected_records_blocked_reasons():
    payload = scanner_service.build_live_pulse_payload(
        raw_candidates=[_candidate("a")],
        normalized_candidates=[_candidate("a")],
        final_candidates=[],
        rejected_candidates=[{"chain": "solana", "base_addr": "a", "reason": "hard_gated"}],
        status="empty",
        now_ts="now",
    )
    assert payload["raw_seen"] == 1
    assert payload["blocked_reasons"]["hard_gated"] == 1
    assert payload["last_empty_reason"] == "all_filtered_hard_gate"


def test_normalize_batch_catches_per_row_exception():
    def normalizer(row):
        if row.get("boom"):
            raise ValueError("bad")
        return row

    result = scanner_service.normalize_scanner_candidates([_candidate("a"), {"boom": True}], normalize_pair_fn=normalizer)
    assert result["diagnostics"]["normalized"] == 1
    assert result["diagnostics"]["reasons"]["normalize_exception"] == 1


def test_filtering_excludes_no_entry_dead_and_hard_gated():
    rows = [_candidate("a", entry_status="NO_ENTRY"), _candidate("b", status="DEAD"), _candidate("c", hard=True)]
    result = scanner_service.filter_live_pulse_candidates(rows, hard_gate_fn=lambda row: {"blocked": bool(row.get("hard"))}, max_candidates=5)
    assert result["final_candidates"] == []
    assert result["diagnostics"]["hard_gated"] >= 2
    assert result["diagnostics"]["scam_or_toxic"] >= 1


def test_filtering_preserves_active_safe_candidate():
    result = scanner_service.filter_live_pulse_candidates([_candidate("safe")], hard_gate_fn=lambda row: {"blocked": False})
    assert [row["base_addr"] for row in result["final_candidates"]] == ["safe"]


def test_filtering_records_duplicate_archive_and_scam_reasons():
    result = scanner_service.filter_live_pulse_candidates(
        [_candidate("dup"), _candidate("dup"), _candidate("arch"), _candidate("scam", risk_flags="scam")],
        archive_rows=[_candidate("arch")],
        hard_gate_fn=lambda row: {"blocked": False},
    )
    assert result["blocked_reasons"]["duplicate"] == 1
    assert result["blocked_reasons"]["archived"] == 1
    assert result["blocked_reasons"]["scam_or_toxic"] == 1


def test_refill_planner_below_target_returns_true():
    result = scanner_service.plan_live_pulse_refill(current_count=1, target_min=3, target_max=5, max_attempts=2, sources_tried=[])
    assert result["should_refill"] is True
    assert result["reason"] == "below_target"


def test_refill_planner_enough_candidates_returns_false():
    result = scanner_service.plan_live_pulse_refill(current_count=3, target_min=3, target_max=5, max_attempts=2, sources_tried=[])
    assert result["should_refill"] is False
    assert result["reason"] == "enough_candidates"


def test_run_cycle_success_saves_payload():
    saved = []
    result = scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=lambda config: {"raw_candidates": [_candidate("a")], "sources_tried": ["fake"]},
        normalize_fn=lambda raw_candidates: {"normalized_candidates": list(raw_candidates), "rejected_candidates": []},
        filter_fn=lambda **kwargs: {"final_candidates": list(kwargs["candidates"]), "rejected_candidates": []},
        save_payload_fn=lambda payload: saved.append(payload),
        monitoring_rows=[],
        portfolio_rows=[],
        now_ts="now",
    )
    assert result["ok"] is True
    assert saved[-1]["final_count"] == 1


def test_run_cycle_empty_saves_empty_payload_with_reason():
    saved = []
    result = scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=lambda config: {"raw_candidates": [_candidate("a")]},
        normalize_fn=lambda raw_candidates: {"normalized_candidates": list(raw_candidates), "rejected_candidates": []},
        filter_fn=lambda **kwargs: {"final_candidates": [], "rejected_candidates": [{"reason": "hard_gated"}]},
        save_payload_fn=lambda payload: saved.append(payload),
        monitoring_rows=[],
        portfolio_rows=[],
        now_ts="now",
    )
    assert result["status"] == "empty"
    assert saved[-1]["last_empty_reason"] == "all_filtered_hard_gate"


def test_run_cycle_fetch_exception_saves_failed_payload():
    saved = []
    def fetch(config):
        raise RuntimeError("boom")
    result = scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=fetch,
        normalize_fn=lambda raw_candidates: {},
        filter_fn=lambda **kwargs: {},
        save_payload_fn=lambda payload: saved.append(payload),
        monitoring_rows=[],
        portfolio_rows=[],
        now_ts="now",
    )
    assert result["status"] == "failed"
    assert saved[-1]["last_empty_reason"] == "worker_failed"


def test_run_cycle_save_failure_returns_not_ok():
    def save(payload):
        raise RuntimeError("save boom")
    result = scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=lambda config: [_candidate("a")],
        normalize_fn=lambda raw_candidates: {"normalized_candidates": raw_candidates},
        filter_fn=lambda **kwargs: {"final_candidates": kwargs["candidates"]},
        save_payload_fn=save,
        monitoring_rows=[],
        portfolio_rows=[],
        now_ts="now",
    )
    assert result["ok"] is False
    assert result["last_empty_reason"] == "save_failed"


def test_cycle_does_not_mutate_input_rows():
    monitoring = [_candidate("m")]
    portfolio = [_candidate("p")]
    before = (deepcopy(monitoring), deepcopy(portfolio))
    scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=lambda config: [_candidate("a")],
        normalize_fn=lambda raw_candidates: {"normalized_candidates": raw_candidates},
        filter_fn=lambda **kwargs: {"final_candidates": kwargs["candidates"]},
        save_payload_fn=lambda payload: None,
        monitoring_rows=monitoring,
        portfolio_rows=portfolio,
        now_ts="now",
    )
    assert (monitoring, portfolio) == before


def test_runtime_no_fail_matrix_includes_scanner_service():
    import scripts.runtime_no_fail_matrix as matrix
    assert "scanner_service" in matrix.CORE_MODULES


def test_app_wrappers_preserve_scan_cycle_runner_signatures():
    maybe = inspect.signature(app.maybe_run_rotating_scanner)
    priority = inspect.signature(app.run_priority_scanner_cycle)
    assert "seeds_raw" in maybe.parameters
    assert "max_items" in maybe.parameters
    assert callable(app.maybe_run_rotating_scanner)
    assert callable(app.run_priority_scanner_cycle)
    assert len(priority.parameters) >= 0


def test_ui_context_contains_live_pulse_and_scanner_diagnostics_keys():
    context = app.build_ui_context(selected_page="Scout", actions={"render_scout": lambda *a, **k: None})
    assert "live_pulse_payload" in context
    assert "live_pulse_debug" in context
    assert "scanner_diagnostics" in context
    assert "last_empty_reason" in context["scanner_diagnostics"]


def test_priority_screenshot_regression_watch_rows_remain_non_empty():
    rows, debug = app.monitoring_core.build_priority_watchlist_rows([_candidate("watch", entry_status="WATCH", priority_score="42")], [])
    assert rows
    assert debug["final_priority_rows"] == 1
