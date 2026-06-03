from __future__ import annotations

from pathlib import Path

import app
import monitoring_service
import scanner_service
import scanner_sources
import scoring
from tests.golden_helpers import assert_subset_matches, load_json_fixture, normalize_dynamic_fields, stable_candidate_id

FIXTURES = Path(__file__).parent / "fixtures" / "scanner_golden"
NOW = "2026-01-01T00:00:00Z"


def _raw():
    return load_json_fixture(FIXTURES / "raw_dex_pairs_mixed.json")


def _conflicts():
    return load_json_fixture(FIXTURES / "monitoring_portfolio_conflicts.json")


def _normalized():
    return scanner_service.normalize_scanner_candidates(_raw())


def _filter(normalized_rows):
    conflicts = _conflicts()
    return scanner_service.filter_live_pulse_candidates(
        normalized_rows,
        monitoring_rows=[],
        portfolio_rows=[conflicts["portfolio_rows"][0]],
        archive_rows=[{"chain": "solana", "base_addr": "ArchiveSoL111111111111111111111111111111", "base_symbol": "ARCH"}],
        hard_gate_fn=lambda row: {"blocked": False},
        max_candidates=10,
    )


def test_normalize_scanner_candidates_matches_golden_fixture():
    result = _normalized()
    expected_norm = load_json_fixture(FIXTURES / "normalized_candidates_expected.json")
    expected_rejected = load_json_fixture(FIXTURES / "rejected_candidates_expected.json")

    assert result["diagnostics"] == expected_norm["diagnostics"]
    actual_subset = [
        {"chain": r["chain"], "base_addr": r["base_addr"], "base_symbol": r.get("base_symbol"), "score": r.get("score"), "entry_status": r.get("entry_status")}
        for r in result["normalized_candidates"]
    ]
    assert actual_subset == expected_norm["candidates"]
    assert result["diagnostics"]["raw_seen"] == 8
    assert result["rejected_candidates"] == expected_rejected["rejected"]
    assert result["diagnostics"]["reasons"] == {"duplicate": 1, "missing_chain": 1}
    assert result["normalized_candidates"][0]["base_addr"] == "SoLmiXeDCase111111111111111111111111111111"


def test_filter_live_pulse_candidates_matches_golden_outcomes():
    normalized = _normalized()["normalized_candidates"]
    result = _filter(normalized)
    final_ids = [stable_candidate_id(row) for row in result["final_candidates"]]

    assert final_ids == [
        "solana|ca:SoLmiXeDCase111111111111111111111111111111",
        "bsc|ca:0xabcdef0000000000000000000000000000000001",
    ]
    assert result["blocked_reasons"] == {
        "hard_gated": 1,
        "scam_or_toxic": 1,
        "archived": 1,
        "portfolio_active": 1,
    }
    assert result["diagnostics"]["portfolio_active"] == 1
    assert result["diagnostics"]["archived"] == 1
    assert result["diagnostics"]["hard_gated"] == 1
    assert result["diagnostics"]["scam_or_toxic"] == 1
    assert all(row.get("base_symbol") not in {"NOPE", "DEADSCAM", "ARCH", "PORT"} for row in result["final_candidates"])


def test_build_live_pulse_payload_success_matches_golden_subset():
    norm = _normalized()
    filt = _filter(norm["normalized_candidates"])
    payload = scanner_service.build_live_pulse_payload(
        raw_candidates=_raw(),
        normalized_candidates=norm["normalized_candidates"],
        final_candidates=filt["final_candidates"],
        rejected_candidates=norm["rejected_candidates"] + filt["rejected_candidates"],
        status="success",
        source="test",
        sources_tried=["dexscreener_search"],
        now_ts=NOW,
    )
    expected = load_json_fixture(FIXTURES / "live_pulse_success_expected.json")
    assert_subset_matches(payload, expected)
    assert payload["raw_seen"] == 8
    assert payload["normalized"] == 6
    assert payload["clean_candidates"] == 2
    assert payload["final_count"] == 2
    assert payload["last_empty_reason"] == ""


def test_build_live_pulse_payload_empty_reason_golden_cases():
    expected = load_json_fixture(FIXTURES / "live_pulse_empty_expected.json")
    cases = {
        "source_api_empty": scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], rejected_candidates=[], status="empty", source_state={"diagnostics": {"sources_total": 1, "sources_empty": 1, "sources_failed": 0, "sources_disabled": 0}}, now_ts=NOW),
        "source_api_failed": scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], rejected_candidates=[], status="empty", source_state={"diagnostics": {"sources_total": 1, "sources_empty": 0, "sources_failed": 1, "sources_disabled": 0}}, now_ts=NOW),
        "scanner_returned_no_candidates": scanner_service.build_live_pulse_payload(raw_candidates=[_raw()[0]], normalized_candidates=[], final_candidates=[], rejected_candidates=[], status="empty", now_ts=NOW),
        "all_filtered_archive_or_duplicate": scanner_service.build_live_pulse_payload(raw_candidates=[_raw()[0]], normalized_candidates=[_normalized()["normalized_candidates"][0]], final_candidates=[], rejected_candidates=[{"chain": "solana", "base_addr": "SoLmiXeDCase111111111111111111111111111111", "reason": "duplicate"}], status="empty", now_ts=NOW),
        "all_filtered_hard_gate": scanner_service.build_live_pulse_payload(raw_candidates=[_raw()[3]], normalized_candidates=[_normalized()["normalized_candidates"][2]], final_candidates=[], rejected_candidates=[{"chain": "solana", "base_addr": "NoEntrySoL11111111111111111111111111111111", "reason": "hard_gated"}], status="empty", now_ts=NOW),
    }
    for name, payload in cases.items():
        assert payload["final_count"] == 0
        assert payload["last_empty_reason"]
        assert {"status": payload["status"], "final_count": payload["final_count"], "last_empty_reason": payload["last_empty_reason"]} == expected[name]


def test_run_scanner_service_cycle_uses_fake_boundaries_and_saves_golden_payload():
    saved = []
    raw = _raw()
    source_state = scanner_sources.collect_scanner_sources(
        source_fns=[lambda: scanner_sources.make_source_result(source="dexscreener_search", ok=True, status="success", raw_candidates=raw, seeds_used=["golden"])],
        max_total=20,
    )

    result = scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=lambda config: source_state,
        normalize_fn=lambda raw_candidates: scanner_service.normalize_scanner_candidates(raw_candidates),
        filter_fn=lambda candidates, monitoring_rows, portfolio_rows, archive_rows: _filter(candidates),
        save_payload_fn=saved.append,
        monitoring_rows=[],
        portfolio_rows=[],
        archive_rows=[],
        config={"source": "test"},
        now_ts=NOW,
    )

    assert result["ok"] is True
    assert len(saved) == 1
    assert result["payload"]["final_count"] == 2
    assert result["payload"]["debug"]["source_diagnostics"]["status"] == "success"
    assert normalize_dynamic_fields(saved[0]) == normalize_dynamic_fields(result["payload"])


def test_priority_screenshot_regression_and_no_entry_diagnostics():
    expected = load_json_fixture(FIXTURES / "priority_watchlist_expected.json")
    rows = [
        {"chain": "solana", "base_addr": "OOO111111111111111111111111111111111111", "base_symbol": "OOO", "entry_status": "EARLY", "priority_score": 266.57},
        {"chain": "solana", "base_addr": "FART11111111111111111111111111111111111", "base_symbol": "FARTCOIN", "entry_status": "EARLY", "priority_score": 127.04},
        {"chain": "solana", "base_addr": "PUNCH1111111111111111111111111111111111", "base_symbol": "PUNCH", "entry_status": "WATCH", "priority_score": 73.39},
        {"chain": "solana", "base_addr": "8X5VQB1111111111111111111111111111O2WN", "base_symbol": "8X5VQB...O2WN", "entry_status": "NO_ENTRY", "priority_score": 47},
        {"chain": "bsc", "base_addr": "0xspx000000000000000000000000000000000000", "base_symbol": "SPX", "entry_status": "NO_ENTRY", "priority_score": 45},
        {"chain": "solana", "base_addr": "BUTT11111111111111111111111111111111111", "base_symbol": "BUTTCOIN", "entry_status": "WATCH", "priority_score": 45.02},
    ]
    priority_rows, debug = monitoring_service.build_priority_watchlist_rows(rows, [])
    assert priority_rows
    assert [r["base_symbol"] for r in priority_rows] == [row["base_symbol"] for row in expected["rows"]]
    assert {"8X5VQB...O2WN", "SPX"}.isdisjoint({r.get("base_symbol") for r in priority_rows})
    assert debug["excluded"]["no_entry"] == 2
    assert debug["final_priority_rows"] > 0


def test_portfolio_material_conflict_regression():
    fixture = _conflicts()
    priority_rows, debug = monitoring_service.build_priority_watchlist_rows(fixture["monitoring_rows"][1:], fixture["portfolio_rows"][1:])
    symbols = {row.get("base_symbol") for row in priority_rows}
    assert "REDWATCH" not in symbols
    assert "TPEARLY" not in symbols
    assert "HOLDWATCH" in symbols
    assert debug["excluded"]["portfolio_material_conflict"] == 2
    assert debug["final_priority_rows"] == 1


def test_score_filter_parity_guard_for_representative_rows():
    high_quality = {"liqUsd": 20000, "txns_m5": 40, "vol_m5": 50000, "pc_m5": 0, "pc_h1": 20, "buys_m5": 35, "sells_m5": 5}
    scam_like = {"liqUsd": 20000, "txns_m5": 12, "vol_m5": 5000, "pc_m5": 99, "pc_h1": 20, "buys_m5": 9, "sells_m5": 4}
    low_liq = {"liqUsd": 100, "txns_m5": 1, "vol_m5": 20, "pc_m5": 2, "pc_h1": 3, "buys_m5": 1, "sells_m5": 5}

    assert scoring.passes_safe_filters(high_quality) is True
    assert scoring.score_row(high_quality) >= 2.0
    assert scoring.score_row(high_quality) >= 0
    assert scoring.passes_safe_filters(scam_like) is False
    assert scoring.score_row(scam_like) >= 0
    assert scoring.passes_safe_filters(low_liq) is False
    assert scoring.score_row(low_liq) >= 0


def test_app_build_ui_context_exposes_scanner_diagnostics_without_scanner_execution(monkeypatch):
    monkeypatch.setattr(scanner_service, "run_scanner_service_cycle", lambda **kwargs: (_ for _ in ()).throw(AssertionError("scanner should not execute")))
    old_shape_payload = {"final_candidates": [{"base_symbol": "OLD"}], "status": "success"}
    context = app.build_ui_context(runtime_state={"live_pulse_payload": old_shape_payload})

    assert context["live_pulse_payload"] == old_shape_payload
    assert context["live_pulse_debug"] == {}
    assert context["scanner_diagnostics"]["last_scan_status"] == "success"
    assert context["scanner_diagnostics"]["raw_seen"] == 0
    assert "priority_watchlist_rows" in context
    assert "priority_watchlist_debug" in context


def test_app_live_pulse_wrapper_delegates_to_scanner_service_when_present():
    if not hasattr(app, "build_live_pulse_payload"):
        return
    kwargs = {"raw_candidates": [], "normalized_candidates": [], "final_candidates": [], "status": "empty", "source": "test", "now_ts": NOW}
    assert app.build_live_pulse_payload(**kwargs) == scanner_service.build_live_pulse_payload(**kwargs)
