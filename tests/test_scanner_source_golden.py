from __future__ import annotations

from pathlib import Path

import scanner_sources
from tests.golden_helpers import load_json_fixture, stable_candidate_id

FIXTURES = Path(__file__).parent / "fixtures" / "scanner_golden"


def test_scanner_golden_fixtures_load_as_valid_json():
    for path in sorted(FIXTURES.glob("*.json")):
        assert load_json_fixture(path), f"{path.name} should not be empty"


def test_dexscreener_source_adapter_fixture_seeds_are_deterministic_and_capped():
    raw = load_json_fixture(FIXTURES / "raw_dex_pairs_mixed.json")

    def fetch(seed, timeout=None):
        assert timeout == 7
        if seed == "alpha":
            return [raw[0], raw[1], raw[2]]
        if seed == "beta":
            return [raw[3], raw[4], raw[5]]
        return []

    result = scanner_sources.fetch_dexscreener_search_source(
        seeds=["alpha", "beta", "empty"],
        fetch_pairs_by_query_fn=fetch,
        max_items=3,
        timeout_sec=7,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["raw_count"] == 3
    assert result["seeds_used"] == ["alpha", "beta", "empty"]
    assert result["meta"]["duplicates"] == 1
    assert result["meta"]["capped"] is True


def test_collect_scanner_sources_golden_status_contracts():
    raw = load_json_fixture(FIXTURES / "raw_dex_pairs_mixed.json")
    success = lambda: scanner_sources.make_source_result(source="success", ok=True, status="success", raw_candidates=raw[:2])
    empty = lambda: scanner_sources.make_source_result(source="empty", ok=True, status="empty")
    failed = lambda: scanner_sources.make_source_result(source="failed", ok=False, status="failed", error="RuntimeError: boom")
    disabled = lambda: scanner_sources.make_source_result(source="disabled", ok=True, status="disabled")

    assert scanner_sources.collect_scanner_sources(source_fns=[success, empty])["status"] == "success"
    assert scanner_sources.collect_scanner_sources(source_fns=[success, failed])["status"] == "partial"
    assert scanner_sources.collect_scanner_sources(source_fns=[failed, failed])["status"] == "failed"
    assert scanner_sources.collect_scanner_sources(source_fns=[empty, empty])["status"] == "empty"
    disabled_result = scanner_sources.collect_scanner_sources(source_fns=[disabled, disabled])
    assert disabled_result["status"] == "empty"
    assert disabled_result["diagnostics"]["sources_disabled"] == 2


def test_source_aggregation_matches_golden_subset():
    raw = load_json_fixture(FIXTURES / "raw_dex_pairs_mixed.json")
    expected = load_json_fixture(FIXTURES / "source_aggregation_expected.json")
    result = scanner_sources.collect_scanner_sources(
        source_fns=[
            lambda: scanner_sources.make_source_result(source="dexscreener_search", ok=True, status="success", raw_candidates=raw[:2], seeds_used=["alpha"]),
            lambda: scanner_sources.make_source_result(source="empty_source", ok=True, status="empty"),
        ],
        max_total=10,
    )
    assert result["status"] == expected["status"]
    assert result["ok"] == expected["ok"]
    assert result["sources_tried"] == expected["sources_tried"]
    assert result["diagnostics"] == expected["diagnostics"]


def test_source_diagnostics_are_sanitized_bounded_and_secret_free():
    result = scanner_sources.fetch_dexscreener_search_source(
        seeds=["alpha"],
        fetch_pairs_by_query_fn=lambda seed, timeout=None: (_ for _ in ()).throw(RuntimeError("api_key=abc123 bearer zzz999")),
    )
    error_text = " ".join(result.get("meta", {}).get("errors_by_seed", {}).values()) + " " + result.get("error", "")
    assert "zzz999" not in error_text
    assert "abc123" not in error_text
    assert "<redacted>" in error_text
    assert len(error_text) < 260


def test_solana_identity_preserves_mixed_case_and_dedupe_key_is_stable():
    raw = load_json_fixture(FIXTURES / "raw_dex_pairs_mixed.json")
    sol = raw[0]
    result = scanner_sources.fetch_dexscreener_search_source(
        seeds=["alpha"],
        fetch_pairs_by_query_fn=lambda seed, timeout=None: [sol, dict(sol)],
        max_items=10,
    )
    assert result["raw_count"] == 1
    assert result["raw_candidates"][0]["baseToken"]["address"] == "SoLmiXeDCase111111111111111111111111111111"
    assert result["meta"]["duplicates"] == 1
    assert stable_candidate_id({"chain": "solana", "base_addr": "SoLmiXeDCase111111111111111111111111111111"}) == "solana|ca:SoLmiXeDCase111111111111111111111111111111"
