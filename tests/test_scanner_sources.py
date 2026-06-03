import inspect
import sys

import pytest

import scanner_service
import scanner_sources


def _pair(addr, *, pair=None, chain="solana", symbol="T"):
    return {"chainId": chain, "pairAddress": pair or f"pair-{addr}", "baseToken": {"address": addr, "symbol": symbol}}


def test_scanner_sources_imports_without_streamlit_or_app():
    sys.modules.pop("scanner_sources", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)
    __import__("scanner_sources")
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_make_source_result_raw_count_matches_candidates():
    result = scanner_sources.make_source_result(source="manual", ok=True, status="success", raw_candidates=[_pair("A")])
    assert result["raw_count"] == len(result["raw_candidates"]) == 1


def test_make_source_result_sanitizes_and_limits_error():
    result = scanner_sources.make_source_result(source="manual", ok=False, status="failed", error="api_key=SECRET " + ("x" * 500))
    assert "SECRET" not in result["error"]
    assert len(result["error"]) <= 240


def test_dexscreener_adapter_empty_seeds_returns_empty():
    result = scanner_sources.fetch_dexscreener_search_source(seeds=[], fetch_pairs_by_query_fn=lambda q: [_pair("A")])
    assert result["ok"] is True
    assert result["status"] == "empty"
    assert result["raw_count"] == 0


def test_dexscreener_adapter_catches_per_seed_exception():
    def fetch(seed, timeout=12):
        if seed == "bad":
            raise RuntimeError("boom")
        return [_pair(seed)]
    result = scanner_sources.fetch_dexscreener_search_source(seeds=["bad", "A"], fetch_pairs_by_query_fn=fetch)
    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["meta"]["seeds_failed"] == 1
    assert "bad" in result["meta"]["errors_by_seed"]


def test_dexscreener_adapter_deduplicates_candidates():
    result = scanner_sources.fetch_dexscreener_search_source(seeds=["one"], fetch_pairs_by_query_fn=lambda seed, timeout=12: [_pair("A"), _pair("A")], max_items=10)
    assert result["raw_count"] == 1
    assert result["meta"]["duplicates"] == 1


def test_dexscreener_adapter_caps_max_items():
    result = scanner_sources.fetch_dexscreener_search_source(seeds=["one"], fetch_pairs_by_query_fn=lambda seed, timeout=12: [_pair("A"), _pair("B")], max_items=1)
    assert result["raw_count"] == 1
    assert result["meta"]["capped"] is True


def test_birdeye_disabled_returns_disabled():
    result = scanner_sources.fetch_birdeye_trending_source(enabled=False)
    assert result["ok"] is True
    assert result["status"] == "disabled"


def test_birdeye_fetch_fn_empty_returns_empty():
    result = scanner_sources.fetch_birdeye_trending_source(enabled=True, fetch_fn=lambda limit=20: [])
    assert result["ok"] is True
    assert result["status"] == "empty"


def test_birdeye_fetch_fn_exception_returns_failed():
    def fetch(limit=20):
        raise RuntimeError("boom")
    result = scanner_sources.fetch_birdeye_trending_source(enabled=True, fetch_fn=fetch)
    assert result["ok"] is False
    assert result["status"] == "failed"


def test_collect_one_success_returns_success():
    result = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="s", ok=True, status="success", raw_candidates=[_pair("A")])])
    assert result["status"] == "success"
    assert result["ok"] is True


def test_collect_success_failed_returns_partial():
    result = scanner_sources.collect_scanner_sources(source_fns=[
        lambda: scanner_sources.make_source_result(source="bad", ok=False, status="failed", error="boom"),
        lambda: scanner_sources.make_source_result(source="s", ok=True, status="success", raw_candidates=[_pair("A")]),
    ])
    assert result["status"] == "partial"


def test_collect_all_empty_returns_empty():
    result = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="e", ok=True, status="empty")])
    assert result["status"] == "empty"


def test_collect_all_failed_returns_failed():
    result = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="f", ok=False, status="failed")])
    assert result["status"] == "failed"
    assert result["ok"] is False


def test_collect_deduplicates_across_sources():
    result = scanner_sources.collect_scanner_sources(source_fns=[
        lambda: scanner_sources.make_source_result(source="a", ok=True, status="success", raw_candidates=[_pair("A")]),
        lambda: scanner_sources.make_source_result(source="b", ok=True, status="success", raw_candidates=[_pair("A")]),
    ])
    assert len(result["raw_candidates"]) == 1
    assert result["diagnostics"]["duplicates"] == 1


def test_collect_caps_max_total():
    result = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="s", ok=True, status="success", raw_candidates=[_pair("A"), _pair("B")])], max_total=1)
    assert len(result["raw_candidates"]) == 1
    assert result["diagnostics"]["capped"] is True


def test_scanner_service_accepts_aggregation_dict_from_fetch_sources_fn():
    saved = []
    aggregation = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="s", ok=True, status="success", raw_candidates=[{"chain": "solana", "base_addr": "A", "score": 1}])])
    result = scanner_service.run_scanner_service_cycle(
        fetch_sources_fn=lambda config: aggregation,
        normalize_fn=lambda raw_candidates: {"normalized_candidates": raw_candidates, "rejected_candidates": []},
        filter_fn=lambda **kwargs: {"final_candidates": kwargs["candidates"], "rejected_candidates": []},
        save_payload_fn=lambda payload: saved.append(payload),
        monitoring_rows=[],
        portfolio_rows=[],
        now_ts="now",
    )
    assert result["ok"] is True
    assert saved[-1]["sources_tried"] == ["s"]


def test_live_pulse_payload_debug_includes_source_diagnostics():
    aggregation = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="s", ok=True, status="success", raw_candidates=[_pair("A")])])
    payload = scanner_service.build_live_pulse_payload(raw_candidates=aggregation["raw_candidates"], normalized_candidates=[], final_candidates=[], status="empty", source_state=aggregation)
    assert payload["debug"]["source_results"][0]["source"] == "s"
    assert payload["debug"]["source_diagnostics"]["sources_total"] == 1


def test_empty_all_failed_sources_maps_to_source_api_failed():
    aggregation = scanner_sources.collect_scanner_sources(source_fns=[lambda: scanner_sources.make_source_result(source="bad", ok=False, status="failed", error="boom")])
    payload = scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], status="failed", source_state=aggregation)
    assert payload["last_empty_reason"] == "source_api_failed"


def test_runtime_no_fail_matrix_includes_scanner_sources():
    import scripts.runtime_no_fail_matrix as matrix
    assert "scanner_sources" in matrix.CORE_MODULES
