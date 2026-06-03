import copy

import app_service


def _ok_context():
    return {
        "runtime_matrix": {"status": "ok"},
        "app_compat": {"status": "ok"},
        "golden_fixtures": {"status": "ok"},
        "scanner_diagnostics": {
            "source_status": "success",
            "last_empty_reason": "",
            "raw_seen": 10,
            "normalized": 8,
            "clean_candidates": 5,
            "final_count": 3,
            "sources_tried": ["dexscreener"],
        },
        "priority_watchlist_rows": [{"symbol": "AAA"}],
        "priority_watchlist_debug": {"final_priority_rows": 1, "eligible_watch_early_rows": 1, "excluded": {}},
        "storage": {"backend": "d1", "last_read_ok": True, "last_write_ok": True, "verify_status": "ok"},
    }


def test_operator_diagnostics_all_ok_context_status_ok():
    diag = app_service.build_operator_diagnostics(_ok_context())
    assert diag["status"] == "ok"
    assert diag["runtime"]["matrix_status"] == "ok"
    assert diag["scanner"]["final_count"] == 3


def test_operator_diagnostics_live_pulse_empty_with_source_api_empty_warning():
    ctx = {
        "scanner_diagnostics": {
            "source_status": "source_api_empty",
            "last_empty_reason": "source_api_empty",
            "raw_seen": 0,
            "normalized": 0,
            "clean_candidates": 0,
            "final_count": 0,
        }
    }
    diag = app_service.build_operator_diagnostics(ctx)
    assert diag["status"] == "warning"
    assert diag["scanner"]["last_empty_reason"] == "source_api_empty"


def test_operator_diagnostics_source_api_failed_degraded():
    diag = app_service.build_operator_diagnostics({"scanner_diagnostics": {"source_status": "source_api_failed"}})
    assert diag["status"] == "degraded"
    assert "source_api_failed" in diag["summary"]


def test_operator_diagnostics_partial_source_failure_with_candidates_warning():
    ctx = {"scanner_diagnostics": {"source_status": "success", "sources_failed": 1, "final_count": 2}}
    diag = app_service.build_operator_diagnostics(ctx)
    assert diag["status"] == "warning"
    assert diag["scanner"]["sources_failed"] == 1


def test_operator_diagnostics_priority_empty_known_exclusions_warning_not_unknown():
    ctx = {
        "priority_watchlist_rows": [],
        "priority_watchlist_debug": {
            "final_priority_rows": 0,
            "eligible_watch_early_rows": 2,
            "excluded": {"no_entry": 1, "hard_gated": 1},
        },
    }
    diag = app_service.build_operator_diagnostics(ctx)
    assert diag["status"] == "warning"
    assert diag["priority"]["excluded"]["no_entry"] == 1
    assert diag["priority"]["excluded"]["hard_gated"] == 1


def test_operator_diagnostics_missing_context_unknown_no_crash():
    diag = app_service.build_operator_diagnostics({})
    assert diag["status"] == "unknown"
    assert diag["runtime"]["stale_jobs"] == []


def test_operator_diagnostics_bounds_source_errors_and_excluded_samples():
    ctx = {
        "scanner_diagnostics": {"source_status": "success", "final_count": 1, "source_errors": ["e1", "e2", "e3", "e4"]},
        "priority_watchlist_debug": {
            "final_priority_rows": 0,
            "eligible_watch_early_rows": 1,
            "excluded": {"unknown": 1},
            "top_excluded_samples": [{"id": i} for i in range(5)],
        },
    }
    diag = app_service.build_operator_diagnostics(ctx)
    assert diag["scanner"]["source_errors"] == ["e1", "e2", "e3"]
    assert len(diag["priority"]["top_excluded_samples"]) == 3


def test_operator_diagnostics_is_pure_and_redacts_secret_keys():
    ctx = _ok_context()
    ctx["priority_watchlist_debug"] = {
        "final_priority_rows": 0,
        "eligible_watch_early_rows": 1,
        "excluded": {"unknown": 1},
        "top_excluded_samples": [{"symbol": "AAA", "api_key": "SECRET", "telegram_token": "TOKEN"}],
    }
    before = copy.deepcopy(ctx)
    diag = app_service.build_operator_diagnostics(ctx)
    assert ctx == before
    rendered = repr(diag)
    assert "SECRET" not in rendered
    assert "TOKEN" not in rendered
    assert "api_key" not in rendered
