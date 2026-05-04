import app


def test_unavailable_from_nested_quota_402() -> None:
    contract = {
        "ok": False,
        "code": "runtime_contract_missing",
        "failures": [
            {
                "table": "runtime_state",
                "code": "supabase_read_failed",
                "http_status": 402,
                "detail": "exceed_egress_quota",
            }
        ],
    }
    assert app.is_supabase_unavailable_status(contract) is True


def test_unavailable_from_nested_503() -> None:
    contract = {
        "ok": False,
        "code": "runtime_contract_missing",
        "failures": [{"table": "locks", "http_status": 503}],
    }
    assert app.is_supabase_unavailable_status(contract) is True


def test_missing_contract_without_unavailable_markers_is_not_unavailable() -> None:
    contract = {
        "ok": False,
        "code": "runtime_contract_missing",
        "message": "required runtime tables are missing",
        "failures": [{"table": "runtime_state", "code": "missing_table"}],
    }
    assert app.is_supabase_unavailable_status(contract) is False


def test_unavailable_from_top_level_402() -> None:
    assert app.is_supabase_unavailable_status({"http_status": 402}) is True
