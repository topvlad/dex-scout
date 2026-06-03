"""Small helpers for scanner golden fixture tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import monitoring_core

_DYNAMIC_KEYS = {"ts_utc", "duration_ms", "run_id", "scan_id", "request_id"}


def load_json_fixture(path: str | Path) -> Any:
    fixture_path = Path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def strip_ts(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: strip_ts(value) for key, value in payload.items() if key not in _DYNAMIC_KEYS}
    if isinstance(payload, list):
        return [strip_ts(item) for item in payload]
    return payload


def normalize_dynamic_fields(payload: Any) -> Any:
    """Remove volatile runtime fields before comparing golden subsets."""
    return strip_ts(payload)


def assert_subset_matches(actual: Any, expected: Any, *, path: str = "payload") -> None:
    """Assert every expected key/value appears in actual with readable paths."""
    actual = normalize_dynamic_fields(actual)
    expected = normalize_dynamic_fields(expected)
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{path}: expected mapping, got {type(actual).__name__}"
        for key, value in expected.items():
            assert key in actual, f"{path}: missing key {key!r}; actual keys={sorted(actual.keys())}"
            assert_subset_matches(actual[key], value, path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) >= len(expected), f"{path}: expected at least {len(expected)} items, got {len(actual)}"
        for idx, value in enumerate(expected):
            assert_subset_matches(actual[idx], value, path=f"{path}[{idx}]")
        return
    assert actual == expected, f"{path}: expected {expected!r}, got {actual!r}"


def stable_candidate_id(row: dict) -> str:
    ident = monitoring_core.build_token_identity(row or {})
    key = str(ident.get("token_key") or "").strip()
    if key:
        return key.replace(":", "|ca:", 1)
    chain = ident.get("chain") or ""
    pair = ident.get("pair_addr") or ""
    return f"{chain}|pair:{pair}" if chain and pair else pair
