import importlib
import json
import sys
from pathlib import Path

import pytest

from storage_repository import StorageRepository, csv_from_text, summarize_storage_result


class MemoryBackend:
    source = "d1"
    verify_source = "d1"

    def __init__(self, initial=None, fail_write=False, fail_message="SECRET_TOKEN_SHOULD_NOT_LEAK"):
        self.data = dict(initial or {})
        self.fail_write = fail_write
        self.fail_message = fail_message
        self.read_calls = []
        self.write_calls = []

    def read_text(self, key: str, bypass_cache: bool = False):
        self.read_calls.append((key, bypass_cache))
        return self.data.get(key)

    def write_text(self, key: str, content: str):
        self.write_calls.append((key, content))
        if self.fail_write:
            raise RuntimeError(self.fail_message)
        self.data[key] = content
        return True


def test_module_imports_without_streamlit_or_app(monkeypatch):
    sys.modules.pop("storage_repository", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)

    module = importlib.import_module("storage_repository")

    assert hasattr(module, "StorageRepository")
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_load_csv_missing_file_returns_empty_list():
    repo = StorageRepository(MemoryBackend(), verify_mode="off")
    assert repo.load_csv("missing.csv", ["a"]) == []


def test_load_csv_fills_missing_fields():
    repo = StorageRepository(MemoryBackend({"x.csv": "a\n1\n"}), verify_mode="off")
    assert repo.load_csv("x.csv", ["a", "b"]) == [{"a": "1", "b": ""}]


def test_load_csv_preserve_extra_true_and_false():
    repo = StorageRepository(MemoryBackend({"x.csv": "a,extra\n1,2\n"}), verify_mode="off")
    assert repo.load_csv("x.csv", ["a"], preserve_extra=True) == [{"a": "1", "extra": "2"}]
    assert repo.load_csv("x.csv", ["a"], preserve_extra=False) == [{"a": "1"}]


def test_empty_and_malformed_csv_returns_empty_list():
    assert csv_from_text("", ["a"]) == []
    assert csv_from_text('a,b\n"unterminated', ["a", "b"]) == []


def test_save_csv_writes_deterministic_csv_with_provided_fields():
    backend = MemoryBackend()
    repo = StorageRepository(backend, verify_mode="off")

    result = repo.save_csv("x.csv", [{"b": "2", "a": "1", "ignored": "z"}], ["a", "b"])

    assert result["write_ok"] is True
    assert backend.data["x.csv"] == "a,b\r\n1,2\r\n"


def test_write_failure_returns_write_not_verified_and_sanitized_error():
    backend = MemoryBackend(fail_write=True, fail_message="super-secret-token")
    repo = StorageRepository(backend, verify_mode="always")

    result = repo.write_text("x", "payload")

    assert result["write_ok"] is False
    assert result["verify_ok"] is False
    assert result["verify_attempted"] is False
    assert "super-secret-token" not in json.dumps(result)


def test_verify_off_returns_verify_skipped_not_verify_ok():
    result = StorageRepository(MemoryBackend(), verify_mode="off").write_text("x", "payload")

    assert result["write_ok"] is True
    assert result["verify_skipped"] is True
    assert result["verify_ok"] is False
    assert summarize_storage_result(result)["verify_status"] == "skipped"


def test_verify_always_bypasses_cache_and_cached_read_not_verification():
    backend = MemoryBackend({"x": "old"})
    repo = StorageRepository(backend, verify_mode="always", read_cache={"x": "cached"})

    assert repo.read_text("x") == "cached"
    result = repo.write_text("x", "new")

    assert result["verify_attempted"] is True
    assert result["verify_ok"] is True
    assert ("x", True) in backend.read_calls
    assert ("x", False) not in backend.read_calls


def test_json_invalid_returns_default_without_mutating_default():
    default = {"items": []}
    repo = StorageRepository(MemoryBackend({"bad.json": "not-json"}), verify_mode="off")

    value = repo.load_json("bad.json", default)

    assert value is default
    assert default == {"items": []}


def test_save_json_roundtrip_uses_compact_deterministic_json():
    backend = MemoryBackend()
    repo = StorageRepository(backend, verify_mode="off")

    repo.save_json("x.json", {"b": 2, "a": "✓"})

    assert backend.data["x.json"] == '{"a":"✓","b":2}'
    assert repo.load_json("x.json", {}) == {"a": "✓", "b": 2}


def test_app_load_csv_save_csv_wrappers_preserve_field_projection(monkeypatch, tmp_path):
    monkeypatch.setenv("DEX_SCOUT_WORKER_MODE", "1")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "USE_D1", False)
    monkeypatch.setattr(app, "USE_SUPABASE", False)
    monkeypatch.setattr(app, "DATA_DIR", str(tmp_path))
    app.st.session_state.clear()

    target = tmp_path / "sample.csv"
    app.save_csv(str(target), [{"a": "1", "b": "2", "extra": "x"}], ["a", "b"])

    assert target.read_text(encoding="utf-8") == "a,b\n1,2\n"
    target.write_text("a,b,extra\n1,2,x\n", encoding="utf-8")
    assert app.load_csv(str(target), ["a", "b"]) == [{"a": "1", "b": "2"}]
