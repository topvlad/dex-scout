"""Import-safe storage repository boundary for DEX Scout.

The repository is intentionally adapter-driven: callers provide a backend with
``read_text`` and ``write_text`` methods, so this module can be imported and
unit-tested without Streamlit, app.py, secrets, or network/storage calls.
"""

from __future__ import annotations

import csv
import io
import json
import random
from typing import Any, Callable, Dict, List, Optional, Protocol


class StorageBackend(Protocol):
    def read_text(self, key: str, bypass_cache: bool = False) -> Optional[str]: ...

    def write_text(self, key: str, content: str) -> Any: ...


WRITE_RESULT_DEFAULT: Dict[str, Any] = {
    "ok": False,
    "write_attempted": False,
    "write_ok": False,
    "verify_attempted": False,
    "verify_ok": False,
    "verify_skipped": True,
    "verify_source": "none",
    "error": "",
}


def _compact_error(code: str, exc: BaseException | None = None) -> str:
    """Return a small diagnostic without exception text or secret-like values."""
    if exc is None:
        return str(code or "storage_error")[:80]
    return f"{str(code or 'storage_error')[:48]}:{type(exc).__name__}"


def _backend_source(backend: StorageBackend) -> str:
    source = str(getattr(backend, "verify_source", "") or getattr(backend, "source", "") or "none").strip().lower()
    return source if source in {"d1", "supabase", "local", "none"} else "none"


def _base_result(**updates: Any) -> Dict[str, Any]:
    result = dict(WRITE_RESULT_DEFAULT)
    result.update(updates)
    result["ok"] = bool(result.get("write_ok")) and (bool(result.get("verify_ok")) or bool(result.get("verify_skipped")))
    if not result.get("write_ok"):
        result["verify_ok"] = False
    return result


def normalize_csv_rows(
    rows: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    *,
    preserve_extra: bool = True,
) -> List[Dict[str, Any]]:
    if not fields:
        return [dict(row or {}) for row in rows]
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        source = dict(row or {})
        out: Dict[str, Any] = {name: source.get(name, "") or "" for name in fields}
        if preserve_extra:
            for key, value in source.items():
                if key is not None and key not in out:
                    out[str(key)] = value if value is not None else ""
        normalized.append(out)
    return normalized


def csv_from_text(
    content: str | None,
    fields: Optional[List[str]] = None,
    *,
    preserve_extra: bool = True,
) -> List[Dict[str, Any]]:
    if not str(content or "").strip():
        return []
    try:
        reader = csv.DictReader(io.StringIO(str(content)), strict=True)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            if row is None or None in row:
                return []
            rows.append(dict(row))
        return normalize_csv_rows(rows, fields, preserve_extra=preserve_extra)
    except (csv.Error, ValueError, TypeError):
        return []


def csv_to_text(rows: List[Dict[str, Any]], fields: List[str]) -> str:
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(fields or []), extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    for row in rows or []:
        writer.writerow({name: (row or {}).get(name, "") for name in fields or []})
    return sio.getvalue()


def summarize_storage_result(result: Dict[str, Any]) -> Dict[str, Any]:
    verify_status = "skipped"
    if result.get("verify_attempted"):
        verify_status = "ok" if result.get("verify_ok") else "failed"
    elif not result.get("verify_skipped"):
        verify_status = "failed"
    return {
        "ok": bool(result.get("ok")),
        "write_ok": bool(result.get("write_ok")),
        "verify_status": verify_status,
        "verify_source": str(result.get("verify_source") or "none"),
        "error": str(result.get("error") or "")[:160],
    }


class StorageRepository:
    def __init__(
        self,
        backend: StorageBackend,
        *,
        verify_mode: str = "off",
        read_cache: Optional[Dict[str, Optional[str]]] = None,
        now_fn: Optional[Callable[[], float]] = None,
        verify_sample_rate: float = 1.0,
    ):
        self.backend = backend
        self.verify_mode = str(verify_mode or "off").strip().lower()
        if self.verify_mode not in {"off", "sampled", "always"}:
            self.verify_mode = "off"
        self.read_cache = read_cache if read_cache is not None else {}
        self.now_fn = now_fn
        self.verify_sample_rate = max(0.0, min(1.0, float(verify_sample_rate)))
        self.last_result: Dict[str, Any] = dict(WRITE_RESULT_DEFAULT)
        self.last_diagnostic: Dict[str, Any] = {}

    def load_csv(self, path: str, fields: Optional[List[str]] = None, preserve_extra: bool = True) -> List[Dict[str, Any]]:
        content = self.read_text(path)
        rows = csv_from_text(content, fields, preserve_extra=preserve_extra)
        if content and not rows:
            self.last_diagnostic = {"op": "load_csv", "key": str(path), "error": "malformed_csv"}
        return rows

    def save_csv(self, path: str, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        if fields is None:
            seen: List[str] = []
            for row in rows or []:
                for key in (row or {}).keys():
                    if key not in seen:
                        seen.append(str(key))
            fields = seen
        return self.write_text(path, csv_to_text(rows or [], list(fields or [])))

    def load_json(self, key: str, default: Any = None) -> Any:
        content = self.read_text(key)
        if content is None or str(content).strip() == "":
            return default
        try:
            return json.loads(str(content))
        except (json.JSONDecodeError, TypeError, ValueError):
            self.last_diagnostic = {"op": "load_json", "key": str(key), "error": "invalid_json"}
            return default

    def save_json(self, key: str, payload: Any) -> Dict[str, Any]:
        try:
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            content = json.dumps(payload, ensure_ascii=False, sort_keys=False, default=str, separators=(",", ":"))
        return self.write_text(key, content)

    def read_text(self, key: str, *, bypass_cache: bool = False) -> Optional[str]:
        cache_key = str(key)
        if not bypass_cache and cache_key in self.read_cache:
            return self.read_cache[cache_key]
        try:
            content = self.backend.read_text(cache_key, bypass_cache=bool(bypass_cache))
        except Exception as exc:  # adapter boundary: never leak raw error text
            self.last_diagnostic = {"op": "read_text", "key": cache_key, "error": _compact_error("read_failed", exc)}
            return None
        if not bypass_cache:
            self.read_cache[cache_key] = content
        return content

    def write_text(self, key: str, content: str) -> Dict[str, Any]:
        key = str(key)
        payload = str(content or "")
        result = _base_result(write_attempted=True, verify_source=_backend_source(self.backend), verify_skipped=self.verify_mode == "off")
        try:
            write_status = self.backend.write_text(key, payload)
            if isinstance(write_status, dict):
                result.update({k: write_status.get(k, result.get(k)) for k in result.keys() if k in write_status})
                result["write_ok"] = bool(write_status.get("write_ok", write_status.get("ok", False)))
                result["error"] = str(write_status.get("error") or "")[:160]
            else:
                result["write_ok"] = bool(write_status)
        except Exception as exc:
            result["write_ok"] = False
            result["error"] = _compact_error("write_failed", exc)
        if not result.get("write_ok"):
            result.update({"ok": False, "verify_attempted": False, "verify_ok": False, "verify_skipped": self.verify_mode == "off"})
            self.last_result = result
            return result

        if self.verify_mode == "off":
            result.update({"verify_attempted": False, "verify_ok": False, "verify_skipped": True})
        elif self.verify_mode == "sampled" and random.random() > self.verify_sample_rate:
            result.update({"verify_attempted": False, "verify_ok": False, "verify_skipped": True})
        else:
            result.update({"verify_attempted": True, "verify_skipped": False})
            try:
                verified_content = self.backend.read_text(key, bypass_cache=True)
                result["verify_ok"] = verified_content == payload
                if not result["verify_ok"] and not result.get("error"):
                    result["error"] = "verify_mismatch"
            except Exception as exc:
                result["verify_ok"] = False
                result["error"] = _compact_error("verify_failed", exc)
        result["ok"] = bool(result.get("write_ok")) and (bool(result.get("verify_ok")) or bool(result.get("verify_skipped")))
        self.last_result = result
        if not result.get("verify_attempted"):
            self.read_cache[key] = payload
        return result
