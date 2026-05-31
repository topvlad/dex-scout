"""Import-safe storage helpers for DEX Scout.

This module centralizes storage constants and pure helper logic that can be used
by runtime entry points without importing the Streamlit app module. It performs
no reads, writes, or network calls at import time.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

VALID_STORAGE_BACKENDS = {"supabase", "d1", "local"}
DEFAULT_STORAGE_BACKEND = "supabase"

RUNTIME_STATE_TABLE = "runtime_state"
LOCKS_TABLE = "locks"
TG_STATE_TABLE = "tg_state"
JOB_HEARTBEATS_TABLE = "job_heartbeats"
JOB_RUNS_TABLE = "job_runs"
RUNTIME_REQUIRED_TABLES: Tuple[str, ...] = (
    RUNTIME_STATE_TABLE,
    LOCKS_TABLE,
    TG_STATE_TABLE,
    JOB_HEARTBEATS_TABLE,
)


def normalize_storage_backend(raw_backend: Any, default: str = DEFAULT_STORAGE_BACKEND) -> str:
    raw_default = str(default or DEFAULT_STORAGE_BACKEND).strip().lower()
    fallback = raw_default if raw_default in VALID_STORAGE_BACKENDS else DEFAULT_STORAGE_BACKEND
    backend = str(raw_backend or fallback).strip().lower()
    return backend if backend in VALID_STORAGE_BACKENDS else fallback


def storage_key_for_path(path: str) -> str:
    # keep it stable and short: data/monitoring.csv -> monitoring.csv
    base = os.path.basename(path)
    return base or path


def runtime_status(ok: bool, code: str, message: str = "", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": bool(ok),
        "code": str(code or "unknown"),
        "message": str(message or ""),
    }
    if extra:
        payload.update(extra)
    return payload


def supabase_configured(url: str, service_role_key: str, backend: str = DEFAULT_STORAGE_BACKEND) -> bool:
    return normalize_storage_backend(backend) == "supabase" and bool(str(url or "").strip() and str(service_role_key or "").strip())


def d1_configured(proxy_url: str, proxy_token: str, backend: str = DEFAULT_STORAGE_BACKEND) -> bool:
    return normalize_storage_backend(backend) == "d1" and bool(str(proxy_url or "").strip() and str(proxy_token or "").strip())


def storage_backend_flags(
    backend: str,
    *,
    supabase_url: str = "",
    supabase_service_role_key: str = "",
    d1_proxy_url: str = "",
    d1_proxy_token: str = "",
) -> Dict[str, bool]:
    normalized = normalize_storage_backend(backend)
    return {
        "use_d1": normalized == "d1",
        "use_supabase": supabase_configured(supabase_url, supabase_service_role_key, normalized),
        "d1_ok": d1_configured(d1_proxy_url, d1_proxy_token, normalized),
    }
