"""Import-safe runtime facade for worker and Telegram webhook roles.

This module intentionally does not import ``app.py`` at module import time.  It
only lazy-loads app when a caller asks for runtime functionality, after marking
this process as worker/runtime mode so app.py can use its import-safe path.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Dict, Optional

_APP_MODULE: Optional[ModuleType] = None
_LAST_RESULT: Optional[Dict[str, Any]] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _worker_fast_mode() -> bool:
    return str(os.getenv("DEX_SCOUT_WORKER_MODE", "")).strip() == "1"


def _failure_payload(runtime: Dict[str, Any], job_mode: str = "") -> Dict[str, Any]:
    payload = {
        "ok": False,
        "status": "app_import_failed",
        "reason": str(runtime.get("error") or "app import failed"),
        "exception_type": str(runtime.get("exception_type") or ""),
    }
    if job_mode:
        payload["job_mode"] = job_mode
    return payload


def load_app_runtime(force_reload: bool = False) -> Dict[str, Any]:
    """Lazy-load app.py and return structured import state.

    Successful imports are cached. Failed imports are cached too so health/status
    checks do not repeatedly crash the runtime, but ``force_reload=True`` retries.
    """

    global _APP_MODULE, _LAST_RESULT

    if _LAST_RESULT is not None and not force_reload:
        return dict(_LAST_RESULT)

    os.environ.setdefault("DEX_SCOUT_WORKER_MODE", "1")
    loaded_at = _utc_now()
    try:
        module = importlib.import_module("app")
        _APP_MODULE = module
        _LAST_RESULT = {
            "ok": True,
            "app": module,
            "error": "",
            "exception_type": "",
            "loaded_at": loaded_at,
            "worker_fast_mode": _worker_fast_mode(),
        }
    except Exception as exc:  # pragma: no cover - covered through monkeypatches
        _APP_MODULE = None
        _LAST_RESULT = {
            "ok": False,
            "app": None,
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "loaded_at": loaded_at,
            "worker_fast_mode": _worker_fast_mode(),
        }
    return dict(_LAST_RESULT)


def get_app(force_reload: bool = False) -> Optional[ModuleType]:
    return load_app_runtime(force_reload=force_reload).get("app")  # type: ignore[return-value]


def get_import_state(force_reload: bool = False) -> Dict[str, Any]:
    state = load_app_runtime(force_reload=force_reload)
    return {
        "ok": bool(state.get("ok")),
        "error": str(state.get("error") or ""),
        "exception_type": str(state.get("exception_type") or ""),
        "loaded_at": str(state.get("loaded_at") or ""),
        "worker_fast_mode": bool(state.get("worker_fast_mode")),
        "app_module_loaded": state.get("app") is not None,
    }


def _call_app(name: str, *args: Any, job_mode: str = "", **kwargs: Any) -> Any:
    runtime = load_app_runtime()
    module = runtime.get("app")
    if not runtime.get("ok") or module is None:
        return _failure_payload(runtime, job_mode=job_mode)
    fn = getattr(module, name)
    return fn(*args, **kwargs)


def _get_app_attr(name: str) -> Any:
    runtime = load_app_runtime()
    module = runtime.get("app")
    if not runtime.get("ok") or module is None:
        raise RuntimeError(
            "app_import_failed:"
            f"{runtime.get('exception_type')}:"
            f"{runtime.get('error')}"
        )
    return getattr(module, name)


def __getattr__(name: str) -> Any:
    return _get_app_attr(name)


def get_worker_runtime_state(*args: Any, **kwargs: Any) -> Any:
    return _call_app("get_worker_runtime_state", *args, **kwargs)


def update_worker_runtime_state(*args: Any, **kwargs: Any) -> Any:
    return _call_app("update_worker_runtime_state", *args, **kwargs)


def update_job_heartbeat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("update_job_heartbeat", *args, **kwargs)


def acquire_lock(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("acquire_lock", *args, **kwargs)


def release_lock(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("release_lock", *args, **kwargs)


def run_storage_maintenance_cycle(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("run_storage_maintenance_cycle", *args, job_mode="maintenance_cycle", **kwargs)


def trigger_digest_notification(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("trigger_digest_notification", *args, job_mode="digest_cycle", **kwargs)


def evaluate_outcome_journals(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("evaluate_outcome_journals", *args, job_mode="outcome_cycle", **kwargs)


def maybe_run_rotating_scanner(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("maybe_run_rotating_scanner", *args, job_mode="scan_cycle", **kwargs)


def run_priority_scanner_cycle(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("run_priority_scanner_cycle", *args, job_mode="monitor_cycle", **kwargs)


def run_auto_notifications(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _call_app("run_auto_notifications", *args, job_mode="notify_cycle", **kwargs)
