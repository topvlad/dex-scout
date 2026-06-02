#!/usr/bin/env python3
"""Safe runtime import/preflight matrix for DEX Scout.

The matrix intentionally performs only import and dry-run resolution checks.  It
must not write storage, send Telegram messages, or start scanner work.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List
from unittest.mock import patch

REQUIRED_ROLES = ("ui_streamlit", "worker", "webhook", "core_modules")
ROLE_ORDER = (*REQUIRED_ROLES[:3], "dash_readonly", REQUIRED_ROLES[3])
CORE_MODULES = (
    "runtime_core",
    "storage_core",
    "storage_repository",
    "notification_core",
    "monitoring_core",
)
REQUIRED_JOB_MODES = (
    "scan_cycle",
    "monitor_cycle",
    "notify_cycle",
    "digest_cycle",
    "outcome_cycle",
    "maintenance_cycle",
)
SECRET_ENV_HINTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CHAT_ID", "D1_PROXY_URL", "SUPABASE_URL")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _StreamlitContext:
    def __enter__(self) -> "_StreamlitContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Callable[..., Any]:
        return lambda *args, **kwargs: None


def _install_streamlit_stub() -> None:
    """Install a minimal Streamlit stub before smoke importing app.py."""
    if "streamlit" in sys.modules:
        return
    stub = ModuleType("streamlit")
    ctx = _StreamlitContext()
    stub.session_state = {}
    stub.secrets = {}
    stub.sidebar = ctx
    stub.set_page_config = lambda *args, **kwargs: None
    stub.cache_data = _cache_decorator()
    stub.cache_resource = _cache_decorator()
    stub.radio = lambda label, options, index=0, **kwargs: list(options)[index]
    stub.columns = lambda spec, *args, **kwargs: [ctx for _ in range(spec if isinstance(spec, int) else len(spec))]
    stub.tabs = lambda labels, *args, **kwargs: [ctx for _ in labels]
    stub.expander = lambda *args, **kwargs: ctx
    stub.container = lambda *args, **kwargs: ctx
    stub.empty = lambda *args, **kwargs: ctx
    stub.checkbox = lambda label, value=False, **kwargs: value
    stub.number_input = lambda label, value=0, **kwargs: value
    stub.text_input = lambda label, value="", **kwargs: value
    stub.text_area = lambda label, value="", **kwargs: value
    stub.button = lambda *args, **kwargs: False
    stub.form_submit_button = lambda *args, **kwargs: False
    for name in (
        "error", "title", "caption", "json", "markdown", "info", "warning", "success",
        "metric", "write", "table", "line_chart", "subheader", "code", "toast",
        "link_button", "divider", "dataframe", "plotly_chart", "selectbox", "multiselect",
        "form", "spinner",
    ):
        setattr(stub, name, (lambda *args, **kwargs: ctx if name in {"form", "spinner"} else None))
    sys.modules["streamlit"] = stub


def _cache_decorator() -> Callable[..., Any]:
    def cache_data(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorate
    cache_data.clear = lambda: None  # type: ignore[attr-defined]
    return cache_data


def _role(status: str = "ok", *, ok: bool = True, errors: Iterable[str] | None = None, **extra: Any) -> Dict[str, Any]:
    payload = {"ok": bool(ok), "status": status, "errors": list(errors or [])}
    payload.update(extra)
    return payload


def _sanitize(value: Any, limit: int = 180) -> str:
    text = str(value or "")
    for key, secret in os.environ.items():
        if not secret or not any(hint in key.upper() for hint in SECRET_ENV_HINTS):
            continue
        text = text.replace(str(secret), "[REDACTED]")
    return text[:limit]


def _check_no_writes(func: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    write_targets = [
        "storage_repository.StorageRepository.write_text",
        "storage_repository.storage_write_text",
        "storage_core.d1_put_storage",
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
    ]
    patches = []
    write_calls: List[str] = []

    def blocked(*args: Any, **kwargs: Any) -> Any:
        write_calls.append("blocked_write")
        raise AssertionError("runtime_matrix_write_blocked")

    for target in write_targets:
        try:
            patches.append(patch(target, blocked))
        except Exception:
            pass
    with contextlib.ExitStack() as stack:
        for p in patches:
            with contextlib.suppress(Exception):
                stack.enter_context(p)
        result = func()
    if write_calls:
        result.setdefault("errors", []).append("write_attempted")
        result["ok"] = False
        result["status"] = "write_attempted"
    return result


def _ui_streamlit_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        os.environ["DEX_SCOUT_WORKER_MODE"] = "1"
        _install_streamlit_stub()
        app = importlib.import_module("app")
        required = ["build_ui_context", "render_selected_page"]
        missing = [name for name in required if not callable(getattr(app, name, None))]
        if missing:
            return _role("missing_router", ok=False, errors=[f"missing:{','.join(missing)}"])
        for page in ("Monitoring", "Archive", "Portfolio", "Scout", "Runtime"):
            ctx = app.build_ui_context(selected_page=page, actions={
                "render_scout": lambda *a, **k: None,
                "render_monitoring": lambda *a, **k: None,
                "render_archive": lambda *a, **k: None,
                "render_portfolio": lambda *a, **k: None,
                "render_runtime": lambda *a, **k: None,
                "render_debug_panel": lambda *a, **k: None,
            })
            app.render_selected_page(ctx)
        return _role("ok")
    return _capture_role(run)


def _verify_facade_forwarding(facade: ModuleType) -> Dict[str, Any]:
    """Exercise facade forwarding with a fake app without touching storage or jobs."""
    errors: List[str] = []
    captured: Dict[str, Any] = {}

    def update_job_heartbeat(job_name: str, job_mode: str, status: str = "alive", meta: Any = None) -> Dict[str, Any]:
        captured.update({"job_name": job_name, "job_mode": job_mode, "status": status, "meta": meta})
        return {"ok": True}

    fake_app = ModuleType("runtime_matrix_fake_app")
    fake_app.update_job_heartbeat = update_job_heartbeat  # type: ignore[attr-defined]

    with patch.object(facade, "load_app_runtime", lambda force_reload=False: {"ok": True, "app": fake_app}):
        heartbeat = facade.update_job_heartbeat(
            job_name="runtime_notify_cycle",
            job_mode="notify_cycle",
            status="started",
            meta={"x": 1},
        )
        missing = facade._call_app("missing_matrix_function", _facade_job_mode="notify_cycle")

    if heartbeat != {"ok": True}:
        errors.append("heartbeat_forward_result_bad")
    expected = {
        "job_name": "runtime_notify_cycle",
        "job_mode": "notify_cycle",
        "status": "started",
        "meta": {"x": 1},
    }
    if captured != expected:
        errors.append(f"heartbeat_kwargs_not_forwarded:{captured}")
    if missing.get("status") != "missing_app_function" or missing.get("function") != "missing_matrix_function":
        errors.append(f"missing_function_not_structured:{missing}")

    forbidden_internal_names = {"job_mode", "chain", "token_addr", "status", "meta", "action", "event_type"}
    signature = inspect.signature(facade._call_app)
    internal_kwargs = [
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY and name not in {"_facade_job_mode"}
    ]
    collisions = sorted(forbidden_internal_names.intersection(internal_kwargs))
    if collisions:
        errors.append(f"facade_internal_kwarg_collision:{','.join(collisions)}")
    if "_facade_job_mode" not in signature.parameters:
        errors.append("missing_reserved_facade_job_mode")

    return {"ok": not errors, "errors": errors, "heartbeat": captured, "missing_function": missing}


def _worker_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        os.environ["DEX_SCOUT_WORKER_MODE"] = "1"
        facade = importlib.import_module("app_runtime_facade")
        import_state = facade.load_app_runtime(force_reload=True)
        if not import_state.get("ok"):
            return _role(
                "app_import_failed",
                ok=False,
                errors=[_sanitize(import_state.get("error"))],
                exception_type=str(import_state.get("exception_type") or ""),
            )
        resolutions = []
        errors = []
        resolver = getattr(facade, "resolve_worker_job_mode", None)
        for mode in REQUIRED_JOB_MODES:
            if callable(resolver):
                res = resolver(mode, dry_run=True)
                resolutions.append(res)
                if not res.get("ok"):
                    errors.append(f"{mode}:{res.get('status')}")
            else:
                errors.append("missing_facade_resolver")
                break
        forwarding = _verify_facade_forwarding(facade)
        errors.extend(forwarding.get("errors", []))
        status = "ok" if not errors else "worker_preflight_failed"
        return _role(status, ok=not errors, errors=errors, job_modes=resolutions, facade_forwarding=forwarding)
    return _capture_role(run)


def _webhook_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        tg = importlib.import_module("tg_webhook")
        errors: List[str] = []
        status_payload: Dict[str, Any] = {}
        degraded_payload: Dict[str, Any] = {}
        try:
            from fastapi.testclient import TestClient
            client = TestClient(tg.app)
            status_payload = client.get("/_import_status").json()
            if not isinstance(status_payload, dict) or "ok" not in status_payload:
                errors.append("bad_import_status_shape")
            with patch("app_runtime_facade.get_import_state", lambda force_reload=False: {
                "ok": False,
                "error": "forced matrix failure",
                "exception_type": "RuntimeError",
                "worker_fast_mode": True,
                "app_module_loaded": False,
            }):
                degraded_payload = client.get("/_import_status").json()
                callback_payload = client.post("/tg_webhook", json={"callback_query": {"id": "m1", "data": "pf:solana:Abcdefghijklmnopqrstuvwxyz", "message": {"chat": {"id": 1}, "message_id": 1}}}).json()
            if degraded_payload.get("ok") is not False or degraded_payload.get("status") != "app_import_failed":
                errors.append("degraded_status_not_explicit")
            if callback_payload.get("ok") is not False or callback_payload.get("error") != "app_import_failed":
                errors.append("callback_not_app_import_failed")
        except Exception as exc:
            errors.append(f"testclient_unavailable:{type(exc).__name__}:{_sanitize(exc)}")
        return _role("ok" if not errors else "webhook_failed", ok=not errors, errors=errors, import_status=status_payload, forced_degraded=degraded_payload)
    return _capture_role(run)


def _dash_readonly_role() -> Dict[str, Any]:
    if not Path("dash_app.py").exists():
        return _role("skipped_absent")
    def run() -> Dict[str, Any]:
        module = importlib.import_module("dash_app")
        return _role("ok", module=getattr(module, "__name__", "dash_app"))
    return _capture_role(lambda: _check_no_writes(run))


def _core_modules_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        snapshot_names = ("app", *CORE_MODULES)
        snapshot = {name: sys.modules.get(name) for name in snapshot_names}
        imported = []
        errors = []
        try:
            sys.modules.pop("app", None)
            for name in CORE_MODULES:
                sys.modules.pop(name, None)
                try:
                    importlib.import_module(name)
                    imported.append(name)
                except Exception as exc:
                    errors.append(f"{name}:{type(exc).__name__}:{_sanitize(exc)}")
            if "app" in sys.modules:
                errors.append("app_imported_by_core_modules")
        finally:
            for name in snapshot_names:
                sys.modules.pop(name, None)
                if snapshot.get(name) is not None:
                    sys.modules[name] = snapshot[name]  # type: ignore[assignment]
        return _role("ok" if not errors else "core_import_failed", ok=not errors, errors=errors, modules=imported)
    return _capture_role(run)


def _capture_role(func: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        result = func()
        result["errors"] = [_sanitize(e) for e in result.get("errors", [])]
        return result
    except Exception as exc:
        return _role("failed", ok=False, errors=[f"{type(exc).__name__}:{_sanitize(exc)}"], exception_type=type(exc).__name__)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_matrix() -> Dict[str, Any]:
    env_overrides = {
        "JOB_DRY_RUN": "true",
        "DEX_SCOUT_WORKER_MODE": "1",
        "RUNTIME_JOBS_DISABLED": "true",
    }
    previous_env = {key: os.environ.get(key) for key in env_overrides}
    try:
        for key, value in env_overrides.items():
            os.environ.setdefault(key, value)
        roles = {
            "ui_streamlit": _ui_streamlit_role(),
            "worker": _worker_role(),
            "webhook": _webhook_role(),
            "dash_readonly": _dash_readonly_role(),
            "core_modules": _core_modules_role(),
        }
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    ok = all(bool(roles[name].get("ok")) for name in REQUIRED_ROLES)
    return {
        "ok": ok,
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "commit": _commit(),
        "roles": roles,
        "secrets_redacted": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DEX Scout runtime no-fail matrix")
    parser.add_argument("--json", dest="json_path", default="", help="Write compact JSON summary to path")
    args = parser.parse_args(argv)
    if args.json_path and Path(args.json_path).is_dir():
        print("--json must be a file path", file=sys.stderr)
        return 2
    summary = build_matrix()
    compact = json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str)
    print(compact)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
