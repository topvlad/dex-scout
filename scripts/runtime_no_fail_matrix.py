#!/usr/bin/env python3
"""Safe runtime import/preflight matrix for DEX Scout.

The matrix intentionally performs only import and dry-run resolution checks.  It
must not write storage, send Telegram messages, or start scanner work.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import types
from urllib.parse import parse_qs, urlsplit
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

REQUIRED_ROLES = ("ui_streamlit", "worker", "webhook", "app_compat", "core_modules", "golden_fixtures", "admin_controls", "external_audit_claims")
ROLE_ORDER = (*REQUIRED_ROLES[:3], "dash_readonly", REQUIRED_ROLES[3], REQUIRED_ROLES[4], REQUIRED_ROLES[5], REQUIRED_ROLES[6], REQUIRED_ROLES[7])
CORE_MODULES = (
    "runtime_core",
    "storage_core",
    "storage_repository",
    "notification_core",
    "monitoring_core",
    "monitoring_service",
    "portfolio_service",
    "app_service",
    "scanner_service",
    "scanner_sources",
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
APP_COMPAT_MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "app_compat_manifest.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_requests_stub() -> None:
    """Install a no-network requests stub when requests is unavailable."""
    if "requests" in sys.modules:
        return
    if importlib.util.find_spec("requests") is not None:
        return

    class _Response:
        status_code = 200
        text = ""
        content = b""
        headers: Dict[str, Any] = {}

        def __init__(self, json_data: Any = None, status_code: int = 200, text: str = "") -> None:
            self._json_data = {} if json_data is None else json_data
            self.status_code = int(status_code)
            self.text = str(text or "")
            self.content = self.text.encode()
            self.headers = {}

        def json(self) -> Any:
            return self._json_data

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def _request(*args: Any, **kwargs: Any) -> _Response:
        return _Response()

    class _Session:
        def request(self, *args: Any, **kwargs: Any) -> _Response:
            return _request(*args, **kwargs)
        def get(self, *args: Any, **kwargs: Any) -> _Response:
            return _request(*args, **kwargs)
        def post(self, *args: Any, **kwargs: Any) -> _Response:
            return _request(*args, **kwargs)

    stub = ModuleType("requests")
    stub.get = _request  # type: ignore[attr-defined]
    stub.post = _request  # type: ignore[attr-defined]
    stub.put = _request  # type: ignore[attr-defined]
    stub.patch = _request  # type: ignore[attr-defined]
    stub.delete = _request  # type: ignore[attr-defined]
    stub.request = _request  # type: ignore[attr-defined]
    stub.Session = _Session  # type: ignore[attr-defined]
    stub.Response = _Response  # type: ignore[attr-defined]
    stub.exceptions = types.SimpleNamespace(RequestException=Exception, Timeout=TimeoutError)  # type: ignore[attr-defined]
    sys.modules["requests"] = stub


def _install_pandas_stub() -> None:
    if "pandas" in sys.modules:
        return
    if importlib.util.find_spec("pandas") is not None:
        return

    class _DataFrame(list):
        def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
            super().__init__(data if isinstance(data, list) else [])
        def copy(self) -> "_DataFrame":
            return _DataFrame(list(self))

    stub = ModuleType("pandas")
    stub.DataFrame = _DataFrame  # type: ignore[attr-defined]
    stub.to_datetime = lambda value, **kwargs: value  # type: ignore[attr-defined]
    stub.to_numeric = lambda value, **kwargs: value  # type: ignore[attr-defined]
    sys.modules["pandas"] = stub


def _install_fastapi_stub() -> None:
    if "fastapi" in sys.modules:
        return
    if importlib.util.find_spec("fastapi") is not None:
        return

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: Any = None) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Response:
        def __init__(self, content: Any = b"", status_code: int = 200, media_type: str | None = None) -> None:
            self.content = content
            self.status_code = int(status_code)
            self.media_type = media_type

    class Request:
        def __init__(self, json_payload: Any = None) -> None:
            self._json_payload = json_payload or {}
        async def json(self) -> Any:
            return self._json_payload

    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.routes: Dict[tuple[str, str], Callable[..., Any]] = {}
        def _decorator(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes[(method.upper(), path)] = func
                return func
            return decorate
        def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._decorator("GET", path)
        def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._decorator("POST", path)
        def head(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._decorator("HEAD", path)

    class _ClientResponse:
        def __init__(self, payload: Any = None, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = int(status_code)
        def json(self) -> Any:
            return self._payload

    class TestClient:
        def __init__(self, app: Any) -> None:
            self.app = app
        def get(self, url: str) -> _ClientResponse:
            return self._request("GET", url)
        def post(self, url: str, json: Any = None) -> _ClientResponse:
            return self._request("POST", url, json_payload=json)
        def _request(self, method: str, url: str, json_payload: Any = None) -> _ClientResponse:
            parsed = urlsplit(url)
            func = self.app.routes[(method.upper(), parsed.path)]
            kwargs = {key: value[-1] for key, value in parse_qs(parsed.query).items()}
            signature = inspect.signature(func)
            if "req" in signature.parameters:
                kwargs["req"] = Request(json_payload)
            result = func(**kwargs)
            if inspect.isawaitable(result):
                import asyncio
                result = asyncio.run(result)
            if isinstance(result, Response):
                return _ClientResponse(result.content, result.status_code)
            return _ClientResponse(result, 200)

    fastapi_stub = ModuleType("fastapi")
    fastapi_stub.FastAPI = FastAPI  # type: ignore[attr-defined]
    fastapi_stub.HTTPException = HTTPException  # type: ignore[attr-defined]
    fastapi_stub.Request = Request  # type: ignore[attr-defined]
    fastapi_stub.Response = Response  # type: ignore[attr-defined]
    testclient_stub = ModuleType("fastapi.testclient")
    testclient_stub.TestClient = TestClient  # type: ignore[attr-defined]
    fastapi_stub.testclient = testclient_stub  # type: ignore[attr-defined]
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.testclient"] = testclient_stub


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
        _install_requests_stub()
        _install_pandas_stub()
        _install_streamlit_stub()
        app = importlib.import_module("app")
        required = ["build_ui_context", "render_selected_page"]
        missing = [name for name in required if not callable(getattr(app, name, None))]
        if missing:
            return _role("missing_router", ok=False, errors=[f"missing:{','.join(missing)}"])
        import app_service
        fixture_diag = app_service.build_operator_diagnostics({
            "scanner_diagnostics": {"source_status": "source_api_empty", "last_empty_reason": "source_api_empty", "final_count": 0},
            "priority_watchlist_debug": {"final_priority_rows": 0, "eligible_watch_early_rows": 1, "excluded": {"no_entry": 1}},
        })
        failed_diag = app_service.build_operator_diagnostics({"scanner_diagnostics": {"source_status": "source_api_failed"}})
        if fixture_diag.get("status") != "warning" or failed_diag.get("status") != "degraded":
            return _role("operator_diagnostics_status_failed", ok=False, errors=[f"unexpected_operator_status:{fixture_diag.get('status')}:{failed_diag.get('status')}"])
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



def _load_app_compat_manifest() -> Dict[str, List[str]]:
    data = json.loads(APP_COMPAT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("app compat manifest must be a JSON object")
    return {str(category): [str(name) for name in names] for category, names in data.items() if isinstance(names, list)}


def _manifest_function_names(manifest: Dict[str, List[str]]) -> List[str]:
    names: List[str] = []
    for values in manifest.values():
        names.extend(values)
    return sorted(set(names))


def _app_compat_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        os.environ["DEX_SCOUT_WORKER_MODE"] = "1"
        _install_requests_stub()
        _install_pandas_stub()
        _install_streamlit_stub()
        manifest = _load_app_compat_manifest()
        required_wrappers = _manifest_function_names(manifest)
        app = importlib.import_module("app")
        facade = importlib.import_module("app_runtime_facade")
        errors: List[str] = []

        for name in required_wrappers:
            if not callable(getattr(app, name, None)):
                errors.append(f"missing_or_not_callable:{name}")

        runner_map = getattr(facade, "WORKER_JOB_MODE_RUNNERS", {})
        resolver = getattr(facade, "resolve_worker_job_mode", None)
        for mode, runner_name in sorted(dict(runner_map).items()):
            if not callable(getattr(app, str(runner_name), None)):
                errors.append(f"runner_missing:{mode}:{runner_name}")
            if callable(resolver):
                resolved = resolver(str(mode), dry_run=True)
                if not resolved.get("ok"):
                    errors.append(f"runner_resolution_failed:{mode}:{resolved.get('status')}")
            else:
                errors.append("missing_facade_resolver")
                break

        from scripts import audit_app_compat
        audit = audit_app_compat.audit_app_path(REPO_ROOT / "app.py")
        duplicates = list(audit.get("duplicates") or [])
        if duplicates:
            errors.append(f"duplicate_top_level_defs:{duplicates}")

        # Verify service modules still import independently of app.py/Streamlit.
        snapshot_names = ("app", "streamlit", *CORE_MODULES)
        snapshot = {name: sys.modules.get(name) for name in snapshot_names}
        try:
            sys.modules.pop("app", None)
            sys.modules.pop("streamlit", None)
            for name in CORE_MODULES:
                sys.modules.pop(name, None)
                importlib.import_module(name)
            if sys.modules.get("app") is not None:
                errors.append("service_modules_imported_app")
            if sys.modules.get("streamlit") is not None:
                errors.append("service_modules_imported_streamlit")
        finally:
            for name in snapshot_names:
                sys.modules.pop(name, None)
                if snapshot.get(name) is not None:
                    sys.modules[name] = snapshot[name]  # type: ignore[assignment]

        return _role(
            "ok" if not errors else "app_compat_failed",
            ok=not errors,
            errors=errors,
            checked_wrappers=len(required_wrappers),
            duplicates=duplicates,
            stale_markers=list(audit.get("stale_markers") or []),
        )
    return _capture_role(run)


def _worker_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        os.environ["DEX_SCOUT_WORKER_MODE"] = "1"
        _install_requests_stub()
        _install_pandas_stub()
        _install_streamlit_stub()
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
        _install_requests_stub()
        _install_fastapi_stub()
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
        _install_pandas_stub()
        module = importlib.import_module("dash_app")
        return _role("ok", module=getattr(module, "__name__", "dash_app"))
    return _capture_role(lambda: _check_no_writes(run))


def _core_modules_role() -> Dict[str, Any]:
    def run() -> Dict[str, Any]:
        snapshot_names = ("app", "streamlit", *CORE_MODULES)
        snapshot = {name: sys.modules.get(name) for name in snapshot_names}
        imported = []
        errors = []
        try:
            sys.modules.pop("app", None)
            sys.modules.pop("streamlit", None)
            for name in CORE_MODULES:
                sys.modules.pop(name, None)
                try:
                    importlib.import_module(name)
                    imported.append(name)
                except Exception as exc:
                    errors.append(f"{name}:{type(exc).__name__}:{_sanitize(exc)}")
            if "app" in sys.modules:
                errors.append("app_imported_by_core_modules")
            if "streamlit" in sys.modules:
                errors.append("streamlit_imported_by_core_modules")
            monitoring_service = sys.modules.get("monitoring_service")
            if monitoring_service is None:
                errors.append("monitoring_service_not_imported")
            else:
                service_file = str(getattr(monitoring_service, "__file__", "") or "")
                imported_for_service = {
                    "streamlit": sys.modules.get("streamlit"),
                    "app": sys.modules.get("app"),
                }
                if imported_for_service["streamlit"] is not None:
                    errors.append(f"monitoring_service_imported_streamlit:{service_file}")
                if imported_for_service["app"] is not None:
                    errors.append(f"monitoring_service_imported_app:{service_file}")
                watch = getattr(monitoring_service, "classify_monitoring_row")({
                    "chain": "solana", "base_addr": "watch", "base_symbol": "WATCH", "entry_status": "WATCH", "priority_score": "10"
                })
                if not (watch.get("is_active") and watch.get("is_actionable") and watch.get("priority_eligible")):
                    errors.append(f"monitoring_service_watch_smoke_failed:{watch}")
                no_entry = getattr(monitoring_service, "classify_monitoring_row")({
                    "chain": "solana", "base_addr": "no", "base_symbol": "NO", "entry_status": "NO_ENTRY", "priority_score": "10"
                })
                if no_entry.get("priority_eligible") or no_entry.get("is_actionable"):
                    errors.append(f"monitoring_service_no_entry_smoke_failed:{no_entry}")
                conflict = getattr(monitoring_service, "classify_monitoring_row")(
                    {"chain": "solana", "base_addr": "exit", "base_symbol": "EXIT", "entry_status": "WATCH", "priority_score": "10"},
                    {"chain": "solana", "base_token_address": "exit", "base_symbol": "EXIT", "final_action": "EXIT"},
                )
                if not conflict.get("portfolio_conflict") or conflict.get("priority_eligible"):
                    errors.append(f"monitoring_service_portfolio_conflict_smoke_failed:{conflict}")

            scanner_service = sys.modules.get("scanner_service")
            if scanner_service is None:
                errors.append("scanner_service_not_imported")
            else:
                if sys.modules.get("streamlit") is not None:
                    errors.append("scanner_service_imported_streamlit")
                if sys.modules.get("app") is not None:
                    errors.append("scanner_service_imported_app")
                payload = scanner_service.build_live_pulse_payload(
                    raw_candidates=[{"chain": "solana", "base_addr": "a"}, {"chain": "solana", "base_addr": "b"}, {"chain": "solana", "base_addr": "c"}],
                    normalized_candidates=[{"chain": "solana", "base_addr": "a"}, {"chain": "solana", "base_addr": "b"}],
                    final_candidates=[{"chain": "solana", "base_addr": "a", "score": 1}],
                    status="success",
                    source="test",
                    now_ts="matrix",
                )
                if payload.get("raw_seen") != 3 or payload.get("normalized") != 2 or payload.get("final_count") != 1:
                    errors.append(f"scanner_service_payload_counters_failed:{payload}")
                failed_payload = scanner_service.build_live_pulse_payload(raw_candidates=[], normalized_candidates=[], final_candidates=[], status="failed", source="test", error="scanner boom", now_ts="matrix")
                if failed_payload.get("last_empty_reason") not in {"worker_failed", "scanner_failed"}:
                    errors.append(f"scanner_service_failed_reason_failed:{failed_payload}")
                refill = scanner_service.plan_live_pulse_refill(current_count=1, target_min=3, target_max=5, max_attempts=2, sources_tried=[])
                if not refill.get("should_refill") or refill.get("reason") != "below_target":
                    errors.append(f"scanner_service_refill_smoke_failed:{refill}")
                filtered = scanner_service.filter_live_pulse_candidates(
                    [
                        {"chain": "solana", "base_addr": "watch", "base_symbol": "WATCH", "entry_status": "WATCH", "score": "10"},
                        {"chain": "solana", "base_addr": "no", "base_symbol": "NO", "entry_status": "NO_ENTRY", "score": "10"},
                        {"chain": "solana", "base_addr": "hard", "base_symbol": "HARD", "entry_status": "WATCH", "score": "10", "risk_flags": "toxic"},
                    ],
                    hard_gate_fn=lambda row: {"blocked": str(row.get("entry_status")) == "NO_ENTRY"},
                    max_candidates=5,
                )
                if filtered.get("diagnostics", {}).get("final_count") != 1 or filtered.get("diagnostics", {}).get("hard_gated") < 1:
                    errors.append(f"scanner_service_filter_smoke_failed:{filtered}")



            scanner_sources = sys.modules.get("scanner_sources")
            if scanner_sources is None:
                errors.append("scanner_sources_not_imported")
            else:
                if sys.modules.get("streamlit") is not None:
                    errors.append("scanner_sources_imported_streamlit")
                if sys.modules.get("app") is not None:
                    errors.append("scanner_sources_imported_app")
                source_result = scanner_sources.make_source_result(source="test", ok=True, status="success", raw_candidates=[{"chainId": "solana", "baseToken": {"address": "A"}}])
                if source_result.get("raw_count") != 1 or len(source_result.get("raw_candidates") or []) != 1:
                    errors.append(f"scanner_sources_make_result_failed:{source_result}")
                disabled = scanner_sources.fetch_birdeye_trending_source(enabled=False)
                if disabled.get("ok") is not True or disabled.get("status") != "disabled":
                    errors.append(f"scanner_sources_disabled_birdeye_failed:{disabled}")
                success = lambda: scanner_sources.make_source_result(source="one", ok=True, status="success", raw_candidates=[{"chainId": "solana", "baseToken": {"address": "A"}}])
                failed = lambda: scanner_sources.make_source_result(source="bad", ok=False, status="failed", error="boom")
                empty = lambda: scanner_sources.make_source_result(source="empty", ok=True, status="empty")
                if scanner_sources.collect_scanner_sources(source_fns=[success]).get("status") != "success":
                    errors.append("scanner_sources_collect_success_failed")
                if scanner_sources.collect_scanner_sources(source_fns=[failed, success]).get("status") != "partial":
                    errors.append("scanner_sources_collect_partial_failed")
                if scanner_sources.collect_scanner_sources(source_fns=[empty, empty]).get("status") != "empty":
                    errors.append("scanner_sources_collect_empty_failed")
                if scanner_sources.collect_scanner_sources(source_fns=[failed, failed]).get("status") != "failed":
                    errors.append("scanner_sources_collect_failed_failed")

            portfolio_service = sys.modules.get("portfolio_service")
            if portfolio_service is None:
                errors.append("portfolio_service_not_imported")
            else:
                if getattr(portfolio_service, "is_material_portfolio_action")({"final_action": "REDUCE", "current_price": ""}) is not True:
                    errors.append("portfolio_material_classifier_smoke_failed")
                conflict = getattr(portfolio_service, "resolve_portfolio_monitoring_conflict")(
                    {"final_action": "REDUCE", "current_price": ""},
                    {"entry_status": "WATCH"},
                )
                if conflict.get("source") != "portfolio" or conflict.get("display_action") != "REDUCE" or not conflict.get("material_portfolio_action"):
                    errors.append(f"portfolio_conflict_resolver_smoke_failed:{conflict}")
        finally:
            for name in snapshot_names:
                sys.modules.pop(name, None)
                if snapshot.get(name) is not None:
                    sys.modules[name] = snapshot[name]  # type: ignore[assignment]
        return _role("ok" if not errors else "core_import_failed", ok=not errors, errors=errors, modules=imported)
    return _capture_role(run)


def _golden_fixtures_role() -> Dict[str, Any]:
    """Cheap golden-fixture smoke with no network, storage write, send, or scan."""
    def run() -> Dict[str, Any]:
        errors: List[str] = []
        fixture_dir = REPO_ROOT / "tests" / "fixtures" / "scanner_golden"
        fixture_paths = sorted(fixture_dir.glob("*.json")) if fixture_dir.exists() else []
        if not fixture_dir.exists():
            errors.append("fixtures_dir_missing")
        fixtures: Dict[str, Any] = {}
        for path in fixture_paths:
            try:
                fixtures[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"fixture_invalid_json:{path.name}:{type(exc).__name__}")
        try:
            import scanner_service
            import scanner_sources
            import monitoring_service

            raw = fixtures.get("raw_dex_pairs_mixed.json") or []
            conflicts = fixtures.get("monitoring_portfolio_conflicts.json") or {}
            norm = scanner_service.normalize_scanner_candidates(raw)
            filtered = scanner_service.filter_live_pulse_candidates(
                norm.get("normalized_candidates", []),
                portfolio_rows=(conflicts.get("portfolio_rows") or [])[:1],
                archive_rows=[{"chain": "solana", "base_addr": "ArchiveSoL111111111111111111111111111111"}],
                hard_gate_fn=lambda row: {"blocked": False},
                max_candidates=5,
            )
            payload = scanner_service.build_live_pulse_payload(
                raw_candidates=raw,
                normalized_candidates=norm.get("normalized_candidates", []),
                final_candidates=filtered.get("final_candidates", []),
                rejected_candidates=(norm.get("rejected_candidates", []) + filtered.get("rejected_candidates", [])),
                status="success",
                source="test",
                now_ts="matrix",
            )
            if payload.get("final_count", 0) <= 0 or payload.get("raw_seen") != len(raw):
                errors.append(f"scanner_fixture_payload_failed:{payload}")

            aggregate = scanner_sources.collect_scanner_sources(
                source_fns=[lambda: scanner_sources.make_source_result(source="fixture", ok=True, status="success", raw_candidates=raw[:2])],
                max_total=5,
            )
            if aggregate.get("status") != "success" or aggregate.get("diagnostics", {}).get("raw_deduped", 0) <= 0:
                errors.append(f"source_fixture_aggregate_failed:{aggregate}")

            priority_fixture = fixtures.get("priority_watchlist_expected.json") or {}
            priority_rows, debug = monitoring_service.build_priority_watchlist_rows([
                {"chain": "solana", "base_addr": "OOO111111111111111111111111111111111111", "base_symbol": "OOO", "entry_status": "EARLY", "priority_score": 266.57},
                {"chain": "solana", "base_addr": "8X5VQB1111111111111111111111111111O2WN", "base_symbol": "8X5VQB...O2WN", "entry_status": "NO_ENTRY", "priority_score": 47},
            ], [])
            if not priority_rows or int(debug.get("final_priority_rows", 0) or 0) <= 0:
                errors.append(f"priority_fixture_failed:{debug}")
            expected_final = int((priority_fixture.get("debug_subset") or {}).get("final_priority_rows", 0) or 0)
            if expected_final <= 0:
                errors.append("priority_fixture_expected_missing")
        except Exception as exc:
            errors.append(f"golden_fixture_exception:{type(exc).__name__}:{_sanitize(exc)}")
        return _role("ok" if not errors else "golden_fixtures_failed", ok=not errors, errors=errors, fixtures_checked=len(fixture_paths))
    return _capture_role(run)


def _admin_controls_role() -> Dict[str, Any]:
    """Import-safe manual recovery smoke with dry-run only checks."""
    def run() -> Dict[str, Any]:
        errors: List[str] = []
        snapshot = {name: sys.modules.get(name) for name in ("admin_controls", "app", "streamlit")}
        try:
            sys.modules.pop("admin_controls", None)
            sys.modules.pop("app", None)
            sys.modules.pop("streamlit", None)
            admin_controls = importlib.import_module("admin_controls")
            if sys.modules.get("app") is not None:
                errors.append("admin_controls_imported_app")
            if sys.modules.get("streamlit") is not None:
                errors.append("admin_controls_imported_streamlit")
            ctx = {
                "operator_diagnostics": {"status": "ok", "runtime": {"matrix_status": "ok", "read_only_status": "ok", "monitor_cycle_status": "ok"}},
                "read_only_status": "ok",
                "locks": [{"lock_key": "scanner_lock_1", "owner": "worker-a", "expires_epoch": 1}],
            }
            plan = admin_controls.build_admin_recovery_plan(ctx)
            registry = admin_controls.get_admin_action_registry()
            if not isinstance(plan, dict) or plan.get("recommended_order", [])[:2] != ["runtime_matrix", "read_only"]:
                errors.append(f"bad_plan:{plan}")
            clear = admin_controls.validate_admin_action("clear_stale_lock", ctx, confirmation="CLEAR STALE LOCK", dry_run=True)
            if not clear.get("enabled") or not clear.get("dry_run"):
                errors.append(f"clear_dry_run_validation_failed:{clear}")
            for action_id, descriptor in registry.items():
                if descriptor.get("type") == "runbook" and not admin_controls.validate_admin_action(action_id, ctx).get("enabled"):
                    errors.append(f"runbook_disabled:{action_id}")
            return _role("ok" if not errors else "admin_controls_failed", ok=not errors, errors=errors, actions_checked=len(registry))
        finally:
            for name, module in snapshot.items():
                sys.modules.pop(name, None)
                if module is not None:
                    sys.modules[name] = module  # type: ignore[assignment]
    return _capture_role(lambda: _check_no_writes(run))


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



def _external_audit_claims_role() -> Dict[str, Any]:
    """Check stale external-review claims without network, writes, sends, or scans."""
    try:
        from scripts import audit_external_review_claims
        report = audit_external_review_claims.audit_claims()
        claims = report.get("claims") if isinstance(report, dict) else {}
        active_claims = [
            key for key, value in (claims or {}).items()
            if isinstance(value, dict) and value.get("status") == "true"
        ]
        errors = [f"active_external_audit_claim:{key}" for key in active_claims]
        return _role(
            "ok" if not errors else "external_audit_claims_failed",
            ok=not errors,
            errors=errors,
            claims_checked=len(claims or {}),
            active_claims=active_claims,
        )
    except Exception as exc:
        return _role(
            "external_audit_claims_exception",
            ok=False,
            errors=[f"external_audit_claims_exception:{type(exc).__name__}:{_sanitize(exc)}"],
            claims_checked=0,
            active_claims=[],
        )

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
            "app_compat": _app_compat_role(),
            "core_modules": _core_modules_role(),
            "golden_fixtures": _golden_fixtures_role(),
            "admin_controls": _admin_controls_role(),
            "external_audit_claims": _external_audit_claims_role(),
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
