#!/usr/bin/env python3
"""Audit app.py compatibility shell for duplicate/stale glue.

The audit is intentionally static and side-effect free: it parses app.py with
AST, reports duplicate top-level functions as failures, and reports service-
owned helper bodies as warning markers for cleanup follow-up.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_PATH = REPO_ROOT / "app.py"

KNOWN_SERVICE_HELPERS = {
    "normalize_portfolio_action": "portfolio_service",
    "is_material_portfolio_action": "portfolio_service",
    "classify_portfolio_row": "portfolio_service",
    "normalize_monitoring_status": "monitoring_service",
    "classify_monitoring_row": "monitoring_service",
    "build_live_pulse_payload": "scanner_service",
    "normalize_scanner_candidates": "scanner_service",
    "filter_live_pulse_candidates": "scanner_service",
    "plan_live_pulse_refill": "scanner_service",
    "build_notification_summary": "app_service",
}

STALE_MARKER_NAMES = set(KNOWN_SERVICE_HELPERS)
SECRET_ENV_HINTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CHAT_ID", "D1_PROXY_URL", "SUPABASE_URL")


def _redact(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    for env_name, secret in os.environ.items():
        if not secret or not any(hint in env_name.upper() for hint in SECRET_ENV_HINTS):
            continue
        text = text.replace(str(secret), "[REDACTED]")
    return text[:limit]


def _function_defs(tree: ast.Module) -> List[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _delegates_to_module(func: ast.FunctionDef | ast.AsyncFunctionDef, module_name: str) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == module_name:
                return True
    return False


def audit_app_path(path: str | Path = DEFAULT_APP_PATH) -> Dict[str, Any]:
    app_path = Path(path)
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))
    funcs = _function_defs(tree)
    by_name: Dict[str, List[int]] = defaultdict(list)
    for func in funcs:
        by_name[func.name].append(int(func.lineno))

    duplicates = [
        {"name": name, "lines": lines}
        for name, lines in sorted(by_name.items())
        if len(lines) > 1
    ]

    stale_markers: List[Dict[str, Any]] = []
    for func in funcs:
        owner = KNOWN_SERVICE_HELPERS.get(func.name)
        if not owner:
            continue
        if not _delegates_to_module(func, owner):
            stale_markers.append({
                "name": func.name,
                "line": int(func.lineno),
                "service_owner": owner,
                "severity": "warning",
                "reason": "service_owned_helper_body_not_delegating",
            })

    checked_functions = [name for name in sorted(STALE_MARKER_NAMES) if name in by_name]
    ok = not duplicates
    return {
        "ok": ok,
        "duplicates": duplicates,
        "stale_markers": stale_markers,
        "checked_functions": checked_functions,
    }


def sanitized_audit(path: str | Path = DEFAULT_APP_PATH) -> Dict[str, Any]:
    raw = audit_app_path(path)
    encoded = json.dumps(raw, sort_keys=True, default=str)
    return json.loads(_redact(encoded, limit=max(240, len(encoded))))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit app.py compatibility wrappers")
    parser.add_argument("--app", default=str(DEFAULT_APP_PATH), help="Path to app.py")
    parser.add_argument("--json", dest="json_path", default="", help="Write audit JSON to this path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = sanitized_audit(args.app)
    text = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)
    print(text)
    if args.json_path:
        out_path = Path(args.json_path)
        if out_path.is_dir():
            raise SystemExit("--json must be a file path")
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
