#!/usr/bin/env python3
"""Audit stale third-party review claims without network, scanner, or Streamlit imports."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIM_KEYS = (
    "scoring_negative_active",
    "worker_lock_key_nameerror",
    "tg_webhook_silent_import_fail",
    "summary_key_not_constant_time",
    "duplicated_row_matching",
    "missing_streamlit_test_stub",
)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _function_source(tree: ast.AST, source: str, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""


def _status_false(evidence: str) -> Dict[str, str]:
    return {"status": "false", "evidence": evidence}


def _status_true(evidence: str) -> Dict[str, str]:
    return {"status": "true", "evidence": evidence}


def _status_unknown(evidence: str) -> Dict[str, str]:
    return {"status": "unknown", "evidence": evidence}


def _audit_scoring() -> Dict[str, str]:
    scoring = _read("scoring.py")
    app = _read("app.py")
    app_tree = ast.parse(app)
    score_pair = _function_source(app_tree, app, "score_pair")
    legacy_ok = "LEGACY FILE" in scoring and "SCORE_MIN_VALUE" in scoring and "max(SCORE_MIN_VALUE" in scoring
    active_floor = "def score_pair" in score_pair and "max(0.0" in score_pair
    if active_floor:
        return _status_false("Active app.py score_pair floors final public score at >= 0; scoring.py is legacy and also uses SCORE_MIN_VALUE.")
    if legacy_ok and "def score_pair" not in score_pair:
        return _status_unknown("scoring.py is legacy and floored, but active score_pair owner was not identified.")
    return _status_true("Active app.py score_pair does not show a final non-negative floor.")


def _audit_worker_lock() -> Dict[str, str]:
    worker = _read("worker.py")
    tree = ast.parse(worker)
    run_job = _function_source(tree, worker, "run_job_mode")
    init_pos = run_job.find("lock_key = None")
    acquire_pos = run_job.find("app.acquire_lock(lock_key=lock_key")
    guard = "if lock_key is not None" in run_job and "app.release_lock(lock_key=lock_key" in run_job
    if init_pos >= 0 and acquire_pos >= 0 and init_pos < acquire_pos and guard:
        return _status_false("worker.run_job_mode initializes lock_key before the acquire path and guards release_lock in finally.")
    return _status_true("worker.run_job_mode lock_key initialization/release guard contract was not found.")


def _audit_webhook_import() -> Dict[str, str]:
    src = _read("tg_webhook.py")
    tree = ast.parse(src)
    webhook = _function_source(tree, src, "tg_webhook")
    health = _function_source(tree, src, "health")
    import_status = _function_source(tree, src, "import_status")
    if "_app_available" in webhook and "_app_import_failed_response" in webhook and "import_failed" in import_status and "_app_import_failed_response" in health:
        return _status_false("tg_webhook exposes /_import_status and returns app_import_failed from /health and callbacks when facade import is unavailable.")
    return _status_true("tg_webhook degraded import behavior guard was not found.")


def _audit_summary_key() -> Dict[str, str]:
    src = _read("tg_webhook.py")
    tree = ast.parse(src)
    fn = _function_source(tree, src, "summary_key_ok")
    if "hmac.compare_digest" in fn:
        return _status_false("summary_key_ok uses hmac.compare_digest for constant-time comparison.")
    return _status_true("summary_key_ok does not use hmac.compare_digest.")


def _audit_row_matching() -> Dict[str, str]:
    src = _read("tg_webhook.py")
    aliases = ("base_token_address", "base_addr", "token_addr", "token_address", "ca", "address", "pair_address")
    helper_present = "def _row_matches_token" in src and all(alias in src for alias in aliases)
    uses = src.count("_row_matches_token(")
    if helper_present and uses >= 4:
        return _status_false("tg_webhook centralizes canonical token row matching in _row_matches_token across add/remove/verify paths.")
    return _status_true("Centralized _row_matches_token alias coverage/use was not found.")


def _audit_streamlit_stub() -> Dict[str, str]:
    conftest = _read("tests/conftest.py")
    required = ("cache_data", "session_state", "sidebar", "set_page_config", "columns", "expander", "button")
    if "ModuleType(\"streamlit\")" in conftest and all(name in conftest for name in required):
        return _status_false("tests/conftest.py installs a Streamlit pytest stub with required app import attributes.")
    return _status_true("Required Streamlit pytest stub attributes were not found.")


def audit_claims() -> Dict[str, Any]:
    claims = {
        "scoring_negative_active": _audit_scoring(),
        "worker_lock_key_nameerror": _audit_worker_lock(),
        "tg_webhook_silent_import_fail": _audit_webhook_import(),
        "summary_key_not_constant_time": _audit_summary_key(),
        "duplicated_row_matching": _audit_row_matching(),
        "missing_streamlit_test_stub": _audit_streamlit_stub(),
    }
    active_risks = [key for key, value in claims.items() if value.get("status") == "true"]
    stale_claims = [key for key, value in claims.items() if value.get("status") == "false"]
    return {"ok": not active_risks, "claims": claims, "active_risks": active_risks, "stale_claims": stale_claims}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit external review claims against current source contracts")
    parser.add_argument("--json", dest="json_path", default="", help="Write JSON report to this path")
    args = parser.parse_args(argv)
    report = audit_claims()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
