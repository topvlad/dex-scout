#!/usr/bin/env python3
"""Emergency runtime reset + workflow_dispatch helper.

Usage:
  SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
  GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo \
  python scripts/emergency_runtime_reset_and_dispatch.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from supabase import Client, create_client

WORKFLOW_FILE = os.getenv("WORKFLOW_FILE", "runtime-jobs.yml")
BLOCKED_PATTERNS = ("duplicate run blocked", "skipped by lock", "exit code 3", "exit_code=3")
RUNTIME_JSON_CANDIDATES = ("state_json", "state", "payload", "data", "runtime_json")


@dataclass
class DispatchResult:
    mode: str
    run_id: int
    html_url: str
    status: str
    conclusion: Optional[str]
    blocked_hits: List[str]


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _sb() -> Client:
    return create_client(_require("SUPABASE_URL"), _require("SUPABASE_SERVICE_ROLE_KEY"))


def _detect_runtime_json_column(sb: Client) -> str:
    for col in RUNTIME_JSON_CANDIDATES:
        try:
            _ = sb.table("runtime_state").select(col).eq("state_key", "worker_runtime").limit(1).execute()
            return col
        except Exception:
            continue
    raise RuntimeError(
        "Unable to detect runtime JSON column in public.runtime_state; tried: "
        + ", ".join(RUNTIME_JSON_CANDIDATES)
    )


def emergency_reset() -> Dict[str, str]:
    sb = _sb()
    runtime_col = _detect_runtime_json_column(sb)

    current_resp = sb.table("runtime_state").select(runtime_col).eq("state_key", "worker_runtime").limit(1).execute()
    data = current_resp.data or []
    current_json = data[0].get(runtime_col) if data else {}
    if not isinstance(current_json, dict):
        current_json = {}
    current_json["last_job_status"] = "finished"

    upd = (
        sb.table("runtime_state")
        .update({runtime_col: current_json})
        .eq("state_key", "worker_runtime")
        .execute()
    )
    runtime_rows_updated = len(upd.data or [])

    # Delete all lock keys with prefix job_mode:
    del_resp = sb.table("locks").delete().like("lock_key", "job_mode:%").execute()
    locks_deleted = len(del_resp.data or [])

    return {
        "runtime_json_column": runtime_col,
        "runtime_rows_updated": str(runtime_rows_updated),
        "locks_deleted": str(locks_deleted),
    }


def _gh_session() -> requests.Session:
    token = _require("GITHUB_TOKEN")
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "dex-scout-ops-script",
        }
    )
    return s


def _repo_parts() -> tuple[str, str]:
    owner, name = _require("GITHUB_REPOSITORY").split("/", 1)
    return owner, name


def _dispatch(mode: str) -> None:
    owner, name = _repo_parts()
    with _gh_session() as s:
        r = s.post(
            f"https://api.github.com/repos/{owner}/{name}/actions/workflows/{WORKFLOW_FILE}/dispatches",
            json={"ref": os.getenv("GITHUB_REF_NAME", "main"), "inputs": {"job_mode": mode}},
            timeout=30,
        )
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"dispatch {mode} failed: {r.status_code} {r.text[:400]}")


def _wait_for_new_dispatch_run(previous_top_id: Optional[int]) -> int:
    owner, name = _repo_parts()
    with _gh_session() as s:
        for _ in range(60):
            r = s.get(
                f"https://api.github.com/repos/{owner}/{name}/actions/workflows/{WORKFLOW_FILE}/runs",
                params={"event": "workflow_dispatch", "per_page": 10},
                timeout=30,
            )
            r.raise_for_status()
            runs = r.json().get("workflow_runs", [])
            if runs:
                top_id = int(runs[0]["id"])
                if previous_top_id is None or top_id != previous_top_id:
                    return top_id
            time.sleep(5)
    raise RuntimeError("Could not observe new workflow_dispatch run")


def _wait_for_completion(run_id: int) -> Dict[str, object]:
    owner, name = _repo_parts()
    with _gh_session() as s:
        for _ in range(120):
            r = s.get(f"https://api.github.com/repos/{owner}/{name}/actions/runs/{run_id}", timeout=30)
            r.raise_for_status()
            run = r.json()
            if str(run.get("status") or "") == "completed":
                return run
            time.sleep(10)
    raise RuntimeError(f"run {run_id} did not complete in time")


def _download_logs_text(run_id: int) -> str:
    owner, name = _repo_parts()
    with _gh_session() as s:
        r = s.get(f"https://api.github.com/repos/{owner}/{name}/actions/runs/{run_id}/logs", timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        chunks: List[str] = []
        for filename in z.namelist():
            if not filename.endswith(".txt"):
                continue
            with z.open(filename) as fh:
                chunks.append(fh.read().decode("utf-8", errors="replace"))
        return "\n".join(chunks)


def _get_latest_dispatch_run_id() -> Optional[int]:
    owner, name = _repo_parts()
    with _gh_session() as s:
        r = s.get(
            f"https://api.github.com/repos/{owner}/{name}/actions/workflows/{WORKFLOW_FILE}/runs",
            params={"event": "workflow_dispatch", "per_page": 1},
            timeout=30,
        )
        r.raise_for_status()
        runs = r.json().get("workflow_runs", [])
        return int(runs[0]["id"]) if runs else None


def run_mode(mode: str) -> DispatchResult:
    previous_top_id = _get_latest_dispatch_run_id()
    _dispatch(mode)
    run_id = _wait_for_new_dispatch_run(previous_top_id)
    run = _wait_for_completion(run_id)
    logs = _download_logs_text(run_id).lower()
    blocked_hits = [p for p in BLOCKED_PATTERNS if p in logs]

    return DispatchResult(
        mode=mode,
        run_id=run_id,
        html_url=str(run.get("html_url") or ""),
        status=str(run.get("status") or ""),
        conclusion=(str(run.get("conclusion")) if run.get("conclusion") is not None else None),
        blocked_hits=blocked_hits,
    )


def main() -> int:
    reset = emergency_reset()
    print("[reset] " + json.dumps(reset, ensure_ascii=False), flush=True)

    results = [run_mode("scan_cycle"), run_mode("all")]
    for result in results:
        print(
            "[dispatch-result] "
            + json.dumps(
                {
                    "mode": result.mode,
                    "run_id": result.run_id,
                    "url": result.html_url,
                    "status": result.status,
                    "conclusion": result.conclusion,
                    "blocked_hits": result.blocked_hits,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    has_blocked = any(r.blocked_hits for r in results)
    has_bad_conclusion = any((r.conclusion or "") != "success" for r in results)
    if has_blocked or has_bad_conclusion:
        print("[result] FAIL", flush=True)
        return 1

    print("[result] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
