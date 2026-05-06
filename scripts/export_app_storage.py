#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

REQUIRED_KEYS = [
    "monitoring.csv",
    "portfolio.csv",
    "monitoring_history.csv",
    "portfolio_reco_log.csv",
    "scanner_state.json",
    "tg_state.json",
    "pulse_history_compact.json",
]
OPTIONAL_KEYS = [
    "signal_journal.csv",
    "portfolio_action_journal.csv",
    "health_override_journal.csv",
    "suppressed_tokens.json",
]
LOCAL_KEY_MAP = {
    "monitoring.csv": "data/monitoring.csv",
    "portfolio.csv": "data/portfolio.csv",
    "monitoring_history.csv": "data/monitoring_history.csv",
    "portfolio_reco_log.csv": "data/portfolio_reco_log.csv",
    "scanner_state.json": "data/scanner_state.json",
    "tg_state.json": "data/tg_state.json",
    "pulse_history_compact.json": "data/pulse_history_compact.json",
    "signal_journal.csv": "data/signal_journal.csv",
    "portfolio_action_journal.csv": "data/portfolio_action_journal.csv",
    "health_override_journal.csv": "data/health_override_journal.csv",
    "suppressed_tokens.json": "data/suppressed_tokens.json",
}


def is_safe_key(key: str) -> bool:
    k = key.lower()
    if k.startswith("backup/") or k.startswith("backup_"):
        return False
    if k.endswith(".bak"):
        return False
    if "snapshot" in k:
        return False
    if k.startswith("scanner_lock_"):
        return False
    if "tmp" in k or "debug" in k:
        return False
    return True


def fetch_supabase_key(base_url: str, token: str, key: str) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/rest/v1/app_storage"
    headers = {"apikey": token, "Authorization": f"Bearer {token}"}
    params = {"select": "storage_value", "storage_key": f"eq.{key}", "limit": "1"}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    return rows[0].get("storage_value")


def list_supabase_keys(base_url: str, token: str) -> List[str]:
    url = f"{base_url.rstrip('/')}/rest/v1/app_storage"
    headers = {"apikey": token, "Authorization": f"Bearer {token}"}
    params = {"select": "storage_key", "order": "storage_key.asc", "limit": "10000"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return [r.get("storage_key", "") for r in resp.json() if r.get("storage_key")]


def export_records(source: str, include_all_safe: bool) -> Tuple[List[Dict], List[str], List[str]]:
    wanted_keys = list(dict.fromkeys(REQUIRED_KEYS + OPTIONAL_KEYS))
    exported: List[Dict] = []
    missing: List[str] = []
    skipped: List[str] = []

    if source == "supabase":
        supabase_url = os.getenv("SUPABASE_URL", "")
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not supabase_url or not service_key:
            raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for --source supabase")

        keys_to_pull: Iterable[str] = wanted_keys
        if include_all_safe:
            listed = list_supabase_keys(supabase_url, service_key)
            keys_to_pull = sorted(set(k for k in listed if is_safe_key(k)))
            skipped.extend(sorted(k for k in listed if not is_safe_key(k)))

        for key in keys_to_pull:
            if not is_safe_key(key):
                skipped.append(key)
                continue
            value = fetch_supabase_key(supabase_url, service_key, key)
            if value is None:
                missing.append(key)
                continue
            exported.append({"key": key, "content": value, "bytes": len(value.encode("utf-8")), "source": "supabase"})
    else:
        for key in wanted_keys:
            if not is_safe_key(key):
                skipped.append(key)
                continue
            path = Path(LOCAL_KEY_MAP.get(key, f"data/{key}"))
            if not path.exists():
                missing.append(key)
                continue
            content = path.read_text(encoding="utf-8")
            exported.append({"key": key, "content": content, "bytes": len(content.encode("utf-8")), "source": "local"})

    return exported, missing, sorted(set(skipped))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export selected app_storage keys to JSONL")
    parser.add_argument("--source", choices=["supabase", "local"], required=True)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--include-all-safe", action="store_true")
    args = parser.parse_args()

    records, missing, skipped = export_records(args.source, args.include_all_safe)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_bytes = sum(r["bytes"] for r in records)
    print(json.dumps({
        "exported_keys": [r["key"] for r in records],
        "missing_keys": missing,
        "skipped_keys": skipped,
        "total_bytes": total_bytes,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
