#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import requests


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Import app_storage JSONL into D1 via proxy")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    base_url = os.getenv("D1_PROXY_URL", "").rstrip("/")
    token = os.getenv("D1_PROXY_TOKEN", "")
    if not base_url or not token:
        raise SystemExit("D1_PROXY_URL and D1_PROXY_TOKEN are required")

    headers = {"Authorization": f"Bearer {token}"}
    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()

    auth_check = requests.get(f"{base_url}/v1/storage-sizes", headers=headers, timeout=20)
    if auth_check.status_code == 401:
        raise SystemExit("Unauthorized: check D1_PROXY_TOKEN")
    auth_check.raise_for_status()

    rows = load_jsonl(Path(args.input_path))
    only = {k.strip() for k in args.only.split(",") if k.strip()}

    imported_keys: List[str] = []
    skipped_keys: List[str] = []
    failed_keys: List[str] = []
    total_bytes = 0

    for row in rows:
        key = row.get("key", "")
        content = row.get("content", "")
        expected_bytes = int(row.get("bytes", len(str(content).encode("utf-8"))))

        if only and key not in only:
            skipped_keys.append(key)
            continue

        if args.dry_run:
            imported_keys.append(key)
            total_bytes += expected_bytes
            continue

        enc_key = quote(key, safe="")
        put_resp = requests.put(
            f"{base_url}/v1/storage/{enc_key}",
            headers=headers,
            json={"value": content},
            timeout=20,
        )
        if put_resp.status_code >= 400:
            failed_keys.append(key)
            continue

        size_resp = requests.get(f"{base_url}/v1/storage-size/{enc_key}", headers=headers, timeout=20)
        if size_resp.status_code >= 400:
            failed_keys.append(key)
            continue
        payload = size_resp.json()
        actual_bytes = int(((payload.get("data") or {}).get("bytes") or 0))
        if actual_bytes != expected_bytes:
            failed_keys.append(key)
            continue

        imported_keys.append(key)
        total_bytes += expected_bytes
        if not args.replace:
            print(f"upserted key: {key}")

    print(json.dumps({
        "imported_keys": imported_keys,
        "skipped_keys": skipped_keys,
        "failed_keys": failed_keys,
        "total_bytes": total_bytes,
    }, ensure_ascii=False, indent=2))

    return 1 if failed_keys else 0


if __name__ == "__main__":
    raise SystemExit(main())
