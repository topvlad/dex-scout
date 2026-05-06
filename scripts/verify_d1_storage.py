#!/usr/bin/env python3
import argparse
import os
from urllib.parse import quote

import requests


def _bytes_to_mb(n: int) -> float:
    return round(n / (1024 * 1024), 4)


def storage_size_url(base_url: str, key: str) -> str:
    return f"{base_url.rstrip('/')}/v1/storage-size/{quote(key, safe='')}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify D1 app_storage keys")
    parser.add_argument("--expect", default="")
    args = parser.parse_args()

    base_url = os.getenv("D1_PROXY_URL", "").rstrip("/")
    token = os.getenv("D1_PROXY_TOKEN", "")
    if not base_url or not token:
        raise SystemExit("D1_PROXY_URL and D1_PROXY_TOKEN are required")

    headers = {"Authorization": f"Bearer {token}"}

    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()

    sizes_resp = requests.get(f"{base_url}/v1/storage-sizes", headers=headers, timeout=20)
    if sizes_resp.status_code == 401:
        raise SystemExit("Unauthorized: check D1_PROXY_TOKEN")
    sizes_resp.raise_for_status()

    sizes_payload = sizes_resp.json()
    rows = sizes_payload.get("rows") or []
    rows = sorted(rows, key=lambda x: int(x.get("bytes") or 0), reverse=True)

    print("Top keys by size:")
    for row in rows[:20]:
        key = row.get("key", "")
        b = int(row.get("bytes") or 0)
        print(f"- {key}: {b} bytes ({_bytes_to_mb(b)} MB)")

    missing = []
    expected = [k.strip() for k in args.expect.split(",") if k.strip()]
    if expected:
        print("\nExpected keys:")
    for key in expected:
        resp = requests.get(storage_size_url(base_url, key), headers=headers, timeout=20)
        if resp.status_code >= 400:
            missing.append(key)
            print(f"- MISSING {key}")
            continue
        payload = resp.json()
        b = int(payload.get("bytes") or 0)
        if b <= 0:
            missing.append(key)
            print(f"- MISSING {key}")
            continue
        print(f"- PRESENT {key}: {b} bytes ({_bytes_to_mb(b)} MB)")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
