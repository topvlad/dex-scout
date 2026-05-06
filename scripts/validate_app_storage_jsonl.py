#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


UNSAFE_PATTERNS = (
    "backup/",
    "backup_",
    "scanner_lock_",
)


def is_safe_key(key: str) -> bool:
    k = str(key or "").strip().lower()
    if not k:
        return False
    if any(k.startswith(prefix) for prefix in UNSAFE_PATTERNS):
        return False
    if k.endswith(".bak"):
        return False
    if "snapshot" in k:
        return False
    if "tmp" in k or "debug" in k:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate app_storage JSONL for safe D1 import")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL path")
    args = parser.parse_args()

    path = Path(args.input_path)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit("input JSONL file missing or empty")

    keys = []
    total_bytes = 0
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                raise SystemExit(f"line {lineno}: empty lines are not allowed")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"line {lineno}: invalid JSON ({exc})") from exc

            key = rec.get("key")
            content = rec.get("content")
            if not isinstance(key, str) or not key.strip():
                raise SystemExit(f"line {lineno}: key must be a non-empty string")
            if not is_safe_key(key):
                raise SystemExit(f"line {lineno}: unsafe key '{key}'")
            if not isinstance(content, str):
                raise SystemExit(f"line {lineno}: content must be a string")

            keys.append(key)
            total_bytes += len(content.encode("utf-8"))

    if not keys:
        raise SystemExit("no records found")

    print(json.dumps({
        "keys": keys,
        "record_count": len(keys),
        "total_bytes": total_bytes,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
