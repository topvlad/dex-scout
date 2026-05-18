import argparse
import csv
import io
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import app

KEYS_REQUIRED = [
    "monitoring.csv",
    "portfolio.csv",
    "monitoring_history.csv",
]
RECO_KEY = "portfolio_reco_log.csv"


def _rows_to_csv(rows: List[Dict[str, str]], fieldnames: List[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _build_fields(rows: List[Dict[str, str]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    for k in ["canonical_token_key", "ca", "token_addr", "pair_address"]:
        if k not in fields:
            fields.append(k)
    return fields


def _repair_rows(rows: List[Dict[str, str]], active_by_key: Dict[str, Dict[str, str]], symbol_chain_to_key: Dict[Tuple[str, str], set]):
    repaired = 0
    unmatched = 0
    unmatched_symbols = set()
    repaired_rows: List[Dict[str, str]] = []
    for row in rows:
        row = dict(row)
        before = dict(row)
        key = app.canonical_token_key(row)
        if not key:
            sym = str(row.get("symbol") or "").strip().upper()
            chain = app.normalize_chain_name(str(row.get("chain") or ""))
            candidates = symbol_chain_to_key.get((sym, chain), set())
            if len(candidates) == 1:
                key = next(iter(candidates))
                row["canonical_token_key"] = key
        if key and key in active_by_key:
            active = active_by_key[key]
            for fld in ["ca", "token_addr", "pair_address"]:
                v = str(row.get(fld) or active.get(fld) or "").strip()
                if v:
                    row[fld] = v
        if row != before:
            repaired += 1
        if not app.canonical_token_key(row):
            unmatched += 1
            sym = str(row.get("symbol") or "").strip().upper()
            if sym:
                unmatched_symbols.add(sym)
        repaired_rows.append(row)
    return repaired_rows, repaired, unmatched, unmatched_symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write repaired rows to storage")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args()

    do_apply = bool(args.apply)
    if not args.apply and not args.dry_run:
        args.dry_run = True

    if app.STORAGE_BACKEND != "d1":
        raise RuntimeError(f"STORAGE_BACKEND must be d1 for this task (got {app.STORAGE_BACKEND!r})")
    if not str(os.getenv("D1_PROXY_URL", "")).strip() or not str(os.getenv("D1_PROXY_TOKEN", "")).strip():
        raise RuntimeError("D1 env missing: D1_PROXY_URL/D1_PROXY_TOKEN are required")

    storage: Dict[str, List[Dict[str, str]]] = {}
    for key in KEYS_REQUIRED + [RECO_KEY]:
        storage[key] = app.load_csv(key)

    if not storage[KEYS_REQUIRED[0]] and not storage[KEYS_REQUIRED[1]]:
        raise RuntimeError("No active tokens found in monitoring.csv/portfolio.csv from D1")

    active_rows = storage["monitoring.csv"] + storage["portfolio.csv"]
    active_by_key = {app.canonical_token_key(r): r for r in active_rows if app.canonical_token_key(r)}
    symbol_chain_to_key: Dict[Tuple[str, str], set] = {}
    for row in active_rows:
        sym = str(row.get("symbol") or "").strip().upper()
        chain = app.normalize_chain_name(str(row.get("chain") or ""))
        key = app.canonical_token_key(row)
        if sym and chain and key:
            symbol_chain_to_key.setdefault((sym, chain), set()).add(key)

    history_keys = ["monitoring_history.csv"] + ([RECO_KEY] if storage[RECO_KEY] else [])
    backups_written: List[str] = []
    repaired_total = 0
    unmatched_total = 0
    unmatched_symbols = set()
    changed_keys: List[str] = []

    for key in history_keys:
        src = storage[key]
        repaired_rows, repaired, unmatched, symbols = _repair_rows(src, active_by_key, symbol_chain_to_key)
        repaired_total += repaired
        unmatched_total += unmatched
        unmatched_symbols.update(symbols)

        if not do_apply:
            continue

        if repaired_rows != src:
            if src and not repaired_rows:
                raise RuntimeError(f"Safety abort: refusing to wipe non-empty key {key}")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_key = f"backup/{ts}_{key}"
            app.storage_write_text(backup_key, _rows_to_csv(src, _build_fields(src)))
            backups_written.append(backup_key)
            app.save_csv(key, repaired_rows, _build_fields(repaired_rows))
            changed_keys.append(key)

    verified = True
    if do_apply and changed_keys:
        for key in changed_keys:
            reloaded = app.load_csv(key)
            if not reloaded:
                verified = False
                break
            repaired_rows, _, _, _ = _repair_rows(reloaded, active_by_key, symbol_chain_to_key)
            if repaired_rows != reloaded:
                verified = False
                break

    if do_apply and changed_keys and not verified:
        raise RuntimeError("Verification failed: verify read does not include repaired rows")

    history_total = len(storage["monitoring_history.csv"]) + len(storage[RECO_KEY])
    print(f"active_tokens={len(active_by_key)}")
    print(f"monitoring_rows={len(storage['monitoring.csv'])}")
    print(f"portfolio_rows={len(storage['portfolio.csv'])}")
    print(f"history_rows_total={history_total}")
    print(f"repaired_rows={repaired_total}")
    print(f"still_unmatched_rows={unmatched_total}")
    print(f"sample_unmatched_symbols={sorted(list(unmatched_symbols))[:10]}")
    print(f"backups_written={len(backups_written)}")
    print(f"verified={str(verified).lower() if do_apply else 'false'}")
    print(f"mode={'apply' if do_apply else 'dry-run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
