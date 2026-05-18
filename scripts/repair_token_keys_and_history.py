import csv
import shutil
from pathlib import Path
import app

ROOT = Path(__file__).resolve().parents[1]
MONITORING = ROOT / "monitoring.csv"
PORTFOLIO = ROOT / "portfolio.csv"
HISTORY = ROOT / "monitoring_history.csv"
RECO = ROOT / "portfolio_reco_log.csv"


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


active_rows = _read_csv(MONITORING) + _read_csv(PORTFOLIO)
active_by_key = {app.canonical_token_key(r): r for r in active_rows if app.canonical_token_key(r)}
symbol_chain_to_key = {}
for row in active_rows:
    sym = str(row.get("symbol") or "").strip().upper()
    chain = app.normalize_chain_name(str(row.get("chain") or ""))
    if not sym or not chain:
        continue
    symbol_chain_to_key.setdefault((sym, chain), set()).add(app.canonical_token_key(row))

history_rows = _read_csv(HISTORY) + _read_csv(RECO)
repaired = 0
still_unmatched = 0
unmatched_symbols = set()
for row in history_rows:
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
        for fld_src, fld_dst in [("ca", "ca"), ("token_addr", "token_addr"), ("pair_address", "pair_address")]:
            v = str(row.get(fld_dst) or row.get(fld_src) or active.get(fld_src) or active.get(fld_dst) or "").strip()
            if v:
                row[fld_dst] = v
    if row != before:
        repaired += 1
    if not app.canonical_token_key(row):
        still_unmatched += 1
        sym = str(row.get("symbol") or "").strip().upper()
        if sym:
            unmatched_symbols.add(sym)

for path in [HISTORY, RECO]:
    if not path.exists():
        continue
    rows = _read_csv(path)
    _backup(path)
    if not rows:
        continue
    # map back from merged rows by index order
    subset = history_rows[: len(rows)]
    history_rows = history_rows[len(rows):]
    fields = list(rows[0].keys())
    for add in ["canonical_token_key", "ca", "token_addr", "pair_address"]:
        if add not in fields:
            fields.append(add)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(subset)

print(f"active_tokens={len(active_by_key)}")
print(f"history_rows_total={len(history_rows) + repaired + still_unmatched}")
print(f"repaired_rows={repaired}")
print(f"still_unmatched_rows={still_unmatched}")
print(f"sample_unmatched_symbols={sorted(list(unmatched_symbols))[:10]}")
