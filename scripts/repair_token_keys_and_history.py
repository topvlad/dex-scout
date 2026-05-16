import csv, shutil
from pathlib import Path
import app

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / 'monitoring.csv', ROOT / 'portfolio.csv']

for fp in FILES:
    if not fp.exists():
        continue
    bak = fp.with_suffix(fp.suffix + '.bak')
    shutil.copy2(fp, bak)
    rows = list(csv.DictReader(fp.open()))
    changed = 0
    for r in rows:
        key = app.canonical_token_key(r)
        if r.get('canonical_token_key') != key:
            r['canonical_token_key'] = key
            changed += 1
    if rows:
        fields = list(rows[0].keys())
        if 'canonical_token_key' not in fields:
            fields.append('canonical_token_key')
        with fp.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    print(f'{fp.name}: changed_rows={changed} backup={bak.name}')
