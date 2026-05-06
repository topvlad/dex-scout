# D1 Migration Runbook (PR #2 tools)

This PR adds **migration scripts only** (export/import/verify). It does **not** switch production to D1 automatically.

## A) Cloudflare setup (already done / required)

1. Create D1 database: `dex-scout`.
2. Apply schema from `cloudflare/d1_schema.sql`.
3. Create Worker: `dex-d1`.
4. Bind D1 as `DB` in Worker.
5. Add Worker secret: `D1_PROXY_TOKEN`.
6. Paste `cloudflare/d1_worker.js` into Worker and deploy.
7. Confirm:
   - `GET /health` -> `{"ok":true}`
   - `GET /v1/storage-sizes` **without token** -> `{"ok":false,"code":"unauthorized"}`

## B) GitHub secrets/variables

Secrets:
- `D1_PROXY_TOKEN`

Variables:
- `D1_PROXY_URL=https://dex-d1.web3mon.workers.dev`
- `D1_TIMEOUT_SEC=12`

Do **not** set `STORAGE_BACKEND=d1` until import + verify are complete.

## C) Migration flow

1. Export from current storage:
   ```bash
   python scripts/export_app_storage.py --source supabase --out .d1_export/app_storage.jsonl
   ```
2. Import into D1:
   ```bash
   python scripts/import_app_storage_to_d1.py --in .d1_export/app_storage.jsonl --replace
   ```
3. Verify required keys:
   ```bash
   python scripts/verify_d1_storage.py --expect monitoring.csv,portfolio.csv,tg_state.json,scanner_state.json
   ```
4. Only after verify passes:
   - set GitHub variable `STORAGE_BACKEND=d1`
5. Manual smoke in Runtime Jobs:
   - `maintenance_cycle`
   - `notify_cycle`
   - `monitor_cycle`
   - `scan_cycle`


## E) Run migration through GitHub Actions

1. Go to **Actions → D1 Migration**.
2. Click **Run workflow**.
3. First run: `action=verify_only`.
4. Then run: `action=export_import_verify`, `source=supabase`, `replace=true`.
5. Confirm verify passes.
6. Only then change repository variable: `STORAGE_BACKEND=d1`.
7. Run **Runtime Jobs** manually:
   - `maintenance_cycle`
   - `notify_cycle`
   - `monitor_cycle`
   - `scan_cycle`

## F) Rollback

Set `STORAGE_BACKEND=supabase`.

## Script quick reference

- Export selected keys:
  - `python scripts/export_app_storage.py --source supabase --out .d1_export/app_storage.jsonl`
  - `python scripts/export_app_storage.py --source local --out .d1_export/app_storage.jsonl`
- Import JSONL into D1:
  - `python scripts/import_app_storage_to_d1.py --in .d1_export/app_storage.jsonl`
  - options: `--dry-run`, `--replace`, `--only key1,key2`
- Verify D1 storage:
  - `python scripts/verify_d1_storage.py`
  - `python scripts/verify_d1_storage.py --expect monitoring.csv,portfolio.csv,tg_state.json,scanner_state.json`
