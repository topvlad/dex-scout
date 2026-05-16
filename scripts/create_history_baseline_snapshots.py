#!/usr/bin/env python3
from datetime import datetime, timezone
import app


def _active(rows):
    return [r for r in rows if str(r.get("active", "1")).strip() == "1"]


def main() -> int:
    now_ts = datetime.now(timezone.utc)
    app.ensure_storage()
    hist_rows = app.load_csv(app.MON_HISTORY_CSV, app.HIST_FIELDS)
    app.backup_csv_snapshot(app.MON_HISTORY_CSV, hist_rows, app.HIST_FIELDS)
    monitoring_rows = _active(app.load_monitoring())
    portfolio_rows = _active(app.load_portfolio())
    added = 0
    existing = 0
    failed = 0
    for row in monitoring_rows:
        ok = app.append_token_history_snapshot(row, source="monitoring", now_ts=now_ts)
        if ok:
            added += 1
        else:
            existing += 1
    for row in portfolio_rows:
        ok = app.append_token_history_snapshot(row, source="portfolio", now_ts=now_ts)
        if ok:
            added += 1
        else:
            existing += 1
    try:
        app.flush_monitoring_history_buffer(force=True)
    except Exception:
        failed += 1
    print(f"baseline snapshots: added={added} existing={existing} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
