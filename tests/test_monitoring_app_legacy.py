import csv

import monitoring_app


def test_save_monitoring_empty_writes_header(tmp_path, monkeypatch):
    path = tmp_path / 'monitoring.csv'
    monkeypatch.setattr(monitoring_app, 'MONITORING_CSV', str(path))
    monitoring_app.save_monitoring([])
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == monitoring_app.MONITORING_FIELDS


def test_now_utc_uses_timezone_aware_datetime(monkeypatch):
    class FakeDateTime:
        @classmethod
        def now(cls, tz):
            assert tz is monitoring_app.timezone.utc
            class FakeNow:
                def strftime(self, fmt):
                    assert fmt == '%Y-%m-%d %H:%M:%S UTC'
                    return '2026-05-31 00:00:00 UTC'
            return FakeNow()

    monkeypatch.setattr(monitoring_app, 'datetime', FakeDateTime)
    assert monitoring_app.now_utc() == '2026-05-31 00:00:00 UTC'
