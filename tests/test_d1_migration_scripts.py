import json
import os
import subprocess
import sys
from pathlib import Path

import scripts.export_app_storage as ex
import scripts.import_app_storage_to_d1 as importer
import scripts.verify_d1_storage as verifier

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exclude_backup_snapshot_scanner_lock_keys():
    assert ex.is_safe_key("monitoring.csv")
    assert not ex.is_safe_key("backup/a.csv")
    assert not ex.is_safe_key("backup_2026.csv")
    assert not ex.is_safe_key("x_snapshot_y.json")
    assert not ex.is_safe_key("scanner_lock_abc")


def test_jsonl_export_format_valid(tmp_path):
    out = tmp_path / "out.jsonl"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "monitoring.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/export_app_storage.py"), "--source", "local", "--out", str(out)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows[0]["key"] == "monitoring.csv"
    assert rows[0]["content"] == "a,b\n1,2\n"


def test_supabase_export_uses_key_content_columns(monkeypatch):
    captured = []

    class Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, headers=None, params=None, timeout=None):
        captured.append(params)
        if params.get("select") == "key":
            return Resp([{"key": "monitoring.csv"}])
        return Resp([{"content": "abc"}])

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret")
    monkeypatch.setattr(ex.requests, "get", fake_get)

    records, missing, skipped = ex.export_records("supabase", include_all_safe=True)

    assert any(p.get("select") == "key" and "storage_key" not in p for p in captured)
    assert any(p.get("select") == "content" and p.get("key") == "eq.monitoring.csv" for p in captured)
    assert records[0]["content"] == "abc"
    assert missing == []
    assert skipped == []


def test_import_storage_url_encodes_key_with_slash():
    assert "a%2Fb.csv" in importer.storage_url("https://d1.example", "a/b.csv")
    assert "a%2Fb.csv" in importer.storage_size_url("https://d1.example", "a/b.csv")


def test_import_script_sends_content_and_reads_top_level_bytes(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"key": "a/b.csv", "content": "x", "bytes": 1}) + "\n", encoding="utf-8")
    helper = tmp_path / "run_import_test.py"
    helper.write_text(
        f"""
import runpy, requests, sys, json
calls = []

class R:
    def __init__(self, code=200, p=None):
        self.status_code = code
        self._p = p or {{"ok": True, "rows": []}}

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def g(url, **kw):
    calls.append({{"method": "GET", "url": url, "json": kw.get("json")}})
    if url.endswith('/health'):
        return R()
    if '/v1/storage-sizes' in url:
        return R(200, {{"ok": True, "rows": []}})
    if '/v1/storage-size/' in url:
        return R(200, {{"ok": True, "key": "a/b.csv", "bytes": 1}})
    return R()


def p(url, **kw):
    calls.append({{"method": "PUT", "url": url, "json": kw.get("json")}})
    return R()

requests.get = g
requests.put = p
sys.argv = ['x', '--in', r'{path}', '--replace']
try:
    runpy.run_path(r'{REPO_ROOT / 'scripts/import_app_storage_to_d1.py'}', run_name='__main__')
except SystemExit as e:
    if int(e.code or 0) != 0:
        raise
print(json.dumps(calls))
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"D1_PROXY_URL": "https://d1.example", "D1_PROXY_TOKEN": "t"})
    proc = subprocess.run([sys.executable, str(helper)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    assert "%2F" in proc.stdout
    assert '"content": "x"' in proc.stdout
    assert '"value"' not in proc.stdout


def test_verify_storage_size_url_encodes_key_with_slash():
    assert "a%2Fb.csv" in verifier.storage_size_url("https://d1.example", "a/b.csv")


def test_verify_script_detects_missing_expected_key(tmp_path):
    helper = tmp_path / "run_verify_test.py"
    helper.write_text(
        f"""
import runpy, requests, sys

class R:
    def __init__(self, code=200, p=None):
        self.status_code = code
        self._p = p or {{"ok": True, "rows": []}}

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def g(url, **kw):
    if url.endswith('/health'):
        return R()
    if '/v1/storage-sizes' in url:
        return R(200, {{"ok": True, "rows": []}})
    return R(200, {{"ok": True, "key": "monitoring.csv", "bytes": 0}})

requests.get = g
sys.argv = ['x', '--expect', 'monitoring.csv']
runpy.run_path(r'{REPO_ROOT / 'scripts/verify_d1_storage.py'}', run_name='__main__')
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"D1_PROXY_URL": "https://d1.example", "D1_PROXY_TOKEN": "t"})
    proc = subprocess.run([sys.executable, str(helper)], env=env)
    assert proc.returncode == 1


def test_verify_script_parses_top_level_rows_and_bytes(tmp_path):
    helper = tmp_path / "run_verify_present_test.py"
    helper.write_text(
        f"""
import runpy, requests, sys

class R:
    def __init__(self, code=200, p=None):
        self.status_code = code
        self._p = p or {{"ok": True}}

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def g(url, **kw):
    if url.endswith('/health'):
        return R()
    if '/v1/storage-sizes' in url:
        return R(200, {{"ok": True, "rows": [{{"key": "monitoring.csv", "bytes": 3}}]}})
    return R(200, {{"ok": True, "key": "monitoring.csv", "bytes": 3}})

requests.get = g
sys.argv = ['x', '--expect', 'monitoring.csv']
runpy.run_path(r'{REPO_ROOT / 'scripts/verify_d1_storage.py'}', run_name='__main__')
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"D1_PROXY_URL": "https://d1.example", "D1_PROXY_TOKEN": "t"})
    proc = subprocess.run([sys.executable, str(helper)], env=env)
    assert proc.returncode == 0
