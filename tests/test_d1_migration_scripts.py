import json
import os
import subprocess
import sys
from pathlib import Path

import scripts.export_app_storage as ex

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


def test_import_script_url_encodes_key_with_slash(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"key": "a/b.csv", "content": "x", "bytes": 1}) + "\n", encoding="utf-8")
    helper = tmp_path / "run_import_test.py"
    helper.write_text(
        f"""
import runpy, requests, sys
calls = []

class R:
    def __init__(self, code=200, p=None):
        self.status_code = code
        self._p = p or {{"ok": True, "data": {{"bytes": 1, "found": True, "rows": []}}}}

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def g(url, **kw):
    calls.append(url)
    if url.endswith('/health'):
        return R()
    if '/v1/storage-sizes' in url:
        return R()
    return R()


def p(url, **kw):
    calls.append(url)
    return R()

requests.get = g
requests.put = p
sys.argv = ['x', '--in', r'{path}']
try:
    runpy.run_path(r'{REPO_ROOT / 'scripts/import_app_storage_to_d1.py'}', run_name='__main__')
except SystemExit:
    pass
print('\\n'.join(calls))
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"D1_PROXY_URL": "https://d1.example", "D1_PROXY_TOKEN": "t"})
    proc = subprocess.run([sys.executable, str(helper)], capture_output=True, text=True, env=env)
    assert "%2F" in proc.stdout


def test_verify_script_detects_missing_expected_key(tmp_path):
    helper = tmp_path / "run_verify_test.py"
    helper.write_text(
        f"""
import runpy, requests, sys

class R:
    def __init__(self, code=200, p=None):
        self.status_code = code
        self._p = p or {{"ok": True, "data": {{"rows": []}}}}

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def g(url, **kw):
    if url.endswith('/health'):
        return R()
    if '/v1/storage-sizes' in url:
        return R(200, {{"ok": True, "data": {{"rows": []}}}})
    return R(200, {{"ok": True, "data": {{"found": False, "bytes": 0}}}})

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
