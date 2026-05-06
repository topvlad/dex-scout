import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_validator(path: Path):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/validate_app_storage_jsonl.py"), "--in", str(path)],
        capture_output=True,
        text=True,
    )


def test_validator_accepts_valid_jsonl(tmp_path):
    file = tmp_path / "app_storage.jsonl"
    file.write_text(json.dumps({"key": "monitoring.csv", "content": "a,b\n1,2\n"}) + "\n", encoding="utf-8")
    proc = run_validator(file)
    assert proc.returncode == 0
    assert '"record_count": 1' in proc.stdout


def test_validator_rejects_unsafe_keys(tmp_path):
    for key in ["backup/x", "x_snapshot_y", "scanner_lock_abc"]:
        file = tmp_path / f"{key.replace('/', '_')}.jsonl"
        file.write_text(json.dumps({"key": key, "content": "x"}) + "\n", encoding="utf-8")
        proc = run_validator(file)
        assert proc.returncode == 1
        assert "unsafe key" in proc.stderr or "unsafe key" in proc.stdout


def test_validator_rejects_missing_content(tmp_path):
    file = tmp_path / "bad.jsonl"
    file.write_text(json.dumps({"key": "monitoring.csv"}) + "\n", encoding="utf-8")
    proc = run_validator(file)
    assert proc.returncode == 1
    assert "content must be a string" in proc.stderr or "content must be a string" in proc.stdout


def test_workflow_has_manual_secret_action_and_preflight_logic():
    workflow = (REPO_ROOT / ".github/workflows/d1-migration.yml").read_text(encoding="utf-8")
    assert "manual_secret_import_verify" in workflow
    assert "missing_env:APP_STORAGE_JSONL_B64" in workflow
    assert "if [[ \"${action_input}\" == \"manual_secret_import_verify\" ]]; then" in workflow
