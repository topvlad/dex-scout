from pathlib import Path

GUIDE = Path("docs/DEX_SCOUT_INTERVENTION_GUIDE.html")


def test_guide_omits_stale_current_state_phrases():
    text = GUIDE.read_text(encoding="utf-8")
    forbidden = [
        "Supabase primary",
        "Немає тестів",
        "tg_webhook silent import fail",
        "Imports app.py at startup — silent fail risk",
        "Render + Supabase",
    ]
    for phrase in forbidden:
        assert phrase not in text
    assert "score_row() від'ємний" not in text or "historical" in text.lower() or "fixed" in text.lower()
    assert "worker lock_key NameError" not in text or "historical" in text.lower() or "fixed" in text.lower()


def test_guide_contains_current_modules_and_runtime_terms():
    text = GUIDE.read_text(encoding="utf-8")
    required = [
        "scanner_service.py",
        "monitoring_service.py",
        "portfolio_service.py",
        "app_service.py",
        "storage_repository.py",
        "app_runtime_facade.py",
        "notification_core.py",
        "monitoring_core.py",
        "D1",
        "runtime_matrix",
        "live_pulse_candidates.json",
    ]
    for phrase in required:
        assert phrase in text


def test_guide_tracks_ui_operator_observability_current_and_next_plan():
    text = GUIDE.read_text(encoding="utf-8")
    assert "#296 UI runtime diagnostics polish / operator observability" in text
    assert "#297 — runbook/admin controls for safe manual recovery" in text
    assert "runtime_matrix</code>/<code>app_compat</code>/<code>golden_fixtures" in text
