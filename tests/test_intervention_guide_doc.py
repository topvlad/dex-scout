from pathlib import Path


def test_intervention_guide_exists_and_contains_current_terms():
    html = Path("docs/DEX_SCOUT_INTERVENTION_GUIDE.html")
    assert html.exists()
    text = html.read_text(encoding="utf-8")
    for needle in ["app_runtime_facade.py", "D1", "#276", "no secrets"]:
        assert needle in text


def test_intervention_guide_does_not_contain_stale_claims_or_secret_values(monkeypatch):
    text = Path("docs/DEX_SCOUT_INTERVENTION_GUIDE.html").read_text(encoding="utf-8")
    for env_name in ["TG_BOT_TOKEN", "D1_PROXY_TOKEN", "SUPABASE_SERVICE_ROLE_KEY"]:
        value = "actual-secret-" + env_name
        monkeypatch.setenv(env_name, value)
        assert value not in text
    for stale in ["Немає жодних тестів", "Supabase primary"]:
        assert stale not in text


def test_intervention_guide_tracks_runtime_matrix_roadmap():
    text = Path("docs/DEX_SCOUT_INTERVENTION_GUIDE.html").read_text(encoding="utf-8")
    assert "runtime facade <code>job_mode</code> kwarg collision fixed" in text
    assert "#285" in text and "Runtime no-fail matrix + service facade guardrails" in text
    assert "#286" in text and "service facade / business glue reduction" in text
    for role in ["ui_streamlit", "worker", "webhook", "dash_readonly", "app_compat", "core_modules"]:
        assert role in text


def test_intervention_guide_tracks_app_compat_audit_current_scope():
    text = Path("docs/DEX_SCOUT_INTERVENTION_GUIDE.html").read_text(encoding="utf-8")
    assert "#292" in text and "app.py compatibility wrapper audit" in text
    assert "#293" in text and "scanner source adapter cleanup / ingest diagnostics" in text
    assert "app.py is now compatibility shell, not business owner" in text
    for stale in [
        "Немає жодних тестів",
        "Supabase primary",
        "tg_webhook silent import fail",
        "score_row negative",
        "worker lock_key NameError",
    ]:
        assert stale not in text
