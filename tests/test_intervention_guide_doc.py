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
    for role in ["ui_streamlit", "worker", "webhook", "dash_readonly", "core_modules"]:
        assert role in text
