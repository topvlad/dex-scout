from fastapi.testclient import TestClient

import tg_webhook


def test_health_stays_200_on_bootstrap_failure(monkeypatch):
    monkeypatch.setattr(
        tg_webhook,
        "BOOTSTRAP_ERROR",
        {"helper": "bootstrap_import", "exception_type": "RuntimeError", "exception_text": "boom"},
    )
    client = TestClient(tg_webhook.app)
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json().get('ok') is True


def test_runtime_returns_500_on_bootstrap_failure(monkeypatch):
    monkeypatch.setattr(
        tg_webhook,
        "BOOTSTRAP_ERROR",
        {"helper": "bootstrap_import", "exception_type": "RuntimeError", "exception_text": "boom"},
    )
    client = TestClient(tg_webhook.app)
    resp = client.get('/runtime')
    assert resp.status_code == 500


def test_tg_digest_returns_500_on_bootstrap_failure(monkeypatch):
    monkeypatch.setattr(
        tg_webhook,
        "BOOTSTRAP_ERROR",
        {"helper": "bootstrap_import", "exception_type": "RuntimeError", "exception_text": "boom"},
    )
    client = TestClient(tg_webhook.app)
    resp = client.get('/tg/digest?key=dummy')
    assert resp.status_code == 500
