import importlib
import sys
from unittest.mock import Mock


def _load_app(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "d1")
    monkeypatch.setenv("D1_PROXY_URL", "https://d1.example")
    monkeypatch.setenv("D1_PROXY_TOKEN", "secret")
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_d1_request_builds_authorization_header(monkeypatch):
    app = _load_app(monkeypatch)
    mock_resp = Mock(status_code=200, text='{"ok":true}')
    mock_resp.json.return_value = {"ok": True}
    req = Mock(return_value=mock_resp)
    monkeypatch.setattr(app.requests, "request", req)
    app.d1_request("GET", "/v1/storage/k")
    assert req.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"


def test_d1_storage_endpoints(monkeypatch):
    app = _load_app(monkeypatch)
    calls = []

    def fake(method, path, json=None, params=None):
        calls.append((method, path, json, params))
        return {"ok": True, "data": {"rows": []}}

    monkeypatch.setattr(app, "d1_request", fake)
    app.d1_get_storage("k")
    app.d1_put_storage("k", "v")
    app.d1_storage_list_sizes(123)
    assert calls[0][1] == "/v1/storage/k"
    assert calls[1][0] == "PUT" and calls[1][1] == "/v1/storage/k"
    assert calls[2][1] == "/v1/storage-sizes"


def test_d1_storage_key_with_slash_is_encoded(monkeypatch):
    app = _load_app(monkeypatch)
    calls = []

    def fake(method, path, json=None, params=None):
        calls.append(path)
        return {"ok": True, "data": {"found": False}}

    monkeypatch.setattr(app, "d1_request", fake)
    app.d1_get_storage("backup/20260502_215026_monitoring.csv")
    assert calls[0].endswith("backup%2F20260502_215026_monitoring.csv")


def test_d1_select_rows_builds_table_params(monkeypatch):
    app = _load_app(monkeypatch)
    captured = {}

    def fake(method, path, json=None, params=None):
        captured["path"] = path
        captured["params"] = params
        return {"ok": True, "data": {"rows": []}}

    monkeypatch.setattr(app, "d1_request", fake)
    app.d1_select_rows("runtime_state", filters={"state_key": "eq.worker_runtime"}, select="state_json", limit=2)
    assert captured["path"] == "/v1/table/runtime_state"
    assert captured["params"]["filter_col"] == "state_key"


def test_d1_http_errors_safe_status(monkeypatch):
    app = _load_app(monkeypatch)
    mock_resp = Mock(status_code=401, text='{"error":"nope"}')
    mock_resp.json.return_value = {"error": "nope"}
    monkeypatch.setattr(app.requests, "request", Mock(return_value=mock_resp))
    status = app.d1_request("GET", "/v1/storage/k")
    assert status["ok"] is False

    mock_resp2 = Mock(status_code=500, text='{"error":"bad"}')
    mock_resp2.json.return_value = {"error": "bad"}
    monkeypatch.setattr(app.requests, "request", Mock(return_value=mock_resp2))
    status2 = app.d1_request("GET", "/v1/storage/k")
    assert status2["ok"] is False
