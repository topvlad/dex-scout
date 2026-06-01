import asyncio
import inspect
import importlib.util
import os
import sys
import types
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STUBS = ROOT / "tests" / "stubs"
for _path in (str(ROOT), str(STUBS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
_parts = [p for p in _existing_pythonpath.split(os.pathsep) if p]
_prefixes = [str(STUBS), str(ROOT)]
for _prefix in reversed(_prefixes):
    if _prefix not in _parts:
        _existing_pythonpath = _prefix + (os.pathsep + _existing_pythonpath if _existing_pythonpath else "")
        _parts.insert(0, _prefix)
os.environ["PYTHONPATH"] = _existing_pythonpath


def _install_requests_stub() -> None:
    """Install a tiny requests stub when the optional dependency is absent."""
    if "requests" in sys.modules:
        return
    if importlib.util.find_spec("requests") is not None:
        return

    class _Response:
        status_code = 200
        text = ""
        content = b""
        headers = {}

        def __init__(self, json_data=None, status_code=200, text=""):
            self._json_data = {} if json_data is None else json_data
            self.status_code = status_code
            self.text = text
            self.content = text.encode()
            self.headers = {}

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def _request(*args, **kwargs):
        return _Response()

    class _Session:
        def request(self, *args, **kwargs):
            return _request(*args, **kwargs)
        def get(self, *args, **kwargs):
            return _request(*args, **kwargs)
        def post(self, *args, **kwargs):
            return _request(*args, **kwargs)

    stub = types.ModuleType("requests")
    stub.get = _request
    stub.post = _request
    stub.put = _request
    stub.delete = _request
    stub.request = _request
    stub.Session = _Session
    stub.Response = _Response
    stub.exceptions = types.SimpleNamespace(RequestException=Exception, Timeout=TimeoutError)
    sys.modules["requests"] = stub


def _install_streamlit_stub() -> None:
    # Tests import app.py many times with sys.modules["app"] reset. Real Streamlit
    # is not safe to re-import repeatedly in bare pytest mode because its
    # DeltaGenerator singleton persists across module reloads, so use a minimal
    # deterministic stub for test collection/imports.
    if "streamlit" in sys.modules and getattr(sys.modules["streamlit"], "_DEX_SCOUT_TEST_STUB", False):
        return

    def cache_data(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]

        def decorate(func):
            return func

        return decorate

    cache_data.clear = lambda: None

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __getattr__(self, name):
            return _noop

    def _noop(*args, **kwargs):
        return None

    def _columns(spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context() for _ in range(max(1, int(count or 1)))]

    stub = types.ModuleType("streamlit")
    stub._DEX_SCOUT_TEST_STUB = True
    stub.cache_data = cache_data
    stub.session_state = {}
    stub.secrets = {}
    stub.sidebar = _Context()
    stub.set_page_config = _noop
    stub.autorefresh = _noop
    stub.rerun = _noop
    stub.columns = _columns
    stub.tabs = lambda labels, *args, **kwargs: [_Context() for _ in labels]
    stub.expander = lambda *args, **kwargs: _Context()
    stub.container = lambda *args, **kwargs: _Context()
    stub.empty = lambda *args, **kwargs: _Context()
    stub.radio = lambda label, options, index=0, **kwargs: options[index] if options else None
    stub.selectbox = lambda label, options, index=0, **kwargs: options[index] if options else None
    stub.checkbox = lambda label, value=False, **kwargs: value
    stub.number_input = lambda label, value=0, **kwargs: value
    stub.text_input = lambda label, value="", **kwargs: value
    stub.text_area = lambda label, value="", **kwargs: value
    stub.button = lambda *args, **kwargs: False
    stub.form_submit_button = lambda *args, **kwargs: False
    for name in (
        "title",
        "caption",
        "info",
        "warning",
        "error",
        "success",
        "metric",
        "markdown",
        "write",
        "json",
        "table",
        "line_chart",
        "subheader",
        "code",
        "toast",
        "link_button",
    ):
        setattr(stub, name, _noop)
    sys.modules["streamlit"] = stub


def _install_pandas_stub() -> None:
    if "pandas" in sys.modules:
        return
    try:
        __import__("pandas")
        return
    except ModuleNotFoundError:
        pass

    class _DataFrame(list):
        def __init__(self, data=None, *args, **kwargs):
            super().__init__(data if isinstance(data, list) else [])

        def copy(self):
            return _DataFrame(list(self))

    stub = types.ModuleType("pandas")
    stub.DataFrame = _DataFrame
    stub.to_datetime = lambda value, **kwargs: value
    stub.to_numeric = lambda value, **kwargs: value
    sys.modules["pandas"] = stub


def _install_fastapi_stub() -> None:
    if "fastapi" in sys.modules:
        return
    try:
        __import__("fastapi")
        return
    except ModuleNotFoundError:
        pass

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Response:
        def __init__(self, content=None, status_code=200, **kwargs):
            self.content = content
            self.status_code = int(status_code)

    class Request:
        def __init__(self, json_payload=None):
            self._json_payload = json_payload or {}

        async def json(self):
            return self._json_payload

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.routes = {}

        def _decorator(self, method, path):
            def decorate(func):
                self.routes[(method.upper(), path)] = func
                return func

            return decorate

        def get(self, path):
            return self._decorator("GET", path)

        def post(self, path):
            return self._decorator("POST", path)

        def head(self, path):
            return self._decorator("HEAD", path)

    class _ClientResponse:
        def __init__(self, payload=None, status_code=200):
            self._payload = payload
            self.status_code = int(status_code)

        def json(self):
            return self._payload

    class TestClient:
        def __init__(self, app):
            self.app = app

        def get(self, url):
            return self._request("GET", url)

        def post(self, url, json=None):
            return self._request("POST", url, json_payload=json)

        def _request(self, method, url, json_payload=None):
            parsed = urlsplit(url)
            func = self.app.routes[(method.upper(), parsed.path)]
            kwargs = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            sig = inspect.signature(func)
            if "req" in sig.parameters:
                kwargs["req"] = Request(json_payload)
            for name, param in sig.parameters.items():
                if name in kwargs and param.annotation is int:
                    kwargs[name] = int(kwargs[name])
            result = func(**kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            if isinstance(result, Response):
                return _ClientResponse(result.content, result.status_code)
            return _ClientResponse(result, 200)

    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Request = Request
    fastapi_stub.Response = Response
    testclient_stub = types.ModuleType("fastapi.testclient")
    testclient_stub.TestClient = TestClient
    fastapi_stub.testclient = testclient_stub
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.testclient"] = testclient_stub


_install_requests_stub()
_install_streamlit_stub()
_install_pandas_stub()
_install_fastapi_stub()
