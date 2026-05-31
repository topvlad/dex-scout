import asyncio
import inspect
import sys
import types
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


_install_streamlit_stub()
_install_pandas_stub()
_install_fastapi_stub()
