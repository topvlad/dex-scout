import importlib
import sys
import types


def _install_import_stubs(monkeypatch):
    class StreamlitStub(types.SimpleNamespace):
        def set_page_config(self, *args, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("st.set_page_config must not run in worker mode")

    def cache_data(*decorator_args, **decorator_kwargs):
        def decorate(func):
            return func
        return decorate

    cache_data.clear = lambda: None
    streamlit_stub = StreamlitStub(cache_data=cache_data, secrets={}, session_state={})
    pandas_stub = types.SimpleNamespace(
        DataFrame=lambda *args, **kwargs: [],
        to_datetime=lambda value, **kwargs: value,
        to_numeric=lambda value, **kwargs: value,
    )
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)


def test_runtime_core_imports_without_streamlit(monkeypatch):
    sys.modules.pop("runtime_core", None)
    sys.modules.pop("streamlit", None)

    module = importlib.import_module("runtime_core")

    assert module.parse_float("1.25", 0.0) == 1.25
    assert "streamlit" not in sys.modules


def test_storage_core_imports_without_streamlit_or_storage_env(monkeypatch):
    for key in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "D1_PROXY_URL", "D1_PROXY_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("storage_core", None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("app", None)

    module = importlib.import_module("storage_core")

    assert module.storage_key_for_path("data/monitoring.csv") == "monitoring.csv"
    assert module.supabase_configured("", "", "supabase") is False
    assert "streamlit" not in sys.modules
    assert "app" not in sys.modules


def test_app_imports_in_worker_mode_without_set_page_config(monkeypatch):
    _install_import_stubs(monkeypatch)
    monkeypatch.setenv("DEX_SCOUT_WORKER_MODE", "1")
    sys.modules.pop("app", None)

    app = importlib.import_module("app")

    assert app.WORKER_FAST_MODE is True


def test_app_runtime_core_helper_parity(monkeypatch):
    _install_import_stubs(monkeypatch)
    monkeypatch.setenv("DEX_SCOUT_WORKER_MODE", "1")
    runtime_core = importlib.import_module("runtime_core")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")

    monkeypatch.setenv("DEX_TEST_INT", "42")
    assert app._env_int("DEX_TEST_INT", 0) == runtime_core._env_int("DEX_TEST_INT", 0)
    assert app.parse_float("3.5", 0.0) == runtime_core.parse_float("3.5", 0.0)
    assert app.parse_int("7.9", 0) == runtime_core.parse_int("7.9", 0)
    assert app.normalize_chain_name("Binance") == runtime_core.normalize_chain_name("Binance")
    assert app.addr_store("bsc", "0xABC") == runtime_core.addr_store("bsc", "0xABC")
    assert app.addr_store("solana", "SoLCase") == runtime_core.addr_store("solana", "SoLCase")
    assert app.safe_get({"a": {"b": 2}}, "a", "b") == runtime_core.safe_get({"a": {"b": 2}}, "a", "b")
    storage_core = importlib.import_module("storage_core")
    assert app.canonical_entity_key("BSC", "0xABC") == runtime_core.canonical_entity_key("BSC", "0xABC")
    assert app.storage_key_for_path("data/monitoring.csv") == storage_core.storage_key_for_path("data/monitoring.csv")


def test_core_modules_do_not_import_app():
    sys.modules.pop("app", None)
    sys.modules.pop("runtime_core", None)
    sys.modules.pop("storage_core", None)

    runtime_core = importlib.import_module("runtime_core")
    storage_core = importlib.import_module("storage_core")

    assert "app" not in sys.modules
    assert "app" not in getattr(runtime_core, "__dict__", {})
    assert "app" not in getattr(storage_core, "__dict__", {})
