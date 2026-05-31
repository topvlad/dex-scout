import subprocess
import sys
import textwrap


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        text=True,
        capture_output=True,
    )


def test_runtime_core_imports_without_streamlit():
    _run_python(
        """
        import sys
        sys.modules.pop("runtime_core", None)
        sys.modules.pop("streamlit", None)

        import runtime_core

        assert runtime_core.parse_float("1.25", 0.0) == 1.25
        assert "streamlit" not in sys.modules
        assert "app" not in sys.modules
        """
    )


def test_storage_core_imports_without_streamlit_or_storage_env():
    _run_python(
        """
        import os
        import sys
        for key in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "D1_PROXY_URL", "D1_PROXY_TOKEN"):
            os.environ.pop(key, None)
        sys.modules.pop("storage_core", None)
        sys.modules.pop("streamlit", None)
        sys.modules.pop("app", None)

        import storage_core

        assert storage_core.storage_key_for_path("data/monitoring.csv") == "monitoring.csv"
        assert storage_core.supabase_configured("", "", "supabase") is False
        assert "streamlit" not in sys.modules
        assert "app" not in sys.modules
        """
    )


def test_app_imports_in_worker_mode_without_set_page_config():
    _run_python(
        """
        import os
        import sys
        import types

        os.environ["DEX_SCOUT_WORKER_MODE"] = "1"

        class StreamlitStub(types.SimpleNamespace):
            def set_page_config(self, *args, **kwargs):
                raise AssertionError("st.set_page_config must not run in worker mode")

        def cache_data(*decorator_args, **decorator_kwargs):
            def decorate(func):
                return func
            return decorate

        cache_data.clear = lambda: None
        sys.modules["streamlit"] = StreamlitStub(cache_data=cache_data, secrets={}, session_state={})
        sys.modules["pandas"] = types.SimpleNamespace(
            DataFrame=lambda *args, **kwargs: [],
            to_datetime=lambda value, **kwargs: value,
            to_numeric=lambda value, **kwargs: value,
        )

        import app

        assert app.WORKER_FAST_MODE is True
        """
    )


def test_app_runtime_core_helper_parity():
    _run_python(
        """
        import os
        import sys
        import types

        os.environ["DEX_SCOUT_WORKER_MODE"] = "1"
        os.environ["DEX_TEST_INT"] = "42"

        class StreamlitStub(types.SimpleNamespace):
            def set_page_config(self, *args, **kwargs):
                raise AssertionError("st.set_page_config must not run in worker mode")

        def cache_data(*decorator_args, **decorator_kwargs):
            def decorate(func):
                return func
            return decorate

        cache_data.clear = lambda: None
        sys.modules["streamlit"] = StreamlitStub(cache_data=cache_data, secrets={}, session_state={})
        sys.modules["pandas"] = types.SimpleNamespace(
            DataFrame=lambda *args, **kwargs: [],
            to_datetime=lambda value, **kwargs: value,
            to_numeric=lambda value, **kwargs: value,
        )

        import app
        import runtime_core
        import storage_core

        assert app._env_int("DEX_TEST_INT", 0) == runtime_core._env_int("DEX_TEST_INT", 0)
        assert app.parse_float("3.5", 0.0) == runtime_core.parse_float("3.5", 0.0)
        assert app.parse_int("7.9", 0) == runtime_core.parse_int("7.9", 0)
        assert app.normalize_chain_name("Binance") == runtime_core.normalize_chain_name("Binance")
        assert app.addr_store("bsc", "0xABC") == runtime_core.addr_store("bsc", "0xABC")
        assert app.addr_store("solana", "SoLCase") == runtime_core.addr_store("solana", "SoLCase")
        assert app.safe_get({"a": {"b": 2}}, "a", "b") == runtime_core.safe_get({"a": {"b": 2}}, "a", "b")
        assert app.canonical_entity_key("BSC", "0xABC") == runtime_core.canonical_entity_key("BSC", "0xABC")
        assert app.storage_key_for_path("data/monitoring.csv") == storage_core.storage_key_for_path("data/monitoring.csv")
        """
    )


def test_core_modules_do_not_import_app():
    _run_python(
        """
        import sys
        sys.modules.pop("app", None)
        sys.modules.pop("runtime_core", None)
        sys.modules.pop("storage_core", None)

        import runtime_core
        import storage_core

        assert "app" not in sys.modules
        assert "app" not in getattr(runtime_core, "__dict__", {})
        assert "app" not in getattr(storage_core, "__dict__", {})
        """
    )
