import importlib
import sys
import types
from pathlib import Path


def _install_streamlit_stub(monkeypatch):
    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    stub = types.ModuleType("streamlit")
    stub.session_state = {}
    stub.secrets = {}
    stub.sidebar = _Ctx()
    stub.error = lambda *args, **kwargs: None
    stub.title = lambda *args, **kwargs: None
    stub.caption = lambda *args, **kwargs: None
    stub.json = lambda *args, **kwargs: None
    stub.markdown = lambda *args, **kwargs: None
    stub.radio = lambda label, options, index=0, **kwargs: options[index]
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    return stub


def _install_app_import_stubs(monkeypatch):
    def cache_data(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]
        def decorate(func):
            return func
        return decorate
    cache_data.clear = lambda: None
    st = _install_streamlit_stub(monkeypatch)
    st.cache_data = cache_data
    st.set_page_config = lambda *args, **kwargs: None
    st.columns = lambda spec, *args, **kwargs: [_install_streamlit_stub(monkeypatch).sidebar for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels, *args, **kwargs: [st.sidebar for _ in labels]
    st.expander = lambda *args, **kwargs: st.sidebar
    st.container = lambda *args, **kwargs: st.sidebar
    st.empty = lambda *args, **kwargs: st.sidebar
    st.checkbox = lambda label, value=False, **kwargs: value
    st.number_input = lambda label, value=0, **kwargs: value
    st.text_input = lambda label, value="", **kwargs: value
    st.text_area = lambda label, value="", **kwargs: value
    st.button = lambda *args, **kwargs: False
    st.form_submit_button = lambda *args, **kwargs: False
    for name in ("info", "warning", "success", "metric", "write", "table", "line_chart", "subheader", "code", "toast", "link_button", "divider"):
        setattr(st, name, lambda *args, **kwargs: None)


def test_ui_modules_import_without_app_or_storage_writes(monkeypatch):
    _install_streamlit_stub(monkeypatch)
    sys.modules.pop("app", None)
    calls = []
    fake_storage = types.ModuleType("storage_repository")
    fake_storage.storage_write_text = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "storage_repository", fake_storage)

    for module_name in [
        "ui.layout",
        "ui.pages_scout",
        "ui.pages_monitoring",
        "ui.pages_portfolio",
        "ui.pages_archive",
        "ui.pages_runtime",
    ]:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)

    assert "app" not in sys.modules
    assert calls == []


def test_ui_render_functions_call_explicit_actions(monkeypatch):
    _install_streamlit_stub(monkeypatch)
    calls = []
    import ui.pages_monitoring as monitoring
    import ui.pages_archive as archive
    import ui.pages_portfolio as portfolio
    import ui.pages_scout as scout

    context = {
        "auto_cfg": {"x": 1},
        "scout_cfg": {"seed_k": 2},
        "actions": {
            "render_monitoring": lambda cfg: calls.append(("monitoring", cfg)),
            "render_archive": lambda: calls.append(("archive", None)),
            "render_portfolio": lambda: calls.append(("portfolio", None)),
            "render_scout": lambda cfg: calls.append(("scout", cfg)),
        },
    }
    monitoring.render_monitoring_page(context)
    archive.render_archive_page(context)
    portfolio.render_portfolio_page(context)
    scout.render_scout_page(context)

    assert calls == [
        ("monitoring", {"x": 1}),
        ("archive", None),
        ("portfolio", None),
        ("scout", {"seed_k": 2}),
    ]


def test_app_imports_in_worker_mode_and_exposes_compat_wrappers(monkeypatch):
    _install_app_import_stubs(monkeypatch)
    monkeypatch.setenv("DEX_SCOUT_WORKER_MODE", "1")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")

    for name in [
        "load_monitoring",
        "save_monitoring",
        "load_portfolio",
        "save_portfolio",
        "page_monitoring",
        "page_archive",
        "page_portfolio",
        "build_ui_context",
        "render_selected_page",
    ]:
        assert callable(getattr(app, name))


def test_page_router_calls_correct_render_function(monkeypatch):
    _install_app_import_stubs(monkeypatch)
    monkeypatch.setenv("DEX_SCOUT_WORKER_MODE", "1")
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    calls = []
    monkeypatch.setattr(app, "render_monitoring_page", lambda context: calls.append(("Monitoring", context["selected_page"])))
    monkeypatch.setattr(app, "render_archive_page", lambda context: calls.append(("Archive", context["selected_page"])))
    monkeypatch.setattr(app, "render_portfolio_page", lambda context: calls.append(("Portfolio", context["selected_page"])))
    monkeypatch.setattr(app, "render_scout_page", lambda context: calls.append(("Scout", context["selected_page"])))
    monkeypatch.setattr(app, "render_runtime_page", lambda context: calls.append(("Runtime", context["selected_page"])))

    for page in ["Monitoring", "Archive", "Portfolio", "Scout", "Runtime"]:
        app.render_selected_page({"selected_page": page})

    assert calls == [
        ("Monitoring", "Monitoring"),
        ("Archive", "Archive"),
        ("Portfolio", "Portfolio"),
        ("Scout", "Scout"),
        ("Runtime", "Runtime"),
    ]


def test_master_guide_current_ui_shell_terms():
    text = Path("docs/DEX_SCOUT_INTERVENTION_GUIDE.html").read_text(encoding="utf-8")
    for needle in [
        "storage_repository.py",
        "notification_core.py",
        "monitoring_core.py",
        "app_runtime_facade.py",
        "D1",
        "#282",
    ]:
        assert needle in text
    for stale in [
        "Немає жодних тестів",
        "Supabase primary",
        "всі три runtime-ролі імпортують app.py",
    ]:
        assert stale not in text
