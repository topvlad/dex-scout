"""Minimal local Streamlit fallback for non-UI test/runtime imports."""

from __future__ import annotations


def _noop(*args, **kwargs):
    return None


class _Context:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def __getattr__(self, name):
        return _noop


def cache_data(*decorator_args, **decorator_kwargs):
    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
        return decorator_args[0]
    def decorate(func):
        return func
    return decorate


cache_data.clear = lambda: None
session_state = {}
secrets = {}
sidebar = _Context()
set_page_config = _noop
autorefresh = _noop
rerun = _noop
columns = lambda spec, *args, **kwargs: [_Context() for _ in range(spec if isinstance(spec, int) else len(spec))]
tabs = lambda labels, *args, **kwargs: [_Context() for _ in labels]
expander = lambda *args, **kwargs: _Context()
container = lambda *args, **kwargs: _Context()
empty = lambda *args, **kwargs: _Context()
radio = lambda label, options, index=0, **kwargs: options[index] if options else None
selectbox = lambda label, options, index=0, **kwargs: options[index] if options else None
checkbox = lambda label, value=False, **kwargs: value
number_input = lambda label, value=0, **kwargs: value
text_input = lambda label, value="", **kwargs: value
text_area = lambda label, value="", **kwargs: value
button = lambda *args, **kwargs: False
form_submit_button = lambda *args, **kwargs: False
for _name in ("title", "caption", "info", "warning", "error", "success", "metric", "markdown", "write", "json", "table", "line_chart", "subheader", "code", "toast", "link_button"):
    globals()[_name] = _noop
