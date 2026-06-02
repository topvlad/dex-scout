"""Runtime/debug page rendering helpers."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_runtime_page(context: Dict[str, Any]) -> None:
    """Render runtime diagnostics from explicit context/action callables."""
    actions = context.get("actions") or {}
    renderer = actions.get("render_runtime") or actions.get("render_debug_panel")
    if callable(renderer):
        renderer()
        return
    runtime_state = context.get("runtime_state") or {}
    st.title("Runtime")
    if runtime_state:
        st.json(runtime_state)
    else:
        st.caption("Runtime state data unavailable.")
