"""Monitoring page rendering adapter.

app.py owns data loading and business actions. This module only renders from an
explicit context and invokes callables supplied by the Streamlit entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_monitoring_page(context: Dict[str, Any]) -> None:
    """Render the Monitoring page using the explicit UI context."""
    actions = context.get("actions") or {}
    renderer = actions.get("render_monitoring")
    if not callable(renderer):
        st.error("Monitoring renderer is unavailable.")
        return
    renderer(context.get("auto_cfg") or {})
