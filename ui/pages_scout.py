"""Scout page rendering adapter."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_scout_page(context: Dict[str, Any]) -> None:
    """Render the Scout page using the explicit UI context."""
    actions = context.get("actions") or {}
    renderer = actions.get("render_scout")
    if not callable(renderer):
        st.error("Scout renderer is unavailable.")
        return
    renderer(context.get("scout_cfg") or {})
