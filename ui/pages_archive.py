"""Archive page rendering adapter."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_archive_page(context: Dict[str, Any]) -> None:
    """Render the Archive page using the explicit UI context."""
    actions = context.get("actions") or {}
    renderer = actions.get("render_archive")
    if not callable(renderer):
        st.error("Archive renderer is unavailable.")
        return
    renderer()
