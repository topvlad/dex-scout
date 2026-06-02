"""Portfolio page rendering adapter."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_portfolio_page(context: Dict[str, Any]) -> None:
    """Render the Portfolio page using the explicit UI context."""
    actions = context.get("actions") or {}
    renderer = actions.get("render_portfolio")
    if not callable(renderer):
        st.error("Portfolio renderer is unavailable.")
        return
    renderer()
