"""Shared Streamlit layout helpers for the DEX Scout shell."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import streamlit as st


def render_page_header(title: str, caption: str = "", *, version: str = "") -> None:
    """Render a standard page title/caption block."""
    st.title(title)
    if caption:
        st.caption(caption)
    if version:
        st.caption(f"DEX Scout {version}")


def render_top_status(context: Dict[str, Any]) -> None:
    """Render lightweight top-level runtime/storage status when provided."""
    version = context.get("version")
    storage_backend = context.get("storage_backend")
    if version:
        st.caption(f"DEX Scout {version}")
    if storage_backend:
        st.caption(f"Storage backend: {storage_backend}")


def render_sidebar(context: Dict[str, Any], pages: Optional[Iterable[str]] = None) -> str:
    """Render a minimal page selector from context.

    The current app.py still owns the production sidebar controls. This helper is
    intentionally small so tests and future extractions can render a sidebar
    without importing app.py or touching storage.
    """
    available_pages = list(pages or context.get("pages") or ["Monitoring", "Archive", "Portfolio"])
    selected_page = str(context.get("selected_page") or (available_pages[0] if available_pages else "Monitoring"))
    if selected_page not in available_pages and not selected_page.startswith("Portfolio"):
        selected_page = available_pages[0] if available_pages else "Monitoring"
    with st.sidebar:
        st.markdown("### DEX Scout")
        choice = st.radio(
            "Page",
            available_pages,
            index=available_pages.index(selected_page) if selected_page in available_pages else 0,
        )
    return "Portfolio" if str(choice).startswith("Portfolio") else str(choice)
