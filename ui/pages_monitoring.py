"""Monitoring page rendering adapter.

app.py owns data loading and business actions. This module only renders from an
explicit context and invokes callables supplied by the Streamlit entrypoint.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import streamlit as st


def render_monitoring_page(context: Dict[str, Any]) -> None:
    """Render the Monitoring page using the explicit UI context."""
    if not isinstance(context, dict):
        context = {}
    canonical_rows = context.get("priority_watchlist_rows")
    if canonical_rows is None:
        for legacy_key in ("priority_rows", "priority_watchlist", "priority_items", "priority_candidates"):
            if legacy_key in context:
                canonical_rows = context.get(legacy_key)
                break
    context["priority_watchlist_rows"] = list(canonical_rows or [])

    actions = context.get("actions") or {}
    renderer = actions.get("render_monitoring")
    if not callable(renderer):
        st.error("Monitoring renderer is unavailable.")
        return

    try:
        signature = inspect.signature(renderer)
        accepts_context = "context" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_context = False

    if accepts_context:
        renderer(context.get("auto_cfg") or {}, context=context)
    else:
        renderer(context.get("auto_cfg") or {})
