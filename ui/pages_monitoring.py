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
    diagnostics = context.get("monitoring_archive_diagnostics")
    if not isinstance(diagnostics, dict):
        debug = context.get("priority_watchlist_debug") if isinstance(context.get("priority_watchlist_debug"), dict) else {}
        diagnostics = debug.get("monitoring_archive_diagnostics") if isinstance(debug.get("monitoring_archive_diagnostics"), dict) else {}
    context["monitoring_archive_diagnostics"] = {
        "rows_seen": int((diagnostics or {}).get("rows_seen", 0) or 0),
        "active": int((diagnostics or {}).get("active", 0) or 0),
        "watch_early": int((diagnostics or {}).get("watch_early", 0) or 0),
        "no_entry": int((diagnostics or {}).get("no_entry", 0) or 0),
        "archived": int((diagnostics or {}).get("archived", 0) or 0),
        "hard_gated": int((diagnostics or {}).get("hard_gated", 0) or 0),
        "portfolio_conflict": int((diagnostics or {}).get("portfolio_conflict", 0) or 0),
        "archive_candidates": int((diagnostics or {}).get("archive_candidates", 0) or 0),
    }

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
