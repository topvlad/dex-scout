"""Monitoring page rendering adapter.

app.py owns data loading and business actions. This module only renders from an
explicit context and invokes callables supplied by the Streamlit entrypoint.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import streamlit as st



EXCLUDED_KEYS = (
    "no_entry",
    "archived",
    "hard_gated",
    "portfolio_material_conflict",
    "missing_score",
    "below_priority_threshold",
    "unknown",
)


def _operator_diagnostics(context: Dict[str, Any]) -> Dict[str, Any]:
    diag = context.get("operator_diagnostics") if isinstance(context.get("operator_diagnostics"), dict) else None
    if diag:
        return diag
    import app_service

    return app_service.build_operator_diagnostics(context)


def _render_priority_diagnostics(context: Dict[str, Any]) -> None:
    diag = _operator_diagnostics(context)
    priority = diag.get("priority", {}) if isinstance(diag.get("priority"), dict) else {}
    rows = context.get("priority_watchlist_rows") if isinstance(context.get("priority_watchlist_rows"), list) else []
    final_rows = int(priority.get("final_priority_rows", len(rows)) or 0)
    if final_rows > 0:
        if hasattr(st, "expander"):
            with st.expander("Priority diagnostics", expanded=False):
                st.json(priority)
        return

    excluded = priority.get("excluded") if isinstance(priority.get("excluded"), dict) else {}
    counts = {key: int(excluded.get(key, 0) or 0) for key in EXCLUDED_KEYS}
    if counts["portfolio_material_conflict"] == 0 and excluded.get("portfolio_conflict"):
        counts["portfolio_material_conflict"] = int(excluded.get("portfolio_conflict", 0) or 0)

    st.markdown("**Priority watchlist diagnostics**")
    st.markdown(str(
        {
            "source_monitoring_rows": priority.get("source_monitoring_rows", 0),
            "active_monitoring_rows": priority.get("active_monitoring_rows", 0),
            "eligible_watch_early_rows": priority.get("eligible_watch_early_rows", 0),
            "final_priority_rows": final_rows,
            "excluded": counts,
        }
    ))
    samples = priority.get("top_excluded_samples") if isinstance(priority.get("top_excluded_samples"), list) else []
    if samples:
        st.caption(f"Top excluded samples: {samples[:3]}")
    if hasattr(st, "expander"):
        with st.expander("Raw Priority diagnostics", expanded=False):
            st.json(priority)


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
        _render_priority_diagnostics(context)
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
    _render_priority_diagnostics(context)
