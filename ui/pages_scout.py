"""Scout page rendering adapter."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st



def _operator_diagnostics(context: Dict[str, Any]) -> Dict[str, Any]:
    diag = context.get("operator_diagnostics") if isinstance(context.get("operator_diagnostics"), dict) else None
    if diag:
        return diag
    import app_service

    return app_service.build_operator_diagnostics(context)


def _render_live_pulse_diagnostics(context: Dict[str, Any]) -> None:
    diag = _operator_diagnostics(context)
    scanner = diag.get("scanner", {}) if isinstance(diag.get("scanner"), dict) else {}
    status = str(diag.get("status") or "unknown")
    final_count = int(scanner.get("final_count", 0) or 0)
    reason = str(scanner.get("last_empty_reason") or "")
    payload = context.get("live_pulse_payload") if isinstance(context.get("live_pulse_payload"), dict) else {}
    old_shape = not payload or not any(key in payload for key in ("status", "final_count", "raw_seen", "debug"))
    should_show = final_count <= 0 or status in {"warning", "degraded", "unknown"}
    if not should_show:
        if hasattr(st, "expander"):
            with st.expander("Live Pulse diagnostics", expanded=False):
                st.json(scanner)
        return

    st.markdown("**Live Pulse diagnostics**")
    if old_shape:
        st.caption("Live Pulse payload missing or old-shape; wait for scan_cycle or run Runtime Smoke/read_only")
    st.markdown(str(
        {
            "status": scanner.get("source_status", status),
            "last_empty_reason": reason or "unknown",
            "raw_seen": scanner.get("raw_seen", 0),
            "normalized": scanner.get("normalized", 0),
            "clean_candidates": scanner.get("clean_candidates", 0),
            "final_count": scanner.get("final_count", 0),
            "sources_tried": scanner.get("sources_tried", []),
            "sources_failed": scanner.get("sources_failed", 0),
            "sources_empty": scanner.get("sources_empty", 0),
            "sources_disabled": scanner.get("sources_disabled", 0),
        }
    ))
    errors = scanner.get("source_errors") if isinstance(scanner.get("source_errors"), list) else []
    if errors:
        st.caption(f"Top source errors: {errors[:3]}")
    if hasattr(st, "expander"):
        with st.expander("Raw Live Pulse diagnostics", expanded=False):
            st.json(scanner)


def render_scout_page(context: Dict[str, Any]) -> None:
    """Render the Scout page using the explicit UI context."""
    if not isinstance(context, dict):
        context = {}
    actions = context.get("actions") or {}
    renderer = actions.get("render_scout")
    if not callable(renderer):
        st.error("Scout renderer is unavailable.")
        _render_live_pulse_diagnostics(context)
        return
    renderer(context.get("scout_cfg") or {})
    _render_live_pulse_diagnostics(context)
