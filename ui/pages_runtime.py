"""Runtime/debug page rendering helpers."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

import admin_controls



JOB_MODES = ("scan_cycle", "monitor_cycle", "notify_cycle", "digest_cycle", "outcome_cycle", "maintenance_cycle")


def _operator_diagnostics(context: Dict[str, Any]) -> Dict[str, Any]:
    diag = context.get("operator_diagnostics") if isinstance(context.get("operator_diagnostics"), dict) else None
    if diag:
        return diag
    import app_service

    return app_service.build_operator_diagnostics(context)


def _fmt(value: Any) -> str:
    text = str(value if value not in (None, "") else "—")
    return text[:240]


def _render_runtime_diagnostics(context: Dict[str, Any]) -> None:
    diag = _operator_diagnostics(context)
    runtime = diag.get("runtime", {}) if isinstance(diag.get("runtime"), dict) else {}
    storage = diag.get("storage", {}) if isinstance(diag.get("storage"), dict) else {}
    runtime_state = context.get("runtime_state") if isinstance(context.get("runtime_state"), dict) else {}

    st.markdown("**Operator diagnostics**")
    st.caption(f"Status: {_fmt(diag.get('status'))} — {_fmt(diag.get('summary'))}")

    st.markdown("**Runtime health**")
    st.markdown(str({
        "runtime_matrix": runtime.get("matrix_status", "unknown"),
        "app_compat": runtime.get("app_compat_status", "unknown"),
        "golden_fixtures": runtime.get("golden_fixtures_status", "unknown"),
    }))
    last_jobs = runtime.get("last_jobs") if isinstance(runtime.get("last_jobs"), dict) else {}
    if last_jobs:
        compact_jobs = {mode: last_jobs.get(mode) for mode in JOB_MODES if mode in last_jobs}
        st.caption(f"Last jobs: {compact_jobs or last_jobs}")
    else:
        st.caption("Last successful jobs unavailable.")

    st.markdown("**Job freshness**")
    freshness = {}
    stale_jobs = runtime.get("stale_jobs") if isinstance(runtime.get("stale_jobs"), list) else []
    stale_text = {str(item.get("mode") or item.get("job") or item) for item in stale_jobs if isinstance(item, dict)} | {str(item) for item in stale_jobs if not isinstance(item, dict)}
    for mode in JOB_MODES:
        job = last_jobs.get(mode) if isinstance(last_jobs, dict) else {}
        status = "stale" if mode in stale_text else _fmt((job or {}).get("last_job_status") if isinstance(job, dict) else "unknown")
        freshness[mode] = status
    st.markdown(str(freshness))

    st.markdown("**Storage health**")
    st.markdown(str({
        "backend": storage.get("backend", context.get("storage_backend", "unknown")),
        "verify_status": storage.get("verify_status", "unknown"),
        "last_read_ok": storage.get("last_read_ok"),
        "last_write_ok": storage.get("last_write_ok"),
    }))
    storage_summary = storage.get("last_summary") or runtime_state.get("last_storage_diag_summary") or runtime_state.get("last_diag_summary")
    if storage_summary:
        st.caption(f"Last storage diagnostic: {_fmt(storage_summary)}")

    st.markdown("**Operator actions reminder**")
    st.caption("Run Runtime Smoke: runtime_matrix")
    st.caption("Run Runtime Smoke: read_only")
    st.caption("Run Runtime Jobs: monitor_cycle / notify_cycle / scan_cycle")

    if hasattr(st, "expander"):
        with st.expander("Raw operator diagnostics", expanded=False):
            st.json(diag)



def _render_manual_recovery(context: Dict[str, Any]) -> None:
    """Render guarded manual recovery controls without mutating on render."""
    st.markdown("**Manual recovery**")
    plan_builder = (context.get("actions") or {}).get("build_admin_recovery_plan")
    if callable(plan_builder):
        plan = plan_builder(context)
    else:
        plan = admin_controls.build_admin_recovery_plan(context)
    st.caption(f"Recovery plan status: {_fmt(plan.get('status'))}")

    order = plan.get("recommended_order") if isinstance(plan.get("recommended_order"), list) else []
    if order:
        st.markdown("Recommended order: " + " → ".join(str(item) for item in order))

    steps = plan.get("manual_workflow_steps") if isinstance(plan.get("manual_workflow_steps"), list) else []
    if steps:
        st.markdown("Runbook steps:")
        for step in steps:
            st.caption(f"• {_fmt(step)}")
            if hasattr(st, "code"):
                st.code(str(step))

    warnings = plan.get("warnings") if isinstance(plan.get("warnings"), list) else []
    for warning in warnings[:4]:
        if hasattr(st, "warning"):
            st.warning(_fmt(warning))
        else:
            st.caption(f"Warning: {_fmt(warning)}")

    validator = (context.get("actions") or {}).get("validate_admin_action")
    executor = (context.get("actions") or {}).get("execute_admin_action")
    validate_fn = validator if callable(validator) else admin_controls.validate_admin_action
    execute_fn = executor if callable(executor) else None

    stale_validation = validate_fn("clear_stale_lock", context, confirmation="", dry_run=True)
    if stale_validation.get("reason") != "stale_lock_not_available_or_not_auditable" or stale_validation.get("enabled"):
        st.markdown("**Guarded stale lock control**")
        st.caption(f"Dry-run preview: {_fmt(stale_validation.get('reason'))}; confirmation required: {_fmt(stale_validation.get('confirmation_phrase'))}")
        confirmation = st.text_input("Confirm stale lock action", value="", key="admin_clear_stale_lock_confirmation")
        dry_run = st.checkbox("Dry-run only", value=True, key="admin_clear_stale_lock_dry_run")
        if st.button("Preview clear stale lock", key="admin_clear_stale_lock_preview"):
            result = admin_controls.execute_admin_action("clear_stale_lock", context, confirmation=confirmation, dry_run=True, adapters={})
            st.caption(f"Preview result: {_fmt(result.get('status'))} — {_fmt(result.get('reason'))}")
            if hasattr(st, "expander"):
                with st.expander("Raw preview", expanded=False):
                    st.json(result)
        if st.button("Execute clear stale lock", key="admin_clear_stale_lock_execute"):
            if execute_fn is None:
                result = admin_controls.execute_admin_action("clear_stale_lock", context, confirmation=confirmation, dry_run=dry_run, adapters={})
            else:
                result = execute_fn("clear_stale_lock", context, confirmation=confirmation, dry_run=dry_run)
            st.caption(f"Action result: {_fmt(result.get('status'))} — {_fmt(result.get('reason'))}")
            if hasattr(st, "expander"):
                with st.expander("Raw action result", expanded=False):
                    st.json(result)
    else:
        st.caption("Stale lock control: not available or no auditable stale lock detected.")

    if hasattr(st, "expander"):
        with st.expander("Raw manual recovery plan", expanded=False):
            st.json(plan)


def render_runtime_page(context: Dict[str, Any]) -> None:
    """Render runtime diagnostics from explicit context/action callables."""
    if not isinstance(context, dict):
        context = {}
    st.title("Runtime")
    _render_runtime_diagnostics(context)
    _render_manual_recovery(context)
    actions = context.get("actions") or {}
    renderer = actions.get("render_runtime") or actions.get("render_debug_panel")
    if callable(renderer):
        renderer()
        return
    runtime_state = context.get("runtime_state") or {}
    if runtime_state:
        if hasattr(st, "expander"):
            with st.expander("Runtime state", expanded=False):
                st.json(runtime_state)
        else:
            st.json(runtime_state)
    else:
        st.caption("Runtime state data unavailable.")
