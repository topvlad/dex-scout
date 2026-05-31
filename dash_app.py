"""Read-only Dash dashboard for DEX Scout D1 data.

This module intentionally uses only the existing D1 proxy/helper read paths from
``app.py``. It does not call scanner, worker, write, delete, Telegram, or alert
functions. Streamlit remains the primary UI; this is a separate Dash entrypoint.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from collections import Counter
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

# Keep importing app.py side-effect-light for Dash/pytest and force this separate entrypoint onto the D1 read path.
os.environ.setdefault("DEX_SCOUT_WORKER_MODE", "1")
os.environ["STORAGE_BACKEND"] = "d1"

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash.dash_table import DataTable

from app import (  # noqa: E402
    D1_PROXY_URL,
    HIST_FIELDS,
    JOB_HEARTBEATS_TABLE,
    JOB_RUNS_TABLE,
    MON_FIELDS,
    MON_HISTORY_CSV,
    MONITORING_CSV,
    PORTFOLIO_CSV,
    PORTFOLIO_FIELDS,
    PORTFOLIO_RECO_LOG_CSV,
    PORTFOLIO_RECO_LOG_FIELDS,
    TG_STATE_TABLE,
    _d1_ok,
    check_runtime_contract,
    d1_get_storage,
    d1_select_rows,
    get_worker_runtime_state,
    parse_float,
    storage_key_for_path,
)

DASH_CACHE_TTL_SEC = max(30, min(60, int(os.getenv("DASH_D1_CACHE_TTL_SEC", "45") or "45")))
DASH_HISTORY_ROWS_LIMIT = max(50, int(os.getenv("DASH_HISTORY_ROWS_LIMIT", "5000") or "5000"))
DASH_TABLE_PAGE_SIZE = max(5, int(os.getenv("DASH_TABLE_PAGE_SIZE", "15") or "15"))

PAGE_LINKS = [
    ("/", "Overview"),
    ("/monitoring", "Monitoring"),
    ("/portfolio", "Portfolio"),
    ("/live-pulse", "Live Pulse"),
    ("/runtime", "Runtime"),
    ("/archive", "Archive"),
]


def ttl_cache(ttl_sec: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tiny process-local TTL cache for low-latency read-only D1 snapshots."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        cache: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}

        @wraps(fn)
        def wrapped(*args: Any) -> Any:
            now = time.time()
            key = tuple(_freeze_arg(arg) for arg in args)
            if key in cache:
                cached_at, value = cache[key]
                if now - cached_at < ttl_sec:
                    return value
            value = fn(*args)
            cache[key] = (now, value)
            return value

        wrapped.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapped

    return decorator


def _freeze_arg(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_arg(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze_arg(v) for v in value)
    return value


def _csv_rows(content: Optional[str], fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not str(content or "").strip():
        return []
    rows = list(csv.DictReader(io.StringIO(str(content))))
    if not fields:
        return rows
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _read_d1_csv(path: str, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Read a CSV blob from D1 app_storage via the existing D1 helper only."""
    return _csv_rows(d1_get_storage(storage_key_for_path(path)), fields=fields)


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_monitoring_rows() -> List[Dict[str, Any]]:
    return _read_d1_csv(MONITORING_CSV, MON_FIELDS)


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_portfolio_rows() -> List[Dict[str, Any]]:
    return _read_d1_csv(PORTFOLIO_CSV, PORTFOLIO_FIELDS)


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_monitoring_history_rows(limit: int = DASH_HISTORY_ROWS_LIMIT) -> List[Dict[str, Any]]:
    rows = _read_d1_csv(MON_HISTORY_CSV, HIST_FIELDS)
    return rows[-max(1, int(limit or DASH_HISTORY_ROWS_LIMIT)) :]


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_portfolio_reco_rows(limit: int = 1000) -> List[Dict[str, Any]]:
    rows = _read_d1_csv(PORTFOLIO_RECO_LOG_CSV, PORTFOLIO_RECO_LOG_FIELDS)
    return rows[-max(1, int(limit or 1000)) :]


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_scanner_state() -> Dict[str, Any]:
    raw = d1_get_storage("scanner_state.json")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": "invalid_scanner_state_json"}
    return data if isinstance(data, dict) else {}


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_tg_state() -> Dict[str, Any]:
    rows, status = d1_select_rows(table=TG_STATE_TABLE, filters={"state_key": "eq.default"}, select="state_json,updated_epoch", limit=1)
    if not status.get("ok") or not rows:
        raw = d1_get_storage("tg_state.json")
        if not raw:
            return {"_status": status}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"_status": status}
        except json.JSONDecodeError:
            return {"_status": status, "_error": "invalid_tg_state_json"}
    payload = rows[0].get("state_json") if isinstance(rows[0], dict) else {}
    return payload if isinstance(payload, dict) else {"_status": status}


@ttl_cache(DASH_CACHE_TTL_SEC)
def read_runtime_snapshot() -> Dict[str, Any]:
    worker_runtime = get_worker_runtime_state()
    scanner_state = read_scanner_state()
    heartbeats, hb_status = d1_select_rows(table=JOB_HEARTBEATS_TABLE, select="*", limit=200)
    runs, runs_status = d1_select_rows(table=JOB_RUNS_TABLE, select="*", limit=200)
    contract = check_runtime_contract()
    return {
        "worker_runtime": worker_runtime if isinstance(worker_runtime, dict) else {},
        "scanner_state": scanner_state,
        "heartbeats": heartbeats or [],
        "heartbeats_status": hb_status,
        "job_runs": runs or [],
        "job_runs_status": runs_status,
        "contract": contract,
    }


def is_active(row: Dict[str, Any]) -> bool:
    active = str(row.get("active", "1")).strip().lower()
    status = str(row.get("status", "")).strip().upper()
    lifecycle = str(row.get("lifecycle_state", "")).strip().upper()
    archived = str(row.get("ts_archived", "")).strip()
    if active in {"0", "false", "no", "n"} or archived:
        return False
    if status in {"ARCHIVED", "SUPPRESSED", "DEAD", "CLOSED"}:
        return False
    if lifecycle in {"ARCHIVED", "SUPPRESSED", "DEAD"}:
        return False
    return True


def token_key(row: Dict[str, Any]) -> str:
    chain = str(row.get("chain") or "").strip().lower()
    addr = str(row.get("base_addr") or row.get("base_token_address") or row.get("token_addr") or row.get("ca") or "").strip()
    symbol = str(row.get("base_symbol") or row.get("symbol") or row.get("token") or "?").strip()
    return f"{chain}:{addr or symbol}"


def display_symbol(row: Dict[str, Any]) -> str:
    return str(row.get("base_symbol") or row.get("symbol") or row.get("token") or token_key(row)).strip() or "?"


def summarize_live_pulse(runtime: Dict[str, Any]) -> Dict[str, Any]:
    scanner = runtime.get("scanner_state") if isinstance(runtime.get("scanner_state"), dict) else {}
    worker = runtime.get("worker_runtime") if isinstance(runtime.get("worker_runtime"), dict) else {}
    pulse = worker.get("live_pulse_candidates") or scanner.get("live_pulse_candidates") or {}
    if not isinstance(pulse, dict):
        pulse = {}
    final_candidates = pulse.get("final_candidates") if isinstance(pulse.get("final_candidates"), list) else []
    raw_cards = pulse.get("cards") if isinstance(pulse.get("cards"), list) else []
    scan_debug = pulse.get("scan_debug") if isinstance(pulse.get("scan_debug"), dict) else {}
    rejected = scan_debug.get("rejected_candidates") if isinstance(scan_debug.get("rejected_candidates"), dict) else {}
    hard_or_dead = [c for c in raw_cards if _candidate_hard_dead(c)]
    return {
        "clean_candidates": int(parse_float(pulse.get("clean_candidates"), len(final_candidates))),
        "final_candidates": final_candidates,
        "raw_cards": raw_cards,
        "hard_dead_candidates": hard_or_dead,
        "scan_debug": scan_debug,
        "rejected_candidates": rejected,
    }


def _candidate_hard_dead(candidate: Dict[str, Any]) -> bool:
    text = " ".join(str(candidate.get(k, "")) for k in ("ui_bucket", "health_label", "weak_reason", "decision_reason", "gate_reason", "risk_flags", "toxic_flags")).lower()
    return any(term in text for term in ("dead", "rug", "scam", "hard", "gate", "critical"))


def dashboard_snapshot() -> Dict[str, Any]:
    monitoring = read_monitoring_rows()
    portfolio = read_portfolio_rows()
    runtime = read_runtime_snapshot()
    worker = runtime.get("worker_runtime") if isinstance(runtime.get("worker_runtime"), dict) else {}
    scanner = runtime.get("scanner_state") if isinstance(runtime.get("scanner_state"), dict) else {}
    pulse = summarize_live_pulse(runtime)
    active_monitoring = [row for row in monitoring if is_active(row)]
    active_portfolio = [row for row in portfolio if is_active(row)]
    archived_monitoring = [row for row in monitoring if not is_active(row)]
    last_stats = worker.get("last_scan_stats") or scanner.get("last_stats") or {}
    if not isinstance(last_stats, dict):
        last_stats = {}
    return {
        "d1_ok": bool(_d1_ok()),
        "d1_proxy_url": D1_PROXY_URL,
        "monitoring": monitoring,
        "portfolio": portfolio,
        "active_monitoring": active_monitoring,
        "active_portfolio": active_portfolio,
        "archive": archived_monitoring,
        "runtime": runtime,
        "pulse": pulse,
        "last_scan_status": str(worker.get("last_scan_status") or scanner.get("last_status") or scanner.get("last_scan_status") or "unknown"),
        "last_scan_ts": str(worker.get("last_scan_ts") or scanner.get("last_run_ts") or ""),
        "last_scan_stats": last_stats,
        "pending": bool(worker.get("scan_request_pending") or scanner.get("scan_request_pending")),
        "pending_ts": str(worker.get("scan_request_ts") or scanner.get("scan_request_ts") or ""),
    }


def metric_card(title: str, value: Any, detail: str = "") -> html.Div:
    return html.Div([html.Div(title, className="metric-title"), html.Div(str(value), className="metric-value"), html.Div(detail, className="metric-detail")], className="metric-card")


def data_table(rows: List[Dict[str, Any]], columns: List[str], table_id: str) -> DataTable:
    return DataTable(
        id=table_id,
        data=[{c: row.get(c, "") for c in columns} for row in rows],
        columns=[{"name": c, "id": c} for c in columns],
        page_size=DASH_TABLE_PAGE_SIZE,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": 12, "textAlign": "left", "maxWidth": 260, "overflow": "hidden", "textOverflow": "ellipsis"},
        style_header={"fontWeight": "700"},
    )


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=260, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def monitoring_sparkline_figure(token: str) -> go.Figure:
    rows = [row for row in read_monitoring_history_rows() if token_key(row) == token]
    if not rows:
        return empty_figure("Select a token with monitoring snapshots")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.get("ts_utc"), errors="coerce")
    df["price_usd_num"] = pd.to_numeric(df.get("price_usd"), errors="coerce")
    df = df.dropna(subset=["ts", "price_usd_num"]).sort_values("ts")
    if df.empty:
        return empty_figure("No plottable price snapshots for this token")
    fig = px.line(df, x="ts", y="price_usd_num", title=f"{token} price sparkline")
    fig.update_layout(height=280, margin={"l": 30, "r": 20, "t": 50, "b": 30}, yaxis_title="price USD", xaxis_title="")
    return fig


def portfolio_action_timeline_figure() -> go.Figure:
    rows = read_portfolio_reco_rows()
    if not rows:
        return empty_figure("No portfolio recommendation log rows found")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.get("ts_utc"), errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return empty_figure("No timestamped portfolio actions found")
    grouped = df.groupby([pd.Grouper(key="ts", freq="1D"), "final_action"]).size().reset_index(name="count")
    fig = px.bar(grouped, x="ts", y="count", color="final_action", title="Portfolio action timeline")
    fig.update_layout(height=320, margin={"l": 30, "r": 20, "t": 50, "b": 30}, xaxis_title="", yaxis_title="actions")
    return fig


def scan_pipeline_counters_figure(runtime: Dict[str, Any]) -> go.Figure:
    runs = runtime.get("job_runs") if isinstance(runtime.get("job_runs"), list) else []
    if not runs:
        stats = dashboard_snapshot().get("last_scan_stats", {})
        counters = {k: parse_float(v, 0) for k, v in stats.items() if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit()}
        if not counters:
            return empty_figure("No scan pipeline counters available")
        fig = px.bar(x=list(counters.keys()), y=list(counters.values()), title="Last scan counters")
        fig.update_layout(height=300, margin={"l": 30, "r": 20, "t": 50, "b": 80}, xaxis_title="counter", yaxis_title="value")
        return fig
    df = pd.DataFrame(runs)
    df["ts"] = pd.to_datetime(df.get("ended_ts").where(df.get("ended_ts") != "", df.get("started_ts")), errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return empty_figure("No timestamped job runs available")
    grouped = df.groupby([pd.Grouper(key="ts", freq="1D"), "status"]).size().reset_index(name="count")
    fig = px.line(grouped, x="ts", y="count", color="status", markers=True, title="Scan/job counters over time")
    fig.update_layout(height=300, margin={"l": 30, "r": 20, "t": 50, "b": 30}, xaxis_title="", yaxis_title="runs")
    return fig


def layout_shell() -> html.Div:
    return html.Div([
        dcc.Location(id="url"),
        dcc.Interval(id="refresh", interval=DASH_CACHE_TTL_SEC * 1000, n_intervals=0),
        html.Div([html.H1("DEX Scout Dash"), html.Div([dcc.Link(label, href=href, className="nav-link") for href, label in PAGE_LINKS], className="nav")], className="topbar"),
        html.Div(id="page"),
    ])


def overview_page() -> html.Div:
    snap = dashboard_snapshot()
    stats = snap["last_scan_stats"]
    return html.Div([
        html.Div([
            metric_card("Active Monitoring", len(snap["active_monitoring"]), "D1 monitoring.csv"),
            metric_card("Portfolio", len(snap["active_portfolio"]), "active positions"),
            metric_card("Live Pulse clean", snap["pulse"]["clean_candidates"], "clean candidates"),
            metric_card("Pending scan", "yes" if snap["pending"] else "no", snap["pending_ts"]),
            metric_card("Last scan", snap["last_scan_status"], snap["last_scan_ts"]),
            metric_card("Runtime health", "ok" if snap["runtime"]["contract"].get("ok") else "check", snap["runtime"]["contract"].get("code", "")),
            metric_card("Hard/dead pulse", len(snap["pulse"]["hard_dead_candidates"]), "gated/dead cards"),
        ], className="metrics-grid"),
        html.H3("Last scan stats"),
        html.Pre(json.dumps(stats, indent=2, ensure_ascii=False) if stats else "No last scan stats available"),
        dcc.Graph(figure=scan_pipeline_counters_figure(snap["runtime"])),
    ], className="page-wrap")


def monitoring_page() -> html.Div:
    snap = dashboard_snapshot()
    options = [{"label": f"{display_symbol(row)} · {token_key(row)}", "value": token_key(row)} for row in snap["active_monitoring"]]
    cols = ["base_symbol", "chain", "active", "last_score", "last_decision", "entry_status", "risk_level", "updated_at", "weak_reason"]
    return html.Div([
        html.H2("Monitoring"),
        metric_card("Active Monitoring", len(snap["active_monitoring"]), "cached D1 read"),
        html.Div([dcc.Dropdown(id="token-select", options=options, value=options[0]["value"] if options else None, placeholder="Select token for details/sparkline"), dcc.Graph(id="token-sparkline")], className="chart-card"),
        data_table(snap["active_monitoring"], cols, "monitoring-table"),
    ], className="page-wrap")


def portfolio_page() -> html.Div:
    snap = dashboard_snapshot()
    cols = ["base_symbol", "chain", "active", "action", "score", "entry_price_usd", "avg_entry_price", "updated_at", "note"]
    return html.Div([html.H2("Portfolio"), metric_card("Portfolio", len(snap["active_portfolio"]), "active D1 rows"), dcc.Graph(figure=portfolio_action_timeline_figure()), data_table(snap["active_portfolio"], cols, "portfolio-table")], className="page-wrap")


def live_pulse_page() -> html.Div:
    snap = dashboard_snapshot()
    pulse = snap["pulse"]
    rows = pulse["final_candidates"] or pulse["raw_cards"]
    hard = pulse["hard_dead_candidates"]
    cols = sorted({k for row in rows[:25] for k in row.keys()})[:18] if rows else ["symbol", "chain", "decision", "weak_reason"]
    hard_cols = sorted({k for row in hard[:25] for k in row.keys()})[:18] if hard else cols
    return html.Div([
        html.H2("Live Pulse"),
        html.Div([metric_card("Clean candidates", pulse["clean_candidates"]), metric_card("Hard-gated/dead", len(hard)), metric_card("Rejected", sum(int(parse_float(v, 0)) for v in pulse["rejected_candidates"].values()))], className="metrics-grid small"),
        html.H3("Clean/final candidates"),
        data_table(rows, cols, "pulse-table"),
        html.H3("Hard-gated / dead candidates"),
        data_table(hard, hard_cols, "pulse-hard-table"),
    ], className="page-wrap")


def runtime_page() -> html.Div:
    snap = dashboard_snapshot()
    runtime = snap["runtime"]
    worker = runtime["worker_runtime"]
    heartbeat_cols = ["job_name", "job_mode", "status", "heartbeat_ts", "heartbeat_epoch"]
    return html.Div([
        html.H2("Runtime"),
        html.Div([metric_card("D1", "ok" if snap["d1_ok"] else "disabled", snap["d1_proxy_url"]), metric_card("Contract", runtime["contract"].get("code"), "ok" if runtime["contract"].get("ok") else "not ok"), metric_card("Last heartbeat", worker.get("last_heartbeat_ts", "")), metric_card("Last loop", worker.get("last_loop_ts", ""))], className="metrics-grid"),
        html.H3("Worker runtime"),
        html.Pre(json.dumps(worker, indent=2, ensure_ascii=False)),
        html.H3("Job heartbeats"),
        data_table(runtime["heartbeats"], heartbeat_cols, "heartbeats-table"),
    ], className="page-wrap")


def archive_page() -> html.Div:
    snap = dashboard_snapshot()
    reason_counts = Counter(str(row.get("archived_reason") or row.get("status") or "unknown") for row in snap["archive"])
    cols = ["base_symbol", "chain", "active", "status", "lifecycle_state", "ts_archived", "archived_reason", "weak_reason", "updated_at"]
    return html.Div([html.H2("Archive"), metric_card("Archived / inactive", len(snap["archive"])), html.Pre(json.dumps(dict(reason_counts.most_common(20)), indent=2)), data_table(snap["archive"], cols, "archive-table")], className="page-wrap")


def route_page(pathname: str) -> html.Div:
    if pathname == "/monitoring":
        return monitoring_page()
    if pathname == "/portfolio":
        return portfolio_page()
    if pathname == "/live-pulse":
        return live_pulse_page()
    if pathname == "/runtime":
        return runtime_page()
    if pathname == "/archive":
        return archive_page()
    return overview_page()


app = Dash(__name__, title="DEX Scout Dash", suppress_callback_exceptions=True)
server = app.server
app.layout = layout_shell

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; background: #f6f7fb; color: #121826; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
            .topbar { background: #121826; color: white; padding: 18px 26px; }
            .topbar h1 { margin: 0 0 12px 0; font-size: 24px; }
            .nav { display: flex; gap: 12px; flex-wrap: wrap; }
            .nav-link { color: #d7e3ff; text-decoration: none; font-weight: 650; }
            .nav-link:hover { color: white; }
            .page-wrap { padding: 22px 26px 42px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 22px; }
            .metrics-grid.small { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
            .metric-card, .chart-card, pre { background: white; border: 1px solid #e3e8f2; border-radius: 14px; padding: 16px; box-shadow: 0 1px 2px rgba(20, 30, 55, 0.05); }
            .metric-title { color: #64708a; font-size: 12px; font-weight: 700; letter-spacing: .03em; text-transform: uppercase; }
            .metric-value { font-size: 30px; font-weight: 800; margin: 7px 0; }
            .metric-detail { color: #64708a; font-size: 12px; overflow-wrap: anywhere; }
            h2, h3 { margin-top: 0; }
            pre { white-space: pre-wrap; overflow-x: auto; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""


@app.callback(Output("page", "children"), Input("url", "pathname"), Input("refresh", "n_intervals"))
def render_page(pathname: str, _n_intervals: int) -> html.Div:
    return route_page(pathname or "/")


@app.callback(Output("token-sparkline", "figure"), Input("token-select", "value"), prevent_initial_call=False)
def render_token_sparkline(selected_token: Optional[str]) -> go.Figure:
    if not selected_token:
        return empty_figure("Select a token to load its history")
    return monitoring_sparkline_figure(selected_token)


if __name__ == "__main__":
    app.run_server(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8050")), debug=os.getenv("DASH_DEBUG", "0") == "1")
