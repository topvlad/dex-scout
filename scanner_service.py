"""Import-safe scanner and Live Pulse service helpers.

This module owns deterministic Live Pulse payload materialization and pure
scanner orchestration boundaries. It intentionally avoids Streamlit, app.py,
storage writes, Telegram, and network calls at import time.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import monitoring_core
import monitoring_service
import portfolio_service
from runtime_core import normalize_chain_name, now_utc_str, parse_float, safe_get

MAX_FINAL_PAYLOAD_CANDIDATES = 50
MAX_REJECTED_PAYLOAD_CANDIDATES = 50


def _now(now_ts: Optional[str] = None) -> str:
    return str(now_ts or now_utc_str() or datetime.now(timezone.utc).isoformat()).strip()


def _copy_dicts(rows: Optional[List[Dict[str, Any]]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    copied = [dict(row or {}) for row in list(rows or []) if isinstance(row, dict)]
    return copied if limit is None else copied[: max(0, int(limit))]


def _compact_candidate(row: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    row = dict(row or {})
    compact = {
        "chain": row.get("chain") or row.get("chainId") or "",
        "base_addr": row.get("base_addr") or row.get("base_token_address") or safe_get(row, "baseToken", "address", default="") or row.get("address") or "",
        "base_symbol": row.get("base_symbol") or row.get("symbol") or safe_get(row, "baseToken", "symbol", default="") or "",
    }
    for key in ("score", "entry_score", "priority_score", "entry_status", "status", "timing", "stage", "key"):
        if key in row and row.get(key) not in (None, ""):
            compact[key] = row.get(key)
    if reason:
        compact["reason"] = reason
    return compact


def _candidate_key(row: Dict[str, Any]) -> str:
    ident = monitoring_core.build_token_identity(row or {})
    key = str(ident.get("token_key") or "").strip()
    if key:
        return key.replace(":", "|ca:", 1)
    chain = normalize_chain_name(row.get("chain") or row.get("chainId"))
    addr = row.get("base_addr") or row.get("base_token_address") or safe_get(row, "baseToken", "address", default="") or row.get("address")
    addr_norm = monitoring_core.normalize_token_address_for_chain(chain, addr)
    return f"{chain}|ca:{addr_norm}" if chain and addr_norm else ""


def _key_set(rows: Optional[List[Dict[str, Any]]], *, active_only: bool = False, portfolio: bool = False) -> Set[str]:
    keys: Set[str] = set()
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        if active_only:
            if portfolio:
                if not monitoring_core.is_active_portfolio_row(row):
                    continue
            elif not monitoring_core.is_active_monitoring_row(row):
                continue
        key = _candidate_key(row)
        if key:
            keys.add(key)
    return keys


def _add_rejection(rejected: List[Dict[str, Any]], blocked: Dict[str, int], row: Dict[str, Any], reason: str, *, limit: int = MAX_REJECTED_PAYLOAD_CANDIDATES) -> None:
    reason = str(reason or "unknown").strip() or "unknown"
    blocked[reason] = int(blocked.get(reason, 0) or 0) + 1
    if len(rejected) < limit:
        rejected.append(_compact_candidate(row, reason))


def _empty_reason(status: str, raw_seen: int, normalized: int, final_count: int, blocked_reasons: Dict[str, int], error: str = "", source_state: dict | None = None) -> str:
    status = str(status or "success").strip().lower()
    state = source_state if isinstance(source_state, dict) else {}
    if final_count <= 0 and raw_seen <= 0 and state:
        diag = state.get("diagnostics") if isinstance(state.get("diagnostics"), dict) else {}
        total = int(diag.get("sources_total", 0) or 0)
        failed = int(diag.get("sources_failed", 0) or 0)
        empty = int(diag.get("sources_empty", 0) or 0)
        disabled = int(diag.get("sources_disabled", 0) or 0)
        if total > 0 and disabled == total:
            return "sources_disabled"
        if total > 0 and failed == total:
            return "source_api_failed"
        if total > 0 and failed > 0 and (empty > 0 or disabled > 0):
            return "source_api_empty_with_failures"
        if total > 0 and (empty + disabled) == total:
            return "source_api_empty"
    if status == "failed" or error:
        error_text = str(error or "").lower()
        return "scanner_failed" if "scanner" in error_text else "worker_failed"
    if final_count > 0:
        return ""
    if raw_seen <= 0:
        return "source_api_empty"
    if normalized <= 0:
        return "scanner_returned_no_candidates"
    if blocked_reasons:
        normalized_reasons = Counter()
        for reason, count in blocked_reasons.items():
            r = str(reason or "").lower()
            c = int(count or 0)
            if "archive" in r or "duplicate" in r or "portfolio_active" in r or "monitoring_active" in r:
                normalized_reasons["all_filtered_archive_or_duplicate"] += c
            elif "scam" in r or "toxic" in r or "dead" in r:
                normalized_reasons["all_filtered_scam"] += c
            elif "hard" in r or "no_entry" in r or "untradeable" in r:
                normalized_reasons["all_filtered_hard_gate"] += c
        if normalized_reasons:
            return normalized_reasons.most_common(1)[0][0]
    return "unknown_empty"


def determine_live_pulse_empty_reason(status: str, raw_seen: int, normalized: int, final_count: int, blocked_reasons: Dict[str, int], error: str = "", source_state: dict | None = None) -> str:
    """Return the canonical non-empty reason for an empty/failed Live Pulse payload."""
    return _empty_reason(status, raw_seen, normalized, final_count, blocked_reasons, error=error, source_state=source_state)


def build_live_pulse_payload(
    *,
    raw_candidates: list[dict],
    normalized_candidates: list[dict] | None = None,
    final_candidates: list[dict] | None = None,
    rejected_candidates: list[dict] | None = None,
    status: str = "success",
    source: str = "scan_cycle",
    sources_tried: list[str] | None = None,
    refill_attempts: int = 0,
    error: str = "",
    now_ts: str | None = None,
    source_state: dict | None = None,
) -> dict:
    raw = _copy_dicts(raw_candidates)
    normalized = _copy_dicts(normalized_candidates if normalized_candidates is not None else raw_candidates)
    final = _copy_dicts(final_candidates, MAX_FINAL_PAYLOAD_CANDIDATES)
    rejected = _copy_dicts(rejected_candidates, MAX_REJECTED_PAYLOAD_CANDIDATES)
    blocked_reasons: Dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("reason") or "unknown").strip() or "unknown"
        blocked_reasons[reason] = int(blocked_reasons.get(reason, 0) or 0) + 1
    clean_candidates = max(len(final), len(normalized) - len(rejected)) if normalized_candidates is not None else len(final)
    payload_status = str(status or "success").strip().lower()
    if payload_status not in {"success", "empty", "failed"}:
        payload_status = "failed" if error else "empty"
    if len(final) <= 0 and payload_status == "success":
        payload_status = "empty"
    reason = _empty_reason(payload_status, len(raw), len(normalized), len(final), blocked_reasons, error=error, source_state=source_state)
    return {
        "ts_utc": _now(now_ts),
        "source": str(source or "unknown") if str(source or "").strip() in {"scan_cycle", "manual", "test", "unknown"} else "unknown",
        "status": payload_status,
        "raw_seen": len(raw),
        "normalized": len(normalized),
        "clean_candidates": int(max(0, clean_candidates)),
        "final_count": len(final),
        "refill_attempts": int(refill_attempts or 0),
        "last_empty_reason": reason,
        "sources_tried": list(sources_tried or []),
        "blocked_reasons": blocked_reasons,
        "final_candidates": [_compact_candidate(row) if not ("row" in row or "best" in row) else dict(row) for row in final],
        "rejected_candidates": rejected,
        "debug": {"error": str(error or ""), "source_results": _source_results_summary(source_state), "source_diagnostics": _source_diagnostics(source_state)},
    }


def normalize_scanner_candidates(raw_candidates: list[dict], *, normalize_pair_fn=None, now_ts: str | None = None) -> dict:
    _ = now_ts
    normalized: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reasons: Dict[str, int] = {}
    seen: Set[str] = set()
    for raw in list(raw_candidates or []):
        if not isinstance(raw, dict):
            _add_rejection(rejected, reasons, {}, "malformed_pair")
            continue
        try:
            row = dict(raw)
            if normalize_pair_fn:
                maybe = normalize_pair_fn(dict(raw))
                if isinstance(maybe, dict):
                    row = maybe
            if not row:
                _add_rejection(rejected, reasons, raw, "malformed_pair")
                continue
            if not (row.get("chain") or row.get("chainId")):
                _add_rejection(rejected, reasons, row, "missing_chain")
                continue
            if not (row.get("base_addr") or row.get("base_token_address") or safe_get(row, "baseToken", "address", default="") or row.get("address")):
                _add_rejection(rejected, reasons, row, "missing_address")
                continue
            chain = normalize_chain_name(row.get("chain") or row.get("chainId"))
            row["chain"] = chain
            addr = row.get("base_addr") or row.get("base_token_address") or safe_get(row, "baseToken", "address", default="") or row.get("address")
            row["base_addr"] = monitoring_core.normalize_token_address_for_chain(chain, addr)
            if not row.get("base_symbol"):
                sym = row.get("symbol") or safe_get(row, "baseToken", "symbol", default="")
                if sym:
                    row["base_symbol"] = str(sym).strip()
            key = _candidate_key(row)
            if not key:
                _add_rejection(rejected, reasons, row, "unknown")
                continue
            if key in seen:
                _add_rejection(rejected, reasons, row, "duplicate")
                continue
            seen.add(key)
            normalized.append(row)
        except Exception as exc:
            sample = dict(raw)
            sample["error"] = type(exc).__name__
            _add_rejection(rejected, reasons, sample, "normalize_exception")
    return {"normalized_candidates": normalized, "rejected_candidates": rejected, "diagnostics": {"raw_seen": len(list(raw_candidates or [])), "normalized": len(normalized), "rejected": len(rejected), "reasons": reasons}}


def filter_live_pulse_candidates(
    candidates: list[dict],
    *,
    monitoring_rows: list[dict] | None = None,
    portfolio_rows: list[dict] | None = None,
    archive_rows: list[dict] | None = None,
    existing_keys: set[str] | None = None,
    hard_gate_fn=None,
    max_candidates: int = 15,
) -> dict:
    rows = [dict(row or {}) for row in list(candidates or []) if isinstance(row, dict)]
    rejected: List[Dict[str, Any]] = []
    blocked: Dict[str, int] = {}
    final: List[Dict[str, Any]] = []
    diag = {"seen": len(rows), "duplicates": 0, "archived": 0, "portfolio_active": 0, "monitoring_active": 0, "scam_or_toxic": 0, "hard_gated": 0, "missing_score": 0, "below_threshold": 0, "final_count": 0}
    seen: Set[str] = set()
    monitor_keys = _key_set(monitoring_rows, active_only=True)
    portfolio_keys = _key_set(portfolio_rows, active_only=True, portfolio=True)
    archive_keys = _key_set(archive_rows)
    existing = {str(k).replace(":", "|ca:", 1) for k in set(existing_keys or set()) if str(k or "").strip()}
    for row in rows:
        key = _candidate_key(row)
        status_text = " ".join(str(row.get(k) or "") for k in ("entry_status", "status", "final_action", "health_label", "risk_flags", "toxic_flags", "archived_reason")).lower()
        if not key or key in seen or key in existing:
            diag["duplicates"] += 1
            _add_rejection(rejected, blocked, row, "duplicate")
            continue
        seen.add(key)
        if key in archive_keys or any(x in status_text for x in ("archived", "archive")):
            diag["archived"] += 1
            _add_rejection(rejected, blocked, row, "archived")
            continue
        if key in portfolio_keys:
            diag["portfolio_active"] += 1
            _add_rejection(rejected, blocked, row, "portfolio_active")
            continue
        if key in monitor_keys:
            diag["monitoring_active"] += 1
            _add_rejection(rejected, blocked, row, "monitoring_active")
            continue
        if any(x in status_text for x in ("scam", "toxic", "honeypot", "rug", "dead")):
            diag["scam_or_toxic"] += 1
            _add_rejection(rejected, blocked, row, "scam_or_toxic")
            continue
        if any(x in status_text for x in ("no_entry", "no entry", "untradeable", "hard_gated")):
            diag["hard_gated"] += 1
            _add_rejection(rejected, blocked, row, "hard_gated")
            continue
        gate = None
        try:
            gate = hard_gate_fn(row) if hard_gate_fn else monitoring_core.hard_gate_monitoring_row(row)
        except Exception as exc:
            gate = {"blocked": True, "reason": f"hard_gate_exception:{type(exc).__name__}"}
        if bool((gate or {}).get("blocked")):
            diag["hard_gated"] += 1
            _add_rejection(rejected, blocked, row, "hard_gated")
            continue
        score = row.get("score", row.get("entry_score", row.get("priority_score")))
        if score is None or str(score).strip() == "":
            diag["missing_score"] += 1
            _add_rejection(rejected, blocked, row, "missing_score")
            continue
        if parse_float(score, 0.0) <= 0:
            diag["below_threshold"] += 1
            _add_rejection(rejected, blocked, row, "below_threshold")
            continue
        if len(final) < max(0, int(max_candidates or 0)):
            final.append(dict(row))
    diag["final_count"] = len(final)
    return {"final_candidates": final, "rejected_candidates": rejected[:MAX_REJECTED_PAYLOAD_CANDIDATES], "blocked_reasons": blocked, "diagnostics": diag}


def plan_live_pulse_refill(*, current_count: int, target_min: int, target_max: int, max_attempts: int, sources_tried: list[str] | None = None) -> dict:
    current = max(0, int(current_count or 0))
    target_min = max(0, int(target_min or 0))
    target_max = max(target_min, int(target_max or target_min))
    attempted = len(list(sources_tried or []))
    allowed = max(0, int(max_attempts or 0) - attempted)
    if current >= target_min:
        reason = "enough_candidates"
    elif allowed <= 0:
        reason = "attempts_exhausted"
    else:
        reason = "below_target"
    return {"should_refill": reason == "below_target", "remaining_slots": max(0, target_max - current), "attempts_allowed": allowed, "target_min": target_min, "target_max": target_max, "reason": reason}



def _source_diagnostics(source_state: dict | None) -> dict:
    if not isinstance(source_state, dict):
        return {}
    diag = source_state.get("diagnostics") if isinstance(source_state.get("diagnostics"), dict) else {}
    out = dict(diag)
    if source_state.get("status"):
        out["status"] = str(source_state.get("status") or "")
    return out


def _source_results_summary(source_state: dict | None) -> list[dict]:
    if not isinstance(source_state, dict):
        return []
    out: List[Dict[str, Any]] = []
    for result in list(source_state.get("source_results") or []):
        if not isinstance(result, dict):
            continue
        out.append({
            "source": str(result.get("source") or "unknown"),
            "ok": bool(result.get("ok")),
            "status": str(result.get("status") or ""),
            "raw_count": int(result.get("raw_count", len(result.get("raw_candidates") or [])) or 0),
            "seeds_used": list(result.get("seeds_used") or [])[:10],
            "error": str(result.get("error") or "")[:240],
            "http_status": result.get("http_status"),
            "duration_ms": result.get("duration_ms"),
        })
    return out


def _normalize_source_fetch_result(fetched: Any) -> tuple[list[dict], list[str], dict]:
    if isinstance(fetched, dict):
        raw = _copy_dicts(fetched.get("raw_candidates") or fetched.get("candidates") or [])
        sources_tried = list(fetched.get("sources_tried") or [])
        if "source_results" in fetched or "diagnostics" in fetched:
            source_state = {
                "status": str(fetched.get("status") or ""),
                "sources_tried": list(sources_tried),
                "source_results": list(fetched.get("source_results") or []),
                "diagnostics": dict(fetched.get("diagnostics") or {}),
            }
            if not sources_tried:
                sources_tried = list(source_state.get("sources_tried") or [])
            return raw, sources_tried, source_state
        return raw, sources_tried, {}
    return _copy_dicts(fetched), [], {}

def run_scanner_service_cycle(
    *,
    fetch_sources_fn,
    normalize_fn,
    filter_fn,
    save_payload_fn,
    monitoring_rows: list[dict],
    portfolio_rows: list[dict],
    archive_rows: list[dict] | None = None,
    config: dict | None = None,
    now_ts: str | None = None,
) -> dict:
    cfg = dict(config or {})
    sources_tried: List[str] = []
    raw: List[Dict[str, Any]] = []
    source_state: Dict[str, Any] = {}
    try:
        fetched = fetch_sources_fn(config=deepcopy(cfg))
        raw, sources_tried, source_state = _normalize_source_fetch_result(fetched)
    except Exception as exc:
        return _failed_cycle(save_payload_fn, raw, [], [], [], sources_tried, f"{type(exc).__name__}:{exc}", now_ts, source_state)
    try:
        norm = normalize_fn(raw_candidates=_copy_dicts(raw))
        normalized = _copy_dicts(norm.get("normalized_candidates") if isinstance(norm, dict) else norm)
        norm_rejected = _copy_dicts(norm.get("rejected_candidates") if isinstance(norm, dict) else [])
    except Exception as exc:
        return _failed_cycle(save_payload_fn, raw, [], [], [], sources_tried, f"{type(exc).__name__}:{exc}", now_ts, source_state)
    try:
        filt = filter_fn(candidates=_copy_dicts(normalized), monitoring_rows=_copy_dicts(monitoring_rows), portfolio_rows=_copy_dicts(portfolio_rows), archive_rows=_copy_dicts(archive_rows))
        final = _copy_dicts(filt.get("final_candidates") if isinstance(filt, dict) else filt)
        rejected = norm_rejected + _copy_dicts(filt.get("rejected_candidates") if isinstance(filt, dict) else [])
    except Exception as exc:
        return _failed_cycle(save_payload_fn, raw, normalized, [], norm_rejected, sources_tried, f"{type(exc).__name__}:{exc}", now_ts, source_state)
    status = "success" if final else "empty"
    payload = build_live_pulse_payload(raw_candidates=raw, normalized_candidates=normalized, final_candidates=final, rejected_candidates=rejected, status=status, source=str(cfg.get("source") or "scan_cycle"), sources_tried=sources_tried, refill_attempts=int(cfg.get("refill_attempts") or 0), now_ts=now_ts, source_state=source_state)
    try:
        save_payload_fn(payload)
    except Exception as exc:
        payload["status"] = "failed"
        payload["last_empty_reason"] = "save_failed"
        payload.setdefault("debug", {})["error"] = f"{type(exc).__name__}:{exc}"
        return {"ok": False, "status": "failed", "payload": payload, "raw_seen": len(raw), "final_count": len(final), "last_empty_reason": "save_failed", "sources_tried": sources_tried, "error": f"{type(exc).__name__}:{exc}"}
    return {"ok": True, "status": status, "payload": payload, "raw_seen": len(raw), "final_count": len(final), "last_empty_reason": payload.get("last_empty_reason", ""), "sources_tried": sources_tried, "error": ""}


def _failed_cycle(save_payload_fn, raw, normalized, final, rejected, sources_tried, error, now_ts, source_state=None):
    payload = build_live_pulse_payload(raw_candidates=raw, normalized_candidates=normalized, final_candidates=final, rejected_candidates=rejected, status="failed", sources_tried=sources_tried, error=error, now_ts=now_ts, source_state=source_state)
    ok = False
    save_error = ""
    try:
        save_payload_fn(payload)
    except Exception as exc:
        save_error = f"{type(exc).__name__}:{exc}"
        payload["last_empty_reason"] = "save_failed"
    return {"ok": ok, "status": "failed", "payload": payload, "raw_seen": len(raw), "final_count": len(final), "last_empty_reason": payload.get("last_empty_reason", "worker_failed"), "sources_tried": list(sources_tried or []), "error": save_error or error}
