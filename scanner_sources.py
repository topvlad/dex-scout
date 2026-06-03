"""Import-safe scanner source adapters.

Network/API work is intentionally confined to explicit adapter functions.  This
module must remain safe to import from runtime smoke checks and tests.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests  # noqa: F401 - allowed for future adapter-owned HTTP calls.
import config  # noqa: F401 - import-safe configuration constants.
from runtime_core import normalize_chain_name, safe_get

_ALLOWED_STATUSES = {"success", "empty", "failed", "skipped", "disabled", "rate_limited", "timeout"}
_SECRET_HINTS = ("token", "secret", "password", "apikey", "api_key", "authorization", "bearer", "chat_id")
_MAX_ERROR_LEN = 240
_MAX_META_ERROR_LEN = 180


def _sanitize_text(value: Any, *, limit: int = _MAX_ERROR_LEN) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    for hint in _SECRET_HINTS:
        lowered = text.lower()
        idx = lowered.find(hint)
        if idx >= 0:
            # Keep the diagnostic class while removing anything that may include credentials.
            text = (text[:idx].rstrip(" :=") + f" {hint}=<redacted>").strip()
            break
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _bounded_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    clean: Dict[str, Any] = {}
    for key, value in meta.items():
        key_s = _sanitize_text(key, limit=80)
        if not key_s or any(h in key_s.lower() for h in _SECRET_HINTS):
            continue
        if isinstance(value, str):
            clean[key_s] = _sanitize_text(value, limit=_MAX_META_ERROR_LEN)
        elif isinstance(value, dict):
            nested: Dict[str, Any] = {}
            for nk, nv in list(value.items())[:20]:
                nk_s = _sanitize_text(nk, limit=80)
                if not nk_s or any(h in nk_s.lower() for h in _SECRET_HINTS):
                    continue
                nested[nk_s] = _sanitize_text(nv, limit=_MAX_META_ERROR_LEN) if isinstance(nv, str) else nv
            clean[key_s] = nested
        elif isinstance(value, list):
            clean[key_s] = value[:50]
        else:
            clean[key_s] = value
    return clean


def make_source_result(
    *,
    source: str,
    ok: bool,
    status: str,
    raw_candidates: list[dict] | None = None,
    seeds_used: list[str] | None = None,
    error: str = "",
    http_status: int | None = None,
    duration_ms: int | None = None,
    meta: dict | None = None,
) -> dict:
    """Return the canonical scanner source result shape."""
    candidates = [dict(row or {}) for row in list(raw_candidates or []) if isinstance(row, dict)]
    status_s = str(status or "failed").strip().lower()
    if status_s not in _ALLOWED_STATUSES:
        status_s = "failed" if not ok else "success"
    duration = 0 if duration_ms is None else max(0, int(duration_ms or 0))
    return {
        "source": str(source or "unknown").strip() or "unknown",
        "ok": bool(ok),
        "status": status_s,
        "raw_count": len(candidates),
        "raw_candidates": candidates,
        "seeds_used": [str(seed) for seed in list(seeds_used or []) if str(seed or "").strip()],
        "error": _sanitize_text(error),
        "http_status": http_status if http_status is None else int(http_status),
        "duration_ms": duration,
        "meta": _bounded_meta(meta),
    }


def _candidate_identity(row: Dict[str, Any]) -> str:
    chain = normalize_chain_name(row.get("chain") or row.get("chainId") or "")
    pair = str(row.get("pairAddress") or row.get("pair_addr") or row.get("pair") or "").strip()
    base = str(row.get("base_addr") or row.get("base_token_address") or safe_get(row, "baseToken", "address", default="") or row.get("address") or "").strip()
    if chain != "solana":
        pair = pair.lower()
        base = base.lower()
    if chain and base:
        return f"{chain}|ca:{base}"
    if chain and pair:
        return f"{chain}|pair:{pair}"
    if base:
        return f"ca:{base}"
    if pair:
        return f"pair:{pair}"
    return ""


def _dedupe_candidates(rows: Iterable[Dict[str, Any]], *, max_items: int) -> tuple[List[Dict[str, Any]], int, bool]:
    out: List[Dict[str, Any]] = []
    seen = set()
    duplicates = 0
    capped = False
    limit = max(0, int(max_items or 0))
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _candidate_identity(row)
        if key and key in seen:
            duplicates += 1
            continue
        if key:
            seen.add(key)
        if len(out) >= limit:
            capped = True
            continue
        out.append(dict(row))
    return out, duplicates, capped


def _call_fetch_pairs(fetch_fn: Callable[..., Any], seed: str, timeout_sec: int) -> list[dict]:
    try:
        result = fetch_fn(seed, timeout=timeout_sec)
    except TypeError:
        result = fetch_fn(seed)
    if isinstance(result, dict):
        result = result.get("pairs") or result.get("raw_candidates") or result.get("candidates") or []
    return [dict(row or {}) for row in list(result or []) if isinstance(row, dict)]


def fetch_dexscreener_search_source(
    *,
    seeds: list[str],
    fetch_pairs_by_query_fn=None,
    max_items: int = 20,
    timeout_sec: int = 12,
) -> dict:
    """Fetch DexScreener search candidates for explicit seeds."""
    started = time.time()
    clean_seeds = [str(seed).strip() for seed in list(seeds or []) if str(seed or "").strip()]
    if not clean_seeds:
        return make_source_result(source="dexscreener_search", ok=True, status="empty", raw_candidates=[], meta={"seeds_requested": 0, "seeds_used": 0, "seeds_failed": 0, "duplicates": 0, "capped": False, "errors_by_seed": {}})
    fetch_fn = fetch_pairs_by_query_fn
    if fetch_fn is None:
        from dex import fetch_pairs_by_query as fetch_fn  # import only when the adapter is explicitly called

    rows: List[Dict[str, Any]] = []
    used: List[str] = []
    errors_by_seed: Dict[str, str] = {}
    for seed in clean_seeds:
        try:
            got = _call_fetch_pairs(fetch_fn, seed, int(timeout_sec or 0))
            if got:
                used.append(seed)
                rows.extend(got)
            elif seed not in used:
                used.append(seed)
        except Exception as exc:  # per-seed visibility without failing whole source
            if len(errors_by_seed) < 5:
                errors_by_seed[seed] = _sanitize_text(f"{type(exc).__name__}:{exc}", limit=_MAX_META_ERROR_LEN)
    deduped, duplicates, capped = _dedupe_candidates(rows, max_items=max_items)
    failed = len(errors_by_seed)
    status = "success" if deduped else ("failed" if failed >= len(clean_seeds) else "empty")
    ok = status != "failed"
    return make_source_result(
        source="dexscreener_search",
        ok=ok,
        status=status,
        raw_candidates=deduped,
        seeds_used=used,
        duration_ms=int((time.time() - started) * 1000),
        meta={"seeds_requested": len(clean_seeds), "seeds_used": len(used), "seeds_failed": failed, "duplicates": duplicates, "capped": capped, "errors_by_seed": errors_by_seed},
    )


def fetch_birdeye_trending_source(
    *,
    enabled: bool,
    fetch_fn=None,
    limit: int = 20,
    timeout_sec: int = 12,
) -> dict:
    """Fetch Birdeye trending candidates through an injected import-safe callable."""
    if not enabled:
        return make_source_result(source="birdeye_trending", ok=True, status="disabled", raw_candidates=[], meta={"limit": int(limit or 0)})
    started = time.time()
    if fetch_fn is None:
        return make_source_result(source="birdeye_trending", ok=False, status="failed", raw_candidates=[], error="fetch_fn_required", duration_ms=int((time.time() - started) * 1000), meta={"limit": int(limit or 0)})
    try:
        try:
            result = fetch_fn(limit=int(limit or 0), timeout=timeout_sec)
        except TypeError:
            result = fetch_fn(limit=int(limit or 0))
        if isinstance(result, dict):
            result = result.get("pairs") or result.get("raw_candidates") or result.get("candidates") or result.get("tokens") or []
        rows = [dict(row or {}) for row in list(result or []) if isinstance(row, dict)]
        deduped, duplicates, capped = _dedupe_candidates(rows, max_items=limit)
        return make_source_result(source="birdeye_trending", ok=True, status="success" if deduped else "empty", raw_candidates=deduped, duration_ms=int((time.time() - started) * 1000), meta={"limit": int(limit or 0), "duplicates": duplicates, "capped": capped})
    except Exception as exc:
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        status = "timeout" if "timeout" in name or "timeout" in text else "rate_limited" if "429" in text or "rate" in text else "failed"
        return make_source_result(source="birdeye_trending", ok=False, status=status, raw_candidates=[], error=f"{type(exc).__name__}:{exc}", duration_ms=int((time.time() - started) * 1000), meta={"limit": int(limit or 0)})


def collect_scanner_sources(*, source_fns: list, max_total: int = 50) -> dict:
    """Run source callables, keep each result visible, and aggregate candidates."""
    results: List[Dict[str, Any]] = []
    for fn in list(source_fns or []):
        try:
            result = fn()
            if not isinstance(result, dict):
                result = make_source_result(source="unknown", ok=False, status="failed", error="source_fn_returned_non_dict")
        except Exception as exc:
            result = make_source_result(source="unknown", ok=False, status="failed", error=f"{type(exc).__name__}:{exc}")
        results.append(result)
    raw_rows: List[Dict[str, Any]] = []
    for result in results:
        raw_rows.extend([dict(row or {}) for row in list(result.get("raw_candidates") or []) if isinstance(row, dict)])
    deduped, duplicates, capped = _dedupe_candidates(raw_rows, max_items=max_total)
    success_count = sum(1 for r in results if r.get("ok") and r.get("status") == "success")
    empty_count = sum(1 for r in results if r.get("ok") and r.get("status") == "empty")
    failed_count = sum(1 for r in results if (not r.get("ok")) or r.get("status") in {"failed", "timeout", "rate_limited"})
    disabled_count = sum(1 for r in results if r.get("status") == "disabled")
    if success_count and failed_count:
        status = "partial"
    elif deduped or success_count:
        status = "success"
    elif results and failed_count == len(results):
        status = "failed"
    elif not results:
        status = "empty"
    else:
        status = "empty"
    return {
        "ok": status != "failed",
        "status": status,
        "raw_candidates": deduped,
        "sources_tried": [str(r.get("source") or "unknown") for r in results],
        "source_results": results,
        "diagnostics": {
            "sources_total": len(results),
            "sources_success": success_count,
            "sources_empty": empty_count,
            "sources_failed": failed_count,
            "sources_disabled": disabled_count,
            "raw_total": len(raw_rows),
            "raw_deduped": len(deduped),
            "duplicates": duplicates,
            "capped": capped,
        },
    }
