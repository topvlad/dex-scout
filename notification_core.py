"""Import-safe notification event keys and exactly-once journal helpers.

This module intentionally has no Streamlit, app, network, or storage imports.  Callers
provide storage adapters where persistence is needed.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

MATERIAL_PORTFOLIO_ACTIONS = {
    "EXIT",
    "REDUCE",
    "TAKE PROFIT",
    "TAKE_PROFIT",
    "PARTIAL_TP",
    "PROTECT",
    "CLOSE",
    "TRIM",
}

_MATERIAL_ACTION_KEYS = {a.replace(" ", "_") for a in MATERIAL_PORTFOLIO_ACTIONS} | {a.replace("_", " ") for a in MATERIAL_PORTFOLIO_ACTIONS}


def normalize_notification_action(value: Any) -> str:
    """Normalize a notification action/verdict token for stable keys."""
    raw = str(value or "").strip().upper().replace("-", "_")
    raw = re.sub(r"\s+", "_", raw)
    return re.sub(r"[^A-Z0-9_]+", "_", raw).strip("_")


def _compact_token(value: Any, default: str = "") -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9_:\-.]+", "_", text).strip("_") or default


def _bucket_token(value: Any, default: str = "stable") -> str:
    text = str(value or "").strip()
    return re.sub(r"[^a-zA-Z0-9_:\-.]+", "_", text).strip("_") or default


def _event_chain(event: Dict[str, Any]) -> str:
    row = event.get("row") if isinstance(event.get("row"), dict) else {}
    return _compact_token(event.get("chain") or row.get("chain") or row.get("network"), "unknown_chain")


def _event_addr(event: Dict[str, Any]) -> str:
    row = event.get("row") if isinstance(event.get("row"), dict) else {}
    return _compact_token(
        event.get("token_address")
        or event.get("token_addr")
        or event.get("base_token_address")
        or event.get("base_addr")
        or row.get("token_address")
        or row.get("token_addr")
        or row.get("base_token_address")
        or row.get("base_addr")
        or row.get("pair_address")
        or row.get("pairAddress"),
        "no_token",
    )


def _event_action(event: Dict[str, Any]) -> str:
    row = event.get("row") if isinstance(event.get("row"), dict) else {}
    signal = event.get("signal") if isinstance(event.get("signal"), dict) else {}
    unified = event.get("unified") if isinstance(event.get("unified"), dict) else {}
    return normalize_notification_action(
        event.get("action")
        or unified.get("final_action")
        or row.get("final_action")
        or row.get("recommended_action")
        or row.get("position_action")
        or row.get("entry_action")
        or row.get("entry")
        or signal.get("bucket")
        or "NONE"
    ) or "NONE"


def _event_verdict(event: Dict[str, Any]) -> str:
    row = event.get("row") if isinstance(event.get("row"), dict) else {}
    return normalize_notification_action(
        event.get("verdict")
        or row.get("verdict")
        or row.get("health_label")
        or row.get("risk_level")
        or row.get("risk")
        or "NA"
    ) or "NA"


def is_material_portfolio_action(value: str | Dict[str, Any]) -> bool:
    """Return whether a raw action or row/event dict carries a material action."""
    candidates = []
    if isinstance(value, dict):
        candidates.extend(
            [
                value.get("final_action"),
                value.get("recommended_action"),
                value.get("position_action"),
                value.get("entry_action"),
                value.get("action"),
                value.get("bucket"),
                value.get("verdict"),
            ]
        )
        row = value.get("row") if isinstance(value.get("row"), dict) else {}
        candidates.extend([row.get("final_action"), row.get("recommended_action"), row.get("position_action"), row.get("entry_action")])
    else:
        candidates.append(value)
    for candidate in candidates:
        norm = normalize_notification_action(candidate)
        if norm and (norm in _MATERIAL_ACTION_KEYS or norm.replace("_", " ") in _MATERIAL_ACTION_KEYS):
            return True
    return False


def build_notification_event_key(event: Dict[str, Any], bucket_ts: str | None = None) -> str:
    """Build a deterministic notification event key from semantic event fields."""
    event = event if isinstance(event, dict) else {}
    event_type = _compact_token(event.get("event_type") or event.get("type") or "notification", "notification")
    source = _compact_token(event.get("source") or "notify_cycle", "notify_cycle")
    chain = _event_chain(event)

    digest_bucket = bucket_ts or event.get("bucket_ts") or event.get("bucket") or event.get("heartbeat_bucket")
    if event_type in {"digest", "heartbeat", "health_warning"} or event.get("digest_path") or event.get("heartbeat_kind"):
        digest_path = _compact_token(event.get("digest_path") or event.get("heartbeat_kind") or event.get("kind") or source, "digest")
        return f"{event_type}|{source}|{chain}|{digest_path}|bucket:{_bucket_token(digest_bucket)}"

    addr = _event_addr(event)
    action = _event_action(event)
    verdict = _event_verdict(event)
    bucket = _bucket_token(event.get("bucket") or bucket_ts)
    return f"{event_type}|{source}|{chain}|{addr}|{action}|{verdict}|{bucket}"


def build_emission_key(event: Dict[str, Any], cooldown_bucket: int | str) -> str:
    event = event if isinstance(event, dict) else {}
    return "|".join(
        [
            _compact_token(event.get("source") or "notify_cycle", "notify_cycle"),
            _event_chain(event),
            _event_addr(event),
            _compact_token(event.get("event_type") or "notification", "notification"),
            normalize_notification_action(event.get("action_bucket") or event.get("action") or "UNKNOWN") or "UNKNOWN",
            _compact_token(event.get("reason_bucket") or event.get("reason") or "unknown_reason", "unknown_reason"),
            str(cooldown_bucket),
        ]
    )


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith(" UTC"):
        text = text[:-4] + "+00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def _ts_float(value: Any) -> float:
    dt = _parse_ts(value)
    return dt.timestamp() if dt else 0.0


def _event_sort_ts(event: Dict[str, Any]) -> float:
    for key in ("last_seen_ts", "send_ts", "first_seen_ts"):
        ts = _ts_float(event.get(key))
        if ts:
            return ts
    return 0.0


def notification_journal_default() -> Dict[str, Any]:
    return {"version": 1, "updated_ts": "", "events": {}}


def trim_notification_event_journal(journal: Dict[str, Any], max_events: Optional[int] = None) -> int:
    journal = journal if isinstance(journal, dict) else notification_journal_default()
    events = journal.get("events") if isinstance(journal.get("events"), dict) else {}
    limit = max(1, int(max_events or len(events) or 1))
    if len(events) <= limit:
        return 0
    overflow = len(events) - limit
    removed = []
    for statuses in (("sent", "skipped"), ("pending",), ("failed",)):
        if len(removed) >= overflow:
            break
        candidates = [
            (key, _event_sort_ts(value if isinstance(value, dict) else {}))
            for key, value in events.items()
            if isinstance(value, dict) and str(value.get("send_status") or "").strip().lower() in statuses
        ]
        candidates.sort(key=lambda item: item[1])
        for key, _ in candidates:
            if len(removed) >= overflow:
                break
            removed.append(key)
    for key in removed:
        events.pop(key, None)
    journal["events"] = events
    journal["notification_journal_trimmed"] = int(journal.get("notification_journal_trimmed", 0) or 0) + len(removed)
    return len(removed)


def summarize_notification_event_journal(journal: Dict[str, Any]) -> Dict[str, Any]:
    journal = journal if isinstance(journal, dict) else {}
    events = journal.get("events") if isinstance(journal.get("events"), dict) else {}
    summary = {
        "journal_size": len(events),
        "sent_count": 0,
        "failed_count": 0,
        "pending_count": 0,
        "skipped_count": 0,
        "last_sent_ts": "",
        "last_failed_ts": "",
        "last_failed_reason": "",
        "trimmed_count": int(journal.get("notification_journal_trimmed", journal.get("trimmed_count", 0)) or 0),
    }
    last_sent = -1.0
    last_failed = -1.0
    for ev in events.values():
        if not isinstance(ev, dict):
            continue
        status = str(ev.get("send_status") or "").strip().lower()
        reason = str(ev.get("last_reason") or "").strip().lower()
        ts = _event_sort_ts(ev)
        if status == "sent":
            summary["sent_count"] += 1
            if ts >= last_sent:
                last_sent = ts
                summary["last_sent_ts"] = str(ev.get("send_ts") or ev.get("last_seen_ts") or "")
        elif status == "failed":
            summary["failed_count"] += 1
            if ts >= last_failed:
                last_failed = ts
                summary["last_failed_ts"] = str(ev.get("last_seen_ts") or ev.get("send_ts") or "")
                summary["last_failed_reason"] = str(ev.get("send_error") or ev.get("last_reason") or "")
        elif status == "pending":
            summary["pending_count"] += 1
        if reason == "skipped_duplicate" or (status == "skipped" and reason in {"", "skipped_duplicate"}):
            summary["skipped_count"] += 1
    return summary


def _base_event(event: Dict[str, Any], event_key: str, now_ts: str, status: str = "pending") -> Dict[str, Any]:
    row = event.get("row") if isinstance(event.get("row"), dict) else {}
    return {
        "event_key": event_key,
        "emission_key": str(event.get("emission_key") or event_key),
        "event_type": str(event.get("event_type") or ""),
        "token_key": str(event.get("token_key") or ""),
        "chain": str(event.get("chain") or row.get("chain") or ""),
        "token_addr": str(event.get("token_addr") or event.get("token_address") or row.get("token_address") or row.get("base_token_address") or ""),
        "first_seen_ts": now_ts,
        "last_seen_ts": now_ts,
        "send_status": status,
        "send_ts": "",
        "send_error": "",
        "send_attempts": 0,
        "last_reason": str(event.get("reason") or ""),
        "source": str(event.get("source") or "notify_cycle"),
    }


def should_skip_notification_event(journal: Dict[str, Any], event_key: str, now_ts: str, retry_cooldown_sec: int) -> Dict[str, Any]:
    events = journal.get("events") if isinstance(journal.get("events"), dict) else {}
    existing = events.get(event_key) if isinstance(events.get(event_key), dict) else {}
    status = str(existing.get("send_status") or "").strip().lower()
    if status == "sent":
        return {"skip": True, "decision": "skip_duplicate", "reason": "skipped_duplicate", "event": existing}
    if status == "failed":
        last = _ts_float(existing.get("last_seen_ts") or existing.get("send_ts") or existing.get("first_seen_ts"))
        now = _ts_float(now_ts)
        if last and now and (now - last) < max(0, int(retry_cooldown_sec or 0)):
            return {"skip": True, "decision": "skip_retry_cooldown", "reason": "retry_cooldown", "event": existing}
    return {"skip": False, "decision": "send", "reason": "send_started", "event": existing}


def record_notification_pending(journal: Dict[str, Any], event: Dict[str, Any], event_key: str, now_ts: str) -> Dict[str, Any]:
    events = journal.setdefault("events", {})
    existing = events.get(event_key) if isinstance(events.get(event_key), dict) else {}
    entry = {**_base_event(event, event_key, now_ts, status="pending"), **existing}
    entry.update({"event_key": event_key, "last_seen_ts": now_ts, "send_status": "pending", "last_reason": str(event.get("reason") or "send_started"), "send_error": ""})
    entry["send_attempts"] = int(float(entry.get("send_attempts") or 0)) + 1
    events[event_key] = entry
    return entry


def record_notification_sent(journal: Dict[str, Any], event_key: str, now_ts: str, reason: str = "sent") -> Dict[str, Any]:
    events = journal.setdefault("events", {})
    entry = events.get(event_key) if isinstance(events.get(event_key), dict) else _base_event({}, event_key, now_ts)
    entry.update({"event_key": event_key, "last_seen_ts": now_ts, "send_status": "sent", "send_ts": now_ts, "send_error": "", "last_reason": reason or "sent"})
    events[event_key] = entry
    return entry


def record_notification_failed(journal: Dict[str, Any], event_key: str, now_ts: str, reason: str = "send_failed", send_error: str = "") -> Dict[str, Any]:
    events = journal.setdefault("events", {})
    entry = events.get(event_key) if isinstance(events.get(event_key), dict) else _base_event({}, event_key, now_ts)
    entry.update({"event_key": event_key, "last_seen_ts": now_ts, "send_status": "failed", "send_error": send_error, "last_reason": reason or "send_failed"})
    events[event_key] = entry
    return entry


def guard_notification_event(
    event: Dict[str, Any],
    load_journal: Callable[[], Dict[str, Any]],
    save_journal: Callable[[Dict[str, Any]], bool],
    now_ts: str,
    retry_cooldown_sec: int,
    max_events: int,
) -> Dict[str, Any]:
    """Reserve a notification event for sending unless exactly-once rules block it."""
    try:
        event = event if isinstance(event, dict) else {}
        event_key = str(event.get("event_key") or build_notification_event_key(event))
        journal = load_journal() or notification_journal_default()
        if not isinstance(journal, dict):
            journal = notification_journal_default()
        events = journal.setdefault("events", {})
        if not isinstance(events, dict):
            events = {}
            journal["events"] = events
        skip = should_skip_notification_event(journal, event_key, now_ts, retry_cooldown_sec)
        if skip.get("skip"):
            existing = events.get(event_key) if isinstance(events.get(event_key), dict) else _base_event(event, event_key, now_ts)
            existing["last_seen_ts"] = now_ts
            existing["last_reason"] = skip.get("reason") or "skipped"
            events[event_key] = existing
            trim_notification_event_journal(journal, max_events=max_events)
            saved = bool(save_journal(journal))
            return {
                "ok": False,
                "event_key": event_key,
                "decision": skip.get("decision") or "blocked",
                "reason": skip.get("reason") or "blocked",
                "journal_updated": saved,
                "send_attempts": int(float(existing.get("send_attempts") or 0)),
            }
        pending = record_notification_pending(journal, event, event_key, now_ts)
        trim_notification_event_journal(journal, max_events=max_events)
        saved = bool(save_journal(journal))
        return {
            "ok": True,
            "event_key": event_key,
            "decision": "send",
            "reason": pending.get("last_reason") or "send_started",
            "journal_updated": saved,
            "send_attempts": int(float(pending.get("send_attempts") or 0)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "event_key": str((event or {}).get("event_key") or ""),
            "decision": "error",
            "reason": f"{type(exc).__name__}:{exc}",
            "journal_updated": False,
            "send_attempts": 0,
        }
