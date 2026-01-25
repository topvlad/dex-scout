# alerts.py
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from config import WATCH_TTL_MINUTES, WATCH_MAX_ITEMS, ALERT_COOLDOWN_SECONDS

Key = Tuple[str, str]  # (chain, pairAddress)

@dataclass
class WatchItem:
    first_seen: float
    last_seen: float
    last_alert: float
    data: dict

def _now() -> float:
    return time.time()

def make_key(row: dict) -> Key:
    return (row.get("chain") or "", row.get("pairAddress") or "")

def purge_expired(store: Dict[Key, WatchItem]) -> None:
    """Drop items older than TTL."""
    ttl = WATCH_TTL_MINUTES * 60
    t = _now()
    dead = [k for k, v in store.items() if (t - v.last_seen) > ttl]
    for k in dead:
        store.pop(k, None)

    # Hard cap
    if len(store) > WATCH_MAX_ITEMS:
        # remove oldest by last_seen
        oldest = sorted(store.items(), key=lambda kv: kv[1].last_seen)[: max(0, len(store) - WATCH_MAX_ITEMS)]
        for k, _ in oldest:
            store.pop(k, None)

def upsert_watchlist(store: Dict[Key, WatchItem], rows: List[dict]) -> None:
    """Update store with new rows; keep last seen snapshot."""
    t = _now()
    for r in rows:
        k = make_key(r)
        if not k[0] or not k[1]:
            continue
        if k not in store:
            store[k] = WatchItem(first_seen=t, last_seen=t, last_alert=0.0, data=r)
        else:
            item = store[k]
            item.last_seen = t
            item.data = r  # overwrite snapshot

def build_alerts(store: Dict[Key, WatchItem], min_score: float) -> List[dict]:
    """Return rows that should alert now (cooldown respected)."""
    t = _now()
    out = []
    for k, item in store.items():
        score = item.data.get("score")
        if score is None:
            continue
        if score < min_score:
            continue
        if (t - item.last_alert) < ALERT_COOLDOWN_SECONDS:
            continue
        # mark alert and emit
        item.last_alert = t
        out.append(item.data)
    # sort high score first
    out.sort(key=lambda r: r.get("score", -1e9), reverse=True)
    return out

def snapshot_watchlist(store: Dict[Key, WatchItem]) -> List[dict]:
    """Return current watchlist data as list of dict, sorted by score."""
    rows = [v.data for v in store.values()]
    rows.sort(key=lambda r: r.get("score", -1e9), reverse=True)
    return rows
