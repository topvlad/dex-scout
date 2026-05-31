import alerts


def test_purge_expired_and_cap(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 1000.0)
    store = {}
    for i in range(alerts.WATCH_MAX_ITEMS + 2):
        store[('bsc', f'p{i}')] = alerts.WatchItem(1.0, float(100 + i), 0.0, {'score': i})
    alerts.purge_expired(store)
    assert len(store) <= alerts.WATCH_MAX_ITEMS


def test_upsert_watchlist_updates_existing(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 200.0)
    store = {}
    row = {'chain': 'sol', 'pairAddress': 'p1', 'score': 1}
    alerts.upsert_watchlist(store, [row])
    assert ('sol', 'p1') in store
    monkeypatch.setattr(alerts, '_now', lambda: 300.0)
    alerts.upsert_watchlist(store, [{'chain': 'sol', 'pairAddress': 'p1', 'score': 5}])
    assert store[('sol', 'p1')].data['score'] == 5
    assert store[('sol', 'p1')].last_seen == 300.0


def test_build_alerts_respects_threshold_and_cooldown(monkeypatch):
    t0 = 1000.0
    monkeypatch.setattr(alerts, '_now', lambda: t0)
    store = {
        ('c', 'a'): alerts.WatchItem(t0, t0, 0.0, {'score': 10, 'pairAddress': 'a'}),
        ('c', 'b'): alerts.WatchItem(t0, t0, 0.0, {'score': 5, 'pairAddress': 'b'}),
    }
    out = alerts.build_alerts(store, min_score=6)
    assert [r['pairAddress'] for r in out] == ['a']
    # immediate second run should cooldown-skip
    out2 = alerts.build_alerts(store, min_score=6)
    assert out2 == []


def test_purge_expired_does_not_sort_under_cap(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 1000.0)
    store = {('bsc', 'p1'): alerts.WatchItem(1.0, 999.0, 0.0, {'score': 1})}
    import builtins
    original_sorted = builtins.sorted
    monkeypatch.setattr(builtins, 'sorted', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unexpected sort')))
    try:
        alerts.purge_expired(store)
    finally:
        monkeypatch.setattr(builtins, 'sorted', original_sorted)
    assert ('bsc', 'p1') in store


def test_purge_expired_overflow_purges_oldest(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 1000.0)
    store = {}
    for i in range(alerts.WATCH_MAX_ITEMS + 1):
        store[('bsc', f'p{i}')] = alerts.WatchItem(1.0, float(100 + i), 0.0, {'score': i})
    alerts.purge_expired(store)
    assert len(store) == alerts.WATCH_MAX_ITEMS
    assert ('bsc', 'p0') not in store


def test_purge_at_cap_without_reserve_does_not_remove_or_sort(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 1000.0)
    store = {
        ('bsc', f'p{i}'): alerts.WatchItem(1.0, float(100 + i), 0.0, {'score': i})
        for i in range(alerts.WATCH_MAX_ITEMS)
    }
    import builtins
    original_sorted = builtins.sorted
    monkeypatch.setattr(builtins, 'sorted', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unexpected sort')))
    try:
        alerts.purge_expired(store, reserve_slot=False)
    finally:
        monkeypatch.setattr(builtins, 'sorted', original_sorted)
    assert len(store) == alerts.WATCH_MAX_ITEMS


def test_purge_at_cap_with_reserve_removes_oldest(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 1000.0)
    store = {
        ('bsc', f'p{i}'): alerts.WatchItem(1.0, float(100 + i), 0.0, {'score': i})
        for i in range(alerts.WATCH_MAX_ITEMS)
    }
    alerts.purge_expired(store, reserve_slot=True)
    assert len(store) == alerts.WATCH_MAX_ITEMS - 1
    assert ('bsc', 'p0') not in store


def test_upsert_at_cap_does_not_exceed_watch_max(monkeypatch):
    monkeypatch.setattr(alerts, '_now', lambda: 1000.0)
    store = {
        ('bsc', f'p{i}'): alerts.WatchItem(1.0, float(100 + i), 0.0, {'score': i})
        for i in range(alerts.WATCH_MAX_ITEMS)
    }
    alerts.upsert_watchlist(store, [{'chain': 'bsc', 'pairAddress': 'new', 'score': 999}])
    assert len(store) == alerts.WATCH_MAX_ITEMS
    assert ('bsc', 'new') in store
    assert ('bsc', 'p0') not in store
