import dex


def test_normalize_pair_maps_fields():
    raw = {
        'chainId': 'bsc', 'dexId': 'pcs', 'pairAddress': '0xpair',
        'baseToken': {'symbol': 'AAA', 'name': 'Token AAA', 'address': '0xaaa'},
        'quoteToken': {'symbol': 'USDT'},
        'priceUsd': '1.23',
        'liquidity': {'usd': '50000'},
        'fdv': '1000000', 'marketCap': '900000',
        'priceChange': {'m5': '1.2', 'h1': '-2.3'},
        'volume': {'m5': '1000', 'h1': '2000', 'h24': '3000'},
        'txns': {'m5': {'buys': '7', 'sells': '3'}},
    }
    n = dex.normalize_pair(raw)
    assert n['chain'] == 'bsc'
    assert n['txns_m5'] == 10
    assert n['priceUsd'] == 1.23


def test_safe_numeric_normalization_handles_invalid_values():
    raw = {
        'baseToken': {}, 'quoteToken': {}, 'priceChange': {'m5': 'x'}, 'volume': {'m5': None},
        'txns': {'m5': {'buys': 'bad', 'sells': None}}, 'liquidity': {'usd': ''},
        'priceUsd': None,
    }
    n = dex.normalize_pair(raw)
    assert n['priceUsd'] is None
    assert n['liqUsd'] is None
    assert n['pc_m5'] is None
    assert n['vol_m5'] is None
    assert n['buys_m5'] == 0 and n['sells_m5'] == 0 and n['txns_m5'] == 0


def test_fetch_pairs_empty_query_returns_empty_without_request(monkeypatch):
    called = []
    monkeypatch.setattr(dex.requests, 'get', lambda *args, **kwargs: called.append(args) or None)
    assert dex.fetch_pairs_by_query(None) == []
    assert dex.fetch_pairs_by_query('') == []
    assert dex.fetch_pairs_by_query('   ') == []
    assert called == []


def test_fetch_pairs_query_is_url_encoded(monkeypatch):
    captured = {}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {'pairs': [{'pairAddress': 'x'}]}

    def fake_get(url, timeout=15):
        captured['url'] = url
        captured['timeout'] = timeout
        return Resp()

    monkeypatch.setattr(dex.requests, 'get', fake_get)
    assert dex.fetch_pairs_by_query('ai agent/$hot') == [{'pairAddress': 'x'}]
    assert 'ai%20agent%2F%24hot' in captured['url']


def test_numeric_helpers_only_swallow_conversion_errors():
    class Exploding:
        def __float__(self):
            raise RuntimeError('boom')

        def __int__(self):
            raise RuntimeError('boom')

    assert dex._to_float('bad') is None
    assert dex._to_int('bad') == 0
    import pytest
    with pytest.raises(RuntimeError):
        dex._to_float(Exploding())
    with pytest.raises(RuntimeError):
        dex._to_int(Exploding())


def test_pair_age_hours_valid_none_and_invalid(monkeypatch):
    monkeypatch.setattr(dex.time, 'time', lambda: 1_700_007_200.0)
    created_ms = 1_700_000_000_000
    assert dex._pair_age_hours(created_ms) == 2.0
    assert dex._pair_age_hours(None) is None
    assert dex._pair_age_hours('bad') is None


def test_normalize_pair_preserves_pair_created_at_and_adds_age(monkeypatch):
    monkeypatch.setattr(dex.time, 'time', lambda: 1_700_003_600.0)
    raw = {'pairCreatedAt': 1_700_000_000_000, 'baseToken': {}, 'quoteToken': {}}
    n = dex.normalize_pair(raw)
    assert n['pairCreatedAt'] == 1_700_000_000_000
    assert n['pair_age_hours'] == 1.0


def test_normalize_pair_core_full_and_compatibility(monkeypatch):
    monkeypatch.setattr(dex.time, 'time', lambda: 1_700_000_000.0)
    raw = {
        'chainId': 'bsc', 'dexId': 'pcs', 'pairAddress': '0xpair',
        'baseToken': {'symbol': 'AAA', 'name': 'Token AAA', 'address': '0xaaa'},
        'quoteToken': {'symbol': 'USDT'},
        'priceUsd': '1.23', 'liquidity': {'usd': '50000'},
        'fdv': '1000000', 'marketCap': '900000',
        'priceChange': {'m5': '1.2', 'h1': '-2.3'},
        'volume': {'m5': '1000', 'h1': '2000', 'h24': '3000'},
        'txns': {'m5': {'buys': '7', 'sells': '3'}},
    }
    core = dex.normalize_pair_core(raw)
    full = dex.normalize_pair_full(raw)
    assert {'fdv', 'mcap', 'vol_h1', 'vol_h24'}.isdisjoint(core)
    assert full['fdv'] == 1_000_000.0
    assert full['mcap'] == 900_000.0
    assert full['vol_h1'] == 2_000.0
    assert full['vol_h24'] == 3_000.0
    assert dex.normalize_pair(raw) == full
