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
