import app


def test_active_score_pair_malformed_numeric_fields_do_not_abort():
    pair = {
        'liquidity': {'usd': 'bad'},
        'volume': {'h24': 'bad', 'm5': 'bad'},
        'priceChange': {'m5': 'bad', 'h1': 'bad'},
        'txns': {'m5': {'buys': 'bad', 'sells': 'bad'}},
    }
    assert app.score_pair(pair) == 0.0


def test_solana_unverified_heuristic_malformed_numeric_fields_do_not_abort():
    pair = {
        'chainId': 'solana',
        'liquidity': {'usd': 'bad'},
        'volume': {'h24': 'bad'},
        'priceChange': {'h1': 'bad'},
    }
    assert app.solana_unverified_heuristic(pair) is True
