import scoring


def test_passes_safe_filters_accepts_good_row():
    row = {
        'liqUsd': 100000,
        'txns_m5': max(scoring.MIN_TXNS_M5, 10),
        'pc_m5': 1.0,
        'pc_h1': min(scoring.MAX_PRICECHANGE_H1, 5.0),
        'vol_m5': max(scoring.MIN_VOLUME_M5, 1000),
        'buys_m5': 20,
        'sells_m5': 5,
    }
    assert scoring.passes_safe_filters(row) is True


def test_passes_safe_filters_rejects_invalid_pc_m5():
    row = {
        'txns_m5': max(scoring.MIN_TXNS_M5, 10),
        'pc_m5': scoring.MAX_ABS_PRICECHANGE_M5 + 0.1,
        'vol_m5': max(scoring.MIN_VOLUME_M5, 1000),
        'buys_m5': 20,
        'sells_m5': 5,
    }
    assert scoring.passes_safe_filters(row) is False


def test_score_row_applies_components_and_low_liq_penalty():
    row = {'buys_m5': 10, 'sells_m5': 4, 'vol_m5': 5000, 'liqUsd': scoring.MIN_LIQ_USD - 1, 'pc_m5': 2.0}
    expected = (
        scoring.W_TXNS_IMBALANCE * 6
        + scoring.W_VOLUME_M5 * (5000 / 10_000)
        + scoring.W_LIQ * ((scoring.MIN_LIQ_USD - 1) / 50_000)
        - scoring.W_PCHG_M5_PENALTY * 2.0
        - 5
    )
    assert scoring.score_row(row) == expected
