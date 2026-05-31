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


def test_score_row_applies_components_and_low_liq_penalty_with_floor():
    row = {'buys_m5': 10, 'sells_m5': 4, 'vol_m5': 5000, 'liqUsd': scoring.MIN_LIQ_USD - 1, 'pc_m5': 2.0}
    raw_expected = (
        scoring.W_TXNS_IMBALANCE * 6
        + scoring.W_VOLUME_M5 * (5000 / scoring.SCORE_VOL_SCALE)
        + scoring.W_LIQ * ((scoring.MIN_LIQ_USD - 1) / scoring.SCORE_LIQ_SCALE)
        - scoring.W_PCHG_M5_PENALTY * 2.0
        - scoring.SCORE_LIQ_BELOW_MIN_PENALTY
    )
    assert scoring.score_row(row) == max(scoring.SCORE_MIN_VALUE, raw_expected)


def test_score_row_never_negative_for_safe_filter_pass():
    row = {
        'liqUsd': scoring.MIN_LIQ_USD,
        'txns_m5': scoring.MIN_TXNS_M5,
        'pc_m5': scoring.MAX_ABS_PRICECHANGE_M5,
        'pc_h1': scoring.MAX_PRICECHANGE_H1,
        'vol_m5': scoring.MIN_VOLUME_M5,
        'buys_m5': 10,
        'sells_m5': 1,
    }
    assert scoring.passes_safe_filters(row) is True
    assert scoring.score_row(row) >= scoring.SCORE_MIN_VALUE


def test_passes_safe_filters_string_buys_sells_do_not_crash():
    row = {
        'liqUsd': '100000',
        'txns_m5': '10',
        'pc_m5': '1.0',
        'pc_h1': '5.0',
        'vol_m5': '2000',
        'buys_m5': '20',
        'sells_m5': '5',
    }
    assert scoring.passes_safe_filters(row) is True
    bad = dict(row, buys_m5='not-a-number')
    assert scoring.passes_safe_filters(bad) is False


def test_pc_m5_boundary_passes_and_score_clamps_not_false_negative():
    row = {
        'liqUsd': scoring.MIN_LIQ_USD,
        'txns_m5': scoring.MIN_TXNS_M5,
        'pc_m5': scoring.MAX_ABS_PRICECHANGE_M5,
        'pc_h1': 0,
        'vol_m5': scoring.MIN_VOLUME_M5,
        'buys_m5': 20,
        'sells_m5': 1,
    }
    assert scoring.passes_safe_filters(row) is True
    assert scoring.score_row(row) >= 0.0


def test_passes_safe_filters_bad_numeric_input_returns_false():
    row = {
        'liqUsd': 'not-a-number',
        'txns_m5': '10',
        'pc_m5': '1.0',
        'pc_h1': '5.0',
        'vol_m5': '2000',
        'buys_m5': '20',
        'sells_m5': '5',
    }
    assert scoring.passes_safe_filters(row) is False
