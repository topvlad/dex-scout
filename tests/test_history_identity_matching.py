import app


def test_history_matching_by_ca():
    cur = {"chain": "solana", "ca": "AbCd123"}
    hist = {"chain": "solana", "token_addr": "AbCd123"}
    assert app.history_row_matches_token(cur, hist)


def test_history_symbol_chain_fallback_when_ca_missing():
    cur = {"chain": "bsc", "symbol": "ABC", "ca": "0x123"}
    hist = {"chain": "bsc", "symbol": "abc"}
    assert app.history_row_matches_token(cur, hist)


def test_solana_case_preserved():
    cur = {"chain": "solana", "ca": "AbCd123"}
    hist = {"chain": "solana", "ca": "abcd123"}
    assert not app.history_row_matches_token(cur, hist)
