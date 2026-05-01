import app


def test_parse_manual_token_input_dex_token_url_keeps_solana_case():
    raw = " https://dexscreener.com/solana/So11111111111111111111111111111111111111112 \n"
    parsed = app.parse_manual_token_input(raw, "solana")
    assert parsed["ok"] is True
    assert parsed["chain"] == "solana"
    assert parsed["token_addr"] == "So11111111111111111111111111111111111111112"


def test_parse_manual_token_input_invalid():
    parsed = app.parse_manual_token_input("   ", "solana")
    assert parsed["ok"] is False
    assert parsed["error"] == "invalid_input"


def test_manual_add_token_to_monitoring_existing_active(monkeypatch):
    monkeypatch.setattr(
        app,
        "load_monitoring",
        lambda: [{"active": "1", "chain": "bsc", "base_addr": "0xabc"}],
    )
    res = app.manual_add_token_to_monitoring("0xabc", "bsc")
    assert res["status"] == "EXISTS_ACTIVE"
