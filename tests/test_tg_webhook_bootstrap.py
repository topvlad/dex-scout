from fastapi.testclient import TestClient

import tg_webhook


def test_health_reports_import_failure_on_bootstrap_failure(monkeypatch):
    monkeypatch.setattr(
        tg_webhook,
        "BOOTSTRAP_ERROR",
        {"helper": "bootstrap_import", "exception_type": "RuntimeError", "exception_text": "boom"},
    )
    client = TestClient(tg_webhook.app)
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', True)
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', 'boom')
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json() == {'ok': False, 'error': 'app_import_failed', 'detail': 'boom'}


def test_runtime_reports_import_failure_on_bootstrap_failure(monkeypatch):
    monkeypatch.setattr(
        tg_webhook,
        "BOOTSTRAP_ERROR",
        {"helper": "bootstrap_import", "exception_type": "RuntimeError", "exception_text": "boom"},
    )
    client = TestClient(tg_webhook.app)
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', True)
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', 'boom')
    resp = client.get('/runtime')
    assert resp.status_code == 200
    assert resp.json()['error'] == 'app_import_failed'


def test_tg_digest_reports_import_failure_on_bootstrap_failure(monkeypatch):
    monkeypatch.setattr(
        tg_webhook,
        "BOOTSTRAP_ERROR",
        {"helper": "bootstrap_import", "exception_type": "RuntimeError", "exception_text": "boom"},
    )
    client = TestClient(tg_webhook.app)
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', True)
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', 'boom')
    resp = client.get('/tg/digest?key=dummy')
    assert resp.status_code == 200
    assert resp.json()['error'] == 'app_import_failed'


def test_import_failure_callback_does_not_mutate(monkeypatch):
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', True)
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', 'boom')
    monkeypatch.setattr(tg_webhook, 'add_contract_to_portfolio', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('mutation attempted')))
    client = TestClient(tg_webhook.app)
    resp = client.post('/tg_webhook', json={'callback_query': {'id': 'cb1', 'data': 'pf|bsc|ABCDEFGHIJKLMNOPQRST', 'message': {'chat': {'id': 1}, 'message_id': 2}}})
    assert resp.json() == {'ok': False, 'error': 'app_import_failed', 'detail': 'boom'}


def test_callback_validation_rejects_invalid_chain_and_ca(monkeypatch):
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', False)
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {})
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', '')
    monkeypatch.setattr(tg_webhook, 'tg_api', lambda *args, **kwargs: {'ok': True})
    client = TestClient(tg_webhook.app)
    resp = client.post('/tg_webhook', json={'callback_query': {'id': 'bad-chain', 'data': 'pf|evil|ABCDEFGHIJKLMNOPQRST', 'message': {}}})
    assert resp.json()['error'] == 'invalid_chain'
    resp = client.post('/tg_webhook', json={'callback_query': {'id': 'bad-ca', 'data': 'pf|bsc|not-valid-!!!', 'message': {}}})
    assert resp.json()['error'] == 'invalid_ca'


def test_callback_replay_does_not_duplicate_action(monkeypatch):
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', False)
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {})
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', '')
    state = {'processed_callback_ids': {}}
    monkeypatch.setattr(tg_webhook, 'TG_STATE', state)
    monkeypatch.setattr(tg_webhook, 'load_tg_state', lambda: state)
    monkeypatch.setattr(tg_webhook, 'save_tg_state', lambda new_state: state.update(new_state))
    monkeypatch.setattr(tg_webhook, 'tg_api', lambda *args, **kwargs: {'ok': True})
    monkeypatch.setattr(tg_webhook, 'canonical_entity_key', lambda chain, ca: f'{chain}:{ca}')
    monkeypatch.setattr(tg_webhook, 'find_token_meta', lambda chain, ca: {})
    calls = []
    monkeypatch.setattr(tg_webhook, 'add_contract_to_portfolio', lambda chain, ca: calls.append((chain, ca)) or True)
    client = TestClient(tg_webhook.app)
    payload = {'callback_query': {'id': 'replay-1', 'data': 'pf|bsc|ABCDEFGHIJKLMNOPQRST', 'message': {}}}
    assert client.post('/tg_webhook', json=payload).json() == {'ok': True}
    assert client.post('/tg_webhook', json=payload).json() == {'ok': True, 'duplicate': True}
    assert calls == [('bsc', 'abcdefghijklmnopqrst')]


def test_summary_key_uses_compare_digest(monkeypatch):
    called = []

    def fake_compare_digest(a, b):
        called.append((a, b))
        return True

    monkeypatch.setenv('TG_SUMMARY_KEY', 'secret')
    monkeypatch.setattr(tg_webhook.hmac, 'compare_digest', fake_compare_digest)
    assert tg_webhook.summary_key_ok('secret') is True
    assert called == [('secret', 'secret')]


def test_row_matches_token_alias_fields_and_solana_case_safe():
    aliases = [
        'base_token_address', 'base_addr', 'token_addr', 'token_address', 'ca', 'address', 'pair_address'
    ]
    for field in aliases:
        assert tg_webhook._row_matches_token({'chain': 'bsc', field: 'ABCDEFGHIJKLMNOPQRST'}, 'bsc', 'abcdefghijklmnopqrst')
    assert tg_webhook._row_matches_token({'chain': 'solana', 'base_addr': 'AbCdEfGhIjKlMnOpQrSt'}, 'solana', 'AbCdEfGhIjKlMnOpQrSt')
    assert not tg_webhook._row_matches_token({'chain': 'solana', 'base_addr': 'AbCdEfGhIjKlMnOpQrSt'}, 'solana', 'abcdefghijklmnopqrst')


def test_find_token_meta_uses_alias_fields(monkeypatch):
    monkeypatch.setattr(tg_webhook, 'load_monitoring', lambda: [{'chain': 'bsc', 'token_address': 'ABCDEFGHIJKLMNOPQRST', 'symbol': 'AAA'}])
    monkeypatch.setattr(tg_webhook, 'load_portfolio', lambda: [])
    assert tg_webhook.find_token_meta('bsc', 'abcdefghijklmnopqrst')['symbol'] == 'AAA'


def test_remove_contract_atomicish_portfolio_save_fail(monkeypatch):
    monkeypatch.setattr(tg_webhook, 'load_portfolio', lambda: [{'chain': 'bsc', 'base_token_address': 'abcdefghijklmnopqrst', 'active': '1', 'archived': '0'}])
    monkeypatch.setattr(tg_webhook, 'load_monitoring', lambda: [{'chain': 'bsc', 'base_addr': 'abcdefghijklmnopqrst', 'active': '1', 'archived': '0'}])
    monkeypatch.setattr(tg_webhook, 'save_portfolio', lambda rows: False)
    monkeypatch.setattr(tg_webhook, 'save_monitoring', lambda rows: (_ for _ in ()).throw(AssertionError('monitoring should not save')))
    out = tg_webhook.remove_contract_everywhere_atomicish('bsc', 'abcdefghijklmnopqrst')
    assert out['ok'] is False
    assert out['portfolio_saved'] is False
    assert out['monitoring_saved'] is False
    assert out['error'] == 'portfolio_save_failed'


def test_remove_contract_atomicish_monitoring_save_fail_reports_partial(monkeypatch):
    p_rows = [{'chain': 'bsc', 'base_token_address': 'abcdefghijklmnopqrst', 'active': '1', 'archived': '0'}]
    m_rows = [{'chain': 'bsc', 'base_addr': 'abcdefghijklmnopqrst', 'active': '1', 'archived': '0'}]
    saved_portfolio = []
    monkeypatch.setattr(tg_webhook, 'load_portfolio', lambda: saved_portfolio[-1] if saved_portfolio else p_rows)
    monkeypatch.setattr(tg_webhook, 'load_monitoring', lambda: m_rows)
    monkeypatch.setattr(tg_webhook, 'save_portfolio', lambda rows: saved_portfolio.append([dict(r) for r in rows]) or None)
    monkeypatch.setattr(tg_webhook, 'save_monitoring', lambda rows: False)
    out = tg_webhook.remove_contract_everywhere_atomicish('bsc', 'abcdefghijklmnopqrst')
    assert out['ok'] is False
    assert out['portfolio_saved'] is True
    assert out['monitoring_saved'] is False
    assert out['error'] == 'monitoring_save_failed'


def test_find_token_meta_uses_ttl_cache(monkeypatch):
    tg_webhook._clear_token_meta_cache()
    calls = {'monitoring': 0, 'portfolio': 0}

    def load_monitoring():
        calls['monitoring'] += 1
        return [{'chain': 'bsc', 'base_addr': 'abcdefghijklmnopqrst', 'symbol': 'AAA'}]

    def load_portfolio():
        calls['portfolio'] += 1
        return []

    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', False)
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {})
    monkeypatch.setattr(tg_webhook, 'load_monitoring', load_monitoring)
    monkeypatch.setattr(tg_webhook, 'load_portfolio', load_portfolio)
    monkeypatch.setattr(tg_webhook.time, 'time', lambda: 1000.0)
    assert tg_webhook.find_token_meta('bsc', 'abcdefghijklmnopqrst')['symbol'] == 'AAA'
    assert tg_webhook.find_token_meta('bsc', 'abcdefghijklmnopqrst')['symbol'] == 'AAA'
    assert calls == {'monitoring': 1, 'portfolio': 1}


def test_find_token_meta_cache_expires_and_alias_fields(monkeypatch):
    tg_webhook._clear_token_meta_cache()
    calls = {'monitoring': 0, 'portfolio': 0}
    now = {'value': 1000.0}

    def load_monitoring():
        calls['monitoring'] += 1
        return []

    def load_portfolio():
        calls['portfolio'] += 1
        return [{'chain': 'bsc', 'pair_address': 'ABCDEFGHIJKLMNOPQRST', 'base_symbol': f"P{calls['portfolio']}"}]

    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', False)
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {})
    monkeypatch.setattr(tg_webhook, 'load_monitoring', load_monitoring)
    monkeypatch.setattr(tg_webhook, 'load_portfolio', load_portfolio)
    monkeypatch.setattr(tg_webhook.time, 'time', lambda: now['value'])
    assert tg_webhook.find_token_meta('bsc', 'abcdefghijklmnopqrst')['symbol'] == 'P1'
    now['value'] += tg_webhook.TG_META_CACHE_TTL_SEC + 1
    assert tg_webhook.find_token_meta('bsc', 'abcdefghijklmnopqrst')['symbol'] == 'P2'
    assert calls == {'monitoring': 2, 'portfolio': 2}


def test_find_token_meta_solana_case_behavior(monkeypatch):
    tg_webhook._clear_token_meta_cache()
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', False)
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {})
    monkeypatch.setattr(tg_webhook, 'load_monitoring', lambda: [{'chain': 'solana', 'base_addr': 'AbCdEfGhIjKlMnOpQrSt', 'symbol': 'SOLX'}])
    monkeypatch.setattr(tg_webhook, 'load_portfolio', lambda: [])
    assert tg_webhook.find_token_meta('solana', 'AbCdEfGhIjKlMnOpQrSt')['symbol'] == 'SOLX'
    assert tg_webhook.find_token_meta('solana', 'abcdefghijklmnopqrst') == {'symbol': '', 'name': ''}


def test_import_status_endpoint_healthy_and_failure(monkeypatch):
    client = TestClient(tg_webhook.app)
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', False)
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', '')
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {})
    data = client.get('/_import_status').json()
    assert data['ok'] is True
    assert data['import_failed'] is False
    assert data['error'] == ''
    assert data['bootstrap_error'] == {}
    assert 'worker_fast_mode' in data
    assert 'app_module_loaded' in data
    monkeypatch.setattr(tg_webhook, 'IMPORT_FAILED', True)
    monkeypatch.setattr(tg_webhook, 'IMPORT_ERROR', 'boom')
    monkeypatch.setattr(tg_webhook, 'BOOTSTRAP_ERROR', {'exception_text': 'boom'})
    data = client.get('/_import_status').json()
    assert data['ok'] is False
    assert data['import_failed'] is True
    assert data['error'] == 'boom'


def test_callback_chain_whitelist_and_override(monkeypatch):
    monkeypatch.setattr(tg_webhook, 'VALID_CHAINS', {'solana', 'bsc'})
    assert tg_webhook._validate_callback_data('pf', 'solana', 'AbCdEfGhIjKlMnOpQrSt')[0]
    assert tg_webhook._validate_callback_data('pf', 'bsc', 'abcdefghijklmnopqrst')[0]
    assert tg_webhook._validate_callback_data('pf', 'eth', 'abcdefghijklmnopqrst')[1] == 'invalid_chain'
    monkeypatch.setattr(tg_webhook, 'VALID_CHAINS', {'eth'})
    assert tg_webhook._validate_callback_data('pf', 'eth', 'abcdefghijklmnopqrst')[0]
    assert tg_webhook._validate_callback_data('pf', 'bsc', 'abcdefghijklmnopqrst')[1] == 'invalid_chain'
