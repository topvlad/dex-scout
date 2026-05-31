import worker


class AppStub:
    def __init__(self):
        self.released = []

    def now_utc_str(self):
        return '2026-01-01T00:00:00Z'

    def parse_float(self, v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    def get_worker_runtime_state(self):
        return {'modes': {'scan_cycle': {'last_job_status': 'idle', 'last_job_started_epoch': 0}}}

    def update_worker_runtime_state(self, updates=None, increments=None):
        return {'ok': True}

    def acquire_lock(self, **kwargs):
        return {'ok': False, 'code': 'lock_held', 'detail': 'busy'}

    def release_lock(self, lock_key, owner):
        self.released.append((lock_key, owner))

    def update_job_heartbeat(self, **kwargs):
        return {'ok': True}


def test_unknown_job_mode_returns_2_without_nameerror(monkeypatch):
    stub = AppStub()
    monkeypatch.setattr(worker, 'app', stub)
    rc = worker.run_job_mode('unknown_mode')
    assert rc == 2


def test_lock_skip_path_returns_3_and_no_nameerror(monkeypatch):
    stub = AppStub()
    monkeypatch.setattr(worker, 'app', stub)
    monkeypatch.setattr(worker, '_update_worker_runtime_with_mode_state', lambda **kwargs: {'ok': True})
    rc = worker.run_job_mode('scan_cycle')
    assert rc == 3


def test_early_runner_exception_still_releases_lock(monkeypatch):
    class LockOkStub(AppStub):
        def acquire_lock(self, **kwargs):
            return {'ok': True}

    stub = LockOkStub()
    monkeypatch.setattr(worker, 'app', stub)
    monkeypatch.setattr(worker, 'JOB_DISPATCH', {'scan_cycle': lambda: (_ for _ in ()).throw(RuntimeError('boom'))})
    monkeypatch.setattr(worker, '_update_worker_runtime_with_mode_state', lambda **kwargs: {'ok': True})
    monkeypatch.setattr(worker, '_finalize_runtime_if_token_matches', lambda **kwargs: True)
    rc = worker.run_job_mode('scan_cycle')
    assert rc == 1
    assert stub.released, 'lock must be released in finally'


def test_dispatch_map_validation_catches_non_callable():
    import pytest
    with pytest.raises(RuntimeError, match='invalid_job_dispatch_non_callable:bad'):
        worker.validate_job_dispatch({'ok': lambda: {}, 'bad': 'nope'})


def test_duplicate_lock_path_does_not_write_running_heartbeat(monkeypatch):
    class DuplicateStub(AppStub):
        def __init__(self):
            super().__init__()
            self.heartbeats = []

        def get_worker_runtime_state(self):
            return {'modes': {'scan_cycle': {'last_job_status': 'running', 'last_job_started_epoch': time.time()}}}

        def update_job_heartbeat(self, **kwargs):
            self.heartbeats.append(kwargs)
            return {'ok': True}

    import time
    stub = DuplicateStub()
    monkeypatch.setattr(worker, 'app', stub)
    monkeypatch.setattr(worker, '_update_worker_runtime_with_mode_state', lambda **kwargs: {'ok': True})
    rc = worker.run_job_mode('scan_cycle')
    assert rc == 3
    assert [h['status'] for h in stub.heartbeats] == ['duplicate_guard']
    assert 'started' not in [h['status'] for h in stub.heartbeats]


def test_lock_acquisition_failure_surfaces_locked_not_running(monkeypatch):
    class LockFailStub(AppStub):
        def __init__(self):
            super().__init__()
            self.heartbeats = []

        def acquire_lock(self, **kwargs):
            return {'ok': False, 'code': 'backend_down', 'detail': 'nope'}

        def update_job_heartbeat(self, **kwargs):
            self.heartbeats.append(kwargs)
            return {'ok': True}

    stub = LockFailStub()
    monkeypatch.setattr(worker, 'app', stub)
    monkeypatch.setattr(worker, '_update_worker_runtime_with_mode_state', lambda **kwargs: {'ok': True})
    rc = worker.run_job_mode('scan_cycle')
    assert rc == 3
    assert [h['status'] for h in stub.heartbeats] == ['lock_failed']
    assert 'started' not in [h['status'] for h in stub.heartbeats]
