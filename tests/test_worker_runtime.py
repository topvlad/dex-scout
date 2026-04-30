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
