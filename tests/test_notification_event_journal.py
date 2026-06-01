import app


def _journal():
    return {"version": 1, "updated_ts": "", "events": {}}


def test_already_sent_event_is_skipped_and_send_not_called(monkeypatch):
    journal = _journal()
    journal["events"]["event-1"] = {"event_key": "event-1", "send_status": "sent", "send_ts": "2026-01-01 00:00:00 UTC"}
    calls = []
    monkeypatch.setattr(app, "load_notification_event_journal", lambda: journal)
    monkeypatch.setattr(app, "save_notification_event_journal", lambda j: True)
    monkeypatch.setattr(app, "send_telegram", lambda *a, **k: calls.append(a) or True)

    out = app.notification_send_once("msg", "event-1", "emit-1", "entry_alert", now_ts=1000)

    assert out["status"] == "skipped"
    assert out["reason"] == "skipped_duplicate"
    assert calls == []


def test_failed_event_can_retry_after_cooldown(monkeypatch):
    journal = _journal()
    journal["events"]["event-1"] = {
        "event_key": "event-1",
        "send_status": "failed",
        "last_seen_ts": "1970-01-01 00:00:00 UTC",
        "send_attempts": 1,
    }
    calls = []
    monkeypatch.setattr(app, "load_notification_event_journal", lambda: journal)
    monkeypatch.setattr(app, "save_notification_event_journal", lambda j: True)
    monkeypatch.setattr(app, "send_telegram", lambda *a, **k: calls.append(a) or True)

    out = app.notification_send_once("msg", "event-1", "emit-1", "entry_alert", now_ts=1000, retry_cooldown_seconds=60)

    assert out["status"] == "sent"
    assert len(calls) == 1
    assert journal["events"]["event-1"]["send_status"] == "sent"
    assert journal["events"]["event-1"]["send_attempts"] == 2


def test_failed_event_before_cooldown_is_skipped(monkeypatch):
    journal = _journal()
    journal["events"]["event-1"] = {
        "event_key": "event-1",
        "send_status": "failed",
        "last_seen_ts": "1970-01-01 00:10:00 UTC",
        "send_attempts": 1,
    }
    calls = []
    monkeypatch.setattr(app, "load_notification_event_journal", lambda: journal)
    monkeypatch.setattr(app, "save_notification_event_journal", lambda j: True)
    monkeypatch.setattr(app, "send_telegram", lambda *a, **k: calls.append(a) or True)

    out = app.notification_send_once("msg", "event-1", "emit-1", "entry_alert", now_ts=620, retry_cooldown_seconds=60)

    assert out["status"] == "skipped"
    assert out["reason"] == "retry_cooldown"
    assert calls == []


def test_successful_send_records_sent_only_after_send_ok(monkeypatch):
    journal = _journal()
    monkeypatch.setattr(app, "load_notification_event_journal", lambda: journal)
    monkeypatch.setattr(app, "save_notification_event_journal", lambda j: True)
    monkeypatch.setattr(app, "send_telegram", lambda *a, **k: True)

    out = app.notification_send_once("msg", "event-1", "emit-1", "entry_alert", now_ts=1000)

    assert out["status"] == "sent"
    assert journal["events"]["event-1"]["send_status"] == "sent"
    assert journal["events"]["event-1"]["send_ts"]


def test_send_telegram_false_records_failed(monkeypatch):
    journal = _journal()
    monkeypatch.setattr(app, "load_notification_event_journal", lambda: journal)
    monkeypatch.setattr(app, "save_notification_event_journal", lambda j: True)
    monkeypatch.setattr(app, "send_telegram", lambda *a, **k: False)

    out = app.notification_send_once("msg", "event-1", "emit-1", "entry_alert", now_ts=1000)

    assert out["status"] == "failed"
    assert journal["events"]["event-1"]["send_status"] == "failed"
    assert journal["events"]["event-1"]["send_error"] == "send_telegram_false"


def test_journal_trims_to_max_size():
    journal = _journal()
    for i in range(5):
        journal["events"][f"sent-{i}"] = {
            "event_key": f"sent-{i}",
            "send_status": "sent",
            "last_seen_ts": f"1970-01-01 00:00:0{i} UTC",
        }
    journal["events"]["failed-recent"] = {
        "event_key": "failed-recent",
        "send_status": "failed",
        "last_seen_ts": "1970-01-01 00:01:00 UTC",
    }

    removed = app.trim_notification_event_journal(journal, max_events=3)

    assert removed == 3
    assert len(journal["events"]) == 3
    assert "failed-recent" in journal["events"]


def test_material_portfolio_action_blocks_no_urgent_heartbeat(monkeypatch):
    sent = []
    journal = _journal()
    monkeypatch.setattr(app, "send_telegram", lambda msg, parse_mode="HTML": sent.append(msg) or True)
    monkeypatch.setattr(app, "load_notification_event_journal", lambda: journal)
    monkeypatch.setattr(app, "save_notification_event_journal", lambda j: True)
    monkeypatch.setattr(app, "load_tg_state", lambda: {"last_position_analysis_dedupe_keys": [], "last_position_analysis_sent_ts": "", "last_position_analysis_heartbeat_ts": ""})
    monkeypatch.setattr(app, "save_tg_state", lambda state: None)
    monkeypatch.setattr(app, "build_notification_candidates", lambda m, p: ([], []))
    monkeypatch.setattr(app, "build_portfolio_meaningful_events", lambda *args, **kwargs: {})
    monkeypatch.setattr(app, "_is_tg_quiet_hours", lambda now_ts: False)
    row = {"active": "1", "chain": "solana", "base_symbol": "ZEREBRO", "base_token_address": "z1", "final_action": "REDUCE"}

    app.run_auto_notifications({}, [], [row], cycle_context={}, trigger_model="ui_scan_sync")

    assert sent
    assert "Portfolio check: no urgent changes" not in sent[0]
    assert any(ev.get("last_reason") == "material_portfolio_actions_present" for ev in journal["events"].values())


def test_same_candidate_has_deterministic_event_key():
    row = {"chain": "solana", "base_token_address": "Token1", "final_action": "REDUCE", "risk_level": "HIGH", "entry_score": "100"}
    signal = {"bucket": "REDUCE"}

    assert app.signal_event_key("portfolio", row, signal) == app.signal_event_key("portfolio", dict(row), dict(signal))


def test_different_action_has_different_event_key():
    row1 = {"chain": "solana", "base_token_address": "Token1", "final_action": "REDUCE", "risk_level": "HIGH", "entry_score": "100"}
    row2 = {**row1, "final_action": "EXIT"}

    assert app.signal_event_key("portfolio", row1, {"bucket": "REDUCE"}) != app.signal_event_key("portfolio", row2, {"bucket": "EXIT"})


def test_digest_bucket_event_key_changes_by_bucket():
    same_a = app.build_digest_event_key("digest_ui", now_ts=0, heartbeat_hours=12)
    same_b = app.build_digest_event_key("digest_ui", now_ts=60, heartbeat_hours=12)
    next_bucket = app.build_digest_event_key("digest_ui", now_ts=12 * 3600, heartbeat_hours=12)

    assert same_a == same_b
    assert same_a != next_bucket
