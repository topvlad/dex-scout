import importlib
import subprocess
import sys

import notification_core as nc


def _journal():
    return {"version": 1, "updated_ts": "", "events": {}}


def test_module_imports_without_streamlit_or_app():
    code = "import sys, notification_core; print('streamlit' in sys.modules, 'app' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "False False"
    assert importlib.import_module("notification_core") is nc


def test_event_key_is_deterministic_for_same_candidate():
    event = {"event_type": "portfolio", "source": "portfolio", "chain": "Solana", "token_address": "Token1", "action": "REDUCE", "verdict": "HIGH"}
    assert nc.build_notification_event_key(event) == nc.build_notification_event_key(dict(event))


def test_action_changes_event_key():
    base = {"event_type": "portfolio", "source": "portfolio", "chain": "solana", "token_address": "Token1", "verdict": "HIGH"}
    assert nc.build_notification_event_key({**base, "action": "REDUCE"}) != nc.build_notification_event_key({**base, "action": "EXIT"})


def test_digest_bucket_changes_event_key():
    event = {"event_type": "digest", "source": "digest_ui", "digest_path": "digest_ui"}
    assert nc.build_notification_event_key(event, bucket_ts="10") == nc.build_notification_event_key(dict(event), bucket_ts="10")
    assert nc.build_notification_event_key(event, bucket_ts="10") != nc.build_notification_event_key(dict(event), bucket_ts="11")


def test_sent_event_skips_duplicate():
    journal = _journal()
    journal["events"]["event-1"] = {"event_key": "event-1", "send_status": "sent", "send_attempts": 1, "last_seen_ts": "1970-01-01 00:00:00 UTC"}
    saved = []
    out = nc.guard_notification_event({"event_key": "event-1", "event_type": "entry_alert"}, lambda: journal, lambda j: saved.append(j) or True, "1970-01-01 00:01:00 UTC", 60, 100)
    assert out["ok"] is False
    assert out["decision"] == "skip_duplicate"
    assert journal["events"]["event-1"]["send_status"] == "sent"
    assert saved


def test_failed_event_retries_after_cooldown():
    journal = _journal()
    journal["events"]["event-1"] = {"event_key": "event-1", "send_status": "failed", "send_attempts": 1, "last_seen_ts": "1970-01-01 00:00:00 UTC"}
    out = nc.guard_notification_event({"event_key": "event-1", "event_type": "entry_alert"}, lambda: journal, lambda j: True, "1970-01-01 00:02:00 UTC", 60, 100)
    assert out["ok"] is True
    assert out["decision"] == "send"
    assert out["send_attempts"] == 2
    assert journal["events"]["event-1"]["send_status"] == "pending"


def test_failed_event_before_cooldown_skips_retry():
    journal = _journal()
    journal["events"]["event-1"] = {"event_key": "event-1", "send_status": "failed", "send_attempts": 1, "last_seen_ts": "1970-01-01 00:00:30 UTC"}
    out = nc.guard_notification_event({"event_key": "event-1", "event_type": "entry_alert"}, lambda: journal, lambda j: True, "1970-01-01 00:01:00 UTC", 60, 100)
    assert out["ok"] is False
    assert out["decision"] == "skip_retry_cooldown"
    assert journal["events"]["event-1"]["send_status"] == "failed"


def test_journal_trim_keeps_max_size():
    journal = _journal()
    for idx in range(5):
        journal["events"][f"sent-{idx}"] = {"event_key": f"sent-{idx}", "send_status": "sent", "last_seen_ts": f"1970-01-01 00:00:0{idx} UTC"}
    removed = nc.trim_notification_event_journal(journal, max_events=2)
    assert removed == 3
    assert len(journal["events"]) == 2


def test_material_portfolio_action_detection():
    assert nc.is_material_portfolio_action("TAKE PROFIT") is True
    assert nc.is_material_portfolio_action("TAKE_PROFIT") is True
    assert nc.is_material_portfolio_action("REDUCE") is True
    assert nc.is_material_portfolio_action({"final_action": "REDUCE"}) is True
    assert nc.is_material_portfolio_action({"final_action": "HOLD"}) is False
    assert nc.is_material_portfolio_action("WATCH") is False
    assert nc.is_material_portfolio_action("") is False


def test_summary_counts_are_correct():
    journal = {
        "notification_journal_trimmed": 3,
        "events": {
            "sent": {"send_status": "sent", "send_ts": "1970-01-01 00:00:01 UTC"},
            "failed": {"send_status": "failed", "last_seen_ts": "1970-01-01 00:00:02 UTC", "send_error": "boom"},
            "pending": {"send_status": "pending"},
            "skipped": {"send_status": "skipped", "last_reason": "skipped_duplicate"},
        },
    }
    summary = nc.summarize_notification_event_journal(journal)
    assert summary == {
        "journal_size": 4,
        "sent_count": 1,
        "failed_count": 1,
        "pending_count": 1,
        "skipped_count": 1,
        "last_sent_ts": "1970-01-01 00:00:01 UTC",
        "last_failed_ts": "1970-01-01 00:00:02 UTC",
        "last_failed_reason": "boom",
        "trimmed_count": 3,
    }
