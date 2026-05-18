import app


def test_post_collapse_forces_no_entry_and_exit():
    facts = {"health": "POST_COLLAPSE", "health_reasons": ["post_pump_collapse"]}
    v = app.build_unified_verdict(facts, "monitoring")
    assert v["entry_action"] == "NO_ENTRY"
    assert v["position_action"] in {"REDUCE", "EXIT", "REVIEW"} or v["position_action"] == "EXIT"


def test_untradeable_never_early():
    v = app.build_unified_verdict({"health": "UNTRADEABLE", "health_reasons": ["no_flow"]}, "portfolio_linked_monitoring")
    assert v["entry_action"] != "EARLY"
    assert v["position_action"] == "EXIT"


def test_unknown_maps_review_not_fake_exit():
    v = app.build_unified_verdict({"health": "UNKNOWN", "health_reasons": []}, "portfolio")
    assert v["position_action"] == "REVIEW"
