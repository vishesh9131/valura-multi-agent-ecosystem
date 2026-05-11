"""Unit tests for deterministic strategy feasibility layer."""
from __future__ import annotations

from src.agents.strategy_feasibility import build_decision_support_audit


def test_infeasible_max_return_zero_risk():
    bundle = {
        "user_question": "I want maximum returns with zero risk.",
        "risk_profile": None,
        "classifier_entities": {},
    }
    audit = build_decision_support_audit(bundle)
    codes = [f["code"] for f in audit["feasibility_findings"]]
    assert "infeasible_tradeoff_max_return_zero_risk" in codes
    assert audit["severity_max"] == "high"
    assert audit["engines_present"]["portfolio_optimizer"] is False
    assert len(audit["constraint_contradictions"]) == 1
    assert "contradict" in audit["constraint_contradictions"][0].lower()
    row = next(f for f in audit["feasibility_findings"] if f["code"].startswith("infeasible"))
    assert row.get("constraint_contradiction") is True
    assert "formal_statement" in row


def test_conservative_plus_aggressive_language():
    bundle = {
        "user_question": "How do I get the highest returns this year?",
        "risk_profile": "conservative",
        "classifier_entities": {},
    }
    audit = build_decision_support_audit(bundle)
    codes = [f["code"] for f in audit["feasibility_findings"]]
    assert "profile_goal_mismatch" in codes


def test_clean_question_no_flags():
    bundle = {
        "user_question": "What does rebalance mean in plain English?",
        "risk_profile": "moderate",
        "classifier_entities": {},
    }
    audit = build_decision_support_audit(bundle)
    assert audit["feasibility_findings"] == []
    assert audit["severity_max"] == "none"
    assert audit["constraint_contradictions"] == []
