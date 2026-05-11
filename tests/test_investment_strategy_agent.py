"""Investment strategy agent — data-driven offline path in tests (no LLM client)."""
from __future__ import annotations

import asyncio

import pytest

from src.pipeline import process_query


def _collect(events):
    out = []

    async def _run():
        async for e in events:
            out.append(e)

    asyncio.run(_run())
    return out


def test_pipeline_investment_strategy_implemented(fake_llm):
    out = _collect(
        process_query(
            query="rebalance my portfolio",
            session_id="sess_inv_strat",
            user_context={
                "user_id": "u_x",
                "risk_profile": "moderate",
                "positions": [{"ticker": "SPY", "quantity": 10, "avg_cost": 400, "currency": "USD"}],
            },
            llm=fake_llm,
            collaborative=False,
        )
    )
    structured = next(e for e in out if e["type"] == "structured")
    payload = structured["payload"]
    assert payload["agent"] == "investment_strategy"
    assert payload["implemented"] is True
    assert "FACTS_JSON" in payload["message"]
    assert "Capability boundary" in payload["message"]
    assert payload["facts_used"]["positions"][0]["ticker"] == "SPY"
    assert payload["decision_support_audit"]["advisory_mode"] == "conversational_framework_only"
    assert "not implemented" not in payload["message"].lower()


def test_strategy_detects_return_risk_language(fake_llm):
    out = _collect(
        process_query(
            query="should i aim for maximum returns with zero risk?",
            session_id="sess_inv_contra",
            user_context={"user_id": "u_y", "risk_profile": "conservative", "positions": []},
            llm=fake_llm,
            collaborative=False,
        )
    )
    structured = next(e for e in out if e["type"] == "structured")
    payload = structured["payload"]
    msg = structured["payload"]["message"].lower()
    audit = payload["decision_support_audit"]
    codes = [f["code"] for f in audit["feasibility_findings"]]
    assert "infeasible_tradeoff_max_return_zero_risk" in codes
    assert audit["severity_max"] == "high"
    assert len(audit["constraint_contradictions"]) == 1
    assert "constraint contradiction" in msg
    assert ("contradict" in msg or "simultaneously" in msg)
