"""investment_debate agent — dialectical routing for bull/bear asks."""
from __future__ import annotations

import asyncio

from src.classifier import classify
from src.pipeline import process_query


def test_classifier_routes_bull_bear_to_debate(fake_llm):
    q = "Bull case and bear case for holding NVDA 5 more years."
    out = classify(q, llm=fake_llm)
    assert out.agent == "investment_debate"
    assert out.task_decomposition.get("primary_theme") == "investment_debate"
    assert "bull_case_generation" in out.task_decomposition.get("sub_tasks", [])


def test_pipeline_debate_emits_three_roles_and_nvda_bull_points(fake_llm):
    out = []

    async def _run():
        async for e in process_query(
            query="Bull case and bear case for holding NVDA 5 more years.",
            session_id="sess_debate_nvda",
            user_context={"user_id": "u_debate", "positions": []},
            llm=fake_llm,
            collaborative=False,
        ):
            out.append(e)

    asyncio.run(_run())
    metas = [e for e in out if e.get("type") == "meta"]
    stages = {m.get("stage") for m in metas}
    assert "debate_bull" in stages and "debate_bear" in stages and "debate_synthesis" in stages
    structured = next(e for e in out if e["type"] == "structured")
    payload = structured["payload"]
    assert payload["agent"] == "investment_debate"
    assert payload["implemented"] is True
    assert payload["primary_intent"] == "investment_debate"
    assert payload["sub_intents"] == [
        "bull_case_generation",
        "bear_case_generation",
        "long_horizon_analysis",
    ]
    msg = payload["message"].lower()
    assert "cuda" in msg or "infrastructure" in msg
    assert "valuation" in msg or "competition" in msg
    assert "synthesis" in msg
