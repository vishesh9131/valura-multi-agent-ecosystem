"""financial_planning agent — retirement / passive-income style asks."""
from __future__ import annotations

import asyncio

from src.pipeline import process_query


def _collect(events):
    out = []

    async def _run():
        async for e in events:
            out.append(e)

    asyncio.run(_run())
    return out


def test_pipeline_financial_planner_retirement_passive_income(fake_llm):
    q = "I want to retire in 15 years with monthly passive income."
    out = _collect(
        process_query(
            query=q,
            session_id="sess_fp_ret",
            user_context={"user_id": "u_fp", "risk_profile": "moderate", "positions": []},
            llm=fake_llm,
            collaborative=False,
        )
    )
    metas = [e for e in out if e.get("type") == "meta"]
    assert any(e.get("stage") == "planner_context" for e in metas)
    structured = next(e for e in out if e["type"] == "structured")
    payload = structured["payload"]
    assert payload["agent"] == "financial_planning"
    assert payload["implemented"] is True
    assert "not implemented" not in payload["message"].lower()
    steps = payload.get("plan_steps") or []
    assert len(steps) >= 4
    facts = payload.get("facts_used") or {}
    assert facts.get("horizon_years") == 15
    assert facts.get("passive_income_interest") is True
    msg = payload["message"].lower()
    assert "planner" in msg or "checkpoint" in msg or "horizon" in msg
