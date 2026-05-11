"""Supervisor ordering + meta stages."""
from __future__ import annotations

import asyncio
from collections import deque

from src.orchestration.supervisor import run_collaborative_supervisor
from src.session import SessionState


def _collect_supervisor(async_gen):
    out = []

    async def _run():
        async for x in async_gen:
            out.append(x)

    asyncio.run(_run())
    return out


def test_supervisor_three_rounds_and_agents_in_order():
    state = SessionState(session_id="s", turns=deque(maxlen=16))
    hist = [{"role": "user", "content": "How is NVDA vs my portfolio?"}]
    classification = {"agent": "portfolio_health", "intent": "health", "entities": {}}
    events = _collect_supervisor(
        run_collaborative_supervisor(
            query="How is NVDA vs my portfolio?",
            session_id="s",
            user_context={
                "user_id": "u1",
                "positions": [{"ticker": "NVDA", "quantity": 10, "avg_cost": 100, "currency": "USD"}],
            },
            conversation_history=hist,
            state=state,
            classification=classification,
            rounds=3,
        )
    )
    stages = [e for e in events if e.get("type") == "meta"]
    assert any(e.get("stage") == "collaborative_start" for e in stages)
    disc = [e for e in stages if e.get("stage") == "agent_discussion"]
    assert len(disc) == 3
    assert [e.get("agent") for e in disc] == ["portfolio", "market", "conversation"]
    assert len(state.discussion_log) == 3
    structured = next(e for e in events if e.get("type") == "structured")
    assert structured["payload"]["agent"] == "multiagent_orchestrator"
    msg = structured["payload"]["message"]
    assert "Educational context only" in msg
    assert "Portfolio lens:" not in msg
    assert "collaboration_signals" in structured["payload"]
    assert structured["payload"].get("multiagent_llm") is False
    cm = structured["payload"].get("collaboration_meta") or {}
    assert cm.get("tension") in {"low", "moderate", "high"}
