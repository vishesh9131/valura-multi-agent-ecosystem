"""Supervisor uses seperate LLM passes when an LLMClient is wired in."""
from __future__ import annotations

import asyncio
from collections import deque

from src.orchestration.supervisor import run_collaborative_supervisor
from src.session import SessionState


class FakeCollabLLM:
    """Minimal Protocol stand-in — no network."""

    provider = "test"
    model = "fake-collab"

    def complete_json(self, messages, **kwargs):  # noqa: ANN001
        system = messages[0]["content"]
        if "Portfolio Analyst agent" in system:
            return {
                "reasoning_trace": ["positions exist", "concentration flag noted"],
                "confidence_0_100": 72,
                "stance_headline": "Book looks concentrated",
                "answer": "Single-name weight is the story here.",
                "abstain": False,
                "abstain_reason": "",
            }
        if "Market Analyst agent" in system:
            return {
                "reasoning_trace": ["read quote", "read 21d window"],
                "confidence_0_100": 68,
                "stance_headline": "Tape looks constructive",
                "answer": "Recent drift isnt scary in isolation.",
                "abstain": False,
                "abstain_reason": "",
            }
        if "Risk / Skeptic agent" in system:
            return {
                "reasoning_trace": ["size vs label", "push back on complacency"],
                "confidence_0_100": 80,
                "stance_headline": "Trim or cap risk",
                "answer": "Id still cut gross exposure even if momentum smiles.",
                "abstain": False,
                "abstain_reason": "",
            }
        if "Momentum / Trend agent" in system:
            return {
                "reasoning_trace": ["trend friend short term", "dont ignore weight"],
                "confidence_0_100": 77,
                "stance_headline": "Trend still has voters",
                "answer": "Momentum crew would drag feet on a panic sale.",
                "abstain": False,
                "abstain_reason": "",
            }
        return {
            "reasoning_trace": ["unknown role"],
            "confidence_0_100": 40,
            "stance_headline": "fallback",
            "answer": "generic",
            "abstain": False,
            "abstain_reason": "",
        }

    async def stream_text(self, messages, **kwargs):  # noqa: ANN001
        yield "Chair merges the panel — "
        yield "risk wants caution, momentum wants patience.\n\n"


def _collect_supervisor(async_gen):
    out = []

    async def _run():
        async for x in async_gen:
            out.append(x)

    asyncio.run(_run())
    return out


def test_llm_supervisor_emits_five_agent_discussion_metas_before_stream():
    state = SessionState(session_id="s_llm", turns=deque(maxlen=16))
    hist = [{"role": "user", "content": "NVDA vs my book?"}]
    classification = {"agent": "portfolio_health", "intent": "health", "entities": {}}
    llm = FakeCollabLLM()
    events = _collect_supervisor(
        run_collaborative_supervisor(
            query="NVDA vs my book?",
            session_id="s_llm",
            user_context={
                "user_id": "u1",
                "positions": [{"ticker": "NVDA", "quantity": 10, "avg_cost": 100, "currency": "USD"}],
            },
            conversation_history=hist,
            state=state,
            classification=classification,
            rounds=3,
            llm=llm,
        )
    )
    disc = [e for e in events if e.get("type") == "meta" and e.get("stage") == "agent_discussion"]
    assert len(disc) == 5
    agents = [e.get("agent") for e in disc]
    assert agents == ["portfolio", "market", "risk", "momentum", "synthesis"]

    structured = next(e for e in events if e.get("type") == "structured")
    assert structured["payload"]["multiagent_llm"] is True
    assert structured["payload"]["collaboration_meta"]["mode"] == "llm_multiagent"
    assert "collaboration_signals" in structured["payload"]
    cm = structured["payload"]["collaboration_meta"]
    assert cm.get("tension") in {"low", "moderate", "high"}

    deltas = [e.get("delta", "") for e in events if e.get("type") == "data"]
    assert "Chair merges" in "".join(deltas)
