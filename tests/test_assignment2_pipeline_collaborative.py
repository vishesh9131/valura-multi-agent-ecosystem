"""Collaborative flag reaches supervisor via pipeline."""
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


def test_collaborative_pipeline_emits_discussion_meta(fake_llm):
    out = _collect(
        process_query(
            query="How is my portfolio doing vs SPY?",
            session_id="sess_collab_smoke",
            user_context={
                "user_id": "u1",
                "positions": [{"ticker": "SPY", "quantity": 2, "avg_cost": 400, "currency": "USD"}],
            },
            llm=fake_llm,
            collaborative=True,
        )
    )
    metas = [e for e in out if e.get("type") == "meta"]
    assert any(e.get("stage") == "collaborative_start" for e in metas)
    assert sum(1 for e in metas if e.get("stage") == "agent_discussion") == 3
    done = next(e for e in out if e.get("type") == "done")
    assert done.get("agent") == "multiagent_orchestrator"
    assert done.get("implemented") is True


def test_default_pipeline_not_collaborative(fake_llm):
    out = _collect(
        process_query(
            query="hi there",
            session_id="sess_non_collab",
            user_context={"user_id": "u1", "positions": []},
            llm=fake_llm,
            collaborative=False,
        )
    )
    done = next(e for e in out if e.get("type") == "done")
    assert done.get("agent") != "multiagent_orchestrator"


def test_collaborative_true_recap_query_skips_supervisor(fake_llm):
    sid = "sess_recap_collab"
    _collect(
        process_query(
            query="How is my portfolio doing?",
            session_id=sid,
            user_context={
                "user_id": "u1",
                "positions": [{"ticker": "SPY", "quantity": 2, "avg_cost": 400, "currency": "USD"}],
            },
            llm=fake_llm,
            collaborative=False,
        )
    )
    out = _collect(
        process_query(
            query="What was my previous query?",
            session_id=sid,
            user_context={
                "user_id": "u1",
                "positions": [{"ticker": "SPY", "quantity": 2, "avg_cost": 400, "currency": "USD"}],
            },
            llm=fake_llm,
            collaborative=True,
        )
    )
    metas = [e for e in out if e.get("type") == "meta"]
    assert not any(e.get("stage") == "collaborative_start" for e in metas)
    orch = next(e for e in metas if e.get("stage") == "orchestration")
    assert orch.get("collaborative_requested") is True
    assert orch.get("collaborative_activated") is False
    assert orch.get("collaborative_skipped_reason") == "recap_thread_query"
    done = next(e for e in out if e.get("type") == "done")
    assert done.get("agent") == "general_query"
    assert done.get("agent") != "multiagent_orchestrator"


def test_collaborative_sparse_portfolio_only_emits_single_discussion_round(fake_llm):
    """Allocation-only ask: no tape hints -> portfolio_only -> one deterministic panel stage."""
    out = _collect(
        process_query(
            query="How diversified is my portfolio?",
            session_id="sess_sparse_alloc",
            user_context={
                "user_id": "u1",
                "positions": [{"ticker": "SPY", "quantity": 2, "avg_cost": 400, "currency": "USD"}],
            },
            llm=fake_llm,
            collaborative=True,
        )
    )
    metas = [e for e in out if e.get("type") == "meta"]
    start = next(e for e in metas if e.get("stage") == "collaborative_start")
    assert start.get("sparse_profile") == "portfolio_only"
    assert start.get("orchestration_plan") == ["portfolio"]
    assert sum(1 for e in metas if e.get("stage") == "agent_discussion") == 1
    done = next(e for e in out if e.get("type") == "done")
    assert done.get("agent") == "multiagent_orchestrator"


def test_investment_strategy_macro_collab_runs_supervisor_with_overlap_narrative(fake_llm):
    """Rebalance + recession language + multi-line book should unlock panel, not lone strategy agent."""
    out = _collect(
        process_query(
            query="Should I rebalance from tech into safer assets given recession fears?",
            session_id="sess_strat_macro_collab",
            user_context={
                "user_id": "u1",
                "positions": [
                    {"ticker": "NVDA", "quantity": 10, "avg_cost": 100, "currency": "USD"},
                    {"ticker": "TSLA", "quantity": 5, "avg_cost": 200, "currency": "USD"},
                    {"ticker": "QQQ", "quantity": 20, "avg_cost": 350, "currency": "USD"},
                ],
            },
            llm=fake_llm,
            collaborative=True,
        )
    )
    done = next(e for e in out if e.get("type") == "done")
    assert done.get("agent") == "multiagent_orchestrator"
    structured = next(e for e in out if e.get("type") == "structured")
    msg = (structured.get("payload") or {}).get("message") or ""
    lower = msg.lower()
    assert "already" in lower
    assert "qqq" in lower or "overlap" in lower or "etf" in lower
    sig = (structured.get("payload") or {}).get("collaboration_signals") or {}
    assert sig.get("etf_equity_overlap_pairs", 0) >= 1
    assert sig.get("growth_tech_cluster_stub") is True


def test_collaborative_requested_but_intent_not_eligible_emits_orchestration_meta(fake_llm):
    out = _collect(
        process_query(
            query="hi there",
            session_id="sess_collab_greeting",
            user_context={"user_id": "u1", "positions": []},
            llm=fake_llm,
            collaborative=True,
        )
    )
    orch = next(e for e in out if e.get("type") == "meta" and e.get("stage") == "orchestration")
    assert orch.get("collaborative_skipped_reason") == "intent_not_eligible"
    done = next(e for e in out if e.get("type") == "done")
    assert done.get("agent") == "general_query"
