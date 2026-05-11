"""Tax-aware decomposition, collaboration gate, and finance toolkit smoke tests."""
from __future__ import annotations

import asyncio

from src.agents.investment_strategy import SYSTEM, _offline_narrative
from src.classifier import classify
from src.intent_decompose import merge_task_decomposition
from src.market_data import Quote
from src.orchestration.toolkits import finance_math as fm
from src.pipeline import process_query


def _collect(events):
    out = []

    async def _run():
        async for e in events:
            out.append(e)

    asyncio.run(_run())
    return out


def test_merge_task_decomposition_heuristic_tax_sell(fake_llm):
    q = "Thinking about selling NVDA — capital gains tax vs waiting until next year?"
    base = classify(q, llm=fake_llm)
    merged = merge_task_decomposition(base, q, {"positions": [{"ticker": "NVDA", "quantity": 1, "avg_cost": 1.0}]})
    td = merged.task_decomposition
    assert td["requires_multi_agent"] is True
    assert "tax" in td["domains"] and "market" in td["domains"]


def test_tax_collaborative_pipeline_supervisor_and_gain_bundle(fake_llm):
    q = "Sell NVDA now or wait — worried about capital gains tax versus market momentum?"
    out = _collect(
        process_query(
            query=q,
            session_id="sess_tax_collab",
            user_context={
                "user_id": "u_tax",
                "positions": [
                    {"ticker": "NVDA", "quantity": 10, "avg_cost": 200.0, "currency": "USD", "purchased_at": "2023-01-01"},
                ],
            },
            llm=fake_llm,
            collaborative=True,
        )
    )
    assert any(e.get("stage") == "collaborative_start" for e in out if e.get("type") == "meta")
    structured = next(e for e in out if e["type"] == "structured")
    payload = structured["payload"]
    est = payload.get("computed_gain_estimate") or {}
    assert est.get("ticker") == "NVDA"
    assert est.get("estimated_unrealized_gain_notional") is not None
    assert est.get("holding_period_class") in {"long_term", "short_term", "unknown"}
    assert "disclaimer" in est


def test_finance_math_toolkit_basis_unknown(monkeypatch):
    monkeypatch.setattr(
        fm,
        "get_quote",
        lambda t: Quote(ticker=t, price=300.0, currency="USD", name=t, sector="Technology"),
    )
    ctx = {"positions": [{"ticker": "ZZZ", "quantity": 2, "avg_cost": 0.0, "currency": "USD"}]}
    bundle = fm.toolkit_tax_gain_bundle(ctx, ticker="ZZZ")
    assert bundle["basis_note"] == "basis_missing_or_zero"
    assert bundle["estimated_unrealized_gain_per_share"] is not None


def test_strategy_system_forbids_empty_audit_contradiction_opener():
    low = SYSTEM.lower()
    assert "empty" in low and "feasibility_findings" in low
    assert "no contradictions" in low


def test_offline_strategy_narrative_no_contradiction_weasel_opener():
    bundle = {
        "user_question": "Tax-wise should I trim NVDA?",
        "risk_profile": "moderate",
        "decision_support_audit": {
            "constraint_contradictions": [],
            "feasibility_findings": [],
            "notes": "",
            "engines_present": {},
        },
        "positions": [],
        "classifier_entities": {"tickers": ["NVDA"], "action": "sell"},
    }
    text = _offline_narrative(bundle).lower()
    assert "no logical contradictions" not in text[:1200]


def test_clarification_meta_when_missing_purchase_date(fake_llm):
    out = _collect(
        process_query(
            query="Should I sell NVDA for capital gains tax reasons this year?",
            session_id="sess_clarif",
            user_context={
                "user_id": "u_cl",
                "positions": [{"ticker": "NVDA", "quantity": 5, "avg_cost": 100.0, "currency": "USD"}],
            },
            llm=fake_llm,
            collaborative=False,
        )
    )
    metas = [e for e in out if e.get("type") == "meta"]
    assert any(e.get("stage") == "clarification_suggested" for e in metas)
