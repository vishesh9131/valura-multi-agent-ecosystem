"""Product recommendation agent — mocked web, no LLM."""
from __future__ import annotations

import asyncio

import pytest

from src.agents.product_recommendation import ProductRecommendationAgent
from src.config import reset_settings_cache
from src.orchestration.toolkits import web_tools as wt


def _collect(agent_gen):
    out = []

    async def _run():
        async for x in agent_gen:
            out.append(x)

    asyncio.run(_run())
    return out


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setenv("WEB_FETCH_ENABLED", "true")
    reset_settings_cache()
    yield
    monkeypatch.delenv("WEB_FETCH_ENABLED", raising=False)
    reset_settings_cache()


def test_product_agent_uses_mocked_search(monkeypatch, web_on):
    def fake_search(query: str, *, max_results=None, settings=None):
        return {
            "ok": True,
            "results": [
                {
                    "title": "Sample ETF overview",
                    "url": "https://example.org/fund",
                    "snippet": "Expense ratio 0.03%.",
                }
            ],
            "note": None,
        }

    def fake_fetch(url: str, *, max_chars=None, settings=None):
        return {"ok": True, "url": url, "text": "Holdings include large caps.", "note": None}

    monkeypatch.setattr(wt, "toolkit_web_search", fake_search)
    monkeypatch.setattr(wt, "toolkit_fetch_url", fake_fetch)

    agent = ProductRecommendationAgent()
    events = _collect(
        agent.run(
            query="recommend a dividend ETF",
            user_context={"user_id": "u1"},
            classification={
                "intent": "product",
                "entities": {"topics": ["dividend"]},
            },
            llm=None,
            conversation_history=[
                {"role": "user", "content": "no tobacco stocks please"},
            ],
        )
    )
    structured = next(e for e in events if e["type"] == "structured")
    payload = structured["payload"]
    assert payload["agent"] == "product_recommendation"
    assert payload["implemented"] is True
    assert payload["web_retrieval_ok"] is True
    assert payload["sources"][0]["url"] == "https://example.org/fund"
    assert "couldnt retrieve" not in payload["message"].lower()
    deltas = "".join(e.get("delta", "") for e in events if e["type"] == "data")
    assert "Sample ETF" in deltas or "expense" in deltas.lower()
