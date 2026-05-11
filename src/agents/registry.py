"""
Agent registry.

Maps the classifier's `agent` field to a runnable agent. Anything not
explicitly registered with a real implementation falls through to a
StubAgent — the router never raises a "no such agent" error.
"""
from __future__ import annotations

from typing import Any

from .base import Agent
from .financial_planner import FinancialPlannerAgent
from .investment_debate import InvestmentDebateAgent
from .investment_strategy import InvestmentStrategyAgent
from .portfolio_health import PortfolioHealthAgent
from .product_recommendation import ProductRecommendationAgent
from .stubs import StubAgent


# Real, fully-implemented agents go here. Add others as they ship.
# general_query is handled by StubAgent (conversational branch in stubs.py).
_REAL: dict[str, Agent] = {
    "portfolio_health": PortfolioHealthAgent(),
    "product_recommendation": ProductRecommendationAgent(),
    "financial_planning": FinancialPlannerAgent(),
    "investment_debate": InvestmentDebateAgent(),
    "investment_strategy": InvestmentStrategyAgent(),
}


# Names the classifier may emit. Anything in this list that isn't in _REAL
# gets a stub. Anything outside this list also gets a stub (defensive).
_KNOWN_NAMES: tuple[str, ...] = (
    "portfolio_health",
    "portfolio_query",
    "market_research",
    "investment_strategy",
    "investment_debate",
    "financial_planning",
    "financial_calculator",
    "risk_assessment",
    "product_recommendation",
    "predictive_analysis",
    "customer_support",
    "general_query",
)


def get_agent(name: str) -> Agent:
    if name in _REAL:
        return _REAL[name]
    fallback_name = name if name in _KNOWN_NAMES else "general_query"
    return StubAgent(fallback_name)


def is_implemented(name: str) -> bool:
    return name in _REAL


def known_agent_names() -> tuple[str, ...]:
    return _KNOWN_NAMES


def register(name: str, agent: Any) -> None:
    """Used by tests to swap an agent without monkeypatching modules."""
    _REAL[name] = agent
