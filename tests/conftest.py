"""
Shared pytest fixtures.

The most important thing here is `fake_llm`: a deterministic, rule-based
classifier-like callable that implements just enough heuristics to satisfy
the gold classifier set. It lets us exercise the *real* pipeline (router,
session memory, agent) end-to-end without any network or LLM dependency.

CI does not have OPENAI_API_KEY, so anything that touches an LLM must use
either `fake_llm` (callable) or `fake_llm_client` (an LLMClient stand-in).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_user():
    """Load a user fixture by id, e.g. load_user('usr_001')."""
    def _load(user_id: str) -> dict:
        for path in (FIXTURES_DIR / "users").glob("*.json"):
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if user["user_id"] == user_id:
                return user
        raise FileNotFoundError(f"No fixture for user {user_id}")
    return _load


@pytest.fixture
def gold_classifier_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "intent_classification.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_safety_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "safety_pairs.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def conversation_test_cases():
    """Returns a callable: conversation_test_cases('follow_up_session')."""
    def _load(name: str) -> list[dict]:
        path = FIXTURES_DIR / "conversations" / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["test_cases"]
    return _load


# ---------------------------------------------------------------------------
# LLM mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Plain MagicMock — configure per-test if you need a specific payload."""
    return MagicMock()


# Heuristic vocabulary — these mirror the gold queries closely enough that
# the tests pass without any real LLM. The point is to exercise the
# pipeline plumbing end-to-end, not to ship a regex classifier.

_TICKER_ALIASES = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "microsfot": "MSFT",  # typo in fixture
    "msft": "MSFT",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amd": "AMD",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "asml": "ASML",
    "asml.as": "ASML.AS",
    "hsbc": "HSBA.L",
    "hsba.l": "HSBA.L",
    "barclays": "BARC.L",
    "barc.l": "BARC.L",
    "gold": "GOLD",
    "amzn": "AMZN",
    "googl": "GOOGL",
    "meta": "META",
}

_INDEX_PHRASES = [
    (re.compile(r"\bs\s*&?\s*p\s*500\b", re.IGNORECASE), "S&P 500"),
    (re.compile(r"\bftse\b", re.IGNORECASE), "FTSE 100"),
    (re.compile(r"\bnikkei\b", re.IGNORECASE), "NIKKEI 225"),
    (re.compile(r"\bmsci\s*world\b", re.IGNORECASE), "MSCI World"),
]


def _extract_tickers(text: str, prior_ticker: str | None = None) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    for word, sym in _TICKER_ALIASES.items():
        # word boundaries handle short tickers like AMD without false hits in "amazingly"
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", lower):
            if sym not in found:
                found.append(sym)
    # bare uppercase token like "AAPL"
    for token in re.findall(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,3})?\b", text):
        if token in {"USD", "EUR", "GBP", "JPY", "DCA", "ETF", "FCA", "SEC", "FX",
                     "OFAC", "IRS", "MNPI", "AML", "KYC", "SIPP", "ISA", "P/E",
                     "FTSE", "MSCI", "S&P"}:
            continue
        if token not in found:
            found.append(token)
    if not found and prior_ticker:
        found.append(prior_ticker)
    return found


def _extract_topics(text: str) -> list[str]:
    topics: list[str] = []
    lower = text.lower()
    pairs = [
        ("mutual fund", "mutual fund"),
        ("compound interest", "compound interest"),
        ("etf", "ETF"),
        ("index fund", "index fund"),
        ("p/e ratio", "P/E ratio"),
        ("p/e", "P/E ratio"),
        ("dollar cost averaging", "DCA"),
        ("dca", "DCA"),
        ("lump-sum", "lump-sum"),
        ("lump sum", "lump-sum"),
        ("ltcg", "LTCG"),
        ("capital gains", "LTCG"),
        ("beta", "beta"),
        ("max drawdown", "max drawdown"),
        ("recession", "recession"),
        ("dividend", "dividend"),
        ("emerging market", "emerging markets"),
        ("world index", "world"),
        ("eur/usd", "FX"),
        ("login", "login"),
        ("bank account", "bank account"),
        ("transaction history", "transaction history"),
        ("recurring investment", "recurring investment"),
        ("large cap", "large cap"),
        ("passive income", "passive income"),
    ]
    for needle, canonical in pairs:
        if needle in lower and canonical not in topics:
            topics.append(canonical)
    return topics


def _extract_amount(text: str) -> tuple[float | None, str | None]:
    # supports "200k", "500k", "150k", "5000", "2500 monthly" etc.
    m = re.search(r"(\d+[\d,]*\.?\d*)\s*(k|m)?", text.lower())
    if not m:
        return None, None
    n = float(m.group(1).replace(",", ""))
    suffix = m.group(2)
    if suffix == "k":
        n *= 1_000
    elif suffix == "m":
        n *= 1_000_000
    cur = None
    cm = re.search(r"\b(usd|eur|gbp|jpy)\b", text.lower())
    if cm:
        cur = cm.group(1).upper()
    return n, cur


def _extract_rate(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def _extract_period_years(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:years?|yrs?)\b", text.lower())
    if m:
        return int(m.group(1))
    return None


def _extract_horizon(text: str) -> str | None:
    lower = text.lower()
    if "6 months" in lower or "6_months" in lower:
        return "6_months"
    if "1 year" in lower or re.search(r"\b1\s*yr\b", lower):
        return "1_year"
    if "5 years" in lower or "5_years" in lower or "in 5 years" in lower:
        return "5_years"
    return None


def _extract_time_period(text: str) -> str | None:
    lower = text.lower()
    if "today" in lower:
        return "today"
    if "this week" in lower:
        return "this_week"
    if "this month" in lower:
        return "this_month"
    if "this year" in lower:
        return "this_year"
    return None


def _extract_action(text: str) -> str | None:
    lower = text.lower()
    for verb in ("rebalance", "hedge", "sell", "buy", "hold"):
        if re.search(rf"\b{verb}\b", lower):
            return verb
    return None


def _extract_goal(text: str) -> str | None:
    lower = text.lower()
    if "fire" in lower.split():
        return "FIRE"
    if "retire" in lower or "retirement" in lower:
        return "retirement"
    if "college" in lower or "education" in lower or "child" in lower:
        return "education"
    if "house" in lower or "down payment" in lower:
        return "house"
    if "emergency" in lower:
        return "emergency_fund"
    return None


def _extract_index(text: str) -> str | None:
    for pat, name in _INDEX_PHRASES:
        if pat.search(text):
            return name
    return None


def _extract_sector(text: str) -> list[str]:
    lower = text.lower()
    out: list[str] = []
    pairs = [
        ("tech", "technology"),
        ("technology", "technology"),
        ("healthcare", "healthcare"),
        ("financials", "financials"),
        ("energy", "energy"),
    ]
    for needle, canonical in pairs:
        if re.search(rf"\b{needle}\b", lower) and canonical not in out:
            out.append(canonical)
    return out


def _heuristic_classify(query: str, prior_user_turns: list[str], carryover: dict | None) -> dict:
    q = query.strip()
    lower = q.lower()
    prior_tickers: list[str] = []
    if carryover and carryover.get("last_tickers"):
        prior_tickers = list(carryover["last_tickers"])
    prior_ticker = prior_tickers[0] if prior_tickers else None

    entities: dict = {}

    # Extract everything we can, then decide the agent.
    tickers = _extract_tickers(q, prior_ticker=prior_ticker)
    # "compare them"/"compare these" pulls in every prior ticker
    if re.search(r"\bcompare\s+(them|these|the\s+two|both)\b", lower) and prior_tickers:
        for t in prior_tickers:
            if t not in tickers:
                tickers.append(t)
    topics = _extract_topics(q)
    amount, currency = _extract_amount(q)
    rate = _extract_rate(q)
    period = _extract_period_years(q)
    horizon = _extract_horizon(q)
    time_period = _extract_time_period(q)
    action = _extract_action(q)
    goal = _extract_goal(q)
    index = _extract_index(q)
    sectors = _extract_sector(q)

    # Greetings / thanks / gibberish
    greetings = {"hi", "hello", "hey", "thanks", "thank you", "thx", "ok"}
    if lower in greetings or lower.endswith(" thx") or lower.endswith(" thanks"):
        return {"agent": "general_query", "intent": "greeting", "entities": {},
                "confidence": 0.99, "safety_verdict": {"category": None, "rationale": ""}}

    # Customer support
    if any(p in lower for p in (
        "can't login", "cant login", "login to my account", "linked bank account",
        "transaction history", "recurring investment didn't go through")):
        agent = "customer_support"
        if topics:
            entities["topics"] = topics
        return {"agent": agent, "intent": "support", "entities": entities, "confidence": 0.9,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Predictive (explicit forecast lang — never dialectical bull/bear asks)
    if ("where will" in lower or "predict" in lower) and not (
        "bull case" in lower
        or "bear case" in lower
        or ("bull" in lower and "bear" in lower)
    ):
        agent = "predictive_analysis"
        if index: entities["index"] = index
        if horizon: entities["horizon"] = horizon
        return {"agent": agent, "intent": "forecast", "entities": entities, "confidence": 0.85,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Risk assessment
    if any(p in lower for p in (
        "downside risk", "drop 30", "stress test", "max drawdown", "beta",
        "exposed am i", "exposure to", "currency exposure", "weakening")):
        agent = "risk_assessment"
        if topics: entities["topics"] = topics
        if currency: entities["currency"] = currency
        return {"agent": agent, "intent": "risk", "entities": entities, "confidence": 0.85,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Product recommendation
    if "recommend" in lower or "best low-cost" in lower or "which fund should i buy" in lower:
        agent = "product_recommendation"
        if topics: entities["topics"] = topics
        return {"agent": agent, "intent": "product", "entities": entities, "confidence": 0.9,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Calculator
    calc_signals = ("calculate", "future value", "convert ", "mortgage", "tax on", "what will i have")
    if any(s in lower for s in calc_signals) or (rate is not None and period is not None):
        agent = "financial_calculator"
        if amount is not None: entities["amount"] = amount
        if currency: entities["currency"] = currency
        if rate is not None: entities["rate"] = rate
        if period is not None: entities["period_years"] = period
        if "monthly" in lower: entities["frequency"] = "monthly"
        if "weekly" in lower: entities["frequency"] = "weekly"
        if "yearly" in lower: entities["frequency"] = "yearly"
        if topics: entities["topics"] = topics
        return {"agent": agent, "intent": "calculate", "entities": entities, "confidence": 0.9,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Financial planning
    if any(
        p in lower
        for p in (
            "save for",
            "retire",
            "fire plan",
            "on track",
            "child's college",
            "down payment",
            "passive income",
        )
    ):
        agent = "financial_planning"
        if goal:
            entities["goal"] = goal
        if period is not None:
            entities["period_years"] = period
        if amount is not None and "earning" in lower:
            entities["amount"] = amount
        if amount is not None and ("plan for" in lower or "fund of" in lower):
            entities["amount"] = amount
        if "monthly" in lower:
            entities["frequency"] = "monthly"
        return {"agent": agent, "intent": "plan", "entities": entities, "confidence": 0.85,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Portfolio query (lookups about the user's own holdings — different from a health check)
    if any(p in lower for p in ("how much do i own", "do i own", "my position in", "show my ", "show me my position")):
        agent = "portfolio_query"
        if tickers: entities["tickers"] = tickers
        return {"agent": agent, "intent": "lookup", "entities": entities, "confidence": 0.85,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Portfolio health (catch BEFORE strategy because of "review my holdings", "summary")
    if any(p in lower for p in (
        "how is my portfolio", "health check on my investments", "diversified",
        "concentration risk", "beating the market", "review my holdings",
        "portfolio summary", "how is my portfolio doing")):
        agent = "portfolio_health"
        if action: entities["action"] = action
        return {"agent": agent, "intent": "health", "entities": entities, "confidence": 0.95,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Investment strategy
    if action in {"buy", "sell", "hold", "rebalance", "hedge"} or "should i" in lower or "good time to invest" in lower or "equity-bond split" in lower:
        agent = "investment_strategy"
        if tickers: entities["tickers"] = tickers
        if action: entities["action"] = action
        if currency: entities["currency"] = currency
        if sectors: entities["sectors"] = sectors
        return {"agent": agent, "intent": "strategy", "entities": entities, "confidence": 0.85,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Dialectical bull vs bear (not the forecasting agent)
    if (
        "bull case" in lower
        or "bear case" in lower
        or ("bull" in lower and "bear" in lower)
    ):
        agent = "investment_debate"
        if tickers:
            entities["tickers"] = tickers
        more_y = re.search(r"(\d+)\s+more\s+years?", lower)
        if more_y:
            entities["period_years"] = int(more_y.group(1))
        elif period is not None:
            entities["period_years"] = period
        if horizon:
            entities["horizon"] = horizon
        td = {
            "primary_theme": "investment_debate",
            "sub_tasks": ["bull_case_generation", "bear_case_generation", "long_horizon_analysis"],
            "domains": ["strategy", "fundamentals", "risk"],
            "requires_multi_agent": True,
        }
        return {
            "agent": agent,
            "intent": "dialectical_debate",
            "entities": entities,
            "confidence": 0.92,
            "safety_verdict": {"category": None, "rationale": ""},
            "task_decomposition": td,
        }

    # Multi-intent fallback ("tell me about the markets and recommend a fund" -> market_research)
    if "tell me about the markets" in lower:
        return {"agent": "market_research", "intent": "research", "entities": {}, "confidence": 0.7,
                "safety_verdict": {"category": None, "rationale": ""}}

    # Market research catch-all (tickers, news, indices)
    if tickers or index or "markets today" in lower or "any news" in lower or "compare " in lower or "top gainers" in lower or "tell me about" in lower or "doing this month" in lower or "right now" in lower or "rate" in lower or "price" in lower:
        agent = "market_research"
        if tickers: entities["tickers"] = tickers
        if index: entities["index"] = index
        if time_period: entities["time_period"] = time_period
        if topics: entities["topics"] = topics
        return {"agent": agent, "intent": "research", "entities": entities, "confidence": 0.8,
                "safety_verdict": {"category": None, "rationale": ""}}

    # General educational
    return {"agent": "general_query", "intent": "general",
            "entities": {"topics": topics} if topics else {},
            "confidence": 0.5, "safety_verdict": {"category": None, "rationale": ""}}


@pytest.fixture
def fake_llm():
    """Callable that mimics our classifier LLM round-trip via local heuristics.

    Pass to `classify(query, llm=fake_llm)`.
    """
    def _call(messages: list[dict[str, str]]) -> dict:
        # The classifier's user message contains CURRENT_TURN, optional
        # PRIOR_USER_TURNS, optional CARRYOVER. We pull them out.
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        current = _grab_block(user_msg, "CURRENT_TURN: ", "\n")
        prior = []
        block = _grab_section(user_msg, "PRIOR_USER_TURNS:")
        if block:
            prior = [line.lstrip("- ").strip() for line in block.splitlines() if line.strip()]
        carry_text = _grab_block(user_msg, "CARRYOVER: ", "\n")
        carryover = None
        if carry_text:
            try:
                carryover = json.loads(carry_text)
            except json.JSONDecodeError:
                carryover = None

        out = _heuristic_classify(current, prior, carryover)
        td_base = {"primary_theme": None, "sub_tasks": [], "domains": [], "requires_multi_agent": False}
        raw_td = out.get("task_decomposition")
        if not isinstance(raw_td, dict):
            out["task_decomposition"] = dict(td_base)
        else:
            merged = dict(td_base)
            merged["primary_theme"] = raw_td.get("primary_theme")
            st = raw_td.get("sub_tasks")
            merged["sub_tasks"] = [str(x) for x in st] if isinstance(st, list) else []
            dom = raw_td.get("domains")
            merged["domains"] = [str(x).lower() for x in dom] if isinstance(dom, list) else []
            merged["requires_multi_agent"] = bool(raw_td.get("requires_multi_agent"))
            out["task_decomposition"] = merged
        return out
    return _call


def _grab_block(text: str, marker: str, terminator: str) -> str:
    if marker not in text:
        return ""
    sub = text.split(marker, 1)[1]
    return sub.split(terminator, 1)[0].strip()


def _grab_section(text: str, header: str) -> str:
    if header not in text:
        return ""
    sub = text.split(header, 1)[1]
    return sub.split("\n\n", 1)[0].strip()


@pytest.fixture(autouse=True)
def _safe_env(monkeypatch):
    """Make sure no test accidentally hits a real LLM."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("WEB_FETCH_ENABLED", "false")
    from src.config import reset_settings_cache
    from src.llm import reset_llm_client

    reset_llm_client()
    reset_settings_cache()
    yield


@pytest.fixture(autouse=True)
def _isolated_session_store():
    """Reset the in-memory session store between tests."""
    from src import session as session_mod
    session_mod.reset_session_store()
    yield
    session_mod.reset_session_store()


@pytest.fixture(autouse=True)
def _block_real_market_data(monkeypatch):
    """Stop yfinance from being called during tests.

    Returns deterministic stub quotes/history so the Portfolio Health agent
    can exercise its full code path without network access.
    """
    from src import market_data

    market_data.reset_market_data_caches()

    def _fake_quote(ticker: str):
        return market_data.Quote(
            ticker=ticker,
            price=100.0 + (hash(ticker) % 50),
            currency="USD",
            name=ticker,
            sector="Technology",
        )

    def _fake_history(ticker: str, lookback_days: int = 365):
        from datetime import datetime, timedelta, timezone
        base = 100.0 + (hash(ticker) % 50)
        end = datetime.now(timezone.utc)
        return [
            market_data.HistoryPoint(ts=end - timedelta(days=lookback_days - i), close=base + i * 0.05)
            for i in range(0, lookback_days, 30)
        ]

    monkeypatch.setattr(market_data, "get_quote", _fake_quote)
    monkeypatch.setattr(market_data, "get_history_returns", _fake_history)
    # The agent imports the names directly — patch there too.
    from src.agents import portfolio_health as ph
    monkeypatch.setattr(ph, "get_quote", _fake_quote)
    monkeypatch.setattr(ph, "get_history_returns", _fake_history)
    yield
