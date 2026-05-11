"""Session store: append, carryover, eviction-window basics."""
from __future__ import annotations

from src.session import InMemorySessionStore


def test_session_get_creates_state():
    s = InMemorySessionStore()
    state = s.get("sess_1")
    assert state.session_id == "sess_1"
    assert state.last_intent is None
    assert state.last_tickers == []


def test_append_turn_and_history():
    s = InMemorySessionStore()
    s.append_turn("sess_1", "user", "tell me about NVDA")
    s.append_turn("sess_1", "assistant", "(streamed reply)")
    s.append_turn("sess_1", "user", "how much do I own?")
    state = s.get("sess_1")
    hist = state.history_for_llm()
    assert hist[-1] == {"role": "user", "content": "how much do I own?"}
    assert state.prior_user_turns() == ["tell me about NVDA"]


def test_carryover_updates_only_when_provided():
    s = InMemorySessionStore()
    s.append_turn("sess_1", "user", "tell me about NVDA")
    s.update_carryover("sess_1", intent="market_research", tickers=["NVDA"])
    state = s.get("sess_1")
    assert state.last_intent == "market_research"
    assert state.last_tickers == ["NVDA"]

    # an empty tickers list should not wipe a prior list (the agent didn't
    # see new tickers, but the carryover should remain).
    s.update_carryover("sess_1", intent="portfolio_health", tickers=[])
    state = s.get("sess_1")
    assert state.last_intent == "portfolio_health"
    assert state.last_tickers == ["NVDA"]


def test_max_turns_evicts_oldest():
    s = InMemorySessionStore(max_turns=2)
    for i in range(10):
        s.append_turn("sess_1", "user", f"q{i}")
    state = s.get("sess_1")
    # max_turns=2 -> deque(maxlen=4); only the last 4 turns survive
    assert len(state.turns) <= 4


def test_reset_clears_state():
    s = InMemorySessionStore()
    s.append_turn("sess_1", "user", "hi")
    s.reset("sess_1")
    state = s.get("sess_1")
    assert list(state.turns) == []
