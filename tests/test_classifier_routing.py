"""
Classifier routing accuracy on the labeled gold set.

Threshold (from ASSIGNMENT.md): >= 85% routing accuracy.

The fake_llm fixture provides a deterministic stand-in for the real LLM
call so we can run in CI without an API key. The point here is to prove
the classifier *integrates* — schema coercion, fallback, taxonomy
clamping, follow-up plumbing — not to claim the heuristic stand-in is the
shipped classifier.
"""
from __future__ import annotations

from src.classifier import classify
from src.session import SessionState
from collections import deque

from .matcher import matches_entities


def test_classifier_routing_accuracy(gold_classifier_queries, fake_llm):
    correct = 0
    misses: list[tuple[str, str, str]] = []
    for case in gold_classifier_queries:
        result = classify(case["query"], llm=fake_llm)
        if result.agent == case["expected_agent"]:
            correct += 1
        else:
            misses.append((case["query"], case["expected_agent"], result.agent))

    accuracy = correct / len(gold_classifier_queries)
    assert accuracy >= 0.85, (
        f"Routing accuracy {accuracy:.2%} below 85%. Misses:\n  "
        + "\n  ".join(f"{q!r}: expected {e}, got {a}" for q, e, a in misses)
    )


def test_classifier_entity_extraction_soft(gold_classifier_queries, fake_llm):
    """Soft signal — does NOT fail the build, just reports."""
    matched = 0
    total = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total += 1
        result = classify(case["query"], llm=fake_llm)
        if matches_entities(result.entities, case["expected_entities"]):
            matched += 1

    rate = matched / total if total else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total})")


def test_classifier_handles_follow_up(conversation_test_cases, fake_llm):
    """Follow-up: ticker should carry over from prior turn into the next intent."""
    cases = conversation_test_cases("follow_up_session")
    for case in cases:
        state = SessionState(
            session_id=f"sess_{case['case_id']}",
            turns=deque(maxlen=16),
        )
        for prior in case["prior_user_turns"]:
            state.append("user", prior)
            state.append("assistant", "(stub)")
        # mark last carryover ticker the same way the pipeline would
        last_tickers: list[str] = []
        for prior in case["prior_user_turns"]:
            for word, sym in (("nvidia", "NVDA"), ("nvda", "NVDA"), ("amd", "AMD"),
                              ("apple", "AAPL"), ("microsoft", "MSFT")):
                if word in prior.lower() and sym not in last_tickers:
                    last_tickers.append(sym)
        if last_tickers:
            state.last_tickers = last_tickers

        # we append the current turn last so prior_user_turns() returns prior
        state.append("user", case["current_user_turn"])
        result = classify(case["current_user_turn"], session=state, llm=fake_llm)

        expected_tickers = {t.upper() for t in case["expected"]["entities"].get("tickers", [])}
        actual_tickers = {t.upper() for t in (result.entities.get("tickers") or [])}
        if expected_tickers:
            assert expected_tickers.issubset(actual_tickers), (
                f"{case['case_id']}: expected tickers {expected_tickers}, got {actual_tickers}"
            )


def test_classifier_handles_multi_intent_topic_switch(conversation_test_cases, fake_llm):
    """Topic switch: each turn classified on its own merits, no inappropriate carry."""
    cases = conversation_test_cases("multi_intent_session")
    for case in cases:
        state = SessionState(
            session_id=f"sess_{case['case_id']}",
            turns=deque(maxlen=16),
        )
        for prior in case["prior_user_turns"]:
            state.append("user", prior)
            state.append("assistant", "(stub)")
        state.append("user", case["current_user_turn"])
        result = classify(case["current_user_turn"], session=state, llm=fake_llm)
        assert result.agent == case["expected"]["agent"], (
            f"{case['case_id']}: expected {case['expected']['agent']}, got {result.agent}"
        )


def test_classifier_falls_back_on_llm_error():
    """LLM raising must NOT crash the request."""
    def boom(_messages):
        raise RuntimeError("simulated provider down")

    out = classify("hello", llm=boom)
    assert out.agent == "general_query"
    assert out.fallback_used is True
    assert out.error and "simulated" in out.error
