"""Offline general-query path replays the stored thread (no LLM in tests)."""
from __future__ import annotations

from src.agents.stubs import _offline_reply_from_thread


def test_offline_first_user_turn_has_no_earlier_question():
    hist = [{"role": "user", "content": "hello"}]
    msg = _offline_reply_from_thread("what did I ask before", hist)
    assert "first user message" in msg.lower() or "earlier question" in msg.lower()


def test_offline_repeats_last_prior_user_line():
    hist = [
        {"role": "user", "content": "how is my portfolio doing today"},
        {"role": "assistant", "content": "stub"},
        {"role": "user", "content": "can you tell my previous query"},
    ]
    msg = _offline_reply_from_thread("can you tell my previous query", hist)
    assert "how is my portfolio doing today" in msg


def test_offline_list_all_numbered_when_user_asks_for_all():
    hist = [
        {"role": "user", "content": "first thing"},
        {"role": "user", "content": "second thing"},
        {"role": "user", "content": "list all my previous queries"},
    ]
    msg = _offline_reply_from_thread("list all my previous queries", hist)
    assert "first thing" in msg and "second thing" in msg
