"""Assignment 2 — discussion_log survives FileBackedSessionStore flush."""
from __future__ import annotations

from pathlib import Path

from src.session import FileBackedSessionStore, SessionState


def test_discussion_log_round_trips_through_json(tmp_path: Path) -> None:
    root = tmp_path / "mem"
    store = FileBackedSessionStore(root, ttl_s=999999)
    s = store.get("sess_disc")
    s.discussion_log.append({"agent": "portfolio", "round": 1, "utterance": "hello"})
    s.orchestrator_meta = {"k": "v"}
    store.flush_session("sess_disc")

    store2 = FileBackedSessionStore(root, ttl_s=999999)
    s2 = store2.get("sess_disc")
    assert len(s2.discussion_log) == 1
    assert s2.discussion_log[0]["agent"] == "portfolio"
    assert s2.orchestrator_meta.get("k") == "v"


def test_new_session_has_empty_discussion_log(tmp_path: Path) -> None:
    store = FileBackedSessionStore(tmp_path / "m", ttl_s=999999)
    s = store.get("fresh")
    assert s.discussion_log == []
    assert s.orchestrator_meta == {}
