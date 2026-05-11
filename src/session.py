"""
Session memory.

Holds the last N user/assistant turns per session_id plus carryover
(last routed agent name as last_intent, last_tickers) for classifier follow-ups.

Non-test environments persist each session to ``Logs/memory/<id>.json`` (path
from SESSION_MEMORY_DIR) so restarts keep the thread. APP_ENV=test stays RAM-only
so pytest stays hermetic.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Protocol


DEFAULT_MAX_TURNS = 8


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    turns: Deque[Turn]
    last_intent: str | None = None
    last_tickers: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    # multi-agent supervisor leaves an audit trail here (also persisted on disk)
    discussion_log: list[dict[str, Any]] = field(default_factory=list)
    orchestrator_meta: dict[str, Any] = field(default_factory=dict)

    def append(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        self.updated_at = time.time()

    def history_for_llm(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def prior_user_turns(self) -> list[str]:
        return [t.content for t in self.turns if t.role == "user"][:-1]


class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState: ...
    def update_carryover(self, session_id: str, *, intent: str | None, tickers: Iterable[str]) -> None: ...
    def append_turn(self, session_id: str, role: str, content: str) -> None: ...
    def flush_session(self, session_id: str) -> None: ...
    def reset(self, session_id: str) -> None: ...


def _safe_file_stem(session_id: str) -> str:
    cleaned = (session_id or "").strip()
    if not cleaned:
        return "anonymous"
    if any(c in cleaned for c in "/\\:\n\x00"):
        return hashlib.sha256(cleaned.encode()).hexdigest()[:48]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in cleaned)
    safe = safe.strip("_") or "session"
    return safe[:180]


class InMemorySessionStore:
    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS, ttl_s: int = 6 * 3600) -> None:
        self._max_turns = max_turns
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def _evict_stale(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items() if now - s.updated_at > self._ttl_s]
        for sid in dead:
            self._sessions.pop(sid, None)

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            self._evict_stale()
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(
                    session_id=session_id,
                    turns=deque(maxlen=self._max_turns * 2),
                )
                self._sessions[session_id] = state
            return state

    def update_carryover(
        self,
        session_id: str,
        *,
        intent: str | None,
        tickers: Iterable[str],
    ) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            if intent:
                state.last_intent = intent
            t_list = list(tickers or [])
            if t_list:
                state.last_tickers = t_list

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        state = self.get(session_id)
        with self._lock:
            state.append(role, content)

    def flush_session(self, session_id: str) -> None:
        return

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class FileBackedSessionStore:
    """Same API as InMemorySessionStore; one JSON file per session on disk."""

    def __init__(self, persist_dir: Path, max_turns: int = DEFAULT_MAX_TURNS, ttl_s: int = 6 * 3600) -> None:
        self._persist_dir = Path(persist_dir)
        self._max_turns = max_turns
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._persist_dir / f"{_safe_file_stem(session_id)}.json"

    def _to_json(self, state: SessionState) -> dict:
        return {
            "session_id": state.session_id,
            "turns": [{"role": t.role, "content": t.content, "ts": t.ts} for t in state.turns],
            "last_intent": state.last_intent,
            "last_tickers": list(state.last_tickers),
            "updated_at": state.updated_at,
            "discussion_log": list(state.discussion_log),
            "orchestrator_meta": dict(state.orchestrator_meta),
        }

    def _from_json(self, data: dict) -> SessionState:
        sid = str(data.get("session_id") or "unknown")
        raw_turns = data.get("turns") or []
        dq: Deque[Turn] = deque(maxlen=self._max_turns * 2)
        for r in raw_turns:
            dq.append(
                Turn(
                    role=str(r["role"]),
                    content=str(r["content"]),
                    ts=float(r.get("ts", time.time())),
                )
            )
        disc = data.get("discussion_log")
        if not isinstance(disc, list):
            disc = []
        orch = data.get("orchestrator_meta")
        if not isinstance(orch, dict):
            orch = {}
        return SessionState(
            session_id=sid,
            turns=dq,
            last_intent=data.get("last_intent"),
            last_tickers=list(data.get("last_tickers") or []),
            updated_at=float(data.get("updated_at", time.time())),
            discussion_log=[dict(x) for x in disc if isinstance(x, dict)],
            orchestrator_meta=dict(orch),
        )

    def _write(self, state: SessionState) -> None:
        path = self._path(state.session_id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(self._to_json(state), ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    def _read(self, session_id: str) -> SessionState | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._from_json(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _evict_stale(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items() if now - s.updated_at > self._ttl_s]
        for sid in dead:
            self._sessions.pop(sid, None)
            try:
                self._path(sid).unlink(missing_ok=True)
            except OSError:
                pass

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            self._evict_stale()
            state = self._sessions.get(session_id)
            if state is None:
                loaded = self._read(session_id)
                if loaded is not None:
                    # keep canonical id from this request
                    loaded.session_id = session_id
                    self._sessions[session_id] = loaded
                    state = loaded
            if state is None:
                state = SessionState(
                    session_id=session_id,
                    turns=deque(maxlen=self._max_turns * 2),
                )
                self._sessions[session_id] = state
                self._write(state)
            return state

    def update_carryover(
        self,
        session_id: str,
        *,
        intent: str | None,
        tickers: Iterable[str],
    ) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                loaded = self._read(session_id)
                if loaded is None:
                    return
                loaded.session_id = session_id
                self._sessions[session_id] = loaded
                state = loaded
            if intent:
                state.last_intent = intent
            t_list = list(tickers or [])
            if t_list:
                state.last_tickers = t_list
            self._write(state)

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        state = self.get(session_id)
        with self._lock:
            state.append(role, content)
            self._write(state)

    def flush_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                loaded = self._read(session_id)
                if loaded is None:
                    return
                loaded.session_id = session_id
                self._sessions[session_id] = loaded
                state = loaded
            state.updated_at = time.time()
            self._write(state)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            try:
                self._path(session_id).unlink(missing_ok=True)
            except OSError:
                pass


_default_store: InMemorySessionStore | FileBackedSessionStore | None = None


def get_session_store() -> SessionStore:
    global _default_store
    if _default_store is None:
        from .config import get_settings

        settings = get_settings()
        ttl = settings.session_ttl_s
        raw_dir = str(settings.session_memory_dir or "").strip()
        use_disk = settings.app_env != "test" and bool(raw_dir)
        if use_disk:
            root = Path(raw_dir).expanduser()
            if not root.is_absolute():
                root = Path.cwd() / root
            _default_store = FileBackedSessionStore(root.resolve(), max_turns=DEFAULT_MAX_TURNS, ttl_s=ttl)
        else:
            _default_store = InMemorySessionStore(max_turns=DEFAULT_MAX_TURNS, ttl_s=ttl)
    return _default_store


def reset_session_store() -> None:
    global _default_store
    _default_store = None
