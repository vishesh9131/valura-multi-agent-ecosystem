"""
Agent contract.

Every specialist follows the same async-iterator shape so the HTTP layer
streams them identically. An agent yields three kinds of dict events:

    {"type": "data",  "delta": <str>}      narrative text token
    {"type": "structured", "payload": {...}}  the structured result
    {"type": "meta", ...}                  optional progress / debug

The router only cares about this protocol; it doesn't know whether the
agent talks to an LLM, runs pure math, or stubs out.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class Agent(Protocol):
    name: str

    async def run(
        self,
        *,
        query: str,
        user_context: dict[str, Any],
        classification: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        ...
