"""Thread / recap helpers — deterministic so pytest doesnt need an LLM."""
from __future__ import annotations

from typing import Any


def toolkit_prior_user_turns(history: list[dict[str, str]]) -> list[str]:
    return [m["content"] for m in history if m.get("role") == "user"]


def toolkit_format_numbered_list(lines: list[str]) -> str:
    out = []
    for i, line in enumerate(lines, start=1):
        out.append(f'{i}. "{line}"')
    return "\n".join(out)


def toolkit_thread_summary_stub(history: list[dict[str, str]], max_user_turns: int = 4, max_chars: int = 140) -> str:
    """Recent **user** turns only — never mirrors assistant text (avoids recursive summary blowups)."""
    users = [m for m in history if m.get("role") == "user"]
    tail = users[-max_user_turns:]
    bits: list[str] = []
    for m in tail:
        content = ((m.get("content") or "").strip().replace("\n", " "))
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        if content:
            bits.append(content)
    return " · ".join(bits) if bits else "(no prior user turns in window)"


def toolkit_disclaimer_block() -> str:
    return (
        "Educational context only — not personalised investment advice. "
        "Past performance does not predict future results."
    )


def toolkit_suggest_handoff_intent(query: str, last_intent: str | None) -> dict[str, Any]:
    q = (query or "").lower()
    hint = last_intent or "general_query"
    if any(k in q for k in ("portfolio", "holding", "allocation")):
        hint = "portfolio_health"
    if any(k in q for k in ("quote", "price", "stock", "ticker", "market")):
        hint = "market_research"
    return {"suggested_intent": hint, "last_intent": last_intent}
