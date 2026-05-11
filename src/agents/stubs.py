"""
Stub agents.

Unimplemented specialists return a polite not-implemented line. The one
exception is `general_query`: same StubAgent class, but that name runs the
conversational path (session thread + optional LLM) so we dont maintain a
second agent type in the registry.
"""
from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator

from ..llm import LLMError, assemble_messages, get_llm_client
from ..safety import MODEL_INJECTION_GUARD


logger = logging.getLogger(__name__)


_NOT_IMPLEMENTED_FRIENDLY = {
    "market_research":        "I'd normally pull live market data, news, and a quick read on this one — but the market research agent isn't wired up in this build yet.",
    "financial_planning":     "Planning conversations normally route to the financial planner agent — if you see this line, routing fell through unexpectedly.",
    "financial_calculator":   "Numerical calculations (DCA, mortgage, future value, FX) are handled by a dedicated calculator — not implemented in this build.",
    "risk_assessment":        "Risk metrics like beta, drawdown, and stress tests come from a dedicated risk agent — not implemented in this build.",
    "predictive_analysis":    "Forecasting questions go through a forecasting agent — not implemented in this build.",
    "customer_support":       "Account and platform questions are handled by support — not implemented in this build.",
    "portfolio_query":        "Portfolio lookup is handled by a dedicated query agent — not implemented in this build.",
}


_GENERAL_QUERY_SYSTEM = """\
You are Valura's conversational assistant for novice investors.

This is **one continuous chat**. The messages you receive are the **full thread**
(user + assistant turns, chronological). That transcript is the only ground
truth about what already happened — dont invent earlier turns or pretend lines
are missing.

When they recap, list prior questions, ask what they asked about stocks or
portfolio, or say "remember what I said":
- Answer **only** from prior turns; quote or paraphrase accurately.
- Portfolio-health questions count as portfolio/markets-related — mention them if
  relevant to what they are asking now.
- For ALL / EVERY / LIST style asks: give each prior USER message in order
  (numbered), excluding only their latest message (the recap request itself).
  Merge obvious duplicates unless they want a verbatim log.
- Never deny a topic (e.g. "you never asked about stocks") if any prior USER
  message clearly touches that topic — point to those lines instead.

Otherwise keep answers short unless they want depth. Do not invent portfolio
facts; USER_PROFILE_HINT is hints only — unknown means unknown. Educational
tone only; not personalised investment advice.

""" + MODEL_INJECTION_GUARD


def _prior_user_from_thread(history: list[dict[str, str]]) -> str | None:
    users = [m["content"] for m in history if m.get("role") == "user"]
    if len(users) >= 2:
        return users[-2]
    return None


def _all_prior_user_messages(history: list[dict[str, str]]) -> list[str]:
    users = [m["content"] for m in history if m.get("role") == "user"]
    if len(users) < 2:
        return []
    return users[:-1]


def _wants_all_prior_questions(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.search(r"\b(all|every|everything|each|list)\b", q))


_RECAPISH = re.compile(
    r"\b(previous|prior|last|what did i ask|what have i asked|recap|repeat (?:my )?question)\b",
    re.IGNORECASE,
)


def _offline_reply_from_thread(query: str, history: list[dict[str, str]]) -> str:
    priors = _all_prior_user_messages(history)
    users = [m for m in history if m.get("role") == "user"]
    if not priors:
        if users:
            if _RECAPISH.search(query or ""):
                return (
                    "This is still your first user message in this session, so there is nothing "
                    "earlier to recap yet. Send a real question first (for example about your portfolio), "
                    "then ask again to repeat or list prior questions."
                )
            return (
                "This looks like your first user message in this session — "
                "there isn't an earlier question to repeat yet."
            )
        return "No conversation thread yet."

    suffix = "\n\n(Chat model unavailable — answers are limited to thread replay.)"
    if _wants_all_prior_questions(query):
        lines = "\n".join(f'{i + 1}. "{u}"' for i, u in enumerate(priors))
        return f"Earlier in this chat you asked:\n{lines}{suffix}"

    last_only = priors[-1]
    return f'Before this message, your last question was: "{last_only}"{suffix}'


async def _general_query_stub_stream(
    *,
    query: str,
    user_context: dict[str, Any],
    classification: dict[str, Any],
    llm: Any | None,
    conversation_history: list[dict[str, str]] | None,
) -> AsyncIterator[dict[str, Any]]:
    hist = list(conversation_history or [])

    client = llm if (llm is not None and hasattr(llm, "stream_text")) else None
    if client is None:
        try:
            client = get_llm_client()
        except Exception:
            client = None
    if client is not None and not hasattr(client, "stream_text"):
        client = None

    slim_ctx = {}
    if user_context:
        slim_ctx = {
            "user_id": user_context.get("user_id"),
            "risk_profile": user_context.get("risk_profile"),
            "base_currency": user_context.get("base_currency"),
        }

    narrative_parts: list[str] = []

    if client is not None:
        tail_hint = ""
        if slim_ctx:
            tail_hint = f"\n\nUSER_PROFILE_HINT: {slim_ctx}"
        messages = assemble_messages(
            system=_GENERAL_QUERY_SYSTEM + tail_hint,
            history=hist,
        )
        try:
            # always stream from model when we have a client — no canned recap shortcut
            async for piece in client.stream_text(messages, temperature=0.3, max_tokens=450):
                narrative_parts.append(piece)
                yield {"type": "data", "delta": piece}
        except LLMError as exc:
            logger.warning("General-query stub LLM stream failed: %s", exc)
            fallback = _offline_reply_from_thread(query, hist)
            narrative_parts.append(fallback)
            yield {"type": "data", "delta": fallback}
    else:
        # no api key / no stream_text on injected llm — thread replay only
        fallback = _offline_reply_from_thread(query, hist)
        if classification.get("fallback_used"):
            err = (classification.get("classifier_error") or "").strip()
            tip = (
                "\n\n[Setup] The classifier could not use the LLM"
                + (f" ({err[:220]})" if err else "")
                + ". Export OPENAI_API_KEY or set LLM_PROVIDER=vllm in `.env`, then restart uvicorn."
            )
            fallback = fallback + tip
        narrative_parts.append(fallback)
        yield {"type": "data", "delta": fallback}

    full_text = "".join(narrative_parts)
    users = [m["content"] for m in hist if m.get("role") == "user"]
    priors = _all_prior_user_messages(hist)
    yield {
        "type": "structured",
        "payload": {
            "agent": "general_query",
            "implemented": False,
            "intent": classification.get("intent"),
            "entities": classification.get("entities") or {},
            "message": full_text,
            "previous_user_query": _prior_user_from_thread(hist),
            "previous_user_queries": priors,
            "thread_user_turns": len(users),
            "prior_user_query_in_thread": _prior_user_from_thread(hist),
            "thread_user_turn_count": len(users),
        },
    }


class StubAgent:
    def __init__(self, agent_name: str) -> None:
        self.name = agent_name

    async def run(
        self,
        *,
        query: str,
        user_context: dict[str, Any],
        classification: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if self.name == "general_query":
            async for ev in _general_query_stub_stream(
                query=query,
                user_context=user_context,
                classification=classification,
                llm=llm,
                conversation_history=conversation_history,
            ):
                yield ev
            return

        message = _NOT_IMPLEMENTED_FRIENDLY.get(
            self.name,
            f"The {self.name} agent isn't implemented in this build.",
        )
        users = [m["content"] for m in (conversation_history or []) if m.get("role") == "user"]
        prior_user = users[-2] if len(users) >= 2 else None

        yield {"type": "data", "delta": message}
        yield {
            "type": "structured",
            "payload": {
                "agent": self.name,
                "implemented": False,
                "intent": classification.get("intent"),
                "entities": classification.get("entities", {}),
                "message": message,
                "prior_user_query_in_thread": prior_user,
                "thread_user_turn_count": len(users),
            },
        }
