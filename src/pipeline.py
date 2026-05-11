"""
Pipeline orchestrator.

One entrypoint, `process_query`, that runs:

    safety guard  ->  classifier  ->  agent  ->  events

It yields normalized event dicts that the HTTP layer turns into SSE frames.
The HTTP layer doesn't reach into safety/classifier/agents directly — only
this module does. Keeps the system testable without a webserver.

Errors are events, not exceptions. The only thing that can stop the loop
mid-stream is the global timeout — and that emits a final `error` event
before closing.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import replace
from typing import Any, AsyncIterator

from . import safety
from .agents.registry import get_agent, is_implemented
from .classifier import classify
from .config import get_settings
from .intent_decompose import merge_task_decomposition
from .llm import LLMClient
from .orchestration.supervisor import run_collaborative_supervisor
from .session import SessionState, get_session_store


logger = logging.getLogger(__name__)

# If the model mislabels chat recap as customer_support, normalize here.
_CHAT_RECAP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bprevious\s+quer",
        r"\bprior\s+quer",
        r"\blast\s+quer",
        r"\bwhat\s+did\s+i\s+ask",
        r"\bwhat\s+have\s+i\s+asked",
        r"\ball\s+(?:my\s+)?previous\s+quer",
        r"\btell\s+(?:me\s+)?(?:all\s+)?(?:my\s+)?previous\s+quer",
        r"\b(?:list|show|repeat|give)\s+(?:me\s+)?(?:all\s+)?(?:my\s+)?(?:previous\s+)?quer",
        r"\bour\s+conversation\b",
        r"\bthis\s+conversation\b",
        r"\bchat\s+history\b",
        r"\bearlier\s+questions?\b",
        r"\bremind\s+me\s+what\s+i\b",
        r"\bwhat\s+was\s+my\s+previous\s+quer",
        r"\btell\s+me\s+what\s+was\s+my\s+previous\s+quer",
    )
)

# Only these classifier routes may invoke the multi-agent supervisor when requested.
_COLLABORATION_AGENT_ALLOWLIST: frozenset[str] = frozenset({"portfolio_health"})

# Strategy asks that mix macro / rebalance language with a multi-line book — panel handles overlap stories better than one agent.
_MACRO_STRATEGY_COLLAB_RE = re.compile(
    r"\b(?:rebalance|recession|macro|safer\s+assets|de-risk|defensive|diversif\w*|volatile|drawdown|economic\s+slowdown|flight\s+to\s+quality)\b",
    re.IGNORECASE,
)


def _agent_supports_collaborative_panel(agent_name: str) -> bool:
    return agent_name in _COLLABORATION_AGENT_ALLOWLIST


def _book_line_count(user_ctx: dict[str, Any]) -> int:
    positions = user_ctx.get("positions") or []
    return sum(1 for p in positions if (p.get("ticker") or "").strip())


def _investment_strategy_macro_collab_eligible(agent: str, query: str, user_ctx: dict[str, Any]) -> bool:
    """Multi-agent path for rebalance/macro-flavored strategy questions with enough holdings to argue overlap."""
    if agent != "investment_strategy":
        return False
    if _book_line_count(user_ctx) < 2:
        return False
    return bool(_MACRO_STRATEGY_COLLAB_RE.search(query or ""))


def _orchestration_profile(classification: Any) -> str | None:
    td = getattr(classification, "task_decomposition", None) or {}
    if not td.get("requires_multi_agent"):
        return None
    domains = {str(d).lower() for d in (td.get("domains") or [])}
    if "tax" in domains and "market" in domains:
        return "tax_market_synthesis"
    return None


def _looks_like_chat_recap_request(query: str) -> bool:
    q = (query or "").strip()
    if len(q) < 8:
        return False
    return any(p.search(q) for p in _CHAT_RECAP_PATTERNS)


def _collaborative_sparse_profile(
    agent: str,
    *,
    entities: dict[str, Any],
    query: str,
) -> str:
    """How much of the collaborative stack to activate for portfolio-style asks."""
    if agent != "portfolio_health":
        return "full"
    tickers = entities.get("tickers") or []
    q = (query or "").lower()
    tape_hints = (
        "price",
        "quote",
        "return",
        "momentum",
        "chart",
        "trading",
        "market",
        "stock",
        "tape",
        "ytd",
        "week",
        "month",
        "nvda",
        "amd",
        "tsla",
        "spy",
        "qqq",
    )
    compare_hints = ("vs ", "versus", "benchmark", "against ")
    if tickers or any(h in q for h in tape_hints) or any(h in q for h in compare_hints):
        return "full"
    return "portfolio_only"


def _effective_collaboration_rounds(sparse_profile: str, settings_rounds: int) -> int:
    """Map sparse tier to supervisor ``rounds`` (1=portfolio only, 3+=full template incl. synthesis round)."""
    sr = max(1, int(settings_rounds))
    if sparse_profile == "portfolio_only":
        return 1
    return max(3, sr)


def _meta_event(**fields: Any) -> dict[str, Any]:
    return {"type": "meta", **fields}


def _error_event(message: str, *, code: str = "internal_error") -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message}


async def process_query(
    *,
    query: str,
    session_id: str,
    user_context: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    settings=None,
    collaborative: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Run the full pipeline. Yields event dicts.

    Event types:
        meta       - progress / debug; metadata events.
        data       - narrative text token (delta to append to UI buffer).
        structured - the agent's structured payload.
        done       - terminal marker.
        error      - terminal marker for failures.
    """
    settings = settings or get_settings()
    user_ctx = user_context or {}
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    yield _meta_event(stage="received", request_id=request_id, session_id=session_id)

    # ---------- safety ----------
    verdict = safety.check(query)
    if verdict.blocked:
        yield _meta_event(
            stage="safety",
            blocked=True,
            category=verdict.category,
            latency_ms=verdict.latency_ms,
        )
        # Send the refusal as both narrative + structured so clients can pick.
        yield {"type": "data", "delta": verdict.message or ""}
        yield {
            "type": "structured",
            "payload": {
                "agent": "safety_guard",
                "blocked": True,
                "category": verdict.category,
                "message": verdict.message,
            },
        }
        yield {"type": "done", "request_id": request_id, "blocked": True}
        return

    yield _meta_event(stage="safety", blocked=False, latency_ms=verdict.latency_ms)

    # ---------- classifier ----------
    store = get_session_store()
    state: SessionState = store.get(session_id)
    store.append_turn(session_id, role="user", content=query)
    # Full thread (user lines include the message we just appended). Agents
    # can use this for recap, pronouns, or richer narratives — not a pipeline hack.
    conversation_history = state.history_for_llm()

    try:
        classification = await asyncio.wait_for(
            asyncio.to_thread(
                classify,
                query,
                session=state,
                user_context=user_ctx,
                llm=llm,
            ),
            timeout=min(settings.request_timeout_s, 90.0),
        )
    except asyncio.TimeoutError:
        yield _error_event("classifier_timeout", code="timeout")
        yield {"type": "done", "request_id": request_id, "blocked": False}
        return
    except Exception as exc:
        logger.exception("Pipeline: classifier crashed")
        yield _error_event(f"classifier_error: {exc}", code="classifier_error")
        yield {"type": "done", "request_id": request_id, "blocked": False}
        return

    if _looks_like_chat_recap_request(query) and classification.agent != "general_query":
        logger.info(
            "Pipeline: chat recap phrasing routed %s -> general_query",
            classification.agent,
        )
        classification = replace(classification, agent="general_query")

    classification = merge_task_decomposition(classification, query, user_ctx)

    yield _meta_event(
        stage="classified",
        agent=classification.agent,
        intent=classification.intent,
        confidence=classification.confidence,
        fallback_used=classification.fallback_used,
        implemented=is_implemented(classification.agent),
        safety_verdict=classification.safety_verdict,
        classifier_error=(classification.error[:500] if classification.error else None),
    )

    store.update_carryover(
        session_id,
        intent=classification.agent,
        tickers=classification.entities.get("tickers") or [],
    )

    orch_prof = _orchestration_profile(classification)

    collaboration_requested = bool(collaborative or settings.multiagent_enabled)
    recap_query = _looks_like_chat_recap_request(query)
    strat_macro_collab = _investment_strategy_macro_collab_eligible(
        classification.agent, query, user_ctx
    )
    agent_panel_eligible = (
        _agent_supports_collaborative_panel(classification.agent)
        or orch_prof == "tax_market_synthesis"
        or strat_macro_collab
    )
    use_collab = collaboration_requested and agent_panel_eligible and not recap_query

    collaborative_skipped_reason: str | None = None
    if collaboration_requested and not use_collab:
        if recap_query:
            collaborative_skipped_reason = "recap_thread_query"
        elif not agent_panel_eligible:
            collaborative_skipped_reason = "intent_not_eligible"
        yield _meta_event(
            stage="orchestration",
            collaborative_requested=True,
            collaborative_activated=False,
            collaborative_skipped_reason=collaborative_skipped_reason,
        )

    sparse_profile = (
        "full"
        if use_collab and (orch_prof == "tax_market_synthesis" or strat_macro_collab)
        else (
            _collaborative_sparse_profile(
                classification.agent,
                entities=classification.entities,
                query=query,
            )
            if use_collab
            else "none"
        )
    )
    effective_rounds = (
        _effective_collaboration_rounds(sparse_profile, int(settings.multiagent_rounds))
        if use_collab
        else 0
    )

    classification_dict: dict[str, Any] = {
        "agent": classification.agent,
        "intent": classification.intent,
        "entities": classification.entities,
        "confidence": classification.confidence,
        "sparse_profile": sparse_profile,
        "collaboration_rounds_effective": effective_rounds,
        "task_decomposition": classification.task_decomposition,
        "orchestration_profile": orch_prof,
        "macro_strategy_collaboration": strat_macro_collab,
    }

    # ---------- routed agent (single) or supervisor (multi-agent) ----------
    response_buffer: list[str] = []
    structured_payload: dict[str, Any] | None = None

    try:
        if use_collab:
            agent_iter = run_collaborative_supervisor(
                query=query,
                session_id=session_id,
                user_context=user_ctx,
                conversation_history=conversation_history,
                state=state,
                classification=classification_dict,
                rounds=effective_rounds,
                llm=llm,
            )
        else:
            agent = get_agent(classification.agent)
            agent_iter = agent.run(
                query=query,
                user_context=user_ctx,
                classification={
                    "agent": classification.agent,
                    "intent": classification.intent,
                    "entities": classification.entities,
                    "confidence": classification.confidence,
                    "fallback_used": classification.fallback_used,
                    "classifier_error": classification.error,
                    "task_decomposition": classification.task_decomposition,
                },
                llm=llm,
                conversation_history=conversation_history,
            )
        async for event in _with_timeout(agent_iter, settings.request_timeout_s, started):
            if event.get("type") == "data":
                response_buffer.append(event.get("delta", ""))
            elif event.get("type") == "structured":
                structured_payload = event.get("payload")
            yield event
    except asyncio.TimeoutError:
        yield _error_event("pipeline_timeout", code="timeout")
        yield {"type": "done", "request_id": request_id, "blocked": False}
        return
    except Exception as exc:
        logger.exception("Pipeline: agent crashed")
        yield _error_event(f"agent_error: {exc}", code="agent_error")
        yield {"type": "done", "request_id": request_id, "blocked": False}
        return

    if response_buffer:
        store.append_turn(session_id, role="assistant", content="".join(response_buffer))

    if use_collab:
        store.flush_session(session_id)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    yield _meta_event(stage="complete", latency_ms=elapsed_ms)
    yield {
        "type": "done",
        "request_id": request_id,
        "blocked": False,
        "agent": "multiagent_orchestrator" if use_collab else classification.agent,
        "implemented": True if use_collab else is_implemented(classification.agent),
        "structured_present": structured_payload is not None,
    }


async def _with_timeout(
    aiter: AsyncIterator[dict[str, Any]],
    total_timeout_s: float,
    started: float,
) -> AsyncIterator[dict[str, Any]]:
    """Yield events while honouring the per-request timeout."""
    while True:
        remaining = total_timeout_s - (time.perf_counter() - started)
        if remaining <= 0:
            raise asyncio.TimeoutError()
        try:
            event = await asyncio.wait_for(_anext(aiter), timeout=remaining)
        except StopAsyncIteration:
            return
        yield event


async def _anext(aiter: AsyncIterator[Any]) -> Any:
    return await aiter.__anext__()
