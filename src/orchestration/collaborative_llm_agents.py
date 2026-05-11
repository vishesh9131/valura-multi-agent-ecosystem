"""Separate LLM inference per collaborative agent — distinct prompts + contexts.

Tools stay deterministic (same bundles MCP/supervisor already compute); each
specialist runs its own ``complete_json`` pass so disagreement can emerge.
Chair streams plain prose so the user sees non-JSON synthesis."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from ..llm import LLMClient, LLMError, assemble_messages

from ..safety import MODEL_INJECTION_GUARD

logger = logging.getLogger(__name__)

# OpenAI strict json_schema wants this shape
_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "title": "collab_agent_turn",
    "properties": {
        "reasoning_trace": {"type": "array", "items": {"type": "string"}},
        "confidence_0_100": {"type": "integer", "minimum": 0, "maximum": 100},
        "stance_headline": {"type": "string"},
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": "string"},
    },
    "required": [
        "reasoning_trace",
        "confidence_0_100",
        "stance_headline",
        "answer",
        "abstain",
        "abstain_reason",
    ],
}

SYS_PORTFOLIO = """You are the Portfolio Analyst agent in a multi-agent panel.

Objective: interpret holdings, concentration, and stated risk profile ONLY from the JSON facts given.
Rules:
- Do not invent positions or prices not in the bundle.
- reasoning_trace is your scratchpad — short bullets (2–6) showing how you moved from facts -> stance.
- stance_headline is one punchy line (your independent view).
- answer is 2–4 sentences for other agents (not the final user essay — chair writes that).
- You have NOT seen market tape yet — dont pretend you did.
- If user_question is only chat/session recap or meta (no holdings math), set abstain=true, abstain_reason one line, answer empty or "n/a".
""" + "\n\n" + MODEL_INJECTION_GUARD

SYS_MARKET = """You are the Market Analyst agent.

Objective: interpret price level, sector, and recent return stats from the JSON facts.
You receive a separate portfolio_facts summary (numbers only from the book) — use it only as context for sizing/concentration, not as gospel truth about risk appetite.
Rules:
- Independent reasoning — do not quote other analysts prose (they dont exist yet).
- reasoning_trace: 2–6 bullets.
- If return data is missing, say so and lower confidence.
- If user_question is only about conversation memory or prior chat turns (no ticker/market substance), abstain=true and explain in abstain_reason — do not fabricate tape commentary.
""" + "\n\n" + MODEL_INJECTION_GUARD

SYS_RISK = """You are the Risk / Skeptic agent.

Objective: stress concentration, gap risk, and mismatch between book structure and user risk label.
You see structured facts plus Portfolio Analyst and Market Analyst JSON outputs — challenge them when needed.
Rules:
- Be willing to disagree with the Market Analyst if momentum looks fine but sizing is reckless.
- reasoning_trace must show at least one explicit pushback or caveat.
- If user_question is purely session/meta/recap, abstain=true — risk lens does not apply.
""" + "\n\n" + MODEL_INJECTION_GUARD

SYS_MOMENTUM = """You are the Momentum / Trend agent.

Objective: argue what recent performance and sector context imply for near-term positioning bias.
You see the same facts and both prior analysts JSON — you may disagree with Risk if tape supports patience.
Rules:
- Dont ignore concentration — acknowledge it, then state whether momentum still matters.
- If user_question is about chat history or session memory only, abstain=true — say momentum agent has nothing to add.
""" + "\n\n" + MODEL_INJECTION_GUARD

SYS_CHAIR = """You write the ONLY text the retail investor will read from this step.

Input is structured analyst notes (JSON-shaped facts + short opinions). Treat it as confidential reasoning — never quote JSON keys, routing labels, or words like "agent", "panel", "tool", "schema".

If an analyst block has abstain=true, skip their substance entirely — do not invent portfolio or tape takes on their behalf.

Output plain prose only: calm, specific, educational. Round percentages sensibly in speech (e.g. "~8.5%" not 8.523). Mention disagreement honestly when views clash.

Cover holdings concentration vs recent tape tone (from non-abstaining voices only), then balanced takeaway with non-prescriptive options (trim gradually, diversify, caps, wait-and-watch). Never expose middleware.

Educational context only — not personalized investment advice.

Stay under ~320 words unless stakes clearly demand more.
""" + "\n\n" + MODEL_INJECTION_GUARD


def _compact_agent(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "stance_headline": obj.get("stance_headline"),
        "confidence_0_100": obj.get("confidence_0_100"),
        "answer": obj.get("answer"),
        "reasoning_trace": (obj.get("reasoning_trace") or [])[:8],
        "abstain": bool(obj.get("abstain")),
        "abstain_reason": str(obj.get("abstain_reason") or ""),
    }


async def _infer_agent(
    llm: LLMClient,
    *,
    system: str,
    user_blob: dict[str, Any],
    temperature: float,
    max_tokens: int = 900,
) -> dict[str, Any]:
    """One structured LLM pass (blocking SDK wrapped for async pipeline)."""

    def _call() -> dict[str, Any]:
        messages = assemble_messages(
            system=system,
            user=json.dumps(user_blob, indent=2, default=str),
        )
        return llm.complete_json(
            messages,
            json_schema=_AGENT_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        raw = await asyncio.to_thread(_call)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc)) from exc

    if not isinstance(raw, dict):
        raise LLMError("collab agent expected dict JSON")
    # models ocasionally drift keys — normalize lightly
    trace = raw.get("reasoning_trace")
    if not isinstance(trace, list):
        trace = [str(trace)] if trace is not None else []
    trace = [str(x).strip() for x in trace if str(x).strip()]
    if not trace:
        trace = ["(model omitted trace)"]
    try:
        conf = int(raw.get("confidence_0_100", 50))
    except (TypeError, ValueError):
        conf = 50
    conf = max(0, min(100, conf))
    abstain = bool(raw.get("abstain"))
    abstain_reason = str(raw.get("abstain_reason") or "").strip()
    ans = str(raw.get("answer") or "").strip() or str(raw.get("stance_headline") or "")
    if abstain and not ans:
        ans = "n/a"
    return {
        "reasoning_trace": trace,
        "confidence_0_100": conf,
        "stance_headline": str(raw.get("stance_headline") or "").strip() or "No headline",
        "answer": ans,
        "abstain": abstain,
        "abstain_reason": abstain_reason,
    }


async def run_portfolio_llm(
    llm: LLMClient, *, query: str, portfolio_tool_bundle: dict[str, Any], temperature: float = 0.38
) -> dict[str, Any]:
    blob = {"user_question": query, "portfolio_tool_bundle": portfolio_tool_bundle}
    return await _infer_agent(llm, system=SYS_PORTFOLIO, user_blob=blob, temperature=temperature)


async def run_market_llm(
    llm: LLMClient,
    *,
    query: str,
    portfolio_facts: dict[str, Any],
    market_tool_bundle: dict[str, Any],
    temperature: float = 0.38,
) -> dict[str, Any]:
    blob = {
        "user_question": query,
        "portfolio_facts_numeric": portfolio_facts,
        "market_tool_bundle": market_tool_bundle,
    }
    return await _infer_agent(llm, system=SYS_MARKET, user_blob=blob, temperature=temperature)


async def run_risk_llm(
    llm: LLMClient,
    *,
    query: str,
    shared_facts: dict[str, Any],
    portfolio_analyst: dict[str, Any],
    market_analyst: dict[str, Any],
    temperature: float = 0.48,
) -> dict[str, Any]:
    blob = {
        "user_question": query,
        "shared_facts": shared_facts,
        "portfolio_analyst": _compact_agent(portfolio_analyst),
        "market_analyst": _compact_agent(market_analyst),
    }
    return await _infer_agent(llm, system=SYS_RISK, user_blob=blob, temperature=temperature)


async def run_momentum_llm(
    llm: LLMClient,
    *,
    query: str,
    shared_facts: dict[str, Any],
    portfolio_analyst: dict[str, Any],
    market_analyst: dict[str, Any],
    temperature: float = 0.48,
) -> dict[str, Any]:
    blob = {
        "user_question": query,
        "shared_facts": shared_facts,
        "portfolio_analyst": _compact_agent(portfolio_analyst),
        "market_analyst": _compact_agent(market_analyst),
    }
    return await _infer_agent(llm, system=SYS_MOMENTUM, user_blob=blob, temperature=temperature)


async def stream_chair_answer(
    llm: LLMClient,
    *,
    query: str,
    portfolio_analyst: dict[str, Any],
    market_analyst: dict[str, Any],
    risk_analyst: dict[str, Any],
    momentum_analyst: dict[str, Any],
    temperature: float = 0.55,
    max_tokens: int = 1100,
) -> AsyncIterator[str]:
    pack = {
        "user_question": query,
        "portfolio_analyst": _compact_agent(portfolio_analyst),
        "market_analyst": _compact_agent(market_analyst),
        "risk_agent": _compact_agent(risk_analyst),
        "momentum_agent": _compact_agent(momentum_analyst),
    }
    messages = assemble_messages(
        system=SYS_CHAIR,
        user=json.dumps(pack, indent=2, default=str),
    )
    try:
        async for piece in llm.stream_text(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield piece
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc)) from exc
