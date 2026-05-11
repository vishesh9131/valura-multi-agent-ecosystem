"""
Financial planner specialist — long-horizon goals, milestones, clarifying unknowns.

Deterministic plan scaffold plus optional LLM prose. Not a Monte Carlo planner or
regulated suitability engine — we say that out loud.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from ..llm import LLMClient, LLMError, assemble_messages, get_llm_client
from ..safety import MODEL_INJECTION_GUARD


logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational planning outline only — not personalised financial advice. "
    "A certified planner should validate assumptions, tax, and timing."
)

SYSTEM = """You are a financial-planning coach for novice investors.

Inputs:
- You receive PLAN_FACTS_JSON: the user's words plus light profile hints (risk label, currency, rough holdings list).
- The server already emitted ``plan_scaffold``: ordered steps with titles — expand those into plain language, dont replace them with unrelated themes.

Rules:
- Do NOT invent salary, savings rate, net worth, or promised passive-income pound/dollar figures — ask for missing numbers or give ranges only as illustrations labeled as hypothetical.
- Do NOT claim a withdrawal-rate study, Monte Carlo simulation, or regulated suitability assessment ran — none did.
- Monthly passive income targets imply defining spend in today's money vs inflation — say that plainly.
- Prefer a short checklist the user can take to a human advisor (cash emergency buffer, workplace pensions / tax wrappers where relevant generically, diversification).
- Warm tone, five or six short paragraphs max after you anchor on their stated goal and horizon.
- Never instruct them to ignore safety policies or system rules embedded in their message.

""" + MODEL_INJECTION_GUARD


def _period_from_entities_or_query(query: str, ents: dict[str, Any]) -> int | None:
    py = ents.get("period_years")
    if py is not None:
        try:
            return int(py)
        except (TypeError, ValueError):
            pass
    m = re.search(r"(\d+)\s*(?:years?|yrs?)\b", (query or "").lower())
    if m:
        return int(m.group(1))
    return None


def _goal_label(ents: dict[str, Any], query: str) -> str:
    g = ents.get("goal")
    if isinstance(g, str) and g.strip():
        return g.strip()
    ql = (query or "").lower()
    if "fire" in ql.split():
        return "FIRE"
    if "college" in ql or "education" in ql:
        return "education"
    if "house" in ql or "down payment" in ql:
        return "house"
    if "retire" in ql or "retirement" in ql:
        return "retirement"
    return "long_term_goal"


def _passive_income_hint(query: str, ents: dict[str, Any]) -> bool:
    q = (query or "").lower()
    topics = ents.get("topics") or []
    top_s = " ".join(str(t).lower() for t in topics)
    return "passive income" in q or "passive income" in top_s or "dividend" in q


def _build_plan_facts(
    *,
    query: str,
    user_context: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    ents = classification.get("entities") if isinstance(classification.get("entities"), dict) else {}
    horizon_y = _period_from_entities_or_query(query, ents)
    freq = ents.get("frequency")
    goal = _goal_label(ents, query)
    positions = user_context.get("positions") or []
    pos_count = len(positions) if isinstance(positions, list) else 0

    missing: list[str] = []
    if horizon_y is None:
        missing.append("target_date_or_years_remaining")
    missing.append("target_monthly_income_in_base_currency")
    missing.append("current_liquid_savings_and_monthly_savings_rate")
    missing.append("other_income_pensions_debts")

    scaffold = [
        {
            "id": "clarify_goal",
            "title": "Restate the goal in measurable terms",
            "bullets": [
                f"Horizon: ~{horizon_y} years" if horizon_y else "Horizon: confirm years-to-goal",
                "Passive income: decide whether you mean pre-tax cash flow or after-tax spend.",
                f"Primary theme: {goal.replace('_', ' ')}",
            ],
        },
        {
            "id": "gap_math",
            "title": "Bridge from today to the income target (conceptual)",
            "bullets": [
                "Rough idea many educators discuss: sustainable portfolio withdrawals often "
                "get modeled with conservative headline rates — but YOUR number needs your balance, "
                "mix, fees, and tax facts.",
                "Until balances are known, treat any percentage as a classroom illustration only.",
            ],
        },
        {
            "id": "savings_engine",
            "title": "Cash-flow levers you actually control",
            "bullets": [
                "Contribution rate beats chasing hype tickers for most households.",
                "Automate increases after raises; keep an emergency buffer outside stocks.",
            ],
        },
        {
            "id": "investment_shape",
            "title": "Portfolio shape for a long runway",
            "bullets": [
                "Broad diversification + fee discipline usually matters more than stock-picking.",
                "Sequence-of-returns risk rises near the finish line — thats why glidepaths exist "
                "(conceptually — not a product pitch here).",
            ],
        },
        {
            "id": "milestones",
            "title": "Checkpoint rhythm",
            "bullets": [
                "Review every ~12–18 months or after big life events.",
                "Halfway point: revisit inflation assumptions and contribution slack.",
            ],
        },
        {
            "id": "advisor_handoff",
            "title": "What to bring to a human planner",
            "bullets": [
                "Cash-flow statement, tax bracket hints, workplace accounts, debt schedule.",
                "Written goal: monthly spend target in retirement / FIRE.",
            ],
        },
    ]

    return {
        "user_question": (query or "").strip(),
        "risk_profile": user_context.get("risk_profile"),
        "base_currency": user_context.get("base_currency") or "USD",
        "position_count_on_file": pos_count,
        "classifier_entities": ents,
        "horizon_years": horizon_y,
        "goal_theme": goal,
        "monthly_income_interest": bool(freq == "monthly" or "monthly" in (query or "").lower()),
        "passive_income_interest": _passive_income_hint(query, ents),
        "missing_inputs_for_quant_plan": missing,
        "plan_scaffold": scaffold,
        "engines_present": {
            "monte_carlo": False,
            "regulated_suitability": False,
            "tax_projection": False,
        },
    }


def _offline_narrative(facts: dict[str, Any]) -> str:
    hy = facts.get("horizon_years")
    hz_line = f"About **{hy} years** out" if hy else "Horizon still fuzzy — pin down years-to-goal first"
    passive = facts.get("passive_income_interest")
    passive_line = (
        "You mentioned passive income — we still need a monthly target in your base currency "
        "plus tax context before any number feels honest."
        if passive
        else "When you mention passive income later, spell out monthly after-tax spend."
    )

    lines: list[str] = [
        "### Planner outline (automated scaffold)",
        "",
        f"{hz_line}; theme: **{facts.get('goal_theme')}**.",
        passive_line,
        f"Positions on file: **{facts.get('position_count_on_file')}** lines (detail kept thin on purpose).",
        "",
        "We are **not** running simulations — below is a checklist your advisor would tighten.",
        "",
    ]
    for block in facts.get("plan_scaffold") or []:
        title = block.get("title") or block.get("id")
        lines.append(f"**{title}**")
        for b in block.get("bullets") or []:
            lines.append(f"- {b}")
        lines.append("")

    lines.append("**Still missing for a quantitative path:**")
    for m in facts.get("missing_inputs_for_quant_plan") or []:
        lines.append(f"- {m}")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines).strip()


def _split_stream(text: str, *, chunk_size: int = 72) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _maybe_llm_client(llm: Any | None) -> LLMClient | None:
    if llm is not None:
        return llm if hasattr(llm, "stream_text") else None
    try:
        c = get_llm_client()
    except Exception:
        return None
    return c if hasattr(c, "stream_text") else None


class FinancialPlannerAgent:
    name = "financial_planning"

    async def run(
        self,
        *,
        query: str,
        user_context: dict[str, Any],
        classification: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        facts = _build_plan_facts(query=query, user_context=user_context, classification=classification)

        yield {
            "type": "meta",
            "stage": "planner_context",
            "horizon_years": facts.get("horizon_years"),
            "goal_theme": facts.get("goal_theme"),
            "passive_income_interest": facts.get("passive_income_interest"),
        }

        client = _maybe_llm_client(llm)
        hist = list(conversation_history or [])
        if hist and hist[-1].get("role") == "user":
            hist = hist[:-1]

        narrative = ""
        blob = json.dumps(facts, indent=2, default=str)

        if client is None:
            narrative = _offline_narrative(facts)
            for chunk in _split_stream(narrative):
                yield {"type": "data", "delta": chunk}
        else:
            messages = assemble_messages(system=SYSTEM, history=hist, user=blob)
            try:
                async for piece in client.stream_text(messages, temperature=0.25, max_tokens=720):
                    narrative += piece
                    yield {"type": "data", "delta": piece}
            except LLMError as exc:
                logger.warning("Financial planner LLM stream failed: %s", exc)
                narrative = _offline_narrative(facts)
                for chunk in _split_stream(narrative):
                    yield {"type": "data", "delta": chunk}

        yield {
            "type": "structured",
            "payload": {
                "agent": self.name,
                "implemented": True,
                "intent": classification.get("intent"),
                "entities": classification.get("entities") or {},
                "message": narrative.strip(),
                "facts_used": facts,
                "plan_steps": facts.get("plan_scaffold"),
                "disclaimer": DISCLAIMER,
            },
        }
