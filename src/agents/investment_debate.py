"""
Dialectical specialist: bull vs bear framing, not price targets.

Surfaces three voices (bull, bear, synthesis) as one streamed answer so the
router can stay a single `agent` label while the UX still feels multi-perspective.
"""
from __future__ import annotations

import re
from typing import Any, AsyncIterator

DISCLAIMER = (
    "Educational debate framing only — not a buy/sell recommendation. "
    "Check assumptions with a licensed professional before acting."
)


def _horizon_phrase(ents: dict[str, Any]) -> str:
    py = ents.get("period_years")
    if py is not None:
        try:
            n = int(py)
            return f"over roughly the next {n} years"
        except (TypeError, ValueError):
            pass
    h = (ents.get("horizon") or "").strip()
    if h == "5_years":
        return "over roughly a five-year horizon"
    if h == "1_year":
        return "over roughly the next year"
    if h == "6_months":
        return "over roughly the next several months"
    return "over a multi-year holding horizon"


def _ticker(ents: dict[str, Any], query: str) -> str:
    tickers = ents.get("tickers") or []
    if tickers:
        return str(tickers[0]).strip().upper()
    m = re.search(r"\b([A-Z]{1,5})\b", query or "")
    return m.group(1).upper() if m else "the name"


def _bull_block(symbol: str) -> str:
    if symbol == "NVDA":
        return (
            "Bull case — arguments often cited FOR holding NVDA:\n"
            "• AI infrastructure demand and data-center buildout tailwinds\n"
            "• CUDA / software stack that many workloads still lean on\n"
            "• Hyperscaler and enterprise spending on accelerated compute\n"
            "• Inference growth as models move from training into production use\n"
            "• Optionality in robotics, simulation, and autonomous systems\n"
            "• Ecosystem of tools and ISVs that raises switching costs\n\n"
        )
    return (
        f"Bull case — arguments often cited FOR holding {symbol}:\n"
        "• Secular demand in the end markets this name serves\n"
        "• Product or technology differentiation vs peers\n"
        "• Scale economics and cash generation if the cycle cooperates\n"
        "• Balance sheet room to invest or return capital in a downdraft\n\n"
    )


def _bear_block(symbol: str) -> str:
    if symbol == "NVDA":
        return (
            "Bear case — arguments often cited AGAINST or for caution on NVDA:\n"
            "• Valuation multiple compression if growth normalizes\n"
            "• Competition from AMD, in-house accelerators, and custom silicon at large customers\n"
            "• Risk of AI capex cycles overshooting then digesting\n"
            "• Regulatory, export, and geopolitical friction in key regions\n"
            "• Semiconductor industry history of sharp inventory and pricing cycles\n"
            "• Margin pressure if mix shifts or customers renegotiate harder\n\n"
        )
    return (
        f"Bear case — arguments often cited AGAINST or for caution on {symbol}:\n"
        "• Cyclical or competitive pressure in the industry\n"
        "• Customer concentration or single-theme dependency\n"
        "• Regulation, litigation, or policy shifts\n"
        "• Execution risk if the growth story decelerates\n\n"
    )


def _synthesis_block(symbol: str, horizon: str) -> str:
    return (
        f"Synthesis — {horizon} on {symbol}:\n"
        "• The decision usually hinges on whether you believe the bull drivers are already priced in and how wide the outcomes are if the bear risks show up.\n"
        "• Watch what would *break* each story: e.g. evidence of demand air pockets, share loss in key accelerators, or a sustained multiple de-rating on peers too.\n"
        "• Practical monitoring: capex plans from large buyers, product roadmaps vs competition, inventory and pricing tone in semis, and your own position size vs risk tolerance.\n"
        "• Many investors size these names as a *range* of outcomes — not a single forecast — and rebalance when the thesis or the weight drifts.\n\n"
    )


def _split_stream(text: str, *, chunk_size: int = 64) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class InvestmentDebateAgent:
    name = "investment_debate"

    async def run(
        self,
        *,
        query: str,
        user_context: dict[str, Any],
        classification: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        ents = classification.get("entities") if isinstance(classification.get("entities"), dict) else {}
        sym = _ticker(ents, query)
        hz = _horizon_phrase(ents)
        td = classification.get("task_decomposition") if isinstance(classification.get("task_decomposition"), dict) else {}

        yield {
            "type": "meta",
            "stage": "dialectical_start",
            "ticker": sym,
            "orchestration_mode": "dialectical_bull_bear_synthesis",
            "task_decomposition": td,
        }

        sections: list[dict[str, str]] = []

        bull = _bull_block(sym)
        yield {"type": "meta", "stage": "debate_bull", "role": "bull_agent"}
        sections.append({"role": "bull", "body": bull.strip()})
        for ch in _split_stream(bull):
            yield {"type": "data", "delta": ch}

        bear = _bear_block(sym)
        yield {"type": "meta", "stage": "debate_bear", "role": "bear_agent"}
        sections.append({"role": "bear", "body": bear.strip()})
        for ch in _split_stream(bear):
            yield {"type": "data", "delta": ch}

        syn = _synthesis_block(sym, hz)
        yield {"type": "meta", "stage": "debate_synthesis", "role": "synthesis_agent"}
        sections.append({"role": "synthesis", "body": syn.strip()})
        for ch in _split_stream(syn):
            yield {"type": "data", "delta": ch}

        disclaimer_line = DISCLAIMER + "\n"
        for ch in _split_stream(disclaimer_line):
            yield {"type": "data", "delta": ch}

        full_text = bull + bear + syn + disclaimer_line
        yield {
            "type": "structured",
            "payload": {
                "agent": self.name,
                "implemented": True,
                "intent": classification.get("intent"),
                "entities": ents,
                "message": full_text.strip(),
                "debate_sections": sections,
                "primary_intent": "investment_debate",
                "sub_intents": ["bull_case_generation", "bear_case_generation", "long_horizon_analysis"],
                "disclaimer": DISCLAIMER,
            },
        }
