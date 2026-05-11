"""
Investment strategy agent.

Builds answers from the live ``user_context`` + classifier payload. Streams via
LLM when available; otherwise emits a narrative assembled only from those facts
(so we arent shipping one static stub paragraph).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from ..llm import LLMClient, LLMError, assemble_messages, get_llm_client
from ..safety import MODEL_INJECTION_GUARD
from .strategy_feasibility import build_decision_support_audit


logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is educational information only, not personalised investment advice. "
    "Consult a qualified professional before acting."
)

SYSTEM = """You are an investment-strategy coach for novice investors.

Inputs:
- You receive FACTS_JSON from the server. It always includes ``decision_support_audit``: deterministic feasibility /
  capability flags computed WITHOUT any portfolio optimizer, suitability engine, or uncertainty model.

Rules:
- If ``decision_support_audit.constraint_contradictions`` is non-empty, your **first 1–2 sentences** must restate that
  contradiction in plain words (you may paraphrase ``formal_statement`` inside each matching finding, but keep the
  logical impossibility explicit — not hedged into a vague tradeoff essay).
- If ``constraint_contradictions`` is **empty** and ``feasibility_findings`` is **empty**, **do not** lead with
  "no contradictions", "no feasibility issues", "nothing contradicts", or similar absence-of-problem framing — open by
  restating the users tax / timing / trade question and name key unknowns (basis, holding window, jurisdiction) before teaching.
- If ``decision_support_audit.feasibility_findings`` is non-empty, address those findings before general education.
- Use ONLY FACTS_JSON for user-specific facts; never invent holdings or risk labels.
- Never claim mean-variance optimization, stochastic simulation, regulated suitability scoring, or constraint solving ran — they did not.
- After stating contradictions, you may briefly explain *why* markets link risk and return in ordinary portfolios.
- Prefer frameworks and clarifying questions — avoid definitive buy/sell commands.
- About five short paragraphs total after the opening contradiction paragraph(s).
- Close with the reminder that professionals handle allocation and suitability properly.

""" + MODEL_INJECTION_GUARD


def _slim_positions(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in raw or []:
        if not isinstance(p, dict):
            continue
        t = p.get("ticker")
        if not t:
            continue
        row: dict[str, Any] = {
            "ticker": str(t).strip().upper(),
            "quantity": p.get("quantity"),
            "avg_cost": p.get("avg_cost"),
            "currency": (p.get("currency") or "USD"),
        }
        if "purchased_at" in p:
            row["purchased_at"] = p.get("purchased_at")
        out.append(row)
    return out


def _facts_bundle(
    *,
    query: str,
    user_context: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    ents = classification.get("entities")
    if not isinstance(ents, dict):
        ents = {}
    return {
        "user_question": (query or "").strip(),
        "risk_profile": user_context.get("risk_profile"),
        "base_currency": user_context.get("base_currency") or "USD",
        "user_id": user_context.get("user_id"),
        "positions": _slim_positions(user_context.get("positions") or []),
        "classifier_intent": classification.get("intent"),
        "classifier_entities": ents,
        "classifier_confidence": classification.get("confidence"),
    }


def _offline_capability_preamble(bundle: dict[str, Any]) -> list[str]:
    audit = bundle.get("decision_support_audit") or {}
    lines: list[str] = [
        "Capability boundary (deterministic system layer — not LLM prose):",
        audit.get("notes", ""),
        "",
        "Subsystems NOT invoked on this path:",
    ]
    for name, on in (audit.get("engines_present") or {}).items():
        lines.append(f"  - {name}: {on}")

    contradictions = audit.get("constraint_contradictions") or []
    if contradictions:
        lines.append("")
        lines.append("Constraint contradiction — formal identification:")
        for stmt in contradictions:
            lines.append(f"  • {stmt}")

    ff = audit.get("feasibility_findings") or []
    if ff:
        lines.append("")
        lines.append("Supporting feasibility records:")
        for f in ff:
            lines.append(f"  [{f.get('severity')}] {f.get('code')}")
            if f.get("constraint_contradiction"):
                lines.append(f"      constraint_contradiction=true")
            lines.append(f"      {f.get('detail')}")
    lines.append("")
    return lines


def _offline_narrative(bundle: dict[str, Any]) -> str:
    """Compose text from bundle + audit; avoids sounding like an optimizer ran."""
    lines: list[str] = []
    lines.extend(_offline_capability_preamble(bundle))

    q = bundle.get("user_question") or ""
    rp = bundle.get("risk_profile")
    if rp:
        lines.append(f"Stated risk profile in FACTS_JSON: {rp!r}.")

    ents = bundle.get("classifier_entities") or {}
    tickers = ents.get("tickers") or []
    sectors = ents.get("sectors") or []
    action = ents.get("action")
    topics = ents.get("topics") or []

    ce_bits: list[str] = []
    if action:
        ce_bits.append(f"action={action!r}")
    if tickers:
        ce_bits.append("tickers=" + ", ".join(str(x) for x in tickers[:12]))
    if sectors:
        ce_bits.append("sectors=" + ", ".join(str(x) for x in sectors[:12]))
    if topics:
        ce_bits.append("topics=" + ", ".join(str(x) for x in topics[:12]))
    if ce_bits:
        lines.append("Classifier entity snapshot: " + "; ".join(ce_bits) + ".")

    pos = bundle.get("positions") or []
    if pos:
        names = ", ".join(str(p.get("ticker")) for p in pos if p.get("ticker"))
        lines.append(f"Holdings on file ({len(pos)} line(s)): {names}.")
    else:
        lines.append("No holdings were supplied — keep guidance generic.")

    lines.append(
        "Conceptual framing you can reuse: expected return and risk usually move together; "
        "cash-like instruments damp volatility but historically trail equities over long horizons."
    )

    lines.append("")
    lines.append("FACTS_JSON:")
    lines.append(json.dumps(bundle, indent=2, default=str))
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


_TAXISH = re.compile(
    r"\b(tax|taxes|taxable|capital\s+gains?|ltcg|stcg|short[- ]term|long[- ]term|harvest)\b",
    re.IGNORECASE,
)


def _tax_sell_missing_holding_period(query: str, ents: dict[str, Any], positions_raw: list[Any]) -> bool:
    """True when user asks tax+sell flavored question about a ticker we hold but purchase date is blank."""
    q = (query or "").strip()
    if not _TAXISH.search(q):
        return False
    action = ents.get("action")
    sellish = action == "sell" or bool(re.search(r"\b(sell|selling)\b", q.lower()))
    if not sellish:
        return False
    tickers = [str(x).strip().upper() for x in (ents.get("tickers") or []) if str(x).strip()]
    if not tickers:
        return False
    want = set(tickers)
    for p in positions_raw or []:
        if not isinstance(p, dict):
            continue
        t = str(p.get("ticker") or "").strip().upper()
        if t not in want:
            continue
        pa = p.get("purchased_at")
        if pa is None or (isinstance(pa, str) and not pa.strip()):
            return True
    return False


def _maybe_llm_client(llm: Any | None) -> LLMClient | None:
    if llm is not None:
        return llm if hasattr(llm, "stream_text") else None
    try:
        c = get_llm_client()
    except Exception:
        return None
    return c if hasattr(c, "stream_text") else None


def _split_stream(text: str, *, chunk_size: int = 48) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class InvestmentStrategyAgent:
    name = "investment_strategy"

    async def run(
        self,
        *,
        query: str,
        user_context: dict[str, Any],
        classification: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        bundle = _facts_bundle(query=query, user_context=user_context, classification=classification)
        audit = build_decision_support_audit(bundle)
        bundle["decision_support_audit"] = audit

        ents = classification.get("entities") if isinstance(classification.get("entities"), dict) else {}
        pos_raw = user_context.get("positions") or []
        if _tax_sell_missing_holding_period(query, ents, pos_raw):
            yield {
                "type": "meta",
                "stage": "clarification_suggested",
                "missing_fields": ["holding_period_or_purchase_date"],
                "hint": "ST vs LT rates hinge on holding window — add purchase date or say roughly how long youve held.",
            }

        yield {
            "type": "meta",
            "stage": "strategy_context",
            "position_count": len(bundle["positions"]),
            "risk_profile": bundle.get("risk_profile"),
            "decision_support_severity": audit.get("severity_max"),
            "human_advisor_recommended": audit.get("human_advisor_recommended"),
        }

        client = _maybe_llm_client(llm)
        hist = list(conversation_history or [])
        if hist and hist[-1].get("role") == "user":
            hist = hist[:-1]

        user_blob = json.dumps(bundle, indent=2, default=str)
        narrative = ""

        if client is None:
            narrative = _offline_narrative(bundle)
            for chunk in _split_stream(narrative):
                yield {"type": "data", "delta": chunk}
        else:
            messages = assemble_messages(system=SYSTEM, history=hist, user=user_blob)
            try:
                async for piece in client.stream_text(messages, temperature=0.28, max_tokens=650):
                    narrative += piece
                    yield {"type": "data", "delta": piece}
            except LLMError as exc:
                logger.warning("Investment strategy LLM stream failed: %s", exc)
                narrative = _offline_narrative(bundle)
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
                "facts_used": bundle,
                "decision_support_audit": audit,
                "disclaimer": DISCLAIMER,
            },
        }
