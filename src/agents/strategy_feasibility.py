"""
Deterministic feasibility / capability boundary — not an optimizer.

Finance-grade systems need convex solvers, suitability rules, scenario trees, etc.
This module only surfaces lexical + profile tensions so we dont *sound* like those
ran when they didnt.
"""
from __future__ import annotations

import re
from typing import Any


def _wants_effectively_zero_risk(q: str) -> bool:
    ql = q.lower()
    if "risk-free" in ql or "risk free" in ql:
        return True
    if re.search(r"\b(zero|no)\s+risk\b", ql):
        return True
    if "without risk" in ql or "no downside" in ql:
        return True
    return False


def _wants_maximum_return(q: str) -> bool:
    ql = q.lower()
    if "maximum return" in ql or "max return" in ql or "maximum returns" in ql:
        return True
    if re.search(r"\b(max|maximum|highest)\b", ql) and ("return" in ql or "gain" in ql or "growth" in ql):
        return True
    return False


def _conservative_profile(bundle: dict[str, Any]) -> bool:
    rp = (bundle.get("risk_profile") or "").lower()
    return any(x in rp for x in ("conservative", "low risk", "cautious", "capital preservation"))


def _aggressive_goal_language(q: str) -> bool:
    ql = q.lower()
    return _wants_maximum_return(ql) or "beat the market" in ql or "best performing" in ql


def build_decision_support_audit(bundle: dict[str, Any]) -> dict[str, Any]:
    """Structured honesty layer for API consumers + LLM grounding."""
    q = bundle.get("user_question") or ""
    findings: list[dict[str, str]] = []

    if _wants_effectively_zero_risk(q) and _wants_maximum_return(q):
        findings.append(
            {
                "code": "infeasible_tradeoff_max_return_zero_risk",
                "severity": "high",
                "constraint_contradiction": True,
                "objectives_in_conflict": [
                    "maximize_returns_or_seek_maximum_upside",
                    "eliminate_or_zero_out_investment_risk",
                ],
                "formal_statement": (
                    "Under the usual definitions in liquid public markets, no investment simultaneously maximizes "
                    "returns and eliminates all risk — those two objectives contradict each other unless you "
                    "narrow one of them (for example capped upside with principal protection)."
                ),
                "detail": (
                    "Constraint contradiction (optimization language): you cannot jointly optimize for unconstrained "
                    "maximum return and strictly zero investment risk; relaxing one objective or re-scoping "
                    "(time horizon, capital guarantee, insured caps) is required for a feasible problem statement."
                ),
            }
        )

    if _conservative_profile(bundle) and _aggressive_goal_language(q):
        findings.append(
            {
                "code": "profile_goal_mismatch",
                "severity": "moderate",
                "detail": (
                    "Stated risk profile reads conservative while the question uses aggressive return-max language — "
                    "needs explicit preference reconciliation before any concrete plan."
                ),
            }
        )

    severities = [f.get("severity", "info") for f in findings]
    rank = {"high": 3, "moderate": 2, "info": 1}
    severity_max = max((rank.get(s, 0) for s in severities), default=0)
    severity_label = {3: "high", 2: "moderate", 1: "info", 0: "none"}[severity_max]

    contradictions = [str(f["formal_statement"]) for f in findings if f.get("constraint_contradiction")]

    return {
        "advisory_mode": "conversational_framework_only",
        "engines_present": {
            "portfolio_optimizer": False,
            "allocation_engine": False,
            "suitability_model": False,
            "uncertainty_model": False,
            "constraint_solver": False,
            "retrieval": False,
        },
        "feasibility_findings": findings,
        "constraint_contradictions": contradictions,
        "severity_max": severity_label,
        "human_advisor_recommended": bool(findings),
        "notes": (
            "No numerical optimization, suitability scoring, or probabilistic scenario engine executed — "
            "only deterministic checks above plus optional LLM prose."
        ),
    }
