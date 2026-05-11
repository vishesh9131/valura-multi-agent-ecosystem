"""
Entity matcher used by the classifier routing tests.

Implements the rules in fixtures/README.md:
    tickers      : case-fold + drop exchange suffix; subset match
    topics/sectors: lowercase substring per element; subset match
    amount/rate  : within +/- 5%
    period_years : exact integer
    currency     : ISO 4217 exact
    index        : exact match against the canonical name (S&P 500 etc.)
    action/goal/frequency/horizon/time_period : vocabulary token, case-insensitive

We deviate from the upstream skeleton matcher in one place: `index`
is matched after collapsing whitespace ("S&P 500" == "S&P500").
"""
from __future__ import annotations

import re
from typing import Any


def normalize_ticker(t: str) -> str:
    return t.upper().split(".")[0]


def normalize_index(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Subset match with normalization. `actual` must contain every expected value."""
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {normalize_ticker(t) for t in exp_value}
            act_set = {normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            act_lc = [s.lower() for s in act_value]
            for needle in exp_value:
                n = needle.lower()
                if not any(n in hay for hay in act_lc):
                    return False
        elif field in ("amount", "rate"):
            try:
                a = float(act_value)
                e = float(exp_value)
            except (TypeError, ValueError):
                return False
            if abs(a - e) > abs(e) * 0.05:
                return False
        elif field == "period_years":
            try:
                if int(act_value) != int(exp_value):
                    return False
            except (TypeError, ValueError):
                return False
        elif field == "index":
            if normalize_index(act_value) != normalize_index(exp_value):
                return False
        else:
            # vocabulary tokens: action/goal/frequency/horizon/time_period/currency/etc.
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True
