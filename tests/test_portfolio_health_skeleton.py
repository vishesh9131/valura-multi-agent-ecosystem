"""Portfolio Health agent — synchronous compute layer."""
from __future__ import annotations

from src.agents.portfolio_health import assess


def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user):
    user = load_user("usr_004")
    response = assess(user)
    assert response is not None
    assert "disclaimer" in response
    assert response["mode"] == "build"
    assert response["next_steps"]
    assert response["concentration_risk"]["flag"] == "n/a"


def test_portfolio_health_flags_concentration(load_user):
    user = load_user("usr_003")
    response = assess(user)
    assert response["concentration_risk"]["flag"] in {"high", "warning"}
    # concentrated holding for usr_003 is NVDA
    assert response["concentration_risk"]["top_holding"] == "NVDA"


def test_portfolio_health_includes_disclaimer(load_user):
    user = load_user("usr_001")
    response = assess(user)
    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()


def test_portfolio_health_observations_are_short(load_user):
    user = load_user("usr_001")
    response = assess(user)
    assert 1 <= len(response["observations"]) <= 6
    for obs in response["observations"]:
        assert obs["severity"] in {"info", "warning", "critical"}
        assert obs["text"]


def test_portfolio_health_handles_multi_currency(load_user):
    user = load_user("usr_006")
    response = assess(user)
    assert "concentration_risk" in response
    assert "performance" in response
    assert response["benchmark_comparison"]["benchmark"] in {"MSCI World", None}


def test_portfolio_health_retiree_income_focus(load_user):
    user = load_user("usr_008")
    response = assess(user)
    assert response["mode"] == "monitor"
    # retiree holds dividend payers (VYM/SCHD/JNJ) — no income-gap nudge expected
    income_obs = [o for o in response["observations"] if "dividend" in o["text"].lower()]
    assert income_obs == []
