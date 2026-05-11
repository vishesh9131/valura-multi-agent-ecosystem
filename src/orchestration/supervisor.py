"""Supervisor: orchestrator-mediated rounds (portfolio → market → conversation)."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, AsyncIterator

from ..llm import LLMClient, LLMError
from ..session import SessionState
from . import collaborative_llm_agents as collab_llm
from .toolkits import conversation_tools as conv_tools
from .toolkits import finance_math as fin_math
from .toolkits import market_tools as mkt_tools
from .toolkits import overlap_tools as ov_tools
from .toolkits import portfolio_tools as port_tools

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# prose throws off naive ALLCAPS scan — "What should I know" yields token I
_BOGUS_QUERY_TICKERS = frozenset(
    {
        "I",
        "A",
        "AN",
        "AS",
        "AT",
        "BE",
        "BY",
        "DO",
        "GO",
        "IF",
        "IN",
        "IS",
        "IT",
        "ME",
        "MY",
        "NO",
        "OF",
        "ON",
        "OR",
        "SO",
        "TO",
        "UP",
        "WE",
        "OK",
    }
)


def _extract_tickers(query: str, positions: list[dict[str, Any]]) -> list[str]:
    """Prefer holdings tickers; from the query only take 2–5 letter tokens that arent english noise."""
    from_positions = [
        str(p.get("ticker") or "").strip().upper()
        for p in positions
        if (p.get("ticker") or "").strip()
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for t in from_positions:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    for raw in _TICKER_RE.findall(query or ""):
        t = raw.upper()
        if len(t) < 2:
            continue
        if t in _BOGUS_QUERY_TICKERS:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    return ordered[:5]


def _fmt_pct(x: Any, digits: int = 1) -> str:
    try:
        return f"{float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_px(x: Any) -> str:
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _overlap_bundle(tool_results: dict[str, Any]) -> dict[str, Any]:
    stub = tool_results.get("factor_overlap_stub")
    return stub if isinstance(stub, dict) else {}


def _market_tool_results_from_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for e in entries:
        if e.get("agent") == "market":
            return e.get("tool_results") or {}
    if len(entries) > 1:
        return entries[1].get("tool_results") or {}
    return {}


def _portfolio_facts(tool_results: dict[str, Any]) -> dict[str, Any]:
    alloc = tool_results.get("summarize_allocation") or {}
    conc = tool_results.get("flag_concentration") or {}
    snap = tool_results.get("snapshot_risk_profile") or {}
    return {
        "position_count": int(alloc.get("position_count") or 0),
        "conc_flag": conc.get("flag"),
        "top_holding": conc.get("top_holding"),
        "top_share_pct": conc.get("top_share_pct"),
        "risk_profile": snap.get("risk_profile"),
        "base_currency": snap.get("base_currency") or "USD",
    }


def _market_facts(tool_results: dict[str, Any]) -> dict[str, Any]:
    q = tool_results.get("get_quote") or {}
    r = tool_results.get("get_recent_returns") or {}
    ret = r.get("return_pct")
    insuf = (r.get("note") or "") == "insufficient_history"
    mom_word = "mixed"
    try:
        if ret is None or insuf:
            mom_word = "unknown"
        elif float(ret) >= 5:
            mom_word = "strong"
        elif float(ret) >= 0:
            mom_word = "constructive"
        else:
            mom_word = "weak"
    except (TypeError, ValueError):
        mom_word = "unknown"
    return {
        "ticker": q.get("ticker"),
        "price": q.get("price"),
        "sector": q.get("sector"),
        "return_pct": ret,
        "returns_note": r.get("note"),
        "momentum_label": mom_word,
    }


def _confidence_word(score: float) -> str:
    if score >= 72:
        return "high"
    if score >= 48:
        return "medium"
    return "low"


def _debate_state(pf: dict[str, Any], mf: dict[str, Any]) -> dict[str, Any]:
    """Tension + confidence labels for structured payloads (deterministic heuristics)."""
    risk_s, mom_s = _risk_momentum_scores(pf, mf)
    strat_s = 58.0
    risk_w, mom_w, strat_w = (
        _confidence_word(risk_s),
        _confidence_word(mom_s),
        _confidence_word(strat_s),
    )
    tension = "low"
    debate_note = "Panels mostly align — still worth spelling tradeoffs."
    if risk_s >= 65 and mom_s >= 65:
        tension = "high"
        debate_note = (
            "Tension: RiskAgent wants trim/diversification while MomentumAgent reads trend support — "
            "binary all-in vs all-out is usually the wrong frame."
        )
    elif risk_s >= 65 and mom_s < 50:
        tension = "moderate"
        debate_note = "Tension: Risk concern dominates weak momentum — de-risking themes sound louder."
    elif risk_s < 50 and mom_s >= 65:
        tension = "moderate"
        debate_note = "Tension: Tape strength runs ahead of book worry — watch oversizing into extension."
    return {
        "tension": tension,
        "debate_note": debate_note,
        "risk_confidence": risk_w,
        "momentum_confidence": mom_w,
        "strategy_confidence": strat_w,
        "risk_score": round(risk_s, 1),
        "momentum_score": round(mom_s, 1),
        "strategy_score": round(strat_s, 1),
    }


def _risk_momentum_scores(pf: dict[str, Any], mf: dict[str, Any]) -> tuple[float, float]:
    """Deterministic 0–100 confidence-ish scores for logging (not a statistical model)."""
    pct = pf.get("top_share_pct")
    flag = pf.get("conc_flag")
    risk_s = 35.0
    if flag == "warning":
        risk_s = 62.0
        try:
            if pct is not None and float(pct) >= 50:
                risk_s = 78.0
            elif pct is not None and float(pct) >= 40:
                risk_s = 70.0
        except (TypeError, ValueError):
            pass
    elif flag == "n/a":
        risk_s = 20.0

    mom_s = 40.0
    label = mf.get("momentum_label")
    if label == "strong":
        mom_s = 82.0
    elif label == "constructive":
        mom_s = 62.0
    elif label == "weak":
        mom_s = 38.0
    elif label == "unknown":
        mom_s = 28.0
    return risk_s, mom_s


def _first_sentence(text: str, max_len: int = 220) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for sep in ".?!":
        idx = t.find(sep)
        if 0 < idx < max_len:
            return t[: idx + 1].strip()
    return (t[:max_len] + "…") if len(t) > max_len else t


def _apply_llm_public_layer(entry: dict[str, Any], llm_out: dict[str, Any]) -> None:
    """Keep full model prose internal; user-visible caption stays short."""
    entry["llm"] = llm_out
    body = str(llm_out.get("answer") or "").strip()
    headline = str(llm_out.get("stance_headline") or "").strip()
    entry["utterance"] = headline or _first_sentence(body) or body[:240]
    inner = dict(entry.get("internal") or {})
    inner["model_answer"] = body
    inner["reasoning_trace"] = llm_out.get("reasoning_trace")
    entry["internal"] = inner


def _collaboration_signals(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Structured facts for API consumers — not user prose."""
    if not entries:
        return {}
    pf = _portfolio_facts(entries[0].get("tool_results") or {})
    mf = _market_facts(_market_tool_results_from_entries(entries))
    ret = mf.get("return_pct")
    ret_out: float | None
    try:
        ret_out = round(float(ret), 2) if ret is not None else None
    except (TypeError, ValueError):
        ret_out = None
    signals: dict[str, Any] = {
        "book_concentration_flag": pf.get("conc_flag"),
        "top_holding": pf.get("top_holding"),
        "top_holding_share_pct": pf.get("top_share_pct"),
        "stated_risk_profile": pf.get("risk_profile"),
        "primary_quote_symbol": mf.get("ticker"),
        "sector": mf.get("sector"),
        "momentum_regime": mf.get("momentum_label"),
        "approx_return_21d_pct": ret_out,
    }
    agree: list[str] = []
    disagree: list[str] = []
    mom = mf.get("momentum_label")
    if pf.get("conc_flag") == "warning":
        disagree.append("elevated_single_name_weight")
    if mom in {"strong", "constructive"}:
        agree.append("constructive_recent_performance")
    if pf.get("conc_flag") == "warning" and mom in {"strong", "constructive"}:
        disagree.append("momentum_vs_concentration")
    signals["agreement"] = agree
    signals["disagreement"] = disagree

    ol = _overlap_bundle(entries[0].get("tool_results") or {})
    pairs = ol.get("overlap_pairs") if isinstance(ol.get("overlap_pairs"), list) else []
    signals["etf_equity_overlap_pairs"] = len(pairs)
    signals["growth_tech_cluster_stub"] = ol.get("cluster_theme") == "high_beta_growth_tech_tilt"

    if pf.get("conc_flag") == "warning" and mom in {"strong", "constructive"}:
        signals["recommended_action_hint"] = "partial_diversification_or_position_caps"
    elif pf.get("conc_flag") == "warning":
        signals["recommended_action_hint"] = "review_concentration_and_risk_budget"
    else:
        signals["recommended_action_hint"] = "monitor_and_rebalance_on_rules"

    last_meta = entries[-1].get("collaboration_meta") if entries else None
    if isinstance(last_meta, dict):
        if last_meta.get("tension"):
            signals["panel_tension"] = last_meta["tension"]
        if last_meta.get("mode") == "llm_multiagent":
            signals["risk_confidence_proxy"] = last_meta.get("risk_score_proxy")
            signals["momentum_confidence_proxy"] = last_meta.get("momentum_score_proxy")
    return signals


def _portfolio_round(user_context: dict[str, Any]) -> dict[str, Any]:
    positions = port_tools.list_positions(user_context)
    # overlap toolkit is stubbed — still useful so downstream rounds dont pretend tickers are uncorrelated
    stub_out = ov_tools.toolkit_factor_overlap_stub(user_context)
    tools_out = {
        "list_positions": positions,
        "summarize_allocation": port_tools.summarize_allocation(user_context),
        "flag_concentration": port_tools.flag_concentration(user_context),
        "match_tickers_to_positions": port_tools.match_tickers_to_positions(
            user_context, _extract_tickers("", positions)
        ),
        "snapshot_risk_profile": port_tools.snapshot_risk_profile(user_context),
        "factor_overlap_stub": stub_out,
    }
    pf = _portfolio_facts(tools_out)
    rp = pf["risk_profile"] or "unspecified"
    cc = pf["base_currency"]
    top = pf["top_holding"] or "—"
    pct_s = _fmt_pct(pf["top_share_pct"])
    if pf["position_count"] <= 0:
        public = (
            "There are no positions on file yet, so we cant relate concentration or sector tilts to your actual account."
        )
        internal = f"[tools/portfolio] empty_book positions=0"
    elif pf["conc_flag"] == "warning":
        public = (
            f"Your largest line ({top}) is roughly {pct_s} of share count — meaningful single-name weight "
            f"for someone with a {rp} risk posture."
        )
        internal = (
            f"[tools/portfolio] positions={pf['position_count']} concentration=warning "
            f"top={top} top_share={pct_s} risk_profile={rp} currency={cc}"
        )
    else:
        public = (
            f"{top} is your biggest position at about {pct_s} of share count — not flashing our coarse concentration alarm, "
            f"but it still drives outcomes if the name gaps."
        )
        internal = (
            f"[tools/portfolio] positions={pf['position_count']} concentration=ok top={top} "
            f"top_share={pct_s} risk_profile={rp} currency={cc}"
        )

    ol_pairs = stub_out.get("overlap_pairs") or []
    ol_cluster = stub_out.get("cluster_theme")
    if pf["position_count"] > 0 and ol_cluster == "high_beta_growth_tech_tilt":
        public = (
            "Portfolio lens: you already run multiple growth / Nasdaq-tech-linked lines — thats current concentration, "
            "not a hypothetical future tilt. "
            + public
        )
    if pf["position_count"] > 0 and ol_pairs:
        tail = (stub_out.get("summary_public") or "").strip()
        if tail:
            public = f"{public} {tail}".strip()

    internal = (
        internal
        + f" overlap_pairs={len(ol_pairs)} cluster={ol_cluster}"
    )

    return {
        "agent": "portfolio",
        "round": 1,
        "tool_results": tools_out,
        "utterance": public,
        "internal": {"deliberation": internal, "facts": pf},
    }


def _market_round(
    user_context: dict[str, Any],
    query: str,
    portfolio_peer: dict[str, Any],
    *,
    round_no: int = 2,
) -> dict[str, Any]:
    positions = port_tools.list_positions(user_context)
    tickers = _extract_tickers(query, positions)
    primary = tickers[0] if tickers else "SPY"
    qinfo = mkt_tools.toolkit_get_quote(primary)
    sector = qinfo.get("sector") if isinstance(qinfo, dict) else None
    tools_out = {
        "get_quote": qinfo,
        "get_recent_returns": mkt_tools.toolkit_get_recent_returns(primary, days=21),
        "sector_benchmark_stub": mkt_tools.toolkit_sector_benchmark_stub(sector),
        "symbol_resolve": mkt_tools.toolkit_symbol_resolve(primary),
        "market_hours_stub": mkt_tools.toolkit_market_hours_stub("US"),
    }
    peer_tr = portfolio_peer.get("tool_results") or {}
    pf = _portfolio_facts(peer_tr)
    mf = _market_facts(tools_out)
    px = _fmt_px(mf["price"])
    sec = mf["sector"] or "unknown sector"
    ret = mf["return_pct"]
    mom_lab = mf["momentum_label"]
    internal = (
        f"[tools/market] symbol={primary} last_px_usd={px} sector={sec} ret21d={ret} "
        f"momentum_label={mom_lab} book_flag={pf['conc_flag']} book_top={pf['top_holding']}"
    )
    top_name = pf["top_holding"] or "your top holding"
    share_w = _fmt_pct(pf["top_share_pct"])
    if pf["conc_flag"] == "warning":
        conc_phrase = "meaningfully tilted toward one dominant line"
    elif pf["conc_flag"] == "ok":
        conc_phrase = "not extreme under this coarse lens"
    else:
        conc_phrase = "not scored yet"
    if ret is None or mf.get("returns_note") == "insufficient_history":
        public = (
            f"{primary} trades in {sec}. We dont have a solid multi-week return window in this feed, "
            f"so treat momentum stories lightly — structure still matters with {top_name} around {share_w} of share count "
            f"({conc_phrase})."
        )
    else:
        try:
            rpct = float(ret)
            rough = round(abs(rpct), 1)
            direction = "up" if rpct >= 0 else "down"
            public = (
                f"{primary} has been {direction} roughly {rough}% over the past few weeks, "
                f"with shares near {px} USD in {sec}. Against your book, {top_name} is still about {share_w} of share count "
                f"while the sleeve looks {conc_phrase}."
            )
        except (TypeError, ValueError):
            public = (
                f"{primary} sits in {sec} near {px} USD; return stats didnt parse cleanly — "
                f"pair that uncertainty with how concentrated you are on {top_name} (~{share_w})."
            )

    ol_peer = peer_tr.get("factor_overlap_stub") if isinstance(peer_tr.get("factor_overlap_stub"), dict) else {}
    if ol_peer.get("overlap_pairs"):
        public = (
            public.rstrip()
            + " Tape/context angle: megacap indices like QQQ already wrap several growth-tech singles — "
            "pairing the ETF with some of those names stacks the same macro shock instead of spreading it."
        )

    return {
        "agent": "market",
        "round": round_no,
        "tool_results": tools_out,
        "utterance": public,
        "internal": {"deliberation": internal, "facts": mf},
    }


def _tax_math_round(
    user_context: dict[str, Any],
    query: str,
    classification: dict[str, Any],
    portfolio_peer: dict[str, Any],
    *,
    round_no: int = 2,
) -> dict[str, Any]:
    ents = classification.get("entities") if isinstance(classification.get("entities"), dict) else {}
    tickers = ents.get("tickers") or []
    positions = port_tools.list_positions(user_context)
    ordered = _extract_tickers(query, positions)
    ticker = (tickers[0] if tickers else None) or (ordered[0] if ordered else "")
    bundle = fin_math.toolkit_tax_gain_bundle(user_context, ticker=str(ticker))
    utterance = bundle.get("summary_public") or ""
    internal = bundle.get("summary_internal") or ""
    return {
        "agent": "tax_math",
        "round": round_no,
        "tool_results": {"tax_gain_bundle": bundle},
        "utterance": utterance,
        "internal": {"deliberation": internal, "facts": bundle},
    }


def _tax_market_synthesis_round(
    query: str,
    portfolio_peer: dict[str, Any],
    tax_peer: dict[str, Any],
    market_peer: dict[str, Any],
    *,
    round_no: int = 4,
) -> dict[str, Any]:
    tb = (tax_peer.get("tool_results") or {}).get("tax_gain_bundle") or {}
    mf = _market_facts(market_peer.get("tool_results") or {})
    pf = _portfolio_facts(portfolio_peer.get("tool_results") or {})
    sym = tb.get("ticker") or mf.get("ticker") or pf.get("top_holding") or "the name"
    mom = mf.get("momentum_label") or "mixed"
    gn = tb.get("estimated_unrealized_gain_notional")
    try:
        gn_s = f"{float(gn):,.2f}" if gn is not None else "n/a"
    except (TypeError, ValueError):
        gn_s = "n/a"
    utterance = (
        f"Synthesis: youre threading tax efficiency versus keeping exposure on {sym}. "
        f"Desk-side gain magnitude lands near {gn_s} USD notional on the stub quote — tape momentum reads `{mom}` "
        f"in this feed. That pairing is exactly why we dont broadcast a definitive sell — "
        "size any decision against brackets and timelines with someone licensed."
    )
    return {
        "agent": "synthesis",
        "round": round_no,
        "tool_results": {},
        "utterance": utterance,
        "internal": {"deliberation": "[synthesis/tax_market]", "facts": {"tax": tb, "market": mf}},
        "collaboration_meta": {"mode": "tax_market_synthesis"},
    }


def _compose_tax_market_user_message(entries: list[dict[str, Any]], query: str) -> str:
    chunks = [f"You asked: {query.strip()}", ""]
    for e in entries:
        u = (e.get("utterance") or "").strip()
        if u:
            chunks.append(u)
            chunks.append("")
    chunks.append(conv_tools.toolkit_disclaimer_block())
    return "\n".join(chunks).strip()


def _conversation_round(
    history: list[dict[str, str]],
    query: str,
    last_intent: str | None,
    portfolio_peer: dict[str, Any],
    market_peer: dict[str, Any],
) -> dict[str, Any]:
    priors = conv_tools.toolkit_prior_user_turns(history)
    tools_out = {
        "prior_user_turns": priors,
        "format_numbered_list": conv_tools.toolkit_format_numbered_list(priors[-5:]),
        "thread_summary_stub": conv_tools.toolkit_thread_summary_stub(history),
        "disclaimer_block": conv_tools.toolkit_disclaimer_block(),
        "suggest_handoff_intent": conv_tools.toolkit_suggest_handoff_intent(query, last_intent),
    }
    pf = _portfolio_facts(portfolio_peer.get("tool_results") or {})
    mf = _market_facts(market_peer.get("tool_results") or {})
    stub_pf = _overlap_bundle(portfolio_peer.get("tool_results") or {})
    overlap_pairs = stub_pf.get("overlap_pairs") or []
    macro_worry = ov_tools.recession_query_hint(query)
    dstate = _debate_state(pf, mf)
    risk_w, mom_w, strat_w = (
        dstate["risk_confidence"],
        dstate["momentum_confidence"],
        dstate["strategy_confidence"],
    )
    tension = dstate["tension"]
    debate_note = dstate["debate_note"]
    top = pf["top_holding"] or "the largest line"
    ticker = mf["ticker"] or top
    mom_lab = mf["momentum_label"]

    if pf["position_count"] > 0 and macro_worry and overlap_pairs:
        risk_line = (
            f"RiskAgent (confidence {risk_w}): "
            "recession narratives squeeze overlapping growth factors — ETF sleeves plus some of the same singles dont diversify that scenario."
        )
    elif pf["conc_flag"] == "warning":
        risk_line = (
            f"RiskAgent (confidence {risk_w}): "
            f'the book tilts hard into "{top}" (~{_fmt_pct(pf.get("top_share_pct"))} of share count); '
            f"for a {pf.get('risk_profile') or 'unspecified'} profile thats a live sizing debate."
        )
    elif pf["position_count"] <= 0:
        risk_line = "RiskAgent (confidence low): no positions on file — nothing to stress-test yet."
    else:
        risk_line = (
            f"RiskAgent (confidence {risk_w}): "
            f"concentration looks tame in this coarse scan — still watch gap risk on {top}."
        )

    if mom_lab == "unknown":
        mom_line = (
            f"MomentumAgent (confidence {mom_w}): "
            f"tape trend on {ticker} is unclear from the stub window — dont overfit a story."
        )
    elif mom_lab == "strong":
        mom_line = (
            f"MomentumAgent (confidence {mom_w}): "
            f"{ticker} shows strong recent performance — trend riders argue patience; risk still argues size."
        )
    elif mom_lab == "constructive":
        mom_line = (
            f"MomentumAgent (confidence {mom_w}): "
            "drift looks constructive — not a melt-up, but not fighting you."
        )
    else:
        mom_line = (
            f"MomentumAgent (confidence {mom_w}): "
            "recent path looks weak — clashes with a heavy book weight if you need stability."
        )

    strat_line = (
        f"StrategyAgent (confidence {strat_w}): "
        "square the circle with actions not slogans — partial hedge, staged sells, "
        "or hard caps on single-name weight beat pretending theres one truth."
    )

    internal_deliberation = "\n".join(
        [
            "[orchestration/internal]",
            f"user_turn_digest={tools_out['thread_summary_stub']}",
            f"routing_hint={tools_out['suggest_handoff_intent']}",
            risk_line,
            mom_line,
            strat_line,
            f"panel_tension={tension}",
            debate_note,
        ]
    )

    if pf["position_count"] > 0 and macro_worry and overlap_pairs:
        public = (
            "Chair note: macro fear plus stacked Nasdaq-style overlap means shock sensitivity runs hotter than separate tickers imply — "
            "staged trims toward bonds, IG credit, dividend growers, defensive sectors can dampen without pretending we timed the cycle."
        )
    elif pf["position_count"] <= 0:
        public = (
            "Until we see actual holdings, market commentary stays hypothetical — wire positions when you want a grounded read."
        )
    elif tension == "high":
        public = (
            "Structurally the book looks concentrated while the tape still reads constructive — those forces disagree, "
            "so the adult version of this decision is about how much single-name risk you want to carry, not about declaring a winner."
        )
    elif tension == "moderate":
        public = (
            "Portfolio shape and recent performance arent telling the same simple story — worth translating that tension into sizing rules "
            "rather than an all-in or all-out call."
        )
    else:
        public = (
            "Across holdings and recent tape behavior nothing here demands a dramatic verdict — still keep an eye on how much of your outcome rides on one or two names."
        )

    return {
        "agent": "conversation",
        "round": 3,
        "tool_results": tools_out,
        "utterance": public,
        "internal": {"deliberation": internal_deliberation},
        "collaboration_meta": dstate,
    }


def _compose_final_user_message(entries: list[dict[str, Any]], query: str) -> str:
    """Clean investor-facing narrative — no routing vocabulary, no raw tool dumps."""
    if not entries:
        return conv_tools.toolkit_disclaimer_block()
    pf_tr = entries[0].get("tool_results") or {}
    pf = _portfolio_facts(pf_tr)
    mf = _market_facts(_market_tool_results_from_entries(entries))
    ol = _overlap_bundle(pf_tr)
    cluster = ol.get("cluster_theme")
    pairs = ol.get("overlap_pairs") or []
    held = ol.get("held_tickers") or []
    macro_worry = ov_tools.recession_query_hint(query)
    risk_s, mom_s = _risk_momentum_scores(pf, mf)
    top = pf["top_holding"] or "your largest position"
    ticker = mf.get("ticker") or top

    if pf["position_count"] <= 0:
        para1 = (
            "We dont see holdings on file yet, so we cant relate market moves to how youre actually positioned."
        )
    elif cluster == "high_beta_growth_tech_tilt" and pf["position_count"] > 0:
        sleeve = ", ".join(str(x) for x in held[:8]) if held else "these holdings"
        rp = pf["risk_profile"] or "moderate"
        para1 = (
            f"You already skew growth / Nasdaq-style tech across {sleeve} — thats live sizing, not a heads-up about some future allocation drift. "
            f"Largest line stays {top} (~{_fmt_pct(pf['top_share_pct'])} of share count) with a stated {rp} risk posture."
        )
    elif pf["conc_flag"] == "warning":
        rp = pf["risk_profile"] or "moderate"
        para1 = (
            f"You do appear meaningfully exposed through {top}, roughly {_fmt_pct(pf['top_share_pct'])} of share count "
            f"alongside a {rp} risk posture. Concentrated sleeves often feel great on the way up but bounce harder when sentiment turns."
        )
    else:
        para1 = (
            f"Your largest line is {top} at about {_fmt_pct(pf['top_share_pct'])} of share count — not flashing our coarse concentration alarm, "
            "yet outcomes still swing with that name."
        )

    overlap_para = ""
    if pairs:
        overlap_para = (
            "Overlap angle (illustrative map): liquid Nasdaq ETFs such as QQQ embed several megacap tech names — holding the basket plus some of those singles "
            "layers the same factor shocks, so headline diversification can oversell how many independent bets you truly have."
        )

    if mf.get("returns_note") == "insufficient_history" or mf.get("return_pct") is None:
        para2 = (
            f"{ticker} sits in {mf.get('sector') or 'its sector'}, though we lack a reliable multi-week performance snapshot here — "
            "treat momentum chatter as directional color only."
        )
    else:
        try:
            rpct = float(mf["return_pct"])
            rough = round(abs(rpct), 1)
            verb = "gained" if rpct >= 0 else "lost"
            para2 = (
                f"{ticker} has {verb} roughly {rough}% over roughly the past three weeks while trading in {mf.get('sector') or 'its sector'}. "
                "Strength doesnt erase sizing risk when one name drives much of the book."
            )
        except (TypeError, ValueError):
            para2 = (
                f"{ticker} trades in {mf.get('sector') or 'its sector'} — performance figures didnt parse cleanly, "
                "so lean on structure (how concentrated you are) more than a precise momentum score."
            )

    if macro_worry and cluster == "high_beta_growth_tech_tilt":
        para3 = (
            "Pulling threads together: recession fear tends to punish high-beta growth as a block — and thats largely what you already own."
        )
        if pairs:
            para3 += (
                " With ETF-plus-single overlap in play, effective tech sensitivity can exceed what three separate tickers naively imply — gradual sleeves into Treasuries, "
                "investment-grade bonds, dividend-led equities, and defensive sectors are the usual pressure-release valves while keeping intentional growth if you still want it."
            )
        else:
            para3 += " Match any de-risk to your timeline rather than pretending anyone clocks macro perfectly."
    elif risk_s >= 65 and mom_s >= 65:
        para3 = (
            "Taken together, supportive tape and chunky single-name weight often coexist — the uncomfortable part is drawdown risk if both fade. "
            "Many investors respond by keeping core conviction while diversifying gradually or adding explicit caps instead of dramatic all-or-nothing moves."
        )
    elif risk_s >= 65:
        para3 = (
            "The structural concentration story is louder than the tape cheerleading here — focus on whether your largest positions match the volatility you can tolerate."
        )
    elif mom_s >= 65:
        para3 = (
            "Momentum still looks constructive, but oversized weights turn decent charts into scary portfolios when trends crack — align trend trades with sleep-at-night sizing."
        )
    else:
        para3 = (
            "Signals are mixed rather than decisive — use this as a checklist against your own targets rather than a verdict from the outside."
        )

    disclaimer = conv_tools.toolkit_disclaimer_block()
    chunks = [para1]
    if overlap_para:
        chunks.append(overlap_para)
    chunks.extend([para2, para3, disclaimer])
    return "\n\n".join(chunks)


def _use_llm_collab(llm: Any) -> bool:
    return llm is not None and isinstance(llm, LLMClient)


def _orchestration_plan(*, rounds: int, llm: LLMClient | None) -> list[str]:
    """Which supervisor stages will run — mirros the branches below (logs / API hints)."""
    r = max(1, int(rounds))
    if _use_llm_collab(llm) and r >= 3:
        return ["portfolio", "market", "risk", "momentum", "synthesis"]
    if _use_llm_collab(llm) and r == 2:
        return ["portfolio", "market"]
    if _use_llm_collab(llm):
        return ["portfolio"]
    if r >= 3:
        return ["portfolio", "market", "conversation"]
    if r == 2:
        return ["portfolio", "market"]
    return ["portfolio"]


def _portfolio_bundle_for_llm(tr: dict[str, Any]) -> dict[str, Any]:
    pos = tr.get("list_positions") or []
    slim: list[dict[str, Any]] = []
    for p in pos[:24]:
        if isinstance(p, dict):
            slim.append(
                {
                    "ticker": p.get("ticker"),
                    "quantity": p.get("quantity"),
                    "avg_cost": p.get("avg_cost"),
                    "currency": p.get("currency"),
                }
            )
    return {
        "positions": slim,
        "summarize_allocation": tr.get("summarize_allocation"),
        "flag_concentration": tr.get("flag_concentration"),
        "snapshot_risk_profile": tr.get("snapshot_risk_profile"),
        "match_tickers_to_positions": tr.get("match_tickers_to_positions"),
        "factor_overlap_stub": tr.get("factor_overlap_stub"),
    }


def _market_bundle_for_llm(tr: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "get_quote",
        "get_recent_returns",
        "sector_benchmark_stub",
        "symbol_resolve",
        "market_hours_stub",
    )
    return {k: tr.get(k) for k in keys}


def _shared_facts_for_llm(p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, Any]:
    tr1 = p1.get("tool_results") or {}
    tr2 = p2.get("tool_results") or {}
    return {
        "portfolio_facts": _portfolio_facts(tr1),
        "market_facts": _market_facts(tr2),
    }


def _llm_tension_note(risk_o: dict[str, Any], mom_o: dict[str, Any]) -> dict[str, Any]:
    rr = int(risk_o.get("confidence_0_100") or 50)
    mm = int(mom_o.get("confidence_0_100") or 50)
    tension = "low"
    if rr >= 62 and mm >= 62:
        tension = "high"
    elif rr >= 62 or mm >= 62:
        tension = "moderate"
    return {"tension": tension, "risk_score_proxy": rr, "momentum_score_proxy": mm}


def _discussion_for_response(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Client-safe rows: captions only — tool names and raw traces stay on disk via SessionState."""
    rows: list[dict[str, Any]] = []
    for e in entries:
        row: dict[str, Any] = {
            "round": e.get("round"),
            "agent": e.get("agent"),
            "utterance": e.get("utterance"),
        }
        if e.get("collaboration_meta") is not None:
            row["collaboration_meta"] = e["collaboration_meta"]
        rows.append(row)
    return rows


def _format_collaborative_report(light: list[dict[str, Any]]) -> str:
    lines = ["Collaborative summary (user-facing captions)", ""]
    for row in light:
        ag = str(row.get("agent") or "?").upper()
        lines.append(f"Round {row.get('round')} — {ag}")
        lines.append("-" * 44)
        lines.append((row.get("utterance") or "").strip())
        lines.append("")
    lines.append(conv_tools.toolkit_disclaimer_block())
    return "\n".join(lines).strip()


async def run_collaborative_supervisor(
    *,
    query: str,
    session_id: str,
    user_context: dict[str, Any],
    conversation_history: list[dict[str, str]],
    state: SessionState,
    classification: dict[str, Any],
    rounds: int = 3,
    llm: LLMClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Mutates ``state.discussion_log``; caller should ``flush_session`` after.

    When ``llm`` is an :class:`LLMClient` and ``rounds >= 3``, runs five seperate
    model passes (portfolio, market, risk, momentum) plus a streaming chair —
    each with its own system prompt and JSON context bundle.
    """
    meta_hint: dict[str, Any] = {
        "session_id": session_id,
        "classification_agent": classification.get("agent"),
        "classification_intent": classification.get("intent"),
        "sparse_profile": classification.get("sparse_profile"),
        "collaboration_rounds_effective": classification.get("collaboration_rounds_effective"),
        "orchestration_plan": _orchestration_plan(rounds=rounds, llm=llm),
    }
    orch_prof = classification.get("orchestration_profile")
    if orch_prof == "tax_market_synthesis":
        meta_hint["orchestration_plan"] = ["portfolio", "tax_math", "market", "synthesis"]
        meta_hint["orchestration_profile"] = orch_prof
    if _use_llm_collab(llm):
        meta_hint["multiagent_llm"] = True
    state.orchestrator_meta = meta_hint
    state.updated_at = time.time()

    yield {"type": "meta", "stage": "collaborative_start", **meta_hint}

    def _preview(entry: dict[str, Any]) -> str:
        lm = entry.get("llm")
        if isinstance(lm, dict) and lm.get("stance_headline"):
            return str(lm["stance_headline"])[:280]
        return (entry.get("utterance") or "")[:280]

    def _stub_agent(answer: str) -> dict[str, Any]:
        return {
            "reasoning_trace": ["(fallback)"],
            "confidence_0_100": 50,
            "stance_headline": answer[:120],
            "answer": answer,
            "abstain": False,
            "abstain_reason": "",
        }

    entries: list[dict[str, Any]] = []

    if classification.get("orchestration_profile") == "tax_market_synthesis":
        p1 = _portfolio_round(user_context)
        entries.append(p1)
        state.discussion_log.append(p1)
        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p1["round"],
            "agent": p1["agent"],
            "utterance_preview": _preview(p1),
        }

        p_tax = _tax_math_round(user_context, query, classification, p1, round_no=2)
        entries.append(p_tax)
        state.discussion_log.append(p_tax)
        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p_tax["round"],
            "agent": p_tax["agent"],
            "utterance_preview": _preview(p_tax),
        }

        p_mkt = _market_round(user_context, query, p1, round_no=3)
        entries.append(p_mkt)
        state.discussion_log.append(p_mkt)
        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p_mkt["round"],
            "agent": p_mkt["agent"],
            "utterance_preview": _preview(p_mkt),
        }

        p_syn = _tax_market_synthesis_round(query, p1, p_tax, p_mkt, round_no=4)
        entries.append(p_syn)
        state.discussion_log.append(p_syn)
        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p_syn["round"],
            "agent": p_syn["agent"],
            "utterance_preview": _preview(p_syn),
        }

        final_text = _compose_tax_market_user_message(entries, query)
        yield {"type": "data", "delta": final_text}

        light = _discussion_for_response(entries)
        tax_bundle = (p_tax.get("tool_results") or {}).get("tax_gain_bundle") or {}
        structured = {
            "agent": "multiagent_orchestrator",
            "implemented": True,
            "intent": classification.get("intent"),
            "entities": classification.get("entities") or {},
            "message": final_text,
            "readable_report": _format_collaborative_report(light),
            "discussion_rounds": len(entries),
            "discussion_log": light,
            "prior_user_queries": conv_tools.toolkit_prior_user_turns(conversation_history)[:-1],
            "collaboration_meta": {"mode": "tax_market_synthesis"},
            "collaboration_signals": _collaboration_signals(entries),
            "multiagent_llm": False,
            "computed_gain_estimate": tax_bundle,
        }
        if len(state.discussion_log) > 32:
            state.discussion_log[:] = state.discussion_log[-32:]
        yield {"type": "structured", "payload": structured}
        return

    # ----- Full LLM panel -----
    if _use_llm_collab(llm) and rounds >= 3:
        assert llm is not None
        p1 = _portfolio_round(user_context)
        entries.append(p1)
        state.discussion_log.append(p1)
        try:
            po = await collab_llm.run_portfolio_llm(
                llm,
                query=query,
                portfolio_tool_bundle=_portfolio_bundle_for_llm(p1["tool_results"] or {}),
            )
            _apply_llm_public_layer(p1, po)
        except LLMError as exc:
            logger.warning("Collaborative portfolio LLM degraded: %s", exc)

        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p1["round"],
            "agent": p1["agent"],
            "utterance_preview": _preview(p1),
            "reasoning_agent": "portfolio",
        }

        p2 = _market_round(user_context, query, p1)
        entries.append(p2)
        state.discussion_log.append(p2)
        try:
            mo = await collab_llm.run_market_llm(
                llm,
                query=query,
                portfolio_facts=_portfolio_facts(p1.get("tool_results") or {}),
                market_tool_bundle=_market_bundle_for_llm(p2["tool_results"] or {}),
            )
            _apply_llm_public_layer(p2, mo)
        except LLMError as exc:
            logger.warning("Collaborative market LLM degraded: %s", exc)

        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p2["round"],
            "agent": p2["agent"],
            "utterance_preview": _preview(p2),
            "reasoning_agent": "market",
        }

        shared = _shared_facts_for_llm(p1, p2)
        portfolio_lm = p1.get("llm") if isinstance(p1.get("llm"), dict) else {}
        market_lm = p2.get("llm") if isinstance(p2.get("llm"), dict) else {}
        portfolio_pack = portfolio_lm or _stub_agent(p1.get("utterance") or "")
        market_pack = market_lm or _stub_agent(p2.get("utterance") or "")

        risk_entry: dict[str, Any] = {"agent": "risk", "round": 3, "tool_results": {}, "utterance": ""}
        entries.append(risk_entry)
        state.discussion_log.append(risk_entry)
        risk_lm: dict[str, Any] = {}
        try:
            risk_lm = await collab_llm.run_risk_llm(
                llm,
                query=query,
                shared_facts=shared,
                portfolio_analyst=portfolio_pack,
                market_analyst=market_pack,
            )
            _apply_llm_public_layer(risk_entry, risk_lm)
        except LLMError as exc:
            logger.warning("Collaborative risk LLM degraded: %s", exc)
            risk_entry["utterance"] = "Risk specialist pass unavailable — continuing with holdings and tape facts only."

        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": risk_entry["round"],
            "agent": risk_entry["agent"],
            "utterance_preview": _preview(risk_entry),
            "reasoning_agent": "risk",
        }

        mom_entry: dict[str, Any] = {"agent": "momentum", "round": 4, "tool_results": {}, "utterance": ""}
        entries.append(mom_entry)
        state.discussion_log.append(mom_entry)
        mom_lm: dict[str, Any] = {}
        try:
            mom_lm = await collab_llm.run_momentum_llm(
                llm,
                query=query,
                shared_facts=shared,
                portfolio_analyst=portfolio_pack,
                market_analyst=market_pack,
            )
            _apply_llm_public_layer(mom_entry, mom_lm)
        except LLMError as exc:
            logger.warning("Collaborative momentum LLM degraded: %s", exc)
            mom_entry["utterance"] = "Momentum specialist pass unavailable — treat tape comments conservatively."

        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": mom_entry["round"],
            "agent": mom_entry["agent"],
            "utterance_preview": _preview(mom_entry),
            "reasoning_agent": "momentum",
        }

        syn_entry: dict[str, Any] = {"agent": "synthesis", "round": 5, "tool_results": {}, "utterance": ""}
        entries.append(syn_entry)

        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": syn_entry["round"],
            "agent": syn_entry["agent"],
            "utterance_preview": "Chair synthesizing panel outputs…",
            "reasoning_agent": "synthesis",
        }

        buf: list[str] = []
        chair_ok = True
        risk_pack = risk_lm or _stub_agent(risk_entry.get("utterance") or "")
        mom_pack = mom_lm or _stub_agent(mom_entry.get("utterance") or "")
        try:
            async for piece in collab_llm.stream_chair_answer(
                llm,
                query=query,
                portfolio_analyst=portfolio_pack,
                market_analyst=market_pack,
                risk_analyst=risk_pack,
                momentum_analyst=mom_pack,
            ):
                buf.append(piece)
                yield {"type": "data", "delta": piece}
        except LLMError as exc:
            chair_ok = False
            logger.warning("Collaborative chair stream degraded: %s", exc)

        disclaimer = conv_tools.toolkit_disclaimer_block()
        if chair_ok and buf:
            final_body = "".join(buf).strip()
            final_text = (
                final_body
                if disclaimer.lower() in final_body.lower()
                else final_body + "\n\n" + disclaimer
            )
        else:
            final_text = _compose_final_user_message([p1, p2], query)

        syn_entry["utterance"] = final_text
        state.discussion_log.append(syn_entry)

        tension = _llm_tension_note(risk_lm or {}, mom_lm or {})
        collab_meta = {
            "mode": "llm_multiagent",
            "integration": "streamed_chair",
            **tension,
        }
        syn_entry["collaboration_meta"] = collab_meta

        light = _discussion_for_response(entries)
        structured: dict[str, Any] = {
            "agent": "multiagent_orchestrator",
            "implemented": True,
            "intent": classification.get("intent"),
            "entities": classification.get("entities") or {},
            "message": final_text,
            "readable_report": _format_collaborative_report(light),
            "discussion_rounds": len(entries),
            "discussion_log": light,
            "prior_user_queries": conv_tools.toolkit_prior_user_turns(conversation_history)[:-1],
            "collaboration_meta": collab_meta,
            "collaboration_signals": _collaboration_signals(entries),
            "multiagent_llm": True,
        }
        if len(state.discussion_log) > 32:
            state.discussion_log[:] = state.discussion_log[-32:]
        yield {"type": "structured", "payload": structured}
        return

    # ----- Partial LLM (tools + 1–2 analyst passes, deterministic synthesis) -----
    if _use_llm_collab(llm) and rounds in (1, 2):
        assert llm is not None
        p1 = _portfolio_round(user_context)
        entries.append(p1)
        state.discussion_log.append(p1)
        try:
            po = await collab_llm.run_portfolio_llm(
                llm,
                query=query,
                portfolio_tool_bundle=_portfolio_bundle_for_llm(p1["tool_results"] or {}),
            )
            _apply_llm_public_layer(p1, po)
        except LLMError as exc:
            logger.warning("Collaborative portfolio LLM degraded: %s", exc)

        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p1["round"],
            "agent": p1["agent"],
            "utterance_preview": _preview(p1),
            "reasoning_agent": "portfolio",
        }

        if rounds >= 2:
            p2 = _market_round(user_context, query, p1)
            entries.append(p2)
            state.discussion_log.append(p2)
            try:
                mo = await collab_llm.run_market_llm(
                    llm,
                    query=query,
                    portfolio_facts=_portfolio_facts(p1.get("tool_results") or {}),
                    market_tool_bundle=_market_bundle_for_llm(p2["tool_results"] or {}),
                )
                _apply_llm_public_layer(p2, mo)
            except LLMError as exc:
                logger.warning("Collaborative market LLM degraded: %s", exc)

            yield {
                "type": "meta",
                "stage": "agent_discussion",
                "round": p2["round"],
                "agent": p2["agent"],
                "utterance_preview": _preview(p2),
                "reasoning_agent": "market",
            }

        final_text = _compose_final_user_message(entries, query)
        yield {"type": "data", "delta": final_text}

        light = _discussion_for_response(entries)
        structured = {
            "agent": "multiagent_orchestrator",
            "implemented": True,
            "intent": classification.get("intent"),
            "entities": classification.get("entities") or {},
            "message": final_text,
            "readable_report": _format_collaborative_report(light),
            "discussion_rounds": len(entries),
            "discussion_log": light,
            "prior_user_queries": conv_tools.toolkit_prior_user_turns(conversation_history)[:-1],
            "collaboration_meta": {"mode": "llm_partial", "rounds_requested": rounds},
            "collaboration_signals": _collaboration_signals(entries),
            "multiagent_llm": True,
        }
        if len(state.discussion_log) > 32:
            state.discussion_log[:] = state.discussion_log[-32:]
        yield {"type": "structured", "payload": structured}
        return

    # ----- Deterministic template panel (no LLM client — offline tests) -----
    p1 = _portfolio_round(user_context)
    entries.append(p1)
    state.discussion_log.append(p1)
    yield {
        "type": "meta",
        "stage": "agent_discussion",
        "round": p1["round"],
        "agent": p1["agent"],
        "utterance_preview": _preview(p1),
    }

    if rounds >= 2:
        p2 = _market_round(user_context, query, p1)
        entries.append(p2)
        state.discussion_log.append(p2)
        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p2["round"],
            "agent": p2["agent"],
            "utterance_preview": _preview(p2),
        }

    if rounds >= 3:
        last_intent = classification.get("agent")
        market_peer = entries[-1]
        p3 = _conversation_round(conversation_history, query, last_intent, p1, market_peer)
        entries.append(p3)
        state.discussion_log.append(p3)
        yield {
            "type": "meta",
            "stage": "agent_discussion",
            "round": p3["round"],
            "agent": p3["agent"],
            "utterance_preview": _preview(p3),
        }

    final_text = _compose_final_user_message(entries, query)
    yield {"type": "data", "delta": final_text}

    light = _discussion_for_response(entries)
    collab_meta = entries[-1].get("collaboration_meta") if entries else None
    structured = {
        "agent": "multiagent_orchestrator",
        "implemented": True,
        "intent": classification.get("intent"),
        "entities": classification.get("entities") or {},
        "message": final_text,
        "readable_report": _format_collaborative_report(light),
        "discussion_rounds": len(entries),
        "discussion_log": light,
        "prior_user_queries": conv_tools.toolkit_prior_user_turns(conversation_history)[:-1],
        "collaboration_meta": collab_meta,
        "collaboration_signals": _collaboration_signals(entries),
        "multiagent_llm": False,
    }
    if len(state.discussion_log) > 32:
        state.discussion_log[:] = state.discussion_log[-32:]
    yield {"type": "structured", "payload": structured}
