"""
Product recommendation agent.

Uses th same shared web toolkit as ``mcp_servers.web_server``. Answers are
educational only (no personalised buy/sell); snippets come from search/fetch.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from ..config import Settings, get_settings
from ..llm import LLMClient, LLMError, assemble_messages, get_llm_client
from ..orchestration.toolkits import web_tools as wt
from ..safety import MODEL_INJECTION_GUARD


logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is not investment advice. Web excerpts may be incomplete or outdated; "
    "always read official fund documents and consult a qualified advisor before investing."
)

SYSTEM = """You help novice investors think about funds and ETFs in an educational way.

Rules:
- Ground every factual claim in the WEB_CONTEXT block only — dont invent tickers, yields, or fees.
- If WEB_CONTEXT is thin or empty, say what you cant verify and stick to general checklist-style guidance.
- Respect EXCLUSIONS / values constraints when summarising what to look for (e.g. screens for tobacco, gambling, weapons).
- Short paragraphs, plain language. End by reminding the user to verify facts themselves.
- Do not tell them what they personally must buy or sell.

""" + MODEL_INJECTION_GUARD


def _build_search_queries(query: str, classification: dict[str, Any]) -> list[str]:
    entities = classification.get("entities") or {}
    topics = entities.get("topics") or []
    out: list[str] = []
    q = (query or "").strip()
    if q:
        out.append(q[:400])
    for t in topics[:3]:
        if isinstance(t, str) and t.strip():
            out.append(f"{t.strip()} ETF fund factsheet overview fees")
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq[:3]


def _merge_hits(search_batches: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_url: dict[str, dict[str, str]] = {}
    for batch in search_batches:
        for row in batch.get("results") or []:
            url = (row.get("url") or "").strip()
            if not url.startswith("https://"):
                continue
            if url not in by_url:
                by_url[url] = {
                    "title": (row.get("title") or url).strip(),
                    "url": url,
                    "snippet": (row.get("snippet") or "").strip(),
                }
    return list(by_url.values())


def _exclusion_blob(conversation_history: list[dict[str, str]] | None) -> str:
    """Pull recent user lines so model sees values constraints."""
    lines = [m.get("content", "").strip() for m in (conversation_history or []) if m.get("role") == "user"]
    tail = lines[-4:] if lines else []
    if not tail:
        return "(none stated)"
    return "\n".join(f"- {t}" for t in tail)


def _web_context_block(hits: list[dict[str, str]], page_texts: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits[:8], start=1):
        parts.append(f"[{i}] {h.get('title')}\nURL: {h.get('url')}\n{h.get('snippet')}")
    for url, txt in page_texts:
        parts.append(f"\n--- PAGE {url} ---\n{txt[:4000]}")
    return "\n\n".join(parts) if parts else "(no web context retrieved)"


def _deterministic_answer(hits: list[dict[str, str]], *, web_ok: bool) -> str:
    if not web_ok or not hits:
        return (
            "I couldnt retrieve live web excerpts just now (disabled or unreachable). "
            "For dividend-focused, values-aligned funds people usually compare provider screens, "
            "expense ratios, tracking difference, and holdings overlap — then read the statutory prospectus. "
            f"\n\n{DISCLAIMER}"
        )
    lines = ["Here are sources from a quick web scan — double-check before relying on them:\n"]
    for i, h in enumerate(hits[:6], start=1):
        sn = h.get("snippet") or "(no snippet)"
        lines.append(f"{i}. {h.get('title')}\n   {sn}\n   {h.get('url')}")
    lines.append(f"\n{DISCLAIMER}")
    return "\n".join(lines)


def _maybe_llm_client(llm: Any | None) -> LLMClient | None:
    """Use streaming client from caller, or env default only when ``llm`` was omitted."""
    if llm is not None:
        return llm if hasattr(llm, "stream_text") else None
    try:
        c = get_llm_client()
    except Exception:
        return None
    return c if hasattr(c, "stream_text") else None


def _split_stream(text: str, *, chunk_size: int = 40) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class ProductRecommendationAgent:
    name = "product_recommendation"

    async def run(
        self,
        *,
        query: str,
        user_context: dict[str, Any],
        classification: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        settings = get_settings()
        queries_used = _build_search_queries(query, classification)

        yield {
            "type": "meta",
            "stage": "product_web",
            "web_queries": queries_used,
            "web_fetch_enabled": settings.web_fetch_enabled,
        }

        batches: list[dict[str, Any]] = []
        hits: list[dict[str, str]] = []
        cap = max(1, min(int(settings.web_search_max_results), 10))
        for sq in queries_used:
            batches.append(await asyncio.to_thread(wt.toolkit_web_search, sq, settings=settings))
            hits = _merge_hits(batches)
            if len(hits) >= cap:
                break

        retrieval_ok = len(hits) > 0

        page_texts: list[tuple[str, str]] = []
        if settings.web_fetch_enabled and hits:
            for h in hits[:1]:
                url = h.get("url") or ""
                fetched = await asyncio.to_thread(wt.toolkit_fetch_url, url, settings=settings)
                if fetched.get("ok") and (fetched.get("text") or "").strip():
                    page_texts.append((str(fetched.get("url") or url), str(fetched.get("text") or "")))

        ctx = _web_context_block(hits, page_texts)
        exclusions = _exclusion_blob(conversation_history)

        client = _maybe_llm_client(llm)
        hist = list(conversation_history or [])
        if hist and hist[-1].get("role") == "user":
            hist = hist[:-1]

        user_blob = (
            f"USER_QUESTION:\n{query}\n\n"
            f"CLASSIFIER_INTENT:\n{classification.get('intent')}\n\n"
            f"RECENT_USER_LINES:\n{exclusions}\n\n"
            f"WEB_CONTEXT:\n{ctx}\n"
        )

        narrative = ""
        if client is None:
            narrative = _deterministic_answer(hits, web_ok=retrieval_ok)
            for chunk in _split_stream(narrative):
                yield {"type": "data", "delta": chunk}
        else:
            messages = assemble_messages(system=SYSTEM, history=hist, user=user_blob)
            try:
                async for piece in client.stream_text(messages, temperature=0.25, max_tokens=700):
                    narrative += piece
                    yield {"type": "data", "delta": piece}
            except LLMError as exc:
                logger.warning("Product recommendation LLM stream failed: %s", exc)
                narrative = _deterministic_answer(hits, web_ok=retrieval_ok)
                for chunk in _split_stream(narrative):
                    yield {"type": "data", "delta": chunk}

        sources = [{"title": h.get("title"), "url": h.get("url")} for h in hits[:12]]
        yield {
            "type": "structured",
            "payload": {
                "agent": self.name,
                "implemented": True,
                "intent": classification.get("intent"),
                "entities": classification.get("entities") or {},
                "message": narrative.strip(),
                "sources": sources,
                "web_queries_used": queries_used,
                "disclaimer": DISCLAIMER,
                "web_retrieval_ok": retrieval_ok,
            },
        }
