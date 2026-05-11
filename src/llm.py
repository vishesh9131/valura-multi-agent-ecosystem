"""
LLM client + provider factory.

Why one file: the surface we actually need is small (one structured call,
one streaming call). Splitting this into 4 files would be ceremony.

Why the openai SDK for both providers: VLLM exposes the OpenAI Chat
Completions API by design, so we point the same client at a different
`base_url` and we are done. One dependency, one set of error types, one
retry policy.

The classifier wants strict JSON; the agent wants a free-form stream.
We expose both shapes here and keep the `LLMClient` Protocol thin so
tests can hand in a fake without dragging in the real SDK.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterable, Protocol, runtime_checkable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import Settings, get_settings


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the upstream LLM call cannot be completed.

    Callers must treat this as recoverable: degrade gracefully (fallback
    classification, error SSE event), do NOT crash the request.
    """


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM surface. Implementations: OpenAI/VLLM, FakeLLM in tests."""

    provider: str
    model: str

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        ...

    async def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> AsyncIterator[str]:
        ...


# ---------------------------------------------------------------------------
# OpenAI / VLLM concrete client
# ---------------------------------------------------------------------------

class OpenAICompatClient:
    """Drives both real OpenAI and a VLLM endpoint that speaks the same wire format.

    For VLLM we ask for `response_format={"type": "json_object"}` because most
    served models support that but not full JSON-schema. For OpenAI proper we
    try the schema-enforced path first and fall back if the model rejects it.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # imported lazily so tests don't need the openai package wired in
        from openai import OpenAI

        self._settings = settings or get_settings()
        self.provider = self._settings.llm_provider

        if self.provider == "vllm":
            self.model = self._settings.vllm_model
            self._client = OpenAI(
                api_key=self._settings.vllm_api_key,
                base_url=self._settings.vllm_base_url,
            )
        else:
            if not self._settings.openai_api_key:
                raise LLMError(
                    "OPENAI_API_KEY missing. Set it, or switch LLM_PROVIDER=vllm."
                )
            self.model = self._settings.openai_model
            self._client = OpenAI(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
            )

    # -- structured ---------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.3, max=2),
        retry=retry_if_exception_type(Exception),
    )
    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        # try the strictest path the provider can take, then fall back.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.provider == "openai" and json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "schema": json_schema,
                    "strict": True,
                },
            }
        else:
            # vllm + plain openai both accept this everywhere
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # second pass without response_format if the model rejected it
            logger.warning("LLM call failed (%s) — retrying without response_format", exc)
            kwargs.pop("response_format", None)
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as exc2:
                raise LLMError(str(exc2)) from exc2

        content = (resp.choices[0].message.content or "").strip()
        return _safe_json_loads(content)

    # -- streaming ----------------------------------------------------------

    async def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> AsyncIterator[str]:
        # The openai SDK stream is a sync iterator; we wrap it as async
        # so the FastAPI handler can await without blocking the loop.
        # It is fine to do this here because the SDK uses httpx under the
        # hood, which releases the GIL on network reads.
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
            except (IndexError, AttributeError):
                continue
            piece = getattr(delta, "content", None)
            if piece:
                yield piece


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Parse model output; tolerate code-fenced JSON blocks."""
    if not text:
        raise LLMError("Empty LLM response")
    # strip ```json ... ``` fences if present (some vllm models still emit them)
    if text.startswith("```"):
        text = text.strip("`")
        # drop optional leading 'json' marker
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON from LLM: {exc.msg}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a process-wide LLM client honouring the current settings."""
    global _singleton
    if _singleton is None:
        _singleton = OpenAICompatClient()
    return _singleton


def reset_llm_client() -> None:
    """Drop the cached client. Used by tests after env changes."""
    global _singleton
    _singleton = None


# ---------------------------------------------------------------------------
# Helpers used by tests / the agent layer
# ---------------------------------------------------------------------------

def assemble_messages(
    *,
    system: str,
    history: Iterable[dict[str, str]] = (),
    user: str | None = None,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    msgs.extend(history)
    if user is not None:
        msgs.append({"role": "user", "content": user})
    return msgs
