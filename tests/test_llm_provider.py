"""
LLM provider switch.

We don't make a real network call here. We just assert that:
  - settings honour the env var
  - the OpenAICompatClient picks the right base_url + model when LLM_PROVIDER=vllm
  - the OpenAI path raises a clear LLMError if no key is set (as opposed
    to silently using SDK defaults which may try to find a key on disk)
"""
from __future__ import annotations

import pytest

from src.config import Settings, reset_settings_cache
from src.llm import LLMError, reset_llm_client


def test_settings_default_provider_is_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reset_settings_cache()
    s = Settings()
    assert s.llm_provider == "openai"


def test_settings_vllm_defaults_point_at_corerec(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    reset_settings_cache()
    s = Settings()
    assert s.llm_provider == "vllm"
    assert s.vllm_base_url == "https://vllm.corerec.online/v1"
    assert s.active_model == s.vllm_model


def test_openai_provider_without_key_raises_clearly():
    # pass a Settings object directly so pydantic-settings never touches .env
    # (monkeypatching env alone won't work when .env contains the key)
    s = Settings(llm_provider="openai", openai_api_key=None)
    from src.llm import OpenAICompatClient
    with pytest.raises(LLMError):
        OpenAICompatClient(settings=s)


def test_vllm_provider_uses_vllm_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "https://vllm.corerec.online/v1")
    monkeypatch.setenv("VLLM_MODEL", "test-model")
    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    reset_settings_cache()
    reset_llm_client()

    from src.llm import OpenAICompatClient
    c = OpenAICompatClient()
    assert c.provider == "vllm"
    assert c.model == "test-model"
    # underlying SDK client should know the base url
    assert "vllm.corerec.online" in str(getattr(c._client, "base_url", ""))
