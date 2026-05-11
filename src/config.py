"""
Settings.

One source of truth for env-driven config. Read once, share everywhere.
We use pydantic-settings so that the same object validates env vars in prod
and accepts overrides in tests via constructor kwargs.

Two LLM providers are supported and they are swapped purely via env:

    LLM_PROVIDER=openai      -> hits api.openai.com
    LLM_PROVIDER=vllm        -> hits a self-hosted VLLM endpoint

VLLM exposes the OpenAI Chat Completions wire format, so we reuse the
official `openai` SDK and just override `base_url` + the model name. No
second client library to maintain.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


LLMProvider = Literal["openai", "vllm"]

# Resolve `.env` next to this package (`Assignment_2/.env`), not from OS cwd.
# Otherwise `uvicorn` started from another directory never sees your key, and
# an empty `OPENAI_API_KEY` in the shell overrides the file unless we load explicitly.
_ASSIGNMENT2_ROOT = Path(__file__).resolve().parent.parent
_DOTENV = _ASSIGNMENT2_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_DOTENV),
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "production", "test"] = "development"
    request_timeout_s: float = 30.0
    # ^ end-to-end pipeline timeout. 30s is generous for a chat agent;
    # p95 target is 6s so anything north of that is already a tail.

    # --- LLM provider switch ---
    llm_provider: LLMProvider = "openai"

    # --- OpenAI ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None  # honoured if set, else SDK default

    # --- VLLM (OpenAI-compatible) ---
    # Defaults point at the cloudflare-tunnel-fronted VLLM the team runs.
    vllm_base_url: str = "https://vllm.corerec.online/v1"
    vllm_model: str = "default"
    # Many self-hosted VLLM deployments dont check the key, but the SDK
    # still wants something non-empty. Keep a placeholder if unset.
    vllm_api_key: str = "EMPTY"

    # --- Multi-agent (Assignment 2): collaborative SSE path ---
    multiagent_rounds: int = Field(default=3, ge=1, le=5, validation_alias="MULTIAGENT_ROUNDS")
    # When true, POST body can omit collaborative=true and still run the supervisor (demo flag).
    multiagent_enabled: bool = Field(default=False, validation_alias="MULTIAGENT_ENABLED")

    # --- Sessions ---
    session_ttl_s: int = 60 * 60 * 6  # six hours of memory is plenty for a chat session
    # Persist chat turns + carryover under this directory (JSON per session).
    # Ignored when APP_ENV=test. Set empty to disable disk persistence (RAM only).
    session_memory_dir: str = Field(default="Logs/memory", validation_alias="SESSION_MEMORY_DIR")

    # --- Market data ---
    market_data_cache_ttl_s: int = 60 * 15
    # Prices in a quarter hour bucket are accurate enough for a health check
    # and stop yfinance from rate-limiting us in dev.

    # --- Web fetch (product recommendation + MCP web_server) ---
    web_fetch_enabled: bool = Field(default=True, validation_alias="WEB_FETCH_ENABLED")
    web_fetch_timeout_s: float = Field(default=12.0, validation_alias="WEB_FETCH_TIMEOUT_S")
    web_fetch_connect_timeout_s: float = Field(default=15.0, validation_alias="WEB_FETCH_CONNECT_TIMEOUT_S")
    web_fetch_max_bytes: int = Field(default=524_288, ge=4096, validation_alias="WEB_FETCH_MAX_BYTES")
    web_fetch_max_chars: int = Field(default=12_000, ge=500, validation_alias="WEB_FETCH_MAX_CHARS")
    web_search_max_results: int = Field(default=5, ge=1, le=10, validation_alias="WEB_SEARCH_MAX_RESULTS")
    web_fetch_user_agent: str = Field(default="", validation_alias="WEB_FETCH_USER_AGENT")
    # Comma-separated host suffixes; empty allowlist = any host that passes DNS global check.
    web_fetch_host_allowlist: str = Field(default="", validation_alias="WEB_FETCH_HOST_ALLOWLIST")
    web_fetch_host_denylist: str = Field(default="", validation_alias="WEB_FETCH_HOST_DENYLIST")

    @property
    def active_model(self) -> str:
        return self.vllm_model if self.llm_provider == "vllm" else self.openai_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    # tests want a fresh Settings after monkeypatching env
    get_settings.cache_clear()
