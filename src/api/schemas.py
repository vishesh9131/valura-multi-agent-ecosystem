"""Request/response schemas for the HTTP layer."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Position(BaseModel):
    ticker: str
    exchange: str | None = None
    quantity: float
    # Optional for quick API demos; agents treat 0 as unknown cost basis.
    avg_cost: float = 0.0
    currency: str = "USD"
    purchased_at: str | None = None


class UserContext(BaseModel):
    user_id: str
    name: str | None = None
    age: int | None = None
    country: str | None = None
    base_currency: str = "USD"
    risk_profile: str | None = None
    kyc: dict[str, Any] | None = None
    positions: list[Position] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str
    session_id: str
    user_context: UserContext | None = None
    # Assignment 2: portfolio + market + conversation supervisor with MCP-backed tools.
    collaborative: bool = False
