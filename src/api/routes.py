"""HTTP routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import get_settings
from ..llm import LLMError, get_llm_client
from ..pipeline import process_query
from .schemas import ChatRequest
from .sse import stream_events


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz() -> JSONResponse:
    s = get_settings()
    return JSONResponse(
        {
            "status": "ok",
            "llm_provider": s.llm_provider,
            "model": s.active_model,
            "app_env": s.app_env,
        }
    )


@router.post("/v1/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """SSE stream of pipeline events.

    SSE is the only response mode. There is no JSON fallback because the
    assignment forbids one and because mixing two response shapes leads
    to clients that quietly skip streaming.
    """
    user_context = req.user_context.model_dump() if req.user_context else {}
    try:
        llm = get_llm_client()
    except LLMError:
        llm = None
    events = process_query(
        query=req.query,
        session_id=req.session_id,
        user_context=user_context,
        collaborative=req.collaborative,
        llm=llm,
    )
    return StreamingResponse(
        stream_events(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx hint, harmless elsewhere
        },
    )
