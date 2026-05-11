"""
SSE serialization.

We do not use sse-starlette directly here because we want full control
over event names and payload shape, and the protocol itself is tiny.
The pipeline yields normalized event dicts; this module turns them into
the wire-format frames a browser EventSource consumer expects.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator


def _encode(
    event_name: str,
    payload: dict[str, Any],
    *,
    indent: int | None = None,
) -> str:
    """One SSE message: event line + data line(s) + blank line.

    Token deltas stay one `data:` row (cheap for the UI). Heavier payloads
    use pretty JSON so curl / logs dont look like one giant escaped blob —
    spec allows multiple `data:` lines; clients join them with newlines.
    """
    body = json.dumps(payload, ensure_ascii=False, indent=indent)
    if indent is None:
        return f"event: {event_name}\ndata: {body}\n\n"
    data_lines = "\n".join(f"data: {line}" for line in body.splitlines())
    return f"event: {event_name}\n{data_lines}\n\n"


def event_to_sse(event: dict[str, Any]) -> str:
    """Map a pipeline event dict to an SSE frame."""
    etype = event.get("type", "message")
    if etype == "data":
        # narrative streaming — keep payload minimal so the browser can
        # concatenate `delta` cheaply
        return _encode("token", {"delta": event.get("delta", "")}, indent=None)
    if etype == "structured":
        return _encode("structured", event.get("payload", {}), indent=2)
    if etype == "meta":
        return _encode("meta", {k: v for k, v in event.items() if k != "type"}, indent=2)
    if etype == "error":
        return _encode("error", {k: v for k, v in event.items() if k != "type"}, indent=2)
    if etype == "done":
        return _encode("done", {k: v for k, v in event.items() if k != "type"}, indent=2)
    return _encode(etype, event, indent=2)


async def stream_events(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield event_to_sse(event)
