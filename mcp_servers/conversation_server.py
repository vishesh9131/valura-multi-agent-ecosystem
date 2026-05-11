"""Conversation / recap MCP server — stdio. Run: ``PYTHONPATH=. python -m mcp_servers.conversation_server``."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.orchestration.toolkits import conversation_tools as ct

app = FastMCP("valura_conversation_mcp")


@app.tool()
def prior_user_turns(messages_json: str) -> str:
    """Pass chat messages as JSON list of {role, content}."""
    try:
        hist = json.loads(messages_json)
        if not isinstance(hist, list):
            return json.dumps([])
        norm = [x for x in hist if isinstance(x, dict) and "role" in x and "content" in x]
        return json.dumps(ct.toolkit_prior_user_turns(norm))  # type: ignore[arg-type]
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid_json"})


@app.tool()
def format_numbered_list(lines_json: str) -> str:
    """JSON array of strings -> numbered multi-line text."""
    try:
        lines = json.loads(lines_json)
        if not isinstance(lines, list):
            return ""
        str_lines = [str(x) for x in lines]
        return ct.toolkit_format_numbered_list(str_lines)
    except json.JSONDecodeError:
        return ""


@app.tool()
def thread_summary_stub(messages_json: str, max_lines: int = 6) -> str:
    """Short textual recap of the tail of the thread."""
    try:
        hist = json.loads(messages_json)
        if not isinstance(hist, list):
            return ct.toolkit_thread_summary_stub([])
        norm = [x for x in hist if isinstance(x, dict)]
        return ct.toolkit_thread_summary_stub(norm, max_lines=max_lines)  # type: ignore[arg-type]
    except json.JSONDecodeError:
        return ct.toolkit_thread_summary_stub([])


@app.tool()
def disclaimer_block() -> str:
    """Standard educational disclaimer string."""
    return ct.toolkit_disclaimer_block()


@app.tool()
def suggest_handoff_intent(query: str, last_intent: str | None = None) -> str:
    """Lightweight keyword router suggestion for orchestrators."""
    return json.dumps(ct.toolkit_suggest_handoff_intent(query, last_intent))


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
