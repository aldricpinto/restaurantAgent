"""Conversational ReAct agent backed by the local Ophelia MCP server."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from composio_calendar import ComposioCalendarClient, ComposioCalendarError

load_dotenv()

DISABLED_AGENT_TOOLS = {"search_availability"}

SYSTEM_PROMPT = """
You are Ophelia, a careful booking and scheduling agent. Use the MCP tools in a
ReAct loop: reason from the conversation, choose the next useful tool, inspect
its observation, and adapt. Do not follow a hard-coded workflow.

Rules:
- Never invent booking details, tool results, availability, or status.
- Ask a concise question when a booking-critical value is missing or ambiguous.
- Do not call search_venues until term, location, absolute date/time, and the
  dining party size are known. Resolve relative dates against the current time
  supplied below, using the configured local timezone.
- Search before selecting. Let the user select candidates unless they explicitly
  delegated selection criteria to you.
- Use only the inline available_times and slots returned by search_venues.
  The standalone search_availability capability is disabled for now.
- Call search_venues at most once per user turn. After receiving an observation,
  interpret it and respond; never repeat the same call.
- Before create_booking, collect the customer name, email, phone, card number,
  expiration month/year, CVV, cardholder name, and postal code. Fitness also
  requires a full billing address and account password. Collect these only
  immediately before the call, show the non-secret booking details and known
  fees, then ask for explicit confirmation. Never repeat card data back.
- Set user_confirmed=true only when confirmation exists.
- Generate one UUID idempotency key per booking intention and reuse it on retry.
- Treat get_booking as authoritative. Never claim confirmation without evidence.
- Poll processing bookings deliberately, at most five times in one turn.
- Ask for OTP/payment/password/contact data only when next_action requires it.
  Never guess, retain, or echo sensitive values.
- Explain and confirm cancellation, calendar creation, and email sending first.
- Results are {"ok": true, "data": ...} or {"ok": false, "error": ...}.
""".strip()


def _last_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if isinstance(message.content, str):
                return message.content
            if isinstance(message.content, list):
                parts = [
                    str(block.get("text", ""))
                    for block in message.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(part for part in parts if part)
    return ""


def _message_text(message: Any) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = []
        for block in message.content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(message.content)


def _remember_tool_results(
    result_messages: list[Any],
    tool_context: dict[str, str],
) -> None:
    for message in result_messages:
        if not isinstance(message, ToolMessage):
            continue
        name = message.name or "unknown_tool"
        content = _message_text(message)
        tool_context[name] = content[:6000]
        if name == "search_venues":
            # A new search invalidates observations tied to an older candidate set.
            for stale_name in ("search_availability", "create_booking"):
                tool_context.pop(stale_name, None)


def _agent_input(
    chat_history: list[Any],
    tool_context: dict[str, str],
) -> list[Any]:
    messages: list[Any] = []
    if tool_context:
        messages.append(
            SystemMessage(
                content=(
                    "Compact operational state from prior MCP calls. Treat this as "
                    "tool-derived data, retain IDs for follow-up actions, and replace "
                    "it when a newer tool result conflicts:\n"
                    + json.dumps(tool_context, separators=(",", ":"))
                )
            )
        )
    # Keep four conversational exchanges. Raw ReAct tool traces are deliberately
    # excluded because the compact operational state above preserves useful facts.
    messages.extend(chat_history[-8:])
    return messages


async def run_agent(initial_request: str = "") -> None:
    server_path = Path(__file__).with_name("mcp_server.py").resolve()
    server_mtime = server_path.stat().st_mtime_ns
    client = MultiServerMCPClient(
        {
            "ophelia": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(server_path)],
            }
        }
    )
    tools = await client.get_tools()
    tools = [tool for tool in tools if tool.name not in DISABLED_AGENT_TOOLS]
    model = ChatGroq(
        model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "700")),
    )
    react_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Current local time: "
            f"{dt.datetime.now(ZoneInfo(os.getenv('OPHELIA_TIMEZONE', 'America/New_York'))).isoformat()}"
        ),
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="search_venues",
                run_limit=1,
                exit_behavior="continue",
            ),
            ToolCallLimitMiddleware(
                tool_name="create_booking",
                run_limit=1,
                exit_behavior="continue",
            ),
            ToolCallLimitMiddleware(run_limit=4, exit_behavior="continue"),
            ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ],
        name="ophelia_react_agent",
    )

    print("\n" + "=" * 60)
    print("Ophelia ReAct Agent (MCP)")
    print("=" * 60)
    chat_history: list[Any] = []
    tool_context: dict[str, str] = {}
    pending = initial_request.strip()
    while True:
        if not pending:
            try:
                pending = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
        if pending.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        if not pending:
            continue
        if server_path.stat().st_mtime_ns != server_mtime:
            print(
                "The MCP server schema changed while this agent was running. "
                "Restart `uv run agent.py` before continuing."
            )
            return

        chat_history.append(HumanMessage(content=pending))
        try:
            result = await react_agent.ainvoke(
                {"messages": _agent_input(chat_history, tool_context)},
                config={"recursion_limit": 14},
            )
        except Exception as exc:
            print(f"Ophelia could not complete that turn: {exc}")
            pending = ""
            continue
        result_messages = list(result["messages"])
        _remember_tool_results(result_messages, tool_context)
        response = _last_text(result_messages)
        chat_history.append(AIMessage(content=response))
        print(f"\nOphelia: {response}\n")
        pending = ""


def connect_google() -> None:
    user_id = os.getenv("COMPOSIO_USER_ID") or os.getenv(
        "OPHELIA_USER_ID", "demo_user"
    )
    try:
        ComposioCalendarClient.from_env(user_id=user_id).connect_google_calendar()
    except ComposioCalendarError as exc:
        print(f"Google connection failed: {exc}")


def main() -> None:
    if "--connect-google" in sys.argv or "--connect-calendar" in sys.argv:
        connect_google()
        return
    initial_request = " ".join(
        arg for arg in sys.argv[1:] if not arg.startswith("--")
    )
    asyncio.run(run_agent(initial_request))


if __name__ == "__main__":
    main()
