"""
The agent loop -- Chapter 3, "the loop that is the whole trick."

There is no secret to an agent. It is a conversation where the model may answer
with tool calls instead of text. We run each call, hand the result back, and ask
the model again -- plan -> act -> observe -- until it answers with plain text and
no tool calls. Everything impressive an agent does emerges from this loop plus a
capable model. This hand-rolls that loop against a LOCAL model, so it runs on
your laptop for free -- no API key, no cloud.

It talks to any OpenAI-compatible local server:
  * LM Studio  -> http://localhost:1234/v1   (default; model gpt-oss-20b)
  * Ollama     -> http://localhost:11434/v1  (set the two env vars below)
"""

import os
import json
from openai import OpenAI

from tools import TOOLS, run_tool, read_memory

# Point at your local server. Defaults to LM Studio + gpt-oss-20b.
# For Ollama:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.2:1b
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = 1024

# OpenAI-format tool specs, derived from the schemas in tools.py.
OAI_TOOLS = [
    {"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]

SYSTEM_TEMPLATE = """You are nano, a personal AI agent that lives on the user's laptop.
You are concise and direct, and you *act* -- when a task needs the shell, a file, or
a fact looked up, call a tool rather than describing what you would do.

You have durable memory in a file called AGENT.md, shown below. When you learn
something worth keeping across conversations (the user's name, a preference, a
decision), call the `remember` tool.

--- AGENT.md ---
{memory}
--- end AGENT.md ---
"""


class Agent:
    def __init__(self, client=None, ctx=None):
        # `client` kept for signature compatibility; we make our own local one.
        self.client = OpenAI(base_url=BASE_URL, api_key="not-needed")
        self.ctx = ctx

    def handle(self, user_text: str, on_event=None) -> str:
        system = SYSTEM_TEMPLATE.format(memory=read_memory(self.ctx) or "(empty)")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        while True:
            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=OAI_TOOLS,
            )
            msg = resp.choices[0].message

            if msg.content and msg.content.strip() and on_event:
                on_event("text", msg.content.strip())

            # No tool calls -> the turn is over. This is the whole termination rule.
            if not msg.tool_calls:
                return (msg.content or "(done)").strip()

            # Record the assistant turn (content + the tool_calls it asked for).
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool, then feed results back as `tool` messages.
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if on_event:
                    on_event("tool", f"{tc.function.name} {args}")
                out, is_err = run_tool(tc.function.name, args, self.ctx)
                if on_event:
                    on_event("result", out)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out,
                })
