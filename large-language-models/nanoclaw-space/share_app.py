"""
nanoclaw-mini -- public browser demo, powered by your LOCAL model.

Run:   python3 share_app.py
It prints a public https://xxxx.gradio.live link (valid ~1 week) that anyone can
open. The brain is your local gpt-oss-20b (via LM Studio), so it costs nothing.

Safe by design: the only tools are a calculator, the clock, and per-chat memory.
There is NO shell or filesystem access -- a public link must never be able to
touch your machine.
"""

import os
import ast
import json
import operator
import datetime

import gradio as gr
from openai import OpenAI

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
client = OpenAI(base_url=BASE_URL, api_key="not-needed")

# ---------------------------------------------------------------- safe tools
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}


def _calc(expr: str) -> str:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        return str(ev(ast.parse(expr, mode="eval").body))
    except Exception:
        return "Error: basic arithmetic only (+ - * / ** %)."


def run_tool(name, args, memory):
    if name == "calculator":
        return _calc(str(args.get("expression", "")))
    if name == "current_time":
        return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")
    if name == "remember":
        f = str(args.get("fact", "")).strip()
        if f:
            memory.append(f); return "Saved to memory."
        return "Nothing to save."
    if name == "recall":
        return "\n".join(f"- {m}" for m in memory) if memory else "(memory is empty)"
    return f"Unknown tool: {name}"


TOOLS = [
    {"type": "function", "function": {"name": "calculator", "description": "Do basic arithmetic.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "current_time", "description": "Get today's date and the current time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "remember", "description": "Save a fact about the user for later in this chat.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}},
    {"type": "function", "function": {"name": "recall", "description": "List everything remembered so far.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = ("You are nano, a friendly little AI agent. You can call tools to act: a "
          "calculator, a clock, and a memory. Use them when helpful, then answer "
          "briefly in plain language. You have no access to the user's computer.")


# ---------------------------------------------------------------- agent loop
def respond(message, history, memory):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in history:
        if t.get("role") in ("user", "assistant") and t.get("content"):
            msgs.append({"role": t["role"], "content": t["content"]})
    msgs.append({"role": "user", "content": message})

    for _ in range(5):
        r = client.chat.completions.create(model=MODEL, max_tokens=500, messages=msgs, tools=TOOLS)
        m = r.choices[0].message
        if not m.tool_calls:
            return (m.content or "(done)").strip()
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in m.tool_calls]})
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = run_tool(tc.function.name, args, memory)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "That took too many steps — try rephrasing?"


INTRO = """# 🤖 nanoclaw-mini — a tiny AI agent, built from scratch

An AI **agent** isn't magic — it's a *loop*: the model answers with a **tool
call** instead of text, the code runs the tool, hands back the result, and asks
again (**plan → act → observe**) until it replies with plain text.

Running on a **local open-source model** — no cloud, no API. Safe demo tools only
(calculator, clock, memory); no access to any real system.

**Try:** *"what is 47 × 89?"* · *"remember my name is Priya"* · *"what's the date, and what did I ask you to remember?"*
"""


def chat_fn(message, history, memory):
    return respond(message, history, memory)


with gr.Blocks(title="nanoclaw-mini") as demo:
    gr.Markdown(INTRO)
    mem = gr.State([])
    gr.ChatInterface(fn=chat_fn, additional_inputs=[mem], type="messages")


if __name__ == "__main__":
    demo.launch(share=True)
