"""
nanoclaw-mini (web demo) -- a from-scratch AI agent you can try in the browser.

This is the SAFE, hosted cousin of the local terminal agent. The core idea is
identical: an agent is a *loop*. The model can answer with a tool call instead of
text; we run the tool, hand back the result, and ask again -- plan -> act ->
observe -- until it replies with plain text.

Two deliberate differences from the local version, both for safety:
  * No shell / filesystem tools. A public demo must never run commands on the
    server (that's the "lethal trifecta" -- untrusted input + a real machine).
    The tools here are pure and harmless: a calculator, the clock, and memory.
  * The model runs on Hugging Face's servers via your token, not on this box.

Set one secret in your Space:  HF_TOKEN  (a free token from hf.co/settings/tokens)
"""

import os
import re
import ast
import json
import operator
import datetime

import gradio as gr
from huggingface_hub import InferenceClient

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
client = InferenceClient(model=MODEL, token=os.environ.get("HF_TOKEN"))
MAX_STEPS = 4


# ---------------------------------------------------------------- safe tools
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}


def _calc(expr: str) -> str:
    """Evaluate arithmetic safely (no eval, no names, just numbers + operators)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expr, mode="eval").body))
    except Exception:
        return "Error: I can only do basic arithmetic (+ - * / ** %)."


def run_tool(name: str, args: dict, memory: list) -> str:
    if name == "calculator":
        return _calc(str(args.get("expression", "")))
    if name == "current_time":
        return datetime.datetime.utcnow().strftime("%A, %d %B %Y, %H:%M UTC")
    if name == "remember":
        fact = str(args.get("fact", "")).strip()
        if fact:
            memory.append(fact)
            return "Saved to memory."
        return "Nothing to save."
    if name == "recall":
        return "\n".join(f"- {m}" for m in memory) if memory else "(memory is empty)"
    return f"Unknown tool: {name}"


TOOL_DOCS = """calculator   -> args: {"expression": "2*(3+4)"}   basic arithmetic
current_time -> args: {}                          today's date and time (UTC)
remember     -> args: {"fact": "..."}             save a fact for later this chat
recall       -> args: {}                          list everything you've remembered"""

SYSTEM = f"""You are nano, a small AI agent. You can use tools to act.

To use a tool, reply with EXACTLY one line and nothing else:
ACTION: <tool_name> <json-args>

Available tools:
{TOOL_DOCS}

After you see the OBSERVATION, either use another tool or give your final answer
as plain text (no ACTION line). Keep answers short and friendly.

Example:
User: what is 21 * 19?
ACTION: calculator {{"expression": "21*19"}}
OBSERVATION: 399
The answer is 399."""

_ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)\s*(\{.*\})?", re.IGNORECASE | re.DOTALL)


def parse_action(text: str):
    m = _ACTION_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip().lower()
    try:
        args = json.loads(m.group(2)) if m.group(2) else {}
    except json.JSONDecodeError:
        args = {}
    return name, args


# ---------------------------------------------------------------- agent loop
def agent_reply(user_msg: str, history: list, memory: list) -> tuple[str, str]:
    """Returns (final_answer, trace). Trace shows the plan->act->observe steps."""
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in history:                       # gradio 'messages' format
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})

    trace = []
    for _ in range(MAX_STEPS):
        out = client.chat_completion(messages=messages, max_tokens=400, temperature=0.3)
        text = (out.choices[0].message.content or "").strip()

        action = parse_action(text)
        if not action:
            return text or "(no reply)", "\n".join(trace)

        name, args = action
        obs = run_tool(name, args, memory)
        trace.append(f"🔧 {name}({json.dumps(args)}) → {obs}")
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})

    return "I got stuck taking too many steps — try rephrasing?", "\n".join(trace)


# ---------------------------------------------------------------- UI
INTRO = """# 🤖 nanoclaw-mini — a tiny AI agent, built from scratch

An AI **agent** isn't magic — it's a *loop*: the model answers with a tool call
instead of text, the code runs the tool, hands back the result, and asks again
(**plan → act → observe**) until it replies with plain text.

This is a safe browser demo. It has three harmless tools — **calculator**,
**clock**, and **memory** — and no access to any real system. The full version
runs on your own laptop with a local model and real tools.

**Try:** *"what is 47 × 89?"* · *"remember that my name is Priya"* · *"what's the date, and what did I ask you to remember?"*
"""


def respond(message, history, memory):
    # `memory` is a gr.State list; mutating it in place persists across turns
    # in this session (run_tool appends to it).
    answer, trace = agent_reply(message, history, memory)
    if trace:
        answer = answer + f"\n\n<details><summary>see how it thought</summary>\n\n{trace}\n\n</details>"
    return answer


with gr.Blocks(title="nanoclaw-mini") as demo:
    gr.Markdown(INTRO)
    mem = gr.State([])
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[mem],
        type="messages",
    )
    gr.Markdown("Built from the *Build Your Own AI Agent* handbook · "
                "the full local version runs real tools on your own machine.")


if __name__ == "__main__":
    demo.launch()
