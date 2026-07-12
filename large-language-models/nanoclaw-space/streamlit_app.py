"""
nanoclaw-mini -- a tiny AI agent you can try in the browser (Streamlit).

An agent isn't magic -- it's a *loop*: the model answers with a tool call instead
of text; the code runs the tool, hands back the result, and asks again
(plan -> act -> observe) until it replies with plain text.

This hosted demo is deliberately SAFE: the only tools are a calculator, the
clock, and per-chat memory. No shell, no filesystem, no access to any real
machine. The model runs on Hugging Face's servers via a token you store as a
Streamlit secret (HF_TOKEN).
"""

import os
import re
import ast
import json
import operator
import datetime

import streamlit as st
from huggingface_hub import InferenceClient


def _get_token() -> str:
    # Streamlit secrets first, then env var.
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "")


MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


@st.cache_resource
def get_client():
    return InferenceClient(model=MODEL, token=_get_token())


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


TOOL_DOCS = """calculator   -> args: {"expression": "2*(3+4)"}   basic arithmetic
current_time -> args: {}                          today's date and time
remember     -> args: {"fact": "..."}             save a fact for later this chat
recall       -> args: {}                          list everything remembered"""

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
    try:
        args = json.loads(m.group(2)) if m.group(2) else {}
    except json.JSONDecodeError:
        args = {}
    return m.group(1).lower(), args


def agent_reply(user_msg, history, memory):
    client = get_client()
    msgs = [{"role": "system", "content": SYSTEM}]
    for role, content in history:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_msg})

    for _ in range(4):
        out = client.chat_completion(messages=msgs, max_tokens=400, temperature=0.3)
        text = (out.choices[0].message.content or "").strip()
        action = parse_action(text)
        if not action:
            return text or "(no reply)"
        name, args = action
        obs = run_tool(name, args, memory)
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user", "content": f"OBSERVATION: {obs}"})
    return "I took too many steps — try rephrasing?"


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="nanoclaw-mini", page_icon="🤖")
st.title("🤖 nanoclaw-mini")
st.caption("A tiny AI agent, built from scratch. An agent is just a loop: "
           "plan → act → observe, until it answers with plain text. "
           "Safe demo tools only — calculator, clock, memory.")

if not _get_token():
    st.error("This demo needs an `HF_TOKEN` secret (Settings → Secrets on "
             "Streamlit Cloud). Get a free token at huggingface.co/settings/tokens.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []      # list of (role, content)
if "memory" not in st.session_state:
    st.session_state.memory = []

with st.sidebar:
    st.markdown("**Try asking**")
    st.markdown("- what is 47 × 89?\n- remember my name is Priya\n- what's the date, "
                "and what did I ask you to remember?")
    if st.session_state.memory:
        st.markdown("**Remembered this chat**")
        for m in st.session_state.memory:
            st.markdown(f"- {m}")

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Ask nano something…"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("thinking…"):
            reply = agent_reply(prompt, st.session_state.messages[:-1], st.session_state.memory)
        st.markdown(reply)
    st.session_state.messages.append(("assistant", reply))
    st.rerun()
