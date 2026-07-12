"""
The agent's tools -- Chapter 6 of the handbook.

Each tool is a JSON schema (so the model knows how to call it) plus a Python
function that actually runs it. The agent loop in agent.py calls run_tool()
whenever the model emits a tool_use block.

Security note (Chapter 9 -- the lethal trifecta): run_bash can do anything, so
it is gated behind an approval callback. By default every command must be
approved by you before it runs. That's the whole point of promoting bash to a
"dedicated tool" -- the harness gets a hook it can gate.
"""

import os
import subprocess

# ---- tool schemas the model sees -----------------------------------------
TOOLS = [
    {
        "name": "run_bash",
        "description": "Run a bash command on the user's machine and return its "
                       "output. Use for real actions: listing files, running a "
                       "script, fetching a URL, checking the date. Every command "
                       "is shown to the user for approval before it runs.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file from the agent's workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the workspace"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file in the agent's workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace"},
                "content": {"type": "string", "description": "The full file contents"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "remember",
        "description": "Save a durable fact about the user or the ongoing work to "
                       "long-term memory (AGENT.md). Use this for anything worth "
                       "recalling in a future conversation -- names, preferences, "
                       "decisions. Memory survives restarts.",
        "input_schema": {
            "type": "object",
            "properties": {"fact": {"type": "string", "description": "One concise fact to remember"}},
            "required": ["fact"],
        },
    },
]

MEMORY_HEADER = "## Memories"


class Context:
    """Everything the tools need: where the workspace and memory live, and how
    to ask the user to approve a command."""
    def __init__(self, workspace: str, memory_path: str, approver):
        self.workspace = os.path.abspath(workspace)
        self.memory_path = memory_path
        self.approver = approver          # approver(command:str) -> bool
        os.makedirs(self.workspace, exist_ok=True)


def _safe_path(ctx: Context, rel: str) -> str | None:
    """Resolve rel inside the workspace; return None if it escapes."""
    full = os.path.abspath(os.path.join(ctx.workspace, rel))
    if os.path.commonpath([full, ctx.workspace]) != ctx.workspace:
        return None
    return full


def run_tool(name: str, args: dict, ctx: Context) -> tuple[str, bool]:
    """Execute a tool. Returns (result_text, is_error)."""
    try:
        if name == "run_bash":
            cmd = args["command"]
            if not ctx.approver(cmd):
                return "The user declined to run this command.", True
            proc = subprocess.run(
                cmd, shell=True, cwd=ctx.workspace,
                capture_output=True, text=True, timeout=30)
            out = (proc.stdout + proc.stderr).strip() or "(no output)"
            return out[:4000], proc.returncode != 0

        if name == "read_file":
            full = _safe_path(ctx, args["path"])
            if not full:
                return "Path escapes the workspace.", True
            with open(full) as f:
                return f.read()[:4000], False

        if name == "write_file":
            full = _safe_path(ctx, args["path"])
            if not full:
                return "Path escapes the workspace.", True
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(args["content"])
            return f"Wrote {len(args['content'])} bytes to {args['path']}.", False

        if name == "remember":
            _append_memory(ctx, args["fact"])
            return "Saved to memory.", False

        return f"Unknown tool: {name}", True
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s.", True
    except Exception as e:
        return f"{type(e).__name__}: {e}", True


def _append_memory(ctx: Context, fact: str) -> None:
    text = read_memory(ctx)
    if MEMORY_HEADER not in text:
        text = text.rstrip() + f"\n\n{MEMORY_HEADER}\n"
    text = text.rstrip() + f"\n- {fact}\n"
    with open(ctx.memory_path, "w") as f:
        f.write(text)


def read_memory(ctx: Context) -> str:
    if os.path.exists(ctx.memory_path):
        with open(ctx.memory_path) as f:
            return f.read()
    return ""
