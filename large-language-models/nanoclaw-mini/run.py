"""
nanoclaw-mini -- a personal AI agent you built from scratch, running on a
LOCAL model (no API key, no cloud, no cost).

Prereq: a local OpenAI-compatible server is running. Default is LM Studio with
gpt-oss-20b on port 1234. (For Ollama, set LLM_BASE_URL / LLM_MODEL -- see agent.py.)

Run:  python run.py

This wires the four pieces together in one process, but a real message still
flows the full path from the handbook:

    you type  ->  host writes inbound.db  ->  worker polls it  ->  agent loop
    (plan/act/observe with tools)  ->  worker writes outbound.db  ->  host
    prints the reply.

Type 'quit' to exit.  Your agent's memory persists in AGENT.md between runs.
"""

import os
import sys

from bus import SessionBus
from agent import Agent
from tools import Context

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(HERE, "session")
WORKSPACE = os.path.join(HERE, "workspace")
MEMORY = os.path.join(HERE, "AGENT.md")

# --- colours (fall back to plain text if not a terminal) -------------------
def _c(code):
    return code if sys.stdout.isatty() else ""
DIM, CYAN, AMBER, GREEN, RESET = _c("\033[2m"), _c("\033[36m"), _c("\033[33m"), _c("\033[32m"), _c("\033[0m")


def approver(command: str) -> bool:
    """The security gate: show the command, let the user allow or deny it."""
    print(f"{AMBER}  ┌─ the agent wants to run:{RESET}")
    print(f"{AMBER}  │  {command}{RESET}")
    ans = input(f"{AMBER}  └─ allow? [y/N] {RESET}").strip().lower()
    return ans in ("y", "yes")


def on_event(kind: str, payload: str) -> None:
    if kind == "text":
        print(f"{CYAN}nano>{RESET} {payload}")
    elif kind == "tool":
        print(f"{DIM}  · calling {payload}{RESET}")
    elif kind == "result":
        first = payload.splitlines()[0] if payload else ""
        print(f"{DIM}  · -> {first[:80]}{RESET}")


def main() -> None:
    bus = SessionBus(SESSION_DIR)
    ctx = Context(WORKSPACE, MEMORY, approver)
    agent = Agent(ctx=ctx)   # talks to your local model server

    print(f"{GREEN}nanoclaw-mini{RESET}  (type 'quit' to exit)")
    print(f"{DIM}memory: {MEMORY}   workspace: {WORKSPACE}{RESET}\n")

    while True:
        try:
            user = input(f"{GREEN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue

        # 1. host drops the message on the bus
        bus.put_inbound(user)

        # 2. worker: drain inbound, run the agent, write the reply -- and mark
        #    the row done only AFTER the reply is written (crash-safe ordering)
        for row in bus.pending_inbound():
            reply = agent.handle(row.text, on_event=on_event)
            bus.put_outbound(reply)
            bus.mark_done(row.seq)

        # 3. host: deliver whatever is waiting in outbound
        for row in bus.pending_outbound():
            bus.mark_delivered(row.seq)


if __name__ == "__main__":
    main()
