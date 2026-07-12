# nanoclaw-mini

A personal AI agent, built from scratch in ~350 lines of Python — a lightweight
take on the *Build Your Own AI Agent* handbook. You chat with it in your
terminal; it runs shell commands (with your approval), reads and writes files,
and **remembers facts about you across restarts**. Every layer is code you can
read.

**It runs entirely on a local open-source model** (gpt-oss-20b via LM Studio, or
any Ollama model) — no API key, no cloud, no cost.

It's built around the handbook's four pillars:

| Pillar | File | What it does |
|---|---|---|
| **Message bus** | `bus.py` | Two SQLite files as IPC — `inbound.db` / `outbound.db`, even/odd sequencing, crash-safe ordering |
| **Agent loop** | `agent.py` | Hand-rolled plan → act → observe loop against the raw Messages API |
| **Tools + memory** | `tools.py`, `AGENT.md` | Real tools (bash, files, memory); durable memory in a Markdown file |
| **Security gate** | `run.py` | Every bash command is shown to you for approval before it runs |

## The data path

```
   you type a message
        │
        ▼
   host  ──writes──►  inbound.db   (even seq: 2, 4, 6…)
                          │
                          ▼  worker polls
                     agent loop  ── plan → act → observe ──┐
                     (Claude + tools: bash, files, memory)  │
                          │  ◄───────────────────────────────┘
                          ▼  writes reply
                     outbound.db  (odd seq: 1, 3, 5…)
                          │
                          ▼
   host  ──reads──►  prints the reply
```

No sockets between host and worker — just two SQLite files and a poll. The
worker marks an inbound message "done" only **after** its reply is written, so a
crash mid-turn means the message is retried, not lost.

## Run it

First, have a local model server running. Easiest is **LM Studio**: load
`openai/gpt-oss-20b` and start its server (`lms server start` + `lms load
openai/gpt-oss-20b`, or use the LM Studio app's Server tab). Then:

```bash
pip install -r requirements.txt
python run.py
```

**Prefer Ollama?** Point the agent at it instead — no code change:

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2:3b python run.py
```

Then talk to it:

```
you>  what year is it? use the shell
  ┌─ the agent wants to run:
  │  date +%Y
  └─ allow? [y/N] y
nano> It's 2026.

you>  my name is Raghunath, remember that
nano> Got it — I'll remember your name is Raghunath.

you>  quit
```

Restart `python run.py` and ask "what's my name?" — it still knows, because the
fact was written to `AGENT.md`.

## What each tool does

- **run_bash** — runs a shell command in `./workspace/` after you approve it.
- **read_file / write_file** — read/write text files, sandboxed to `./workspace/`.
- **remember** — appends a durable fact to `AGENT.md`, which is loaded into the
  system prompt every turn, so memory outlasts the conversation.

## The one idea to take away

Strip the hype and an agent is a **loop**: the model answers with a tool call,
your code runs it, you hand back the result, and you ask again — until the model
replies with plain text and no tool calls. That's the entire termination
condition (`agent.py`). Everything impressive an agent does emerges from this
loop plus a capable model; nobody wrote a "recover from a failed command"
feature.

## Model

Runs on whatever local server you point it at (set in `agent.py` or via the
`LLM_BASE_URL` / `LLM_MODEL` env vars). Default: `gpt-oss-20b` on LM Studio
(`http://localhost:1234/v1`). It's an OpenAI-compatible client, so any local
server that speaks that API works — swap in a bigger model for more capability
or a smaller one for speed.

## Safety

`run_bash` can do anything you can do at a shell. That's why it's gated behind an
approval prompt — the point of promoting bash to a dedicated tool is that the
harness gets a hook it can gate (handbook Chapter 9, the "lethal trifecta").
Keep the approval on unless you're sandboxed.

## Files

```
nanoclaw-mini/
├── bus.py            # SQLite message bus (IPC)
├── tools.py          # tool schemas + implementations + workspace sandbox
├── agent.py          # the hand-rolled agent loop
├── run.py            # host + worker wired together; the chat CLI
├── AGENT.md          # the agent's personality + durable memory
├── workspace/        # sandbox the agent reads/writes in
└── requirements.txt
```
