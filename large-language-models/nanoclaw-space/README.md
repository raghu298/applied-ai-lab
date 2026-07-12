---
title: nanoclaw-mini
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: A tiny AI agent built from scratch — plan, act, observe.
---

# nanoclaw-mini

A tiny AI **agent**, built from scratch, that you can try in the browser.

An agent isn't magic — it's a *loop*. The model can answer with a **tool call**
instead of text; the code runs the tool, hands back the result, and asks again
(**plan → act → observe**) until it replies with plain text. That's the whole
trick.

This hosted demo is deliberately **safe**: it has three harmless tools —
`calculator`, `current_time`, and `remember`/`recall` — and **no access to any
real system**. The full version runs on your own laptop with a local open-source
model and real tools (shell, files), gated behind an approval prompt.

## Setup (for the Space owner)

1. Create this Space (SDK = Gradio) on Hugging Face.
2. Add a **secret** named `HF_TOKEN` — a free token from
   https://huggingface.co/settings/tokens (read access is enough).
   The demo uses it to call a chat model on HF's servers.
3. (Optional) Set a `MODEL_ID` variable to try a different instruct model.
   Default: `Qwen/Qwen2.5-7B-Instruct`.

## Files

- `app.py` — the Gradio UI + the agent loop + the safe tools
- `requirements.txt` — gradio, huggingface_hub
