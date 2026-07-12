#!/usr/bin/env bash
# Launch nanoclaw-mini: make sure the local model server is up, then run.
set -e
cd "$(dirname "$0")"

MODEL="${LLM_MODEL:-openai/gpt-oss-20b}"

# Is a local server already answering on :1234 (LM Studio)?
if ! curl -s http://localhost:1234/v1/models >/dev/null 2>&1; then
  echo "Starting LM Studio server + loading $MODEL ..."
  lms server start
  lms load "$MODEL"
else
  echo "Local model server already running."
fi

echo
python3 run.py
