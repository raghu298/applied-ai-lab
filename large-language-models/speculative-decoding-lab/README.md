# Speculative Decoding — from scratch, on two real models

A small, runnable lab that implements **speculative decoding** (Leviathan et al. /
Chen et al., 2022) on two real language models and measures what it actually buys you.

- **Target model:** `gpt2` (124M) — the model whose output we want, verbatim
- **Draft model:** `distilgpt2` (82M) — a smaller, faster guesser (same tokenizer)

## The idea

Decoding one token normally costs a full read of the big model's weights, while the
GPU's compute units sit idle (decode is *memory-bound*). But *verifying* whether k
tokens are what the model would have produced costs a **single** forward pass.

So: let the cheap draft model guess k tokens ahead, then let the big model verify all
k at once and keep the longest correct prefix. Accepted guesses are free tokens
squeezed out of bandwidth you were already wasting.

## What the experiment proves

Run it:

```bash
pip install torch transformers      # first run downloads ~500 MB of weights
python speculative_decoding.py
```

1. **Correctness** — greedy speculative output is *identical* to greedy target
   decoding, token for token. It's exact, not an approximation.
2. **The mechanism** — draft proposes k; target verifies in one pass; keep the
   matched prefix + one free correction token.
3. **The payoff** — the big model runs far fewer forward passes for the same text.

## Measured results (gpt2 + distilgpt2, k=4, 48 new tokens, 4 prompts)

| Metric | Result |
|---|---|
| Outputs identical to target-only decoding | **4 / 4** |
| Acceptance rate (alpha) | ~0.55 |
| Avg tokens per target forward pass | **~3.05** |
| Target forward passes | **192 → 63** (3.05× fewer) |

The forward-pass reduction is hardware-independent: it measures trips through the big
model's weights directly. On a real GPU with a larger target/draft size gap, that
reduction turns into 2–3× lower latency — which is exactly why every serious serving
engine (vLLM, TGI, TensorRT-LLM) ships speculative decoding.

## Honest caveats (in the code's output too)

- It's a **latency** tool, not a throughput tool. At large batch sizes the GPU is
  already busy and verification FLOPs compete with other users' work.
- High-temperature / creative sampling drafts poorly → low acceptance → little gain.
- Predictable text (chat, code, RAG that quotes its context) drafts best.

## Files

- `speculative_decoding.py` — the full, commented implementation + measurement
- `speculative_results.json` — machine-readable results from the last run
