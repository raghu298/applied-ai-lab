# The Token Tax — same sentence, very different bills

Extends my LLM course Lab 2 (tokenizer comparison) into a measured experiment.

LLM APIs charge per **token**, and context windows are measured in tokens. If a
tokenizer splits your language into more pieces, you pay more per sentence and fit
less conversation into the same window — a hidden "tax" on non-English languages.

## The experiment

The same sentence — *"What is the weather today?"* — translated into 5 languages,
tokenized by three generations of tokenizers:

| Tokenizer | Year | Vocab |
|---|---|---|
| gpt2 | 2019 | 50k |
| cl100k_base (GPT-4) | 2023 | 100k |
| o200k_base (GPT-4o) | 2024 | 200k |

```bash
pip install tiktoken transformers
python token_tax.py
```

## Measured results (tokens for the same meaning)

| Language | gpt2 (2019) | cl100k (GPT-4) | o200k (GPT-4o) |
|---|---|---|---|
| English | 6 | 6 | 6 |
| Spanish | 12 | 6 | 5 |
| Hindi | 24 | 18 | **6** |
| Tamil | **85** | 41 | 9 |
| Chinese | 15 | 10 | 5 |

**Token tax vs English:** Tamil paid **14.2×** in 2019, **6.8×** with GPT-4, and
**1.5×** with GPT-4o. Hindi reached parity (1.0×).

The visceral detail: gpt2 splits `आज मौसम कैसा है?` (16 characters) into 24
byte-fallback fragments — not one of them is a real word piece.

## Why it happens

Old tokenizers were trained mostly on English web text. Scripts they rarely saw
(Devanagari, Tamil) fall back to raw UTF-8 bytes — 3 bytes per character — so every
character costs ~2–3 tokens. Newer tokenizers trained on multilingual data with
bigger vocabularies (50k → 100k → 200k) learn real subwords for these scripts.

## Why it still matters

- Per-token pricing means the tax is a real cost multiplier on API bills.
- More tokens = fewer conversations fit in the context window.
- More tokens = more decode steps = higher latency.
- Many open models still ship tokenizers with a substantial tax on Indic scripts.

## Files

- `token_tax.py` — the experiment
- `token_tax_results.json` — machine-readable results from the last run
