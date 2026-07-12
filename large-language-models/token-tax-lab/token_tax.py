"""
The Token Tax — same sentence, very different bills.
====================================================

Extends Lab 2 (tokenizer comparison). LLM APIs charge per TOKEN, not per word
or per character. Context windows are measured in tokens too. So if a tokenizer
splits your language into more pieces, you literally pay more per sentence and
fit less conversation into the same window.

This experiment tokenizes the SAME sentence ("What is the weather today?")
translated into 5 languages, across three generations of tokenizers:

  gpt2         (2019, 50k vocab)   - trained mostly on English web text
  cl100k_base  (2023, 100k vocab)  - GPT-4's tokenizer
  o200k_base   (2024, 200k vocab)  - GPT-4o's tokenizer, multilingual-aware

Run:  python token_tax.py
Deps: tiktoken, transformers
"""

import json
import tiktoken
from transformers import AutoTokenizer

SENTENCES = {
    "English":  "What is the weather today?",
    "Spanish":  "¿Qué tiempo hace hoy?",
    "Hindi":    "आज मौसम कैसा है?",
    "Tamil":    "இன்று வானிலை எப்படி இருக்கிறது?",
    "Chinese":  "今天天气怎么样？",
}

def main():
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    cl100k = tiktoken.get_encoding("cl100k_base")   # GPT-4
    o200k = tiktoken.get_encoding("o200k_base")     # GPT-4o

    tokenizers = {
        "gpt2 (2019)":    lambda s: len(gpt2(s)["input_ids"]),
        "cl100k (GPT-4)": lambda s: len(cl100k.encode(s)),
        "o200k (GPT-4o)": lambda s: len(o200k.encode(s)),
    }

    results = {}
    print(f"{'Language':<10}", *[f"{n:>16}" for n in tokenizers], sep="")
    print("-" * 60)
    for lang, text in SENTENCES.items():
        counts = {n: f(text) for n, f in tokenizers.items()}
        results[lang] = {"text": text, **counts}
        print(f"{lang:<10}", *[f"{c:>16}" for c in counts.values()], sep="")

    print("\nTOKEN TAX vs English (same meaning, x times the tokens):")
    print(f"{'Language':<10}", *[f"{n:>16}" for n in tokenizers], sep="")
    print("-" * 60)
    eng = results["English"]
    for lang in SENTENCES:
        if lang == "English":
            continue
        taxes = [f"{results[lang][n] / eng[n]:>15.1f}x" for n in tokenizers]
        print(f"{lang:<10}", *taxes, sep="")

    # Show the actual pieces for Hindi on the oldest tokenizer - the visceral bit
    hindi = SENTENCES["Hindi"]
    pieces = gpt2.convert_ids_to_tokens(gpt2(hindi)["input_ids"])
    print(f"\ngpt2 splits {hindi!r} ({len(hindi)} chars) into {len(pieces)} tokens:")
    print("  ", pieces)

    with open("token_tax_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved -> token_tax_results.json")


if __name__ == "__main__":
    main()
