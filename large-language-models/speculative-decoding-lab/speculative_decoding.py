"""
Speculative decoding, from scratch, on two REAL language models.
================================================================

The claim (Leviathan et al. / Chen et al., 2022):
  A small, fast "draft" model can guess several tokens ahead. A big "target"
  model then VERIFIES all of them in a single forward pass and keeps the longest
  correct prefix. The text you get out is *identical* to running the big model
  alone -- but the big model runs far fewer times.

  Why that matters: decoding is memory-bound. Every token normally costs one full
  read of the big model's weights. Verifying k guesses costs ONE read. So accepted
  guesses are, in effect, free tokens squeezed out of bandwidth you were wasting.

This file proves three things on real models (gpt2 = target, distilgpt2 = draft):

  1. CORRECTNESS  - greedy speculative output is character-for-character identical
                    to greedy target decoding. Exact, not approximate.
  2. THE MECHANISM- draft proposes k tokens; target verifies in one pass; we keep
                    the matched prefix plus one "correction" token (free from the
                    same pass).
  3. THE PAYOFF   - to produce N tokens, the target model runs only M forward
                    passes (M << N). That ratio is the memory-traffic saving, and
                    it does not depend on your hardware.

Run:  python speculative_decoding.py
Deps: torch, transformers   (CPU is fine; first run downloads ~500 MB of weights)
"""

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_NAME = "gpt2"        # 124M params - the model whose output we want, verbatim
DRAFT_NAME  = "distilgpt2"  # 82M  params - a smaller, faster guesser (SAME tokenizer)

torch.manual_seed(0)
DEVICE = "cpu"


def load():
    print(f"Loading target={TARGET_NAME}  draft={DRAFT_NAME} ...")
    tok = AutoTokenizer.from_pretrained(TARGET_NAME)
    target = AutoModelForCausalLM.from_pretrained(TARGET_NAME).to(DEVICE).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_NAME).to(DEVICE).eval()
    return tok, target, draft


@torch.no_grad()
def next_token_logits(model, ids):
    """Logits over the vocabulary for the position AFTER the given ids."""
    out = model(torch.tensor([ids], device=DEVICE))
    return out.logits[0, -1]


@torch.no_grad()
def baseline_greedy(model, prefix, n_new):
    """Ordinary autoregressive decoding: ONE target forward pass per token."""
    ids = list(prefix)
    passes = 0
    for _ in range(n_new):
        logits = next_token_logits(model, ids)
        passes += 1
        ids.append(int(logits.argmax()))
    return ids[len(prefix):], passes


@torch.no_grad()
def speculative_greedy(target, draft, prefix, n_new, k):
    """
    Speculative decoding (greedy). Returns (tokens, target_passes, proposed, accepted).
    Guaranteed to equal baseline_greedy(target, ...) token for token.
    """
    ids = list(prefix)
    target_passes = proposed = accepted = 0
    limit = len(prefix) + n_new

    while len(ids) < limit:
        # 1. DRAFT: the small model guesses k tokens, one at a time (cheap).
        guesses = []
        cur = list(ids)
        for _ in range(k):
            g = int(next_token_logits(draft, cur).argmax())
            guesses.append(g)
            cur.append(g)
        proposed += k

        # 2. VERIFY: the big model scores the whole block in ONE forward pass.
        out = target(torch.tensor([ids + guesses], device=DEVICE))
        target_passes += 1
        logits = out.logits[0]                 # one row of logits per input position
        base = len(ids) - 1                    # row that predicts guesses[0]

        # 3. Keep the longest prefix the target agrees with.
        n_ok = 0
        for i in range(k):
            target_choice = int(logits[base + i].argmax())
            if target_choice == guesses[i]:
                n_ok += 1
            else:
                break
        accepted += n_ok

        ids.extend(guesses[:n_ok])             # accepted guesses
        # 4. FREE CORRECTION: the same pass already told us the target's own next
        #    token at the divergence point. Append it -> stays exactly on target.
        correction = int(logits[base + n_ok].argmax())
        ids.append(correction)

        if len(ids) >= limit:
            break

    return ids[len(prefix):len(prefix) + n_new], target_passes, proposed, accepted


def run():
    tok, target, draft = load()

    prompts = [
        "The capital of France is Paris, and the capital of Italy is",
        "def add(a, b):\n    return a +",
        "In machine learning, a neural network is a model that",
        "Once upon a time, there was a little robot who loved to",
    ]
    N_NEW = 48
    K = 4

    print("\n" + "=" * 68)
    print(f"EXPERIMENT   target={TARGET_NAME}  draft={DRAFT_NAME}  k={K}  new_tokens={N_NEW}")
    print("=" * 68)

    agg = {"identical": 0, "base_passes": 0, "spec_passes": 0,
           "proposed": 0, "accepted": 0, "base_time": 0.0, "spec_time": 0.0}

    for p in prompts:
        prefix = tok(p)["input_ids"]

        t0 = time.time()
        base_ids, base_passes = baseline_greedy(target, prefix, N_NEW)
        t_base = time.time() - t0

        t0 = time.time()
        spec_ids, spec_passes, proposed, accepted = speculative_greedy(
            target, draft, prefix, N_NEW, K)
        t_spec = time.time() - t0

        identical = base_ids == spec_ids
        alpha = accepted / proposed if proposed else 0.0
        toks_per_pass = N_NEW / spec_passes

        agg["identical"] += int(identical)
        agg["base_passes"] += base_passes
        agg["spec_passes"] += spec_passes
        agg["proposed"] += proposed
        agg["accepted"] += accepted
        agg["base_time"] += t_base
        agg["spec_time"] += t_spec

        print(f"\nprompt: {p!r}")
        print(f"  identical output ......... {'YES' if identical else 'NO'}")
        print(f"  target passes: baseline {base_passes:>3}  ->  speculative {spec_passes:>3}")
        print(f"  acceptance rate (alpha) .. {alpha:5.2f}")
        print(f"  tokens per target pass ... {toks_per_pass:4.2f}")

    print("\n" + "-" * 68)
    a = agg["accepted"] / agg["proposed"]
    tpp = agg["base_passes"] / agg["spec_passes"]        # = tokens per target pass
    pass_reduction = agg["base_passes"] / agg["spec_passes"]
    print("AGGREGATE OVER ALL PROMPTS")
    print(f"  outputs identical to target-only decoding : {agg['identical']}/{len(prompts)}")
    print(f"  measured acceptance rate  (alpha)         : {a:.2f}")
    print(f"  avg tokens per target forward pass        : {tpp:.2f}")
    print(f"  target forward passes: {agg['base_passes']}  ->  {agg['spec_passes']}"
          f"   ({pass_reduction:.2f}x fewer trips through the big weights)")
    print(f"  wall clock (CPU, noisy): {agg['base_time']:.1f}s -> {agg['spec_time']:.1f}s")

    # Theory check: expected tokens per pass = (1 - a^(k+1)) / (1 - a)   [book 7.3]
    theo = (1 - a ** (K + 1)) / (1 - a)
    print(f"\n  theory  (1 - a^(k+1))/(1 - a) at a={a:.2f}, k={K} : {theo:.2f} tokens/pass")
    print(f"  measured                                        : {tpp:.2f} tokens/pass")

    # Sweep k to show the tradeoff, using the measured alpha.
    print("\n  draft length k vs expected tokens/pass (measured alpha):")
    sweep = {}
    for k in (1, 2, 3, 4, 6, 8):
        e = (1 - a ** (k + 1)) / (1 - a)
        sweep[k] = round(e, 2)
        bar = "#" * int(e * 6)
        print(f"    k={k}: {e:4.2f}  {bar}")

    results = {
        "target": TARGET_NAME, "draft": DRAFT_NAME, "k": K, "n_new": N_NEW,
        "identical": f"{agg['identical']}/{len(prompts)}",
        "alpha": round(a, 3),
        "tokens_per_pass": round(tpp, 2),
        "base_passes": agg["base_passes"], "spec_passes": agg["spec_passes"],
        "pass_reduction": round(pass_reduction, 2),
        "theory_tokens_per_pass": round(theo, 2),
        "k_sweep": sweep,
    }
    with open("speculative_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> speculative_results.json")
    return results


if __name__ == "__main__":
    run()
