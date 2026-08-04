"""LLMOps: choose the human-review threshold from evidence rather than by guess.

The triage classifier does not have to be right on its own. It has to know when
it is unsure, so that uncertain cases reach a triage nurse instead of being auto
routed. This script sweeps the confidence threshold over the held-out test split
and reports, for each candidate value, how much of the workload is handled
automatically and how accurate the model is on that portion.

Output: artifacts/threshold_calibration.csv and .png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

# The sweep range follows the classifier. The original distilbert averaged
# 0.54 confidence, so candidates sat between 0.15 and 0.70; the ModernBERT
# replacement averages 0.97, so the informative region moved upward.
CANDIDATES = [0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]


def main():
    ds = load_dataset(config.FINETUNE_DATASET)["test"]
    texts = [str(x) for x in ds["input_text"]]
    golds = [str(x) for x in ds["output_text"]]

    tokenizer = AutoTokenizer.from_pretrained(str(config.FINETUNED_TRIAGE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config.FINETUNED_TRIAGE_DIR)
    )
    model.eval()

    enc = tokenizer(
        texts, truncation=True, max_length=256, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)

    confidence, predicted = probs.max(dim=-1)
    gold_ids = [model.config.label2id[g] for g in golds]
    correct = np.array(
        [int(p == g) for p, g in zip(predicted.tolist(), gold_ids)]
    )
    conf = confidence.numpy()

    rows = []
    for t in CANDIDATES:
        auto = conf >= t
        n_auto = int(auto.sum())
        n_flag = int((~auto).sum())
        rows.append(
            {
                "threshold": t,
                "auto_handled_pct": round(100.0 * n_auto / len(conf), 1),
                "accuracy_when_auto": round(
                    100.0 * correct[auto].mean(), 1
                )
                if n_auto
                else None,
                "accuracy_when_flagged": round(
                    100.0 * correct[~auto].mean(), 1
                )
                if n_flag
                else None,
                "cases_sent_for_review": n_flag,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = config.ARTIFACT_DIR / "threshold_calibration.csv"
    df.to_csv(out_csv, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["threshold"], df["auto_handled_pct"], marker="o",
            label="Workload handled automatically (%)")
    ax.plot(df["threshold"], df["accuracy_when_auto"], marker="s",
            label="Accuracy on automatically handled cases (%)")
    ax.axvline(config.TRIAGE_REVIEW_THRESHOLD, linestyle="--", color="grey")
    ax.annotate(
        f"chosen threshold {config.TRIAGE_REVIEW_THRESHOLD}",
        xy=(config.TRIAGE_REVIEW_THRESHOLD, 50),
        xytext=(config.TRIAGE_REVIEW_THRESHOLD + 0.03, 45),
        fontsize=9,
    )
    ax.set_xlabel("Confidence threshold for automatic routing")
    ax.set_ylabel("Percent")
    ax.set_title("Triage classifier: automation rate against accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.ARTIFACT_DIR / "threshold_calibration.png", dpi=150)
    plt.close(fig)

    print(f"Mean confidence on the test split: {conf.mean():.3f}")
    print(df.to_string(index=False))
    print(f"\nWritten to {out_csv}")


if __name__ == "__main__":
    main()
