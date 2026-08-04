"""Requirement 8 - fine-tune a small language model on a healthcare dataset.

Base model : distilbert-base-uncased (66M parameter SLM)
Dataset    : gretelai/symptom_to_diagnosis
             853 training and 212 test records, 22 diagnosis classes, where the
             input is a free-text symptom description written the way a patient
             would speak it, and the label is the corresponding condition.
Task       : single-label sequence classification, used by sub-task 2.

The script evaluates the base model before training to establish a baseline,
trains, evaluates again, writes a confusion matrix and a per-class report to the
artifacts directory, and logs the run to MLflow.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import config

SEED = 42
EPOCHS = 6
BATCH_SIZE = 16
# 5e-5 suits ModernBERT-base; the earlier distilbert run used 3e-5.
LEARNING_RATE = 5e-5
MAX_LENGTH = 256


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def build_dataset(tokenizer, labels: list[str]):
    ds = load_dataset(config.FINETUNE_DATASET)
    label2id = {name: i for i, name in enumerate(labels)}

    def prepare(batch):
        enc = tokenizer(
            batch["input_text"], truncation=True, max_length=MAX_LENGTH
        )
        enc["labels"] = [label2id[x] for x in batch["output_text"]]
        return enc

    keep = ["input_ids", "attention_mask", "labels"]
    tokenized = ds.map(prepare, batched=True, remove_columns=ds["train"].column_names)
    tokenized = tokenized.select_columns(keep)
    return ds, tokenized


def plot_confusion(cm, labels, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    ax.set_title("Fine-tuned triage classifier - confusion matrix on held-out test set")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset:", config.FINETUNE_DATASET)
    raw = load_dataset(config.FINETUNE_DATASET)
    labels = sorted(set(raw["train"]["output_text"]))
    print(f"  train={len(raw['train'])}  test={len(raw['test'])}  classes={len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_CLASSIFIER)
    _, tokenized = build_dataset(tokenizer, labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_CLASSIFIER,
        num_labels=len(labels),
        id2label={i: n for i, n in enumerate(labels)},
        label2id={n: i for i, n in enumerate(labels)},
    )

    args = TrainingArguments(
        output_dir=str(config.MODEL_DIR / "triage-checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=1,
        seed=SEED,
        report_to=[],
        use_cpu=not torch.backends.mps.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nBaseline evaluation before fine-tuning")
    baseline = trainer.evaluate()
    print({k: round(v, 4) for k, v in baseline.items() if isinstance(v, float)})

    print("\nFine-tuning")
    started = time.perf_counter()
    train_out = trainer.train()
    train_seconds = time.perf_counter() - started

    print("\nEvaluation after fine-tuning")
    final = trainer.evaluate()
    print({k: round(v, 4) for k, v in final.items() if isinstance(v, float)})

    # Per-class report and confusion matrix on the held-out test split
    preds_out = trainer.predict(tokenized["test"])
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true = preds_out.label_ids

    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0, digits=3
    )
    (config.ARTIFACT_DIR / "finetune_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    plot_confusion(
        confusion_matrix(y_true, y_pred, labels=range(len(labels))),
        labels,
        config.ARTIFACT_DIR / "finetune_confusion_matrix.png",
    )

    print("\nSaving model to", config.FINETUNED_TRIAGE_DIR)
    trainer.save_model(str(config.FINETUNED_TRIAGE_DIR))
    tokenizer.save_pretrained(str(config.FINETUNED_TRIAGE_DIR))

    summary = {
        "base_model": config.BASE_CLASSIFIER,
        "dataset": config.FINETUNE_DATASET,
        "train_records": len(raw["train"]),
        "test_records": len(raw["test"]),
        "num_classes": len(labels),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_seconds": round(train_seconds, 1),
        "train_loss": round(float(train_out.training_loss), 4),
        "baseline": {
            k.replace("eval_", ""): round(float(v), 4)
            for k, v in baseline.items()
            if k.startswith("eval_") and isinstance(v, float)
        },
        "finetuned": {
            k.replace("eval_", ""): round(float(v), 4)
            for k, v in final.items()
            if k.startswith("eval_") and isinstance(v, float)
        },
    }
    (config.ARTIFACT_DIR / "finetune_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="finetune-triage-distilbert"):
            mlflow.log_params(
                {
                    "base_model": config.BASE_CLASSIFIER,
                    "dataset": config.FINETUNE_DATASET,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_classes": len(labels),
                }
            )
            mlflow.log_metrics(
                {f"baseline_{k}": v for k, v in summary["baseline"].items()}
            )
            mlflow.log_metrics(
                {f"finetuned_{k}": v for k, v in summary["finetuned"].items()}
            )
            mlflow.log_metric("train_seconds", summary["train_seconds"])
            mlflow.log_artifact(
                str(config.ARTIFACT_DIR / "finetune_classification_report.txt")
            )
            mlflow.log_artifact(
                str(config.ARTIFACT_DIR / "finetune_confusion_matrix.png")
            )
        print("Logged run to MLflow")
    except Exception as exc:
        print("MLflow logging skipped:", type(exc).__name__, exc)

    print("\nDone.")
    print(f"  accuracy  {summary['baseline'].get('accuracy')} -> {summary['finetuned'].get('accuracy')}")
    print(f"  f1_macro  {summary['baseline'].get('f1_macro')} -> {summary['finetuned'].get('f1_macro')}")


if __name__ == "__main__":
    main()
