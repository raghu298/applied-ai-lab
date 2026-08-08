"""Sub-task 2 - Text Classification: map the complaint to a likely condition
and an urgency band.

Category: Natural Language Processing
Model:    answerdotai/ModernBERT-base fine-tuned on gretelai/symptom_to_diagnosis
          (the fine-tuning deliverable of this assignment)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from llmops.metrics import tracked
from tasks import model_hub

ROUTING = {
    "EMERGENCY": "Route to emergency department, notify on-call physician now",
    "HIGH": "Book a same-day consultation with the relevant specialist",
    "ROUTINE": "Schedule a standard out-patient appointment within 7 days",
}


def _urgency(condition: str, transcript: str) -> tuple[str, str | None]:
    """Return the urgency band and, for emergencies, the red flag that fired.

    The trigger is surfaced in the interface so an emergency banner can lead
    with the actual reason for the routing instead of the model's (overridden
    and possibly wrong) condition label.
    """
    lowered = transcript.lower()
    for phrase in config.EMERGENCY_PHRASES:
        if phrase in lowered:
            return "EMERGENCY", phrase
    # Co-occurrence rules catch red-flag presentations whose words are
    # separated in the sentence ("pain ... in the center of my chest"),
    # which a literal phrase list cannot see.
    for group in config.EMERGENCY_COOCCURRENCE:
        if all(term in lowered for term in group):
            return "EMERGENCY", " with ".join(group)
    if condition.lower() in config.HIGH_URGENCY_CONDITIONS:
        return "HIGH", None
    return "ROUTINE", None


@tracked("2_triage_classification", "fine-tuned ModernBERT")
def classify(transcript: str, top_k: int = 3) -> dict:
    """Predict the most likely condition and derive a triage decision."""
    clf = model_hub.triage_classifier()
    if clf is None:
        raise RuntimeError(
            "Fine-tuned triage model not found. Run: "
            "python src/finetune/train_triage.py"
        )

    scores = clf(transcript, truncation=True, max_length=256)[0]
    ranked = sorted(scores, key=lambda d: d["score"], reverse=True)[:top_k]
    best = ranked[0]

    urgency, red_flag = _urgency(best["label"], transcript)
    # The EMERGENCY band is only ever reached through a red-flag phrase, never
    # from the classifier itself, so an emergency always means the rule layer
    # overrode whatever the model predicted.
    escalated = urgency == "EMERGENCY"
    review_required = float(best["score"]) < config.TRIAGE_REVIEW_THRESHOLD

    return {
        "condition": best["label"],
        "confidence": float(best["score"]),
        "alternatives": [
            {"condition": r["label"], "score": round(float(r["score"]), 4)}
            for r in ranked[1:]
        ],
        "urgency": urgency,
        "routing_action": ROUTING[urgency],
        "rule_escalated": escalated,
        "red_flag": red_flag,
        "review_required": review_required,
        "review_reason": (
            f"model confidence {float(best['score']):.0%} is below the "
            f"{config.TRIAGE_REVIEW_THRESHOLD:.0%} threshold"
            if review_required
            else None
        ),
    }


@tracked("2_patient_distress", config.SENTIMENT_MODEL)
def distress(transcript: str) -> dict:
    """Estimate patient distress, used to prioritise within an urgency band."""
    out = model_hub.sentiment()(transcript, truncation=True, max_length=256)[0]
    negative = out["label"].upper() == "NEGATIVE"
    return {
        "distressed": negative,
        "label": "Distressed" if negative else "Calm",
        "confidence": float(out["score"]),
    }
