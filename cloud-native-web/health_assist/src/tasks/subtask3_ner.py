"""Sub-task 3 - Named Entity Recognition: pull structured clinical facts out of
the free-text complaint.

Category: Natural Language Processing
Model:    d4data/biomedical-ner-all
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from llmops.metrics import tracked
from tasks import model_hub

# Entity groups the triage desk actually acts on, mapped to display names.
GROUPS_OF_INTEREST = {
    "Sign_symptom": "Symptom",
    "Disease_disorder": "Condition",
    "Biological_structure": "Body site",
    "Severity": "Severity",
    "Duration": "Duration",
    "Frequency": "Frequency",
    "Medication": "Medication",
    "Age": "Age",
    "Sex": "Sex",
    "Lab_value": "Lab value",
    "Detailed_description": "Descriptor",
}


@tracked("3_clinical_ner", config.NER_MODEL)
def extract_entities(transcript: str, min_score: float = 0.35) -> dict:
    """Return de-duplicated clinical entities grouped by type."""
    raw = model_hub.ner()(transcript)

    grouped: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    scores = []

    for ent in raw:
        score = float(ent["score"])
        if score < min_score:
            continue
        group = ent["entity_group"]
        display = GROUPS_OF_INTEREST.get(group)
        if display is None:
            continue
        text = ent["word"].strip(" ,.;:")
        # Drop wordpiece debris the aggregator sometimes leaks on long
        # sentences ("##useous", "na", "##head") - fragments, not findings.
        if not text or "##" in text or len(text) < 3:
            continue
        key = (display, text.lower())
        if key in seen:
            continue
        seen.add(key)
        scores.append(score)
        grouped.setdefault(display, []).append(
            {"text": text, "score": round(score, 4)}
        )

    for items in grouped.values():
        items.sort(key=lambda d: d["score"], reverse=True)

    total = sum(len(v) for v in grouped.values())
    return {
        "entities": grouped,
        "entity_count": total,
        "symptom_count": len(grouped.get("Symptom", [])),
        "confidence": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


def as_clinical_note(entities: dict) -> str:
    """Flatten the extracted entities into a single structured line."""
    parts = []
    for label, items in entities.get("entities", {}).items():
        values = ", ".join(i["text"] for i in items)
        parts.append(f"{label}: {values}")
    return " | ".join(parts) if parts else "No structured entities detected."
