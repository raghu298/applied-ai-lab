"""Sub-task 4 - Summarisation: condense the consultation into a handover note
the attending physician can read in a few seconds.

Category: Natural Language Processing
Model:    facebook/bart-large-cnn
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from llmops.metrics import tracked
from tasks import model_hub

MIN_WORDS_TO_SUMMARISE = 25


def _drop_incomplete_tail(text: str) -> str:
    """Remove a trailing fragment left behind when generation hits the length cap.

    The summariser stops at max_length, which frequently leaves a dangling
    clause such as "... a few steps . The". Anything after the last sentence
    terminator is discarded so the handover note always ends cleanly.
    """
    import re

    text = re.sub(r"\s+([.,;])", r"\1", text).strip()
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last > 0:
        return text[: last + 1].strip()
    return text


@tracked("4_clinical_summary", config.SUMMARISER_MODEL)
def summarise(transcript: str, clinical_note: str = "", condition: str = "") -> dict:
    """Produce an abstractive summary of the patient encounter."""
    # Design rule, adopted after the summariser dropped the assessment
    # sentence on one case and paraphrased the review marker on another:
    # the model only ever summarises what the PATIENT said; every clinical
    # fact (assessment, override, review status) is prepended by code after
    # summarisation, so its wording is identical on every run.
    parts = [f"The patient reports the following. {transcript.strip()}"]
    if clinical_note:
        parts.append(f"Recorded findings are as follows. {clinical_note}")
    source = " ".join(parts)

    words = len(source.split())
    if words < MIN_WORDS_TO_SUMMARISE:
        # Below this length an abstractive model tends to copy or hallucinate,
        # so the original text is passed through unchanged. The deterministic
        # clinical header is still applied, same as on the abstractive path.
        note = transcript.strip()
        if condition:
            note = f"The triage assessment for this patient is {condition}. {note}"
        return {
            "summary": note,
            "compression_ratio": 1.0,
            "source_words": words,
            "summary_words": len(note.split()),
            "abstractive": False,
            "confidence": 1.0,
        }

    # The band was widened when bart-large-cnn replaced distilbart: the larger
    # model spends tokens restating the assessment sentence, and the earlier
    # cap made it drop the recorded findings (the rash never reached the
    # dengue handover note).
    max_len = max(60, min(160, int(words * 0.7)))
    min_len = max(25, int(max_len * 0.45))

    out = model_hub.summariser()(
        source,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )[0]["summary_text"].strip()

    out = _drop_incomplete_tail(out)

    # Deterministic clinical header: identical wording on every run, immune
    # to whatever the summariser chose to keep.
    if condition:
        out = f"The triage assessment for this patient is {condition}. {out}"

    summary_words = len(out.split())
    ratio = summary_words / words if words else 1.0

    return {
        "summary": out,
        "compression_ratio": round(ratio, 4),
        "source_words": words,
        "summary_words": summary_words,
        "abstractive": True,
        # A summary between 20 and 60 percent of the source is the useful band.
        "confidence": round(1.0 - min(abs(ratio - 0.35) / 0.35, 1.0), 4),
    }
