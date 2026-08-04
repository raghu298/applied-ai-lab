"""Sub-task 5 - Question Answering: answer a patient question from the hospital
knowledge base.

Category: Natural Language Processing
Models:   sentence-transformers/all-MiniLM-L6-v2 for passage retrieval
          distilbert-base-cased-distilled-squad for extractive answering

The retrieval step keeps the answer grounded in an approved hospital document,
which is the control that stops the assistant from inventing medical advice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from llmops.metrics import tracked
from tasks import model_hub

KB_PATH = config.DATA_DIR / "knowledge_base.json"

_KB: list[dict] | None = None
_KB_VECTORS = None

# Below this retrieval score the question is treated as out of scope.
RETRIEVAL_FLOOR = 0.25
ANSWER_FLOOR = 0.10


def load_kb() -> list[dict]:
    global _KB
    if _KB is None:
        with open(KB_PATH, encoding="utf-8") as fh:
            _KB = json.load(fh)
    return _KB


def _kb_vectors():
    global _KB_VECTORS
    if _KB_VECTORS is None:
        kb = load_kb()
        texts = [f"{d['title']}. {d['text']}" for d in kb]
        _KB_VECTORS = model_hub.embedder().encode(texts, normalize_embeddings=True)
    return _KB_VECTORS


def _containing_sentence(context: str, span: str) -> str:
    """Return the full sentence from the approved text that holds the span."""
    import re

    if not span:
        return span
    sentences = re.split(r"(?<=[.!?])\s+", context)
    for sentence in sentences:
        if span in sentence:
            return sentence.strip()
    return span


@tracked("5_question_answering", "MiniLM retriever + DistilBERT SQuAD")
def answer_question(question: str, top_k: int = 3) -> dict:
    """Retrieve the most relevant passages, then extract a span answer."""
    kb = load_kb()
    q_vec = model_hub.embedder().encode([question], normalize_embeddings=True)[0]
    sims = _kb_vectors() @ q_vec

    order = np.argsort(-sims)[:top_k]
    retrieved = [
        {
            "id": kb[i]["id"],
            "title": kb[i]["title"],
            "text": kb[i]["text"],
            "score": round(float(sims[i]), 4),
        }
        for i in order
    ]
    best_passage = retrieved[0]

    if best_passage["score"] < RETRIEVAL_FLOOR:
        return {
            "answer": (
                "That question is outside the hospital knowledge base. "
                "Please speak to the help desk on extension 100."
            ),
            "in_scope": False,
            "source_id": None,
            "source_title": None,
            "retrieval_score": best_passage["score"],
            "retrieved": retrieved,
            "confidence": 0.0,
        }

    context = " ".join(p["text"] for p in retrieved)
    out = model_hub.qa()(question=question, context=context)
    span = out["answer"].strip()
    span_score = float(out["score"])

    if span_score < ANSWER_FLOOR or not span:
        # Retrieval succeeded but span extraction was weak, so the whole
        # approved passage is returned rather than a low-quality fragment.
        answer = best_passage["text"]
        confidence = best_passage["score"] * 0.5
    else:
        # An extracted span is often a bare fragment such as "8 AM to 10 PM".
        # Returning the sentence that contains it keeps the answer grounded in
        # the approved text while staying readable to the patient.
        answer = _containing_sentence(context, span)
        confidence = span_score

    return {
        "answer": answer,
        "in_scope": True,
        "source_id": best_passage["id"],
        "source_title": best_passage["title"],
        "retrieval_score": best_passage["score"],
        "span_score": round(span_score, 4),
        "retrieved": retrieved,
        "confidence": round(float(confidence), 4),
    }
