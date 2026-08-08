"""Central configuration for the Patient Voice Triage and Clinical Support Assistant."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """Read a .env file into the process environment.

    Written by hand rather than pulling in python-dotenv, to keep the
    dependency list to what the models actually need. Values already present in
    the environment win, so an exported variable can override the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env(ROOT / ".env")

# Token presence is reported by the application; it is never printed.
HF_TOKEN_PRESENT = bool(os.environ.get("HF_TOKEN"))

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

for _d in (DATA_DIR, ARTIFACT_DIR, MODEL_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Sub-task 1 (Speech Recognition): patient voice note to text.
# Chosen by two rounds of measurement on real recordings. whisper-base.en
# garbled the clinically relevant phrase "burning feeling while peeing" into
# "burning, feeling, oil pain"; whisper-small.en transcribed it correctly.
# whisper-large-v3-turbo then matched small.en on every test clip while
# running about 40 percent faster warm (its decoder is only four layers),
# and it is multilingual, so accented or code-switched speech does not
# derail it the way it would an English-only checkpoint.
ASR_MODEL = "openai/whisper-large-v3-turbo"

# Sub-task 2 (NLP): triage classification, fine-tuned in this project.
# ModernBERT-base replaced distilbert-base-uncased after a controlled
# comparison on the same data, seed and epochs: distilbert 92.0 percent
# accuracy, roberta-base 92.9, ModernBERT-base 95.8 (macro F1 0.957).
BASE_CLASSIFIER = "answerdotai/ModernBERT-base"
FINETUNED_TRIAGE_DIR = MODEL_DIR / "triage-modernbert"

# Sub-task 3 (NLP): biomedical named entity recognition
NER_MODEL = "d4data/biomedical-ner-all"

# Sub-task 4 (NLP): clinical summarisation.
# bart-large-cnn replaced distilbart-cnn-12-6 after a side-by-side test:
# distilbart reordered the complaint (opening with the rash that appeared
# last), bart-large preserved the clinical sequence of events.
SUMMARISER_MODEL = "facebook/bart-large-cnn"

# Sub-task 5 (NLP): question answering over the hospital knowledge base.
# roberta-base-squad2 is trained on SQuAD 2.0, which includes unanswerable
# questions, so it scores nonsense spans much lower than the SQuAD 1.1
# distilbert did. That makes the low-confidence fallback to the full
# approved passage fire more reliably on out-of-scope questions.
QA_MODEL = "deepset/roberta-base-squad2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Sub-task 6 (NLP): patient guidance text generation.
# Qwen2.5-1.5B-Instruct is an instruction-tuned SLM. It was chosen over
# flan-t5-base, which paraphrased the prompt instead of writing a reply, and
# over Qwen2.5-0.5B-Instruct, which suggested specific medicines and so failed
# the safety requirement that the assistant must not prescribe.
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Supporting model used for patient distress detection
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# ---------------------------------------------------------------------------
# Cloud-lite mode: Streamlit Community Cloud allows about 2.7 GB of RAM, far
# below what the full model set needs, so HEALTH_ASSIST_LITE=1 swaps every
# stage to a small model that fits. The pipeline, rule layer, thresholds and
# safety guardrails are identical; only model capacity changes. The lite
# triage checkpoint (bert-mini, 11M parameters) is small enough to live in
# the git repository, so the cloud app needs no training step at boot.
# ---------------------------------------------------------------------------
LITE_MODE = os.environ.get("HEALTH_ASSIST_LITE") == "1"
if LITE_MODE:
    # whisper-tiny, not whisper-tiny.en. The multilingual checkpoint has the
    # same 39M parameters and the same memory cost as the English-only one,
    # so multilingual intake is free here. It matters because an English-only
    # checkpoint does not refuse foreign speech, it hallucinates confident
    # English from it, and that transcript would flow into triage unflagged.
    ASR_MODEL = "openai/whisper-tiny"
    FINETUNED_TRIAGE_DIR = ROOT / "models_lite" / "triage-mini"
    SUMMARISER_MODEL = "Falconsai/medical_summarization"
    QA_MODEL = "distilbert-base-cased-distilled-squad"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # No generative reply model in lite mode: the patient message is composed
    # by a deterministic template instead (see subtask6_guidance).
    GENERATOR_MODEL = None

# Fine-tuning dataset (healthcare domain)
FINETUNE_DATASET = "gretelai/symptom_to_diagnosis"

# LLMOps
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "health-assist-llmops"
METRICS_LOG = ARTIFACT_DIR / "llmops_metrics.jsonl"

# Triage urgency policy. These are the conditions within the 22 classes the
# fine-tuned model can predict that warrant a same-day specialist review rather
# than a routine appointment.
HIGH_URGENCY_CONDITIONS = {
    "pneumonia",
    "jaundice",
    "dengue",
    "malaria",
    "typhoid",
    "diabetes",
    "hypertension",
    "bronchial asthma",
    "drug reaction",
}

# A prediction below this probability is not trusted on its own and the case is
# marked for human review by a triage nurse. Recalibrated for the ModernBERT
# classifier, whose mean confidence on the test split is 0.97 (the earlier
# distilbert averaged 0.54, hence its lower 0.30 threshold): at 0.90 the
# system automates 93.9 percent of cases at 98.0 percent accuracy, and the
# flagged cases would only have been 84.6 percent accurate.
TRIAGE_REVIEW_THRESHOLD = 0.90

# Red-flag phrases that force an emergency route regardless of model output
EMERGENCY_PHRASES = [
    "chest pain",
    "heart attack",
    "cannot breathe",
    "can't breathe",
    "unconscious",
    "severe bleeding",
    "suicidal",
    "stroke",
    "slurred speech",
    "blue lips",
]

# Word co-occurrence red flags. A phrase list alone missed a textbook cardiac
# complaint phrased as "pain or pressure in the center of my chest" - the
# literal substring "chest pain" never appears. Each group below forces the
# emergency route when ALL of its terms appear anywhere in the complaint.
# Over-triggering is the accepted direction: a needless escalation is
# inefficient, a missed emergency is catastrophic.
EMERGENCY_COOCCURRENCE = [
    ("chest", "pain"),
    ("chest", "pressure"),
    ("chest", "tight"),
    ("chest", "crushing"),
    ("chest", "heaviness"),
    ("chest", "squeez"),
]
