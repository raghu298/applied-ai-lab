# Patient Voice Triage and Clinical Support Assistant

Assignment II for CCZG506 (API-driven Cloud Native Solutions).

Domain: Healthcare. Categories: Speech Recognition and Natural Language
Processing. Six sub-tasks operate on one patient episode.

## Sub-tasks and models

| # | Sub-task | Category | Model |
|---|----------|----------|-------|
| 1 | Automatic Speech Recognition | Speech Recognition | openai/whisper-large-v3-turbo |
| 2 | Text Classification | NLP | answerdotai/ModernBERT-base, fine-tuned here |
| 3 | Named Entity Recognition | NLP | d4data/biomedical-ner-all |
| 4 | Summarisation | NLP | facebook/bart-large-cnn |
| 5 | Question Answering | NLP | all-mpnet-base-v2 + deepset/roberta-base-squad2 |
| 6 | Text Generation | NLP | Qwen/Qwen2.5-1.5B-Instruct |

Model choices were settled by measurement, not by size: whisper-base.en
garbled a symptom phrase on a real voice note, small.en fixed it, and
large-v3-turbo then matched small.en on every clip while running about 40
percent faster warm, with multilingual robustness as the tie-breaker;
ModernBERT-base beat distilbert (92.0) and roberta-base (92.9) on the same
data and seed; bart-large-cnn preserved symptom order where distilbart
reordered it; roberta-base-squad2 scores out-of-scope questions low enough
to trigger the approved-passage fallback. Larger candidates that lost on
evidence were rejected: a DeBERTa medical NER model produced noisier
entities than the current one, and Qwen3-1.7B leaked think-tags and edged
toward stating a diagnosis.

## Pipeline

```
voice note
  -> [1] Speech Recognition       transcript
  -> [2] Text Classification      condition + urgency band
  -> [3] Named Entity Recognition structured clinical findings
  -> [5] Question Answering       governing hospital policy
  -> [4] Summarisation            physician handover note
  -> [6] Text Generation          reply sent to the patient
```

## Results

- Fine-tuned triage classifier: accuracy 3.8 percent to 97.2 percent,
  macro F1 0.005 to 0.973, over 22 conditions, about 3 minutes on Apple
  silicon (MPS).
- Human-review threshold calibrated to 0.90: 93.9 percent of cases handled
  automatically at 98.0 percent accuracy; the flagged cases would only have
  been 84.6 percent accurate.
- Seven LLMOps metrics instrumented, mirrored into MLflow.

## Setup

Two things are deliberately not in this repository: the `.env` holding the
Hugging Face token, and the fine-tuned model weights. Both are regenerated
locally by the steps below.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Token, only needed for the hosted Inference API tab. A read-scoped
# token is sufficient. Every other part of the project runs without one.
cp .env.example .env
# then edit .env and set HF_TOKEN
```

The 574 MB fine-tuned checkpoint exceeds GitHub's 100 MB file limit, so it is
excluded. Running the fine-tuning step below reproduces it in about three
minutes, and the reported metrics are seeded with a fixed random seed.

## Run

```
# Fine-tune the triage model (about three minutes on Apple silicon)
.venv/bin/python src/finetune/train_triage.py

# Calibrate the human-review threshold
.venv/bin/python src/llmops/calibrate_threshold.py

# Launch the application
.venv/bin/streamlit run src/app.py

# One episode from the command line
.venv/bin/python src/pipeline.py

# MLflow dashboard
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Layout

```
src/
  app.py                    Streamlit interface
  pipeline.py               orchestrates the six sub-tasks
  config.py                 model names, thresholds, policy tables
  build_report.py           generates the submission document
  capture_screenshots.py    drives the app to capture figures
  tasks/                    one module per sub-task
  llmops/                   instrumentation and threshold calibration
  finetune/train_triage.py  fine-tuning script
data/
  knowledge_base.json       15 approved hospital passages
  audio/                    sample patient voice notes
models/triage-modernbert/   the fine-tuned model
artifacts/                  metrics, charts, evaluation reports
docs/                       report and screenshots
```

## Safety design

Red-flag phrases such as chest pain force an emergency route regardless of the
model prediction. Predictions below the calibrated confidence threshold are
queued for a triage nurse. Patient questions are answered only by extraction
from approved hospital passages. The generation stage is instructed never to
name a medication or state a definite diagnosis, and every message carries a
disclaimer. All models run locally, so no patient text leaves the machine.
