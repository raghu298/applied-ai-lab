"""Hugging Face API access layer (requirement 6).

Three distinct uses of the Hugging Face API are demonstrated.

1. Hub REST API. Model metadata for every model backing the six sub-tasks is
   retrieved live from huggingface.co. This is what populates the model
   provenance table in the application, and it needs no authentication.

2. transformers model API. Every sub-task resolves and downloads its model
   from the Hub on first use through the transformers library, then runs it
   locally. This is the default execution path for the whole application.

3. Serverless Inference API. The same sub-task can be served either locally
   through the transformers library or remotely through the hosted inference
   endpoint. ``run_remote`` performs the remote call when a valid token is
   present and reports cleanly when it is not, so the application never depends
   on network availability during a demonstration.

The local path is the default. Patient audio and transcripts stay on the
machine, which is the privacy position the project takes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from llmops.metrics import tracked

MODELS_IN_USE = [
    ("1. Speech Recognition", config.ASR_MODEL),
    ("2. Text Classification (base)", config.BASE_CLASSIFIER),
    ("3. Named Entity Recognition", config.NER_MODEL),
    ("4. Summarisation", config.SUMMARISER_MODEL),
    ("5. Question Answering", config.QA_MODEL),
    ("5. Retrieval", config.EMBEDDING_MODEL),
    ("6. Text Generation", config.GENERATOR_MODEL),
]


def _fmt_downloads(n) -> str:
    if not n:
        return "not reported"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


@tracked("api_hub_metadata", "huggingface_hub REST API")
def fetch_model_metadata() -> dict:
    """Retrieve live metadata from the Hub for every model the project uses."""
    from huggingface_hub import HfApi

    api = HfApi()
    rows, failures = [], 0

    for role, model_id in MODELS_IN_USE:
        try:
            info = api.model_info(model_id)
            rows.append(
                {
                    "sub_task": role,
                    "model_id": model_id,
                    "pipeline_tag": info.pipeline_tag or "not declared",
                    "library": info.library_name or "transformers",
                    "downloads_30d": _fmt_downloads(info.downloads),
                    "likes": info.likes or 0,
                    "last_modified": str(info.last_modified)[:10]
                    if info.last_modified
                    else "unknown",
                }
            )
        except Exception as exc:
            failures += 1
            rows.append(
                {
                    "sub_task": role,
                    "model_id": model_id,
                    "pipeline_tag": f"lookup failed: {type(exc).__name__}",
                    "library": "-",
                    "downloads_30d": "-",
                    "likes": "-",
                    "last_modified": "-",
                }
            )

    return {
        "models": rows,
        "retrieved": len(rows) - failures,
        "failed": failures,
        "confidence": 1.0 if failures == 0 else 0.0,
    }


def token_status() -> dict:
    """Report whether a usable Inference API token is configured."""
    try:
        from huggingface_hub import HfApi, get_token

        token = get_token()
        if not token:
            return {"present": False, "valid": False,
                    "detail": "No Hugging Face token found on this machine."}
        try:
            who = HfApi().whoami(token=token)
            return {"present": True, "valid": True,
                    "detail": f"Authenticated as {who.get('name')}."}
        except Exception as exc:
            return {
                "present": True,
                "valid": False,
                "detail": f"A token is stored but it was rejected "
                          f"({type(exc).__name__}). Run 'hf auth login' to "
                          f"refresh it.",
            }
    except Exception as exc:
        return {"present": False, "valid": False,
                "detail": f"{type(exc).__name__}: {exc}"}


# Canonical Hub IDs for the two sub-task models that the hosted hf-inference
# provider still serves. The provider requires the fully qualified name, which
# is why the organisation prefix is spelled out here.
REMOTE_SUMMARISER = "sshleifer/distilbart-cnn-12-6"
REMOTE_QA = "distilbert/distilbert-base-cased-distilled-squad"


@tracked("api_remote_summarise", "HF Serverless Inference API")
def remote_summarise(text: str) -> dict:
    """Run sub-task 4 through the hosted Inference API instead of locally."""
    status = token_status()
    if not status["valid"]:
        return {"available": False, "reason": status["detail"],
                "model_id": REMOTE_SUMMARISER, "confidence": 0.0}
    try:
        from huggingface_hub import InferenceClient

        started = time.perf_counter()
        out = InferenceClient().summarization(text, model=REMOTE_SUMMARISER)
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "available": True,
            "model_id": REMOTE_SUMMARISER,
            "summary": out.summary_text.strip(),
            "latency_ms": round(elapsed, 1),
            "confidence": 1.0,
        }
    except Exception as exc:
        return {"available": False,
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                "model_id": REMOTE_SUMMARISER, "confidence": 0.0}


@tracked("api_remote_qa", "HF Serverless Inference API")
def remote_question_answering(question: str, context: str) -> dict:
    """Run sub-task 5 through the hosted Inference API instead of locally."""
    status = token_status()
    if not status["valid"]:
        return {"available": False, "reason": status["detail"],
                "model_id": REMOTE_QA, "confidence": 0.0}
    try:
        from huggingface_hub import InferenceClient

        started = time.perf_counter()
        out = InferenceClient().question_answering(
            question=question, context=context, model=REMOTE_QA
        )
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "available": True,
            "model_id": REMOTE_QA,
            "answer": out.answer,
            "score": round(float(out.score), 4),
            "latency_ms": round(elapsed, 1),
            "confidence": float(out.score),
        }
    except Exception as exc:
        return {"available": False,
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                "model_id": REMOTE_QA, "confidence": 0.0}


def compare_local_and_remote(text: str) -> dict:
    """Serve the same sub-task both ways and compare.

    The same model, sshleifer/distilbart-cnn-12-6, is run locally through the
    transformers library and remotely through the hosted endpoint. This is the
    measurement behind the deployment argument in the report: the hosted path
    removes the local compute cost but adds network round-trip and a cold-start
    penalty, and it sends patient text off the machine.
    """
    from tasks import subtask4_summary

    started = time.perf_counter()
    local = subtask4_summary.summarise(text)
    local_ms = (time.perf_counter() - started) * 1000

    remote = remote_summarise(text)

    return {
        "model_id": REMOTE_SUMMARISER,
        "local": {
            "summary": local["summary"],
            "latency_ms": round(local_ms, 1),
        },
        "remote": remote,
        "delta_ms": round(remote["latency_ms"] - local_ms, 1)
        if remote.get("available")
        else None,
    }


if __name__ == "__main__":
    meta = fetch_model_metadata()
    print(f"Retrieved {meta['retrieved']} of {len(MODELS_IN_USE)} models\n")
    for row in meta["models"]:
        print(f"  {row['sub_task']:32} {row['model_id']}")
        print(f"  {'':32} task={row['pipeline_tag']}  "
              f"downloads={row['downloads_30d']}  updated={row['last_modified']}")
    print("\nInference API token:", token_status()["detail"])

    demo = (
        "The patient reports a high fever for four days with severe headache, "
        "pain behind the eyes, joint and muscle ache, and a rash that appeared "
        "on the arms yesterday. The patient feels weak and nauseous."
    )
    print("\nSame model, both execution paths:")
    cmp = compare_local_and_remote(demo)
    print(f"  model  : {cmp['model_id']}")
    print(f"  local  : {cmp['local']['latency_ms']} ms")
    if cmp["remote"].get("available"):
        print(f"  remote : {cmp['remote']['latency_ms']} ms "
              f"(difference {cmp['delta_ms']} ms)")
    else:
        print(f"  remote : unavailable, {cmp['remote']['reason']}")
