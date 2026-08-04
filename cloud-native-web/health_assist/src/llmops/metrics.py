"""LLMOps instrumentation layer.

Every sub-task in the application is wrapped by the ``tracked`` decorator. The
decorator records one observation per invocation into a JSON-lines store and
mirrors the numeric metrics into MLflow, so the run history survives restarts of
the Streamlit process and can be inspected on the MLflow dashboard.

Metrics captured per invocation:
    1. latency_ms          wall-clock time of the sub-task
    2. input_tokens        size of the request handed to the model
    3. output_tokens       size of the response produced by the model
    4. throughput_tps      output tokens per second
    5. confidence          model self-reported probability, where applicable
    6. success             1 on completion, 0 on exception
    7. low_confidence      1 when confidence falls under the review threshold
"""

from __future__ import annotations

import functools
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import METRICS_LOG, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

_LOCK = threading.Lock()
_MLFLOW_READY = False

# Generic instrumentation flag applied uniformly across all sub-tasks. This is
# deliberately not the triage review threshold (config.TRIAGE_REVIEW_THRESHOLD),
# which is calibrated per classifier: stages such as the QA reader report
# confidences on a different scale, so one shared 0.30 flag keeps the
# low_confidence metric comparable across stages and across model swaps.
CONFIDENCE_THRESHOLD = 0.30


def _approx_tokens(value: Any) -> int:
    """Whitespace token count, used as a lightweight cost proxy."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_approx_tokens(v) for v in value)
    if isinstance(value, dict):
        return sum(_approx_tokens(v) for v in value.values())
    if isinstance(value, (int, float)):
        return 1
    return len(str(value).split())


def _ensure_mlflow():
    global _MLFLOW_READY
    if _MLFLOW_READY:
        return True
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        _MLFLOW_READY = True
    except Exception:
        _MLFLOW_READY = False
    return _MLFLOW_READY


def write_record(record: dict) -> None:
    """Append one observation to the JSON-lines metric store."""
    with _LOCK:
        with open(METRICS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def log_to_mlflow(task: str, record: dict) -> None:
    if not _ensure_mlflow():
        return
    try:
        import mlflow

        with mlflow.start_run(run_name=f"{task}-{record['timestamp']}", nested=False):
            mlflow.set_tag("subtask", task)
            mlflow.log_metrics(
                {
                    "latency_ms": record["latency_ms"],
                    "input_tokens": record["input_tokens"],
                    "output_tokens": record["output_tokens"],
                    "throughput_tps": record["throughput_tps"],
                    "confidence": record["confidence"],
                    "success": record["success"],
                    "low_confidence": record["low_confidence"],
                }
            )
    except Exception:
        # Metric logging must never break the user-facing request.
        pass


def tracked(task: str, model_name: str = "") -> Callable:
    """Instrument a sub-task function with the LLMOps metric set."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            success, confidence, result, error = 1, 0.0, None, ""
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    confidence = float(result.get("confidence", 0.0) or 0.0)
            except Exception as exc:
                success, error = 0, f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                out_tokens = _approx_tokens(result)
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "task": task,
                    "model": model_name,
                    "latency_ms": round(elapsed_ms, 2),
                    "input_tokens": _approx_tokens(args) + _approx_tokens(kwargs),
                    "output_tokens": out_tokens,
                    # Throughput is only meaningful for a real model call. The
                    # manual-entry path returns in microseconds, which would
                    # otherwise produce a meaningless millions-of-tokens rate.
                    "throughput_tps": round(out_tokens / (elapsed_ms / 1000.0), 2)
                    if elapsed_ms >= 1.0
                    else 0.0,
                    "confidence": round(confidence, 4),
                    "success": success,
                    "low_confidence": int(0 < confidence < CONFIDENCE_THRESHOLD),
                    "error": error,
                }
                write_record(record)
                log_to_mlflow(task, record)
            return result

        return wrapper

    return decorator


def load_metrics():
    """Return the metric store as a pandas DataFrame."""
    import pandas as pd

    if not METRICS_LOG.exists():
        return pd.DataFrame()
    rows = []
    with open(METRICS_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def summarise_metrics():
    """Aggregate the seven runtime metrics by sub-task."""
    df = load_metrics()
    if df.empty:
        return df
    grouped = (
        df.groupby("task")
        .agg(
            invocations=("task", "count"),
            avg_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
            avg_throughput_tps=("throughput_tps", "mean"),
            avg_confidence=("confidence", "mean"),
            success_rate=("success", "mean"),
            low_confidence_rate=("low_confidence", "mean"),
            total_output_tokens=("output_tokens", "sum"),
        )
        .reset_index()
    )
    for col in grouped.columns:
        if col != "task":
            grouped[col] = grouped[col].astype(float).round(3)
    return grouped
