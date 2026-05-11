"""Logging module to save analysis results to CSV."""

import csv
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "analysis_history.csv")

FIELDNAMES = [
    "timestamp",
    "indicator_type",
    "indicator_value",
    "classification",
    "risk_score",
    "reason",
    "recommended_action",
]


def ensure_log_file():
    """Create the log file with headers if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_result(result: dict):
    """Append an analysis result to the CSV log."""
    ensure_log_file()
    row = {
        "timestamp": datetime.now().isoformat(),
        "indicator_type": result.get("indicator_type", ""),
        "indicator_value": result.get("indicator_value", ""),
        "classification": result.get("classification", ""),
        "risk_score": result.get("risk_score", ""),
        "reason": result.get("reason", ""),
        "recommended_action": result.get("recommended_action", ""),
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def get_history() -> list[dict]:
    """Read the full analysis history from CSV."""
    ensure_log_file()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
