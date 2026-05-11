"""Module to extract threat indicators from uploaded log files and reports."""

import re


# Patterns for extracting indicators from text
PATTERNS = {
    "IP Address": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    "URL": r"https?://[^\s\"'<>]+",
    "Domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
    "CVE ID": r"CVE-\d{4}-\d{4,}",
    "File Hash": r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b",
    "Email Address": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}


def extract_indicators_from_text(text: str) -> list[dict]:
    """Extract all threat indicators from a text string."""
    results = []
    seen = set()

    for ind_type, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        for match in matches:
            if match not in seen:
                seen.add(match)
                results.append({"type": ind_type, "value": match})

    return results


def parse_uploaded_file(uploaded_file) -> str:
    """Read the content of an uploaded file (Streamlit UploadedFile)."""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return content
    except Exception as e:
        return f"Error reading file: {e}"
