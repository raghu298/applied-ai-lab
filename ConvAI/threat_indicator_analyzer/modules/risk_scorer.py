"""Risk scoring utilities."""


def get_risk_color(risk_score: int) -> str:
    """Return a color based on risk score for UI display."""
    if risk_score >= 80:
        return "🔴"
    elif risk_score >= 50:
        return "🟠"
    elif risk_score >= 25:
        return "🟡"
    return "🟢"


def get_classification_color(classification: str) -> str:
    """Return display styling for classification."""
    colors = {
        "Critical": "🔴 Critical",
        "Malicious": "🔴 Malicious",
        "Suspicious": "🟠 Suspicious",
        "Safe": "🟢 Safe",
    }
    return colors.get(classification, classification)
