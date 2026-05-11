"""Threat analysis module using rule-based logic and optional LLM API."""

import os
import json
import re

try:
    import requests as _requests
except ImportError:
    _requests = None


# Suspicious / malicious keyword lists
MALICIOUS_KEYWORDS = [
    "phishing", "malware", "trojan", "ransomware", "exploit", "botnet",
    "keylogger", "rootkit", "backdoor", "c2", "command-and-control",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "bank", "paypal", "alert", "suspend", "unusual", "click",
]

KNOWN_MALICIOUS_IPS = [
    "185.220.101.45", "45.33.32.156", "192.168.1.100",
]

KNOWN_MALICIOUS_DOMAINS = [
    "fake-update-security.com", "secure-bank-login.verify-now.com",
]

HIGH_SEVERITY_CVE_PATTERNS = ["2023", "2024", "2025", "2026"]


def _keyword_score(value: str, keywords: list[str]) -> int:
    value_lower = value.lower()
    return sum(1 for kw in keywords if kw in value_lower)


def analyze_ip(value: str) -> dict:
    """Analyze an IP address for threats."""
    if value in KNOWN_MALICIOUS_IPS:
        return {
            "classification": "Malicious",
            "risk_score": 90,
            "reason": f"IP {value} is found in known malicious IP database.",
            "recommended_action": "Block the IP immediately and investigate related traffic.",
        }

    # Private IP ranges
    if value.startswith(("10.", "172.16.", "192.168.")):
        return {
            "classification": "Safe",
            "risk_score": 10,
            "reason": f"IP {value} is a private/internal address.",
            "recommended_action": "No action needed for internal IPs.",
        }

    return {
        "classification": "Suspicious",
        "risk_score": 45,
        "reason": f"IP {value} is not in known threat lists but is external.",
        "recommended_action": "Monitor traffic from this IP and check threat intelligence feeds.",
    }


def analyze_url(value: str) -> dict:
    """Analyze a URL for threats."""
    mal_score = _keyword_score(value, MALICIOUS_KEYWORDS)
    sus_score = _keyword_score(value, SUSPICIOUS_KEYWORDS)

    if mal_score >= 2 or (mal_score >= 1 and sus_score >= 2):
        return {
            "classification": "Critical",
            "risk_score": 95,
            "reason": f"URL contains multiple malicious indicators: {value}",
            "recommended_action": "Block the URL immediately, notify affected users, and investigate.",
        }
    if sus_score >= 3:
        return {
            "classification": "Malicious",
            "risk_score": 85,
            "reason": f"URL contains phishing-style keywords: {value}",
            "recommended_action": "Block the URL and notify affected users.",
        }
    if sus_score >= 1:
        return {
            "classification": "Suspicious",
            "risk_score": 55,
            "reason": f"URL contains some suspicious keywords.",
            "recommended_action": "Investigate the URL further and monitor access logs.",
        }

    return {
        "classification": "Safe",
        "risk_score": 15,
        "reason": "URL does not match known threat patterns.",
        "recommended_action": "No immediate action needed.",
    }


def analyze_domain(value: str) -> dict:
    """Analyze a domain for threats."""
    if value in KNOWN_MALICIOUS_DOMAINS:
        return {
            "classification": "Malicious",
            "risk_score": 90,
            "reason": f"Domain {value} is in the known malicious domain list.",
            "recommended_action": "Block the domain and investigate DNS queries.",
        }

    sus_score = _keyword_score(value, SUSPICIOUS_KEYWORDS)
    if sus_score >= 2:
        return {
            "classification": "Suspicious",
            "risk_score": 60,
            "reason": f"Domain contains suspicious keywords.",
            "recommended_action": "Monitor and investigate the domain.",
        }

    return {
        "classification": "Safe",
        "risk_score": 10,
        "reason": "Domain does not match known threat patterns.",
        "recommended_action": "No action needed.",
    }


def analyze_cve(value: str) -> dict:
    """Analyze a CVE ID."""
    year_match = re.search(r"CVE-(\d{4})", value, re.IGNORECASE)
    year = year_match.group(1) if year_match else ""

    if year in HIGH_SEVERITY_CVE_PATTERNS:
        return {
            "classification": "Critical",
            "risk_score": 85,
            "reason": f"{value} is a recent CVE that may be actively exploited.",
            "recommended_action": "Patch affected systems immediately and check for indicators of compromise.",
        }

    return {
        "classification": "Suspicious",
        "risk_score": 50,
        "reason": f"{value} is an older CVE. Check if systems are patched.",
        "recommended_action": "Verify patch status for this vulnerability.",
    }


def analyze_file_hash(value: str) -> dict:
    """Analyze a file hash."""
    known_malicious_hashes = ["44d88612fea8a8f36de82e1278abb02f"]
    if value.lower() in known_malicious_hashes:
        return {
            "classification": "Malicious",
            "risk_score": 95,
            "reason": f"Hash {value} matches known malware signature (EICAR test).",
            "recommended_action": "Quarantine the file and scan the system.",
        }

    return {
        "classification": "Suspicious",
        "risk_score": 40,
        "reason": "Hash not found in known malware databases. Could be unknown.",
        "recommended_action": "Submit to a sandbox for analysis (e.g., VirusTotal).",
    }


def analyze_email(value: str) -> dict:
    """Analyze an email address for threats."""
    sus_score = _keyword_score(value, SUSPICIOUS_KEYWORDS)
    domain = value.split("@")[-1] if "@" in value else ""

    if domain in KNOWN_MALICIOUS_DOMAINS or sus_score >= 2:
        return {
            "classification": "Malicious",
            "risk_score": 85,
            "reason": f"Email {value} uses a suspicious/phishing domain.",
            "recommended_action": "Block the sender and warn users about phishing.",
        }

    if sus_score >= 1:
        return {
            "classification": "Suspicious",
            "risk_score": 50,
            "reason": f"Email {value} contains suspicious keywords.",
            "recommended_action": "Verify the sender before taking any action.",
        }

    return {
        "classification": "Safe",
        "risk_score": 10,
        "reason": "Email does not match known threat patterns.",
        "recommended_action": "No action needed.",
    }


ANALYZERS = {
    "IP Address": analyze_ip,
    "URL": analyze_url,
    "Domain": analyze_domain,
    "CVE ID": analyze_cve,
    "File Hash": analyze_file_hash,
    "Email Address": analyze_email,
}


def analyze_indicator(indicator_type: str, value: str) -> dict:
    """Analyze a threat indicator and return classification results."""
    analyzer = ANALYZERS.get(indicator_type)
    if analyzer:
        result = analyzer(value.strip())
    else:
        result = {
            "classification": "Suspicious",
            "risk_score": 50,
            "reason": "Unknown indicator type — manual review recommended.",
            "recommended_action": "Review the indicator manually.",
        }

    result["indicator_type"] = indicator_type
    result["indicator_value"] = value.strip()
    return result


def analyze_with_llm(indicator_type: str, value: str) -> dict | None:
    """Optionally analyze using an LLM API via OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not _requests:
        return None

    prompt = (
        f"Analyze this cybersecurity threat indicator.\n"
        f"Type: {indicator_type}\nValue: {value}\n\n"
        f"Classify as: Safe, Suspicious, Malicious, or Critical.\n"
        f"Give a risk score (0-100), reason, and recommended action.\n"
        f"Respond in JSON format with keys: classification, risk_score, reason, recommended_action"
    )

    try:
        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["indicator_type"] = indicator_type
            result["indicator_value"] = value.strip()
            return result
    except Exception:
        pass

    return None
