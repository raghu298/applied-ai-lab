"""Input validation module for threat indicators."""

import re


def validate_ip(value: str) -> bool:
    pattern = r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
    return bool(re.match(pattern, value.strip()))


def validate_url(value: str) -> bool:
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, value.strip(), re.IGNORECASE))


def validate_domain(value: str) -> bool:
    pattern = r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
    return bool(re.match(pattern, value.strip()))


def validate_cve(value: str) -> bool:
    pattern = r"^CVE-\d{4}-\d{4,}$"
    return bool(re.match(pattern, value.strip(), re.IGNORECASE))


def validate_file_hash(value: str) -> bool:
    pattern = r"^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$"
    return bool(re.match(pattern, value.strip()))


def validate_email(value: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, value.strip()))


VALIDATORS = {
    "IP Address": validate_ip,
    "URL": validate_url,
    "Domain": validate_domain,
    "CVE ID": validate_cve,
    "File Hash": validate_file_hash,
    "Email Address": validate_email,
}


def validate_indicator(indicator_type: str, value: str) -> bool:
    """Validate an indicator value based on its type."""
    validator = VALIDATORS.get(indicator_type)
    if validator:
        return validator(value.strip())
    return True
