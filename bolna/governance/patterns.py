"""Compiled PII regexes. Deterministic, no network, no LLM."""

from __future__ import annotations

import re

# Placeholders never contain the original span — receipts only store these labels.
PLACEHOLDERS = {
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "account_number": "[ACCOUNT_NUMBER]",
}

DEFAULT_PII_TYPES = ("ssn", "credit_card", "email", "phone")

# Dashed / spaced US SSN only. Bare 9-digit runs collide with order ids.
_SSN = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")

# 13–19 digits with optional separators; Luhn-checked at match time.
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# E.164 or common North-American spoken forms. Require a separator or leading +.
_PHONE = re.compile(r"(?<!\d)(?:\+?1[-.\s]*)?(?:\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}|\+\d{10,15})(?!\d)")

# Long digit runs that are not phones (12–17). Off unless the agent opts in.
_ACCOUNT = re.compile(r"\b\d{12,17}\b")

_COMPILED = {
    "ssn": _SSN,
    "credit_card": _CREDIT_CARD,
    "email": _EMAIL,
    "phone": _PHONE,
    "account_number": _ACCOUNT,
}


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def find_and_redact(text: str, pii_types: tuple[str, ...] | list[str]) -> tuple[str, list[str]]:
    """Replace matches with placeholders. Returns (redacted_text, labels_found).

    Labels are unique and ordered by first appearance. Original spans are discarded.
    """
    if not text:
        return text, []
    found: list[str] = []
    redacted = text
    for pii_type in pii_types:
        pattern = _COMPILED.get(pii_type)
        if pattern is None:
            continue
        placeholder = PLACEHOLDERS[pii_type]

        def _sub(match, _type=pii_type, _ph=placeholder):
            raw = match.group(0)
            if _type == "credit_card" and not luhn_ok(raw):
                return raw
            if _type not in found:
                found.append(_type)
            return _ph

        redacted = pattern.sub(_sub, redacted)
    return redacted, found
