"""Parsing for the agent's comma-separated `keywords` field, shared by every transcriber."""

from typing import List, Optional


def keyword_entries(keywords: Optional[str]) -> List[str]:
    """Entries as written, weights included, for engines that take the agent's own syntax."""
    return [entry.strip() for entry in (keywords or "").split(",") if entry.strip()]


def keyword_terms(keywords: Optional[str]) -> List[str]:
    """Entries with any `:weight` suffix dropped, for engines that take a bare term list."""
    terms = (_strip_weight(entry) for entry in keyword_entries(keywords))
    return [term for term in terms if term]


def _strip_weight(entry: str) -> str:
    """`bolna:2` becomes `bolna`. A colon that is not a weight belongs to the term, as in `3:30 pm`."""
    term, separator, weight = entry.rpartition(":")
    if not separator:
        return entry
    try:
        float(weight)
    except ValueError:
        return entry
    return term.strip()
