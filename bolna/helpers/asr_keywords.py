"""Parsing for the agent's comma-separated `keywords` field, shared by every transcriber.

One agent-level string, written as `term` or `term:weight`, feeds engines that want a bare term
list, a weighted list, or the string itself.
"""

from typing import List, Optional, Tuple

DEFAULT_KEYWORD_WEIGHT = 1.0


def keyword_entries(keywords: Optional[str]) -> List[str]:
    """Entries as written, weights included, for engines that take the agent's own syntax."""
    return [entry.strip() for entry in (keywords or "").split(",") if entry.strip()]


def parse_keywords(keywords: Optional[str]) -> List[Tuple[str, float]]:
    """`[(term, weight)]`, defaulting the weight where the entry carries none."""
    parsed = []
    for entry in keyword_entries(keywords):
        term, separator, weight = entry.rpartition(":")
        try:
            parsed.append((term.strip(), float(weight)) if separator else (entry, DEFAULT_KEYWORD_WEIGHT))
        except ValueError:
            # A colon that is not a weight belongs to the term.
            parsed.append((entry, DEFAULT_KEYWORD_WEIGHT))
    return [(term, weight) for term, weight in parsed if term]


def keyword_terms(keywords: Optional[str]) -> List[str]:
    """Terms alone, for engines that take an unweighted list."""
    return [term for term, _ in parse_keywords(keywords)]


def format_weighted_keywords(parsed: List[Tuple[str, float]]) -> str:
    """`term:weight` comma-separated, one explicit weight per entry."""
    return ",".join(f"{term}:{weight:g}" for term, weight in parsed)
