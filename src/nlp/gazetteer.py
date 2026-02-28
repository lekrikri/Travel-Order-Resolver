"""Shared gazetteer utilities for NLP engines.

Provides a singleton gazetteer loader and common functions:
- resolve_city(): normalize a raw string to a known station name
- get_city_set(): O(1) membership check for feature extraction
- assign_roles(): keyword-window heuristic for departure/destination
"""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from pathlib import Path

from unidecode import unidecode

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# Module-level singleton -- loaded once, shared by all engines
_gazetteer: dict[str, str] | None = None
_gazetteer_keys: list[str] | None = None
_city_set: set[str] | None = None

FUZZY_CUTOFF = 0.75
FUZZY_MAX_RESULTS = 1


def _load() -> None:
    """Load gazetteer.json into module-level singletons."""
    global _gazetteer, _gazetteer_keys, _city_set
    with open(PROCESSED_DIR / "gazetteer.json", encoding="utf-8") as f:
        _gazetteer = json.load(f)
    _gazetteer_keys = list(_gazetteer.keys())
    _city_set = set(_gazetteer.values())


def get_gazetteer() -> dict[str, str]:
    """Return the gazetteer dict (alias -> canonical name)."""
    if _gazetteer is None:
        _load()
    return _gazetteer  # type: ignore[return-value]


def get_gazetteer_keys() -> list[str]:
    """Return gazetteer keys as a list (for fuzzy matching)."""
    if _gazetteer_keys is None:
        _load()
    return _gazetteer_keys  # type: ignore[return-value]


def get_city_set() -> set[str]:
    """Return canonical station names as a set (O(1) membership check)."""
    if _city_set is None:
        _load()
    return _city_set  # type: ignore[return-value]


def resolve_city(raw: str, fuzzy: bool = True) -> str | None:
    """Try to match a raw string to a known station name.

    Strategy:
    1. Exact lookup (lowercase, unidecode, dash/space variants)
    2. Prefix match (handles truncated names like "toulous")
    3. Levenshtein fuzzy match (handles typos)
    """
    gaz = get_gazetteer()
    keys = get_gazetteer_keys()

    raw = raw.strip().strip(".,;:!?")
    candidates = [
        raw.lower(),
        unidecode(raw).lower(),
        raw.lower().replace("-", " "),
        unidecode(raw).lower().replace("-", " "),
    ]
    for candidate in candidates:
        if candidate in gaz:
            return gaz[candidate]

    if not fuzzy:
        return None

    normalized = unidecode(raw).lower()
    if len(normalized) < 3:
        return None

    # Prefix match
    for key in keys:
        if key.startswith(normalized):
            return gaz[key]

    # Levenshtein fuzzy match
    matches = get_close_matches(
        normalized, keys, n=FUZZY_MAX_RESULTS, cutoff=FUZZY_CUTOFF
    )
    if matches:
        return gaz[matches[0]]

    return None


# Keywords used by assign_roles() to determine departure vs destination
_DEPARTURE_KEYWORDS = {"de", "depuis", "partant", "partance", "depart", "quitte"}
_DESTINATION_KEYWORDS = {"a", "à", "vers", "pour", "direction", "destination", "jusque"}


def assign_roles(
    text: str, entities: list[str]
) -> tuple[str | None, str | None]:
    """Assign departure/destination roles using a keyword-window heuristic.

    For engines that only detect LOC entities (spaCy, transformer) without
    distinguishing departure from destination, this function looks at the
    words surrounding each entity mention to guess the role.

    Args:
        text: The original input text.
        entities: List of resolved station names (in order of appearance).

    Returns:
        (departure, destination) or (None, None) if less than 2 entities.
    """
    if len(entities) < 2:
        if len(entities) == 1:
            return None, None
        return None, None

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)

    if len(unique) < 2:
        return None, None

    text_lower = unidecode(text).lower()

    # Score each entity: positive = destination, negative = departure
    scores: dict[str, float] = {e: 0.0 for e in unique}

    for entity in unique:
        # Find entity position in the text
        entity_lower = unidecode(entity).lower()
        pos = text_lower.find(entity_lower)
        if pos < 0:
            # Try shorter forms (first word of the station name)
            first_word = entity_lower.split()[0] if entity_lower.split() else ""
            pos = text_lower.find(first_word)

        if pos < 0:
            continue

        # Look at the 5 words preceding the entity
        before = text_lower[:pos].split()
        window = before[-5:] if len(before) >= 5 else before

        for word in window:
            clean = re.sub(r"[^a-z]", "", word)
            if clean in _DEPARTURE_KEYWORDS:
                scores[entity] -= 2.0
            elif clean in _DESTINATION_KEYWORDS:
                scores[entity] += 2.0

    # Sort: most negative = departure, most positive = destination
    ranked = sorted(unique, key=lambda e: scores[e])
    departure = ranked[0]
    destination = ranked[-1]

    if departure == destination:
        # Fallback: first mention = departure, second = destination
        departure = unique[0]
        destination = unique[1]

    return departure, destination
