"""Rule-based NLP engine using regex patterns and a station gazetteer."""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from pathlib import Path
from unidecode import unidecode

from src.nlp.base import NLPEngine

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class RuleBasedEngine(NLPEngine):
    """Extract departure and destination using French keyword patterns.

    Strategy:
    1. Normalize the input text (lowercase, remove accents)
    2. Try directional patterns ("de X a Y", "depuis X vers Y", etc.)
    3. If patterns fail, find all city mentions and use position heuristic
    """

    # Patterns ordered by specificity (most specific first)
    # Each pattern has named groups: dep (departure) and arr (arrival)
    PATTERNS = [
        # "de <dep> a/à <arr>"
        r"(?:partir |aller |voyager )?de (?P<dep>.+?) (?:jusqu'?[aà]|[aà]) (?P<arr>.+?)(?:\s*[?.!]|$)",
        # "depuis <dep> vers/direction/pour <arr>"
        r"depuis (?P<dep>.+?) (?:vers|direction|pour|en direction de) (?P<arr>.+?)(?:\s*[?.!]|$)",
        # "en partance de <dep> .* destination <arr>"
        r"en partance de (?P<dep>.+?) .*?(?:destination|vers|pour) (?P<arr>.+?)(?:\s*[?.!]|$)",
        # "<dep> direction <arr>"
        r"(?:^|train |billet |trajet )(?P<dep>.+?) direction (?P<arr>.+?)(?:\s*[?.!]|$)",
        # "vers <arr> depuis <dep>" (reversed order)
        r"vers (?P<arr>.+?) (?:depuis|de|en partance de) (?P<dep>.+?)(?:\s*[?.!]|$)",
        # "un billet <dep>-<arr>" or "<dep> <arr>" (dash or arrow separated)
        r"(?:billet|trajet|voyage|aller) (?P<dep>.+?)(?:\s*[-–—→]\s*| )(?P<arr>.+?)(?:\s*[?.!]|$)",
        # "habite à <dep> ... pars/vais à <arr>"
        r"habite (?:a|à) (?P<dep>.+?)(?:\.|,|\s+.+?\s+)(?:pars|vais|voyage|me rends) (?:a|à) (?P<arr>.+?)(?:\s*[?.!]|$)",
        # reversed: "pars à <arr> ... habite à <dep>"
        r"(?:pars|vais|voyage|me rends) (?:a|à) (?P<arr>.+?)(?:\.|,|\s+.+?\s+)habite (?:a|à) (?P<dep>.+?)(?:\s*[?.!]|$)",
    ]

    # Context scoring: words preceding a city mention that boost/penalize its score
    CITY_BOOST = {"à", "a", "vers", "pour", "de", "depuis"}       # +2.0
    PERSON_PENALTY = {"avec"}                                       # -3.0
    ET_PENALTY = -1.5  # "et" between name-like words

    # Verbs before "à <city>" that indicate travel role
    DEPARTURE_VERBS = {"habite", "vis", "réside", "demeure"}       # role=departure
    DESTINATION_VERBS = {"pars", "vais", "va", "voyage", "rends"}  # role=destination

    FUZZY_CUTOFF = 0.75
    FUZZY_MAX_RESULTS = 1

    def __init__(self) -> None:
        self._gazetteer: dict[str, str] | None = None
        self._gazetteer_keys: list[str] | None = None
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATTERNS
        ]

    @property
    def name(self) -> str:
        return "rule-based"

    @property
    def description(self) -> str:
        return "Regex patterns + station gazetteer lookup. Fast, no ML dependency."

    @property
    def gazetteer(self) -> dict[str, str]:
        if self._gazetteer is None:
            with open(PROCESSED_DIR / "gazetteer.json", encoding="utf-8") as f:
                self._gazetteer = json.load(f)
            self._gazetteer_keys = list(self._gazetteer.keys())
        return self._gazetteer

    @property
    def gazetteer_keys(self) -> list[str]:
        _ = self.gazetteer  # ensure loaded
        return self._gazetteer_keys  # type: ignore[return-value]

    def normalize(self, text: str) -> str:
        """Normalize input text for matching."""
        text = text.lower().strip()
        # Normalize unicode but keep original for gazetteer lookup
        return text

    def resolve_city(self, raw: str, fuzzy: bool = True) -> str | None:
        """Try to match a raw string to a known station name."""
        raw = raw.strip().strip(".,;:!?")
        candidates = [
            raw.lower(),
            unidecode(raw).lower(),
            raw.lower().replace("-", " "),
            unidecode(raw).lower().replace("-", " "),
        ]
        for candidate in candidates:
            if candidate in self.gazetteer:
                return self.gazetteer[candidate]

        if not fuzzy:
            return None

        # Fuzzy matching fallback: handle misspelled city names
        normalized = unidecode(raw).lower()
        if len(normalized) < 3:
            return None

        # Try prefix match first (fast) -- e.g. "toulous" matches "toulouse"
        for key in self.gazetteer_keys:
            if key.startswith(normalized):
                return self.gazetteer[key]

        # Levenshtein-based fuzzy match (slower, last resort)
        matches = get_close_matches(
            normalized,
            self.gazetteer_keys,
            n=self.FUZZY_MAX_RESULTS,
            cutoff=self.FUZZY_CUTOFF,
        )
        if matches:
            return self.gazetteer[matches[0]]

        return None

    def _score_city_mention(self, position: int, words: list[str]) -> tuple[float, str]:
        """Score a city mention by surrounding context.

        Returns (score, role) where role is 'departure'|'destination'|'unknown'.
        """
        score = 1.0
        role = "unknown"

        # Check the word immediately before the city
        if position >= 1:
            prev = words[position - 1]
            if prev in self.CITY_BOOST:
                score += 2.0
            elif prev in self.PERSON_PENALTY:
                score -= 3.0
            elif prev == "et":
                score += self.ET_PENALTY

        # Check verb before preposition: "habite à <city>" or "pars à <city>"
        if position >= 2:
            prev_prev = words[position - 2]
            prev = words[position - 1]
            if prev in {"à", "a"}:
                if prev_prev in self.DEPARTURE_VERBS:
                    score += 3.0
                    role = "departure"
                elif prev_prev in self.DESTINATION_VERBS:
                    score += 2.5
                    role = "destination"

        return max(score, 0.0), role

    def _assign_roles(
        self, cities: list[tuple[str, float, str]],
    ) -> tuple[str | None, str | None]:
        """From scored+role-tagged cities, assign departure and destination.

        Each entry is (city_name, score, role).
        """
        tagged_dep = [c for c in cities if c[2] == "departure"]
        tagged_arr = [c for c in cities if c[2] == "destination"]
        unknowns = [c for c in cities if c[2] == "unknown"]

        dep: str | None = None
        arr: str | None = None

        if tagged_dep and tagged_arr:
            dep = tagged_dep[0][0]
            arr = tagged_arr[0][0]
        elif tagged_dep and unknowns:
            dep = tagged_dep[0][0]
            arr = unknowns[0][0]
        elif tagged_arr and unknowns:
            dep = unknowns[0][0]
            arr = tagged_arr[0][0]
        elif len(cities) >= 2:
            dep = cities[0][0]
            arr = cities[1][0]

        return dep, arr

    def extract(self, text: str) -> tuple[str | None, str | None, float]:
        normalized = self.normalize(text)

        # Try each pattern
        for pattern in self._compiled_patterns:
            match = pattern.search(normalized)
            if match:
                raw_dep = match.group("dep")
                raw_arr = match.group("arr")
                dep = self.resolve_city(raw_dep)
                arr = self.resolve_city(raw_arr)
                if dep and arr and dep != arr:
                    return dep, arr, 0.9

        # Fallback: find all city mentions and score them by context
        words = normalized.split()
        found_cities: list[tuple[int, str]] = []
        # Try longest matches first (multi-word city names)
        for n in range(4, 0, -1):
            for i in range(len(words) - n + 1):
                chunk = " ".join(words[i : i + n])
                city = self.resolve_city(chunk, fuzzy=False)
                if city:
                    # Check this position wasn't already matched
                    if not any(abs(pos - i) < n for pos, _ in found_cities):
                        found_cities.append((i, city))

        # Score each city mention by surrounding context
        scored: list[tuple[str, float, str, int]] = []  # (city, score, role, pos)
        for pos, city in found_cities:
            score, role = self._score_city_mention(pos, words)
            scored.append((city, score, role, pos))

        # Sort by score (descending), then position as tiebreaker
        scored.sort(key=lambda x: (-x[1], x[3]))

        # Deduplicate by city name (keep highest score)
        seen: set[str] = set()
        unique: list[tuple[str, float, str]] = []
        for city, score, role, _ in scored:
            if city not in seen:
                seen.add(city)
                unique.append((city, score, role))

        if len(unique) >= 2:
            dep, arr = self._assign_roles(unique)
            if dep and arr and dep != arr:
                return dep, arr, 0.5

        return None, None, 0.0
