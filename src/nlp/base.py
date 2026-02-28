"""Base interface for NLP engines (Strategy pattern)."""

from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class NLPResult:
    """Result returned by any NLP engine."""

    departure: str | None = None
    destination: str | None = None
    confidence: float = 0.0
    is_valid: bool = False
    # Metrics (filled automatically by NLPEngine.run)
    time_ms: float = 0.0
    memory_kb: float = 0.0
    engine_name: str = ""


class NLPEngine(ABC):
    """Abstract base class for all NLP engines.

    Every engine must implement `extract()` which takes raw text
    and returns (departure, destination) or None for invalid orders.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this engine (e.g. 'rule-based', 'spacy')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of how this engine works."""

    @abstractmethod
    def extract(self, text: str) -> tuple[str | None, str | None, float]:
        """Extract departure and destination from text.

        Returns:
            (departure, destination, confidence) where confidence is 0.0-1.0.
            Return (None, None, 0.0) for invalid orders.
        """

    def run(self, text: str) -> NLPResult:
        """Run extraction with automatic timing and memory measurement."""
        tracemalloc.start()
        start = time.perf_counter()

        dep, dest, conf = self.extract(text)

        elapsed_ms = (time.perf_counter() - start) * 1000
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return NLPResult(
            departure=dep,
            destination=dest,
            confidence=conf,
            is_valid=dep is not None and dest is not None,
            time_ms=round(elapsed_ms, 3),
            memory_kb=round(peak_kb / 1024, 2),
            engine_name=self.name,
        )
