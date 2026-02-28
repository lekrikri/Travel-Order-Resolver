"""SpaCy NER engine using the fr_core_news_md French model."""

from __future__ import annotations

from src.nlp.base import NLPEngine
from src.nlp.gazetteer import assign_roles, resolve_city


class SpaCyNEREngine(NLPEngine):
    """Extract departure and destination using spaCy's named-entity recognition.

    Pipeline: spaCy NER -> filter LOC entities -> resolve via gazetteer -> assign_roles()

    The French model (fr_core_news_md, ~50 MB) is lazy-loaded on first call
    so the server starts instantly.
    """

    def __init__(self) -> None:
        self._nlp = None

    @property
    def name(self) -> str:
        return "spacy-ner"

    @property
    def description(self) -> str:
        return "spaCy fr_core_news_md NER pipeline. Detects LOC entities, resolves via gazetteer."

    def _load_model(self):
        """Lazy-load the spaCy French model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("fr_core_news_md")
        return self._nlp

    def extract(self, text: str) -> tuple[str | None, str | None, float]:
        nlp = self._load_model()
        doc = nlp(text)

        # Collect LOC entities and resolve them to station names
        resolved: list[str] = []
        for ent in doc.ents:
            if ent.label_ == "LOC":
                station = resolve_city(ent.text)
                if station and station not in resolved:
                    resolved.append(station)

        if len(resolved) < 2:
            return None, None, 0.0

        dep, dest = assign_roles(text, resolved)
        if dep and dest and dep != dest:
            return dep, dest, 0.7

        return None, None, 0.0
