"""SpaCy NER engines using our custom-trained models (DEPART/ARRIVEE)."""

from __future__ import annotations

import time
from pathlib import Path

from src.nlp.base import NLPEngine

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class SpaCyNEREngine(NLPEngine):
    """Generic spaCy NER engine loading any of our custom-trained models."""

    def __init__(self, model_dir: str, display_name: str, desc: str) -> None:
        self._model_path = MODELS_DIR / model_dir
        self._display_name = display_name
        self._desc = desc
        self._nlp = None
        self.loaded = False
        self.load_time_ms = 0.0

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._desc

    def _load_model(self):
        if self._nlp is None:
            import spacy
            t0 = time.perf_counter()
            self._nlp = spacy.load(str(self._model_path))
            self.load_time_ms = round((time.perf_counter() - t0) * 1000, 1)
            self.loaded = True
        return self._nlp

    def extract(self, text: str) -> tuple[str | None, str | None, float]:
        nlp = self._load_model()
        doc = nlp(text)

        dep, dest = None, None
        extra_departs: list[str] = []
        for ent in doc.ents:
            if ent.label_ == "DEPART":
                if dep is None:
                    dep = ent.text
                else:
                    extra_departs.append(ent.text)
            elif ent.label_ == "ARRIVEE" and dest is None:
                dest = ent.text

        # Fallback: if model tagged two DEPART and zero ARRIVEE,
        # treat the second one as ARRIVEE (order of appearance)
        if dep and dest is None and extra_departs:
            dest = extra_departs[0]

        if dep and dest:
            return dep, dest, 0.999

        return None, None, 0.0
