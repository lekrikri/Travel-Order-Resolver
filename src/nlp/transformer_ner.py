"""CamemBERT NER engine using Hugging Face transformers.

Uses the Jean-Baptiste/camembert-ner model (~400 MB) for French
named-entity recognition. The model is auto-downloaded from HF Hub
on first use.
"""

from __future__ import annotations

from src.nlp.base import NLPEngine
from src.nlp.gazetteer import assign_roles, resolve_city


class TransformerNEREngine(NLPEngine):
    """Extract departure and destination using CamemBERT NER.

    Pipeline: HF token-classification -> filter LOC entities ->
              resolve via gazetteer -> assign_roles()

    The model is lazy-loaded on first call so the server starts instantly.
    """

    def __init__(self) -> None:
        self._pipeline = None

    @property
    def name(self) -> str:
        return "transformer-ner"

    @property
    def description(self) -> str:
        return "CamemBERT NER (Jean-Baptiste/camembert-ner). Transformer-based French NER."

    def _load_pipeline(self):
        """Lazy-load the HF NER pipeline."""
        if self._pipeline is None:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            model_name = "Jean-Baptiste/camembert-ner"
            # Use the base CamemBERT tokenizer (has tokenizer.json
            # compatible with transformers v5+). Same vocabulary.
            tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        return self._pipeline

    def extract(self, text: str) -> tuple[str | None, str | None, float]:
        pipe = self._load_pipeline()
        entities = pipe(text)

        # Filter for LOC entities and resolve to station names
        resolved: list[str] = []
        for ent in entities:
            if ent["entity_group"] == "LOC":
                station = resolve_city(ent["word"])
                if station and station not in resolved:
                    resolved.append(station)

        if len(resolved) < 2:
            return None, None, 0.0

        dep, dest = assign_roles(text, resolved)
        if dep and dest and dep != dest:
            return dep, dest, 0.75

        return None, None, 0.0
