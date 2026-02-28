"""CRF sequence labeling engine with synthetic training data.

Uses sklearn-crfsuite to train a Conditional Random Field model that directly
predicts B-DEP/I-DEP/B-DEST/I-DEST/O tags, so no separate role-assignment
heuristic is needed.
"""

from __future__ import annotations

import random
from unidecode import unidecode

from src.nlp.base import NLPEngine
from src.nlp.gazetteer import get_city_set, get_gazetteer, resolve_city

# French sentence templates for synthetic data generation.
# {dep} and {dest} are replaced by city names.
_TEMPLATES = [
    "je veux aller de {dep} a {dest}",
    "je voudrais aller de {dep} a {dest}",
    "un billet de {dep} a {dest}",
    "un trajet de {dep} a {dest}",
    "un voyage de {dep} a {dest}",
    "partir de {dep} pour aller a {dest}",
    "je pars de {dep} pour {dest}",
    "depuis {dep} vers {dest}",
    "depuis {dep} jusqu'a {dest}",
    "depuis {dep} direction {dest}",
    "je vais de {dep} a {dest}",
    "de {dep} a {dest} s'il vous plait",
    "trajet {dep} {dest}",
    "{dep} direction {dest}",
    "{dep} vers {dest}",
    "je souhaite partir de {dep} et arriver a {dest}",
    "j'habite a {dep} et je vais a {dest}",
    "en partance de {dep} pour {dest}",
    "en partance de {dep} destination {dest}",
    "voyage de {dep} vers {dest}",
    "billet {dep} {dest}",
    "je cherche un train de {dep} a {dest}",
    "aller de {dep} a {dest}",
    "je me rends a {dest} depuis {dep}",
    "vers {dest} depuis {dep}",
    "pour aller de {dep} a {dest}",
    "prendre le train de {dep} a {dest}",
    "reservation de {dep} a {dest}",
]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with punctuation splitting."""
    tokens: list[str] = []
    for word in text.split():
        # Split trailing punctuation
        while word and word[-1] in ".,;:!?":
            word = word[:-1]
        if word:
            tokens.append(word)
    return tokens


def _bio_tag(
    tokens: list[str], dep_tokens: list[str], dest_tokens: list[str]
) -> list[str]:
    """Assign BIO tags to tokens given departure and destination token sequences."""
    tags = ["O"] * len(tokens)
    lower_tokens = [t.lower() for t in tokens]

    for label, entity_tokens in [("DEP", dep_tokens), ("DEST", dest_tokens)]:
        entity_lower = [t.lower() for t in entity_tokens]
        elen = len(entity_lower)
        for i in range(len(lower_tokens) - elen + 1):
            if lower_tokens[i : i + elen] == entity_lower:
                tags[i] = f"B-{label}"
                for j in range(1, elen):
                    tags[i + j] = f"I-{label}"
                break

    return tags


class CRFNEREngine(NLPEngine):
    """CRF sequence labeling NER for French train queries.

    Generates synthetic training data from French templates and gazetteer
    city names, then trains a CRF model on first extract() call (~2-5s).
    The model is cached in memory for subsequent calls.
    """

    def __init__(self) -> None:
        self._model = None
        self._trained = False

    @property
    def name(self) -> str:
        return "crf-ner"

    @property
    def description(self) -> str:
        return "CRF sequence labeler (sklearn-crfsuite). BIO tagging with synthetic training data."

    def _build_training_data(
        self,
    ) -> list[tuple[list[str], list[str]]]:
        """Generate synthetic training sentences with BIO tags."""
        gaz = get_gazetteer()
        city_set = get_city_set()

        # Get a list of canonical city names for sampling
        canonical = list(city_set)
        if len(canonical) < 2:
            return []

        # Build alias -> canonical lookup for short names
        short_aliases: list[str] = []
        for alias in gaz:
            # Keep aliases that are single words or short (good for templates)
            if len(alias.split()) <= 3:
                short_aliases.append(alias)

        # Sample city pairs
        rng = random.Random(42)  # deterministic
        n_pairs = min(200, len(canonical) * (len(canonical) - 1))
        pairs: list[tuple[str, str]] = []
        attempts = 0
        while len(pairs) < n_pairs and attempts < n_pairs * 5:
            a, b = rng.sample(canonical, 2)
            if a != b:
                pairs.append((a, b))
            attempts += 1

        # Generate sentences
        data: list[tuple[list[str], list[str]]] = []
        for dep_canonical, dest_canonical in pairs:
            # Pick a short alias form for template insertion
            dep_form = unidecode(dep_canonical).lower()
            dest_form = unidecode(dest_canonical).lower()

            template = rng.choice(_TEMPLATES)
            sentence = template.format(dep=dep_form, dest=dest_form)
            tokens = _tokenize(sentence)
            dep_tokens = _tokenize(dep_form)
            dest_tokens = _tokenize(dest_form)
            tags = _bio_tag(tokens, dep_tokens, dest_tokens)

            data.append((tokens, tags))

        return data

    def _word_features(self, tokens: list[str], idx: int) -> dict:
        """Extract features for a single token."""
        city_set = get_city_set()
        word = tokens[idx]
        word_lower = word.lower()
        word_uni = unidecode(word_lower)

        features: dict = {
            "word.lower": word_lower,
            "word.uni": word_uni,
            "word.istitle": word.istitle(),
            "word.isupper": word.isupper(),
            "word.isdigit": word.isdigit(),
            "word.shape": _word_shape(word),
            "word.len": len(word),
            "word.in_gazetteer": word_lower in city_set or word_uni in city_set,
        }

        # Bigram gazetteer check (current + next word)
        if idx < len(tokens) - 1:
            bigram = f"{word_lower} {tokens[idx + 1].lower()}"
            bigram_uni = unidecode(bigram)
            features["bigram.in_gazetteer"] = (
                bigram in city_set or bigram_uni in city_set
            )
        else:
            features["bigram.in_gazetteer"] = False

        # Travel verb detection
        travel_verbs = {"aller", "partir", "voyager", "prendre", "cherche", "veux", "voudrais", "souhaite", "rends", "vais", "pars"}
        features["word.is_travel_verb"] = word_lower in travel_verbs

        # Previous word context
        if idx >= 1:
            prev = tokens[idx - 1].lower()
            features["prev.lower"] = prev
            features["prev.is_prep"] = prev in {"de", "depuis", "a", "vers", "pour", "direction", "destination"}
        else:
            features["BOS"] = True

        # Next word context
        if idx < len(tokens) - 1:
            nxt = tokens[idx + 1].lower()
            features["next.lower"] = nxt
        else:
            features["EOS"] = True

        return features

    def _sentence_features(self, tokens: list[str]) -> list[dict]:
        """Extract features for all tokens in a sentence."""
        return [self._word_features(tokens, i) for i in range(len(tokens))]

    def _train(self) -> None:
        """Train the CRF model on synthetic data."""
        import pycrfsuite

        data = self._build_training_data()
        if not data:
            return

        X = [self._sentence_features(tokens) for tokens, _ in data]
        y = [tags for _, tags in data]

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X, y):
            trainer.append(xseq, yseq)

        trainer.set_params({
            "c1": 0.1,
            "c2": 0.1,
            "max_iterations": 100,
            "feature.possible_transitions": True,
        })

        # Train to a temporary file (pycrfsuite requires file-based model)
        import tempfile
        import os

        model_path = os.path.join(tempfile.gettempdir(), "crf_ner_model.crfsuite")
        trainer.train(model_path)

        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        self._model = tagger
        self._trained = True

    def extract(self, text: str) -> tuple[str | None, str | None, float]:
        if not self._trained:
            self._train()

        if self._model is None:
            return None, None, 0.0

        tokens = _tokenize(unidecode(text).lower())
        if not tokens:
            return None, None, 0.0

        features = self._sentence_features(tokens)
        tags = self._model.tag(features)

        # Extract entity spans from BIO tags
        dep_tokens: list[str] = []
        dest_tokens: list[str] = []

        current_label = None
        current_tokens: list[str] = []

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                # Save previous entity
                if current_label == "DEP" and current_tokens:
                    dep_tokens = current_tokens
                elif current_label == "DEST" and current_tokens:
                    dest_tokens = current_tokens
                current_label = tag[2:]
                current_tokens = [token]
            elif tag.startswith("I-") and current_label == tag[2:]:
                current_tokens.append(token)
            else:
                if current_label == "DEP" and current_tokens:
                    dep_tokens = current_tokens
                elif current_label == "DEST" and current_tokens:
                    dest_tokens = current_tokens
                current_label = None
                current_tokens = []

        # Don't forget the last entity
        if current_label == "DEP" and current_tokens:
            dep_tokens = current_tokens
        elif current_label == "DEST" and current_tokens:
            dest_tokens = current_tokens

        # Resolve to station names
        dep_raw = " ".join(dep_tokens) if dep_tokens else None
        dest_raw = " ".join(dest_tokens) if dest_tokens else None

        dep = resolve_city(dep_raw) if dep_raw else None
        dest = resolve_city(dest_raw) if dest_raw else None

        if dep and dest and dep != dest:
            return dep, dest, 0.8

        return None, None, 0.0


def _word_shape(word: str) -> str:
    """Reduce a word to its shape (e.g. 'Paris' -> 'Xxxxx')."""
    shape = []
    for c in word[:5]:  # truncate for efficiency
        if c.isupper():
            shape.append("X")
        elif c.islower():
            shape.append("x")
        elif c.isdigit():
            shape.append("d")
        else:
            shape.append(c)
    return "".join(shape)
