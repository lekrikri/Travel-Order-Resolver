"""Evaluate all 8 NLP engines on the updated test set.

Models:
  1. rule-based     (RuleBasedEngine — regex + gazetteer)
  2. spacy-blank    (models/ner_blank/model-best)
  3. spacy-sm       (models/ner_sm)
  4. spacy-md       (models/ner_md)
  5. spacy-lg       (models/ner_lg)
  6. spacy-10k      (models/ner_10k/model-best)
  7. crf-ner        (CRFNEREngine — sklearn-crfsuite)
  8. transformer    (TransformerNEREngine — CamemBERT fine-tuned)

Metrics (per engine):
  - Exact Match %  : both dep AND dest correctly extracted (city-level, accent-normalized)
  - F1-Score       : span-level precision/recall/F1 on DEPART/ARRIVEE entities
  - Precision / Recall (same span-level)
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed" / "test.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase + strip accents for fuzzy matching."""
    nfkd = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower().strip()


def cities_match(pred: str | None, gold: str | None) -> bool:
    """City-level exact match ignoring accents/case."""
    if not pred or not gold:
        return False
    return normalize(pred) == normalize(gold)


def load_test() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Span-level F1 (same logic as evaluate_all.py, but works on extract() output)
# ---------------------------------------------------------------------------

def eval_extract_fn(extract_fn, test_data: list[dict]) -> dict:
    """
    Evaluate an engine that exposes extract(text) → (dep, dest, score).

    Returns precision, recall, f1, exact_match.
    """
    tp = fp = fn = 0
    exact_both = 0
    total = len(test_data)

    for ex in test_data:
        text = ex["text"]
        gold_dep = ex.get("depart", "")
        gold_dest = ex.get("arrivee", "")
        gold_valid = ex.get("valid", True)

        try:
            pred_dep, pred_dest, _ = extract_fn(text)
        except Exception:
            pred_dep, pred_dest = None, None

        # Exact match (city-level)
        dep_ok = cities_match(pred_dep, gold_dep) if gold_valid else (pred_dep is None)
        dest_ok = cities_match(pred_dest, gold_dest) if gold_valid else (pred_dest is None)
        if dep_ok and dest_ok:
            exact_both += 1

        # Span-level counts
        if gold_valid:
            # Gold has 2 entities
            predicted = sum([pred_dep is not None, pred_dest is not None])
            correct = sum([cities_match(pred_dep, gold_dep), cities_match(pred_dest, gold_dest)])
            tp += correct
            fp += predicted - correct
            fn += 2 - correct
        else:
            # Gold has 0 entities; any prediction is FP
            fp += sum([pred_dep is not None, pred_dest is not None])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact = exact_both / total if total > 0 else 0

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "exact_match": round(exact * 100, 2),
        "tp": tp, "fp": fp, "fn": fn,
        "total": total,
    }


def eval_spacy_model(model_path: str, test_data: list[dict]) -> dict | None:
    """Wrap a spaCy NER model as an extract function."""
    import spacy
    try:
        nlp = spacy.load(model_path)
    except Exception as e:
        print(f"  [ERREUR] Impossible de charger {model_path}: {e}")
        return None

    def extract(text: str):
        doc = nlp(text)
        dep = dest = None
        for ent in doc.ents:
            if ent.label_ == "DEPART" and dep is None:
                dep = ent.text
            elif ent.label_ == "ARRIVEE" and dest is None:
                dest = ent.text
        return dep, dest, 0.9

    return eval_extract_fn(extract, test_data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ÉVALUATION — 8 MODÈLES NER (dataset mis à jour hard_v2)")
    print("=" * 70)

    test_data = load_test()
    positive = [e for e in test_data if e.get("valid", True)]
    print(f"\nJeu de test: {len(test_data)} exemples ({len(positive)} positifs, {len(test_data)-len(positive)} négatifs)")
    print()

    results = {}

    # ------------------------------------------------------------------
    # 1. Rule-based
    # ------------------------------------------------------------------
    print("1/8  rule-based …", flush=True)
    try:
        from src.nlp.rule_based import RuleBasedEngine
        engine = RuleBasedEngine()
        results["rule-based"] = eval_extract_fn(engine.extract, test_data)
        print(f"     F1={results['rule-based']['f1']}%  Exact={results['rule-based']['exact_match']}%")
    except Exception as e:
        print(f"     [ERREUR] {e}")

    # ------------------------------------------------------------------
    # 2-6. spaCy variants
    # ------------------------------------------------------------------
    spacy_models = [
        ("spacy-blank", "models/ner_blank/model-best"),
        ("spacy-sm",    "models/ner_sm"),
        ("spacy-md",    "models/ner_md"),
        ("spacy-lg",    "models/ner_lg"),
        ("spacy-10k",   "models/ner_10k/model-best"),
    ]
    for idx, (name, path) in enumerate(spacy_models, start=2):
        print(f"{idx}/8  {name} …", flush=True)
        r = eval_spacy_model(str(ROOT / path), test_data)
        if r:
            results[name] = r
            print(f"     F1={r['f1']}%  Exact={r['exact_match']}%")
        else:
            results[name] = None

    # ------------------------------------------------------------------
    # 7. CRF-NER
    # ------------------------------------------------------------------
    print("7/8  crf-ner …", flush=True)
    try:
        from src.nlp.crf_ner import CRFNEREngine
        engine = CRFNEREngine()
        results["crf-ner"] = eval_extract_fn(engine.extract, test_data)
        print(f"     F1={results['crf-ner']['f1']}%  Exact={results['crf-ner']['exact_match']}%")
    except Exception as e:
        print(f"     [ERREUR] {e}")

    # ------------------------------------------------------------------
    # 8. Transformer (CamemBERT)
    # ------------------------------------------------------------------
    print("8/8  transformer-ner (CamemBERT) …", flush=True)
    try:
        from src.nlp.transformer_ner import TransformerNEREngine
        engine = TransformerNEREngine()
        results["transformer-ner"] = eval_extract_fn(engine.extract, test_data)
        print(f"     F1={results['transformer-ner']['f1']}%  Exact={results['transformer-ner']['exact_match']}%")
    except Exception as e:
        print(f"     [ERREUR] {e}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    header = f"{'Modèle':<22} {'Exact%':>8} {'F1%':>8} {'Prec%':>8} {'Rec%':>8}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        if r:
            print(f"{name:<22} {r['exact_match']:>7.2f}% {r['f1']:>7.2f}% {r['precision']:>7.2f}% {r['recall']:>7.2f}%")
        else:
            print(f"{name:<22}   N/A")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    out = {name: r for name, r in results.items() if r}
    (RESULTS_DIR / "evaluation_results_8models.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRésultats sauvegardés → results/evaluation_results_8models.json")
    return results


if __name__ == "__main__":
    main()
