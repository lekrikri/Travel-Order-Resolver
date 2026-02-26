"""
ÉTAPE 7 : Évaluation comparative de tous les modèles
=====================================================
Compare spaCy (blank, sm, md, lg) et CamemBERT sur le jeu de test.

Exécuter avec : python3 src/evaluation/evaluate_all.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from tabulate import tabulate

# =============================================================================
# CHARGEMENT DES DONNÉES DE TEST
# =============================================================================

def charger_test_data(filepath="data/processed/test.json"):
    """Charge le jeu de test."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# ÉVALUATION SPACY
# =============================================================================

def evaluer_spacy(model_path, test_data):
    """
    Évalue un modèle spaCy sur les données de test.

    Returns:
        dict avec precision, recall, f1
    """
    import spacy

    try:
        nlp = spacy.load(model_path)
    except Exception as e:
        print(f"  Erreur chargement {model_path}: {e}")
        return None

    tp = {"DEPART": 0, "ARRIVEE": 0}
    fp = {"DEPART": 0, "ARRIVEE": 0}
    fn = {"DEPART": 0, "ARRIVEE": 0}

    # Labels qu'on évalue
    labels_valides = {"DEPART", "ARRIVEE"}

    for exemple in test_data:
        text = exemple["text"]
        entities_attendues = exemple["entities"]

        # Prédiction
        doc = nlp(text)

        # Entités prédites (seulement DEPART/ARRIVEE)
        preds = set()
        for ent in doc.ents:
            if ent.label_ in labels_valides:
                preds.add((ent.start_char, ent.end_char, ent.label_))

        # Entités attendues
        golds = set()
        for start, end, label in entities_attendues:
            if label in labels_valides:
                golds.add((start, end, label))

        # Calculer TP, FP, FN
        for pred in preds:
            if pred in golds:
                tp[pred[2]] += 1
            else:
                fp[pred[2]] += 1

        for gold in golds:
            if gold not in preds:
                fn[gold[2]] += 1

    # Calculer métriques globales
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


# =============================================================================
# ÉVALUATION CAMEMBERT
# =============================================================================

def evaluer_camembert(model_path, test_data):
    """
    Évalue le modèle CamemBERT sur les données de test.
    """
    from transformers import CamembertTokenizerFast, CamembertForTokenClassification

    try:
        tokenizer = CamembertTokenizerFast.from_pretrained(model_path)
        model = CamembertForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"  Erreur chargement CamemBERT: {e}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Labels
    id2label = model.config.id2label

    tp = {"DEPART": 0, "ARRIVEE": 0}
    fp = {"DEPART": 0, "ARRIVEE": 0}
    fn = {"DEPART": 0, "ARRIVEE": 0}

    for exemple in test_data:
        text = exemple["text"]
        entities_attendues = exemple["entities"]

        # Tokenizer
        inputs = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=128
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Prédiction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

        # Extraire les entités prédites
        preds = set()
        current_entity = None
        current_start = None

        for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == 0 and end == 0:
                continue

            label = id2label[pred_id]

            if label.startswith("B-"):
                # Sauvegarder l'entité précédente
                if current_entity:
                    preds.add((current_start, current_end, current_entity))

                current_entity = label[2:]  # Enlever "B-"
                current_start = start
                current_end = end

            elif label.startswith("I-") and current_entity == label[2:]:
                current_end = end
            else:
                # Fin d'entité
                if current_entity:
                    preds.add((current_start, current_end, current_entity))
                current_entity = None

        # Dernière entité
        if current_entity:
            preds.add((current_start, current_end, current_entity))

        # Entités attendues
        golds = set()
        for start, end, label in entities_attendues:
            golds.add((start, end, label))

        # Calculer TP, FP, FN
        for pred in preds:
            if pred in golds:
                tp[pred[2]] += 1
            else:
                fp[pred[2]] += 1

        for gold in golds:
            if gold not in preds:
                fn[gold[2]] += 1

    # Calculer métriques globales
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ÉVALUATION COMPARATIVE DES MODÈLES NER")
    print("=" * 70)

    # Charger les données de test
    print("\nChargement du jeu de test...")
    test_data = charger_test_data()
    print(f"  {len(test_data)} exemples de test")

    # Modèles à évaluer
    modeles = [
        ("spaCy blank", "models/ner_blank/model-best", "spacy"),
        ("spaCy fr_core_news_sm", "models/ner_sm", "spacy"),
        ("spaCy fr_core_news_md", "models/ner_md", "spacy"),
        ("spaCy fr_core_news_lg", "models/ner_lg", "spacy"),
        ("spaCy 10k (meilleur)", "models/ner_10k/model-best", "spacy"),
        ("CamemBERT", "models/ner_camembert/best", "camembert"),
    ]

    resultats = []

    print("\nÉvaluation des modèles...")
    print("-" * 70)

    for nom, chemin, type_modele in modeles:
        print(f"\n{nom}...")

        if type_modele == "spacy":
            metrics = evaluer_spacy(chemin, test_data)
        else:
            metrics = evaluer_camembert(chemin, test_data)

        if metrics:
            resultats.append({
                "Modèle": nom,
                "Precision": f"{metrics['precision']:.2f}%",
                "Recall": f"{metrics['recall']:.2f}%",
                "F1-Score": f"{metrics['f1']:.2f}%",
                "TP": metrics["tp"],
                "FP": metrics["fp"],
                "FN": metrics["fn"]
            })
            print(f"  Precision: {metrics['precision']:.2f}%")
            print(f"  Recall:    {metrics['recall']:.2f}%")
            print(f"  F1-Score:  {metrics['f1']:.2f}%")
        else:
            resultats.append({
                "Modèle": nom,
                "Precision": "N/A",
                "Recall": "N/A",
                "F1-Score": "N/A",
                "TP": "-",
                "FP": "-",
                "FN": "-"
            })

    # Afficher le tableau récapitulatif
    print("\n" + "=" * 70)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 70)

    headers = ["Modèle", "Precision", "Recall", "F1-Score", "TP", "FP", "FN"]
    table = [[r[h] for h in headers] for r in resultats]
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # Sauvegarder les résultats
    output_path = Path("results/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés: {output_path}")

    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
