"""
ÉTAPE 4 : Entraînement avec différents modèles de base
=======================================================
Compare : blank (fait), sm, md, lg

Exécuter avec : python3 src/models/train_all_models.py
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_TO_TRAIN = [
    ("fr_core_news_sm", "ner_sm"),
    ("fr_core_news_md", "ner_md"),
    ("fr_core_news_lg", "ner_lg"),
]

TRAIN_PATH = "data/processed/train.spacy"
VAL_PATH = "data/processed/val.spacy"
OUTPUT_DIR = "models"

N_EPOCHS = 20  # Nombre d'epochs d'entraînement


# =============================================================================
# FONCTIONS
# =============================================================================

def charger_donnees(path, nlp):
    """Charge les données depuis un fichier .spacy"""
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs


def creer_examples(docs, nlp):
    """Crée des Examples pour l'entraînement spaCy"""
    examples = []
    for doc in docs:
        # Créer un Example avec le texte et les annotations
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        example = Example.from_dict(
            nlp.make_doc(doc.text),
            {"entities": entities}
        )
        examples.append(example)
    return examples


def entrainer_modele(base_model, output_name, n_epochs=20):
    """
    Entraîne un NER custom à partir d'un modèle de base.

    Args:
        base_model: Nom du modèle spaCy (ex: "fr_core_news_sm")
        output_name: Nom du dossier de sortie (ex: "ner_sm")
        n_epochs: Nombre d'epochs
    """
    print(f"\n{'='*60}")
    print(f"ENTRAINEMENT: {base_model} -> {output_name}")
    print(f"{'='*60}")

    # 1. Charger le modèle de base
    print(f"Chargement de {base_model}...")
    nlp = spacy.load(base_model)

    # 2. Ajouter ou récupérer le NER
    if "ner" in nlp.pipe_names:
        ner = nlp.get_pipe("ner")
    else:
        ner = nlp.add_pipe("ner")

    # 3. Ajouter nos labels
    ner.add_label("DEPART")
    ner.add_label("ARRIVEE")

    # 4. Charger les données
    print("Chargement des donnees...")
    train_docs = charger_donnees(TRAIN_PATH, nlp)
    val_docs = charger_donnees(VAL_PATH, nlp)

    print(f"  Train: {len(train_docs)} exemples")
    print(f"  Val: {len(val_docs)} exemples")

    # 5. Créer les Examples
    train_examples = creer_examples(train_docs, nlp)
    val_examples = creer_examples(val_docs, nlp)

    # 6. Configurer l'entraînement
    # On ne veut entraîner que le NER, pas les autres composants
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # 7. Entraîner
    print(f"\nEntrainement ({n_epochs} epochs)...")

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        for epoch in range(n_epochs):
            random.shuffle(train_examples)
            losses = {}

            # Entraîner par batch
            for i in range(0, len(train_examples), 32):
                batch = train_examples[i:i+32]
                nlp.update(batch, sgd=optimizer, losses=losses)

            # Évaluer sur validation
            scores = nlp.evaluate(val_examples)

            print(f"  Epoch {epoch+1:2}/{n_epochs} | "
                  f"Loss: {losses.get('ner', 0):8.2f} | "
                  f"F1: {scores['ents_f']*100:5.1f}% | "
                  f"P: {scores['ents_p']*100:5.1f}% | "
                  f"R: {scores['ents_r']*100:5.1f}%")

    # 8. Sauvegarder
    output_path = Path(OUTPUT_DIR) / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)

    print(f"\nModele sauvegarde: {output_path}")

    return scores


def main():
    print("="*60)
    print("COMPARAISON DES MODELES SPACY")
    print("="*60)

    resultats = []

    # Entraîner chaque modèle
    for base_model, output_name in MODELS_TO_TRAIN:
        try:
            scores = entrainer_modele(base_model, output_name, N_EPOCHS)
            resultats.append({
                "modele": output_name,
                "base": base_model,
                "f1": round(scores['ents_f'] * 100, 2),
                "precision": round(scores['ents_p'] * 100, 2),
                "recall": round(scores['ents_r'] * 100, 2)
            })
        except Exception as e:
            print(f"ERREUR avec {base_model}: {e}")

    # Afficher le résumé
    print("\n" + "="*60)
    print("RESUME DES RESULTATS")
    print("="*60)
    print(f"{'Modele':<15} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-"*50)

    for r in resultats:
        print(f"{r['modele']:<15} {r['f1']:>9.1f}% {r['precision']:>11.1f}% {r['recall']:>9.1f}%")

    # Sauvegarder les résultats
    with open("results/comparaison_modeles.json", 'w') as f:
        json.dump(resultats, f, indent=2)

    print(f"\nResultats sauvegardes: results/comparaison_modeles.json")


if __name__ == "__main__":
    # Créer le dossier results
    Path("results").mkdir(exist_ok=True)
    main()
