"""
ÉTAPE 2 : Conversion JSON → Format .spacy
==========================================
Ce script convertit nos fichiers JSON annotés en format binaire .spacy
pour l'entraînement du modèle NER.

Exécuter avec : python3 src/dataset/convert_to_spacy.py
"""

import json
import spacy
from spacy.tokens import DocBin
from pathlib import Path


def charger_json(filepath):
    """Charge un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def convertir_vers_spacy(data, nlp):
    """
    Convertit une liste d'exemples annotés en DocBin spaCy.

    Args:
        data: Liste de dicts avec 'text' et 'entities'
        nlp: Pipeline spaCy (pour la tokenisation)

    Returns:
        DocBin contenant tous les documents annotés
    """
    doc_bin = DocBin()

    erreurs = 0
    succes = 0

    for exemple in data:
        texte = exemple['text']
        entities = exemple['entities']

        # Créer un Doc à partir du texte
        doc = nlp.make_doc(texte)

        # Convertir les annotations en Spans
        ents = []
        for start, end, label in entities:
            # char_span crée un Span à partir des positions de caractères
            span = doc.char_span(start, end, label=label)

            if span is None:
                # Parfois les positions ne correspondent pas aux tokens
                erreurs += 1
            else:
                ents.append(span)

        # Assigner les entités au Doc
        doc.ents = ents

        # Ajouter au DocBin
        doc_bin.add(doc)
        succes += 1

    print(f"  Convertis: {succes}, Erreurs d'alignement: {erreurs}")
    return doc_bin


def main():
    print("=" * 60)
    print("CONVERSION JSON → FORMAT .spacy")
    print("=" * 60)

    # Charger un modèle vide pour la tokenisation
    # On utilise blank("fr") car on veut juste tokeniser, pas prédire
    print("\nChargement du tokenizer francais...")
    nlp = spacy.blank("fr")

    # Chemins
    input_dir = Path("data/processed")
    output_dir = Path("data/processed")

    # Convertir chaque split
    for split in ["train", "val", "test"]:
        json_path = input_dir / f"{split}.json"
        spacy_path = output_dir / f"{split}.spacy"

        print(f"\n{split.upper()}:")
        print(f"  Lecture: {json_path}")

        # Charger les données
        data = charger_json(json_path)
        print(f"  {len(data)} exemples")

        # Convertir
        doc_bin = convertir_vers_spacy(data, nlp)

        # Sauvegarder
        doc_bin.to_disk(spacy_path)
        print(f"  Sauvegarde: {spacy_path}")

    print("\n" + "=" * 60)
    print("CONVERSION TERMINEE!")
    print("=" * 60)
    print("\nFichiers crees:")
    print("  - data/processed/train.spacy")
    print("  - data/processed/val.spacy")
    print("  - data/processed/test.spacy")
    print("\nProchaine etape: Entrainer le modele NER!")


if __name__ == "__main__":
    main()
