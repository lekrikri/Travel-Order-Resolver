#!/usr/bin/env python3
"""
Lancement du warm-up Phase 1.1
"""

import sys
import os

# Ajouter src au path
sys.path.insert(0, './src/1_spacy_baseline')

from warmup import SpacyNERWarmup

def main():
    print("="*70)
    print("PHASE 1.1 - WARM-UP : Test NER avec spaCy")
    print("="*70)

    # Initialiser
    print("\n[1/4] Initialisation de spaCy fr_core_news_md...")
    warmup = SpacyNERWarmup(model_name="fr_core_news_md")

    # Charger le dataset
    print("\n[2/4] Chargement du dataset Kaggle...")
    dataset_path = "./data/raw/ner_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"ERREUR: {dataset_path} non trouvé!")
        print("Téléchargez-le depuis:")
        print("https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus")
        return

    df = warmup.load_ner_dataset(dataset_path)

    # Extraire les phrases
    print("\n[3/4] Extraction des phrases...")
    sentences = warmup.extract_sentences(df)
    print(f"       Total phrases extraites: {len(sentences)}")

    # Afficher quelques exemples
    print("\nExemples de phrases:")
    for i, (text, tags) in enumerate(sentences[:3]):
        print(f"  {i+1}. {text[:80]}...")
        print(f"     Tags: {' '.join(tags[:10])}...")

    # Évaluer (sur 100 premières phrases pour rapidité)
    print("\n[4/4] Evaluation sur 100 premières phrases...")
    results = warmup.evaluate(sentences)

    # Afficher les résultats
    print("\n" + "="*70)
    print("RESULTATS WARM-UP - spaCy NER")
    print("="*70)
    print(f"Dataset:   ner_dataset.csv ({len(sentences)} phrases au total)")
    print(f"Evalué sur: 100 premières phrases")
    print("-"*70)
    print(f"F1-Score:   {results['f1_score']:.3f}")
    print(f"Precision:  {results['precision']:.3f}")
    print(f"Recall:     {results['recall']:.3f}")
    print("="*70)

    print("\nAnalyse:")
    if results['f1_score'] > 0.8:
        print("✓ Excellent: spaCy NER fonctionne très bien sur ce dataset")
    elif results['f1_score'] > 0.6:
        print("✓ Bon: spaCy NER fonctionne correctement")
    else:
        print("✗ Moyen: spaCy a du mal avec ce dataset")

    print("\nConclusion Phase 1.1:")
    print("- spaCy est capable de faire du NER sur des phrases anglaises")
    print("- Prochaine étape: Phase 1.2 - Training sur bottins.csv (français)")
    print("="*70)


if __name__ == "__main__":
    main()
