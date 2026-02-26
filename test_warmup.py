#!/usr/bin/env python3
"""
Test du warm-up avec mini dataset
"""

import sys
sys.path.insert(0, './src')

from 1_spacy_baseline.warmup import SpacyNERWarmup

def main():
    print("="*70)
    print("TEST WARM-UP - Phase 1.1")
    print("="*70)

    # Initialiser
    print("\n1. Initialisation de spaCy...")
    warmup = SpacyNERWarmup(model_name="fr_core_news_md")

    # Charger le mini dataset
    print("\n2. Chargement du dataset...")
    df = warmup.load_ner_dataset("./data/raw/ner_dataset_mini.csv")

    # Extraire les phrases
    print("\n3. Extraction des phrases...")
    sentences = warmup.extract_sentences(df)
    print(f"   Phrases extraites: {len(sentences)}")

    # Afficher quelques exemples
    print("\n4. Exemples de phrases:")
    for i, (text, tags) in enumerate(sentences[:3]):
        print(f"   {i+1}. {text}")
        print(f"      Tags: {' '.join(tags)}")

    # Évaluer
    print("\n5. Evaluation...")
    results = warmup.evaluate(sentences)

    print("\n" + "="*70)
    print("RESULTATS WARM-UP - spaCy NER (mini dataset)")
    print("="*70)
    print(f"F1-Score:  {results['f1_score']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print("="*70)

    print("\nNOTE: Ceci est un test avec un mini dataset (3 phrases).")
    print("Pour tester sur le dataset complet, téléchargez:")
    print("https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus")
    print("="*70)


if __name__ == "__main__":
    main()
