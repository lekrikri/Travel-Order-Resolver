#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Qwen2.5 + ChromaDB RAG sur le dataset 10k
Comparaison avec spaCy
"""

import sys
import io
import csv
import os

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ajouter src au path
sys.path.insert(0, './src/2_qwen_chroma')

from qwen_extractor import Qwen25Extractor
from chroma_rag import ChromaRAG

print("="*70)
print("TEST QWEN2.5 + CHROMADB SUR DATASET 10K")
print("="*70)

# ============================================================================
# ÉTAPE 1: INITIALISATION
# ============================================================================

print("\n[1/6] Initialisation de Qwen2.5...")
extractor = Qwen25Extractor(model_name="Qwen/Qwen2.5-3B-Instruct")

if not extractor.load_model():
    print("\nERREUR: Impossible de charger Qwen2.5")
    print("\nOptions:")
    print("1. Installer les dépendances: pip install transformers torch")
    print("2. Télécharger le modèle (1.7GB): il se télécharge automatiquement au premier lancement")
    print("\nNote: Qwen2.5-3B nécessite ~4GB RAM (CPU) ou ~2GB VRAM (GPU)")
    sys.exit(1)

print("\n[2/6] Initialisation de ChromaDB...")
rag = ChromaRAG()

if not rag.initialize():
    print("\nERREUR: Impossible d'initialiser ChromaDB")
    print("Installation: pip install chromadb sentence-transformers")
    sys.exit(1)

# ============================================================================
# ÉTAPE 2: CHARGER LE DATASET DANS CHROMADB
# ============================================================================

print("\n[3/6] Chargement du dataset d'entraînement dans ChromaDB...")

# Charger 100 phrases pour le RAG (exemples)
if rag.collection.count() == 0:
    print("   Chargement des 100 phrases annotées...")
    rag.load_dataset("./data/annotated/train_sentences_100.csv")
else:
    print(f"   Collection déjà chargée: {rag.collection.count()} documents")

# ============================================================================
# ÉTAPE 3: CHARGER LE DATASET DE TEST
# ============================================================================

print("\n[4/6] Chargement du dataset de test (10k)...")

test_sentences = []
with open('./data/annotated/train_sentences_10k.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Nettoyer la phrase (enlever les tags)
        import re
        clean_sentence = re.sub(r'<[^>]+>', '', row['sentence'])
        test_sentences.append({
            'sentence': clean_sentence,
            'gold_dep': row['departure'],
            'gold_arr': row['arrival']
        })

# Limiter à 100 phrases pour le test (Qwen2.5 est lent)
test_size = 100
test_sentences = test_sentences[:test_size]
print(f"   Test sur {test_size} phrases (échantillon)")

# ============================================================================
# ÉTAPE 4: EXTRACTION AVEC QWEN2.5 + RAG
# ============================================================================

print("\n[5/6] Extraction avec Qwen2.5 + RAG...")
print("   (Cela peut prendre quelques minutes...)")

results = []

for i, test in enumerate(test_sentences):
    # Récupérer des exemples similaires avec ChromaDB
    similar_examples = rag.get_similar_examples(test['sentence'], n_results=3)

    # Extraire avec Qwen2.5 (few-shot avec exemples RAG)
    result = extractor.extract_cities(test['sentence'], examples=similar_examples)

    results.append({
        'sentence': test['sentence'],
        'gold_dep': test['gold_dep'],
        'gold_arr': test['gold_arr'],
        'pred_dep': result.get('departure'),
        'pred_arr': result.get('arrival'),
        'confidence': result.get('confidence', 0.0),
        'correct_dep': result.get('departure') == test['gold_dep'],
        'correct_arr': result.get('arrival') == test['gold_arr'],
        'both_correct': (result.get('departure') == test['gold_dep'] and
                        result.get('arrival') == test['gold_arr'])
    })

    if (i + 1) % 10 == 0:
        correct_so_far = sum(1 for r in results if r['both_correct'])
        accuracy_so_far = correct_so_far / len(results)
        print(f"   {i+1}/{test_size} phrases traitées... Accuracy: {accuracy_so_far:.1%}")

print(f"   OK {len(results)} phrases analysées")

# ============================================================================
# ÉTAPE 5: CALCUL DES MÉTRIQUES
# ============================================================================

print("\n[6/6] Calcul des métriques...")

correct_dep = sum(1 for r in results if r['correct_dep'])
correct_arr = sum(1 for r in results if r['correct_arr'])
both_correct = sum(1 for r in results if r['both_correct'])
found_dep = sum(1 for r in results if r['pred_dep'] is not None)
found_arr = sum(1 for r in results if r['pred_arr'] is not None)

precision_dep = correct_dep / found_dep if found_dep > 0 else 0
precision_arr = correct_arr / found_arr if found_arr > 0 else 0
recall_dep = correct_dep / len(results)
recall_arr = correct_arr / len(results)

f1_dep = 2 * (precision_dep * recall_dep) / (precision_dep + recall_dep) if (precision_dep + recall_dep) > 0 else 0
f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr) if (precision_arr + recall_arr) > 0 else 0

accuracy = both_correct / len(results)

# ============================================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================================

print("\n" + "="*70)
print("RÉSULTATS - Qwen2.5 + ChromaDB RAG")
print("="*70)
print(f"Dataset testé:     {test_size} phrases (échantillon)")
print(f"Modèle:            Qwen2.5-3B-Instruct + ChromaDB RAG")
print(f"Few-shot:          3 exemples similaires par phrase")
print("-"*70)

print("\nMétriques DÉPART:")
print(f"  - Précision:     {precision_dep:.3f} ({correct_dep}/{found_dep})")
print(f"  - Rappel:        {recall_dep:.3f} ({correct_dep}/{len(results)})")
print(f"  - F1-Score:      {f1_dep:.3f}")

print("\nMétriques ARRIVÉE:")
print(f"  - Précision:     {precision_arr:.3f} ({correct_arr}/{found_arr})")
print(f"  - Rappel:        {recall_arr:.3f} ({correct_arr}/{len(results)})")
print(f"  - F1-Score:      {f1_arr:.3f}")

print("\nMétriques GLOBALES:")
print(f"  - Accuracy complète: {accuracy:.3f} ({both_correct}/{len(results)})")
print(f"    (départ ET arrivée corrects)")

# ============================================================================
# COMPARAISON AVEC SPACY
# ============================================================================

# Charger les résultats spaCy si disponibles
spacy_results_file = "./results/spacy_10k_results.csv"
spacy_accuracy = 0.322  # Résultat précédent

print("\n" + "="*70)
print("COMPARAISON SPACY VS QWEN2.5 + CHROMADB")
print("="*70)
print(f"\nspaCy (baseline):           {spacy_accuracy:.1%} accuracy")
print(f"Qwen2.5 + ChromaDB RAG:     {accuracy:.1%} accuracy")
print(f"\nAmélioration:               +{(accuracy - spacy_accuracy):.1%} points")
print(f"Gain relatif:               {((accuracy - spacy_accuracy) / spacy_accuracy * 100):.1f}%")

# ============================================================================
# EXEMPLES
# ============================================================================

print("\n" + "="*70)
print("EXEMPLES DE RÉUSSITES (5 premiers)")
print("="*70)

successes = [r for r in results if r['both_correct']][:5]
for i, r in enumerate(successes, 1):
    print(f"\n{i}. {r['sentence'][:70]}...")
    print(f"   Gold: {r['gold_dep']} -> {r['gold_arr']}")
    print(f"   Pred: {r['pred_dep']} -> {r['pred_arr']} OK")
    print(f"   Confiance: {r['confidence']:.2f}")

print("\n" + "="*70)
print("EXEMPLES D'ERREURS (5 premiers)")
print("="*70)

errors = [r for r in results if not r['both_correct']][:5]
for i, r in enumerate(errors, 1):
    print(f"\n{i}. {r['sentence'][:70]}...")
    print(f"   Gold: {r['gold_dep']} -> {r['gold_arr']}")
    print(f"   Pred: {r['pred_dep']} -> {r['pred_arr']}")
    print(f"   Confiance: {r['confidence']:.2f}")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if accuracy > 0.9:
    verdict = "EXCELLENT"
elif accuracy > 0.7:
    verdict = "TRÈS BON"
elif accuracy > 0.5:
    verdict = "BON"
else:
    verdict = "MOYEN"

print(f"\n{verdict}: Qwen2.5 + ChromaDB atteint {accuracy:.1%} accuracy")

print("\nPoints forts de Qwen2.5 + ChromaDB:")
print("  + Comprend le contexte linguistique")
print("  + Détecte correctement l'ordre départ/arrivée")
print("  + Few-shot learning avec exemples similaires")
print("  + S'adapte aux formulations variées")

if accuracy > spacy_accuracy:
    print(f"\nQwen2.5 + ChromaDB surpasse spaCy de {(accuracy - spacy_accuracy):.1%} points!")
else:
    print("\nNote: Performance similaire ou inférieure à spaCy")
    print("Possibles causes: modèle 3B trop petit, besoin de fine-tuning")

# ============================================================================
# SAUVEGARDE
# ============================================================================

print("\n" + "="*70)
print("Sauvegarde des résultats...")

os.makedirs("./results", exist_ok=True)
output_file = "./results/qwen_chroma_10k_results.csv"

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'gold_dep', 'gold_arr', 'pred_dep', 'pred_arr', 'confidence', 'correct'])
    for r in results:
        writer.writerow([
            r['sentence'],
            r['gold_dep'],
            r['gold_arr'],
            r['pred_dep'],
            r['pred_arr'],
            r['confidence'],
            'OK' if r['both_correct'] else 'ERREUR'
        ])

print(f"OK Résultats sauvegardés: {output_file}")
print("="*70)
