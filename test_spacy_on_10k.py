#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de spaCy sur le dataset de 10,000 phrases
Évaluation de l'extraction départ/arrivée
"""

import csv
import re
import spacy
import sys
import io
from collections import Counter

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("TEST SPACY SUR DATASET 10K")
print("="*70)

# ============================================================================
# CHARGEMENT
# ============================================================================

print("\n[1/5] Chargement du dataset 10k...")

sentences = []
with open('./data/annotated/train_sentences_10k.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row)

print(f"   OK {len(sentences):,} phrases chargées")

# Limiter à 1000 pour la rapidité (test sur échantillon)
test_size = 1000
sentences = sentences[:test_size]
print(f"   Test sur {test_size} premières phrases (échantillon)")

# ============================================================================
# CHARGEMENT SPACY
# ============================================================================

print("\n[2/5] Chargement de spaCy fr_core_news_md...")

try:
    nlp = spacy.load("fr_core_news_md")
    print("   OK Modèle chargé")
except:
    print("   ERREUR: fr_core_news_md non installé")
    print("   Installation: python -m spacy download fr_core_news_md")
    sys.exit(1)

# ============================================================================
# EXTRACTION AVEC SPACY
# ============================================================================

print("\n[3/5] Extraction des villes avec spaCy...")

def extract_cities_spacy(text):
    """Extraire les villes avec spaCy NER"""
    # Enlever les tags d'annotation pour avoir le texte brut
    clean_text = re.sub(r'<[^>]+>([^<]+)</[^>]+>', r'\1', text)

    doc = nlp(clean_text)
    cities = []

    for ent in doc.ents:
        if ent.label_ in ['LOC', 'GPE']:  # Lieux et entités géopolitiques
            cities.append(ent.text)

    return cities

results = []
for i, row in enumerate(sentences):
    sentence = row['sentence']
    gold_dep = row['departure']
    gold_arr = row['arrival']

    # Extraire avec spaCy
    cities = extract_cities_spacy(sentence)

    # Essayer de déterminer départ/arrivée
    # Stratégie simple: premier = départ, dernier = arrivée
    if len(cities) >= 2:
        pred_dep = cities[0]
        pred_arr = cities[-1]
    elif len(cities) == 1:
        pred_dep = cities[0]
        pred_arr = None
    else:
        pred_dep = None
        pred_arr = None

    results.append({
        'sentence': sentence,
        'gold_dep': gold_dep,
        'gold_arr': gold_arr,
        'pred_dep': pred_dep,
        'pred_arr': pred_arr,
        'cities_found': cities,
        'correct_dep': pred_dep == gold_dep if pred_dep else False,
        'correct_arr': pred_arr == gold_arr if pred_arr else False,
        'both_correct': (pred_dep == gold_dep and pred_arr == gold_arr) if (pred_dep and pred_arr) else False
    })

    if (i + 1) % 200 == 0:
        print(f"   {i+1}/{test_size} phrases traitées...")

print(f"   OK {len(results)} phrases analysées")

# ============================================================================
# CALCUL DES MÉTRIQUES
# ============================================================================

print("\n[4/5] Calcul des métriques...")

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

print(f"   OK Métriques calculées")

# ============================================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================================

print("\n[5/5] Génération du rapport...")

print("\n" + "="*70)
print("RÉSULTATS - spaCy sur Dataset 10k")
print("="*70)
print(f"Dataset testé:     {test_size:,} phrases (échantillon)")
print(f"Modèle:            spaCy fr_core_news_md")
print(f"Tâche:             Extraction départ/arrivée")
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

print("\n" + "="*70)
print("ANALYSE DES ERREURS")
print("="*70)

# Analyser les types d'erreurs
errors = [r for r in results if not r['both_correct']]

# Compter les problèmes
no_cities_found = sum(1 for r in errors if len(r['cities_found']) == 0)
one_city_found = sum(1 for r in errors if len(r['cities_found']) == 1)
wrong_order = sum(1 for r in errors if len(r['cities_found']) >= 2 and r['pred_dep'] == r['gold_arr'] and r['pred_arr'] == r['gold_dep'])
partial_match = sum(1 for r in errors if r['correct_dep'] or r['correct_arr'])

print(f"\nTypes d'erreurs ({len(errors)} erreurs sur {len(results)}):")
print(f"  - Aucune ville détectée:     {no_cities_found} ({no_cities_found*100//len(errors) if errors else 0}%)")
print(f"  - Une seule ville détectée:  {one_city_found} ({one_city_found*100//len(errors) if errors else 0}%)")
print(f"  - Ordre inversé:             {wrong_order} ({wrong_order*100//len(errors) if errors else 0}%)")
print(f"  - Correspondance partielle:  {partial_match} ({partial_match*100//len(errors) if errors else 0}%)")

print("\n" + "="*70)
print("EXEMPLES DE RÉUSSITES (10 premiers)")
print("="*70)

successes = [r for r in results if r['both_correct']][:10]
for i, r in enumerate(successes, 1):
    clean_sentence = re.sub(r'<[^>]+>', '', r['sentence'])
    print(f"\n{i}. {clean_sentence[:70]}...")
    print(f"   Gold: {r['gold_dep']} -> {r['gold_arr']}")
    print(f"   Pred: {r['pred_dep']} -> {r['pred_arr']} OK")

print("\n" + "="*70)
print("EXEMPLES D'ERREURS (10 premiers)")
print("="*70)

for i, r in enumerate(errors[:10], 1):
    clean_sentence = re.sub(r'<[^>]+>', '', r['sentence'])
    print(f"\n{i}. {clean_sentence[:70]}...")
    print(f"   Gold: {r['gold_dep']} -> {r['gold_arr']}")
    print(f"   Pred: {r['pred_dep']} -> {r['pred_arr']}")
    print(f"   Villes trouvées: {r['cities_found']}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if accuracy > 0.7:
    print(f"\nEXCELLENT: spaCy fonctionne bien ({accuracy:.1%} accuracy)")
elif accuracy > 0.5:
    print(f"\nBON: spaCy fonctionne correctement ({accuracy:.1%} accuracy)")
elif accuracy > 0.3:
    print(f"\nMOYEN: spaCy a des limites ({accuracy:.1%} accuracy)")
else:
    print(f"\nFAIBLE: spaCy n'est pas adapté à cette tâche ({accuracy:.1%} accuracy)")

print("\nPoints forts de spaCy:")
print("  + Rapide et simple à utiliser")
print("  + Pas besoin d'entraînement")
print("  + Détecte bien les noms de villes connues")

print("\nLimitations observées:")
print("  - Difficulté avec l'ordre départ/arrivée")
print("  - Confusion sur certaines villes (banlieues, noms composés)")
print("  - Pas de compréhension du contexte linguistique")

print("\nRecommandation:")
print("  -> Passer à Phase 2 avec Qwen2.5 + ChromaDB")
print("  -> Les LLMs comprendront mieux le contexte et l'ordre")
print("="*70)

# Sauvegarder les résultats
print("\nSauvegarde des résultats détaillés...")
output_file = "./results/spacy_10k_results.csv"
import os
os.makedirs("./results", exist_ok=True)

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'gold_dep', 'gold_arr', 'pred_dep', 'pred_arr', 'cities_found', 'correct'])
    for r in results:
        writer.writerow([
            r['sentence'],
            r['gold_dep'],
            r['gold_arr'],
            r['pred_dep'],
            r['pred_arr'],
            '; '.join(r['cities_found']),
            'OK' if r['both_correct'] else 'ERREUR'
        ])

print(f"OK Résultats sauvegardés: {output_file}")
print("="*70)
