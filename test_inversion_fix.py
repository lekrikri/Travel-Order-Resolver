#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test de la correction des inversions sur les 7 erreurs identifiées"""

import sys
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

# Les 7 phrases avec inversions
test_phrases = [
    (583, "<Dep>Montauban</Dep> -> <Arr>Avignon</Arr>"),
    (2876, "<Dep>Versailles</Dep> -> <Arr>Courbevoie</Arr>"),
    (3176, "Vite, un train <Dep>Nice</Dep> <Arr>Angers</Arr>"),
    (3645, "<Dep>Nice</Dep> -> <Arr>Cholet</Arr>"),
    (6129, "Réserver pour mardi en fin d'après-midi : <Dep>Montauban</Dep> vers <Arr>Cherbourg</Arr>"),
    (8477, "Avec mes parents, train <Dep>Dijon</Dep> <Arr>Drancy</Arr> dans 2 jours"),
    (9689, "Billet <Dep>Lille</Dep> <Arr>Pantin</Arr> en début de semaine Pour un mariage"),
]

print("="*100)
print("TEST DE CORRECTION DES INVERSIONS")
print("="*100)

# Initialiser le pipeline
print("\nInitialisation du pipeline...")
pipeline = TravelOrderPipeline(use_rag=True)

print("\n" + "="*100)
print("TRAITEMENT DES 7 PHRASES AVEC INVERSION")
print("="*100)

for sentence_id, sentence in test_phrases:
    print(f"\n[ID {sentence_id}] {sentence}")

    result = pipeline.process_sentence(sentence_id, sentence, metric="time")

    if result["error"]:
        print(f"  ❌ ERREUR: {result['error']}")
    else:
        nlp = result["nlp_result"]
        route = result["route_result"]

        inversion_fixed = nlp.get("inversion_corrected", False)

        print(f"  📍 NLP: {nlp['departure']} → {nlp['arrival']}")
        if inversion_fixed:
            print(f"  ✅ INVERSION CORRIGÉE AUTOMATIQUEMENT")
        print(f"  🛤️  Route: {route['num_stops']} gares, {route['total_time_minutes']:.0f} min")
        print(f"  📄 Output: {result['final_output'][:100]}...")

print("\n" + "="*100)
print("TEST TERMINÉ")
print("="*100)
