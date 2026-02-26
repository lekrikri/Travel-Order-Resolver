#!/usr/bin/env python3
"""Recalcule uniquement les 35 phrases corrigées"""

import sys
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

# Les 35 IDs à recalculer
error_ids = [583, 673, 1370, 1372, 1466, 2752, 2869, 2876, 3008, 3176, 3645, 3875,
             4053, 4713, 4741, 5038, 5041, 5157, 6008, 6014, 6129, 6268, 7210, 7214,
             7535, 8477, 8578, 8613, 8636, 8917, 8916, 9268, 9387, 9656, 9689]

print("="*100)
print("RECALCUL DES 35 PHRASES CORRIGÉES")
print("="*100)

# Charger les phrases depuis le fichier d'entrée
sentences = {}
with open('./data/processed/input_10k.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',', 1)
            if len(parts) == 2:
                sentence_id = int(parts[0])
                sentence = parts[1]
                sentences[sentence_id] = sentence

print(f"\nPhrases chargées: {len(sentences)}")
print(f"IDs à recalculer: {len(error_ids)}")

# Initialiser le pipeline
print("\nInitialisation du pipeline...")
pipeline = TravelOrderPipeline(use_rag=True)

# Recalculer chaque phrase
results = []
errors = 0

for i, sentence_id in enumerate(error_ids, 1):
    if sentence_id not in sentences:
        print(f"\n[{i}/{len(error_ids)}] ❌ ID {sentence_id} non trouvé dans input")
        continue

    sentence = sentences[sentence_id]
    print(f"\n[{i}/{len(error_ids)}] ID {sentence_id}")
    print(f"   Phrase: {sentence[:80]}...")

    result = pipeline.process_sentence(sentence_id, sentence, metric="time")

    if result["error"]:
        print(f"   ❌ Erreur: {result['error']}")
        errors += 1
    else:
        nlp = result["nlp_result"]
        route = result["route_result"]
        print(f"   ✅ NLP: {nlp['departure']} → {nlp['arrival']}")
        print(f"   ✅ Route: {route['num_stops']} gares, {route['total_time_minutes']:.0f} min")

    results.append((sentence_id, result["final_output"]))

# Sauvegarder les résultats
print("\n" + "="*100)
print("MISE À JOUR DU FICHIER OUTPUT")
print("="*100)

# Lire le fichier actuel
existing_lines = {}
with open('./results/output_pipeline_10k.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sentence_id = int(line.split(',')[0])
            existing_lines[sentence_id] = line

# Ajouter les nouveaux résultats
for sentence_id, final_output in results:
    existing_lines[sentence_id] = final_output

# Réécrire le fichier dans l'ordre
with open('./results/output_pipeline_10k.txt', 'w', encoding='utf-8') as f:
    for sentence_id in sorted(existing_lines.keys()):
        f.write(existing_lines[sentence_id] + '\n')

print(f"\n✅ Fichier mis à jour: ./results/output_pipeline_10k.txt")
print(f"   Total lignes: {len(existing_lines)}")
print(f"   Recalculées: {len(results)}")
print(f"   Erreurs: {errors}")

print("\n" + "="*100)
print("RECALCUL TERMINÉ")
print("="*100)
