#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test de toutes les 35 erreurs avec les corrections appliquées"""

import sys
import pandas as pd
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

# Les 35 IDs d'erreurs identifiées
error_ids = [583, 673, 1370, 1372, 1466, 2752, 2869, 2876, 3008, 3176, 3645, 3875,
             4053, 4713, 4741, 5038, 5041, 5157, 6008, 6014, 6129, 6268, 7210, 7214,
             7535, 8477, 8578, 8613, 8636, 8917, 8916, 9268, 9387, 9656, 9689]

# Charger les phrases et le ground truth
gt_df = pd.read_csv('./data/annotated/train_sentences_10k.csv').head(10000)

print("="*100)
print("TEST DES 35 ERREURS APRÈS CORRECTIONS")
print("="*100)

# Initialiser le pipeline
print("\nInitialisation du pipeline...")
pipeline = TravelOrderPipeline(use_rag=True)

print("\n" + "="*100)
print("TRAITEMENT")
print("="*100)

results = []
inversions_fixed = 0
still_errors = 0
now_correct = 0

for idx, error_id in enumerate(error_ids, 1):
    # Récupérer la phrase et le GT
    row = gt_df.iloc[error_id - 1]
    sentence = row['sentence']
    gt_dep = row['departure'].strip().lower()
    gt_arr = row['arrival'].strip().lower()

    print(f"\n[{idx}/35] ID {error_id}: {sentence[:80]}...")

    result = pipeline.process_sentence(error_id, sentence, metric="time")

    if result["error"]:
        print(f"  ❌ ERREUR: {result['error']}")
        still_errors += 1
    else:
        nlp = result["nlp_result"]
        pred_dep = nlp['departure'].lower()
        pred_arr = nlp['arrival'].lower()

        # Récupérer la route finale (après normalization)
        final_output = result["final_output"]
        parts = final_output.split(',')
        final_dep = parts[1].strip().lower() if len(parts) > 1 else ""
        final_arr = parts[-1].strip().lower() if len(parts) > 1 else ""

        inversion_fixed = nlp.get("inversion_corrected", False)

        is_correct = (final_dep == gt_dep and final_arr == gt_arr)

        if inversion_fixed:
            print(f"  🔄 INVERSION CORRIGÉE")
            inversions_fixed += 1

        print(f"  GT:   {gt_dep} → {gt_arr}")
        print(f"  PRED: {final_dep} → {final_arr}")

        if is_correct:
            print(f"  ✅ MAINTENANT CORRECT !")
            now_correct += 1
        else:
            print(f"  ❌ TOUJOURS INCORRECT")
            still_errors += 1

print("\n" + "="*100)
print("RÉSULTATS")
print("="*100)
print(f"Total d'erreurs testées: 35")
print(f"Inversions corrigées: {inversions_fixed}")
print(f"Maintenant correctes: {now_correct}")
print(f"Toujours incorrectes: {still_errors}")
print(f"Amélioration: +{now_correct} erreurs corrigées ({100*now_correct/35:.1f}%)")
print(f"Nouvelles erreurs restantes: {35 - now_correct}")
print("="*100)
