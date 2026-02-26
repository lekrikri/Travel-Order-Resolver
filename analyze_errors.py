#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse des 35 erreurs du test 10k"""

import pandas as pd

# Charger le ground truth
gt_df = pd.read_csv('./data/annotated/train_sentences_10k.csv').head(10000)
gt_dict = {}
for idx, row in gt_df.iterrows():
    sentence_id = idx + 1
    gt_dict[sentence_id] = {
        'departure': row['departure'].strip().lower(),
        'arrival': row['arrival'].strip().lower(),
        'sentence': row['sentence']
    }

# Charger les résultats
with open('./results/output_pipeline_10k.txt', 'r', encoding='utf-8') as f:
    results = [line.strip() for line in f.readlines()]

# Trouver les erreurs et les catégoriser
inversions = []
partial_errors = []
complete_errors = []

for result_line in results:
    parts = result_line.split(',')
    sentence_id = int(parts[0])

    if len(parts) < 2 or parts[1] == 'INVALID':
        continue

    predicted_departure = parts[1].strip().lower()
    predicted_arrival = parts[-1].strip().lower()

    gt = gt_dict.get(sentence_id)
    if gt:
        dep_ok = predicted_departure == gt['departure']
        arr_ok = predicted_arrival == gt['arrival']

        if not (dep_ok and arr_ok):
            # Vérifier si c'est une inversion
            if predicted_departure == gt['arrival'] and predicted_arrival == gt['departure']:
                inversions.append({
                    'id': sentence_id,
                    'sentence': gt['sentence'],
                    'gt_dep': gt['departure'],
                    'gt_arr': gt['arrival'],
                    'pred_dep': predicted_departure,
                    'pred_arr': predicted_arrival
                })
            elif dep_ok or arr_ok:
                partial_errors.append({
                    'id': sentence_id,
                    'sentence': gt['sentence'],
                    'gt_dep': gt['departure'],
                    'gt_arr': gt['arrival'],
                    'pred_dep': predicted_departure,
                    'pred_arr': predicted_arrival,
                    'error_type': 'DEPART' if not dep_ok else 'ARRIVEE'
                })
            else:
                complete_errors.append({
                    'id': sentence_id,
                    'sentence': gt['sentence'],
                    'gt_dep': gt['departure'],
                    'gt_arr': gt['arrival'],
                    'pred_dep': predicted_departure,
                    'pred_arr': predicted_arrival
                })

print('='*100)
print('ANALYSE DES 35 ERREURS')
print('='*100)
print()
print(f'1. INVERSIONS (départ ↔ arrivée): {len(inversions)} erreurs')
print(f'2. ERREURS PARTIELLES (un seul côté): {len(partial_errors)} erreurs')
print(f'3. ERREURS COMPLETES (les deux côtés): {len(complete_errors)} erreurs')
print()

print('='*100)
print(f'ERREURS PARTIELLES ({len(partial_errors)} erreurs)')
print('='*100)
for err in partial_errors:
    print(f'ID {err["id"]}: {err["sentence"]}')
    print(f'  GT:   {err["gt_dep"]} → {err["gt_arr"]}')
    print(f'  PRED: {err["pred_dep"]} → {err["pred_arr"]}')
    print(f'  ❌ {err["error_type"]} INCORRECT')
    print()

print('='*100)
print(f'ERREURS COMPLETES ({len(complete_errors)} erreurs - Hallucinations NLP)')
print('='*100)
for err in complete_errors:
    print(f'ID {err["id"]}: {err["sentence"]}')
    print(f'  GT:   {err["gt_dep"]} → {err["gt_arr"]}')
    print(f'  PRED: {err["pred_dep"]} → {err["pred_arr"]}')
    print()
