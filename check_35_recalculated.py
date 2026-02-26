#!/usr/bin/env python3
"""Vérifier les 35 phrases recalculées"""
import csv

error_ids = [583, 673, 1370, 1372, 1466, 2752, 2869, 2876, 3008, 3176, 3645, 3875,
             4053, 4713, 4741, 5038, 5041, 5157, 6008, 6014, 6129, 6268, 7210, 7214,
             7535, 8477, 8578, 8613, 8636, 8917, 8916, 9268, 9387, 9656, 9689]

# Charger le ground truth
gt_dict = {}
with open('./data/annotated/train_sentences_10k.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=1):
        if idx > 10000:
            break
        gt_dict[idx] = {
            "departure": row['departure'].strip().lower(),
            "arrival": row['arrival'].strip().lower()
        }

# Charger les résultats
results_dict = {}
with open('./results/output_pipeline_10k.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',')
            sentence_id = int(parts[0])
            results_dict[sentence_id] = parts

# Vérifier les 35 IDs
correct = 0
incorrect = []

for error_id in error_ids:
    result = results_dict.get(error_id)
    gt = gt_dict.get(error_id)

    if not result or not gt:
        continue

    if len(result) < 2 or result[1] == 'INVALID':
        # Check si c'est un INVALID valide
        incorrect.append({
            'id': error_id,
            'result': 'INVALID',
            'gt': f"{gt['departure']} → {gt['arrival']}"
        })
        continue

    pred_dep = result[1].strip().lower()
    pred_arr = result[-1].strip().lower()

    if pred_dep == gt['departure'] and pred_arr == gt['arrival']:
        correct += 1
    else:
        incorrect.append({
            'id': error_id,
            'pred': f"{pred_dep} → {pred_arr}",
            'gt': f"{gt['departure']} → {gt['arrival']}"
        })

print("="*100)
print("VÉRIFICATION DES 35 PHRASES RECALCULÉES")
print("="*100)
print(f"\nRésultat: {correct}/{len(error_ids)} correct ({100*correct/len(error_ids):.1f}%)")
print(f"Erreurs: {len(incorrect)}")

if incorrect:
    print("\n" + "="*100)
    print("ERREURS DÉTAILLÉES")
    print("="*100)
    for err in incorrect:
        print(f"\nID {err['id']}:")
        print(f"  GT:   {err['gt']}")
        print(f"  PRED: {err.get('result', err.get('pred', '?'))}")

print("\n" + "="*100)
