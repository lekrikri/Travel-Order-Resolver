#!/usr/bin/env python3
"""Évaluation finale de la précision après corrections"""
import csv

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
results = []
with open('./results/output_pipeline_10k.txt', 'r', encoding='utf-8') as f:
    results = [line.strip() for line in f.readlines()]

# Compter les résultats
correct_departure = 0
correct_arrival = 0
correct_both = 0
invalid_count = 0
errors = []

for result_line in results:
    parts = result_line.split(',')
    sentence_id = int(parts[0])

    if len(parts) < 2 or parts[1] == 'INVALID':
        invalid_count += 1
        continue

    predicted_departure = parts[1].strip().lower()
    predicted_arrival = parts[-1].strip().lower()

    gt = gt_dict.get(sentence_id)
    if gt:
        dep_ok = predicted_departure == gt["departure"]
        arr_ok = predicted_arrival == gt["arrival"]

        if dep_ok:
            correct_departure += 1
        if arr_ok:
            correct_arrival += 1
        if dep_ok and arr_ok:
            correct_both += 1
        else:
            errors.append({
                'id': sentence_id,
                'gt_dep': gt["departure"],
                'gt_arr': gt["arrival"],
                'pred_dep': predicted_departure,
                'pred_arr': predicted_arrival
            })

total = len(results)
valid = total - invalid_count

print("="*100)
print("ÉVALUATION FINALE - APRÈS CORRECTIONS")
print("="*100)
print(f"\nTotal de phrases: {total}")
print(f"Phrases valides: {valid} ({100*valid/total:.2f}%)")
print(f"Phrases invalides: {invalid_count} ({100*invalid_count/total:.2f}%)")

if valid > 0:
    print(f"\nPrécision départ: {100*correct_departure/valid:.2f}% ({correct_departure}/{valid})")
    print(f"Précision arrivée: {100*correct_arrival/valid:.2f}% ({correct_arrival}/{valid})")
    print(f"Précision complète (départ + arrivée): {100*correct_both/valid:.2f}% ({correct_both}/{valid})")
    print(f"\nErreurs restantes: {len(errors)} ({100*len(errors)/valid:.2f}%)")

print("\n" + "="*100)
print("COMPARAISON AVANT/APRÈS")
print("="*100)
print(f"Avant corrections: 99.60% (8,712/8,747 correct)")
print(f"Après corrections: {100*correct_both/valid:.2f}% ({correct_both}/{valid} correct)")
print(f"Amélioration: +{correct_both - 8712} phrases corrigées")

if len(errors) > 0 and len(errors) <= 10:
    print("\n" + "="*100)
    print(f"ERREURS RESTANTES ({len(errors)})")
    print("="*100)
    for err in errors:
        print(f"\nID {err['id']}:")
        print(f"  GT:   {err['gt_dep']} → {err['gt_arr']}")
        print(f"  PRED: {err['pred_dep']} → {err['pred_arr']}")

print("\n" + "="*100)
