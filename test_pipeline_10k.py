#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du pipeline complet sur 10k phrases
"""

import sys
import pandas as pd
import time
from datetime import datetime

# Imports
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

def prepare_input_file(input_csv, output_txt, max_rows=10000):
    """Prépare le fichier d'entrée"""
    print("\n[1/4] PREPARATION DU FICHIER D'ENTREE")
    print("="*70)

    df = pd.read_csv(input_csv).head(max_rows)
    print(f"Dataset charge: {len(df)} phrases")

    output_lines = []
    for idx, row in df.iterrows():
        sentence_id = idx + 1
        sentence = row['sentence']
        output_lines.append(f"{sentence_id},{sentence}")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"Fichier d'entree cree: {output_txt}")
    return output_txt

def evaluate_results(output_file, ground_truth_csv, max_rows=10000):
    """Evalue la précision"""
    print("\n[4/4] EVALUATION DES RESULTATS")
    print("="*70)

    gt_df = pd.read_csv(ground_truth_csv).head(max_rows)
    gt_dict = {}
    for idx, row in gt_df.iterrows():
        sentence_id = idx + 1
        gt_dict[sentence_id] = {
            "departure": row['departure'].strip().lower(),
            "arrival": row['arrival'].strip().lower()
        }

    with open(output_file, 'r', encoding='utf-8') as f:
        results = [line.strip() for line in f.readlines()]

    correct_departure = 0
    correct_arrival = 0
    correct_both = 0
    invalid_count = 0

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

    total = len(results)
    valid = total - invalid_count

    print("\nMETRIQUES DE PERFORMANCE:")
    print(f"   Total de phrases: {total}")
    print(f"   Phrases valides: {valid} ({100*valid/total:.1f}%)")
    print(f"   Phrases invalides: {invalid_count} ({100*invalid_count/total:.1f}%)")

    if valid > 0:
        print(f"\n   Precision depart: {100*correct_departure/valid:.1f}% ({correct_departure}/{valid})")
        print(f"   Precision arrivee: {100*correct_arrival/valid:.1f}% ({correct_arrival}/{valid})")
        print(f"   Precision complete (depart + arrivee): {100*correct_both/valid:.1f}% ({correct_both}/{valid})")

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid_count,
        "accuracy_departure": 100*correct_departure/valid if valid > 0 else 0,
        "accuracy_arrival": 100*correct_arrival/valid if valid > 0 else 0,
        "accuracy_both": 100*correct_both/valid if valid > 0 else 0
    }

if __name__ == "__main__":
    print("="*70)
    print("TEST DU PIPELINE SUR 10 000 PHRASES")
    print("="*70)
    print(f"Heure de debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Chemins
    input_csv = "./data/annotated/train_sentences_10k.csv"
    input_txt = "./data/processed/input_10k.txt"
    output_txt = "./results/output_pipeline_10k.txt"

    # Preparer l'entree
    prepare_input_file(input_csv, input_txt, max_rows=10000)

    # Initialiser le pipeline
    print("\n[2/4] INITIALISATION DU PIPELINE")
    print("="*70)
    pipeline = TravelOrderPipeline(use_rag=True)

    # Traiter le fichier
    print("\n[3/4] TRAITEMENT DES 10 000 PHRASES")
    print("="*70)
    print("Estimation: 5-6 heures avec GPU CUDA...")
    print("(~0.28 phrases/sec)")

    pipeline.process_file(input_txt, output_txt, metric="time")

    # Evaluer
    metrics = evaluate_results(output_txt, input_csv, max_rows=10000)

    # Temps total
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "="*70)
    print("TEST 10K TERMINE")
    print("="*70)
    print(f"Heure de fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temps total: {hours}h {minutes}m {seconds}s")
    print(f"Vitesse moyenne: {10000/elapsed:.2f} phrases/seconde")
    print(f"\nFichier de sortie: {output_txt}")
    print(f"Precision complete: {metrics['accuracy_both']:.1f}%")
    print("="*70)
