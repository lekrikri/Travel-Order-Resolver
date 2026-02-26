#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rapide du pipeline complet sur 100 phrases - Version simplifiée
"""

import sys
import pandas as pd
import time
from datetime import datetime

# Imports
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

def prepare_input_file(input_csv, output_txt, max_rows=100):
    """Prépare le fichier d'entrée"""
    print("\n[1/4] PREPARATION DU FICHIER D'ENTREE")
    print("="*70)

    df = pd.read_csv(input_csv).head(max_rows)
    print("Dataset charge: {} phrases".format(len(df)))

    output_lines = []
    for idx, row in df.iterrows():
        sentence_id = idx + 1
        sentence = row['sentence']
        output_lines.append("{},{}".format(sentence_id, sentence))

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print("Fichier d'entree cree: {}".format(output_txt))
    return output_txt

def evaluate_results(output_file, ground_truth_csv, max_rows=100):
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
    details = []

    for result_line in results:
        parts = result_line.split(',')
        sentence_id = int(parts[0])

        if len(parts) < 2 or parts[1] == 'INVALID':
            invalid_count += 1
            details.append("ID {}: INVALID".format(sentence_id))
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

            status = "OK" if (dep_ok and arr_ok) else "KO"
            details.append("{} ID {}: {} -> {} (GT: {} -> {})".format(
                status, sentence_id, predicted_departure, predicted_arrival,
                gt['departure'], gt['arrival']
            ))

    total = len(results)
    valid = total - invalid_count

    print("\nMETRIQUES DE PERFORMANCE:")
    print("   Total de phrases: {}".format(total))
    print("   Phrases valides: {} ({:.1f}%)".format(valid, 100*valid/total))
    print("   Phrases invalides: {} ({:.1f}%)".format(invalid_count, 100*invalid_count/total))

    if valid > 0:
        print("\n   Precision depart: {:.1f}% ({}/{})".format(
            100*correct_departure/valid, correct_departure, valid))
        print("   Precision arrivee: {:.1f}% ({}/{})".format(
            100*correct_arrival/valid, correct_arrival, valid))
        print("   Precision complete (depart + arrivee): {:.1f}% ({}/{})".format(
            100*correct_both/valid, correct_both, valid))

    print("\nPREMIERS RESULTATS (10 premiers):")
    for detail in details[:10]:
        print("   {}".format(detail))

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
    print("TEST RAPIDE DU PIPELINE SUR 100 PHRASES")
    print("="*70)
    print("Heure de debut: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    start_time = time.time()

    # Chemins
    input_csv = "./data/annotated/train_sentences_10k.csv"
    input_txt = "./data/processed/input_100.txt"
    output_txt = "./results/output_pipeline_100.txt"

    # Preparer l'entree
    prepare_input_file(input_csv, input_txt, max_rows=100)

    # Initialiser le pipeline
    print("\n[2/4] INITIALISATION DU PIPELINE")
    print("="*70)
    pipeline = TravelOrderPipeline(use_rag=True)

    # Traiter le fichier
    print("\n[3/4] TRAITEMENT DES 100 PHRASES")
    print("="*70)
    print("Estimation: 3-5 minutes...")

    pipeline.process_file(input_txt, output_txt, metric="time")

    # Evaluer
    metrics = evaluate_results(output_txt, input_csv, max_rows=100)

    # Temps total
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "="*70)
    print("TEST RAPIDE TERMINE")
    print("="*70)
    print("Heure de fin: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("Temps total: {}m {}s".format(minutes, seconds))
    print("Vitesse moyenne: {:.2f} phrases/seconde".format(100/elapsed))
    print("\nFichier de sortie: {}".format(output_txt))
    print("Precision complete: {:.1f}%".format(metrics['accuracy_both']))
    print("="*70)
