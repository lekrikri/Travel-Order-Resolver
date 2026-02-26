#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du pipeline complet sur le dataset 10k phrases
"""

import sys
import io
import pandas as pd
import time
from datetime import datetime

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Imports
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

def prepare_input_file(input_csv: str, output_txt: str):
    """
    Prépare le fichier d'entrée au format attendu par le pipeline
    Format de sortie: sentenceID,sentence (sans en-tête)
    """
    print("\n[1/4] PRÉPARATION DU FICHIER D'ENTRÉE")
    print("="*70)

    # Charger le dataset avec en-tête
    df = pd.read_csv(input_csv)
    print(f"✅ Dataset chargé: {len(df)} phrases")

    # Créer le format attendu: sentenceID,sentence
    output_lines = []
    for idx, row in df.iterrows():
        sentence_id = idx + 1
        sentence = row['sentence']
        output_lines.append(f"{sentence_id},{sentence}")

    # Sauvegarder
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"✅ Fichier d'entrée créé: {output_txt}")
    return output_txt

def evaluate_results(output_file: str, ground_truth_csv: str):
    """
    Évalue la précision du pipeline en comparant avec les annotations
    """
    print("\n[4/4] ÉVALUATION DES RÉSULTATS")
    print("="*70)

    # Charger les ground truth
    gt_df = pd.read_csv(ground_truth_csv)
    gt_dict = {}
    for idx, row in gt_df.iterrows():
        sentence_id = idx + 1
        gt_dict[sentence_id] = {
            "departure": row['departure'].strip().lower(),
            "arrival": row['arrival'].strip().lower()
        }

    # Charger les résultats
    with open(output_file, 'r', encoding='utf-8') as f:
        results = [line.strip() for line in f.readlines()]

    # Comparer
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

        # Extraire départ et arrivée prédits
        predicted_departure = parts[1].strip().lower()
        predicted_arrival = parts[-1].strip().lower()

        # Comparer avec ground truth
        gt = gt_dict.get(sentence_id)
        if gt:
            if predicted_departure == gt["departure"]:
                correct_departure += 1
            if predicted_arrival == gt["arrival"]:
                correct_arrival += 1
            if predicted_departure == gt["departure"] and predicted_arrival == gt["arrival"]:
                correct_both += 1

    # Calculer les métriques
    total = len(results)
    valid = total - invalid_count

    print(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
    print(f"   Total de phrases: {total}")
    print(f"   Phrases valides: {valid} ({100*valid/total:.1f}%)")
    print(f"   Phrases invalides: {invalid_count} ({100*invalid_count/total:.1f}%)")
    print(f"\n   Précision départ: {100*correct_departure/valid:.1f}%")
    print(f"   Précision arrivée: {100*correct_arrival/valid:.1f}%")
    print(f"   Précision complète (départ + arrivée): {100*correct_both/valid:.1f}%")

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid_count,
        "accuracy_departure": 100*correct_departure/valid,
        "accuracy_arrival": 100*correct_arrival/valid,
        "accuracy_both": 100*correct_both/valid
    }

if __name__ == "__main__":
    print("="*70)
    print("TEST COMPLET DU PIPELINE SUR 10K PHRASES")
    print("="*70)
    print(f"Heure de début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Chemins
    input_csv = "./data/annotated/train_sentences_10k.csv"
    input_txt = "./data/processed/input_10k.txt"
    output_txt = "./results/output_pipeline_10k.txt"

    # Préparer l'entrée
    prepare_input_file(input_csv, input_txt)

    # Initialiser le pipeline
    print("\n[2/4] INITIALISATION DU PIPELINE")
    print("="*70)
    pipeline = TravelOrderPipeline(use_rag=True)

    # Traiter le fichier
    print("\n[3/4] TRAITEMENT DES 10K PHRASES")
    print("="*70)
    print("⚠️  Cela peut prendre plusieurs heures...")

    pipeline.process_file(input_txt, output_txt, metric="time")

    # Évaluer
    metrics = evaluate_results(output_txt, input_csv)

    # Temps total
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "="*70)
    print("✅ TEST COMPLET TERMINÉ")
    print("="*70)
    print(f"Heure de fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temps total: {hours}h {minutes}m {seconds}s")
    print(f"Vitesse moyenne: {10000/elapsed:.1f} phrases/seconde")
    print(f"\n📄 Fichier de sortie: {output_txt}")
    print(f"📊 Précision complète: {metrics['accuracy_both']:.1f}%")
    print("="*70)
