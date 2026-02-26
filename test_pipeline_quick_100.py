#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rapide du pipeline complet sur 100 phrases
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

def prepare_input_file(input_csv: str, output_txt: str, max_rows: int = 100):
    """
    Prépare le fichier d'entrée au format attendu par le pipeline
    Format de sortie: sentenceID,sentence (sans en-tête)
    """
    print("\n[1/4] PRÉPARATION DU FICHIER D'ENTRÉE")
    print("="*70)

    # Charger le dataset avec en-tête
    df = pd.read_csv(input_csv)

    # Limiter au nombre de lignes demandé
    df = df.head(max_rows)
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

def evaluate_results(output_file: str, ground_truth_csv: str, max_rows: int = 100):
    """
    Évalue la précision du pipeline en comparant avec les annotations
    """
    print("\n[4/4] ÉVALUATION DES RÉSULTATS")
    print("="*70)

    # Charger les ground truth
    gt_df = pd.read_csv(ground_truth_csv).head(max_rows)
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

    details = []

    for result_line in results:
        parts = result_line.split(',')
        sentence_id = int(parts[0])

        if len(parts) < 2 or parts[1] == 'INVALID':
            invalid_count += 1
            details.append(f"ID {sentence_id}: INVALID")
            continue

        # Extraire départ et arrivée prédits
        predicted_departure = parts[1].strip().lower()
        predicted_arrival = parts[-1].strip().lower()

        # Comparer avec ground truth
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

            # Détails
            status = "✅" if (dep_ok and arr_ok) else "❌"
            details.append(f"{status} ID {sentence_id}: {predicted_departure} → {predicted_arrival} (GT: {gt['departure']} → {gt['arrival']})")

    # Calculer les métriques
    total = len(results)
    valid = total - invalid_count

    print(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
    print(f"   Total de phrases: {total}")
    print(f"   Phrases valides: {valid} ({100*valid/total:.1f}%)")
    print(f"   Phrases invalides: {invalid_count} ({100*invalid_count/total:.1f}%)")

    if valid > 0:
        print(f"\n   Précision départ: {100*correct_departure/valid:.1f}% ({correct_departure}/{valid})")
        print(f"   Précision arrivée: {100*correct_arrival/valid:.1f}% ({correct_arrival}/{valid})")
        print(f"   Précision complète (départ + arrivée): {100*correct_both/valid:.1f}% ({correct_both}/{valid})")

    # Afficher quelques exemples
    print(f"\n📝 PREMIERS RÉSULTATS (10 premiers):")
    for detail in details[:10]:
        print(f"   {detail}")

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
    print(f"Heure de début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Chemins
    input_csv = "./data/annotated/train_sentences_10k.csv"
    input_txt = "./data/processed/input_100.txt"
    output_txt = "./results/output_pipeline_100.txt"

    # Préparer l'entrée
    prepare_input_file(input_csv, input_txt, max_rows=100)

    # Initialiser le pipeline
    print("\n[2/4] INITIALISATION DU PIPELINE")
    print("="*70)
    pipeline = TravelOrderPipeline(use_rag=True)

    # Traiter le fichier
    print("\n[3/4] TRAITEMENT DES 100 PHRASES")
    print("="*70)
    print("⏱️  Estimation: 3-5 minutes...")

    pipeline.process_file(input_txt, output_txt, metric="time")

    # Évaluer
    metrics = evaluate_results(output_txt, input_csv, max_rows=100)

    # Temps total
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "="*70)
    print("✅ TEST RAPIDE TERMINÉ")
    print("="*70)
    print(f"Heure de fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temps total: {minutes}m {seconds}s")
    print(f"Vitesse moyenne: {100/elapsed:.2f} phrases/seconde")
    print(f"\n📄 Fichier de sortie: {output_txt}")
    print(f"📊 Précision complète: {metrics['accuracy_both']:.1f}%")
    print("="*70)
