#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Qwen2.5 seul (sans RAG) sur 100 phrases - GPU
"""

import sys
import io
import csv
import re
import time

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, './src/2_qwen_chroma')

import torch
from qwen_extractor import Qwen25Extractor

def flush_print(msg):
    """Print avec flush immédiat"""
    print(msg, flush=True)

flush_print("="*70)
flush_print("TEST QWEN2.5 (SANS RAG) SUR 100 PHRASES - GPU")
flush_print("="*70)

# Vérifier GPU
flush_print(f"\n✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ÉTAPE 1: Charger Qwen2.5
flush_print("\n[1/4] Chargement de Qwen2.5...")
extractor = Qwen25Extractor(model_name="Qwen/Qwen2.5-3B-Instruct")

if not extractor.load_model():
    flush_print("\n❌ ERREUR: Impossible de charger Qwen2.5")
    sys.exit(1)

flush_print("✅ Qwen2.5 chargé sur GPU")

# ÉTAPE 2: Charger dataset de test
flush_print("\n[2/4] Chargement du dataset de test...")
test_sentences = []
with open('./data/annotated/train_sentences_10k.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        clean_sentence = re.sub(r'<[^>]+>', '', row['sentence'])
        test_sentences.append({
            'sentence': clean_sentence,
            'gold_dep': row['departure'],
            'gold_arr': row['arrival']
        })

test_sentences = test_sentences[:100]
flush_print(f"✅ {len(test_sentences)} phrases chargées")

# ÉTAPE 3: Extraction avec Qwen2.5
flush_print("\n[3/4] Extraction avec Qwen2.5...")
flush_print("   (Progression affichée toutes les 10 phrases)")

results = []
start_time = time.time()

for i, test in enumerate(test_sentences):
    # Extraction
    result = extractor.extract_cities(test['sentence'])

    # Stocker résultat
    results.append({
        'sentence': test['sentence'],
        'gold_dep': test['gold_dep'],
        'gold_arr': test['gold_arr'],
        'pred_dep': result.get('departure'),
        'pred_arr': result.get('arrival'),
        'confidence': result.get('confidence', 0.0)
    })

    # Afficher progression
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (len(test_sentences) - i - 1) / rate
        flush_print(f"   {i+1}/{len(test_sentences)} - {elapsed:.1f}s écoulées, ~{remaining:.0f}s restantes")

total_time = time.time() - start_time
flush_print(f"\n✅ Extraction terminée en {total_time:.1f}s ({total_time/len(test_sentences):.2f}s/phrase)")

# ÉTAPE 4: Calculer métriques
flush_print("\n[4/4] Calcul des métriques...")

correct_both = 0
correct_dep = 0
correct_arr = 0

for r in results:
    if r['pred_dep'] == r['gold_dep']:
        correct_dep += 1
    if r['pred_arr'] == r['gold_arr']:
        correct_arr += 1
    if r['pred_dep'] == r['gold_dep'] and r['pred_arr'] == r['gold_arr']:
        correct_both += 1

accuracy = correct_both / len(results) * 100
precision_dep = correct_dep / len(results) * 100
precision_arr = correct_arr / len(results) * 100

flush_print("\n" + "="*70)
flush_print("RÉSULTATS QWEN2.5 (SANS RAG)")
flush_print("="*70)
flush_print(f"\n📊 Phrases testées: {len(results)}")
flush_print(f"⏱️  Temps total: {total_time:.1f}s")
flush_print(f"🚀 Vitesse: {total_time/len(results):.2f}s/phrase")
flush_print(f"\n✅ Accuracy (les 2 correctes): {accuracy:.1f}%")
flush_print(f"🏙️  Départ correct: {precision_dep:.1f}%")
flush_print(f"🏙️  Arrivée correcte: {precision_arr:.1f}%")

# Sauvegarder résultats
output_file = './results/qwen_only_100_results.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sentence', 'gold_dep', 'gold_arr', 'pred_dep', 'pred_arr', 'confidence'])
    writer.writeheader()
    writer.writerows(results)

flush_print(f"\n💾 Résultats sauvegardés: {output_file}")
flush_print("="*70)
