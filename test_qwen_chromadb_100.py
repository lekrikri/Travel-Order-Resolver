#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Qwen2.5 + ChromaDB RAG sur 100 phrases - GPU
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
from chroma_rag import ChromaRAG

def flush_print(msg):
    """Print avec flush immédiat"""
    print(msg, flush=True)

flush_print("="*70)
flush_print("TEST QWEN2.5 + CHROMADB RAG SUR 100 PHRASES - GPU")
flush_print("="*70)

# Vérifier GPU
flush_print(f"\n✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ÉTAPE 1: Charger Qwen2.5
flush_print("\n[1/5] Chargement de Qwen2.5...")
extractor = Qwen25Extractor(model_name="Qwen/Qwen2.5-3B-Instruct")

if not extractor.load_model():
    flush_print("\n❌ ERREUR: Impossible de charger Qwen2.5")
    sys.exit(1)

flush_print("✅ Qwen2.5 chargé sur GPU")

# ÉTAPE 2: Initialiser ChromaDB
flush_print("\n[2/5] Initialisation de ChromaDB RAG...")
flush_print("   Création de l'instance ChromaRAG...")
rag = ChromaRAG()

flush_print("   Appel de rag.initialize()...")
if not rag.initialize():
    flush_print("\n❌ ERREUR: Impossible d'initialiser ChromaDB")
    sys.exit(1)

flush_print("✅ ChromaDB initialisé")

# ÉTAPE 3: Charger dataset d'entraînement dans ChromaDB
flush_print("\n[3/5] Chargement du dataset dans ChromaDB...")
if rag.collection.count() == 0:
    flush_print("   Chargement des 100 phrases annotées...")
    rag.load_dataset("./data/annotated/train_sentences_100.csv")
    flush_print(f"   ✅ {rag.collection.count()} documents chargés")
else:
    flush_print(f"   ✅ Collection déjà chargée: {rag.collection.count()} documents")

# ÉTAPE 4: Charger dataset de test
flush_print("\n[4/5] Chargement du dataset de test...")
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

# ÉTAPE 5: Extraction avec Qwen2.5 + RAG
flush_print("\n[5/5] Extraction avec Qwen2.5 + ChromaDB RAG...")
flush_print("   (Progression affichée toutes les 10 phrases)")

results = []
start_time = time.time()

for i, test in enumerate(test_sentences):
    # Récupérer exemples similaires avec ChromaDB
    similar_examples = rag.get_similar_examples(test['sentence'], n_results=3)

    # Extraction avec Qwen2.5 (few-shot avec exemples RAG)
    result = extractor.extract_cities(test['sentence'], examples=similar_examples)

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

# CALCUL DES MÉTRIQUES
flush_print("\nCalcul des métriques...")

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
flush_print("RÉSULTATS QWEN2.5 + CHROMADB RAG")
flush_print("="*70)
flush_print(f"\n📊 Phrases testées: {len(results)}")
flush_print(f"⏱️  Temps total: {total_time:.1f}s")
flush_print(f"🚀 Vitesse: {total_time/len(results):.2f}s/phrase")
flush_print(f"\n✅ Accuracy (les 2 correctes): {accuracy:.1f}%")
flush_print(f"🏙️  Départ correct: {precision_dep:.1f}%")
flush_print(f"🏙️  Arrivée correcte: {precision_arr:.1f}%")

# Sauvegarder résultats
output_file = './results/qwen_chromadb_100_results.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sentence', 'gold_dep', 'gold_arr', 'pred_dep', 'pred_arr', 'confidence'])
    writer.writeheader()
    writer.writerows(results)

flush_print(f"\n💾 Résultats sauvegardés: {output_file}")
flush_print("="*70)
