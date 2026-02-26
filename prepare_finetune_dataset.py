#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Préparation du dataset 10k pour fine-tuning de Qwen2.5
Convertit le CSV en format instruction-following
"""

import sys
import io
import csv
import json
import re
from datasets import Dataset

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("PRÉPARATION DU DATASET POUR FINE-TUNING")
print("="*70)

print("\n[1/4] Chargement du dataset 10k...")

# Charger le dataset
data = []
with open('./data/annotated/train_sentences_10k.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Nettoyer la phrase (enlever les tags)
        clean_sentence = re.sub(r'<[^>]+>', '', row['sentence'])
        data.append({
            'sentence': clean_sentence,
            'departure': row['departure'],
            'arrival': row['arrival']
        })

print(f"✅ {len(data)} phrases chargées")

print("\n[2/4] Conversion au format instruction-tuning...")

# Template d'instruction
INSTRUCTION = """Tu es un assistant spécialisé dans l'extraction d'informations de voyage en train.

Ta tâche : extraire la ville de DÉPART et la ville d'ARRIVÉE d'une phrase en français.

Format de sortie (JSON strict) :
{"departure": "ville_de_depart", "arrival": "ville_d_arrivee"}

Si impossible à extraire, retourne :
{"departure": null, "arrival": null}"""

# Convertir en format instruction
formatted_data = []
for item in data:
    formatted_item = {
        'instruction': INSTRUCTION,
        'input': item['sentence'],
        'output': json.dumps({
            'departure': item['departure'],
            'arrival': item['arrival']
        }, ensure_ascii=False)
    }
    formatted_data.append(formatted_item)

print(f"✅ {len(formatted_data)} exemples formatés")

print("\n[3/4] Séparation train/validation (90/10)...")

# Séparer en train (90%) et validation (10%)
split_idx = int(len(formatted_data) * 0.9)
train_data = formatted_data[:split_idx]
val_data = formatted_data[split_idx:]

print(f"   Train: {len(train_data)} exemples")
print(f"   Validation: {len(val_data)} exemples")

print("\n[4/4] Sauvegarde des datasets...")

# Sauvegarder en JSONL (format standard pour fine-tuning)
with open('./data/finetune/train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('./data/finetune/val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Créer aussi des datasets HuggingFace pour compatibilité
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

train_dataset.save_to_disk('./data/finetune/train_hf')
val_dataset.save_to_disk('./data/finetune/val_hf')

print(f"✅ Datasets sauvegardés:")
print(f"   ./data/finetune/train.jsonl ({len(train_data)} exemples)")
print(f"   ./data/finetune/val.jsonl ({len(val_data)} exemples)")
print(f"   ./data/finetune/train_hf/ (format HuggingFace)")
print(f"   ./data/finetune/val_hf/ (format HuggingFace)")

print("\n" + "="*70)
print("✅ DATASET PRÊT POUR LE FINE-TUNING")
print("="*70)

# Afficher quelques exemples
print("\n📝 Exemples formatés:")
for i in range(min(3, len(train_data))):
    example = train_data[i]
    print(f"\n--- Exemple {i+1} ---")
    print(f"Input: {example['input']}")
    print(f"Output: {example['output']}")
