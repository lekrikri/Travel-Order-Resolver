#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple Qwen2.5 sur GPU
"""

import sys
import io

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ajouter src au path
sys.path.insert(0, './src/2_qwen_chroma')

import torch
from qwen_extractor import Qwen25Extractor

print("="*70)
print("TEST SIMPLE QWEN2.5 SUR GPU")
print("="*70)

# Vérifier GPU
print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n[1/2] Initialisation de Qwen2.5...")
extractor = Qwen25Extractor(model_name="Qwen/Qwen2.5-3B-Instruct")

if not extractor.load_model():
    print("\n❌ ERREUR: Impossible de charger Qwen2.5")
    sys.exit(1)

print("\n[2/2] Test d'extraction sur 1 phrase...")
test_sentence = "Je veux aller de Paris à Lyon demain matin"

result = extractor.extract_cities(test_sentence)

print("\n" + "="*70)
print("RÉSULTAT")
print("="*70)
print(f"\n📝 Phrase: {test_sentence}")
print(f"🏙️  Départ prédit: {result.get('departure')}")
print(f"🏙️  Arrivée prédite: {result.get('arrival')}")
print(f"📊 Confiance: {result.get('confidence', 0.0):.2f}")

if result.get('departure') == 'Paris' and result.get('arrival') == 'Lyon':
    print("\n✅ TEST RÉUSSI!")
else:
    print("\n⚠️  Résultat inattendu")
    if 'raw_response' in result:
        print(f"Réponse brute: {result['raw_response']}")

print("="*70)
