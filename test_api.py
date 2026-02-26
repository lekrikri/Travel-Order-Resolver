#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'API REST
"""

import requests
import json

# Configuration
API_URL = "http://localhost:8000"

print("="*70)
print("TEST DE L'API REST")
print("="*70)

# Test 1: Health check
print("\n[1/4] Health check...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"✅ Status: {response.status_code}")
    print(f"   {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except requests.exceptions.ConnectionError:
    print("❌ ERREUR: L'API n'est pas démarrée")
    print("   Lancez-la avec: python api.py")
    exit(1)

# Test 2: Liste des modèles
print("\n[2/4] Liste des modèles...")
response = requests.get(f"{API_URL}/models")
print(f"✅ Status: {response.status_code}")
print(f"   {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Test 3: Extraction avec RAG
print("\n[3/4] Extraction avec RAG...")
test_sentences = [
    "Je voudrais un billet de train de Paris à Lyon pour demain",
    "Besoin d'un trajet Marseille Bordeaux",
    "Aller simple Lille Nice s'il vous plaît"
]

for sentence in test_sentences:
    response = requests.post(
        f"{API_URL}/extract",
        json={"sentence": sentence, "use_rag": True}
    )
    result = response.json()
    print(f"\n   Phrase: {sentence}")
    print(f"   → Départ: {result['departure']}")
    print(f"   → Arrivée: {result['arrival']}")
    print(f"   → Confiance: {result['confidence']:.0%}")
    print(f"   → Méthode: {result['method']}")

# Test 4: Comparaison avec/sans RAG
print("\n[4/4] Comparaison avec/sans RAG...")
sentence = "Un billet Strasbourg Toulouse merci"

print(f"\n   Phrase: {sentence}")

# Sans RAG
response_no_rag = requests.post(
    f"{API_URL}/extract",
    json={"sentence": sentence, "use_rag": False}
)
result_no_rag = response_no_rag.json()
print(f"\n   Sans RAG:")
print(f"   → Départ: {result_no_rag['departure']}")
print(f"   → Arrivée: {result_no_rag['arrival']}")

# Avec RAG
response_rag = requests.post(
    f"{API_URL}/extract",
    json={"sentence": sentence, "use_rag": True}
)
result_rag = response_rag.json()
print(f"\n   Avec RAG:")
print(f"   → Départ: {result_rag['departure']}")
print(f"   → Arrivée: {result_rag['arrival']}")

print("\n" + "="*70)
print("✅ TESTS TERMINÉS")
print("="*70)
print(f"\n💡 Documentation interactive: {API_URL}/docs")
print(f"💡 Page d'accueil: {API_URL}/")
print("="*70)
