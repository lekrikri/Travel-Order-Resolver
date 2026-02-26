#!/usr/bin/env python3
"""Test de debug pour identifier le probleme"""
import sys
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

print("="*70)
print("TEST DEBUG - IDENTIFICATION DU PROBLEME")
print("="*70)

# Test 1: Import
print("\n[1] Test des imports...")
try:
    from qwen_extractor import Qwen25Extractor
    print("   OK - qwen_extractor")
except Exception as e:
    print(f"   ERREUR qwen_extractor: {e}")
    sys.exit(1)

try:
    from chroma_rag import ChromaRAG
    print("   OK - chroma_rag")
except Exception as e:
    print(f"   ERREUR chroma_rag: {e}")
    sys.exit(1)

try:
    from pathfinder import TrainPathfinder
    print("   OK - pathfinder")
except Exception as e:
    print(f"   ERREUR pathfinder: {e}")
    sys.exit(1)

# Test 2: Chargement modele
print("\n[2] Test chargement Qwen2.5...")
try:
    extractor = Qwen25Extractor()
    if extractor.load_model():
        print("   OK - Qwen2.5 charge")
    else:
        print("   ERREUR - Impossible de charger Qwen2.5")
        sys.exit(1)
except Exception as e:
    print(f"   ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: ChromaDB
print("\n[3] Test ChromaDB...")
try:
    rag = ChromaRAG()
    if rag.initialize():
        print(f"   OK - ChromaDB initialise ({rag.collection.count()} exemples)")
    else:
        print("   ERREUR - ChromaDB non initialise")
except Exception as e:
    print(f"   ERREUR: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Pathfinder
print("\n[4] Test Pathfinder...")
try:
    pathfinder = TrainPathfinder()
    pathfinder.initialize()
    print("   OK - Pathfinder initialise")
except Exception as e:
    print(f"   ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Pipeline complet
print("\n[5] Test extraction simple...")
try:
    result = extractor.extract_cities("Je veux aller de Paris a Lyon")
    print(f"   Depart: {result.get('departure')}")
    print(f"   Arrivee: {result.get('arrival')}")
    print("   OK - Extraction fonctionne")
except Exception as e:
    print(f"   ERREUR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TESTS TERMINES")
print("="*70)
