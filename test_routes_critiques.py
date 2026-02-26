#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test automatique pour vérifier les itinéraires SNCF
Tests spécifiques pour les cas limites et ambiguïtés
"""

import sys
import io

# Fix encoding
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (ValueError, AttributeError):
        pass

sys.path.insert(0, './src/3_pathfinder')
from pathfinder import TrainPathfinder

# Phrases de test critiques avec critères de validation
TEST_CASES = [
    # (from_city, to_city, max_stops, description, expected_stations)

    # === Tests de base (gares terminus correctes) ===
    ("Paris", "Lyon", 2, "Paris-Lyon doit utiliser Lyon Part Dieu (TGV)", ["Lyon Part Dieu"]),
    ("Paris", "Marseille", 3, "Paris-Marseille doit utiliser Marseille Saint-Charles", ["Marseille Saint-Charles"]),
    ("Lyon", "Marseille", 2, "Lyon-Marseille direct via LGV", ["Lyon Part Dieu", "Marseille Saint-Charles"]),
    ("Paris", "Bordeaux", 2, "Paris-Bordeaux LGV Atlantique", ["Bordeaux Saint-Jean"]),
    ("Paris", "Lille", 2, "Paris-Lille LGV Nord", []),
    ("Paris", "Strasbourg", 2, "Paris-Strasbourg LGV Est", []),

    # === Tests d'ambiguïté géographique ===
    ("Marseille", "Paris", 3, "Marseille-Paris (ordre inversé)", ["Marseille Saint-Charles"]),
    ("Lyon", "Paris", 2, "Lyon-Paris (ordre inversé)", ["Lyon Part Dieu"]),

    # === Tests de trajets complexes (plusieurs correspondances) ===
    ("Lille", "Nice", 6, "Trajet long Lille-Nice via Paris-Lyon-Marseille", []),
    ("Strasbourg", "Toulouse", 5, "Strasbourg-Toulouse via Paris-Bordeaux", []),
    ("Bordeaux", "Marseille", 6, "Bordeaux-Marseille (trajet indirect)", []),
    ("Nantes", "Lyon", 5, "Nantes-Lyon via Paris", []),

    # === Tests de villes avec plusieurs gares ===
    ("Paris", "Tours", 2, "Paris-Tours LGV", []),
    ("Paris", "Nantes", 2, "Paris-Nantes LGV", []),
    ("Lyon", "Grenoble", 2, "Lyon-Grenoble TER", []),
    ("Lyon", "Chambery", 2, "Lyon-Chambéry TER", []),

    # === Tests de correspondances logiques ===
    ("Toulouse", "Marseille", 4, "Toulouse-Marseille via Montpellier", []),
    ("Nice", "Marseille", 2, "Nice-Marseille (Côte d'Azur)", []),
    ("Bordeaux", "Toulouse", 2, "Bordeaux-Toulouse Intercités", []),
]

def run_tests():
    print("="*70)
    print("TESTS AUTOMATIQUES DES ITINERAIRES SNCF")
    print("="*70)

    # Initialiser le pathfinder
    print("\n[INIT] Initialisation du pathfinder...")
    pathfinder = TrainPathfinder()
    pathfinder.initialize()

    results = []
    passed = 0
    failed = 0
    warnings = 0

    print("\n" + "="*70)
    print("EXECUTION DES TESTS")
    print("="*70)

    for from_city, to_city, max_stops, description, expected_stations in TEST_CASES:
        print(f"\n[TEST] {description}")
        print(f"  Trajet: {from_city} -> {to_city}")

        route = pathfinder.find_route(from_city, to_city, metric="time")

        if "error" in route:
            print(f"  [FAIL] Erreur: {route['error']}")
            failed += 1
            results.append((description, False, route.get('error'), []))
            continue

        num_stops = route['num_stops']
        stations = route['stations']

        # Vérifier le nombre de gares
        if num_stops <= max_stops:
            # Vérifier que les gares attendues sont présentes
            missing_expected = [s for s in expected_stations if not any(s.lower() in st.lower() for st in stations)]

            if missing_expected:
                print(f"  [WARN] {num_stops} gares (OK) mais gare(s) attendue(s) manquante(s): {', '.join(missing_expected)}")
                print(f"      Itineraire: {' -> '.join(stations)}")
                warnings += 1
                results.append((description, "warning", f"Gares manquantes: {missing_expected}", stations))
            else:
                print(f"  [PASS] {num_stops} gares (<= {max_stops})")
                print(f"      Itineraire: {' -> '.join(stations)}")
                passed += 1
                results.append((description, True, f"{num_stops} gares", stations))
        else:
            print(f"  [FAIL] {num_stops} gares (attendu <= {max_stops})")
            print(f"      Itineraire: {' -> '.join(stations[:3])} ... {stations[-1]}")
            failed += 1
            results.append((description, False, f"{num_stops} gares au lieu de <={max_stops}", stations))

    # Résumé
    print("\n" + "="*70)
    print("RESUME DES TESTS")
    print("="*70)
    total = len(TEST_CASES)
    print(f"Total: {total} tests")
    print(f"[PASS] Reussis: {passed}/{total} ({passed*100//total}%)")
    print(f"[WARN] Avertissements: {warnings}/{total}")
    print(f"[FAIL] Echoues: {failed}/{total}")

    if failed > 0:
        print("\n[FAIL] Tests echoues:")
        for desc, success, details, stations in results:
            if success == False:
                print(f"  - {desc}: {details}")

    if warnings > 0:
        print("\n[WARN] Tests avec avertissement:")
        for desc, success, details, stations in results:
            if success == "warning":
                print(f"  - {desc}: {details}")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
