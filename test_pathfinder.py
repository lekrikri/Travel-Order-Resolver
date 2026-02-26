#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du module Pathfinder seul
"""

import sys
sys.path.insert(0, './src/3_pathfinder')

from pathfinder import TrainPathfinder

print("="*70)
print("TEST DU MODULE PATHFINDER")
print("="*70)

# Initialiser
pathfinder = TrainPathfinder()
pathfinder.initialize()

# Stats
pathfinder.graph.stats()

# Test de routes
print("\n🚂 TEST DE RECHERCHE D'ITINÉRAIRES")
print("="*70)

test_routes = [
    ("Paris", "Lyon"),
    ("Marseille", "Bordeaux"),
    ("Lille", "Nice"),
    ("Strasbourg", "Toulouse"),
    ("Nantes", "Montpellier")
]

for from_city, to_city in test_routes:
    print(f"\n{from_city} → {to_city}")

    route = pathfinder.find_route(from_city, to_city, metric="time")

    if "error" in route:
        print(f"   ❌ Erreur: {route['error']}")
    else:
        print(f"   ✅ {route['num_stops']} gares, {route['num_changes']} changement(s)")
        print(f"   ⏱️  Temps estimé: {route['total_time_minutes']:.0f} minutes")
        print(f"   🛤️  Itinéraire: {' → '.join(route['stations'])}")

        # Format de sortie
        output = pathfinder.format_route_for_output(1, route)
        print(f"   📄 Format: {output}")

print("\n" + "="*70)
print("✅ TESTS TERMINÉS")
print("="*70)
