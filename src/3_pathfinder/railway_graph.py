#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphe du réseau ferroviaire
Construit un graphe orienté pour le calcul d'itinéraires
"""

import networkx as nx
import pandas as pd
import math
from typing import List, Dict, Optional, Tuple
from sncf_data_loader import SNCFDataLoader


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance à vol d'oiseau entre deux points GPS (en km).
    Utilisé comme heuristique pour A*.
    """
    if None in (lat1, lon1, lat2, lon2):
        return 0  # Pas de coordonnées = heuristique nulle

    R = 6371  # Rayon de la Terre en km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

class RailwayGraph:
    """Graphe du réseau ferroviaire français"""

    def __init__(self):
        """Initialise le graphe vide"""
        self.graph = nx.Graph()  # Graphe non-orienté (les trains vont dans les 2 sens)
        self.stations = {}  # Dictionnaire code_uic -> infos gare
        self.loader = None

    def build_from_sncf_data(self, loader: SNCFDataLoader):
        """
        Construit le graphe à partir des données SNCF

        Args:
            loader: Instance de SNCFDataLoader avec les données chargées
        """
        self.loader = loader

        # Charger les gares
        stations_df = loader.get_all_stations()
        connections_df = loader.get_all_connections()

        print("[BUILD] Construction du graphe ferroviaire...")

        # Ajouter les nœuds (gares)
        for _, station in stations_df.iterrows():
            # Normaliser le code UIC (gérer les codes composites)
            code_uic = str(station["code_uic"]).split(";")[0]
            self.stations[code_uic] = {
                "nom": station["nom_gare"],
                "commune": station["commune"],
                "latitude": station.get("latitude"),
                "longitude": station.get("longitude")
            }
            self.graph.add_node(code_uic, **self.stations[code_uic])

        print(f"   [OK] {len(self.graph.nodes)} gares ajoutees au graphe")

        # Ajouter les arêtes (connexions)
        for _, conn in connections_df.iterrows():
            # Normaliser les codes UIC (gérer les codes composites)
            from_uic = str(conn["from_uic"]).split(";")[0]
            to_uic = str(conn["to_uic"]).split(";")[0]

            # Poids = temps de trajet en minutes
            weight = conn["time_minutes"]

            # Ajouter l'arête (bidirectionnelle)
            self.graph.add_edge(
                from_uic,
                to_uic,
                weight=weight,
                distance_km=conn["distance_km"]
            )

        print(f"   [OK] {len(self.graph.edges)} connexions ajoutees au graphe")

        # Vérifier la connectivité
        if nx.is_connected(self.graph):
            print("   [OK] Le graphe est entierement connecte")
        else:
            n_components = nx.number_connected_components(self.graph)
            print(f"   [WARNING] Le graphe a {n_components} composantes connexes")

    def find_shortest_path(
        self,
        from_uic: str,
        to_uic: str,
        metric: str = "time"
    ) -> Optional[Tuple[List[str], float]]:
        """
        Trouve le chemin le plus court entre deux gares (Dijkstra)

        Args:
            from_uic: Code UIC de la gare de départ
            to_uic: Code UIC de la gare d'arrivée
            metric: Métrique à minimiser ("time" ou "distance")

        Returns:
            (chemin, coût) ou None si aucun chemin
        """
        if from_uic not in self.graph or to_uic not in self.graph:
            return None

        try:
            # Algorithme de Dijkstra
            if metric == "time":
                weight = "weight"  # Temps en minutes
            else:
                weight = "distance_km"  # Distance en km

            path = nx.shortest_path(
                self.graph,
                source=from_uic,
                target=to_uic,
                weight=weight
            )

            # Calculer le coût total
            cost = nx.shortest_path_length(
                self.graph,
                source=from_uic,
                target=to_uic,
                weight=weight
            )

            return path, cost

        except nx.NetworkXNoPath:
            return None

    def find_shortest_path_astar(
        self,
        from_uic: str,
        to_uic: str,
        metric: str = "time"
    ) -> Optional[Tuple[List[str], float]]:
        """
        Trouve le chemin le plus court avec A* (plus rapide que Dijkstra).

        Utilise la distance géographique comme heuristique.

        Args:
            from_uic: Code UIC de la gare de départ
            to_uic: Code UIC de la gare d'arrivée
            metric: Métrique à minimiser ("time" ou "distance")

        Returns:
            (chemin, coût) ou None si aucun chemin
        """
        if from_uic not in self.graph or to_uic not in self.graph:
            return None

        # Récupérer les coordonnées de la destination pour l'heuristique
        dest_info = self.stations.get(to_uic, {})
        dest_lat = dest_info.get("latitude")
        dest_lon = dest_info.get("longitude")

        def heuristic(node, target):
            """Heuristique A* : distance à vol d'oiseau vers la destination."""
            node_info = self.stations.get(node, {})
            node_lat = node_info.get("latitude")
            node_lon = node_info.get("longitude")

            dist_km = haversine_distance(node_lat, node_lon, dest_lat, dest_lon)

            if metric == "time":
                # Convertir en minutes (vitesse moyenne TGV ~250 km/h)
                return dist_km / 250 * 60
            else:
                return dist_km

        try:
            if metric == "time":
                weight = "weight"
            else:
                weight = "distance_km"

            # Algorithme A*
            path = nx.astar_path(
                self.graph,
                source=from_uic,
                target=to_uic,
                heuristic=heuristic,
                weight=weight
            )

            # Calculer le coût total
            cost = 0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                cost += edge_data.get(weight, 0)

            return path, cost

        except nx.NetworkXNoPath:
            return None

    def get_station_info(self, code_uic: str) -> Optional[Dict]:
        """Retourne les informations d'une gare"""
        return self.stations.get(code_uic)

    def find_station_by_name(self, name: str) -> Optional[str]:
        """
        Recherche le code UIC d'une gare par son nom

        Args:
            name: Nom de la gare ou de la ville

        Returns:
            Code UIC ou None
        """
        if self.loader is None:
            return None

        station = self.loader.get_station_by_name(name)
        if station:
            return str(station["code_uic"])
        return None

    def get_path_details(self, path: List[str]) -> List[Dict]:
        """
        Récupère les détails d'un itinéraire

        Args:
            path: Liste de codes UIC

        Returns:
            Liste de dictionnaires avec les infos de chaque étape
        """
        details = []

        for i, uic in enumerate(path):
            station = self.get_station_info(uic)
            step = {
                "step_number": i + 1,
                "code_uic": uic,
                "nom_gare": station["nom"] if station else "Inconnue",
                "commune": station["commune"] if station else None,
                "latitude": station.get("latitude") if station else None,
                "longitude": station.get("longitude") if station else None
            }

            # Ajouter infos sur le trajet vers la prochaine gare
            if i < len(path) - 1:
                next_uic = path[i + 1]
                edge_data = self.graph.get_edge_data(uic, next_uic)
                if edge_data:
                    step["next_time_minutes"] = edge_data.get("weight")
                    step["next_distance_km"] = edge_data.get("distance_km")

            details.append(step)

        return details

    def stats(self):
        """Affiche des statistiques sur le graphe"""
        print("\n[STATS] STATISTIQUES DU GRAPHE FERROVIAIRE")
        print("="*70)
        print(f"Nombre de gares (noeuds):      {len(self.graph.nodes)}")
        print(f"Nombre de connexions (aretes): {len(self.graph.edges)}")

        if len(self.graph.nodes) > 0:
            avg_degree = sum(dict(self.graph.degree()).values()) / len(self.graph.nodes)
            print(f"Degre moyen par gare:          {avg_degree:.1f}")

        if nx.is_connected(self.graph):
            diameter = nx.diameter(self.graph)
            print(f"Diametre du graphe:            {diameter} etapes")
        else:
            print(f"Composantes connexes:          {nx.number_connected_components(self.graph)}")

        print("="*70)

# Test
if __name__ == "__main__":
    print("="*70)
    print("TEST DU RAILWAY GRAPH")
    print("="*70)

    # Charger les données SNCF
    loader = SNCFDataLoader()
    loader.download_stations()
    loader.generate_connections_from_stations()

    # Construire le graphe
    railway = RailwayGraph()
    railway.build_from_sncf_data(loader)

    # Afficher statistiques
    railway.stats()

    # Test de pathfinding
    print("\n🔍 TEST DE PATHFINDING")
    print("="*70)

    test_routes = [
        ("Paris", "Lyon"),
        ("Marseille", "Bordeaux"),
        ("Lille", "Nice")
    ]

    for city_from, city_to in test_routes:
        print(f"\n{city_from} → {city_to}:")

        # Trouver les codes UIC
        from_uic = railway.find_station_by_name(city_from)
        to_uic = railway.find_station_by_name(city_to)

        if not from_uic or not to_uic:
            print("   [ERROR] Gare(s) non trouvee(s)")
            continue

        # Calculer l'itinéraire
        result = railway.find_shortest_path(from_uic, to_uic, metric="time")

        if result:
            path, total_time = result
            print(f"   [OK] Itineraire trouve: {len(path)} etapes, {total_time:.0f} min")

            # Afficher le détail
            details = railway.get_path_details(path)
            for step in details:
                step_info = f"      {step['step_number']}. {step['nom_gare']}"
                if "next_time_minutes" in step:
                    step_info += f" -> {step['next_time_minutes']:.0f} min"
                print(step_info)
        else:
            print("   [ERROR] Aucun itineraire trouve")
