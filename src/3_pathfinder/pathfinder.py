#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pathfinder - Calcul d'itinéraires ferroviaires
Module principal pour trouver des itinéraires entre villes
"""

import sys
import io
from typing import List, Dict, Optional, Tuple
from sncf_data_loader import SNCFDataLoader
from railway_graph import RailwayGraph

# Fix encoding (only if not already fixed)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (ValueError, AttributeError):
        pass  # stdout already configured

class TrainPathfinder:
    """Calcule les itinéraires ferroviaires entre villes"""

    def __init__(self, cache_dir: str = "./data/sncf"):
        """
        Initialise le pathfinder

        Args:
            cache_dir: Répertoire de cache pour les données SNCF
        """
        self.loader = SNCFDataLoader(cache_dir=cache_dir)
        self.graph = RailwayGraph()
        self.initialized = False

    @staticmethod
    def normalize_station_name(station_name: str) -> str:
        """
        Normalise un nom de gare SNCF en nom de ville simple

        Exemples:
            "La Pauline Hyères" -> "Hyères"
            "Belfort - Montbéliard TGV" -> "Belfort"
            "Bordeaux Saint-Jean" -> "Bordeaux"
            "Paris Est" -> "Paris"

        Args:
            station_name: Nom complet de la gare SNCF

        Returns:
            Nom de ville normalisé
        """
        import re

        name = station_name.strip()

        # Supprimer les suffixes courants
        suffixes_to_remove = [
            r'\s*-\s*TGV$',
            r'\s*-\s*Montbéliard\s*TGV$',
            r'\s*TGV\s*Rhône-Alpes\s*Sud$',
            r'\s*Bretagne\s*Sud$',
            r'\s*Saint-Jean$',
            r'\s*Rive\s*Droite$',
            r'\s*Rive\s*Gauche$',
            r'\s*Est$',
            r'\s*Ouest$',
            r'\s*Nord$',
            r'\s*Sud$',
            r'\s*Centre$',
            r'\s*Gare$',
            r'\s*Station$',
            r'\s*Hôpital-Université$',
            r'\s*Universités?\s*-\s*\w+$',
            r'\s*Universités?$',
            r'\s*Val\s+de\s+Seine$',
            r'\s*Stade\s+de\s+France.*$',
            r'\s*La\s+Bocca$',
            r'\s*Montjovis$',
            r'\s*Bénédictins$',
            r'\s*Sainte-Musse$',
            r'\s*Pompadour$',
            r'\s*Musées$',
            r'\s*Mésanges$',
            r'\s*Saint-Joseph$',
            r'\s*Médoquine$',
            r'\s*-\s*Challes-les-Eaux$',
        ]

        for suffix in suffixes_to_remove:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)

        # Supprimer les préfixes type "La Pauline", "Le Bosquet", etc.
        # Mais garder les "La/Le/Les" qui font partie du nom de ville (La Rochelle, Le Mans, etc.)
        prefixes_to_remove = [
            r'^La\s+Pauline\s+',
            r'^Le\s+Bosquet\s+',
            r'^Les\s+Fontinettes\s+',
            r'^La\s+Garenne-',
            r'^La\s+Plaine\s+',
            r'^Chemin\s+d\'',
            r'^Saint-Hilaire\s+-\s+',
            r'^Morières-lès-',
            r'^Joué-lès-',
        ]

        for prefix in prefixes_to_remove:
            name = re.sub(prefix, '', name, flags=re.IGNORECASE)

        # Gérer les cas spéciaux de noms composés
        # "Strasbourg Roethig" -> "Strasbourg"
        # "Saint-Quentin-Fallavier" -> "Saint-Quentin" (MAIS "Saint-Étienne" reste "Saint-Étienne")
        # Important: Les patterns sont évalués dans l'ordre, donc mettre les plus spécifiques en premier
        special_cases = {
            # Cas spéciaux avec tirets à ne PAS couper
            r'^Saint-Étienne\s+.*$': 'Saint-Étienne',
            r'^Saint-Quentin-Fallavier$': 'Saint-Quentin',  # Cas spécial
            r'^Clermont-Ferrand\s+.*$': 'Clermont-Ferrand',
            r'^Aix-en-Provence\s+.*$': 'Aix-en-Provence',

            # Grandes villes avec variantes
            r'^Strasbourg\s+.*$': 'Strasbourg',
            r'^Tours\s+.*$': 'Tours',
            r'^Limoges\s+.*$': 'Limoges',
            r'^Toulouse\s+.*$': 'Toulouse',
            r'^Lyon\s+.*$': 'Lyon',
            r'^Marseille\s+.*$': 'Marseille',
            r'^Paris\s+.*$': 'Paris',
            r'^Bordeaux\s+.*$': 'Bordeaux',
            r'^Lille\s+.*$': 'Lille',
            r'^Nantes\s+.*$': 'Nantes',
            r'^Nice\s+.*$': 'Nice',
            r'^Montpellier\s+.*$': 'Montpellier',
            r'^Rennes\s+.*$': 'Rennes',
            r'^Reims\s+.*$': 'Reims',
            r'^Le\s+Havre\s+.*$': 'Le Havre',
            r'^Le\s+Mans\s+.*$': 'Le Mans',
            r'^Grenoble\s+.*$': 'Grenoble',
            r'^Dijon\s+.*$': 'Dijon',
            r'^Angers\s+.*$': 'Angers',
            r'^Nîmes\s+.*$': 'Nîmes',
            r'^Toulon\s+.*$': 'Toulon',
            r'^Clermont\s+.*$': 'Clermont',
            r'^Saint-Denis\s+.*$': 'Saint-Denis',
            r'^Saint-Quentin\s+.*$': 'Saint-Quentin',
            r'^Argenteuil\s+.*$': 'Argenteuil',
            r'^Rouen\s+.*$': 'Rouen',
            r'^Montreuil\s+.*$': 'Montreuil',
            r'^Caen\s+.*$': 'Caen',
            r'^Nancy\s+.*$': 'Nancy',
            r'^Amiens\s+.*$': 'Amiens',
            r'^Brest\s+.*$': 'Brest',
            r'^Metz\s+.*$': 'Metz',
            r'^Besançon\s+.*$': 'Besançon',
            r'^Orléans\s+.*$': 'Orléans',
            r'^Mulhouse\s+.*$': 'Mulhouse',
            r'^Perpignan\s+.*$': 'Perpignan',
            r'^Boulogne\s+.*$': 'Boulogne',
            r'^Antony\s+.*$': 'Antony',
            r'^Valence\s+.*$': 'Valence',
            r'^Troyes\s+.*$': 'Troyes',
            r'^Chambéry\s+.*$': 'Chambéry',
            r'^Niort\s+.*$': 'Niort',
            r'^Lorient\s+.*$': 'Lorient',
            r'^Mérignac\s+.*$': 'Mérignac',
            r'^Belfort\s+.*$': 'Belfort',
            r'^Montauban\s+.*$': 'Montauban',
        }

        for pattern, replacement in special_cases.items():
            if re.match(pattern, name, flags=re.IGNORECASE):
                return replacement

        # Nettoyer les espaces multiples
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def initialize(self, force_refresh: bool = False):
        """
        Initialise le pathfinder (télécharge données SNCF et construit le graphe)

        Args:
            force_refresh: Forcer le téléchargement des données
        """
        if self.initialized and not force_refresh:
            print("[OK] Pathfinder deja initialise")
            return

        print("="*70)
        print("INITIALISATION DU PATHFINDER")
        print("="*70)

        # Télécharger les données SNCF
        print("\n[1/3] Telechargement des gares SNCF...")
        self.loader.download_stations(force_refresh=force_refresh)

        print("\n[2/3] Generation des connexions ferroviaires...")
        self.loader.generate_connections_from_stations()

        # Construire le graphe
        print("\n[3/3] Construction du graphe ferroviaire...")
        self.graph.build_from_sncf_data(self.loader)

        self.initialized = True

        print("\n" + "="*70)
        print("[OK] PATHFINDER PRET")
        print("="*70)

    def _find_station_with_fallback(self, city: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Cherche une gare par nom. Si introuvable, essaie le fuzzy matching.

        Args:
            city: Nom de la ville ou gare

        Returns:
            (code_uic, warning_message) — warning est None si trouvé exactement
        """
        # Tentative 1 : recherche exacte via alias + contain
        uic = self.graph.find_station_by_name(city)
        if uic:
            return uic, None

        # Tentative 2 : fallback fuzzy matching (gare la plus proche)
        fallback = self.loader.get_nearest_station_fuzzy(city, threshold=45)
        if fallback:
            fallback_name = fallback["nom_gare"]
            # Récupérer le code UIC de la gare trouvée
            uic = self.graph.find_station_by_name(fallback_name)
            if uic:
                warning = fallback.get("fuzzy_warning", f"Ville '{city}' → gare la plus proche: '{fallback_name}'")
                return uic, warning

        return None, None

    def find_route(
        self,
        from_city: str,
        to_city: str,
        metric: str = "time",
        algorithm: str = "astar"
    ) -> Optional[Dict]:
        """
        Trouve un itinéraire entre deux villes.

        Args:
            from_city: Ville de départ
            to_city: Ville d'arrivée
            metric: Métrique à minimiser — "time", "distance" ou "changes"
            algorithm: Algorithme — "dijkstra" ou "astar" (ignoré si metric="changes")

        Returns:
            Dictionnaire avec l'itinéraire ou erreur
        """
        if not self.initialized:
            raise RuntimeError("Le pathfinder n'est pas initialisé. Appelez initialize() d'abord.")

        warnings = []

        # Trouver les gares correspondantes (avec fallback fuzzy si nécessaire)
        from_uic, from_warning = self._find_station_with_fallback(from_city)
        to_uic, to_warning = self._find_station_with_fallback(to_city)

        if from_warning:
            warnings.append(from_warning)
        if to_warning:
            warnings.append(to_warning)

        if not from_uic:
            return {
                "error": f"Gare de départ introuvable pour: '{from_city}' (aucune gare similaire dans le réseau SNCF)",
                "from_city": from_city,
                "to_city": to_city
            }

        if not to_uic:
            return {
                "error": f"Gare d'arrivée introuvable pour: '{to_city}' (aucune gare similaire dans le réseau SNCF)",
                "from_city": from_city,
                "to_city": to_city
            }

        # Calculer l'itinéraire selon la métrique
        if metric == "changes":
            # Stratégie : minimum de correspondances (BFS)
            result = self.graph.find_path_min_changes(from_uic, to_uic)
            if result:
                path_uic, nb_changes = result
                total_cost = nb_changes
            else:
                result = None
        elif algorithm == "astar":
            result = self.graph.find_shortest_path_astar(from_uic, to_uic, metric=metric)
            if result:
                path_uic, total_cost = result
        else:
            result = self.graph.find_shortest_path(from_uic, to_uic, metric=metric)
            if result:
                path_uic, total_cost = result

        if not result:
            return {
                "error": "Aucun itinéraire trouvé entre ces deux gares",
                "from_city": from_city,
                "to_city": to_city
            }

        # Récupérer les détails de l'itinéraire
        path_details = self.graph.get_path_details(path_uic)
        stations = [step["nom_gare"] for step in path_details]

        # Calculer les métriques complémentaires
        total_time = sum(
            self.graph.graph.get_edge_data(path_uic[i], path_uic[i+1], {}).get("weight", 0)
            for i in range(len(path_uic) - 1)
        )
        total_dist = sum(
            self.graph.graph.get_edge_data(path_uic[i], path_uic[i+1], {}).get("distance_km", 0)
            for i in range(len(path_uic) - 1)
        )

        return {
            "from_city": from_city,
            "to_city": to_city,
            "from_station": stations[0],
            "to_station": stations[-1],
            "stations": stations,
            "num_stops": len(stations),
            "num_changes": max(0, len(stations) - 2),
            "total_time_minutes": total_time,
            "total_distance_km": total_dist,
            "metric_used": metric,
            "algorithm_used": "bfs" if metric == "changes" else algorithm,
            "details": path_details,
            "warnings": warnings if warnings else None,
            "original_from": from_city,
            "original_to": to_city
        }

    def format_route_for_output(
        self,
        sentence_id: int,
        route: Dict
    ) -> str:
        """
        Formate l'itinéraire au format demandé: sentenceID,Departure,Step1,Step2,...,Destination

        Args:
            sentence_id: ID de la phrase
            route: Résultat de find_route()

        Returns:
            Ligne CSV formatée
        """
        if "error" in route:
            # En cas d'erreur, retourner INVALID
            return f"{sentence_id},INVALID"

        # Extraire les gares et les normaliser
        stations_raw = route["stations"]
        stations_normalized = [self.normalize_station_name(s).lower() for s in stations_raw]

        # Utiliser les noms originaux pour le départ et l'arrivée si disponibles
        if "original_from" in route and "original_to" in route:
            # CAS SPÉCIAL: Si une seule station (départ = arrivée physiquement)
            # Mais logiquement ce sont deux villes différentes
            if len(stations_normalized) == 1:
                # Créer une route avec départ et arrivée distincts
                stations_normalized = [
                    route["original_from"].lower(),
                    route["original_to"].lower()
                ]
            else:
                # Remplacer le premier et le dernier par les noms originaux (en lowercase)
                stations_normalized[0] = route["original_from"].lower()
                stations_normalized[-1] = route["original_to"].lower()

        # Formater: ID,Départ,Étape1,Étape2,...,Arrivée
        result = [str(sentence_id)] + stations_normalized

        return ",".join(result)

    def process_batch(
        self,
        input_file: str,
        output_file: str,
        metric: str = "time"
    ):
        """
        Traite un fichier CSV avec plusieurs itinéraires

        Format d'entrée: sentenceID,Departure,Destination
        Format de sortie: sentenceID,Departure,Step1,Step2,...,Destination

        Args:
            input_file: Fichier CSV d'entrée
            output_file: Fichier CSV de sortie
            metric: Métrique à minimiser
        """
        import pandas as pd

        if not self.initialized:
            self.initialize()

        print(f"\n📥 Traitement du fichier: {input_file}")

        # Charger le fichier d'entrée
        df = pd.read_csv(input_file, header=None, names=["sentence_id", "departure", "destination"])

        print(f"   {len(df)} itinéraires à calculer")

        results = []

        for idx, row in df.iterrows():
            sentence_id = row["sentence_id"]
            departure = row["departure"]
            destination = row["destination"]

            print(f"\r   Traitement {idx+1}/{len(df)}: {departure} → {destination}", end="")

            # Calculer l'itinéraire
            route = self.find_route(departure, destination, metric=metric)

            # Formater pour la sortie
            output_line = self.format_route_for_output(sentence_id, route)
            results.append(output_line)

        print()  # Nouvelle ligne

        # Sauvegarder les résultats
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))

        print(f"[OK] Resultats sauvegardes: {output_file}")

        return results

# Test
if __name__ == "__main__":
    print("="*70)
    print("TEST DU PATHFINDER")
    print("="*70)

    # Initialiser le pathfinder
    pathfinder = TrainPathfinder()
    pathfinder.initialize()

    # Afficher stats du graphe
    pathfinder.graph.stats()

    # Test sur quelques routes
    print("\n[TEST] RECHERCHE D'ITINERAIRES")
    print("="*70)

    test_routes = [
        ("Paris", "Lyon"),
        ("Marseille", "Bordeaux"),
        ("Lille", "Nice"),
        ("Strasbourg", "Toulouse")
    ]

    for from_city, to_city in test_routes:
        print(f"\n{from_city} -> {to_city}")

        route = pathfinder.find_route(from_city, to_city, metric="time")

        if "error" in route:
            print(f"   [ERROR] Erreur: {route['error']}")
        else:
            print(f"   [OK] {route['num_stops']} gares, {route['num_changes']} changement(s)")
            print(f"   [TIME] Temps estime: {route['total_time_minutes']:.0f} minutes")
            print(f"   [ROUTE] Itineraire:")

            for i, station in enumerate(route['stations']):
                prefix = "      " + ("[START]" if i == 0 else "[END]" if i == len(route['stations']) - 1 else "[STOP]")
                print(f"{prefix} {station}")

            # Format de sortie
            output = pathfinder.format_route_for_output(1, route)
            print(f"   [OUTPUT] Format sortie: {output}")

    print("\n" + "="*70)
    print("[OK] TESTS TERMINES")
    print("="*70)
