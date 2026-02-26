#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chargeur de données SNCF Open Data
Télécharge et traite les gares et liaisons ferroviaires
"""

import sys
import io

# Fix encoding (Windows compatibility)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (ValueError, AttributeError):
        pass  # stdout already configured

import requests
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class SNCFDataLoader:
    """Charge les données SNCF Open Data (gares et liaisons)"""

    # URLs des datasets SNCF Open Data (API v2.1)
    BASE_URL = "https://ressources.data.sncf.com/api/explore/v2.1"
    STATIONS_DATASET = "gares-de-voyageurs"

    # Clé API SNCF
    API_KEY = "125dad1b8c7a2ea1dd03af036d4e849927387a5a6465f5437068f7f1"

    def __init__(self, cache_dir: str = "./data/sncf", api_key: str = None):
        """
        Initialise le chargeur

        Args:
            cache_dir: Répertoire de cache pour les données téléchargées
            api_key: Clé API SNCF (optionnel, utilise la clé par défaut si non fournie)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stations_file = self.cache_dir / "stations.csv"
        self.connections_file = self.cache_dir / "connections.csv"

        self.stations_df = None
        self.connections_df = None

        # Utiliser la clé API fournie ou celle par défaut
        self.api_key = api_key if api_key else self.API_KEY

    def download_stations(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Télécharge le référentiel des gares SNCF

        Args:
            force_refresh: Forcer le téléchargement même si le cache existe

        Returns:
            DataFrame avec les gares
        """
        # Utiliser le cache si disponible
        if self.stations_file.exists() and not force_refresh:
            print(f"[OK] Chargement des gares depuis le cache: {self.stations_file}")
            self.stations_df = pd.read_csv(self.stations_file)
            return self.stations_df

        print("[DOWNLOAD] Telechargement du referentiel des gares SNCF...")

        # Télécharger toutes les gares (API paginée v2.1)
        all_stations = []
        limit = 100
        offset = 0

        # Construire l'URL de l'API v2.1
        api_url = f"{self.BASE_URL}/catalog/datasets/{self.STATIONS_DATASET}/records"

        while True:
            params = {
                "limit": limit,
                "offset": offset
            }

            # Headers avec clé API
            headers = {
                "Authorization": f"Apikey {self.api_key}"
            }

            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            # API v2.1 retourne les résultats directement
            results = data.get("results", [])
            if not results:
                break

            # Extraire les champs pertinents
            for record in results:
                # Dans API v2.1, les champs sont directement dans record
                pos_geo = record.get("position_geographique")

                # Extraire lat/lon si disponible
                latitude = pos_geo.get("lat") if pos_geo else None
                longitude = pos_geo.get("lon") if pos_geo else None

                all_stations.append({
                    "code_uic": record.get("codes_uic"),
                    "code_ci": record.get("libellecourt"),
                    "nom_gare": record.get("nom"),
                    "commune": record.get("nom"),  # Utiliser le nom de la gare comme commune
                    "departement": record.get("codeinsee", "")[:2] if record.get("codeinsee") else None,
                    "latitude": latitude,
                    "longitude": longitude,
                    "is_suggestable": record.get("segment_drg") in ["A", "B", "C"]  # Gares principales
                })

            offset += limit
            print(f"   Téléchargé {offset} gares...", end="\r")

            # Limite pour éviter trop de requêtes
            if offset >= 5000:  # Maximum ~3000 gares en réalité
                break

        print(f"\n[OK] {len(all_stations)} gares telechargees")

        # Créer DataFrame
        self.stations_df = pd.DataFrame(all_stations)

        # Nettoyer les données
        self.stations_df = self.stations_df.dropna(subset=["code_uic", "nom_gare"])
        self.stations_df["code_uic"] = self.stations_df["code_uic"].astype(str)

        # Gérer les codes UIC composites (ex: "87686030;87686006") en prenant le premier
        self.stations_df["code_uic"] = self.stations_df["code_uic"].apply(
            lambda x: x.split(";")[0] if ";" in str(x) else x
        )

        # Sauvegarder dans le cache
        self.stations_df.to_csv(self.stations_file, index=False, encoding="utf-8")
        print(f"[SAVED] Gares sauvegardees: {self.stations_file}")

        return self.stations_df

    def generate_connections_from_stations(self) -> pd.DataFrame:
        """
        Génère des connexions ferroviaires à partir du fichier railway_lines.csv

        Utilise les vraies lignes ferroviaires françaises (LGV, Intercités, TER)
        au lieu de connexions par proximité géographique.

        Returns:
            DataFrame avec les connexions
        """
        if self.stations_df is None:
            raise ValueError("Chargez d'abord les gares avec download_stations()")

        # Utiliser le cache si disponible
        if self.connections_file.exists():
            print(f"[OK] Chargement des connexions depuis le cache: {self.connections_file}")
            self.connections_df = pd.read_csv(self.connections_file)
            return self.connections_df

        print("[GENERATE] Generation des connexions ferroviaires depuis railway_lines.csv...")

        # Charger le fichier railway_lines.csv
        railway_lines_file = self.cache_dir / "railway_lines.csv"

        if not railway_lines_file.exists():
            raise FileNotFoundError(
                f"Le fichier {railway_lines_file} n'existe pas. "
                "Veuillez créer ce fichier avec les vraies lignes ferroviaires."
            )

        # Lire le fichier CSV (en ignorant les commentaires)
        railway_lines = pd.read_csv(railway_lines_file, comment='#')

        print(f"   {len(railway_lines)} lignes ferroviaires chargées")

        # Créer les connexions à partir des lignes ferroviaires
        connections = []
        not_found_stations = set()

        for _, line in railway_lines.iterrows():
            from_city = line["from_station"]
            to_city = line["to_station"]

            # Trouver les gares correspondantes
            from_station = self.get_station_by_name(from_city)
            to_station = self.get_station_by_name(to_city)

            if not from_station:
                not_found_stations.add(from_city)
                continue

            if not to_station:
                not_found_stations.add(to_city)
                continue

            # Normaliser les codes UIC (gérer les codes composites)
            from_uic = str(from_station["code_uic"]).split(";")[0]
            to_uic = str(to_station["code_uic"]).split(";")[0]

            # Créer la connexion bidirectionnelle (aller)
            connections.append({
                "from_uic": from_uic,
                "from_name": from_station["nom_gare"],
                "to_uic": to_uic,
                "to_name": to_station["nom_gare"],
                "distance_km": line["distance_km"],
                "time_minutes": line["time_minutes"],
                "line_type": line["line_type"],
                "line_code": line["line_code"]
            })

            # Connexion bidirectionnelle (retour)
            connections.append({
                "from_uic": to_uic,
                "from_name": to_station["nom_gare"],
                "to_uic": from_uic,
                "to_name": from_station["nom_gare"],
                "distance_km": line["distance_km"],
                "time_minutes": line["time_minutes"],
                "line_type": line["line_type"],
                "line_code": line["line_code"]
            })

        if not_found_stations:
            print(f"   [WARNING] Gares non trouvees: {', '.join(sorted(not_found_stations)[:5])}...")

        self.connections_df = pd.DataFrame(connections)

        # Sauvegarder dans le cache
        self.connections_df.to_csv(self.connections_file, index=False, encoding="utf-8")
        print(f"[SAVED] {len(connections)} connexions bidirectionnelles sauvegardees: {self.connections_file}")

        return self.connections_df

    def get_station_by_name(self, name: str) -> Optional[Dict]:
        """
        Recherche une gare par son nom (fuzzy matching + alias)

        Args:
            name: Nom de la gare ou de la ville

        Returns:
            Dictionnaire avec les infos de la gare, ou None
        """
        if self.stations_df is None:
            raise ValueError("Chargez d'abord les gares avec download_stations()")

        # Dictionnaire de correspondances pour les villes sans gare SNCF
        # ou avec des noms différents
        # IMPORTANT: Ne mapper QUE vers des gares qui existent vraiment dans la base SNCF
        city_aliases = {
            # PRINCIPALES GARES TERMINUS (Privilégier les gares centrales TGV)
            'paris': 'paris gare de lyon',  # Principale gare pour TGV Sud-Est
            'lyon': 'lyon part dieu',  # Gare TGV principale (SANS TIRET!)
            'marseille': 'marseille saint-charles',  # Terminus principal (pas Vitrolles Aéroport)
            'lille': 'lille flandres',  # Gare centrale (ou Lille Europe pour TGV)
            'bordeaux': 'bordeaux saint-jean',  # Gare principale
            'toulouse': 'toulouse matabiau',  # Gare principale
            'nantes': 'nantes',  # OK tel quel
            'strasbourg': 'strasbourg',  # OK tel quel (Strasbourg-Ville)
            'nice': 'nice',  # Gare simple "Nice" (code NIC)
            'aix-en-provence': 'aix-en-provence tgv',  # Gare TGV principale

            # Banlieues parisiennes -> Gares SNCF vérifiées qui existent
            'levallois-perret': 'clichy - levallois',  # Gare existante
            'neuilly-sur-seine': 'neuilly porte maillot',  # Gare existante
            'villeneuve-d\'ascq': 'ascq',  # Gare existante
            'villeneuve-dascq': 'ascq',  # Gare existante
            'issy-les-moulineaux': 'issy val de seine',  # Gare existante
            'maisons-alfort': 'alfortville',  # Gare proche
            'clichy': 'clichy - levallois',  # Gare existante

            # Villes SANS gare SNCF proche correcte -> PAS D'ALIAS
            # (retourneront INVALID mais le nom original sera préservé grâce à original_from/to)
            # 'saint-maur-des-fossés': Pas de gare proche -> INVALID
            # 'champigny-sur-marne': "Champigny" trouve "Champigny sur Yonne" (loin) -> INVALID
            # 'boulogne-billancourt': "Boulogne" trouve "Boulogne-sur-Mer" (Nord) -> INVALID
            # 'rueil-malmaison': Pas de gare trouvée -> INVALID

            # Alias courants (avec/sans accents)
            'st-nazaire': 'saint-nazaire',
            'saint nazaire': 'saint-nazaire',
            'st-etienne': 'saint-étienne',
            'saint etienne': 'saint-étienne',
            'chambery': 'chambéry',  # Sans accent -> avec accent
            'valence': 'valence tgv',  # Valence Drôme (pas Valence Espagne)
        }

        # Vérifier si c'est un alias
        name_lower = name.lower().strip()
        if name_lower in city_aliases:
            name_lower = city_aliases[name_lower]

        # Recherche exacte (insensible à la casse)
        matches = self.stations_df[
            self.stations_df["nom_gare"].str.lower().str.contains(name_lower, na=False) |
            self.stations_df["commune"].str.lower().str.contains(name_lower, na=False)
        ]

        # Fuzzy matching désactivé car il introduit trop d'erreurs
        # (ex: "Champigny-sur-Marne" trouvé comme "Champigny-sur-Yonne")
        # Préférer des alias explicites plutôt que du fuzzy matching agressif

        if len(matches) == 0:
            return None

        # Prioriser les gares suggestables
        suggestable = matches[matches["is_suggestable"] == True]
        if len(suggestable) > 0:
            matches = suggestable

        # Retourner la première correspondance
        return matches.iloc[0].to_dict()

    def get_all_stations(self) -> pd.DataFrame:
        """Retourne toutes les gares chargées"""
        if self.stations_df is None:
            self.download_stations()
        return self.stations_df

    def get_all_connections(self) -> pd.DataFrame:
        """Retourne toutes les connexions"""
        if self.connections_df is None:
            if self.stations_df is None:
                self.download_stations()
            self.generate_connections_from_stations()
        return self.connections_df

# Test
if __name__ == "__main__":
    loader = SNCFDataLoader()

    print("="*70)
    print("TEST DU SNCF DATA LOADER")
    print("="*70)

    # Télécharger les gares
    stations = loader.download_stations()
    print(f"\n📊 {len(stations)} gares chargées")
    print("\nExemple de gares:")
    print(stations[["nom_gare", "commune", "departement"]].head(10))

    # Générer les connexions
    connections = loader.generate_connections_from_stations()
    print(f"\n🔗 {len(connections)} connexions générées")
    print("\nExemple de connexions:")
    print(connections[["from_name", "to_name", "distance_km", "time_minutes"]].head(10))

    # Test de recherche
    print("\n🔍 Test de recherche:")
    for city in ["Paris", "Lyon", "Marseille", "Bordeaux"]:
        station = loader.get_station_by_name(city)
        if station:
            print(f"   {city} → {station['nom_gare']} (UIC: {station['code_uic']})")
        else:
            print(f"   {city} → Non trouvée")
