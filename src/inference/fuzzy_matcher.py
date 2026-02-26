"""
ÉTAPE 5 : Fuzzy Matching pour les gares SNCF
=============================================
Corrige les fautes d'orthographe vers les vraies gares.

Exécuter avec : python3 src/inference/fuzzy_matcher.py
"""

import csv
from pathlib import Path
from rapidfuzz import fuzz, process


# =============================================================================
# CHARGEMENT DES GARES SNCF DEPUIS LE FICHIER CSV
# =============================================================================

def charger_gares_sncf(filepath="data/sncf/stations.csv"):
    """
    Charge les noms de gares depuis le fichier CSV SNCF.

    Args:
        filepath: Chemin vers stations.csv

    Returns:
        Liste des noms de gares
    """
    gares = []

    # Trouver le fichier (chercher depuis la racine du projet)
    possible_paths = [
        Path(filepath),
        Path(__file__).parent.parent.parent / filepath,
        Path("/home/lekrikri/Projects/T-AIA-911-PAR_14") / filepath,
    ]

    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break

    if not csv_path:
        print(f"ATTENTION: Fichier {filepath} non trouve!")
        print("Utilisation d'une liste de secours...")
        return [
            "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
            "Nice", "Nantes", "Strasbourg", "Montpellier", "Lille"
        ]

    # Lire le CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nom = row.get('nom_gare', '').strip()
            if nom:
                gares.append(nom)

    return gares


# Charger les gares au démarrage
GARES_SNCF = charger_gares_sncf()


# =============================================================================
# CLASSE FUZZY MATCHER
# =============================================================================

class FuzzyStationMatcher:
    """
    Trouve la gare SNCF la plus proche d'un texte donné.
    Utilise les 2786 gares officielles SNCF.
    """

    def __init__(self, stations=None):
        """
        Args:
            stations: Liste des gares. Si None, utilise les gares SNCF.
        """
        self.stations = stations or GARES_SNCF
        print(f"FuzzyMatcher initialise avec {len(self.stations)} gares SNCF")

    def match(self, query, threshold=60):
        """
        Trouve la gare la plus proche.

        Args:
            query: Texte à matcher (ex: "Marsei")
            threshold: Score minimum (0-100) pour accepter un match

        Returns:
            (gare, score) ou (None, 0) si pas de match
        """
        if not query or query.upper() in ["INVALID", "UNKNOWN", ""]:
            return None, 0

        # Normaliser
        query_normalized = query.lower().strip()

        # 1. Match EXACT (insensible à la casse)
        for station in self.stations:
            if station.lower() == query_normalized:
                return station, 100

        # 2. Si query contient "gare de" ou un suffixe de gare, chercher un match précis
        #    Ex: "Paris Gare de Lyon" doit matcher exactement, pas "Paris Est"
        gare_keywords = ["gare de", "gare du", "part-dieu", "saint-charles",
                         "saint-jean", "matabiau", "montparnasse", "saint-lazare",
                         "perrache", " est", " nord", " sud", " ouest"]

        is_specific_station = any(kw in query_normalized for kw in gare_keywords)

        if is_specific_station:
            # Fuzzy matching strict pour les gares spécifiques
            result = process.extractOne(
                query_normalized,
                self.stations,
                scorer=fuzz.ratio,  # Score strict
                score_cutoff=70,
                processor=lambda x: x.lower()
            )
            if result:
                return result[0], result[1]

        # 3. Pour les noms simples (Paris, Lyon...), garder le nom tel quel
        #    Le pathfinder choisira la bonne gare selon la destination
        simple_cities = ["paris", "lyon", "marseille", "bordeaux", "toulouse",
                        "nice", "nantes", "strasbourg", "montpellier", "lille",
                        "rennes", "grenoble", "dijon", "annecy", "avignon",
                        "reims", "metz", "nancy", "tours", "orleans", "rouen",
                        "toulon", "clermont-ferrand", "limoges", "brest", "caen"]

        if query_normalized in simple_cities:
            # Retourner le nom capitalisé (le pathfinder gèrera)
            return query.capitalize(), 100

        # 4. Chercher si query est le DÉBUT d'une gare
        starts_with = []
        for station in self.stations:
            station_lower = station.lower()
            # Vérifier que c'est un début de mot, pas juste de string
            if station_lower.startswith(query_normalized + " ") or \
               station_lower.startswith(query_normalized + "-") or \
               station_lower == query_normalized:
                starts_with.append(station)

        if starts_with:
            best = min(starts_with, key=len)
            return best, 95

        # 5. Fuzzy matching classique
        result = process.extractOne(
            query_normalized,
            self.stations,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
            processor=lambda x: x.lower()
        )

        if result:
            return result[0], result[1]
        return None, 0

    def match_top_n(self, query, n=3, threshold=50):
        """
        Retourne les N meilleures correspondances.
        Utile pour proposer des suggestions.

        Args:
            query: Texte à matcher
            n: Nombre de résultats
            threshold: Score minimum

        Returns:
            Liste de (gare, score)
        """
        if not query:
            return []

        results = process.extract(
            query,
            self.stations,
            scorer=fuzz.WRatio,
            limit=n,
            score_cutoff=threshold
        )

        return [(name, score) for name, score, _ in results]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU FUZZY MATCHING")
    print("=" * 60)

    # Créer le matcher
    matcher = FuzzyStationMatcher()

    # Cas de test
    tests = [
        # Exact
        ("Paris", "Match exact"),
        ("Lyon", "Match exact"),
        ("Marseille", "Match exact"),

        # Fautes d'orthographe
        ("Marsei", "Faute - manque lettres"),
        ("Pari", "Faute - manque 's'"),
        ("Lyone", "Faute - 'e' en trop"),
        ("Bordeau", "Faute - manque 'x'"),
        ("Toulouze", "Faute - 'z' au lieu de 's'"),

        # Variations
        ("paris", "Minuscules"),
        ("LYON", "Majuscules"),
        ("St Etienne", "Abréviation"),
        ("Aix en Provence", "Sans tirets"),

        # Cas difficiles
        ("Gare de Lyon", "Confusion possible avec Paris Gare de Lyon"),
        ("Montpelier", "Faute courante"),
        ("Straburg", "Faute"),
    ]

    print(f"\n{'Query':<20} {'Match':<25} {'Score':>6}  Note")
    print("-" * 70)

    for query, note in tests:
        match, score = matcher.match(query)
        match_str = match if match else "(pas de match)"
        print(f"{query:<20} {match_str:<25} {score:>5.0f}%  {note}")

    # Test avec suggestions
    print("\n" + "=" * 60)
    print("TEST DES SUGGESTIONS (top 3)")
    print("=" * 60)

    query = "Marsei"
    suggestions = matcher.match_top_n(query, n=3)
    print(f"\nPour '{query}':")
    for gare, score in suggestions:
        print(f"  - {gare} ({score:.0f}%)")
