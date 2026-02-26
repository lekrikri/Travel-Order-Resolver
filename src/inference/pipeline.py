"""
Pipeline Intelligent : NER + Fuzzy Matching + Post-traitement
==============================================================
Combine l'extraction NER avec validation et correction.

Exécuter avec : python3 src/inference/pipeline.py
"""

import sys
from pathlib import Path

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

import spacy
from fuzzy_matcher import FuzzyStationMatcher, GARES_SNCF


class TravelOrderPipeline:
    """
    Pipeline complet pour extraire et valider les itinéraires.

    Étapes :
    1. NER : Extrait DEPART et ARRIVEE
    2. Validation : Vérifie que les villes existent (fuzzy matcher)
    3. Post-traitement : Gère les cas ambigus (Florence prénom vs ville)
    """

    def __init__(self, model_path="models/ner_10k/model-best"):
        print("=" * 50)
        print("INITIALISATION DU PIPELINE")
        print("=" * 50)

        # Charger le NER
        print(f"Chargement NER: {model_path}")
        self.nlp = spacy.load(model_path)

        # Charger le Fuzzy Matcher
        print("Chargement Fuzzy Matcher...")
        self.matcher = FuzzyStationMatcher()

        # Liste des prénoms courants (à filtrer)
        self.prenoms = {
            "florence", "marie", "jean", "pierre", "sophie", "lucas",
            "emma", "louis", "paul", "julie", "thomas", "camille",
            "nicolas", "antoine", "claire", "laurent", "michel", "bernard"
        }

        print("Pipeline pret!\n")

    def _is_valid_station(self, name):
        """
        Vérifie si un nom correspond à une gare SNCF valide.

        Returns:
            (is_valid, matched_name, score)
        """
        if not name:
            return False, None, 0

        matched, score = self.matcher.match(name)

        # Score minimum de 80 pour considérer comme valide
        if matched and score >= 80:
            return True, matched, score
        return False, None, score

    def _is_likely_prenom(self, name, context):
        """
        Détermine si un nom est probablement un prénom plutôt qu'une ville.

        Args:
            name: Le nom détecté
            context: La phrase complète

        Returns:
            True si c'est probablement un prénom
        """
        name_lower = name.lower()

        # 1. Est-ce dans notre liste de prénoms ?
        if name_lower in self.prenoms:
            # Vérifier le contexte : "Florence et moi", "avec Jean"
            prenom_patterns = [
                f"{name_lower} et moi",
                f"{name_lower} et ",
                f"avec {name_lower}",
                f"pour {name_lower}",
                f"rejoindre {name_lower}",
                f"attendre {name_lower}",
            ]
            context_lower = context.lower()
            for pattern in prenom_patterns:
                if pattern in context_lower:
                    return True

        return False

    def extract(self, text):
        """
        Extrait départ et arrivée avec validation et post-traitement.

        Returns:
            dict avec tous les détails de l'extraction
        """
        # 1. Extraction NER brute
        doc = self.nlp(text)

        departs_raw = []
        arrivees_raw = []

        for ent in doc.ents:
            if ent.label_ == "DEPART":
                departs_raw.append(ent.text)
            elif ent.label_ == "ARRIVEE":
                arrivees_raw.append(ent.text)

        # 2. Filtrer et valider chaque entité
        departs_valid = []
        arrivees_valid = []

        for dep in departs_raw:
            # Vérifier si c'est un prénom
            if self._is_likely_prenom(dep, text):
                continue  # Ignorer les prénoms

            # Vérifier si la ville existe
            is_valid, matched, score = self._is_valid_station(dep)
            if is_valid:
                departs_valid.append({
                    "raw": dep,
                    "matched": matched,
                    "score": score
                })

        for arr in arrivees_raw:
            if self._is_likely_prenom(arr, text):
                continue

            is_valid, matched, score = self._is_valid_station(arr)
            if is_valid:
                arrivees_valid.append({
                    "raw": arr,
                    "matched": matched,
                    "score": score
                })

        # 3. Post-traitement : sélectionner le meilleur
        # Pour DEPART : prendre le dernier valide (souvent le bon avec "depuis X")
        # Pour ARRIVEE : prendre le premier valide
        depart_final = None
        arrivee_final = None

        if departs_valid:
            # Prendre le dernier DEPART (gère "Florence et moi... depuis Paris")
            best = departs_valid[-1]
            depart_final = best["matched"]

        if arrivees_valid:
            # Prendre la première ARRIVEE
            best = arrivees_valid[0]
            arrivee_final = best["matched"]

        # 4. Résultat
        return {
            "text": text,
            "departs_raw": departs_raw,
            "arrivees_raw": arrivees_raw,
            "departs_valid": departs_valid,
            "arrivees_valid": arrivees_valid,
            "depart": depart_final,
            "arrivee": arrivee_final,
            "valid": bool(depart_final and arrivee_final)
        }

    def resolve(self, text):
        """
        Version simplifiée qui retourne juste départ/arrivée.
        """
        result = self.extract(text)
        return result["depart"], result["arrivee"]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Créer le pipeline
    pipeline = TravelOrderPipeline()

    # Phrases de test
    tests = [
        # Cas normaux
        "Je veux aller de Paris a Lyon",
        "Direction Marseille depuis Bordeaux",
        "Train Nantes Rennes pour demain",

        # Cas avec prénom (doit filtrer Florence)
        "Florence et moi voulons aller a Annecy depuis Paris",
        "Avec Jean, on part de Lille pour Nice",
        "Je rejoins Marie a Lyon, je suis a Paris",

        # Cas inversés
        "A destination de Toulouse depuis Bordeaux",
        "Vers Strasbourg en partant de Metz",

        # Phrases sans itinéraire
        "Bonjour, comment allez-vous ?",
        "Quel temps fait-il ?",

        # Fautes d'orthographe (fuzzy matching)
        "Je pars de Bordeau pour Toulouze",
        "Billet Marsei Lyon svp",
    ]

    print("=" * 70)
    print("TEST DU PIPELINE INTELLIGENT")
    print("=" * 70)

    print(f"\n{'Phrase':<55} {'Depart':<12} {'Arrivee':<12} {'OK'}")
    print("-" * 85)

    for phrase in tests:
        result = pipeline.extract(phrase)

        depart = result['depart'] or "-"
        arrivee = result['arrivee'] or "-"
        valid = "OK" if result['valid'] else "X"

        # Tronquer la phrase si trop longue
        phrase_display = phrase[:53] + ".." if len(phrase) > 55 else phrase
        print(f"{phrase_display:<55} {depart:<12} {arrivee:<12} {valid}")

        # Afficher les détails si filtrage
        if result['departs_raw'] != [d['raw'] for d in result['departs_valid']]:
            filtered = set(result['departs_raw']) - {d['raw'] for d in result['departs_valid']}
            if filtered:
                print(f"  [filtre: {filtered} (prenom ou invalide)]")

    print("\n" + "=" * 70)
    print("PIPELINE INTELLIGENT FONCTIONNEL!")
    print("=" * 70)
