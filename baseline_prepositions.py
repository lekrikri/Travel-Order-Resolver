#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASELINE PRÉPOSITIONS
=====================
Baseline simple pour l'extraction de villes depuis des ordres de voyage.
Utilise des règles basées sur les prépositions françaises (regex).

Sert de référence pour comparer avec spaCy et CamemBERT.

Usage:
    python baseline_prepositions.py
    python baseline_prepositions.py --eval data/annotated/test.csv
"""

import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Tuple


# =============================================================================
# STOP WORDS (mots à ignorer comme noms de villes)
# =============================================================================

STOP_WORDS = {
    'demain', 'train', 'billet', 'ticket', 'place', 'svp', 'stp', 'matin',
    'soir', 'prochain', 'prochaine', 'weekend', 'samedi', 'dimanche', 'lundi',
    'mardi', 'mercredi', 'jeudi', 'vendredi', 'aller', 'retour', 'simple',
    'direct', 'rapide', 'express', 'tgv', 'ter', 'intercites', 'intercités',
    'heure', 'heures', 'minute', 'minutes', 'classe', 'voyage', 'voyageur',
    'gare', 'bonsoir', 'bonjour', 'merci', 'svp', 'stp', 'please', 'voudrais',
    'veux', 'besoin', 'cherche', 'trouver', 'réserver', 'réservation', 'prendre',
    'aujourd', 'hui', 'ce', 'cet', 'cette', 'prochain', 'prochaine', 'semaine',
    'mois', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet',
    'août', 'septembre', 'octobre', 'novembre', 'décembre'
}


# =============================================================================
# PATTERNS REGEX
# Format : chaque pattern capture (départ, arrivée) dans les groupes 1 et 2
# =============================================================================

# Motif générique pour une ville : 1 ou 2 mots avec tirets/apostrophes
_C = r"([A-ZÀ-Ÿa-zà-ÿ][a-zà-ÿA-ZÀ-Ÿ'\-]{1,25}(?:[-\s][A-ZÀ-Ÿa-zà-ÿ][a-zà-ÿA-ZÀ-Ÿ'\-]{1,25})?)"

PATTERNS = [
    # -----------------------------------------------------------------------
    # Patterns avec "de ... à/vers/pour"   (les plus fréquents)
    # "Je veux aller de Paris à Lyon"
    # "Un billet de Marseille pour Bordeaux"
    # -----------------------------------------------------------------------
    (rf"(?:^|[\s,]+)de\s+{_C}\s+(?:à|vers|pour|jusqu'à|jusqu'a)\s+{_C}", "de...à"),

    # "aller de X à Y" / "partir de X vers Y"
    (rf"(?:aller|partir|voyager|trajets?)\s+de\s+{_C}\s+(?:à|vers|pour|jusqu'à|jusqu'a)\s+{_C}", "aller de...à"),

    # -----------------------------------------------------------------------
    # Patterns avec "depuis"
    # "depuis Lyon vers Paris"
    # "depuis Lyon à Paris"
    # -----------------------------------------------------------------------
    (rf"depuis\s+{_C}\s+(?:à|vers|pour|jusqu'à|jusqu'a|en\s+direction\s+de)\s+{_C}", "depuis...à"),

    # -----------------------------------------------------------------------
    # Patterns avec marqueurs explicites départ/arrivée
    # "départ Paris arrivée Lyon"
    # "départ de Paris arrivée à Lyon"
    # -----------------------------------------------------------------------
    (rf"(?:départ\s+(?:de\s+)?){_C}\s+(?:arrivée\s+(?:à\s+)?|destination\s+(?:finale\s+)?(?:à\s+)?|à\s+){_C}", "départ...arrivée"),

    # -----------------------------------------------------------------------
    # Patterns avec flèche ou tiret
    # "Paris → Lyon" / "Paris -> Lyon"
    # -----------------------------------------------------------------------
    (rf"^{_C}\s*(?:→|->|►|➔)\s*{_C}$", "X→Y"),
    (rf"{_C}\s*(?:→|->|►|➔)\s*{_C}", "X→Y"),

    # -----------------------------------------------------------------------
    # Patterns avec "direction" ou "destination"
    # "direction Lyon depuis Paris"
    # "destination Lyon de Paris"
    # -----------------------------------------------------------------------
    (rf"(?:direction|destination)\s+{_C}[,\s]+(?:depuis|de|en\s+partant\s+de|partant\s+de)\s+{_C}", "direction Y depuis X"),

    # "en direction de Lyon depuis Paris"
    (rf"en\s+direction\s+de\s+{_C}[,\s]+(?:depuis|de|en\s+partant\s+de)\s+{_C}", "en direction de Y depuis X"),

    # -----------------------------------------------------------------------
    # Patterns SMS / informels
    # "Lyon Paris demain soir"  → difficile à capturer sans NER
    # "Paris Lyon svp"
    # -----------------------------------------------------------------------
    # On essaie de capturer "VILLE VILLE" quand entourés de mots déclencheurs
    (rf"(?:train|billet|trajet|tgv|ter)\s+{_C}\s+{_C}", "billet X Y"),
]


# =============================================================================
# CLASSE BASELINE PRÉPOSITIONS
# =============================================================================

class BaselinePrepositions:
    """
    Baseline simple pour extraire départ/arrivée via les prépositions.

    Avantages:
        - Rapide (0.01ms / phrase)
        - Aucun entraînement requis
        - Interprétable

    Limites:
        - N'apprend pas de nouveaux patterns
        - Échoue sur les phrases sans prépositions explicites
        - Sensible à l'ordre des mots
    """

    def __init__(self):
        self.patterns = [
            (re.compile(pattern, re.IGNORECASE | re.UNICODE), name)
            for pattern, name in PATTERNS
        ]

    def _clean_city(self, city: str) -> Optional[str]:
        """Nettoie un nom de ville capturé."""
        if not city:
            return None
        city = city.strip().strip(',').strip()
        # Vérifier que ce n'est pas un stop word
        if city.lower() in STOP_WORDS:
            return None
        # Vérifier longueur minimale
        if len(city) < 2:
            return None
        return city

    def extract(self, sentence: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrait la ville de départ et d'arrivée depuis une phrase.

        Args:
            sentence: Phrase en langage naturel

        Returns:
            (départ, arrivée) ou (None, None) si non trouvé
        """
        # Nettoyer la phrase (enlever tags XML si présents)
        clean = re.sub(r'<[^>]+>', '', sentence).strip()

        for regex, pattern_name in self.patterns:
            match = regex.search(clean)
            if match:
                group1 = self._clean_city(match.group(1))
                group2 = self._clean_city(match.group(2)) if match.lastindex >= 2 else None

                # "direction Y depuis X" → inverser
                if pattern_name in ("direction Y depuis X", "en direction de Y depuis X"):
                    group1, group2 = group2, group1

                if group1 and group2:
                    return group1, group2

        return None, None

    def evaluate(self, test_file: str) -> dict:
        """
        Évalue la baseline sur un fichier CSV annoté.

        Format attendu: sentence,departure,arrival
        (avec ou sans tags XML <Dep>...</Dep> <Arr>...</Arr>)

        Args:
            test_file: Chemin vers le fichier CSV de test

        Returns:
            Dictionnaire de métriques
        """
        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"Fichier de test non trouvé: {test_file}")

        total = 0
        exact_match = 0
        dep_correct = 0
        arr_correct = 0
        not_extracted = 0

        with open(test_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentence = row.get('sentence', '')
                true_dep = row.get('departure', '').strip().lower()
                true_arr = row.get('arrival', '').strip().lower()

                if not sentence or not true_dep or not true_arr:
                    continue

                total += 1
                pred_dep, pred_arr = self.extract(sentence)

                if pred_dep is None or pred_arr is None:
                    not_extracted += 1
                    continue

                pred_dep = pred_dep.lower().strip()
                pred_arr = pred_arr.lower().strip()

                dep_ok = (pred_dep == true_dep)
                arr_ok = (pred_arr == true_arr)

                if dep_ok:
                    dep_correct += 1
                if arr_ok:
                    arr_correct += 1
                if dep_ok and arr_ok:
                    exact_match += 1

        if total == 0:
            return {"error": "Aucune donnée à évaluer"}

        extracted = total - not_extracted
        extraction_rate = extracted / total

        # Métriques sur les phrases extraites
        em_rate = exact_match / total
        dep_acc = dep_correct / total
        arr_acc = arr_correct / total
        avg_acc = (dep_acc + arr_acc) / 2

        # Pseudo F1 (ici accuracy = precision = recall car extraction binaire)
        f1 = em_rate

        return {
            "total_samples":    total,
            "extracted":        extracted,
            "not_extracted":    not_extracted,
            "extraction_rate":  round(extraction_rate * 100, 2),
            "exact_match":      round(em_rate * 100, 2),
            "departure_acc":    round(dep_acc * 100, 2),
            "arrival_acc":      round(arr_acc * 100, 2),
            "avg_accuracy":     round(avg_acc * 100, 2),
            "f1_score":         round(f1 * 100, 2),
        }


# =============================================================================
# TEST & DEMO
# =============================================================================

def demo():
    """Démontre la baseline sur des phrases exemples."""
    baseline = BaselinePrepositions()

    print("=" * 65)
    print("BASELINE PRÉPOSITIONS — Exemples")
    print("=" * 65)

    examples = [
        # Formulations classiques
        "Je veux aller de Paris à Lyon demain",
        "Un billet de Marseille pour Bordeaux s'il vous plaît",
        "Comment aller depuis Strasbourg à Toulouse ?",
        "Billet de train de Lille vers Nice pour vendredi",

        # Formulations SMS / informelles
        "Paris Lyon stp",
        "train Bordeaux Nantes",
        "billet Toulouse Montpellier demain matin",

        # Avec flèche
        "Paris → Lyon",
        "Marseille -> Nice",

        # Avec départ/arrivée explicite
        "départ Paris arrivée Lyon",
        "départ de Bordeaux arrivée à Brest",

        # Cas difficiles (sans prépositions claires)
        "Je dois être à Lyon ce soir",
        "Direction Grenoble depuis Lyon",

        # Majuscules/minuscules/fautes
        "de paris à lyon stp",
        "DE MARSEILLE POUR NICE",
    ]

    print(f"\n{'Phrase':<45} {'Départ':<18} {'Arrivée':<18}")
    print("-" * 85)

    correct = 0
    for phrase in examples:
        dep, arr = baseline.extract(phrase)
        dep_str = dep if dep else "(non détecté)"
        arr_str = arr if arr else "(non détecté)"
        ok = "✓" if (dep and arr) else "✗"
        if dep and arr:
            correct += 1
        print(f"  {ok}  {phrase[:42]:<42} {dep_str:<18} {arr_str:<18}")

    print(f"\n  Taux de détection: {correct}/{len(examples)} ({100*correct/len(examples):.0f}%)")


def evaluate_on_file(test_file: str):
    """Évalue la baseline sur un fichier CSV."""
    baseline = BaselinePrepositions()

    print("=" * 65)
    print(f"ÉVALUATION — {test_file}")
    print("=" * 65)

    metrics = baseline.evaluate(test_file)

    print(f"\n  Échantillons total :   {metrics['total_samples']}")
    print(f"  Phrases extraites :    {metrics['extracted']} ({metrics['extraction_rate']}%)")
    print(f"  Phrases non extraites: {metrics['not_extracted']}")
    print()
    print(f"  Exact Match:           {metrics['exact_match']}%")
    print(f"  Accuracy Départ:       {metrics['departure_acc']}%")
    print(f"  Accuracy Arrivée:      {metrics['arrival_acc']}%")
    print(f"  Accuracy Moyenne:      {metrics['avg_accuracy']}%")
    print(f"  F1-Score (EM-based):   {metrics['f1_score']}%")
    print()
    print("  NOTE: Cette baseline sert de RÉFÉRENCE PLANCHER.")
    print("  spaCy 10k = 99.88% F1 | CamemBERT = 98.15% F1")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline prépositions pour NER voyage")
    parser.add_argument("--eval", type=str, help="Fichier CSV à évaluer", default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate_on_file(args.eval)
    else:
        demo()

        # Tenter évaluation automatique si le fichier de test existe
        default_test = "data/annotated/train_sentences_10k.csv"
        if Path(default_test).exists():
            print("\n" + "=" * 65)
            print("ÉVALUATION AUTOMATIQUE sur 10k (premiers 500 exemples)")
            print("=" * 65)
            # Évaluer sur un sous-ensemble pour la démo
            import tempfile, shutil
            with open(default_test, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False,
                                              encoding='utf-8')
            tmp.writelines(lines[:501])  # header + 500 exemples
            tmp.close()
            evaluate_on_file(tmp.name)
            import os; os.unlink(tmp.name)
