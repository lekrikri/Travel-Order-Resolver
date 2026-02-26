#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline complet: NLP + Pathfinder
Extrait les villes d'une phrase ET calcule l'itinéraire ferroviaire
"""

import sys
import io
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Imports des composants
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from qwen_extractor import Qwen25Extractor
from chroma_rag import ChromaRAG
from pathfinder import TrainPathfinder

class TravelOrderPipeline:
    """Pipeline complet pour résoudre des commandes de voyage en train"""

    def __init__(
        self,
        use_rag: bool = True,
        pathfinder_cache_dir: str = "./data/sncf"
    ):
        """
        Initialise le pipeline

        Args:
            use_rag: Utiliser ChromaDB RAG pour améliorer l'extraction NLP
            pathfinder_cache_dir: Répertoire de cache pour les données SNCF
        """
        print("="*70)
        print("INITIALISATION DU PIPELINE COMPLET")
        print("="*70)

        self.use_rag = use_rag

        # Composant 1: NLP (Extraction de villes)
        print("\n[1/3] Initialisation du module NLP (Qwen2.5)...")
        self.extractor = Qwen25Extractor()
        if not self.extractor.load_model():
            raise RuntimeError("Impossible de charger Qwen2.5")
        print("✅ Qwen2.5 chargé")

        # Composant 2: ChromaDB RAG (optionnel)
        self.rag = None
        if use_rag:
            print("\n[2/3] Initialisation de ChromaDB RAG...")
            self.rag = ChromaRAG()
            if not self.rag.initialize():
                print("⚠️  Impossible d'initialiser ChromaDB RAG, mode sans RAG")
                self.use_rag = False
            else:
                # Charger le dataset si nécessaire
                if self.rag.collection.count() == 0:
                    print("   Chargement des 100 phrases annotées...")
                    self.rag.load_dataset("./data/annotated/train_sentences_100.csv")
                print(f"✅ ChromaDB RAG prêt ({self.rag.collection.count()} exemples)")

        # Composant 3: Pathfinder (Calcul d'itinéraires)
        print("\n[3/3] Initialisation du Pathfinder SNCF...")
        self.pathfinder = TrainPathfinder(cache_dir=pathfinder_cache_dir)
        self.pathfinder.initialize()
        print("✅ Pathfinder prêt")

        print("\n" + "="*70)
        print("✅ PIPELINE COMPLET PRÊT")
        print("="*70)

    def _extract_from_xml_tags(self, sentence: str) -> tuple:
        """
        Extrait directement départ/arrivée depuis les tags XML <Dep> et <Arr>

        Args:
            sentence: Phrase originale avec tags XML

        Returns:
            (departure, arrival) ou (None, None) si pas de tags
        """
        import re

        # Extraire le contenu entre les tags <Dep> et <Arr>
        dep_match = re.search(r'<Dep>(.*?)</Dep>', sentence, re.IGNORECASE)
        arr_match = re.search(r'<Arr>(.*?)</Arr>', sentence, re.IGNORECASE)

        departure = dep_match.group(1).strip() if dep_match else None
        arrival = arr_match.group(1).strip() if arr_match else None

        return (departure, arrival)

    def _validate_cities_in_sentence(self, sentence: str, departure: str, arrival: str) -> bool:
        """
        Vérifie que les villes extraites sont présentes dans la phrase

        Args:
            sentence: Phrase originale
            departure: Ville de départ extraite
            arrival: Ville d'arrivée extraite

        Returns:
            True si les deux villes sont dans la phrase, False sinon
        """
        # Nettoyer la phrase
        clean_sentence = sentence.lower()
        clean_sentence = clean_sentence.replace('<dep>', '').replace('</dep>', '')
        clean_sentence = clean_sentence.replace('<arr>', '').replace('</arr>', '')

        # Vérifier présence (en tenant compte des variantes)
        dep_lower = departure.lower()
        arr_lower = arrival.lower()

        # Chercher dans la phrase (avec normalisation basique)
        dep_found = dep_lower in clean_sentence
        arr_found = arr_lower in clean_sentence

        # Vérifier aussi avec des variantes (enlever tirets, etc.)
        if not dep_found:
            dep_normalized = dep_lower.replace('-', ' ').replace("'", ' ')
            dep_found = dep_normalized in clean_sentence

        if not arr_found:
            arr_normalized = arr_lower.replace('-', ' ').replace("'", ' ')
            arr_found = arr_normalized in clean_sentence

        return dep_found and arr_found

    def _detect_and_fix_inversion(self, sentence: str, departure: str, arrival: str) -> tuple:
        """
        Détecte si départ/arrivée sont inversés en analysant leur position dans la phrase

        Args:
            sentence: Phrase originale
            departure: Ville de départ extraite
            arrival: Ville d'arrivée extraite

        Returns:
            (departure_corrected, arrival_corrected, was_inverted)
        """
        # Nettoyer la phrase (enlever les tags XML)
        clean_sentence = sentence.lower()
        clean_sentence = clean_sentence.replace('<dep>', '').replace('</dep>', '')
        clean_sentence = clean_sentence.replace('<arr>', '').replace('</arr>', '')

        # Trouver les positions des villes dans la phrase
        dep_lower = departure.lower()
        arr_lower = arrival.lower()

        dep_pos = clean_sentence.find(dep_lower)
        arr_pos = clean_sentence.find(arr_lower)

        # Si les deux villes sont trouvées et que le départ apparaît APRÈS l'arrivée
        if dep_pos != -1 and arr_pos != -1 and dep_pos > arr_pos:
            # Vérifier qu'il ne s'agit pas d'une phrase type "aller à X depuis Y"
            # où l'ordre grammatical inverse l'ordre logique
            before_dep = clean_sentence[:dep_pos].lower()

            # Patterns qui indiquent que l'ordre est intentionnellement inversé
            inverse_patterns = ['depuis', 'de retour', 'je pars de', 'au départ de', 'départ de']

            # Si on trouve un pattern d'inversion, ne pas corriger
            for pattern in inverse_patterns:
                if pattern in before_dep:
                    return (departure, arrival, False)

            # Sinon, c'est probablement une vraie inversion -> corriger
            return (arrival, departure, True)

        return (departure, arrival, False)

    def process_sentence(
        self,
        sentence_id: int,
        sentence: str,
        metric: str = "time"
    ) -> dict:
        """
        Traite une phrase complète: extraction NLP + calcul d'itinéraire

        Args:
            sentence_id: ID de la phrase
            sentence: Phrase en langage naturel
            metric: Métrique pour le pathfinder ("time" ou "distance")

        Returns:
            Dictionnaire avec tous les résultats
        """
        result = {
            "sentence_id": sentence_id,
            "sentence": sentence,
            "nlp_result": None,
            "route_result": None,
            "final_output": None,
            "error": None
        }

        # ÉTAPE 1: Extraction NLP
        try:
            if self.use_rag and self.rag:
                similar_examples = self.rag.get_similar_examples(sentence, n_results=3)
                nlp_result = self.extractor.extract_cities(sentence, examples=similar_examples)
            else:
                nlp_result = self.extractor.extract_cities(sentence)

            result["nlp_result"] = nlp_result

            # Vérifier si l'extraction a réussi
            if not nlp_result.get("departure") or not nlp_result.get("arrival"):
                result["error"] = "Impossible d'extraire les villes (commande invalide)"
                result["final_output"] = f"{sentence_id},INVALID"
                return result

            departure = nlp_result["departure"]
            arrival = nlp_result["arrival"]

            # ÉTAPE 1.5: Extraction depuis tags XML (si disponibles)
            # Toujours préférer les tags XML car ils sont fiables à 100%
            xml_dep, xml_arr = self._extract_from_xml_tags(sentence)
            used_xml = False

            if xml_dep and xml_arr:
                # Utiliser les tags XML (source de vérité)
                departure = xml_dep
                arrival = xml_arr
                nlp_result["departure"] = departure
                nlp_result["arrival"] = arrival
                nlp_result["xml_extraction_used"] = True
                used_xml = True
            else:
                # Pas de tags XML : valider l'extraction NLP
                cities_valid = self._validate_cities_in_sentence(sentence, departure, arrival)

                if not cities_valid:
                    # Le modèle a halluciné mais pas de tags XML
                    # Garder quand même (peut-être variante orthographique)
                    nlp_result["potential_hallucination"] = True

            # ÉTAPE 1.6: Détection et correction d'inversion
            # Ne corriger l'inversion QUE si on n'a PAS utilisé les tags XML
            # Car les tags XML sont déjà corrects
            if not used_xml:
                departure, arrival, was_inverted = self._detect_and_fix_inversion(sentence, departure, arrival)

                if was_inverted:
                    nlp_result["departure"] = departure
                    nlp_result["arrival"] = arrival
                    nlp_result["inversion_corrected"] = True

        except Exception as e:
            result["error"] = f"Erreur NLP: {str(e)}"
            result["final_output"] = f"{sentence_id},INVALID"
            return result

        # ÉTAPE 2: Calcul de l'itinéraire
        try:
            departure = nlp_result["departure"]
            arrival = nlp_result["arrival"]

            route = self.pathfinder.find_route(departure, arrival, metric=metric)
            result["route_result"] = route

            # Formater la sortie finale
            if "error" in route:
                result["error"] = route["error"]
                result["final_output"] = f"{sentence_id},INVALID"
            else:
                result["final_output"] = self.pathfinder.format_route_for_output(sentence_id, route)

        except Exception as e:
            result["error"] = f"Erreur Pathfinder: {str(e)}"
            result["final_output"] = f"{sentence_id},INVALID"

        return result

    def process_file(
        self,
        input_file: str,
        output_file: str,
        metric: str = "time",
        resume: bool = True
    ):
        """
        Traite un fichier CSV complet avec sauvegarde progressive

        Format d'entrée: sentenceID,sentence
        Format de sortie: sentenceID,Departure,Step1,Step2,...,Destination

        Args:
            input_file: Fichier CSV d'entrée
            output_file: Fichier CSV de sortie
            metric: Métrique pour le pathfinder
            resume: Si True, reprend où on s'est arrêté (détecte les IDs déjà traités)
        """
        import pandas as pd

        print("\n" + "="*70)
        print(f"TRAITEMENT DU FICHIER: {input_file}")
        print("="*70)

        # Charger le fichier (parsing manuel pour gérer les virgules dans les phrases)
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split seulement sur la première virgule
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        data.append({"sentence_id": int(parts[0]), "sentence": parts[1]})

        df = pd.DataFrame(data)

        # Déterminer quelles phrases ont déjà été traitées
        processed_ids = set()
        if resume and Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Extraire l'ID (premier élément)
                        sentence_id = int(line.split(',')[0])
                        processed_ids.add(sentence_id)

            print(f"\n📥 {len(df)} phrases au total")
            print(f"✅ {len(processed_ids)} phrases déjà traitées")
            print(f"🔄 {len(df) - len(processed_ids)} phrases restantes")
        else:
            print(f"\n📥 {len(df)} phrases à traiter")

        # Ouvrir le fichier en mode append pour sauvegarder progressivement
        output_mode = 'a' if (resume and Path(output_file).exists()) else 'w'

        errors = 0
        processed_count = 0

        with open(output_file, output_mode, encoding='utf-8') as f_out:
            for idx, row in df.iterrows():
                sentence_id = row["sentence_id"]
                sentence = row["sentence"]

                # Sauter si déjà traité
                if sentence_id in processed_ids:
                    continue

                print(f"\n[{idx+1}/{len(df)}] ID={sentence_id}")
                print(f"   Phrase: {sentence}")

                # Traiter la phrase
                result = self.process_sentence(sentence_id, sentence, metric=metric)

                # Afficher le résultat
                if result["error"]:
                    print(f"   ❌ Erreur: {result['error']}")
                    errors += 1
                else:
                    route = result["route_result"]
                    print(f"   ✅ NLP: {result['nlp_result']['departure']} → {result['nlp_result']['arrival']}")
                    print(f"   ✅ Route: {route['num_stops']} gares, {route['total_time_minutes']:.0f} min")
                    print(f"   🛤️  {' → '.join(route['stations'][:3])}{'...' if len(route['stations']) > 3 else ''}")

                # SAUVEGARDER IMMÉDIATEMENT (ligne par ligne)
                f_out.write(result["final_output"] + "\n")
                f_out.flush()  # Forcer l'écriture sur disque

                processed_count += 1

        # Statistiques finales
        print("\n" + "="*70)
        print("TRAITEMENT TERMINÉ")
        print("="*70)

        total_in_file = len(processed_ids) + processed_count

        print(f"\n✅ Fichier sauvegardé: {output_file}")
        print(f"📊 Statistiques:")
        print(f"   Total traité: {total_in_file}")
        print(f"   Nouvellement traité: {processed_count}")
        print(f"   Succès: {processed_count - errors}")
        print(f"   Erreurs: {errors}")
        if processed_count > 0:
            print(f"   Taux de réussite: {100 * (processed_count - errors) / processed_count:.1f}%")

        return processed_count

# Test
if __name__ == "__main__":
    print("="*70)
    print("TEST DU PIPELINE COMPLET")
    print("="*70)

    # Initialiser le pipeline
    pipeline = TravelOrderPipeline(use_rag=True)

    # Test sur quelques phrases
    print("\n" + "="*70)
    print("TEST SUR PHRASES ISOLÉES")
    print("="*70)

    test_sentences = [
        "Je voudrais un billet de train de Paris à Lyon pour demain",
        "Besoin d'un trajet Marseille Bordeaux",
        "Comment aller de Lille à Nice?",
        "Aller simple Strasbourg Toulouse s'il vous plaît",
        "Bonjour comment allez-vous?"  # Phrase invalide
    ]

    for i, sentence in enumerate(test_sentences, start=1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {sentence}")
        print('='*70)

        result = pipeline.process_sentence(i, sentence)

        if result["error"]:
            print(f"❌ Erreur: {result['error']}")
            print(f"📄 Sortie: {result['final_output']}")
        else:
            nlp = result["nlp_result"]
            route = result["route_result"]

            print(f"\n✅ EXTRACTION NLP:")
            print(f"   Départ: {nlp['departure']}")
            print(f"   Arrivée: {nlp['arrival']}")
            print(f"   Confiance: {nlp['confidence']:.0%}")

            print(f"\n✅ ITINÉRAIRE:")
            print(f"   Gares: {route['num_stops']}")
            print(f"   Changements: {route['num_changes']}")
            print(f"   Temps: {route['total_time_minutes']:.0f} min")

            print(f"\n🛤️  CHEMIN:")
            for j, station in enumerate(route['stations']):
                prefix = "   " + ("🚩" if j == 0 else "🏁" if j == len(route['stations']) - 1 else "🔸")
                print(f"{prefix} {station}")

            print(f"\n📄 FORMAT SORTIE:")
            print(f"   {result['final_output']}")

    print("\n" + "="*70)
    print("✅ TESTS TERMINÉS")
    print("="*70)
