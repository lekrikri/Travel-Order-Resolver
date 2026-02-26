"""
ÉTAPE 1 : Génération du Dataset Annoté pour NER
================================================
Ce script génère des phrases annotées avec les labels DEPART et ARRIVEE.

Exécuter avec : python src/dataset/generate_dataset.py
"""

import random
import json
from pathlib import Path

# =============================================================================
# DONNÉES DE BASE
# =============================================================================

# Liste des gares SNCF (échantillon)
GARES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
    "Nice", "Nantes", "Strasbourg", "Montpellier", "Lille",
    "Rennes", "Grenoble", "Dijon", "Annecy", "Chambery",
    "Avignon", "Cannes", "Toulon", "Metz", "Nancy",
    "Reims", "Le Havre", "Rouen", "Clermont-Ferrand", "Tours",
    "Orleans", "Poitiers", "Limoges", "Brest", "Caen"
]

# Prénoms (pour créer des phrases complexes - ce ne sont PAS des villes!)
PRENOMS = ["Florence", "Marie", "Jean", "Pierre", "Sophie", "Lucas", "Emma", "Louis"]

# =============================================================================
# TEMPLATES DE PHRASES
# {D} = Départ, {A} = Arrivée, {P} = Prénom
# =============================================================================

TEMPLATES = [
    # --- STANDARD (départ mentionné avant arrivée) ---
    ("Je veux aller de {D} a {A}", "standard"),
    ("Un billet de {D} vers {A}", "standard"),
    ("Je pars de {D} pour {A}", "standard"),
    ("Train {D} {A}", "standard"),
    ("{D} - {A} s'il vous plait", "standard"),
    ("Aller simple de {D} a {A}", "standard"),
    ("Je voudrais aller de {D} jusqu'a {A}", "standard"),
    ("Billet {D} {A} pour demain", "standard"),
    ("Je dois me rendre a {A} depuis {D}", "standard"),

    # --- INVERSÉ (arrivée mentionnée avant départ dans la phrase) ---
    ("Direction {A} depuis {D}", "inverse"),
    ("Vers {A} en partant de {D}", "inverse"),
    ("Pour aller a {A} je pars de {D}", "inverse"),
    ("A destination de {A} depuis {D}", "inverse"),
    ("{A} au depart de {D}", "inverse"),

    # --- COMPLEXE (avec prénoms qui ne sont PAS des villes) ---
    ("{P} et moi voulons aller a {A} depuis {D}", "complexe"),
    ("Avec {P} on va de {D} a {A}", "complexe"),
    ("{P} m'attend a {A} je suis a {D}", "complexe"),
    ("Je rejoins {P} a {A} depuis {D}", "complexe"),
    ("{P} veut qu'on parte de {D} pour aller a {A}", "complexe"),
]

# Phrases SANS itinéraire (le modèle doit apprendre à les ignorer)
TEMPLATES_INVALIDES = [
    "Bonjour comment allez-vous",
    "Quel temps fait-il aujourd'hui",
    "Je cherche un hotel",
    "Ou sont les toilettes",
    "Merci beaucoup",
    "Un cafe s'il vous plait",
    "Quelle heure est-il",
    "Je voudrais un sandwich",
    "Combien ca coute",
    "Au revoir et bonne journee",
]


# =============================================================================
# FONCTIONS
# =============================================================================

def trouver_position(texte, mot):
    """
    Trouve la position (start, end) d'un mot dans un texte.

    Exemple:
        texte = "Je vais a Paris"
        trouver_position(texte, "Paris") → (10, 15)
    """
    start = texte.find(mot)
    if start == -1:
        return None
    end = start + len(mot)
    return (start, end)


def generer_phrase_valide():
    """
    Génère UNE phrase avec un itinéraire annoté.

    Returns:
        {
            "text": "Je veux aller de Paris a Lyon",
            "entities": [(17, 22, "DEPART"), (25, 29, "ARRIVEE")],
            "depart": "Paris",
            "arrivee": "Lyon",
            "valid": True
        }
    """
    # 1. Choisir un template
    template, pattern_type = random.choice(TEMPLATES)

    # 2. Choisir 2 villes DIFFÉRENTES
    depart, arrivee = random.sample(GARES, 2)

    # 3. Choisir un prénom (si le template en a besoin)
    prenom = random.choice(PRENOMS)

    # 4. Construire la phrase
    texte = template.format(D=depart, A=arrivee, P=prenom)

    # 5. Trouver les positions des entités
    entities = []

    pos_depart = trouver_position(texte, depart)
    pos_arrivee = trouver_position(texte, arrivee)

    if pos_depart:
        entities.append((pos_depart[0], pos_depart[1], "DEPART"))
    if pos_arrivee:
        entities.append((pos_arrivee[0], pos_arrivee[1], "ARRIVEE"))

    # 6. Trier par position (spaCy l'exige)
    entities.sort(key=lambda x: x[0])

    return {
        "text": texte,
        "entities": entities,
        "depart": depart,
        "arrivee": arrivee,
        "pattern_type": pattern_type,
        "valid": True
    }


def generer_phrase_invalide():
    """
    Génère une phrase SANS itinéraire.
    Le modèle doit apprendre à ne rien extraire de ces phrases.
    """
    texte = random.choice(TEMPLATES_INVALIDES)

    return {
        "text": texte,
        "entities": [],  # Aucune entité !
        "depart": "INVALID",
        "arrivee": "INVALID",
        "pattern_type": "invalid",
        "valid": False
    }


def generer_dataset(n_valides=8000, n_invalides=2000, seed=42):
    """
    Génère le dataset complet.

    Args:
        n_valides: Nombre de phrases avec itinéraire (80%)
        n_invalides: Nombre de phrases sans itinéraire (20%)
        seed: Graine aléatoire pour reproductibilité
    """
    random.seed(seed)

    data = []

    # Phrases valides
    print(f"Generation de {n_valides} phrases valides...")
    for i in range(n_valides):
        data.append(generer_phrase_valide())
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_valides}")

    # Phrases invalides
    print(f"Generation de {n_invalides} phrases invalides...")
    for _ in range(n_invalides):
        data.append(generer_phrase_invalide())

    # Mélanger
    random.shuffle(data)

    return data


def split_dataset(data, train_ratio=0.70, val_ratio=0.15):
    """
    Divise le dataset en 3 parties:
    - train (70%): pour entraîner le modèle
    - val (15%): pour ajuster pendant l'entraînement
    - test (15%): pour évaluer à la fin
    """
    n = len(data)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


def sauvegarder(splits, output_dir="data/processed"):
    """Sauvegarde les splits en JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for nom, donnees in splits.items():
        filepath = output_path / f"{nom}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)

        print(f"Sauvegarde: {filepath} ({len(donnees)} exemples)")


def afficher_exemples(splits, n=3):
    """Affiche quelques exemples du dataset."""
    print("\n" + "=" * 60)
    print("EXEMPLES DU DATASET")
    print("=" * 60)

    for i, exemple in enumerate(splits['train'][:n]):
        print(f"\nExemple {i+1}:")
        print(f"  Texte: \"{exemple['text']}\"")
        print(f"  Depart: {exemple['depart']}")
        print(f"  Arrivee: {exemple['arrivee']}")
        print(f"  Entities: {exemple['entities']}")
        print(f"  Type: {exemple['pattern_type']}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATION DU DATASET ANNOTE POUR NER")
    print("=" * 60)

    # 1. Générer les données
    data = generer_dataset(n_valides=8000, n_invalides=2000)
    print(f"\nTotal genere: {len(data)} phrases")

    # 2. Splitter en train/val/test
    splits = split_dataset(data)

    # 3. Afficher les stats
    print("\n" + "-" * 60)
    print("STATISTIQUES:")
    print("-" * 60)
    for nom, donnees in splits.items():
        valides = sum(1 for d in donnees if d['valid'])
        print(f"  {nom.upper():6}: {len(donnees):5} phrases ({valides} valides, {len(donnees)-valides} invalides)")

    # 4. Sauvegarder
    print("\n" + "-" * 60)
    sauvegarder(splits)

    # 5. Afficher des exemples
    afficher_exemples(splits)

    print("\n" + "=" * 60)
    print("DATASET GENERE AVEC SUCCES!")
    print("=" * 60)
    print("\nFichiers crees:")
    print("  - data/processed/train.json (70%)")
    print("  - data/processed/val.json   (15%)")
    print("  - data/processed/test.json  (15%)")
