# ÉTAPE 1 : Créer le Dataset

## 🎯 Objectif

Créer un dataset **annoté** pour entraîner notre NER custom.

## 📚 C'est quoi un dataset annoté ?

C'est une liste de phrases où on a **marqué** les entités qu'on veut que le modèle apprenne.

### Format spaCy :
```python
(
    "Je veux aller de Paris à Lyon",  # Le texte
    {
        "entities": [
            (17, 22, "DEPART"),   # "Paris" commence à 17, finit à 22
            (25, 29, "ARRIVEE")   # "Lyon" commence à 25, finit à 29
        ]
    }
)
```

### Comment trouver les positions ?

```python
texte = "Je veux aller de Paris à Lyon"
#        0123456789...

# "Paris" commence où ?
print(texte.find("Paris"))  # → 17

# "Paris" finit où ?
print(texte.find("Paris") + len("Paris"))  # → 17 + 5 = 22
```

## 📊 Le Split Train/Val/Test

**Pourquoi splitter ?**

| Split | % | Utilité |
|-------|---|---------|
| **Train** | 70% | Pour entraîner le modèle |
| **Validation** | 15% | Pour ajuster les paramètres pendant l'entraînement |
| **Test** | 15% | Pour évaluer à la fin (jamais vu par le modèle!) |

**Analogie :**
- **Train** = Les exercices d'entraînement
- **Validation** = Les exercices de révision
- **Test** = L'examen final (surprise!)

## 🔧 Le Script de Génération

Crée le fichier `src/dataset/generate_dataset.py` :

```python
"""
ÉTAPE 1 : Génération du Dataset Annoté
=====================================
Ce script génère des phrases annotées pour entraîner un NER custom.
"""

import random
import json
from pathlib import Path

# =============================================================================
# DONNÉES DE BASE
# =============================================================================

# Liste des gares (à enrichir avec le vrai fichier SNCF)
GARES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
    "Nice", "Nantes", "Strasbourg", "Montpellier", "Lille",
    "Rennes", "Grenoble", "Dijon", "Annecy", "Chambéry",
    "Avignon", "Cannes", "Toulon", "Metz", "Nancy",
    "Reims", "Le Havre", "Rouen", "Clermont-Ferrand", "Tours"
]

# Prénoms (pour les phrases complexes - ce ne sont PAS des villes!)
PRENOMS = ["Florence", "Marie", "Jean", "Pierre", "Sophie", "Lucas", "Emma"]

# =============================================================================
# TEMPLATES DE PHRASES
# =============================================================================

# Format: (template, type)
# {D} = Départ, {A} = Arrivée, {P} = Prénom

TEMPLATES = [
    # --- Patterns STANDARD (départ avant arrivée) ---
    ("Je veux aller de {D} à {A}", "standard"),
    ("Un billet de {D} vers {A}", "standard"),
    ("Je pars de {D} pour {A}", "standard"),
    ("Train {D} {A}", "standard"),
    ("{D} - {A} s'il vous plaît", "standard"),
    ("Aller simple de {D} à {A}", "standard"),
    ("Je voudrais aller de {D} jusqu'à {A}", "standard"),

    # --- Patterns INVERSÉS (arrivée mentionnée avant départ) ---
    ("Direction {A} depuis {D}", "inverse"),
    ("Vers {A} en partant de {D}", "inverse"),
    ("Pour aller à {A}, je pars de {D}", "inverse"),
    ("À destination de {A} depuis {D}", "inverse"),
    ("{A} au départ de {D}", "inverse"),

    # --- Patterns COMPLEXES (avec prénoms/contexte) ---
    ("{P} et moi voulons aller à {A} depuis {D}", "complexe"),
    ("Avec {P}, on va de {D} à {A}", "complexe"),
    ("{P} m'attend à {A}, je suis à {D}", "complexe"),
    ("Je rejoins {P} à {A} depuis {D}", "complexe"),
]

# Phrases sans itinéraire (INVALIDES)
TEMPLATES_INVALIDES = [
    "Bonjour, comment allez-vous ?",
    "Quel temps fait-il ?",
    "Je cherche un hôtel",
    "Où sont les toilettes ?",
    "Merci beaucoup",
    "Un café s'il vous plaît",
    "Quelle heure est-il ?",
]

# =============================================================================
# FONCTIONS
# =============================================================================

def trouver_position(texte, mot):
    """
    Trouve la position d'un mot dans un texte.

    Returns:
        (start, end) ou None si pas trouvé
    """
    start = texte.find(mot)
    if start == -1:
        return None
    end = start + len(mot)
    return (start, end)


def generer_phrase_valide():
    """
    Génère UNE phrase valide avec ses annotations.

    Returns:
        dict avec text, entities, depart, arrivee, valid
    """
    # Choisir un template au hasard
    template, pattern_type = random.choice(TEMPLATES)

    # Choisir 2 villes différentes
    depart, arrivee = random.sample(GARES, 2)

    # Choisir un prénom (si besoin)
    prenom = random.choice(PRENOMS)

    # Construire la phrase
    texte = template.format(D=depart, A=arrivee, P=prenom)

    # Trouver les positions des entités
    entities = []

    pos_depart = trouver_position(texte, depart)
    pos_arrivee = trouver_position(texte, arrivee)

    if pos_depart:
        entities.append((pos_depart[0], pos_depart[1], "DEPART"))
    if pos_arrivee:
        entities.append((pos_arrivee[0], pos_arrivee[1], "ARRIVEE"))

    # Trier par position (important pour spaCy)
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
    """
    texte = random.choice(TEMPLATES_INVALIDES)

    return {
        "text": texte,
        "entities": [],  # Pas d'entités !
        "depart": "INVALID",
        "arrivee": "INVALID",
        "pattern_type": "invalid",
        "valid": False
    }


def generer_dataset(n_valides=8000, n_invalides=2000, seed=42):
    """
    Génère le dataset complet.

    Args:
        n_valides: Nombre de phrases avec itinéraire
        n_invalides: Nombre de phrases sans itinéraire
        seed: Pour la reproductibilité
    """
    random.seed(seed)  # Toujours le même résultat

    data = []

    # Générer les phrases valides
    print(f"Génération de {n_valides} phrases valides...")
    for i in range(n_valides):
        data.append(generer_phrase_valide())
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_valides}")

    # Générer les phrases invalides
    print(f"Génération de {n_invalides} phrases invalides...")
    for _ in range(n_invalides):
        data.append(generer_phrase_invalide())

    # Mélanger
    random.shuffle(data)

    return data


def split_dataset(data, train_ratio=0.70, val_ratio=0.15):
    """
    Divise le dataset en train/validation/test.

    Args:
        data: Liste des exemples
        train_ratio: % pour l'entraînement (0.70 = 70%)
        val_ratio: % pour la validation (0.15 = 15%)
        # test = le reste (0.15 = 15%)
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
    """
    Sauvegarde les splits en fichiers JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for nom, donnees in splits.items():
        filepath = output_path / f"{nom}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)

        print(f"✅ Sauvegardé: {filepath} ({len(donnees)} exemples)")


def afficher_stats(splits):
    """
    Affiche les statistiques du dataset.
    """
    print("\n" + "=" * 50)
    print("📊 STATISTIQUES DU DATASET")
    print("=" * 50)

    for nom, donnees in splits.items():
        valides = sum(1 for d in donnees if d['valid'])
        invalides = len(donnees) - valides

        print(f"\n{nom.upper()}:")
        print(f"  Total:     {len(donnees)}")
        print(f"  Valides:   {valides} ({valides/len(donnees)*100:.1f}%)")
        print(f"  Invalides: {invalides} ({invalides/len(donnees)*100:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🚀 GÉNÉRATION DU DATASET ANNOTÉ")
    print("=" * 50)

    # 1. Générer
    data = generer_dataset(n_valides=8000, n_invalides=2000)
    print(f"\n✅ Total généré: {len(data)} phrases")

    # 2. Splitter
    splits = split_dataset(data)

    # 3. Sauvegarder
    sauvegarder(splits)

    # 4. Stats
    afficher_stats(splits)

    # 5. Exemple
    print("\n" + "=" * 50)
    print("📝 EXEMPLE D'ANNOTATION")
    print("=" * 50)
    exemple = splits['train'][0]
    print(f"Texte: {exemple['text']}")
    print(f"Entités: {exemple['entities']}")
    print(f"Départ: {exemple['depart']}")
    print(f"Arrivée: {exemple['arrivee']}")
```

## ▶️ Commandes à exécuter

```bash
# 1. Aller dans le projet
cd /home/lekrikri/Projects/T-AIA-911-PAR_14

# 2. Créer les dossiers
mkdir -p src/dataset data/processed

# 3. Créer le script (copie le code ci-dessus)
nano src/dataset/generate_dataset.py

# 4. Exécuter
python src/dataset/generate_dataset.py
```

## 📁 Résultat

Après exécution, tu auras :
```
data/processed/
├── train.json   # 7000 phrases (70%)
├── val.json     # 1500 phrases (15%)
└── test.json    # 1500 phrases (15%)
```

## 🔍 Vérifier le résultat

```python
import json

# Charger
with open("data/processed/train.json", 'r') as f:
    train = json.load(f)

# Voir un exemple
print(train[0])
```

**Output :**
```json
{
  "text": "Je veux aller de Paris à Lyon",
  "entities": [[17, 22, "DEPART"], [25, 29, "ARRIVEE"]],
  "depart": "Paris",
  "arrivee": "Lyon",
  "valid": true
}
```

---

**Prochaine étape →** [ETAPE_2_SPACY_BASICS.md](ETAPE_2_SPACY_BASICS.md)
