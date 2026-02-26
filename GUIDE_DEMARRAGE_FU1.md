# 🚀 Guide de Démarrage - Travel Order Resolver (T-AIA-911)
## Pour le Follow-Up 1 (FU1)

---

## 📋 Objectif du Projet

Créer un système NLP capable de :
1. **Extraire** les villes de départ et d'arrivée d'une phrase en français
2. **Calculer** l'itinéraire ferroviaire optimal sur le réseau SNCF

**Exemple :**
- Input: `"Je voudrais aller de Paris à Lyon"`
- Output: `Départ: Paris, Arrivée: Lyon, Itinéraire: Paris → Dijon → Lyon`

---

## 🎯 Attendus FU1 (Minimum requis)

| Critère | Description | Priorité |
|---------|-------------|----------|
| Dataset draft | Premier jeu de données pour l'entraînement NER | ⭐⭐⭐ Obligatoire |
| Baseline simple | Approche basée sur les prépositions | ⭐⭐⭐ Obligatoire |
| Métriques | Quantifier la performance de la baseline | ⭐⭐⭐ Obligatoire |
| Données SNCF | Télécharger et étudier la structure | ⭐⭐ Bonus |
| Approche spaCy | NER avec spaCy (intermédiaire) | ⭐⭐ Bonus |

---

## 📚 ÉTAPE 1 : Comprendre le Problème (Jour 1)

### 1.1 Le NER (Named Entity Recognition)
Le NER permet d'identifier des entités nommées dans un texte :
- **LOC** : Lieux (Paris, Lyon, Marseille)
- **PER** : Personnes
- **ORG** : Organisations

Pour ce projet, on s'intéresse aux **lieux (LOC)** qui sont des gares SNCF.

### 1.2 Le défi spécifique
Ce n'est pas juste du NER classique ! Il faut comprendre :
- Quelle ville est le **DÉPART** ?
- Quelle ville est l'**ARRIVÉE** ?

**Exemples complexes :**
```
"Je pars de Marseille pour aller à Nice"     → Départ: Marseille, Arrivée: Nice
"Direction Lyon depuis Paris"                 → Départ: Paris, Arrivée: Lyon
"Bordeaux-Toulouse s'il vous plaît"          → Départ: Bordeaux, Arrivée: Toulouse
```

---

## 📊 ÉTAPE 2 : Créer le Dataset (Jour 1-2)

### 2.1 Structure du Dataset
Créer un fichier CSV avec cette structure :

```csv
phrase,depart,arrivee,valid
"Je veux aller de Paris à Lyon",Paris,Lyon,true
"Direction Marseille depuis Bordeaux",Bordeaux,Marseille,true
"Billet pour Nice",INVALID,INVALID,false
```

### 2.2 Script de génération du dataset

Créer `generate_dataset.py` :

```python
import csv
import random

# Liste de gares SNCF courantes (pour commencer)
GARES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
    "Nice", "Nantes", "Strasbourg", "Montpellier", "Lille",
    "Rennes", "Grenoble", "Rouen", "Toulon", "Dijon"
]

# Templates de phrases
TEMPLATES_VALIDES = [
    "Je voudrais aller de {depart} à {arrivee}",
    "Un billet de {depart} vers {arrivee} s'il vous plaît",
    "Je pars de {depart} pour aller à {arrivee}",
    "Direction {arrivee} depuis {depart}",
    "{depart} - {arrivee} pour demain",
    "Aller simple de {depart} à {arrivee}",
    "Je dois me rendre à {arrivee} depuis {depart}",
    "Train de {depart} vers {arrivee}",
    "Voyage {depart} {arrivee}",
]

TEMPLATES_INVALIDES = [
    "Quel temps fait-il aujourd'hui ?",
    "Je cherche un hôtel à {ville}",
    "Bonjour, comment ça va ?",
    "Quelle heure est-il ?",
    "Un café s'il vous plaît",
]

def generate_dataset(n_valid=800, n_invalid=200):
    data = []

    # Phrases valides
    for _ in range(n_valid):
        template = random.choice(TEMPLATES_VALIDES)
        depart, arrivee = random.sample(GARES, 2)
        phrase = template.format(depart=depart, arrivee=arrivee)
        data.append({
            "phrase": phrase,
            "depart": depart,
            "arrivee": arrivee,
            "valid": "true"
        })

    # Phrases invalides
    for _ in range(n_invalid):
        template = random.choice(TEMPLATES_INVALIDES)
        if "{ville}" in template:
            phrase = template.format(ville=random.choice(GARES))
        else:
            phrase = template
        data.append({
            "phrase": phrase,
            "depart": "INVALID",
            "arrivee": "INVALID",
            "valid": "false"
        })

    random.shuffle(data)
    return data

def save_dataset(data, filename="dataset_fu1.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "depart", "arrivee", "valid"])
        writer.writeheader()
        writer.writerows(data)
    print(f"Dataset sauvegardé : {filename} ({len(data)} phrases)")

if __name__ == "__main__":
    dataset = generate_dataset(n_valid=800, n_invalid=200)
    save_dataset(dataset)
```

### 2.3 Exécuter
```bash
python generate_dataset.py
```

**Résultat :** Un fichier `dataset_fu1.csv` avec 1000 phrases.

---

## 🔧 ÉTAPE 3 : Baseline avec Prépositions (Jour 2-3)

### 3.1 L'idée
Utiliser des règles simples basées sur les prépositions :
- `"de X à Y"` → Départ: X, Arrivée: Y
- `"depuis X vers Y"` → Départ: X, Arrivée: Y
- `"X - Y"` → Départ: X, Arrivée: Y

### 3.2 Script baseline

Créer `baseline_prepositions.py` :

```python
import re
import csv

# Patterns regex pour détecter départ/arrivée
PATTERNS = [
    # "de Paris à Lyon"
    r"de\s+([A-Z][a-zéèêëàâäôöùûü\-]+)\s+(?:à|vers|pour)\s+([A-Z][a-zéèêëàâäôöùûü\-]+)",
    # "depuis Paris vers Lyon"
    r"depuis\s+([A-Z][a-zéèêëàâäôöùûü\-]+)\s+(?:vers|à|pour)\s+([A-Z][a-zéèêëàâäôöùûü\-]+)",
    # "Paris - Lyon" ou "Paris Lyon"
    r"([A-Z][a-zéèêëàâäôöùûü\-]+)\s*[\-–]\s*([A-Z][a-zéèêëàâäôöùûü\-]+)",
    # "direction Lyon depuis Paris"
    r"direction\s+([A-Z][a-zéèêëàâäôöùûü\-]+)\s+depuis\s+([A-Z][a-zéèêëàâäôöùûü\-]+)",
]

def extract_baseline(phrase):
    """Extrait départ/arrivée avec la méthode baseline (prépositions)"""

    for pattern in PATTERNS:
        match = re.search(pattern, phrase, re.IGNORECASE)
        if match:
            groups = match.groups()
            # Attention: certains patterns inversent l'ordre
            if "direction" in pattern:
                return groups[1].capitalize(), groups[0].capitalize()  # Inversé
            return groups[0].capitalize(), groups[1].capitalize()

    return "INVALID", "INVALID"

def evaluate_baseline(dataset_path="dataset_fu1.csv"):
    """Évalue la baseline sur le dataset"""

    correct = 0
    total = 0
    errors = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phrase = row['phrase']
            expected_depart = row['depart']
            expected_arrivee = row['arrivee']

            predicted_depart, predicted_arrivee = extract_baseline(phrase)

            total += 1
            if (predicted_depart.lower() == expected_depart.lower() and
                predicted_arrivee.lower() == expected_arrivee.lower()):
                correct += 1
            else:
                errors.append({
                    'phrase': phrase,
                    'expected': f"{expected_depart} → {expected_arrivee}",
                    'predicted': f"{predicted_depart} → {predicted_arrivee}"
                })

    precision = (correct / total) * 100

    print("=" * 60)
    print("RÉSULTATS BASELINE (Prépositions)")
    print("=" * 60)
    print(f"Total phrases testées : {total}")
    print(f"Prédictions correctes : {correct}")
    print(f"Précision : {precision:.2f}%")
    print("=" * 60)

    # Afficher quelques erreurs
    print("\nExemples d'erreurs (10 premières) :")
    for err in errors[:10]:
        print(f"  Phrase: {err['phrase']}")
        print(f"  Attendu: {err['expected']}")
        print(f"  Prédit: {err['predicted']}")
        print()

    return precision, errors

if __name__ == "__main__":
    precision, errors = evaluate_baseline()
```

### 3.3 Exécuter
```bash
python baseline_prepositions.py
```

**Résultat attendu :** ~40-60% de précision (la baseline est volontairement simple).

---

## 🧠 ÉTAPE 4 : Approche spaCy (Bonus - Jour 3-4)

### 4.1 Installation
```bash
pip install spacy
python -m spacy download fr_core_news_md
```

### 4.2 Script spaCy

Créer `baseline_spacy.py` :

```python
import spacy
import csv

# Charger le modèle français
nlp = spacy.load("fr_core_news_md")

def extract_spacy(phrase):
    """Extrait les villes avec spaCy NER"""
    doc = nlp(phrase)

    # Trouver toutes les entités de type LOC (lieu)
    locations = [ent.text for ent in doc.ents if ent.label_ == "LOC"]

    if len(locations) >= 2:
        # Prendre les 2 premières : supposer départ puis arrivée
        return locations[0], locations[1]
    elif len(locations) == 1:
        return locations[0], "INVALID"
    else:
        return "INVALID", "INVALID"

def evaluate_spacy(dataset_path="dataset_fu1.csv"):
    """Évalue spaCy sur le dataset"""

    correct = 0
    total = 0
    errors = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phrase = row['phrase']
            expected_depart = row['depart']
            expected_arrivee = row['arrivee']

            predicted_depart, predicted_arrivee = extract_spacy(phrase)

            total += 1
            if (predicted_depart.lower() == expected_depart.lower() and
                predicted_arrivee.lower() == expected_arrivee.lower()):
                correct += 1
            else:
                errors.append({
                    'phrase': phrase,
                    'expected': f"{expected_depart} → {expected_arrivee}",
                    'predicted': f"{predicted_depart} → {predicted_arrivee}"
                })

    precision = (correct / total) * 100

    print("=" * 60)
    print("RÉSULTATS SPACY NER")
    print("=" * 60)
    print(f"Total phrases testées : {total}")
    print(f"Prédictions correctes : {correct}")
    print(f"Précision : {precision:.2f}%")
    print("=" * 60)

    return precision, errors

if __name__ == "__main__":
    precision, errors = evaluate_spacy()
```

### 4.3 Problème connu avec spaCy
spaCy détecte les villes mais **ne comprend pas le contexte** :
- Il ne sait pas différencier départ/arrivée
- Il peut inverser l'ordre
- Performance attendue : ~30-35%

---

## 🚂 ÉTAPE 5 : Données SNCF (Bonus)

### 5.1 Télécharger les données
Aller sur : https://ressources.data.sncf.com/

Datasets utiles :
1. **Liste des gares** : `liste-des-gares.csv`
2. **Référentiel des lignes** : `referentiel-des-lignes.csv`

### 5.2 Script pour charger les gares

```python
import pandas as pd

def load_sncf_stations(filepath="liste-des-gares.csv"):
    """Charge la liste des gares SNCF"""
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')

    # Colonnes importantes
    # - Nom de la gare
    # - Code UIC
    # - Coordonnées GPS

    print(f"Nombre de gares : {len(df)}")
    print(f"Colonnes : {df.columns.tolist()}")

    return df

# Exemple d'utilisation
stations = load_sncf_stations()
print(stations.head())
```

---

## 📈 ÉTAPE 6 : Métriques d'Évaluation

### 6.1 Métriques à calculer

```python
def calculate_metrics(y_true, y_pred):
    """Calcule les métriques d'évaluation"""

    # Accuracy (précision globale)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    # Pour un rapport plus détaillé :
    # - True Positives (TP) : bonnes prédictions valides
    # - True Negatives (TN) : bonnes prédictions INVALID
    # - False Positives (FP) : prédit valide mais faux
    # - False Negatives (FN) : prédit INVALID mais était valide

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(y_true)
    }
```

### 6.2 Tableau comparatif à produire

| Approche | Précision | Temps/phrase | Avantages | Inconvénients |
|----------|-----------|--------------|-----------|---------------|
| Baseline (prépositions) | ~50% | <0.01s | Simple, rapide | Ne comprend pas le contexte |
| spaCy NER | ~32% | ~0.01s | Détecte les lieux | Confond départ/arrivée |

---

## 📁 Structure du Projet FU1

```
T-AIA-911-PAR_XX/
├── data/
│   ├── dataset_fu1.csv          # Dataset généré
│   └── liste-des-gares.csv      # (bonus) Données SNCF
├── src/
│   ├── generate_dataset.py      # Génération du dataset
│   ├── baseline_prepositions.py # Baseline simple
│   └── baseline_spacy.py        # (bonus) Approche spaCy
├── results/
│   └── evaluation_fu1.md        # Résultats des métriques
├── requirements.txt             # Dépendances
└── README.md                    # Documentation
```

### requirements.txt
```
spacy>=3.0.0
pandas>=1.5.0
```

---

## ✅ Checklist FU1

- [ ] **Dataset créé** (~1000 phrases minimum)
- [ ] **Baseline prépositions** implémentée et évaluée
- [ ] **Métriques calculées** (précision en %)
- [ ] **Documentation** des résultats
- [ ] (Bonus) Données SNCF téléchargées
- [ ] (Bonus) Approche spaCy testée

---

## 🎯 Ce qui vient après (FU2, FU3...)

Une fois le FU1 validé, les prochaines étapes seront :
1. **FU2** : Améliorer avec un LLM (Qwen2.5, Mistral...)
2. **FU3** : Ajouter le RAG (ChromaDB) pour ancrer sur les vraies gares
3. **Final** : API REST + Frontend + Pathfinding

**Notre projet final a atteint 100% de précision** en utilisant :
- Qwen2.5-3B-Instruct (LLM)
- ChromaDB (RAG)
- NetworkX (Pathfinding)
- FastAPI + React (Production)

---

## 💡 Conseils

1. **Commence simple** : La baseline avec prépositions suffit pour le FU1
2. **Documente tout** : Note tes résultats et tes observations
3. **Ne vise pas 100%** : L'objectif du FU1 est d'avoir une référence, pas la perfection
4. **Teste sur des cas variés** : Ajoute des phrases complexes dans ton dataset

---

*Guide créé le 22/01/2025 - Projet T-AIA-911*
