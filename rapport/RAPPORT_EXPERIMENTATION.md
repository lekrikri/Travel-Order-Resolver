# Rapport d'Expérimentation - Travel Order Resolver

**Projet** : T-AIA-911-PAR_14
**Équipe** : Epitech MSc Pro 2026
**Date** : Janvier 2026

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Description du Problème](#2-description-du-problème)
3. [Architecture du Système](#3-architecture-du-système)
4. [Dataset](#4-dataset)
5. [Modèles NER](#5-modèles-ner)
6. [Résultats Quantitatifs](#6-résultats-quantitatifs)
7. [Analyse Qualitative](#7-analyse-qualitative)
8. [Pathfinder](#8-pathfinder)
9. [Conclusion](#9-conclusion)
10. [Annexes](#10-annexes)

---

## 1. Introduction

### 1.1 Contexte

Le projet **Travel Order Resolver** vise à extraire automatiquement les villes de départ et d'arrivée à partir de phrases en langage naturel français, puis à calculer un itinéraire ferroviaire optimal.

**Exemple :**
- Entrée : *"Je veux aller de Paris à Lyon"*
- Sortie : `DEPART: Paris` | `ARRIVEE: Lyon` | Itinéraire : Paris → Lyon

### 1.2 Objectifs

1. **Extraction NER** : Identifier les entités DEPART et ARRIVEE dans des phrases françaises
2. **Fuzzy Matching** : Corriger les fautes d'orthographe et valider les gares SNCF
3. **Pathfinding** : Calculer l'itinéraire optimal entre deux gares

### 1.3 Contraintes du Jury

- Dataset avec split train/validation/test
- Comparaison de plusieurs modèles spaCy (sm, md, lg, blank)
- Au moins un modèle transformer (CamemBERT)
- Métriques quantitatives d'évaluation
- Interface CLI (pas de Jupyter)
- Fuzzy matching pour la correction orthographique

---

## 2. Description du Problème

### 2.1 Tâche NER (Named Entity Recognition)

La reconnaissance d'entités nommées consiste à identifier et classifier des segments de texte en catégories prédéfinies.

**Notre tâche :**
- **DEPART** : Ville/gare de départ
- **ARRIVEE** : Ville/gare d'arrivée

### 2.2 Défis Identifiés

| Défi | Exemple | Solution |
|------|---------|----------|
| Ambiguïté prénom/ville | "Florence et moi..." | Filtrage contextuel |
| Fautes d'orthographe | "Bordeau", "Marsei" | Fuzzy matching |
| Formulations variées | "depuis X vers Y", "de X à Y" | Dataset varié |
| Gares multiples | Paris Est, Paris Lyon | Distinction par contexte |

### 2.3 Format BIO

Nous utilisons le schéma d'annotation **BIO** (Begin, Inside, Outside) :

| Token | Label |
|-------|-------|
| Je | O |
| veux | O |
| aller | O |
| de | O |
| Paris | **B-DEPART** |
| à | O |
| Lyon | **B-ARRIVEE** |

---

## 3. Architecture du Système

### 3.1 Pipeline Global

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAVEL ORDER RESOLVER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Phrase  │───▶│   NER    │───▶│  Fuzzy   │───▶│  Path    │  │
│  │  (input) │    │  spaCy/  │    │ Matching │    │  finder  │  │
│  │          │    │ CamemBERT│    │  (2785   │    │(Dijkstra)│  │
│  │          │    │          │    │  gares)  │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                       │               │               │         │
│                       ▼               ▼               ▼         │
│                   DEPART          Validation      Itinéraire    │
│                   ARRIVEE         SNCF           Optimal        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Composants

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| NER | spaCy / CamemBERT | Extraction des entités |
| Fuzzy Matcher | RapidFuzz | Correction orthographique |
| Pathfinder | NetworkX (Dijkstra) | Calcul d'itinéraire |
| API | FastAPI | Interface REST |
| Frontend | React + TypeScript | Interface utilisateur |

---

## 4. Dataset

### 4.1 Génération des Données

Le dataset a été généré automatiquement avec des templates variés :

**Templates utilisés :**
- "Je veux aller de {DEPART} à {ARRIVEE}"
- "Direction {ARRIVEE} depuis {DEPART}"
- "Train {DEPART} {ARRIVEE} pour demain"
- "Vers {ARRIVEE} en partant de {DEPART}"
- ... (30+ templates)

### 4.2 Répartition

| Split | Nombre d'exemples | Pourcentage |
|-------|------------------|-------------|
| Train | 7000 | 70% |
| Validation | 1500 | 15% |
| Test | 1500 | 15% |
| **Total** | **10000** | **100%** |

### 4.3 Format des Données

```json
{
  "text": "Je veux aller de Paris à Lyon",
  "entities": [
    [17, 22, "DEPART"],
    [25, 29, "ARRIVEE"]
  ]
}
```

### 4.4 Caractéristiques

- **Villes utilisées** : 26 grandes villes françaises
- **Formulations** : 30+ templates différents
- **Distribution** : Équilibrée entre les villes

---

## 5. Modèles NER

### 5.1 Modèles spaCy Testés

| Modèle | Description | Embeddings |
|--------|-------------|------------|
| **blank** | Modèle vide, entraîné from scratch | Appris |
| **fr_core_news_sm** | Petit modèle français pré-entraîné | Statiques |
| **fr_core_news_md** | Modèle moyen avec word vectors | Word2Vec |
| **fr_core_news_lg** | Grand modèle avec word vectors | Word2Vec |

### 5.2 CamemBERT

**CamemBERT** est un modèle transformer basé sur RoBERTa, entraîné sur 138GB de texte français.

| Caractéristique | Valeur |
|-----------------|--------|
| Architecture | RoBERTa |
| Paramètres | 110M |
| Vocabulaire | 32000 tokens |
| Contexte | 512 tokens |

**Avantages :**
- Embeddings contextuels (vs statiques pour spaCy)
- Compréhension profonde du français
- Transfer learning efficace

### 5.3 Configuration d'Entraînement

**spaCy :**
```python
EPOCHS = 30
BATCH_SIZE = 128
DROPOUT = 0.35
LEARNING_RATE = 0.001
```

**CamemBERT :**
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
```

---

## 6. Résultats Quantitatifs

### 6.1 Tableau Comparatif

| Modèle | Precision | Recall | F1-Score | TP | FP | FN |
|--------|-----------|--------|----------|-----|-----|-----|
| spaCy blank | 70.22% | 68.07% | 69.13% | 2042 | 866 | 958 |
| spaCy fr_core_news_sm | 74.25% | 77.10% | 75.65% | 2313 | 802 | 687 |
| spaCy fr_core_news_md | 84.08% | 88.90% | 86.42% | 2667 | 505 | 333 |
| spaCy fr_core_news_lg | 79.02% | 84.13% | 81.50% | 2524 | 670 | 476 |
| **spaCy 10k** | **99.90%** | **99.87%** | **99.88%** | 2996 | 3 | 4 |
| CamemBERT | 96.40% | 99.97% | 98.15% | 2999 | 112 | 1 |

### 6.2 Analyse des Résultats

**Observations clés :**

1. **spaCy 10k est le meilleur modèle** (F1 = 99.88%)
   - Entraîné sur notre dataset spécifique
   - Très peu d'erreurs (3 FP, 4 FN sur 1500 tests)

2. **CamemBERT** (F1 = 98.15%)
   - Excellent recall (99.97%) - détecte presque tout
   - Precision plus faible (96.40%) - quelques faux positifs
   - 112 faux positifs vs 3 pour spaCy 10k

3. **Modèles pré-entraînés** (sm, md, lg)
   - Performance décevante car non fine-tunés sur notre tâche
   - md meilleur que lg (surprenant)

### 6.3 Graphique Comparatif

```
F1-Score par modèle :

spaCy 10k     ████████████████████████████████████████████████████ 99.88%
CamemBERT     ████████████████████████████████████████████████░░░░ 98.15%
spaCy md      ██████████████████████████████████████░░░░░░░░░░░░░░ 86.42%
spaCy lg      ████████████████████████████████████░░░░░░░░░░░░░░░░ 81.50%
spaCy sm      █████████████████████████████████░░░░░░░░░░░░░░░░░░░ 75.65%
spaCy blank   █████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░ 69.13%
              0%                        50%                     100%
```

---

## 7. Analyse Qualitative

### 7.1 Comparaison des Approches

| Critère | spaCy 10k | CamemBERT |
|---------|-----------|-----------|
| **Performance** | 99.88% F1 | 98.15% F1 |
| **Vitesse d'inférence** | ~1ms (CPU) | ~50ms (GPU) |
| **Mémoire** | ~50 MB | ~500 MB |
| **GPU requis** | Non | Oui (recommandé) |
| **Temps d'entraînement** | ~5 min | ~30 min |

### 7.2 Cas d'Erreurs Analysés

**Erreurs spaCy 10k (7 erreurs / 1500 tests) :**

| Type | Exemple | Problème |
|------|---------|----------|
| FP (3) | "Avec Jean, on part..." | "Jean" détecté comme ville |
| FN (4) | "Direction sud vers..." | "sud" non détecté |

**Erreurs CamemBERT (113 erreurs / 1500 tests) :**

| Type | Exemple | Problème |
|------|---------|----------|
| FP (112) | Divers | Sur-détection d'entités |
| FN (1) | Cas rare | Sous-détection |

### 7.3 Gestion des Prénoms

Le pipeline intelligent filtre les prénoms courants grâce à :
- Une liste de prénoms français (Florence, Marie, Jean...)
- Une analyse contextuelle ("Florence et moi" → prénom, "Direction Florence" → ville)

```python
# Patterns détectés comme prénoms
"Florence et moi..."  → Prénom (filtré)
"Avec Jean on part..." → Prénom (filtré)

# Patterns détectés comme villes
"Direction Florence"  → Ville (gardé)
"De Jean vers Paris"  → Erreur possible
```

### 7.4 Recommandation

**Pour la production, nous recommandons spaCy 10k** :
- Meilleure performance (99.88% vs 98.15%)
- Pas de GPU nécessaire
- Inférence plus rapide
- Modèle plus léger

**CamemBERT** est intéressant pour :
- La recherche/expérimentation
- Les cas où le recall est critique
- L'extension à d'autres langues

---

## 8. Pathfinder

### 8.1 Données SNCF

| Donnée | Valeur |
|--------|--------|
| Gares | 2785 |
| Connexions | 8510 |
| Composante connexe | 1 (graphe connecté) |

### 8.2 Algorithme

**Dijkstra** (implémenté via NetworkX) :
- Trouve le chemin le plus court
- Métriques : temps (minutes) ou distance (km)
- Complexité : O((V + E) log V)

### 8.3 Exemple de Route

```
Paris → Lyon (optimisé par temps)

[START] Paris Gare de Lyon
[STOP]  Dijon Ville
[END]   Lyon Part-Dieu

Temps total : 118 minutes
Distance : 427 km
```

### 8.4 Extensions Possibles

- **Algorithme A*** : Avec heuristique euclidienne pour accélérer
- **Critère prix** : Intégrer les tarifs SNCF
- **Multi-critères** : Optimisation Pareto (temps + prix)

---

## 9. Conclusion

### 9.1 Objectifs Atteints

✅ Dataset avec split train/val/test (10000 phrases)
✅ Comparaison de modèles spaCy (blank, sm, md, lg)
✅ Modèle CamemBERT entraîné
✅ Métriques quantitatives (Precision, Recall, F1)
✅ Fuzzy matching avec 2785 gares SNCF
✅ Pathfinder avec Dijkstra
✅ Interface CLI

### 9.2 Performance Finale

| Métrique | Valeur |
|----------|--------|
| **Meilleur modèle** | spaCy 10k |
| **F1-Score** | 99.88% |
| **Precision** | 99.90% |
| **Recall** | 99.87% |

### 9.3 Leçons Apprises

1. **Le fine-tuning est crucial** : Un modèle simple (spaCy) bien entraîné sur la tâche bat un transformer généraliste

2. **La qualité des données prime** : 10k exemples bien générés suffisent

3. **L'architecture transformer n'est pas toujours nécessaire** : Pour une tâche NER simple avec peu de classes, spaCy suffit

### 9.4 Perspectives

- Ajouter A* avec heuristique géographique
- Intégrer les prix SNCF
- Étendre à d'autres langues avec mBERT
- Déployer en production avec monitoring

---

## 10. Annexes

### A. Structure du Projet

```
T-AIA-911-PAR_14/
├── data/
│   ├── processed/          # Dataset JSON
│   └── sncf/               # Données SNCF
├── models/
│   ├── ner_10k/            # Meilleur modèle spaCy
│   └── ner_camembert/      # Modèle CamemBERT
├── src/
│   ├── dataset/            # Génération dataset
│   ├── models/             # Entraînement
│   ├── inference/          # Pipeline + Fuzzy
│   ├── evaluation/         # Métriques
│   └── 3_pathfinder/       # Calcul d'itinéraires
├── frontend/               # Interface React
└── rapport/                # Ce document
```

### B. Commandes

```bash
# Entraîner spaCy
python3 src/models/train_all_models.py

# Entraîner CamemBERT
python3 src/models/train_camembert.py

# Évaluer tous les modèles
python3 src/evaluation/evaluate_all.py

# CLI interactive
python3 cli.py

# API + Frontend
uvicorn api:app --reload
npm run dev
```

### C. Dépendances

```
spacy==3.7+
transformers==4.35+
torch==2.0+
rapidfuzz==3.0+
networkx==3.0+
pandas==2.0+
```

---

**Fin du rapport**

*Document généré le 29/01/2026*
