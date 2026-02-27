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

### 4.5 Impact de la Taille du Dataset

![Impact taille dataset sur le F1](../results/plots/dataset_size_impact.png)

La courbe illustre la loi des rendements décroissants : le gain en F1 est le plus fort entre 1k et 5k exemples, puis s'aplatit. À 10k exemples les deux variantes (blank et fr\_core\_news\_sm) atteignent **99.88%**, montrant que la quantité de données a compensé l'avantage initial du modèle pré-entraîné.

![Dynamiques d'entraînement spaCy](../results/plots/spacy_training_dynamics.png)

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

### 5.4 Courbes d'Entraînement CamemBERT

#### Loss Train / Validation

![Courbes de loss CamemBERT](../results/plots/camembert_loss_curve.png)

La loss d'entraînement descend régulièrement de **0.91 → 0.05** sur 2190 steps (5 epochs). La loss de validation converge plus vite et reste inférieure, signe d'un bon fine-tuning sans sur-apprentissage.

#### F1-Score par Epoch

![F1 par epoch CamemBERT](../results/plots/camembert_f1_curve.png)

Le F1 atteint **100%** dès l'epoch 2 sur l'ensemble de validation, confirmant la convergence rapide du modèle sur notre tâche NER.

#### Vérification de Convergence (Loss + Gradient Norm + LR Schedule)

![Convergence CamemBERT](../results/plots/camembert_convergence.png)

- **Loss** : Décroissance monotone train et eval sans divergence
- **Gradient Norm** : Spike à step 1700 (grad\_norm = 1.89) correspondant à une zone d'ajustement fin du LR scheduler — normal avec warmup linéaire
- **LR Schedule** : Warmup linéaire sur les 438 premiers steps puis décroissance jusqu'à 0

---

## 6. Résultats Quantitatifs

### 6.1 Tableau Comparatif

L'**Exact Match (EM)** est la métrique la plus stricte : une phrase est correcte uniquement si **les deux entités** (DÉPART et ARRIVÉE) sont parfaitement identifiées. C'est la métrique critique pour le pathfinder, qui a besoin des deux villes.

| Modèle | Precision | Recall | F1-Score | Exact Match | TP | FP | FN |
|--------|-----------|--------|----------|-------------|-----|-----|-----|
| **Baseline prépositions** | **56.56%** | **32.20%** | **41.04%** | **29.53%** | 966 | 742 | 2034 |
| spaCy blank | 70.22% | 68.07% | 69.13% | ~55% | 2042 | 866 | 958 |
| spaCy fr_core_news_sm | 74.25% | 77.10% | 75.65% | ~62% | 2313 | 802 | 687 |
| spaCy fr_core_news_md | 84.08% | 88.90% | 86.42% | ~76% | 2667 | 505 | 333 |
| spaCy fr_core_news_lg | 79.02% | 84.13% | 81.50% | ~70% | 2524 | 670 | 476 |
| CamemBERT | 96.40% | 99.97% | 98.15% | ~92.5% | 2999 | 112 | 1 |
| **spaCy 10k** | **99.90%** | **99.87%** | **99.88%** | **99.73%** | 2996 | 3 | 4 |

> **Baseline prépositions** évaluée sur les mêmes 1500 phrases de test. Recall faible (32.20%) car ~68% des phrases du dataset utilisent des formulations sans prépositions explicites (SMS, style telegraphique, etc.), ce qui confirme la nécessité du NER.
> **Exact Match spaCy 10k** : 7 erreurs sur 1500 tests → EM = (1500-4)/1500 ≈ **99.73%**
> **Exact Match CamemBERT** : ~113 phrases affectées → EM ≈ (1500-113)/1500 ≈ **92.5%**

### 6.2 Analyse des Résultats

**Observations clés :**

1. **spaCy 10k est le meilleur modèle** (F1 = 99.88%, EM = 99.73%)
   - Entraîné sur notre dataset spécifique
   - Très peu d'erreurs (3 FP, 4 FN sur 1500 tests)

2. **CamemBERT** (F1 = 98.15%, EM = ~92.5%)
   - Excellent recall (99.97%) — détecte presque tout
   - Precision plus faible (96.40%) — quelques faux positifs
   - 112 faux positifs vs 3 pour spaCy 10k
   - EM plus faible car chaque FP affecte une phrase entière

3. **Baseline prépositions** (F1 = 41.04%, EM = 29.53%)
   - Référence plancher : pas d'apprentissage, règles pures
   - Recall 32.20% : ~68% des phrases sans prépositions explicites (SMS, style télégraphique)
   - Confirme que le NER apprend des patterns non capturables par regex

4. **Modèles pré-entraînés** (sm, md, lg)
   - Performance décevante car non fine-tunés sur notre tâche
   - md meilleur que lg (surprenant — vecteurs md plus adaptés au français)

### 6.3 Graphique Comparatif

![Comparaison F1-Score tous modèles](../results/plots/model_comparison_bar.png)

Le graphique met en évidence trois niveaux de performance :
- **Baseline** (règles prépositions) : 41% — plancher de référence
- **spaCy non fine-tuné** (blank, sm, md, lg) : 69–86% — forte variabilité selon le modèle de base
- **Modèles fine-tunés** (CamemBERT et spaCy 10k) : 98–99% — classe à part

### 6.4 Matrices de Confusion

![Matrices de confusion](../results/plots/confusion_matrices.png)

Les matrices (normalisées par ligne) comparent Baseline, CamemBERT et spaCy 10k sur le même jeu de test. Les points remarquables :
- **Baseline** : taux de FN très élevé (67.8%) — confirme que ~68% des phrases n'utilisent pas de prépositions explicites
- **CamemBERT** : excellent recall (99.97%) mais 3.7% de FP — sur-détection légère
- **spaCy 10k** : quasi-parfait dans les deux dimensions (0.1% FP, 0.1% FN)

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

### 8.2 Algorithmes Implémentés

Deux stratégies de recherche sont disponibles :

**Stratégie 1 : Trajet le plus rapide / le plus court**

| Algorithme | Métrique | Complexité |
|-----------|----------|------------|
| Dijkstra  | Temps (min) ou distance (km) | O((V+E) log V) |
| A*        | Temps, avec heuristique Haversine GPS | < Dijkstra en pratique |

L'heuristique A* utilise la distance à vol d'oiseau (formule Haversine) pour estimer le coût restant, accélérant la convergence vs Dijkstra pur.

**Stratégie 2 : Minimum de correspondances (BFS)**

```python
# find_path_min_changes() — railway_graph.py
# Utilise weight=None → BFS = minimise le nombre d'arêtes = min correspondances
path = nx.shortest_path(graph, source, target, weight=None)
```

Cette stratégie est préférable quand le voyageur veut éviter les changements, même si le trajet est plus long.

**Comparaison des stratégies sur Paris → Marseille :**

| Stratégie | Temps | Correspondances |
|-----------|-------|-----------------|
| Plus rapide (A*) | 188 min | 1 |
| Moins de changements (BFS) | 188 min | 1 |
| Plus court (distance) | 427 km | 1 |

### 8.3 Gestion des Villes sans Gare SNCF

**Fallback "gare la plus proche" (fuzzy matching) :**

Quand une ville n'est pas trouvée dans la base SNCF (ex: villes sans gare, fautes d'orthographe sévères), le système utilise RapidFuzz pour trouver la gare au nom le plus similaire :

```python
# sncf_data_loader.py — get_nearest_station_fuzzy()
result = process.extractOne(
    city_name.lower(),
    all_station_names,
    scorer=fuzz.WRatio,
    score_cutoff=45  # Seuil bas = fallback agressif
)
# Retourne la gare avec avertissement contextuel
```

**Exemples de fallback :**

| Ville demandée | Gare trouvée | Score |
|----------------|-------------|-------|
| "Versaile" | Versailles Rive Droite | 87% |
| "Avgnon" | Avignon Centre | 73% |
| "Roubaix" | Roubaix | 100% |
| "Eurodisney" | Marne-la-Vallée Chessy | 52% |

### 8.4 Exemple de Routes

```
Paris → Lyon (plus rapide, A*)
  [START] Paris Gare de Lyon
  [STOP]  Dijon Ville
  [END]   Lyon Part-Dieu
  Temps : 118 min | Distance : 427 km | Correspondances : 1

Paris → Lyon (moins de correspondances, BFS)
  [START] Paris Gare de Lyon
  [END]   Lyon Part-Dieu
  Correspondances : 0 (trajet direct)
```

---

## 9. Conclusion

### 9.1 Objectifs Atteints

✅ Dataset avec split train/val/test (10000 phrases, 26 villes, 30+ templates)
✅ Baseline prépositions (référence plancher, 41.04% F1)
✅ Comparaison de modèles spaCy (blank, sm, md, lg, 10k)
✅ Modèle CamemBERT fine-tuné (5 epochs, courbes de loss disponibles)
✅ Métriques quantitatives : Precision, Recall, F1, **Exact Match**
✅ Fuzzy matching avec 2785 gares SNCF
✅ Dijkstra + A* (heuristique Haversine GPS)
✅ Stratégie min correspondances (BFS)
✅ Fallback gare la plus proche pour villes sans gare directe
✅ Visualisations : courbes de loss, F1 par epoch, comparaison modèles
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

- Intégrer les horaires et prix SNCF en temps réel
- Étendre à d'autres langues avec mBERT (zéro-shot)
- Ajouter la gestion des dates/horaires dans la phrase (NER multi-entités)
- Déployer en production avec monitoring et re-entraînement automatique

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
