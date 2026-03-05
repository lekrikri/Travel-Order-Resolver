# Analyse Comparative des Engines NLP - Travel Order Resolver

**Projet** : T-AIA-911-PAR_14
**Équipe** : Epitech MSc Pro 2026
**Date** : Mars 2026

---

## Table des Matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture des 4 engines](#2-architecture-des-4-engines)
3. [Approche par engine](#3-approche-par-engine)
4. [Comparaison des coûts opérationnels](#4-comparaison-des-coûts-opérationnels)
5. [Pertinence par rapport au cas d'usage](#5-pertinence-par-rapport-au-cas-dusage)
6. [Impact du fine-tuning](#6-impact-du-fine-tuning)
7. [Conclusion et recommandation](#7-conclusion-et-recommandation)

---

## 1. Vue d'ensemble

Le projet implémente 4 engines NLP sous un **Strategy Pattern** commun (`NLPEngine`), chacun représentant un niveau de complexité croissant :

| Engine | Type | Apprentissage | Extraction des rôles (dep/dest) |
|--------|------|---------------|--------------------------------|
| **Rule-based** | Regex + heuristiques | Aucun | Direct (named groups dans les regex) |
| **CRF** | ML classique (modèle linéaire) | Automatique (données synthétiques) | Direct (tags BIO : B-DEP / B-DEST) |
| **spaCy** | Réseau de neurones (NER pré-entraîné) | Pré-entraîné (fr_core_news_md) | Indirect (LOC + heuristique `assign_roles`) |
| **CamemBERT** | Transformer (110M paramètres) | Pré-entraîné (camembert-ner) | Indirect (LOC + heuristique `assign_roles`) |

---

## 2. Architecture des 4 engines

### 2.1 Pipeline commun

```
Texte utilisateur
       │
       ▼
  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │  Engine  │────▶│  Gazetteer   │────▶│  (dep, dest, │
  │  NLP     │     │  Resolution  │     │   confidence)│
  └──────────┘     └──────────────┘     └──────────────┘
       │
       ├── Rule-based : regex patterns → resolve_greedy()
       ├── CRF        : features → tags BIO → resolve_city()
       ├── spaCy      : NER → LOC entities → assign_roles()
       └── CamemBERT  : NER → LOC entities → assign_roles()
```

### 2.2 Distinction fondamentale

Les 4 engines partagent le même gazetteer (2785 gares SNCF) mais diffèrent sur deux axes :

- **Comment ils trouvent les villes** dans le texte (regex, features manuelles, embeddings, attention)
- **Comment ils assignent les rôles** départ/destination (direct vs heuristique)

Le CRF et le rule-based prédisent directement "ceci est un départ, ceci est une destination". SpaCy et CamemBERT ne détectent que des "lieux" (label LOC) et dépendent d'une heuristique par mots-clés (`assign_roles()`) pour distinguer le rôle.

---

## 3. Approche par engine

### 3.1 Rule-based (regex patterns)

**Fichier** : `src/nlp/rule_based.py`

Utilise 8 patterns regex ordonnés par spécificité, chacun avec des named groups `(?P<dep>...)` et `(?P<arr>...)` :

| Pattern | Exemple |
|---------|---------|
| `de <dep> à <arr>` | "de Paris à Lyon" |
| `depuis <dep> vers <arr>` | "depuis Nantes vers Bordeaux" |
| `en partance de <dep> ... destination <arr>` | "en partance de Lille destination Marseille" |
| `<dep> direction <arr>` | "train Toulouse direction Nice" |
| `vers <arr> depuis <dep>` (inversé) | "vers Lyon depuis Paris" |
| `billet <dep>-<arr>` | "billet Strasbourg-Rennes" |
| `habite à <dep> ... pars à <arr>` | "habite à Albi, pars à Angers" |
| (variante inversée du précédent) | "pars à Angers, habite à Albi" |

**Fallback** : si aucun regex ne matche, un système de scoring contextuel scanne tous les mots, identifie les villes connues du gazetteer, et les classe par score selon les mots environnants ("habite à" = départ, "vers" = destination, "avec mon amie" = pénalité personne).

**Forces** : aucune dépendance ML, inférence instantanée, déterministe.
**Faiblesses** : chaque formulation imprévue nécessite d'ajouter un nouveau pattern manuellement.

### 3.2 CRF (Conditional Random Field)

**Fichier** : `src/nlp/crf_ner.py`

Modèle de séquence linéaire (sklearn-crfsuite) qui assigne un tag BIO à chaque token :

- `B-DEP` / `I-DEP` : début/intérieur d'une entité départ
- `B-DEST` / `I-DEST` : début/intérieur d'une entité destination
- `O` : aucune entité

**Entraînement** : généré automatiquement à partir de 65 templates français et de paires de villes du gazetteer (~800 phrases synthétiques). L'entraînement prend ~2-5 secondes au premier appel.

**Features manuelles** (15 features par token) :

| Feature | Rôle |
|---------|------|
| `word.in_gazetteer` | Le token est-il une ville connue ? |
| `prev.is_prep` | Le mot précédent est-il une préposition (de, vers, à...) ? |
| `prev2.is_habite_verb` | 2 mots avant : verbe d'habitation (habite, vis...) ? |
| `word.is_person_ctx` | Contexte de personne (ami, copain, avec...) ? |
| `bigram.in_gazetteer` | Le bigramme (2 mots consécutifs) est-il une ville ? |
| `word.shape` | Forme du mot (Xxxxx, xxxx, dd...) |

Le CRF apprend les **poids** de chaque feature à partir des données. Son calcul est une somme pondérée : `score = poids_1 x feature_1 + poids_2 x feature_2 + ...`, d'où le terme "modèle linéaire".

**Forces** : léger, rapide, prédit directement dep/dest, pas de dépendance GPU.
**Faiblesses** : limité aux formulations couvertes par les 65 templates d'entraînement.

### 3.3 spaCy (NER pré-entraîné)

**Fichier** : `src/nlp/spacy_ner.py`

Utilise le modèle français `fr_core_news_md` (~50 Mo) avec un pipeline NER pré-entraîné sur de la presse et Wikipedia. Ne détecte que des entités `LOC` (lieu) sans distinguer départ et destination.

**Pipeline** : `texte → spaCy NER → filtrer LOC → resolve_city() → assign_roles()`

La fonction `assign_roles()` (dans `gazetteer.py`) est une heuristique par mots-clés qui scanne les 5 mots précédant chaque entité pour deviner le rôle : les mots "de", "depuis" favorisent le départ ; "vers", "pour" favorisent la destination.

**Forces** : bonne reconnaissance d'entités dans du texte varié, gère les cas ambigus (prénoms vs villes).
**Faiblesses** : modèle généraliste (50 Mo) pour une tâche de domaine fermé ; dépend d'une heuristique pour les rôles.

### 3.4 CamemBERT (Transformer)

**Fichier** : `src/nlp/transformer_ner.py`

Utilise le modèle `Jean-Baptiste/camembert-ner` (~400 Mo), un transformer RoBERTa à 110M de paramètres, fine-tuné pour la NER française. Même pipeline que spaCy : détecte des LOC puis utilise `assign_roles()`.

**Forces** : meilleure compréhension contextuelle (embeddings contextuels sur 12 couches d'attention).
**Faiblesses** : modèle le plus lourd (400 Mo + PyTorch), inférence la plus lente, GPU recommandé, même dépendance à l'heuristique `assign_roles()`.

---

## 4. Comparaison des coûts opérationnels

### 4.1 Ressources par requête

| Métrique | Rule-based | CRF | spaCy | CamemBERT |
|----------|-----------|-----|-------|-----------|
| **RAM** | ~5 Mo | ~15 Mo | ~120 Mo | ~800 Mo |
| **Latence / requête** | < 0.1 ms | ~0.1 ms | ~15 ms | ~80-200 ms |
| **GPU nécessaire** | Non | Non | Non | Recommandé |
| **Dépendances** | unidecode | pycrfsuite | spacy + modèle 50 Mo | transformers + torch + modèle 400 Mo |

### 4.2 Coût de déploiement (projection 10 000 req/jour)

| Métrique | Rule-based | CRF | spaCy | CamemBERT |
|----------|-----------|-----|-------|-----------|
| **Type de serveur** | VPS minimal | VPS minimal | VPS moyen | VPS avec GPU |
| **Coût mensuel estimé** | ~3 EUR | ~3 EUR | ~10 EUR | ~50-200 EUR |
| **Énergie / an** | ~0.1 kWh | ~0.4 kWh | ~18 kWh | ~180 kWh |

### 4.3 Coût d'entraînement (one-shot)

| Métrique | Rule-based | CRF | spaCy (fine-tuné) | CamemBERT (fine-tuné) |
|----------|-----------|-----|-------------------|----------------------|
| **Données nécessaires** | 0 | ~800 (auto-générées) | ~1 000-3 000 | ~3 000-10 000 |
| **Temps** | 0 | ~5 sec CPU | ~5-15 min CPU | ~30-60 min GPU |
| **Annotation manuelle** | Non | Non (templates) | Oui (ou semi-auto) | Oui (ou semi-auto) |

---

## 5. Pertinence par rapport au cas d'usage

### 5.1 Caractéristiques du domaine

Le Travel Order Resolver opère dans un **domaine fermé** avec les contraintes suivantes :

- **Gazetteer exhaustif** : toutes les gares SNCF sont connues (2785 stations)
- **Langue unique** : français
- **Deux entités seulement** : départ et destination
- **Formulations prévisibles** : requêtes de trajet ferroviaire

### 5.2 Classement par pertinence

**1. CRF - Le plus pertinent**

Le CRF occupe le meilleur compromis pour ce domaine :
- Prédit directement B-DEP / B-DEST (pas besoin d'heuristique)
- Ses features sont spécifiquement conçues pour le domaine (gazetteer, prépositions, verbes d'habitation)
- Entraînement automatique en 5 secondes à partir de templates
- Empreinte mémoire minimale (~15 Mo)

**2. Rule-based - Solide en baseline**

- Zéro dépendance ML, déterministe, maintenable
- Les 8 regex couvrent la majorité des formulations françaises de trajet
- Le scoring contextuel en fallback rattrape les cas non couverts
- Limitation : maintenance manuelle à chaque nouvelle formulation

**3. spaCy - Surdimensionné**

- 50 Mo de modèle NER généraliste pour un domaine où le gazetteer est exhaustif
- La capacité de spaCy à reconnaître des entités inconnues n'a pas de valeur ici : toutes les villes sont dans le gazetteer
- Dépend de `assign_roles()` pour distinguer départ/destination

**4. CamemBERT - Le moins pertinent**

- 400 Mo + PyTorch pour la même tâche que le CRF résout en 15 Mo
- Sa puissance de compréhension contextuelle profonde est sous-exploitée sur un domaine fermé
- Même dépendance à `assign_roles()` que spaCy
- Se justifierait dans un domaine ouvert, multi-langues, ou sans gazetteer

### 5.3 Pourquoi le gazetteer change tout

L'existence d'un gazetteer exhaustif est le facteur déterminant. Dans un domaine fermé :

- Les gros modèles NER (spaCy, CamemBERT) sont conçus pour **découvrir des entités inconnues** dans du texte libre. Cette capacité n'a aucune valeur quand toutes les villes sont déjà répertoriées.
- Le CRF et le rule-based exploitent directement le gazetteer comme feature (`word.in_gazetteer`) ou comme étape de résolution (`resolve_city()`), ce qui les rend plus efficaces sur cette tâche spécifique.

---

## 6. Impact du fine-tuning

### 6.1 Situation actuelle (sans fine-tuning spaCy/CamemBERT)

SpaCy et CamemBERT utilisent leurs modèles NER pré-entraînés tels quels. Ils ne détectent que des entités `LOC` génériques et dépendent de l'heuristique `assign_roles()` pour assigner les rôles départ/destination.

### 6.2 Avec fine-tuning

Un fine-tuning permettrait d'entraîner spaCy et CamemBERT à prédire directement `B-DEP` / `B-DEST` (comme le CRF), éliminant le besoin de l'heuristique.

| Aspect | CRF (actuel) | spaCy fine-tuné | CamemBERT fine-tuné |
|--------|-------------|-----------------|---------------------|
| Données nécessaires | ~800 (auto) | ~1 000-3 000 | ~3 000-10 000 |
| Temps de fine-tuning | 5 sec | 5-15 min | 30-60 min (GPU) |
| Annotation | Automatique (templates) | Manuelle ou semi-auto | Manuelle ou semi-auto |
| Taille du modèle | ~1 Mo | ~50 Mo | ~400 Mo |
| Qualité attendue | Bon sur formulations typiques | Meilleure généralisation | Excellente généralisation |

### 6.3 Le fine-tuning se justifie-t-il ?

Le coût d'entraînement est négligeable (one-shot). Le vrai coût est :
- L'effort d'annotation des données (ou la création de plus de templates)
- Le coût d'inférence multiplié par chaque requête (RAM, latence, énergie)
- La complexité de la stack de déploiement (PyTorch, GPU)

Pour un domaine fermé avec un gazetteer exhaustif, le gain de qualité du fine-tuning est **marginal** sur les cas typiques (~95% du trafic) et ne se manifeste que sur des formulations inhabituelles.

---

## 7. Conclusion et recommandation

### 7.1 Synthèse

Le choix de l'engine dépend du compromis entre complexité du modèle et contraintes du domaine :

```
Complexité du modèle
────────────────────────────────────────────────────────────

  Rule-based       CRF            spaCy          CamemBERT
     │               │               │               │
  Regex           Modèle          Réseau de       Transformer
  codées          linéaire        neurones        12 couches
  à la main       (features       (embeddings     (attention
                   manuelles,      pré-entraînés)  contextuelle,
                   poids appris)                    110M params)

  Pas de ML  ◄── ML classique ──── Deep learning ── Transformer ──►
```

Sur un domaine fermé (gazetteer exhaustif, langue unique, 2 entités), les modèles les plus simples sont les plus adaptés. La puissance des modèles complexes (généralisation à des entités inconnues, compréhension contextuelle profonde) est sous-exploitée.

### 7.2 Recommandation

| Scénario | Engine recommandé | Justification |
|----------|-------------------|---------------|
| **Production actuelle** | CRF | Meilleur ratio qualité/coût, prédiction directe dep/dest |
| **Fallback / baseline** | Rule-based | Zéro ML, déterministe, maintenable |
| **Évolution multi-entités** (dates, classes, correspondances) | spaCy fine-tuné | Bon compromis complexité/performance |
| **Évolution multi-langues / domaine ouvert** | CamemBERT fine-tuné | Justifié quand le domaine s'ouvre |

### 7.3 Perspective

Les 4 engines coexistent sous un Strategy Pattern, ce qui permet de basculer entre eux sans modifier le reste du pipeline. Cette architecture laisse la porte ouverte à l'évolution du produit tout en optimisant les ressources pour le cas d'usage actuel.

---

*Document rédigé le 05/03/2026*
