---
title: Travel Order Resolver API
emoji: 🚄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Travel Order Resolver

Extraction d'entités NER (départ/arrivée) depuis des phrases en français naturel, puis calcul d'itinéraire ferroviaire optimal sur le réseau SNCF réel.

**Stack :** Python 3.12 · FastAPI · spaCy · CamemBERT · CRF · React + TypeScript · Whisper

---

## Aperçu

```
"Je veux aller de Bordeaux à Nice"
          ↓  8 moteurs NLP comparés
  DEPART: Bordeaux  |  ARRIVEE: Nice
          ↓  3 algorithmes × 3 critères
  Bordeaux → Lyon Part Dieu → Avignon TGV → ... → Nice  (400 min)
```

**8 moteurs NLP :** rule-based · spaCy blank/sm/md/lg/10k · CRF-NER · CamemBERT
**3 algorithmes :** Dijkstra · BFS · A* (heuristique Haversine GPS)
**3 critères :** temps · pollution CO₂ · correspondances
**Benchmark en temps réel** via streaming SSE avec métriques énergie/mémoire/CPU
**Reconnaissance vocale** via Whisper (STT → NER → itinéraire)
**14 140 connexions SNCF** issues des données GTFS open data

---

## Performances (dataset hard_v2 — SMS, argot, typos, 2602 tests)

| Modèle | Precision | Recall | F1 | Exact Match |
|---|---|---|---|---|
| rule-based | 30.66% | 29.94% | 30.30% | 17.10% |
| spaCy blank | 74.81% | 63.11% | 68.46% | 47.02% |
| spaCy sm | 80.89% | 73.19% | 76.84% | 62.78% |
| spaCy lg | 83.60% | 77.84% | 80.62% | 70.40% |
| spaCy md | 85.55% | 82.25% | 83.87% | 74.21% |
| spaCy 10k | 91.07% | 89.74% | 90.40% | 84.17% |
| CamemBERT | 97.05% | 88.93% | 92.82% | 87.79% |
| **CRF-NER** | **99.79%** | **88.62%** | **93.87%** | **89.20%** |

---

## Prérequis

- Python 3.12+
- Node.js 18+ et npm
- GPU recommandé pour CamemBERT (optionnel, fonctionne en CPU)
- ~8 Go de disque pour tous les modèles

---

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/lekrikri/Travel-Order-Resolver.git
cd Travel-Order-Resolver
```

### 2. Installer les dépendances Python

```bash
# Avec uv (recommandé — plus rapide que pip)
pip install uv
uv sync

# OU avec pip classique
pip install fastapi "uvicorn[standard]" spacy sklearn-crfsuite transformers torch \
            unidecode sentencepiece protobuf python-multipart pandas
```

Installer les modèles spaCy de base (nécessaires pour les variantes sm/md/lg) :

```bash
uv run python -m spacy download fr_core_news_sm
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download fr_core_news_lg
```

### 3. Installer les dépendances frontend

```bash
cd frontend && npm install && cd ..
```

---

## Entraîner les modèles

> **Les modèles ne sont pas inclus dans le repo** (fichiers binaires trop lourds).
> Les données d'entraînement sont incluses dans `data/processed/` et `data/hard_v2/`.

### CRF-NER — entraînement automatique (~30 secondes)

Le CRF se réentraîne automatiquement au 1er appel. Pour le pré-charger :

```bash
uv run python -c "
from src.nlp.crf_ner import CRFNEREngine
e = CRFNEREngine()
e.extract('Je veux aller de Paris à Lyon')
print('CRF entraîné.')
"
```

### spaCy sm / md / lg (~5-10 min)

Fine-tune les 3 variantes de `fr_core_news_*` sur le dataset NER :

```bash
uv run python src/models/train_all_models.py
# → models/ner_sm/   models/ner_md/   models/ner_lg/
```

### spaCy blank (~3 min)

Entraîne un modèle spaCy vide from scratch sur les données standard :

```bash
uv run python -c "
import spacy, json, random
from spacy.tokens import DocBin
from spacy.training import Example
from pathlib import Path

with open('data/processed/train.json') as f:
    data = json.load(f)

nlp = spacy.blank('fr')
nlp.add_pipe('ner')
ner = nlp.get_pipe('ner')
ner.add_label('DEPART')
ner.add_label('ARRIVEE')

examples = []
for item in data:
    doc = nlp.make_doc(item['text'])
    ents = [(e['start'], e['end'], e['label']) for e in item.get('entities', [])]
    examples.append(Example.from_dict(doc, {'entities': ents}))

nlp.initialize(lambda: examples)
for epoch in range(30):
    random.shuffle(examples)
    losses = {}
    for batch in spacy.util.minibatch(examples, size=128):
        nlp.update(batch, losses=losses, drop=0.35)

Path('models/ner_blank/model-best').mkdir(parents=True, exist_ok=True)
nlp.to_disk('models/ner_blank/model-best')
print('Sauvegardé dans models/ner_blank/model-best')
"
```

### spaCy 10k — meilleur modèle spaCy (~5 min)

Entraîne sur le dataset hard_v2 (SMS, argot, fautes de frappe) :

```bash
uv run python -c "
import spacy, json, random
from spacy.training import Example
from pathlib import Path

with open('data/hard_v2/ml_10k.json') as f:
    data = json.load(f)

nlp = spacy.blank('fr')
nlp.add_pipe('ner')
ner = nlp.get_pipe('ner')
ner.add_label('DEPART')
ner.add_label('ARRIVEE')

examples = []
for item in data:
    doc = nlp.make_doc(item['text'])
    ents = [(e['start'], e['end'], e['label']) for e in item.get('entities', [])]
    examples.append(Example.from_dict(doc, {'entities': ents}))

nlp.initialize(lambda: examples)
for epoch in range(30):
    random.shuffle(examples)
    losses = {}
    for batch in spacy.util.minibatch(examples, size=128):
        nlp.update(batch, losses=losses, drop=0.35)
    if epoch % 5 == 0:
        print(f'Epoch {epoch}: {losses}')

Path('models/ner_10k/model-best').mkdir(parents=True, exist_ok=True)
nlp.to_disk('models/ner_10k/model-best')
print('Sauvegardé dans models/ner_10k/model-best')
"
```

### CamemBERT (~30 min GPU / ~3h CPU)

Fine-tune `camembert-base` depuis HuggingFace (téléchargé automatiquement) :

```bash
uv run python src/models/train_camembert.py
# → models/ner_camembert/best/
```

**Configuration** dans `src/models/train_camembert.py` :
- Modèle de base : `camembert-base` (~400 Mo, téléchargé automatiquement)
- 5 epochs, batch 16, LR 2e-5, max_length 128

---

## Lancer l'application

### Démarrage rapide (backend + frontend)

```bash
bash start.sh
# Backend  → http://localhost:8000
# Frontend → http://localhost:5173
```

### Séparé

```bash
# Terminal 1 — Backend
uv run uvicorn src.server.app:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

---

## Utilisation

### Interface web — http://localhost:5173

**Recherche par texte :**
```
"Je veux aller de Paris à Lyon"
"Un billet Marseille direction Bordeaux"
"Emmene moi de Lille a Strasbourg stp"
```

**Recherche vocale :** enregistrer ou importer un `.wav`/`.mp3` → Whisper transcrit → itinéraire calculé

**Benchmark :** comparer les 8 moteurs NLP et 3 algorithmes de graphe sur une même phrase avec métriques temps/énergie/mémoire en temps réel

### API REST — http://localhost:8000

```bash
# Résoudre un itinéraire
curl -X POST http://localhost:8000/api/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Je veux aller de Bordeaux à Nice",
    "nlp_engine": "crf-ner",
    "graph_engine": "dijkstra",
    "optimization": "time"
  }'

# Lister tous les moteurs disponibles
curl http://localhost:8000/api/engines

# Statut du serveur (nb gares, connexions, GPU...)
curl http://localhost:8000/api/health

# Documentation interactive
open http://localhost:8000/docs
```

**Moteurs NLP :** `rule-based` · `spacy-blank` · `spacy-sm` · `spacy-md` · `spacy-lg` · `spacy-ner` · `crf-ner` · `transformer-ner`

**Moteurs graph :** `dijkstra` · `bfs` · `astar`

**Optimisations :** `time` · `pollution` · `hops`

---

## Structure du projet

```
.
├── src/
│   ├── nlp/              # Moteurs NLP (Strategy Pattern)
│   │   ├── rule_based.py      # Règles regex + scoring contextuel
│   │   ├── spacy_ner.py       # spaCy générique (charge n'importe quel modèle)
│   │   ├── crf_ner.py         # CRF avec features manuelles (sklearn-crfsuite)
│   │   └── transformer_ner.py # CamemBERT fine-tuné (HuggingFace)
│   ├── graph/            # Algorithmes pathfinding (Strategy Pattern)
│   │   ├── dijkstra.py        # Plus court chemin (poids configurables)
│   │   ├── bfs.py             # BFS (minimise correspondances)
│   │   ├── astar.py           # A* + heuristique Haversine GPS
│   │   ├── hub.py             # Système hub intra-ville (Paris multi-gares)
│   │   └── base.py            # Interface abstraite + gazetteer
│   ├── server/app.py     # API FastAPI (endpoints + streaming SSE)
│   ├── dataset/          # Générateurs de datasets NER
│   └── models/           # Scripts d'entraînement
│       ├── train_all_models.py   # spaCy sm/md/lg
│       └── train_camembert.py    # CamemBERT fine-tuning
│
├── frontend/             # React + TypeScript (Vite)
│   └── src/
│       ├── components/
│       │   ├── RouteSearch.tsx    # Recherche par texte
│       │   ├── BenchmarkPanel.tsx # Benchmark temps réel SSE
│       │   ├── AudioRecorder.tsx  # Interface vocale Whisper
│       │   ├── RouteDisplay.tsx   # Affichage itinéraire + billets
│       │   └── RouteMap.tsx       # Carte Leaflet interactive
│       └── services/api.ts        # Client API + benchmark stream
│
├── data/
│   ├── processed/             # INCLUS dans le repo
│   │   ├── connections.json   # 14 140 connexions SNCF (GTFS + manuelles)
│   │   ├── gazetteer.json     # 3 491 gares avec aliases + fuzzy matching
│   │   └── stations.json      # Coordonnées GPS des gares
│   ├── hard_v2/               # Dataset SMS/argot 10k phrases (INCLUS)
│   └── sncf/                  # Données lignes ferroviaires sources
│
├── models/               # NON inclus (à entraîner — voir ci-dessus)
│   ├── ner_blank/        # spaCy vide from scratch
│   ├── ner_sm/           # fr_core_news_sm fine-tuné
│   ├── ner_md/           # fr_core_news_md fine-tuné
│   ├── ner_lg/           # fr_core_news_lg fine-tuné
│   ├── ner_10k/          # spaCy blank entraîné sur hard_v2 (meilleur)
│   └── ner_camembert/    # CamemBERT fine-tuné
│
├── rapport/
│   ├── rapport.typ       # Rapport Typst (compiler avec `typst compile rapport.typ`)
│   └── rapport.pdf       # PDF pré-compilé
│
├── results/plots/        # Graphiques comparatifs des modèles
├── pyproject.toml        # Dépendances Python (uv/pip)
└── start.sh              # Démarrage backend + frontend
```

---

## Notes techniques

- **Données SNCF** : `connections.json` contient les données GTFS SNCF open data enrichies de liaisons transversales manquantes (Bordeaux ↔ Lyon Part Dieu, etc.)
- **Fuzzy matching** : le gazetteer couvre 3 491 gares avec résolution des aliases, accents et fautes de frappe
- **Architecture Strategy Pattern** : tous les moteurs NLP et Graph sont interchangeables à l'exécution via les endpoints `nlp_engine` / `graph_engine`
- **Whisper** : chargé une seule fois en mémoire (singleton async) — la 1ère requête prend ~45s, les suivantes ~1-3s
- **CRF-NER** : entraîné à la volée au premier appel (~30s), stocké dans `/tmp/`
- **Paris hub** : les 6 gares parisiennes sont reliées par des arcs de transfert (30 min) pour permettre les connexions inter-gares
