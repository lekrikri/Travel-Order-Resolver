# T-AIA-911-PAR_14

## 🚂 Travel Order Resolver - NER + Fuzzy Matching + Pathfinding

Système complet d'extraction NER pour générer des itinéraires ferroviaires SNCF à partir de phrases en français naturel.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Résultats Finaux

| Modèle | Précision (F1) | Temps/phrase | Utilisation |
|--------|----------------|--------------|-------------|
| spaCy blank | 69.13% | ~0.005s | Baseline |
| spaCy sm | 75.65% | ~0.01s | Léger |
| spaCy md | 86.42% | ~0.02s | Moyen |
| spaCy lg | 81.50% | ~0.03s | Large |
| **spaCy 10k** | **99.88%** ✅ | ~0.01s | **Production** |
| CamemBERT | 98.15% | ~0.15s | Transformer |

**Approche retenue:** spaCy 10k (99.88% F1) + Fuzzy Matching + Pathfinder A*

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                 FRONTEND (React + TypeScript)          │
│  - Interface utilisateur moderne                       │
│  - Gestion d'erreurs complète                          │
│  - Carte interactive (Leaflet)                         │
│  - Mode recherche texte + vocale (Whisper)            │
└────────────────────────────────────────────────────────┘
                           │
                           ↓ HTTP/WebSocket
┌────────────────────────────────────────────────────────┐
│              API REST (FastAPI + Python)               │
│  - Endpoint /route : pipeline complet                  │
│  - Endpoint /extract : NER seule                       │
│  - Endpoint /speech-to-route : transcription audio     │
│  - Gestion d'erreurs structurée                        │
└────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ↓                                     ↓
┌──────────────────┐              ┌──────────────────────┐
│  spaCy NER 10k   │              │  Fuzzy Matcher SNCF  │
│  F1: 99.88%      │              │  RapidFuzz           │
│  Labels:         │              │  2,785 gares         │
│  - DEPART        │              │  Score: 0-100        │
│  - ARRIVEE       │              └──────────────────────┘
└──────────────────┘                         │
                                             ↓
                           ┌──────────────────────────────┐
                           │   Pathfinder (NetworkX)      │
                           │   - Dijkstra / A*            │
                           │   - Heuristique Haversine    │
                           │   - 2,785 gares              │
                           │   - 50 connexions            │
                           └──────────────────────────────┘
```

---

## 📂 Structure du Projet

```
T-AIA-911-PAR_14/
├── frontend/                      # 🎨 Interface React + TypeScript
│   ├── src/
│   │   ├── components/
│   │   │   ├── RouteSearch.tsx    # Recherche par texte
│   │   │   ├── RouteDisplay.tsx   # Affichage itinéraire
│   │   │   ├── RouteMap.tsx       # Carte Leaflet
│   │   │   ├── ErrorDisplay.tsx   # Gestion d'erreurs
│   │   │   └── AudioRecorder.tsx  # Recherche vocale
│   │   ├── services/api.ts        # Client API
│   │   ├── types/index.ts         # Types TypeScript
│   │   └── styles/App.css         # Styles
│   ├── package.json
│   └── vite.config.ts
│
├── src/                           # 🧠 Pipeline NER + Pathfinding
│   ├── 3_pathfinder/              # Pathfinding SNCF
│   │   ├── pathfinder.py          # Classe principale
│   │   ├── railway_graph.py       # Graphe NetworkX + A*
│   │   └── sncf_data_loader.py    # Chargement données
│   │
│   ├── inference/                 # NER + Fuzzy
│   │   ├── fuzzy_matcher.py       # RapidFuzz SNCF
│   │   └── pipeline.py            # Pipeline complet
│   │
│   ├── models/                    # Entraînement NER
│   │   ├── train_all_models.py    # spaCy sm/md/lg/10k
│   │   └── train_camembert.py     # CamemBERT transformers
│   │
│   ├── evaluation/                # Évaluation
│   │   └── evaluate_all.py        # Comparaison modèles
│   │
│   └── dataset/                   # Génération datasets
│       └── generate_dataset.py    # 10k phrases
│
├── data/
│   ├── sncf/                      # Données SNCF
│   │   ├── stations.csv           # 2,785 gares
│   │   ├── connections.csv        # 50 connexions
│   │   └── railway_lines.csv      # Lignes ferroviaires
│   │
│   └── annotated/                 # Datasets NER
│       └── train_sentences_10k.csv
│
├── cours/                         # 📚 Documentation pédagogique
│   ├── ETAPE_0_COMPRENDRE_LE_PROBLEME.md
│   ├── ETAPE_1_DATASET.md
│   ├── ETAPE_2_SPACY_BASICS.md
│   ├── ETAPE_3_ENTRAINEMENT.md
│   ├── ETAPE_4_COMPARAISON_MODELES.md
│   ├── ETAPE_5_FUZZY_MATCHING.md
│   ├── ETAPE_6_CAMEMBERT.md
│   ├── ETAPE_7_EVALUATION.md
│   └── ETAPE_8_PATHFINDER.md
│
├── rapport/
│   └── RAPPORT_EXPERIMENTATION.md  # Rapport complet
│
├── api_spacy.py                    # 🚀 API FastAPI principale
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Backend (API)

```bash
# Cloner le dépôt
git clone https://github.com/EpitechMscProPromo2026/T-AIA-911-PAR_14.git
cd T-AIA-911-PAR_14

# Créer environnement virtuel
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# Installer dépendances
pip install --upgrade pip
pip install fastapi uvicorn spacy pandas networkx rapidfuzz requests

# Charger le modèle spaCy entraîné
# Le modèle est dans models/ner_10k/model-best/
```

### 2. Frontend (React)

```bash
cd frontend
npm install
```

---

## 🚀 Démarrage

### Terminal 1 - API Backend

```bash
cd T-AIA-911-PAR_14
source venv_gpu/bin/activate
uvicorn api_spacy:app --reload --host 0.0.0.0 --port 8000
```

✅ API disponible sur : http://localhost:8000
📖 Documentation : http://localhost:8000/docs

### Terminal 2 - Frontend React

```bash
cd frontend
npm run dev
```

✅ Interface disponible sur : http://localhost:5173

---

## 📡 API Endpoints

### POST /route
Pipeline complet : NER + Fuzzy + Pathfinding

**Request:**
```json
{
  "sentence": "Je veux aller de Paris à Lyon",
  "metric": "time",
  "algorithm": "astar"
}
```

**Response:**
```json
{
  "sentence": "Je veux aller de Paris à Lyon",
  "departure": "Paris",
  "arrival": "Lyon",
  "departure_matched": "Paris Gare de Lyon",
  "arrival_matched": "Lyon Part-Dieu",
  "confidence": 0.98,
  "is_valid": true,
  "route": ["Paris Gare de Lyon", "Lyon Part-Dieu", "Lyon"],
  "route_count": 3,
  "total_distance": 462.5,
  "total_time": 120.0,
  "method": "spaCy NER + Fuzzy + ASTAR",
  "details": [...]
}
```

### POST /extract
Extraction NER seule (sans pathfinding)

### GET /health
Statut de l'API et modèles chargés

### GET /stats
Statistiques SNCF (gares, connexions)

---

## 🎯 Exemples de Phrases

```
✅ "Je veux aller de Paris à Lyon"
   → Départ: Paris, Arrivée: Lyon
   → Route: Paris Gare de Lyon → Lyon Part-Dieu → Lyon

✅ "Comment me rendre à Bordeaux depuis Toulouse ?"
   → Départ: Toulouse, Arrivée: Bordeaux
   → Route: Toulouse → Bordeaux St-Jean

✅ "Avec mon ami Albert, je veux aller de Nice à Monaco"
   → Départ: Nice, Arrivée: Monaco (et non "Albert" !)
   → Route: Nice-Ville → Monaco

❌ "Bonjour comment allez-vous ?"
   → Warning: Aucune ville de départ ou d'arrivée détectée
```

---

## 🛠️ Fonctionnalités

### Backend
- ✅ NER spaCy 10k (F1: 99.88%)
- ✅ Fuzzy matching RapidFuzz (2,785 gares)
- ✅ Pathfinding A* avec heuristique Haversine
- ✅ Validation complète des entrées
- ✅ Gestion d'erreurs structurée
- ✅ Documentation API interactive (Swagger)
- ✅ Transcription audio (Whisper)

### Frontend
- ✅ Interface moderne React + TypeScript
- ✅ Recherche texte + vocale
- ✅ Carte interactive Leaflet
- ✅ Affichage itinéraire détaillé
- ✅ Gestion d'erreurs élégante
- ✅ Mode sombre/clair
- ✅ Responsive design

---

## 📊 Comparaison des Modèles

### Résultats d'Évaluation (Test Set)

```
┌─────────────┬───────────┬────────┬────────┬─────────┐
│   Modèle    │ Précision │ Rappel │   F1   │  Temps  │
├─────────────┼───────────┼────────┼────────┼─────────┤
│ spaCy blank │   68.91%  │ 69.35% │ 69.13% │  5ms    │
│ spaCy sm    │   76.34%  │ 74.97% │ 75.65% │ 10ms    │
│ spaCy md    │   86.89%  │ 85.96% │ 86.42% │ 20ms    │
│ spaCy lg    │   82.12%  │ 80.90% │ 81.50% │ 30ms    │
│ spaCy 10k   │   99.88%  │ 99.88% │ 99.88% │ 10ms ⭐  │
│ CamemBERT   │   98.24%  │ 98.06% │ 98.15% │ 150ms   │
└─────────────┴───────────┴────────┴────────┴─────────┘
```

**Choix final:** spaCy 10k - meilleure précision ET vitesse

---

## 🧪 Tests

```bash
# Évaluer tous les modèles
python src/evaluation/evaluate_all.py

# Tester le pathfinder
python test_pathfinder.py

# Tester l'API
python test_api.py
```

---

## 📚 Documentation

- **[cours/](cours/)** : Tutoriels pas-à-pas (8 étapes)
- **[rapport/RAPPORT_EXPERIMENTATION.md](rapport/RAPPORT_EXPERIMENTATION.md)** : Rapport technique complet
- **[API Docs](http://localhost:8000/docs)** : Documentation interactive Swagger

---

## 🎓 Technologies

### Backend
- **FastAPI** - API REST moderne
- **spaCy 3.7+** - NER + NLP
- **RapidFuzz** - Fuzzy string matching
- **NetworkX** - Graphes + Pathfinding
- **Uvicorn** - Serveur ASGI
- **Whisper** - Speech-to-text (optionnel)

### Frontend
- **React 18** - Framework UI
- **TypeScript** - Typage statique
- **Vite** - Build tool
- **Leaflet** - Cartes interactives
- **Axios** - Client HTTP
- **Lucide React** - Icônes

### Data
- **SNCF Open Data** - 2,785 gares françaises
- **Dataset NER 10k** - 10,000 phrases annotées

---

## 🏆 Points Clés

### Innovations
1. **Modèle spaCy 10k** entraîné sur mesure (99.88% F1)
2. **Fuzzy matching** intelligent pour normalisation SNCF
3. **Pathfinding A*** avec heuristique géographique (Haversine)
4. **Gestion d'erreurs complète** API + Frontend
5. **Interface moderne** avec carte interactive

### Apprentissages
- **NER custom** : entraînement sur domaine spécifique crucial
- **Fuzzy matching** : essentiel pour variations de noms de gares
- **A* vs Dijkstra** : gain de performance avec heuristique GPS
- **Architecture 3-tiers** : séparation Frontend/API/Data
- **Gestion erreurs** : validation multi-niveaux pour robustesse

---

## 👥 Équipe

**T-AIA-911-PAR_14** - Epitech MSc Pro Promo 2026

---

## 📝 Licence

MIT License - Projet académique Epitech

---

## 🔗 Liens Utiles

- [Dépôt GitHub](https://github.com/EpitechMscProPromo2026/T-AIA-911-PAR_14)
- [spaCy Documentation](https://spacy.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [SNCF Open Data](https://www.sncf.com/fr/groupe/open-data)
- [NetworkX Documentation](https://networkx.org)

---

🚂 **99.88% F1-Score - Production Ready!**
