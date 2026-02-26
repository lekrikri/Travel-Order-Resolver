# 🚂 Module Pathfinder - Calcul d'Itinéraires Ferroviaires

Module de calcul d'itinéraires ferroviaires basé sur les données SNCF Open Data et l'algorithme de Dijkstra.

## 📋 Fonctionnalités

- ✅ Téléchargement automatique du référentiel des gares SNCF
- ✅ Génération d'un graphe ferroviaire avec connexions
- ✅ Calcul du chemin le plus court (algorithme de Dijkstra)
- ✅ Optimisation par temps de trajet ou distance
- ✅ Format de sortie standardisé pour intégration
- ✅ Cache local pour performances optimales

## 🏗️ Architecture

```
src/3_pathfinder/
├── sncf_data_loader.py    # Téléchargement données SNCF Open Data
├── railway_graph.py        # Construction du graphe ferroviaire
└── pathfinder.py           # Calcul d'itinéraires (Dijkstra)

data/sncf/                  # Cache des données SNCF
├── stations.csv            # Référentiel des gares (~3000 gares)
└── connections.csv         # Connexions ferroviaires générées

complete_pipeline.py        # Pipeline complet NLP + Pathfinder
test_pathfinder.py          # Tests du module
```

## 🚀 Utilisation

### Test rapide du Pathfinder seul

```bash
# Activer l'environnement
source venv_gpu/bin/activate

# Tester le pathfinder
python test_pathfinder.py
```

**Ce que fait le test:**
1. Télécharge le référentiel des gares SNCF (~3000 gares)
2. Génère les connexions ferroviaires basées sur la proximité géographique
3. Construit un graphe avec NetworkX
4. Teste 5 routes (Paris→Lyon, Marseille→Bordeaux, etc.)
5. Affiche les itinéraires avec format de sortie

**Sortie attendue:**
```
Paris → Lyon
   ✅ 2 gares, 0 changement(s)
   ⏱️  Temps estimé: 120 minutes
   🛤️  Itinéraire: Paris Gare de Lyon → Lyon Part Dieu
   📄 Format: 1,Paris Gare de Lyon,Lyon Part Dieu
```

### Pipeline complet (NLP + Pathfinder)

```bash
# Tester le pipeline complet
python complete_pipeline.py
```

**Ce que fait le pipeline:**
1. Charge Qwen2.5 + ChromaDB RAG (extraction de villes)
2. Initialise le Pathfinder SNCF
3. Traite 5 phrases de test:
   - Extraction NLP: "Paris à Lyon" → départ=Paris, arrivée=Lyon
   - Pathfinding: Paris → Lyon → itinéraire complet
4. Affiche les résultats pour chaque phrase

**Sortie attendue pour chaque phrase:**
```
✅ EXTRACTION NLP:
   Départ: Paris
   Arrivée: Lyon
   Confiance: 95%

✅ ITINÉRAIRE:
   Gares: 2
   Changements: 0
   Temps: 120 min

🛤️  CHEMIN:
   🚩 Paris Gare de Lyon
   🏁 Lyon Part Dieu

📄 FORMAT SORTIE:
   1,Paris Gare de Lyon,Lyon Part Dieu
```

## 📊 Données SNCF Open Data

### Référentiel des gares

**Source:** SNCF Open Data - `referentiel-gares-voyageurs`
**API:** `https://ressources.data.sncf.com/api/records/1.0/search/`

**Champs extraits:**
- `code_uic` - Code UIC unique de la gare
- `nom_gare` - Nom de la gare
- `commune` - Commune
- `departement` - Numéro de département
- `latitude` / `longitude` - Coordonnées GPS
- `is_suggestable` - Gare principale (utilisée pour les connexions)

**Nombre de gares:** ~3000 (dont ~500 gares principales)

### Connexions ferroviaires

**Source:** Générées automatiquement à partir des gares

**Méthode:**
- Pour chaque gare principale, connexion aux 5 voisines les plus proches
- Distance calculée avec coordonnées GPS (approximation)
- Temps estimé: 100 km/h en moyenne

**Nombre de connexions:** ~2500

**Format:**
- `from_uic` - Code UIC de départ
- `to_uic` - Code UIC d'arrivée
- `distance_km` - Distance en km
- `time_minutes` - Temps estimé en minutes

## 🔧 Algorithme de Pathfinding

### Dijkstra

**Implémentation:** NetworkX `nx.shortest_path()`

**Poids des arêtes:**
- Métrique `time`: Temps de trajet en minutes (par défaut)
- Métrique `distance`: Distance en km

**Complexité:** O((V + E) log V) avec heap binaire

### Exemple de graphe

```
Paris ─120min─ Lyon ─90min─ Marseille
  │              │
 180min        150min
  │              │
Lille ─240min─ Nice
```

**Chemin le plus court Paris → Nice:**
- Par temps: Paris → Lyon → Nice (270 min)
- Par distance: Direct si connexion existe

## 📄 Formats de données

### Format d'entrée (NLP)

```csv
sentenceID,sentence
1,Je voudrais un billet de train de Paris à Lyon
2,Besoin d'un trajet Marseille Bordeaux
```

### Format intermédiaire (NLP → Pathfinder)

```csv
sentenceID,Departure,Destination
1,Paris,Lyon
2,Marseille,Bordeaux
```

### Format de sortie final

```csv
sentenceID,Departure,Step1,Step2,...,Destination
1,Paris Gare de Lyon,Lyon Part Dieu
2,Marseille Saint-Charles,Narbonne,Toulouse Matabiau,Bordeaux Saint-Jean
3,INVALID
```

**Règles:**
- Si extraction NLP échoue → `INVALID`
- Si aucune gare trouvée → `INVALID`
- Si aucun itinéraire trouvé → `INVALID`

## 💻 Utilisation programmatique

### Pathfinder seul

```python
from src.3_pathfinder.pathfinder import TrainPathfinder

# Initialiser
pathfinder = TrainPathfinder()
pathfinder.initialize()

# Calculer un itinéraire
route = pathfinder.find_route("Paris", "Lyon", metric="time")

if "error" in route:
    print(f"Erreur: {route['error']}")
else:
    print(f"Itinéraire: {' → '.join(route['stations'])}")
    print(f"Temps: {route['total_time_minutes']} min")
    print(f"Changements: {route['num_changes']}")

# Format de sortie
output = pathfinder.format_route_for_output(1, route)
print(output)  # "1,Paris Gare de Lyon,Lyon Part Dieu"
```

### Pipeline complet

```python
from complete_pipeline import TravelOrderPipeline

# Initialiser le pipeline
pipeline = TravelOrderPipeline(use_rag=True)

# Traiter une phrase
result = pipeline.process_sentence(
    sentence_id=1,
    sentence="Je voudrais un billet de train de Paris à Lyon"
)

print(result["final_output"])
# "1,Paris Gare de Lyon,Lyon Part Dieu"

# Traiter un fichier complet
pipeline.process_file(
    input_file="./data/test_sentences.csv",
    output_file="./results/routes_output.csv"
)
```

### Traitement par batch

```python
# Préparer fichier d'entrée (NLP déjà fait)
# Format: sentenceID,Departure,Destination

pathfinder = TrainPathfinder()
pathfinder.initialize()

pathfinder.process_batch(
    input_file="./results/nlp_output.csv",
    output_file="./results/routes_output.csv",
    metric="time"
)
```

## 📊 Performances

| Métrique | Valeur |
|----------|--------|
| Temps d'initialisation | ~5-10s (première fois) |
| Temps d'initialisation (cache) | ~1s |
| Temps par requête | ~10-50ms |
| Nombre de gares | ~3000 |
| Nombre de connexions | ~2500 |
| Taille cache | ~500 KB |

## 🔍 Dépannage

### Les données SNCF ne se téléchargent pas

```bash
# Vérifier la connexion à l'API SNCF
curl "https://ressources.data.sncf.com/api/records/1.0/search/?dataset=referentiel-gares-voyageurs&rows=1"
```

**Solutions:**
- Vérifier la connexion Internet
- Vérifier que l'API SNCF est accessible
- Forcer le refresh: `pathfinder.initialize(force_refresh=True)`

### Erreur "Gare non trouvée"

La recherche de gare est fuzzy mais limitée. Essayez:
- Nom complet de la ville: "Marseille" au lieu de "Mrslle"
- Nom de la gare: "Paris Gare de Lyon" au lieu de "Paris"

### Erreur "Aucun itinéraire trouvé"

Le graphe peut avoir des composantes non connectées. Vérifiez:
```python
pathfinder.graph.stats()
# Si "Composantes connexes: X" > 1, le graphe n'est pas entièrement connecté
```

### Dépendances manquantes

```bash
pip install networkx scipy pandas requests
```

## 🧪 Tests

### Test unitaire du SNCF Data Loader

```bash
python src/3_pathfinder/sncf_data_loader.py
```

### Test unitaire du Railway Graph

```bash
python src/3_pathfinder/railway_graph.py
```

### Test d'intégration complet

```bash
python test_pathfinder.py
```

### Test du pipeline NLP + Pathfinder

```bash
python complete_pipeline.py
```

## 🎯 Améliorations possibles

### Court terme
- ✅ Téléchargement données SNCF
- ✅ Graphe basique avec proximité
- ✅ Algorithme de Dijkstra
- ⬜ Cache intelligent avec expiration
- ⬜ Logging détaillé

### Moyen terme
- ⬜ Utiliser les horaires SNCF réels (API Transilien)
- ⬜ Intégrer les lignes ferroviaires (TGV, TER, etc.)
- ⬜ Prendre en compte les correspondances
- ⬜ A* au lieu de Dijkstra (heuristique distance)

### Long terme
- ⬜ Calcul multi-critères (temps, prix, confort)
- ⬜ Gestion des perturbations en temps réel
- ⬜ Interface web interactive
- ⬜ Optimisation GPU pour calculs massifs

## 📚 Références

- [SNCF Open Data](https://ressources.data.sncf.com/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Dijkstra Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Référentiel des gares SNCF](https://ressources.data.sncf.com/explore/dataset/referentiel-gares-voyageurs/)

## 📝 License

Projet académique - T-AIA-911-PAR_14 - Epitech MSc Pro 2026

---

🚂 **Ready to explore the French railway network!**
