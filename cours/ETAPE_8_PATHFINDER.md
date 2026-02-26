# ÉTAPE 8 : Pathfinder (Calcul d'Itinéraires)

## 🎯 Objectif

Calculer l'itinéraire optimal entre deux gares SNCF.

## 🗺️ Le Graphe Ferroviaire

### Structure

```
Graphe = (Noeuds, Arêtes)

Noeuds = Gares SNCF (2785)
Arêtes = Connexions ferroviaires (8510)
```

### Exemple

```
Paris ──(2h)──> Lyon ──(1h30)──> Marseille
  │                │
  │              (45min)
  │                ▼
  └──(3h)──> Bordeaux ──(2h)──> Toulouse
```

## 🔍 Algorithmes

### 1. Dijkstra (classique)

**Principe** : Explorer tous les chemins possibles par ordre de coût croissant.

```python
# Pseudo-code Dijkstra
def dijkstra(graph, start, end):
    distances = {start: 0}
    visited = set()

    while end not in visited:
        # Prendre le noeud non visité avec la plus petite distance
        current = min(non_visited, key=distances.get)

        # Explorer ses voisins
        for neighbor in graph.neighbors(current):
            new_dist = distances[current] + graph.edge_weight(current, neighbor)
            if new_dist < distances.get(neighbor, inf):
                distances[neighbor] = new_dist

        visited.add(current)

    return distances[end]
```

**Complexité** : O((V + E) log V)

### 2. A* (optimisé)

**Principe** : Comme Dijkstra, mais avec une **heuristique** qui guide vers la destination.

```python
# f(n) = g(n) + h(n)
# g(n) = coût réel depuis le départ
# h(n) = estimation jusqu'à l'arrivée (heuristique)

def astar(graph, start, end):
    # Heuristique = distance à vol d'oiseau
    def heuristic(node):
        return haversine_distance(node.coords, end.coords)

    # Priorité = coût réel + estimation
    priority = g(node) + heuristic(node)
```

**Heuristique utilisée** : Distance Haversine (à vol d'oiseau)

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance entre 2 points GPS en km."""
    R = 6371  # Rayon Terre
    # ... formule sphérique
    return distance_km
```

## 📊 Comparaison

| Critère | Dijkstra | A* |
|---------|----------|-----|
| Optimal | ✅ Oui | ✅ Oui |
| Noeuds explorés | Beaucoup | Moins |
| Vitesse | Lent | **Rapide** |
| Heuristique | Non | Oui |

### Exemple : Paris → Marseille

| Algorithme | Noeuds explorés | Temps calcul |
|------------|-----------------|--------------|
| Dijkstra | ~800 | 50ms |
| A* | ~150 | **10ms** |

A* est **5x plus rapide** !

## 💻 Utilisation

### Code

```python
from pathfinder import TrainPathfinder

# Initialiser
pf = TrainPathfinder()
pf.initialize()

# Dijkstra
route = pf.find_route("Paris", "Lyon", metric="time", algorithm="dijkstra")

# A* (par défaut)
route = pf.find_route("Paris", "Lyon", metric="time", algorithm="astar")

# Optimiser par distance
route = pf.find_route("Paris", "Lyon", metric="distance")
```

### Métriques disponibles

| Métrique | Description |
|----------|-------------|
| `time` | Temps de trajet (minutes) |
| `distance` | Distance (km) |

## 📁 Fichiers

| Fichier | Description |
|---------|-------------|
| `src/3_pathfinder/pathfinder.py` | Classe principale |
| `src/3_pathfinder/railway_graph.py` | Graphe + Dijkstra + A* |
| `src/3_pathfinder/sncf_data_loader.py` | Chargement données SNCF |

## ✅ Commandes

```bash
# Tester le pathfinder
cd /home/lekrikri/Projects/T-AIA-911-PAR_14/src/3_pathfinder
source ../../venv_gpu/bin/activate
python3 pathfinder.py
```

## 🎁 Bonus : Critère Prix

Pour ajouter le prix comme critère :

1. Intégrer l'API SNCF des tarifs
2. Ajouter `price` comme poids des arêtes
3. Optimisation multi-critères (Pareto)

```python
# Futur
route = pf.find_route("Paris", "Lyon", metric="price")
```
