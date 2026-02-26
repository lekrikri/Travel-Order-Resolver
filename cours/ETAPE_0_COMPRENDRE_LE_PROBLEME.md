# ÉTAPE 0 : Comprendre le Problème

## 🎯 C'est quoi le projet ?

On veut créer un système qui comprend des phrases comme :
```
"Florence et moi voulons aller à Annecy depuis Paris"
```

Et qui extrait :
- **DÉPART** : Paris
- **ARRIVÉE** : Annecy

## 🤔 Pourquoi c'est difficile ?

### Problème 1 : L'ordre des mots change
```
"De Paris à Lyon"           → Départ=Paris, Arrivée=Lyon
"Direction Lyon depuis Paris" → Départ=Paris, Arrivée=Lyon (inversé!)
"Lyon, j'arrive de Paris"    → Départ=Paris, Arrivée=Lyon (encore inversé!)
```

### Problème 2 : Il y a du "bruit"
```
"Florence et moi voulons aller à Annecy depuis Paris"
         ↑
    Ce n'est PAS une ville, c'est un prénom !
```

### Problème 3 : Les fautes d'orthographe
```
"Je veux aller à Marsei" → Marseille
"Pari - Lyon"            → Paris - Lyon
```

## 🧠 La solution : NER (Named Entity Recognition)

**NER** = Reconnaissance d'Entités Nommées

C'est une technique de NLP qui identifie et classifie les "entités" dans un texte :
- **PER** : Personnes (Florence, Jean, Marie)
- **LOC** : Lieux (Paris, Lyon, France)
- **ORG** : Organisations (SNCF, Google)

### Exemple avec spaCy de base :
```python
import spacy
nlp = spacy.load("fr_core_news_md")

doc = nlp("Florence et moi voulons aller à Annecy depuis Paris")

for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")
```

**Résultat :**
```
Florence → PER (Personne - correct!)
Annecy → LOC (Lieu)
Paris → LOC (Lieu)
```

### Le problème de spaCy de base :
- Il détecte les **LOC** (lieux)
- Mais il ne sait pas lequel est le **DÉPART** et lequel est l'**ARRIVÉE** !

## 💡 Notre solution : NER Custom

On va **entraîner** spaCy à reconnaître nos propres labels :
- **DEPART** : La ville de départ
- **ARRIVEE** : La ville d'arrivée

Après entraînement :
```python
doc = nlp("Florence et moi voulons aller à Annecy depuis Paris")

for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")
```

**Résultat voulu :**
```
Annecy → ARRIVEE
Paris → DEPART
```

## 📊 Le pipeline complet

```
┌─────────────────────────────────────────────┐
│  "Je veux aller de Paris à Lyon"            │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  ÉTAPE 1: NER Custom (spaCy entraîné)       │
│  → Détecte: Paris=DEPART, Lyon=ARRIVEE      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  ÉTAPE 2: Fuzzy Matching                    │
│  → Corrige "Pari" en "Paris"                │
│  → Vérifie que la gare existe               │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  ÉTAPE 3: Pathfinding (Dijkstra)            │
│  → Calcule le chemin optimal                │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  RÉSULTAT: Paris → Dijon → Lyon             │
└─────────────────────────────────────────────┘
```

## ✅ Ce qu'on va faire

| Étape | Description | Fichier |
|-------|-------------|---------|
| 1 | Créer un dataset annoté | ETAPE_1_DATASET.md |
| 2 | Comprendre spaCy | ETAPE_2_SPACY_BASICS.md |
| 3 | Entraîner un NER custom | ETAPE_3_ENTRAINEMENT.md |
| 4 | Comparer les modèles | ETAPE_4_COMPARAISON.md |
| 5 | Ajouter le Fuzzy Matching | ETAPE_5_FUZZY.md |
| 6 | Évaluer avec des métriques | ETAPE_6_METRIQUES.md |
| 7 | Créer le CLI | ETAPE_7_CLI.md |

---

**Prochaine étape →** [ETAPE_1_DATASET.md](ETAPE_1_DATASET.md)
