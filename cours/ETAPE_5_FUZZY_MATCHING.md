# ÉTAPE 5 : Fuzzy Matching

## 🎯 Objectif

Corriger les fautes d'orthographe pour matcher les vraies gares SNCF.

## 🤔 Le problème

Le NER extrait ce que l'utilisateur a écrit :
```
"Je veux aller de Marsei à Lyone"
                 ↓          ↓
            "Marsei"    "Lyone"
```

Mais ces noms n'existent pas dans la base SNCF !

## 💡 La solution : Fuzzy Matching

On compare le texte extrait avec la liste des vraies gares et on trouve la plus proche.

```
"Marsei" → Distance de 2 avec "Marseille" → Match !
"Lyone"  → Distance de 1 avec "Lyon" → Match !
```

## 📚 Comment ça marche ?

### Distance de Levenshtein

C'est le nombre d'opérations pour transformer un mot en un autre :
- Insertion
- Suppression
- Substitution

**Exemple :**
```
"Marsei" → "Marseille"
  1. Marsei  → Marseil  (ajout 'l')
  2. Marseil → Marseill (ajout 'l')
  3. Marseill→ Marseille(ajout 'e')
  = 3 opérations
```

### Score de similarité

On utilise un **ratio** de 0 à 100 :
- 100 = identique
- 0 = complètement différent

```
"Marseille" vs "Marseille" = 100
"Marsei" vs "Marseille"    = 80
"Paris" vs "Marseille"     = 30
```

## 🔧 Bibliothèque : rapidfuzz

On utilise `rapidfuzz` (plus rapide que `fuzzywuzzy`).

```python
from rapidfuzz import fuzz, process

# Score entre deux mots
score = fuzz.ratio("Marsei", "Marseille")  # ~80

# Trouver le meilleur match dans une liste
match = process.extractOne("Marsei", ["Paris", "Lyon", "Marseille"])
# ("Marseille", 80)
```

---

## ▶️ Commande à exécuter

Installe rapidfuzz :
```bash
pip install rapidfuzz
```
