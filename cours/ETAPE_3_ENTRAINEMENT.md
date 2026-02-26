# ÉTAPE 3 : Entraîner le NER Custom

## 🎯 Objectif

Entraîner spaCy à reconnaître nos labels **DEPART** et **ARRIVEE**.

## 📚 Comment ça marche ?

### Le concept

1. On montre au modèle des milliers d'exemples :
   - "Je veux aller de **Paris** à **Lyon**" → Paris=DEPART, Lyon=ARRIVEE

2. Le modèle apprend les **patterns** :
   - "de X à Y" → X=DEPART, Y=ARRIVEE
   - "depuis X vers Y" → X=DEPART, Y=ARRIVEE
   - etc.

3. Après entraînement, il peut prédire sur de nouvelles phrases !

### Fichier de configuration

spaCy utilise un fichier `config.cfg` qui définit :
- L'architecture du modèle
- Les paramètres d'entraînement
- Les chemins des données

## 🔧 Les étapes

1. **Créer la config** avec `spacy init config`
2. **Entraîner** avec `spacy train`
3. **Tester** le modèle entraîné

## ⚙️ Paramètres importants

| Paramètre | Description |
|-----------|-------------|
| `max_epochs` | Nombre de fois qu'on parcourt tout le dataset |
| `patience` | Arrêter si pas d'amélioration pendant X epochs |
| `learn_rate` | Vitesse d'apprentissage (trop haut = instable, trop bas = lent) |
| `batch_size` | Nombre d'exemples traités ensemble |

---

## ▶️ Commandes à exécuter

On va d'abord créer une config de base, puis l'entraîner.
