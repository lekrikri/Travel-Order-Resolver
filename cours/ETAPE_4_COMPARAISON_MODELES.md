# ÉTAPE 4 : Comparer les Modèles

## 🎯 Objectif

Le jury veut voir l'**impact des paramètres** sur les performances.

On va entraîner le NER avec différents modèles de base :

| Modèle | Description | Taille |
|--------|-------------|--------|
| `blank` | Aucun pré-entraînement (déjà fait ✅) | ~5 MB |
| `fr_core_news_sm` | Petit modèle pré-entraîné | ~15 MB |
| `fr_core_news_md` | Moyen modèle pré-entraîné | ~50 MB |
| `fr_core_news_lg` | Grand modèle pré-entraîné | ~500 MB |

## 📚 Pourquoi comparer ?

**Modèle blank :**
- Part de zéro
- Apprend uniquement sur notre dataset
- Peut être moins bon sur des cas nouveaux

**Modèles pré-entraînés (sm/md/lg) :**
- Déjà entraînés sur des millions de textes français
- Connaissent déjà la grammaire, les mots, etc.
- On "fine-tune" juste le NER

## 🤔 Hypothèse

> Plus le modèle de base est gros, meilleurs seront les résultats ?

On va vérifier ça avec des métriques !

---

## ▶️ Commandes à exécuter

On va entraîner les 3 autres modèles.
