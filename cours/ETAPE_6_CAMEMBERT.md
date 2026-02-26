# ÉTAPE 6 : Modèle CamemBERT

## 🎯 Objectif

Entraîner un modèle **transformer** (CamemBERT) pour le NER.
C'est ce que le jury demande !

## 📚 C'est quoi CamemBERT ?

**CamemBERT** = BERT entraîné sur du texte français.

- Basé sur l'architecture **RoBERTa**
- Entraîné sur 138GB de texte français
- Comprend mieux le contexte que spaCy classique

## 🔥 Pourquoi c'est mieux ?

| spaCy classique | CamemBERT |
|-----------------|-----------|
| Regarde les mots un par un | Regarde toute la phrase |
| Embedding statique | Embedding contextuel |
| Rapide (CPU) | Plus lent mais précis (GPU) |

## 🏗️ Architecture

```
Phrase: "Je veux aller de Paris à Lyon"
           ↓
    [CamemBERT Encoder]  ← Transformers (attention)
           ↓
    [Couche de classification]
           ↓
    O  O   O    O  B-DEP  O  B-ARR
```

**Labels BIO :**
- **B-DEPART** : Début d'un départ
- **I-DEPART** : Suite d'un départ (pour les noms composés)
- **B-ARRIVEE** : Début d'une arrivée
- **I-ARRIVEE** : Suite d'une arrivée
- **O** : Autre (pas une entité)
