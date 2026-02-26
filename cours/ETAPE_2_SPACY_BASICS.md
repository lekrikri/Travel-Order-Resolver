# ÉTAPE 2 : Comprendre spaCy

## 🎯 Objectif

1. Comprendre comment fonctionne spaCy
2. Convertir notre dataset JSON au format `.spacy`

## 📚 C'est quoi spaCy ?

**spaCy** est une bibliothèque NLP (Natural Language Processing) en Python.

Elle fait plein de choses :
- **Tokenisation** : Découper le texte en mots
- **POS Tagging** : Identifier verbes, noms, adjectifs...
- **NER** : Reconnaître les entités (personnes, lieux, organisations)
- **Et plus...**

## 🔧 Installation

```bash
# Installer spaCy
pip install spacy

# Télécharger le modèle français (medium)
python3 -m spacy download fr_core_news_md
```

## 🧪 Test rapide de spaCy

```python
import spacy

# Charger le modèle français
nlp = spacy.load("fr_core_news_md")

# Analyser une phrase
doc = nlp("Florence et moi voulons aller à Annecy depuis Paris")

# Voir les entités détectées
print("Entités détectées:")
for ent in doc.ents:
    print(f"  {ent.text:15} → {ent.label_}")
```

**Résultat attendu :**
```
Entités détectées:
  Florence        → PER    (Personne - correct!)
  Annecy          → LOC    (Lieu)
  Paris           → LOC    (Lieu)
```

**Problème :** spaCy détecte les LOC mais ne sait pas lequel est DEPART/ARRIVEE !

## 🏗️ Le Pipeline spaCy

Quand tu fais `nlp(texte)`, spaCy exécute un **pipeline** :

```
Texte → [Tokenizer] → [Tagger] → [Parser] → [NER] → Doc
```

| Composant | Rôle |
|-----------|------|
| **Tokenizer** | Découpe en mots ("Je", "veux", "aller"...) |
| **Tagger** | Identifie les types (verbe, nom, préposition...) |
| **Parser** | Analyse la structure grammaticale |
| **NER** | Trouve les entités nommées |

**Notre but :** Remplacer/entraîner le composant **NER** avec nos labels DEPART/ARRIVEE.

## 📦 Format DocBin (.spacy)

Pour entraîner spaCy, on doit convertir notre JSON en format `.spacy` (binaire).

**Pourquoi ?**
- Plus rapide à charger
- Format natif de spaCy
- Optimisé pour l'entraînement

## 🔄 Conversion JSON → .spacy

Le script suivant va :
1. Lire nos fichiers JSON
2. Créer des objets `Doc` spaCy avec les annotations
3. Sauvegarder en `.spacy`

---

## ▶️ Commandes à exécuter

```bash
# 1. D'abord, active le venv si pas fait
source venv_gpu/bin/activate

# 2. Installe spaCy (si pas déjà fait)
pip install spacy

# 3. Télécharge le modèle français
python3 -m spacy download fr_core_news_md
```

Dis-moi quand c'est fait et je te crée le script de conversion !
