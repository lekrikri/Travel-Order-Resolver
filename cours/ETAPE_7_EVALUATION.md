# ÉTAPE 7 : Évaluation et Métriques

## 🎯 Objectif

Comparer **tous les modèles NER** avec des métriques quantitatives sur le jeu de test.

## 📊 Métriques Utilisées

### Precision, Recall, F1-Score

| Métrique | Formule | Signification |
|----------|---------|---------------|
| **Precision** | TP / (TP + FP) | % de prédictions correctes |
| **Recall** | TP / (TP + FN) | % d'entités trouvées |
| **F1-Score** | 2 × (P × R) / (P + R) | Moyenne harmonique |

**Où :**
- **TP** (True Positive) : Entité correctement détectée
- **FP** (False Positive) : Entité détectée mais fausse
- **FN** (False Negative) : Entité manquée

### Exemple

```
Phrase : "Je veux aller de Paris à Lyon"
Attendu : DEPART=Paris, ARRIVEE=Lyon
Prédit  : DEPART=Paris, ARRIVEE=Lyon

→ TP = 2, FP = 0, FN = 0
→ Precision = 100%, Recall = 100%, F1 = 100%
```

## 🔬 Script d'Évaluation

**Fichier** : `src/evaluation/evaluate_all.py`

```python
# Exécuter l'évaluation
python3 src/evaluation/evaluate_all.py
```

## 📈 Résultats

### Tableau Comparatif

| Modèle | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| spaCy blank | 70.22% | 68.07% | 69.13% |
| spaCy sm | 74.25% | 77.10% | 75.65% |
| spaCy md | 84.08% | 88.90% | 86.42% |
| spaCy lg | 79.02% | 84.13% | 81.50% |
| **spaCy 10k** | **99.90%** | **99.87%** | **99.88%** |
| CamemBERT | 96.40% | 99.97% | 98.15% |

### Graphique

```
F1-Score :

spaCy 10k     ████████████████████████████████████████ 99.88%
CamemBERT     ██████████████████████████████████████░░ 98.15%
spaCy md      ██████████████████████████████████░░░░░░ 86.42%
spaCy lg      ████████████████████████████████░░░░░░░░ 81.50%
spaCy sm      ██████████████████████████████░░░░░░░░░░ 75.65%
spaCy blank   ████████████████████████████░░░░░░░░░░░░ 69.13%
```

## 🏆 Meilleur Modèle

**spaCy 10k** avec F1 = **99.88%**

- Seulement 3 faux positifs
- Seulement 4 faux négatifs
- Sur 1500 exemples de test

## 🤔 Analyse

### Pourquoi spaCy 10k > CamemBERT ?

| Critère | spaCy 10k | CamemBERT |
|---------|-----------|-----------|
| Entraînement | Sur notre dataset | Sur notre dataset |
| Architecture | Simple (CNN) | Complexe (Transformer) |
| Precision | 99.90% | 96.40% |
| Faux Positifs | 3 | 112 |

**Conclusion** : CamemBERT sur-détecte (trop de faux positifs).

### Pourquoi md > lg ?

Surprenant ! Le modèle **medium** (md) bat le **large** (lg).

Hypothèse : Le modèle lg a plus de paramètres mais n'a pas été fine-tuné assez longtemps sur notre tâche spécifique.

## ✅ Commandes

```bash
# Évaluer tous les modèles
cd /home/lekrikri/Projects/T-AIA-911-PAR_14
source venv_gpu/bin/activate
python3 src/evaluation/evaluate_all.py
```

## 📁 Fichiers Créés

| Fichier | Description |
|---------|-------------|
| `src/evaluation/evaluate_all.py` | Script d'évaluation |
| `results/evaluation_results.json` | Résultats JSON |

## ➡️ Étape Suivante

**ÉTAPE 8** : CLI de démonstration
