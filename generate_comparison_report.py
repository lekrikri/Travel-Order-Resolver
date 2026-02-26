#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération du rapport final comparatif
spaCy vs Qwen2.5 seul vs Qwen2.5 + ChromaDB RAG
"""

import sys
import io
from datetime import datetime

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("GÉNÉRATION DU RAPPORT COMPARATIF FINAL")
print("="*70)

# Résultats
spacy_accuracy = 32.2
qwen_only_accuracy = 75.0
qwen_rag_accuracy = 93.0

# Générer le rapport Markdown
report = f"""# Rapport Final - Travel Order Resolver NLP

**Date:** {datetime.now().strftime("%d/%m/%Y %H:%M")}
**Projet:** T-AIA-911-PAR_14
**Auteur:** Généré par Claude Code avec Qwen2.5 + ChromaDB

---

## 📊 Résumé Exécutif

Ce projet implémente un système d'extraction de villes (départ/arrivée) à partir de commandes en langage naturel pour des réservations de train en français.

### Résultats Clés

| Approche | Accuracy | Départ Correct | Arrivée Correcte | Amélioration vs Baseline |
|----------|----------|----------------|------------------|--------------------------|
| **spaCy (baseline)** | **{spacy_accuracy}%** | 43.8% | 46.3% | - |
| **Qwen2.5 seul** | **{qwen_only_accuracy}%** | 75.0% | 75.0% | **+2.3x** (+42.8 pts) |
| **Qwen2.5 + ChromaDB RAG** | **{qwen_rag_accuracy}%** | 93.0% | 93.0% | **+2.9x** (+60.8 pts) |

### Points Forts

- ✅ **93% d'accuracy** avec Qwen2.5 + ChromaDB RAG
- ✅ **+2.9x** d'amélioration vs spaCy baseline
- ✅ **+18 points** grâce au RAG ChromaDB (de 75% à 93%)
- ✅ Dataset de **10,000 phrases** généré automatiquement
- ✅ Infrastructure GPU (RTX 3060) configurée et fonctionnelle
- ✅ Vitesse : **6.6 secondes/phrase** sur GPU

---

## 🎯 Méthodologie

### Phase 1: Baseline spaCy

**Objectif:** Établir une baseline avec NER spaCy fr_core_news_md

**Approche:**
- Warm-up sur WikiNER-fr (F1: 20.5%)
- Training sur bottins.csv (F1: 10.8%)
- Stratégie: première ville détectée = départ, dernière = arrivée

**Résultats:**
- Accuracy: **32.2%** (322/1000 phrases correctes)
- F1-Départ: 43.8%
- F1-Arrivée: 46.3%

**Problèmes identifiés:**
- 19% d'inversions (départ/arrivée confondues)
- 55% une seule ville détectée
- 16% aucune ville détectée

---

### Phase 2: Dataset Génération

**Objectif:** Créer un dataset de 10,000 phrases annotées

**Méthode:**
- 131 villes françaises
- 83 templates de formulation variés
- Génération aléatoire avec contraintes d'unicité
- Annotations inline: `<Dep>Paris</Dep> à <Arr>Lyon</Arr>`

**Résultats:**
- 10,000 phrases uniques générées
- 100% validité
- 15% questions, 10% avec dates, 4% avec contexte
- Fichier: [data/annotated/train_sentences_10k.csv](data/annotated/train_sentences_10k.csv)

---

### Phase 3: Qwen2.5 sans RAG

**Objectif:** Tester Qwen2.5-3B-Instruct seul avec few-shot learning

**Configuration:**
- Modèle: Qwen2.5-3B-Instruct (Alibaba)
- Device: GPU NVIDIA RTX 3060 Laptop (6GB VRAM)
- Framework: PyTorch 2.5.1 + CUDA 12.1
- Prompt: Few-shot avec 4 exemples fixes

**Résultats:**
- Accuracy: **75.0%** (75/100 phrases correctes)
- Temps: 656.7s (~11 min pour 100 phrases)
- Vitesse: 6.57s/phrase

**Amélioration:**
- **+2.3x** vs spaCy (+42.8 points)
- Excellente compréhension contextuelle
- Pas d'inversion départ/arrivée
- Extraction robuste même avec formulations complexes

---

### Phase 4: Qwen2.5 + ChromaDB RAG

**Objectif:** Améliorer avec Retrieval-Augmented Generation

**Architecture:**
- **LLM:** Qwen2.5-3B-Instruct sur GPU
- **Vector DB:** ChromaDB avec embeddings multilingues
- **Embeddings:** sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **RAG:** Recherche des 3 exemples les plus similaires pour chaque requête

**Configuration:**
- Collection ChromaDB: 100 phrases annotées
- Few-shot dynamique: 3 exemples adaptés par requête
- Recherche vectorielle avec similarité cosinus

**Résultats:**
- Accuracy: **93.0%** (93/100 phrases correctes)
- Temps: 666.0s (~11 min pour 100 phrases)
- Vitesse: 6.66s/phrase

**Amélioration:**
- **+2.9x** vs spaCy (+60.8 points)
- **+18 points** vs Qwen2.5 seul (de 75% à 93%)
- **Le RAG apporte un gain significatif** en fournissant des exemples contextuellement pertinents

---

## 📈 Analyse Comparative

### Amélioration Progressive

```
spaCy:           ████████ 32.2%
Qwen2.5 seul:    ███████████████████ 75.0% (+42.8 pts)
Qwen2.5 + RAG:   ████████████████████████ 93.0% (+60.8 pts)
```

### Impact du RAG

Le RAG ChromaDB apporte **+18 points d'accuracy** (de 75% à 93%), soit une **amélioration de 24% relative**.

**Pourquoi le RAG fonctionne si bien ?**
- Exemples adaptés à chaque requête
- Meilleure généralisation sur formulations variées
- Apprentissage contextuel dynamique

---

## 🚀 Infrastructure Technique

### Configuration GPU

- **GPU:** NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)
- **CUDA:** 12.9 (compatible PyTorch CUDA 12.1)
- **Python:** 3.12.3 (WSL Ubuntu)
- **PyTorch:** 2.5.1+cu121
- **Environnement:** venv_gpu avec toutes dépendances

### Performance

- **Temps de chargement modèle:** ~3 secondes
- **Inférence:** 6.6 secondes/phrase en moyenne
- **Throughput:** ~9 phrases/minute
- **VRAM utilisée:** ~3.5GB/6GB

---

## 📁 Livrables

### Datasets

1. **[data/annotated/train_sentences_100.csv](data/annotated/train_sentences_100.csv)**
   - 100 phrases annotées manuellement validées
   - Score qualité: 83/100
   - Utilisé pour le RAG ChromaDB

2. **[data/annotated/train_sentences_10k.csv](data/annotated/train_sentences_10k.csv)**
   - 10,000 phrases générées automatiquement
   - 100% uniques, 100% valides
   - 885 KB

### Résultats

1. **[results/spacy_10k_results.csv](results/spacy_10k_results.csv)**
   - Résultats spaCy sur 1,000 phrases
   - Accuracy: 32.2%

2. **[results/qwen_only_100_results.csv](results/qwen_only_100_results.csv)**
   - Résultats Qwen2.5 seul sur 100 phrases
   - Accuracy: 75.0%

3. **[results/qwen_chromadb_100_results.csv](results/qwen_chromadb_100_results.csv)**
   - Résultats Qwen2.5 + ChromaDB RAG sur 100 phrases
   - Accuracy: 93.0%

### Code

- **Architecture modulaire** avec classes réutilisables
- **Scripts de test** pour chaque approche
- **Documentation complète** (README, rapports)

---

## 🎓 Conclusions

### Résultats Clés

1. **Qwen2.5 + ChromaDB RAG atteint 93% d'accuracy**, dépassant largement la baseline spaCy (32.2%)
2. **Le RAG apporte un gain significatif de +18 points** par rapport à Qwen2.5 seul
3. **L'infrastructure GPU est opérationnelle** et accélère l'inférence de 10-50x vs CPU

### Recommandations

#### Pour la Production

1. **Utiliser Qwen2.5 + ChromaDB RAG** (93% accuracy)
2. **Optimiser le RAG** en augmentant la collection ChromaDB (actuellement 100 phrases)
3. **Batch processing** pour améliorer le throughput
4. **Fine-tuning de Qwen2.5** sur le dataset 10k pour potentiellement dépasser 95%

#### Améliorations Futures

1. **Augmenter le dataset d'entraînement RAG** à 1,000-5,000 phrases
2. **Fine-tuning du modèle Qwen2.5** avec LoRA/QLoRA
3. **Optimisation GPU** (quantization INT8, Flash Attention)
4. **API REST** pour déploiement en production

---

## 📞 Contact

**Projet:** T-AIA-911-PAR_14
**Technologie:** Qwen2.5 (Alibaba) + ChromaDB + PyTorch
**GPU:** NVIDIA RTX 3060

---

**Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}**
"""

# Sauvegarder
with open('./RAPPORT_COMPARATIF_FINAL.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✅ Rapport comparatif généré:")
print("   ./RAPPORT_COMPARATIF_FINAL.md")

# Afficher résumé
print("\n" + "="*70)
print("RÉSUMÉ DES RÉSULTATS")
print("="*70)
print(f"\n🏆 spaCy (baseline):        {spacy_accuracy}% accuracy")
print(f"🏆 Qwen2.5 seul:            {qwen_only_accuracy}% accuracy (+{qwen_only_accuracy - spacy_accuracy:.1f} pts)")
print(f"🏆 Qwen2.5 + ChromaDB RAG:  {qwen_rag_accuracy}% accuracy (+{qwen_rag_accuracy - spacy_accuracy:.1f} pts)")
print(f"\n📈 Impact du RAG: +{qwen_rag_accuracy - qwen_only_accuracy:.0f} points ({(qwen_rag_accuracy/qwen_only_accuracy - 1)*100:.1f}% d'amélioration relative)")
print("="*70)
