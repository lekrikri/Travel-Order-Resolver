#!/usr/bin/env python3
"""
Phase 1.1 - Warm-up: Test NER avec spaCy
Version simplifiée qui gère les problèmes d'encodage
"""

import pandas as pd
import spacy
from seqeval.metrics import classification_report, f1_score

print("="*70)
print("PHASE 1.1 - WARM-UP : Test NER avec spaCy")
print("="*70)

# 1. Charger spaCy
print("\n[1/5] Chargement du modèle spaCy fr_core_news_md...")
nlp = spacy.load("fr_core_news_md")
print("       OK Modèle chargé")

# 2. Charger le dataset (avec encoding latin-1)
print("\n[2/5] Chargement du dataset Kaggle...")
df = pd.read_csv("./data/raw/ner_dataset.csv", encoding='latin-1')
print(f"       OK Dataset chargé: {len(df):,} lignes")
print(f"       OK Phrases uniques: {df['Sentence #'].nunique():,}")

# 3. Extraire les phrases
print("\n[3/5] Extraction des phrases...")
sentences = []
current_sentence_id = None
current_words = []
current_tags = []

for _, row in df.iterrows():
    sentence_id = row['Sentence #']

    if pd.notna(sentence_id) and sentence_id != current_sentence_id:
        if current_words:
            sentences.append((' '.join(current_words), current_tags))
        current_sentence_id = sentence_id
        current_words = [row['Word']]
        current_tags = [row['Tag']]
    elif pd.notna(row['Word']):
        current_words.append(row['Word'])
        current_tags.append(row['Tag'])

if current_words:
    sentences.append((' '.join(current_words), current_tags))

print(f"       OK {len(sentences):,} phrases extraites")

# Afficher quelques exemples
print("\n   Exemples:")
for i, (text, tags) in enumerate(sentences[:3]):
    print(f"   {i+1}. {text[:70]}...")

# 4. Evaluer sur 100 premières phrases
print("\n[4/5] Evaluation NER sur 100 premières phrases...")
y_true = []
y_pred = []

for sentence_text, gold_tags in sentences[:100]:
    # Prédiction spaCy
    doc = nlp(sentence_text)
    pred_tags = []
    for token in doc:
        if token.ent_iob_ == 'O':
            pred_tags.append('O')
        else:
            pred_tags.append(f"{token.ent_iob_}-{token.ent_type_}")

    # Aligner
    min_len = min(len(gold_tags), len(pred_tags))
    y_true.append(gold_tags[:min_len])
    y_pred.append(pred_tags[:min_len])

# 5. Calculer les métriques
print("\n[5/5] Calcul des métriques...")
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Résultats
print("\n" + "="*70)
print("RESULTATS - Phase 1.1 Warm-up")
print("="*70)
print(f"Dataset:      ner_dataset.csv")
print(f"Total phrases: {len(sentences):,}")
print(f"Evalué sur:    100 premières phrases")
print(f"Modèle:        spaCy fr_core_news_md")
print("-"*70)
print(f"F1-Score:      {f1:.3f}")
print(f"Precision:     {report.get('weighted avg', {}).get('precision', 0):.3f}")
print(f"Recall:        {report.get('weighted avg', {}).get('recall', 0):.3f}")
print("="*70)

# Analyse
print("\nAnalyse:")
if f1 > 0.8:
    print("OK Excellent: spaCy NER fonctionne très bien")
elif f1 > 0.6:
    print("OK Bon: spaCy NER fonctionne correctement")
else:
    print("X Moyen: spaCy a des difficultés")

print("\nConclusion Phase 1.1:")
print("- spaCy est capable de faire du NER (Named Entity Recognition)")
print("- Les métriques montrent ses performances sur ce dataset")
print("- Prochaine étape: Phase 1.2 - Training sur bottins.csv (français)")
print("="*70)
