#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1.1 - Warm-up avec WikiNER-fr (vrai dataset français)
Test de spaCy français sur données réelles
"""

import sys
import io
import spacy
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("PHASE 1.1 - WARM-UP : WikiNER-fr (Dataset français)")
print("="*70)

# 1. Télécharger WikiNER-fr
print("\n[1/5] Téléchargement de WikiNER-fr depuis HuggingFace...")
print("       (Cela peut prendre 1-2 minutes la première fois)")
try:
    dataset = load_dataset("Babelscape/wikineural-multilingual-ner", "fr")
    train_data = dataset['train']
    print(f"       OK Dataset téléchargé: {len(train_data):,} phrases")
except Exception as e:
    print(f"       ERREUR: {e}")
    print("       Essayons un autre dataset...")
    # Fallback: autre dataset français
    dataset = load_dataset("Jean-Baptiste/wikiner_fr")
    train_data = dataset['train']
    print(f"       OK Dataset alternatif chargé: {len(train_data):,} phrases")

# 2. Charger spaCy français
print("\n[2/5] Chargement du modèle spaCy fr_core_news_md...")
nlp = spacy.load("fr_core_news_md")
print("       OK Modèle français chargé")

# 3. Préparer les données (prendre 100 premières phrases)
print("\n[3/5] Préparation des phrases pour évaluation...")
sentences = []
for i, example in enumerate(train_data):
    if i >= 100:  # Limiter à 100 pour rapidité
        break

    # Format WikiNER: tokens et ner_tags
    if 'tokens' in example and 'ner_tags' in example:
        tokens = example['tokens']
        ner_tags = example['ner_tags']

        # Convertir les IDs de tags en labels
        # Format typique: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, etc.
        tag_names = {
            0: 'O',
            1: 'B-PER', 2: 'I-PER',
            3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC',
            7: 'B-MISC', 8: 'I-MISC'
        }

        # Si le dataset a déjà les noms de tags
        if isinstance(ner_tags[0], str):
            gold_tags = ner_tags
        else:
            gold_tags = [tag_names.get(tag_id, 'O') for tag_id in ner_tags]

        sentence_text = ' '.join(tokens)
        sentences.append((sentence_text, gold_tags))

print(f"       OK {len(sentences)} phrases préparées")

# Afficher exemples
print("\n   Exemples de phrases:")
for i, (text, tags) in enumerate(sentences[:3]):
    print(f"   {i+1}. {text[:70]}...")
    # Afficher quelques tags
    entities = [(tok, tag) for tok, tag in zip(text.split()[:10], tags[:10]) if tag != 'O']
    if entities:
        print(f"      Entités: {entities}")

# 4. Evaluer spaCy
print("\n[4/5] Évaluation de spaCy français sur WikiNER-fr...")
y_true = []
y_pred = []

for sentence_text, gold_tags in sentences:
    # Prédiction spaCy
    doc = nlp(sentence_text)

    pred_tags = []
    for token in doc:
        if token.ent_iob_ == 'O':
            pred_tags.append('O')
        else:
            # Mapper les types d'entités spaCy vers format CoNLL
            entity_type = token.ent_type_
            if entity_type == 'PER':
                entity_type = 'PER'
            elif entity_type in ['GPE', 'LOC']:
                entity_type = 'LOC'
            elif entity_type == 'ORG':
                entity_type = 'ORG'
            else:
                entity_type = 'MISC'

            pred_tags.append(f"{token.ent_iob_}-{entity_type}")

    # Aligner (même longueur)
    min_len = min(len(gold_tags), len(pred_tags))
    y_true.append(gold_tags[:min_len])
    y_pred.append(pred_tags[:min_len])

# 5. Calculer les métriques
print("\n[5/5] Calcul des métriques...")
try:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Résultats
    print("\n" + "="*70)
    print("RESULTATS - Phase 1.1 Warm-up (WikiNER-fr)")
    print("="*70)
    print(f"Dataset:       WikiNER-fr (Wikipedia français)")
    print(f"Total phrases: {len(train_data):,} (dataset complet)")
    print(f"Evalué sur:    {len(sentences)} phrases")
    print(f"Modèle:        spaCy fr_core_news_md")
    print("-"*70)
    print(f"F1-Score:      {f1:.3f}")
    print(f"Precision:     {report.get('weighted avg', {}).get('precision', 0):.3f}")
    print(f"Recall:        {report.get('weighted avg', {}).get('recall', 0):.3f}")
    print("="*70)

    # Analyse
    print("\nAnalyse:")
    if f1 > 0.7:
        print("OK Excellent: spaCy français fonctionne très bien sur WikiNER-fr")
    elif f1 > 0.5:
        print("OK Bon: spaCy français fonctionne correctement")
    else:
        print("X Moyen: spaCy a des difficultés")

    print("\nDétails:")
    print(f"- Dataset: WikiNER-fr (extraits de Wikipédia français)")
    print(f"- Modèle: spaCy fr_core_news_md (entraîné sur actualités françaises)")
    print(f"- Tags NER: PER (personne), LOC (lieu), ORG (organisation), MISC")
    print(f"- Test cross-domain: actualités (modèle) vs encyclopédie (données)")

except Exception as e:
    print(f"\nERREUR lors du calcul des métriques: {e}")
    print("Les tags gold et predicted ne correspondent peut-être pas...")

print("\nConclusion Phase 1.1:")
print("- spaCy français testé sur un vrai dataset français (WikiNER)")
print("- Les métriques montrent ses performances sur du texte encyclopédique")
print("- Prochaine étape: Phase 1.2 - Training et comparaison avec Transformers")
print("="*70)
