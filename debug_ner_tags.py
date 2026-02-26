#!/usr/bin/env python3
"""
Debug: Comparer les tags gold vs predicted
"""

import pandas as pd
import spacy

# Charger spaCy anglais
nlp = spacy.load("en_core_web_sm")

# Charger dataset
df = pd.read_csv("./data/raw/ner_dataset.csv", encoding='latin-1')

# Extraire premiere phrase
sentences = []
current_words = []
current_tags = []

for _, row in df.iterrows():
    if pd.notna(row['Sentence #']) and current_words:
        sentences.append((' '.join(current_words), current_tags))
        if len(sentences) >= 5:  # Prendre 5 phrases
            break
        current_words = []
        current_tags = []

    if pd.notna(row['Word']):
        current_words.append(str(row['Word']))
        current_tags.append(str(row['Tag']))

print("="*70)
print("DEBUG - Comparaison tags GOLD vs PREDICTED")
print("="*70)

for i, (text, gold_tags) in enumerate(sentences[:3]):
    print(f"\nPhrase {i+1}: {text[:60]}...")
    print("-"*70)

    # Prediction spaCy
    doc = nlp(text)

    print(f"GOLD tags:      {' | '.join(gold_tags[:10])}...")

    pred_tags = []
    for token in doc:
        if token.ent_iob_ == 'O':
            pred_tags.append('O')
        else:
            pred_tags.append(f"{token.ent_iob_}-{token.ent_type_}")

    print(f"PREDICTED tags: {' | '.join(pred_tags[:10])}...")

    print(f"\nTokens avec entites:")
    for token in doc:
        if token.ent_type_:
            print(f"  - '{token.text}': {token.ent_iob_}-{token.ent_type_}")

print("\n" + "="*70)
print("Tags GOLD uniques dans le dataset:")
unique_gold = set()
for _, tags in sentences:
    unique_gold.update(tags)
print(sorted(unique_gold))

print("\nTags PREDICTED uniques par spaCy:")
unique_pred = set()
for text, _ in sentences:
    doc = nlp(text)
    for token in doc:
        if token.ent_type_:
            unique_pred.add(f"{token.ent_iob_}-{token.ent_type_}")
unique_pred.add('O')
print(sorted(unique_pred))

print("\n" + "="*70)
print("CONCLUSION:")
print("Les tags gold et predicted utilisent des schemas differents!")
print("- Gold: B-geo, I-geo, B-per, etc. (CoNLL format)")
print("- Predicted: B-GPE, I-PERSON, B-LOC, etc. (spaCy format)")
print("Il faut mapper les tags pour que seqeval fonctionne.")
print("="*70)
