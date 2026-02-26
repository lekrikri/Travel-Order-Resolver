"""
Conversion du dataset annoté existant vers format spaCy
=======================================================
Convertit les balises <Dep>...</Dep> et <Arr>...</Arr> en annotations spaCy.

Exécuter avec : python3 src/dataset/convert_annotated_to_spacy.py
"""

import csv
import re
import json
import random
from pathlib import Path
import spacy
from spacy.tokens import DocBin


def extraire_entites(sentence):
    """
    Extrait les entités depuis une phrase avec balises.

    Input:  "Je pars de <Dep>Paris</Dep> vers <Arr>Lyon</Arr>"
    Output: ("Je pars de Paris vers Lyon", [(11, 16, "DEPART"), (22, 26, "ARRIVEE")])
    """
    entities = []

    # Trouver toutes les balises <Dep>...</Dep>
    for match in re.finditer(r'<Dep>(.*?)</Dep>', sentence):
        ville = match.group(1)
        entities.append({
            'text': ville,
            'label': 'DEPART',
            'original_start': match.start(),
            'original_end': match.end()
        })

    # Trouver toutes les balises <Arr>...</Arr>
    for match in re.finditer(r'<Arr>(.*?)</Arr>', sentence):
        ville = match.group(1)
        entities.append({
            'text': ville,
            'label': 'ARRIVEE',
            'original_start': match.start(),
            'original_end': match.end()
        })

    # Supprimer les balises pour obtenir le texte propre
    clean_text = re.sub(r'</?(?:Dep|Arr)>', '', sentence)

    # Recalculer les positions dans le texte propre
    final_entities = []
    for ent in entities:
        start = clean_text.find(ent['text'])
        if start != -1:
            end = start + len(ent['text'])
            final_entities.append((start, end, ent['label']))

    # Trier par position
    final_entities.sort(key=lambda x: x[0])

    return clean_text, final_entities


def charger_dataset(filepath):
    """Charge le dataset CSV annoté."""
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row.get('sentence', '')
            departure = row.get('departure', '')
            arrival = row.get('arrival', '')

            if sentence:
                clean_text, entities = extraire_entites(sentence)
                data.append({
                    'text': clean_text,
                    'entities': entities,
                    'depart': departure,
                    'arrivee': arrival,
                    'valid': bool(departure and arrival)
                })

    return data


def split_dataset(data, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Divise le dataset en train/val/test."""
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }


def sauvegarder_json(splits, output_dir):
    """Sauvegarde en JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for nom, donnees in splits.items():
        filepath = output_path / f"{nom}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)
        print(f"  {nom}.json: {len(donnees)} exemples")


def convertir_vers_spacy(data, nlp, output_path):
    """Convertit en format .spacy."""
    doc_bin = DocBin()
    erreurs = 0

    for item in data:
        doc = nlp.make_doc(item['text'])
        ents = []

        for start, end, label in item['entities']:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
            else:
                erreurs += 1

        doc.ents = ents
        doc_bin.add(doc)

    doc_bin.to_disk(output_path)
    return erreurs


def main():
    print("=" * 60)
    print("CONVERSION DU DATASET ANNOTÉ EXISTANT")
    print("=" * 60)

    # Chemins
    input_file = Path("/home/lekrikri/Projects/T-AIA-911-PAR_14/data/annotated/train_sentences_10k.csv")
    output_dir = Path("/home/lekrikri/Projects/T-AIA-911-PAR_14/data/processed")

    # 1. Charger le dataset
    print(f"\nChargement de {input_file}...")
    data = charger_dataset(input_file)
    print(f"  {len(data)} phrases chargees")

    # 2. Vérifier quelques exemples
    print("\nExemples de conversion:")
    for i, item in enumerate(data[:3]):
        print(f"  {i+1}. \"{item['text'][:50]}...\"")
        print(f"     Entites: {item['entities']}")

    # 3. Split
    print("\nSplit du dataset...")
    splits = split_dataset(data)

    # 4. Sauvegarder JSON
    print("\nSauvegarde JSON:")
    sauvegarder_json(splits, output_dir)

    # 5. Convertir en .spacy
    print("\nConversion en format .spacy:")
    nlp = spacy.blank("fr")

    for nom, donnees in splits.items():
        output_path = output_dir / f"{nom}.spacy"
        erreurs = convertir_vers_spacy(donnees, nlp, output_path)
        print(f"  {nom}.spacy: {len(donnees)} docs (erreurs alignement: {erreurs})")

    # Stats
    print("\n" + "=" * 60)
    print("CONVERSION TERMINEE!")
    print("=" * 60)
    print(f"\nDataset: {len(data)} phrases")
    print(f"  Train: {len(splits['train'])} (70%)")
    print(f"  Val:   {len(splits['val'])} (15%)")
    print(f"  Test:  {len(splits['test'])} (15%)")


if __name__ == "__main__":
    main()
