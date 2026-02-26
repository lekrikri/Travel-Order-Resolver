# 🔄 Nouvelle Approche : spaCy NER Custom + Fuzzy Matching
## T-AIA-911-PAR_14 | Follow-Up 2

---

## ⚠️ Feedback du Jury

> **"Overkill !"** - L'utilisation d'un LLM (Qwen2.5) est trop lourde pour ce problème.
> On peut faire plus simple avec spaCy et obtenir de bons résultats.

### Nouvelle Direction :
1. **NER Custom** avec spaCy (entraîné sur nos données)
2. **Fuzzy Matching** pour corriger les noms de gares
3. **Comparaison de modèles** (sm, md, lg, blank, CamemBERT)
4. **Métriques quantitatives** rigoureuses

---

## 📋 Attendus FU2

| Critère | Description | Priorité |
|---------|-------------|----------|
| Dataset split | train/validation/test documenté | ⭐⭐⭐ |
| Multi-modèles spaCy | Comparer sm/md/lg/blank | ⭐⭐⭐ |
| Modèle CamemBERT | Au moins un modèle transformer | ⭐⭐⭐ |
| Métriques | Precision, Recall, F1 | ⭐⭐⭐ |
| Critique qualitative | Analyse performance vs ressources | ⭐⭐⭐ |
| CLI | Pas de Jupyter, gestion d'erreurs | ⭐⭐ |
| Bonus graph | Dijkstra/A*/paramètres | ⭐⭐ |

---

## 🏗️ Architecture Simplifiée

```
┌─────────────────────────────────────────────────────────────┐
│                    NOUVELLE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT: "Florence et moi voulons aller à Annecy depuis Paris" │
│                           │                                   │
│                           ▼                                   │
│              ┌────────────────────────┐                      │
│              │   NER spaCy Custom     │                      │
│              │  (entraîné sur notre   │                      │
│              │      dataset)          │                      │
│              └────────────────────────┘                      │
│                           │                                   │
│            Entités: [Paris:DEPART, Annecy:ARRIVEE]           │
│                           │                                   │
│                           ▼                                   │
│              ┌────────────────────────┐                      │
│              │    Fuzzy Matching      │                      │
│              │  (correction noms      │                      │
│              │   vers gares SNCF)     │                      │
│              └────────────────────────┘                      │
│                           │                                   │
│                           ▼                                   │
│              ┌────────────────────────┐                      │
│              │   Pathfinder Graph     │                      │
│              │  (Dijkstra / A*)       │                      │
│              └────────────────────────┘                      │
│                           │                                   │
│                           ▼                                   │
│  OUTPUT: Paris → Lyon → Chambéry → Annecy                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Différence clé :** On entraîne spaCy à reconnaître **directement** DEPART et ARRIVEE (pas juste LOC).

---

## 📁 Structure du Projet FU2

```
T-AIA-911-PAR_14/
├── data/
│   ├── raw/
│   │   └── phrases_raw.csv
│   ├── processed/
│   │   ├── train.spacy        # 70%
│   │   ├── val.spacy          # 15%
│   │   └── test.spacy         # 15%
│   └── sncf/
│       └── gares_sncf.csv
├── src/
│   ├── dataset/
│   │   ├── generate_dataset.py
│   │   └── convert_to_spacy.py
│   ├── models/
│   │   ├── train_ner_spacy.py
│   │   ├── train_camembert.py
│   │   └── config/
│   │       ├── config_sm.cfg
│   │       ├── config_md.cfg
│   │       └── config_lg.cfg
│   ├── evaluation/
│   │   ├── evaluate_models.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── fuzzy_matcher.py
│   │   └── pipeline.py
│   └── cli/
│       └── main.py
├── models/
│   ├── ner_sm/
│   ├── ner_md/
│   ├── ner_lg/
│   ├── ner_blank/
│   └── ner_camembert/
├── results/
│   ├── metrics_comparison.csv
│   └── figures/
├── requirements.txt
└── README.md
```

---

# ÉTAPE 1 : Dataset avec Split

## 1.1 Génération du Dataset Annoté

Le format d'annotation spaCy pour NER :
```python
# Format: (texte, {"entities": [(start, end, label), ...]})
("Je veux aller de Paris à Lyon", {"entities": [(17, 22, "DEPART"), (25, 29, "ARRIVEE")]})
```

### Script : `src/dataset/generate_dataset.py`

```python
"""
Génération du dataset annoté pour NER spaCy
Labels: DEPART, ARRIVEE
"""
import csv
import random
import json
from pathlib import Path

# Gares SNCF (échantillon - à compléter avec le fichier complet)
GARES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
    "Nice", "Nantes", "Strasbourg", "Montpellier", "Lille",
    "Rennes", "Grenoble", "Rouen", "Toulon", "Dijon",
    "Annecy", "Chambéry", "Avignon", "Aix-en-Provence", "Cannes",
    "Saint-Étienne", "Le Havre", "Reims", "Saint-Denis", "Toulon",
    "Clermont-Ferrand", "Villeurbanne", "Metz", "Besançon", "Orléans"
]

# Prénoms pour les phrases complexes
PRENOMS = ["Florence", "Marie", "Jean", "Pierre", "Sophie", "Lucas", "Emma", "Louis"]

# Templates avec positions marquées
# {D} = DEPART, {A} = ARRIVEE
TEMPLATES = [
    # Patterns classiques
    ("Je veux aller de {D} à {A}", "standard"),
    ("Un billet de {D} vers {A} s'il vous plaît", "standard"),
    ("Je pars de {D} pour aller à {A}", "standard"),
    ("Train de {D} à {A}", "standard"),
    ("Aller simple {D} {A}", "standard"),
    ("Je voudrais un aller-retour {D} {A}", "standard"),

    # Patterns inversés (arrivée mentionnée avant départ)
    ("Direction {A} depuis {D}", "inverse"),
    ("Vers {A} en partant de {D}", "inverse"),
    ("À destination de {A} depuis {D}", "inverse"),
    ("Pour aller à {A}, je pars de {D}", "inverse"),

    # Patterns complexes avec contexte
    ("{P} et moi voulons aller à {A} depuis {D}", "complexe"),
    ("Avec {P}, on va de {D} à {A}", "complexe"),
    ("Je dois rejoindre {P} à {A}, je suis à {D}", "complexe"),
    ("{P} m'attend à {A}, je pars de {D}", "complexe"),

    # Patterns avec tiret
    ("{D} - {A}", "tiret"),
    ("{D}-{A} pour demain", "tiret"),
    ("Billet {D}-{A}", "tiret"),
]

TEMPLATES_INVALIDES = [
    "Quel temps fait-il aujourd'hui ?",
    "Bonjour, comment allez-vous ?",
    "Je cherche un hôtel",
    "Quelle heure est-il ?",
    "Merci beaucoup",
    "Au revoir",
    "Je voudrais un café",
    "Où sont les toilettes ?",
]


def find_entity_positions(text, entity, label):
    """Trouve la position d'une entité dans le texte"""
    start = text.find(entity)
    if start == -1:
        return None
    end = start + len(entity)
    return (start, end, label)


def generate_annotated_phrase():
    """Génère une phrase annotée avec DEPART et ARRIVEE"""
    template, pattern_type = random.choice(TEMPLATES)
    depart, arrivee = random.sample(GARES, 2)
    prenom = random.choice(PRENOMS)

    # Remplacer les placeholders
    text = template.replace("{D}", depart).replace("{A}", arrivee).replace("{P}", prenom)

    # Trouver les positions des entités
    entities = []

    dep_pos = find_entity_positions(text, depart, "DEPART")
    arr_pos = find_entity_positions(text, arrivee, "ARRIVEE")

    if dep_pos:
        entities.append(dep_pos)
    if arr_pos:
        entities.append(arr_pos)

    # Trier par position de début
    entities.sort(key=lambda x: x[0])

    return {
        "text": text,
        "entities": entities,
        "pattern_type": pattern_type,
        "depart": depart,
        "arrivee": arrivee,
        "valid": True
    }


def generate_invalid_phrase():
    """Génère une phrase sans itinéraire"""
    text = random.choice(TEMPLATES_INVALIDES)
    return {
        "text": text,
        "entities": [],
        "pattern_type": "invalid",
        "depart": "INVALID",
        "arrivee": "INVALID",
        "valid": False
    }


def generate_dataset(n_valid=8000, n_invalid=2000, seed=42):
    """Génère le dataset complet"""
    random.seed(seed)

    data = []

    # Phrases valides
    for _ in range(n_valid):
        data.append(generate_annotated_phrase())

    # Phrases invalides
    for _ in range(n_invalid):
        data.append(generate_invalid_phrase())

    random.shuffle(data)
    return data


def split_dataset(data, train_ratio=0.7, val_ratio=0.15):
    """Split en train/validation/test"""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


def save_dataset(splits, output_dir="data/processed"):
    """Sauvegarde les splits en JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        filepath = output_path / f"{split_name}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Sauvegardé: {filepath} ({len(split_data)} exemples)")

    # Stats
    print("\n📊 Statistiques du dataset:")
    for split_name, split_data in splits.items():
        valid = sum(1 for d in split_data if d['valid'])
        invalid = len(split_data) - valid
        print(f"  {split_name}: {len(split_data)} total ({valid} valides, {invalid} invalides)")


if __name__ == "__main__":
    print("🔄 Génération du dataset annoté pour NER...")

    # Générer
    data = generate_dataset(n_valid=8000, n_invalid=2000)

    # Split
    splits = split_dataset(data)

    # Sauvegarder
    save_dataset(splits)

    print("\n✅ Dataset généré avec succès!")
```

## 1.2 Conversion au Format spaCy

### Script : `src/dataset/convert_to_spacy.py`

```python
"""
Convertit le dataset JSON au format .spacy pour l'entraînement
"""
import json
import spacy
from spacy.tokens import DocBin
from pathlib import Path


def convert_to_spacy_format(json_path, output_path, nlp):
    """Convertit un fichier JSON en format .spacy"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_bin = DocBin()

    for item in data:
        text = item['text']
        entities = item['entities']

        doc = nlp.make_doc(text)
        ents = []

        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)

        doc.ents = ents
        doc_bin.add(doc)

    doc_bin.to_disk(output_path)
    print(f"Converti: {json_path} → {output_path} ({len(data)} docs)")


def main():
    # Charger un modèle de base pour la tokenisation
    nlp = spacy.blank("fr")

    input_dir = Path("data/processed")
    output_dir = Path("data/processed")

    for split in ["train", "val", "test"]:
        json_path = input_dir / f"{split}.json"
        spacy_path = output_dir / f"{split}.spacy"

        if json_path.exists():
            convert_to_spacy_format(json_path, spacy_path, nlp)


if __name__ == "__main__":
    main()
```

---

# ÉTAPE 2 : Entraînement NER spaCy

## 2.1 Configuration spaCy

### Fichier : `src/models/config/base_config.cfg`

```ini
[paths]
train = "data/processed/train.spacy"
dev = "data/processed/val.spacy"

[system]
gpu_allocator = null

[nlp]
lang = "fr"
pipeline = ["tok2vec", "ner"]

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96
attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
rows = [5000, 2500, 2500, 2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
max_epochs = 30
patience = 5

[training.optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.001

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
size = 1000

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
```

## 2.2 Script d'Entraînement Multi-Modèles

### Script : `src/models/train_ner_spacy.py`

```python
"""
Entraînement de modèles NER spaCy avec différentes configurations
Compare: blank, sm, md, lg
"""
import subprocess
import json
from pathlib import Path
import spacy
from datetime import datetime


def train_model(base_model, output_dir, config_path, train_path, dev_path):
    """Entraîne un modèle NER"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Entraînement: {base_model or 'blank'}")
    print(f"{'='*60}")

    # Commande spacy train
    cmd = [
        "python", "-m", "spacy", "train",
        config_path,
        "--output", str(output_path),
        "--paths.train", train_path,
        "--paths.dev", dev_path,
    ]

    # Ajouter le modèle de base si spécifié
    if base_model:
        # Pour utiliser un modèle pré-entraîné, on doit modifier la config
        cmd.extend(["--training.vectors", base_model])

    print(f"Commande: {' '.join(cmd)}")

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()

    training_time = (end_time - start_time).total_seconds()

    if result.returncode == 0:
        print(f"✅ Succès en {training_time:.1f}s")
    else:
        print(f"❌ Erreur: {result.stderr}")

    return {
        "model": base_model or "blank",
        "success": result.returncode == 0,
        "training_time": training_time,
        "output_dir": str(output_path)
    }


def train_from_pretrained(base_model_name, output_name):
    """Entraîne en partant d'un modèle pré-entraîné"""

    print(f"\n{'='*60}")
    print(f"🚀 Entraînement depuis: {base_model_name}")
    print(f"{'='*60}")

    # Charger le modèle de base
    nlp = spacy.load(base_model_name)

    # Ajouter le NER si pas présent, ou le réinitialiser
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Ajouter nos labels
    ner.add_label("DEPART")
    ner.add_label("ARRIVEE")

    # Charger les données d'entraînement
    from spacy.tokens import DocBin

    train_data = DocBin().from_disk("data/processed/train.spacy")

    # Entraînement
    from spacy.training import Example
    import random

    # Obtenir les documents
    train_docs = list(train_data.get_docs(nlp.vocab))

    # Entraîner
    optimizer = nlp.resume_training()

    for epoch in range(20):
        random.shuffle(train_docs)
        losses = {}

        for doc in train_docs:
            example = Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]})
            nlp.update([example], sgd=optimizer, losses=losses)

        print(f"Epoch {epoch+1}: Loss = {losses.get('ner', 0):.4f}")

    # Sauvegarder
    output_path = Path(f"models/{output_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)

    print(f"✅ Modèle sauvegardé: {output_path}")
    return str(output_path)


def main():
    """Entraîne plusieurs variantes de modèles"""

    results = []

    # Configuration commune
    config_path = "src/models/config/base_config.cfg"
    train_path = "data/processed/train.spacy"
    dev_path = "data/processed/val.spacy"

    # 1. Modèle blank (from scratch)
    results.append(train_model(
        base_model=None,
        output_dir="models/ner_blank",
        config_path=config_path,
        train_path=train_path,
        dev_path=dev_path
    ))

    # 2. Depuis fr_core_news_sm
    try:
        train_from_pretrained("fr_core_news_sm", "ner_sm")
        results.append({"model": "fr_core_news_sm", "success": True})
    except Exception as e:
        print(f"Erreur sm: {e}")
        results.append({"model": "fr_core_news_sm", "success": False, "error": str(e)})

    # 3. Depuis fr_core_news_md
    try:
        train_from_pretrained("fr_core_news_md", "ner_md")
        results.append({"model": "fr_core_news_md", "success": True})
    except Exception as e:
        print(f"Erreur md: {e}")
        results.append({"model": "fr_core_news_md", "success": False, "error": str(e)})

    # 4. Depuis fr_core_news_lg
    try:
        train_from_pretrained("fr_core_news_lg", "ner_lg")
        results.append({"model": "fr_core_news_lg", "success": True})
    except Exception as e:
        print(f"Erreur lg: {e}")
        results.append({"model": "fr_core_news_lg", "success": False, "error": str(e)})

    # Sauvegarder les résultats
    with open("results/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n📊 Résumé de l'entraînement:")
    for r in results:
        status = "✅" if r.get("success") else "❌"
        print(f"  {status} {r['model']}")


if __name__ == "__main__":
    main()
```

---

# ÉTAPE 3 : Modèle CamemBERT

### Script : `src/models/train_camembert.py`

```python
"""
Entraînement d'un modèle NER basé sur CamemBERT
"""
import json
import torch
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score


# Labels
LABEL_LIST = ["O", "B-DEPART", "I-DEPART", "B-ARRIVEE", "I-ARRIVEE"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def load_data(json_path):
    """Charge les données depuis le JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize et aligne les labels avec les tokens"""

    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128,
        return_offsets_mapping=True
    )

    all_labels = []

    for i, (offsets, entities) in enumerate(zip(
        tokenized_inputs["offset_mapping"],
        examples["entities"]
    )):
        labels = ["O"] * len(offsets)

        for start, end, label in entities:
            for j, (token_start, token_end) in enumerate(offsets):
                if token_start is None:
                    continue
                if token_start >= start and token_end <= end:
                    if token_start == start:
                        labels[j] = f"B-{label}"
                    else:
                        labels[j] = f"I-{label}"

        all_labels.append([LABEL2ID.get(l, 0) for l in labels])

    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")

    return tokenized_inputs


def compute_metrics(pred):
    """Calcule les métriques d'évaluation"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []
        for p, l in zip(prediction, label):
            if l != -100:  # Ignorer les tokens de padding
                true_label.append(ID2LABEL[l])
                true_pred.append(ID2LABEL[p])
        true_labels.append(true_label)
        true_predictions.append(true_pred)

    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions)
    }


def main():
    print("🚀 Entraînement du modèle CamemBERT NER")

    # Charger le tokenizer et le modèle
    model_name = "camembert-base"
    tokenizer = CamembertTokenizerFast.from_pretrained(model_name)
    model = CamembertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Charger les données
    train_data = load_data("data/processed/train.json")
    val_data = load_data("data/processed/val.json")

    # Convertir en Dataset
    train_dataset = Dataset.from_dict({
        "text": [d["text"] for d in train_data],
        "entities": [d["entities"] for d in train_data]
    })
    val_dataset = Dataset.from_dict({
        "text": [d["text"] for d in val_data],
        "entities": [d["entities"] for d in val_data]
    })

    # Tokenizer
    train_tokenized = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/ner_camembert",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Entraîner
    trainer.train()

    # Sauvegarder
    trainer.save_model("models/ner_camembert/best")
    tokenizer.save_pretrained("models/ner_camembert/best")

    print("✅ Modèle CamemBERT sauvegardé!")


if __name__ == "__main__":
    main()
```

---

# ÉTAPE 4 : Fuzzy Matching

### Script : `src/inference/fuzzy_matcher.py`

```python
"""
Fuzzy Matching pour corriger les noms de gares vers les vraies gares SNCF
"""
import csv
from rapidfuzz import fuzz, process
from pathlib import Path


class FuzzyStationMatcher:
    """Matcher fuzzy pour les gares SNCF"""

    def __init__(self, stations_file="data/sncf/gares_sncf.csv"):
        self.stations = self._load_stations(stations_file)
        print(f"📍 {len(self.stations)} gares chargées")

    def _load_stations(self, filepath):
        """Charge la liste des gares"""
        stations = []

        # Si le fichier existe
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    # Adapter selon la structure du CSV SNCF
                    name = row.get('Nom de la gare') or row.get('name') or row.get('libelle')
                    if name:
                        stations.append(name.strip())
        else:
            # Fallback avec une liste de base
            stations = [
                "Paris Gare de Lyon", "Paris Montparnasse", "Paris Saint-Lazare",
                "Paris Nord", "Paris Est", "Paris Austerlitz", "Paris Bercy",
                "Lyon Part-Dieu", "Lyon Perrache", "Marseille Saint-Charles",
                "Bordeaux Saint-Jean", "Toulouse Matabiau", "Nice Ville",
                "Nantes", "Strasbourg", "Montpellier Saint-Roch", "Lille Flandres",
                "Lille Europe", "Rennes", "Grenoble", "Rouen Rive Droite",
                "Toulon", "Dijon Ville", "Annecy", "Chambéry", "Avignon TGV",
                "Aix-en-Provence TGV", "Cannes", "Saint-Étienne Châteaucreux",
                "Le Havre", "Reims", "Clermont-Ferrand", "Metz Ville",
                "Besançon Viotte", "Orléans", "Tours", "Poitiers", "La Rochelle",
                "Brest", "Quimper", "Vannes", "Lorient", "Saint-Malo", "Caen"
            ]

        return list(set(stations))  # Dédupliquer

    def match(self, query, threshold=70):
        """
        Trouve la gare la plus proche du query

        Args:
            query: Nom de ville/gare à matcher
            threshold: Score minimum (0-100)

        Returns:
            (station_name, score) ou (None, 0) si pas de match
        """
        if not query or query.upper() in ["INVALID", "UNKNOWN"]:
            return None, 0

        # Recherche fuzzy
        result = process.extractOne(
            query,
            self.stations,
            scorer=fuzz.WRatio,  # Weighted Ratio - bon pour les variations
            score_cutoff=threshold
        )

        if result:
            return result[0], result[1]
        return None, 0

    def match_with_alternatives(self, query, n=3, threshold=60):
        """
        Retourne les N meilleures correspondances

        Utile pour afficher des suggestions à l'utilisateur
        """
        if not query:
            return []

        results = process.extract(
            query,
            self.stations,
            scorer=fuzz.WRatio,
            limit=n,
            score_cutoff=threshold
        )

        return [(name, score) for name, score, _ in results]


# Test
if __name__ == "__main__":
    matcher = FuzzyStationMatcher()

    # Tests
    test_cases = [
        "Paris",           # Exact
        "Pari",            # Coquille
        "Lyon Part Dieu",  # Sans tiret
        "marseile",        # Faute d'orthographe
        "St Etienne",      # Abréviation
        "Aneci",           # Coquille
        "Bordeau",         # Manque lettre
        "Tgv Avignon",     # Ordre inversé
    ]

    print("\n🔍 Tests de Fuzzy Matching:")
    print("-" * 50)

    for query in test_cases:
        match, score = matcher.match(query)
        print(f"'{query}' → '{match}' (score: {score})")
```

---

# ÉTAPE 5 : Évaluation et Métriques

### Script : `src/evaluation/evaluate_models.py`

```python
"""
Évaluation et comparaison des modèles NER
Métriques: Precision, Recall, F1 pour DEPART et ARRIVEE
"""
import json
import spacy
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_test_data(json_path="data/processed/test.json"):
    """Charge les données de test"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_model(model_path, test_data):
    """Évalue un modèle sur les données de test"""

    try:
        nlp = spacy.load(model_path)
    except Exception as e:
        print(f"Erreur chargement {model_path}: {e}")
        return None

    # Compteurs par label
    metrics = {
        "DEPART": {"tp": 0, "fp": 0, "fn": 0},
        "ARRIVEE": {"tp": 0, "fp": 0, "fn": 0}
    }

    correct_full = 0  # Les deux entités correctes
    total = len(test_data)

    for item in test_data:
        text = item["text"]
        expected_entities = {(e[0], e[1], e[2]) for e in item["entities"]}

        # Prédiction
        doc = nlp(text)
        predicted_entities = {(e.start_char, e.end_char, e.label_) for e in doc.ents}

        # Pour chaque label
        for label in ["DEPART", "ARRIVEE"]:
            expected_label = {e for e in expected_entities if e[2] == label}
            predicted_label = {e for e in predicted_entities if e[2] == label}

            # True Positives
            tp = len(expected_label & predicted_label)
            metrics[label]["tp"] += tp

            # False Positives
            fp = len(predicted_label - expected_label)
            metrics[label]["fp"] += fp

            # False Negatives
            fn = len(expected_label - predicted_label)
            metrics[label]["fn"] += fn

        # Vérifier si les deux sont corrects (pour la tâche globale)
        if predicted_entities == expected_entities:
            correct_full += 1

    # Calculer Precision, Recall, F1
    results = {"model": model_path}

    for label in ["DEPART", "ARRIVEE"]:
        tp = metrics[label]["tp"]
        fp = metrics[label]["fp"]
        fn = metrics[label]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[f"{label}_precision"] = round(precision * 100, 2)
        results[f"{label}_recall"] = round(recall * 100, 2)
        results[f"{label}_f1"] = round(f1 * 100, 2)

    # Accuracy globale (les deux corrects)
    results["accuracy"] = round(correct_full / total * 100, 2)

    return results


def main():
    print("📊 Évaluation des modèles NER\n")

    # Charger les données de test
    test_data = load_test_data()
    print(f"Données de test: {len(test_data)} exemples\n")

    # Modèles à évaluer
    models = [
        "models/ner_blank/model-best",
        "models/ner_sm",
        "models/ner_md",
        "models/ner_lg",
        # CamemBERT nécessite une évaluation séparée
    ]

    results = []

    for model_path in models:
        if Path(model_path).exists():
            print(f"Évaluation: {model_path}")
            result = evaluate_model(model_path, test_data)
            if result:
                results.append(result)
                print(f"  → Accuracy: {result['accuracy']}%")
        else:
            print(f"⚠️ Modèle non trouvé: {model_path}")

    # Créer le tableau comparatif
    if results:
        df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("TABLEAU COMPARATIF DES PERFORMANCES")
        print("=" * 80)
        print(df.to_string(index=False))

        # Sauvegarder
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        df.to_csv(output_dir / "metrics_comparison.csv", index=False)
        print(f"\n✅ Résultats sauvegardés: results/metrics_comparison.csv")


if __name__ == "__main__":
    main()
```

---

# ÉTAPE 6 : CLI de Démonstration

### Script : `src/cli/main.py`

```python
#!/usr/bin/env python3
"""
CLI Travel Order Resolver
Pas de Jupyter - Gestion d'erreurs complète
"""
import argparse
import sys
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from inference.fuzzy_matcher import FuzzyStationMatcher


class TravelOrderResolver:
    """Résolveur d'itinéraires ferroviaires"""

    def __init__(self, model_path="models/ner_md"):
        print("🔄 Chargement du système...")

        try:
            self.nlp = spacy.load(model_path)
            print(f"  ✅ Modèle NER chargé: {model_path}")
        except Exception as e:
            print(f"  ❌ Erreur chargement modèle: {e}")
            print("  ⚠️ Utilisation du modèle par défaut fr_core_news_md")
            self.nlp = spacy.load("fr_core_news_md")

        try:
            self.matcher = FuzzyStationMatcher()
            print(f"  ✅ Fuzzy Matcher chargé")
        except Exception as e:
            print(f"  ⚠️ Fuzzy Matcher non disponible: {e}")
            self.matcher = None

        print("✅ Système prêt!\n")

    def extract(self, text):
        """Extrait départ et arrivée d'une phrase"""

        doc = self.nlp(text)

        departure = None
        arrival = None

        for ent in doc.ents:
            if ent.label_ == "DEPART" and not departure:
                departure = ent.text
            elif ent.label_ == "ARRIVEE" and not arrival:
                arrival = ent.text
            # Fallback sur LOC si nos labels custom ne sont pas trouvés
            elif ent.label_ == "LOC":
                if not departure:
                    departure = ent.text
                elif not arrival:
                    arrival = ent.text

        return departure, arrival

    def match_station(self, city):
        """Corrige le nom de ville vers une vraie gare SNCF"""
        if not city or not self.matcher:
            return city, 0

        return self.matcher.match(city)

    def resolve(self, text):
        """
        Pipeline complet: extraction + fuzzy matching

        Returns:
            dict avec departure, arrival, matched_departure, matched_arrival, confidence
        """
        # Extraction NER
        departure, arrival = self.extract(text)

        result = {
            "input": text,
            "departure_raw": departure,
            "arrival_raw": arrival,
            "departure": None,
            "arrival": None,
            "departure_score": 0,
            "arrival_score": 0,
            "valid": False
        }

        # Fuzzy matching
        if departure:
            matched, score = self.match_station(departure)
            result["departure"] = matched
            result["departure_score"] = score

        if arrival:
            matched, score = self.match_station(arrival)
            result["arrival"] = matched
            result["arrival_score"] = score

        # Validité
        result["valid"] = bool(result["departure"] and result["arrival"])

        return result


def interactive_mode(resolver):
    """Mode interactif"""

    print("=" * 60)
    print("🚂 TRAVEL ORDER RESOLVER - Mode Interactif")
    print("=" * 60)
    print("Tapez une phrase pour extraire l'itinéraire.")
    print("Tapez 'quit' ou 'q' pour quitter.\n")

    while True:
        try:
            text = input("📝 Votre phrase: ").strip()

            if text.lower() in ['quit', 'q', 'exit']:
                print("👋 Au revoir!")
                break

            if not text:
                print("⚠️ Veuillez entrer une phrase.\n")
                continue

            # Résoudre
            result = resolver.resolve(text)

            print("\n📊 Résultat:")
            print(f"  Départ brut:    {result['departure_raw'] or 'Non détecté'}")
            print(f"  Arrivée brute:  {result['arrival_raw'] or 'Non détectée'}")
            print(f"  Départ corrigé: {result['departure'] or 'INVALID'} (score: {result['departure_score']})")
            print(f"  Arrivée corrigée: {result['arrival'] or 'INVALID'} (score: {result['arrival_score']})")
            print(f"  Valide: {'✅ Oui' if result['valid'] else '❌ Non'}")
            print()

        except KeyboardInterrupt:
            print("\n👋 Interruption. Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}\n")


def batch_mode(resolver, input_file, output_file):
    """Mode batch - traite un fichier"""
    import json

    print(f"📂 Lecture de {input_file}...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
                texts = [d.get('text', d.get('phrase', '')) for d in data]
            else:
                texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lecture: {e}")
        sys.exit(1)

    print(f"📝 {len(texts)} phrases à traiter...")

    results = []
    for i, text in enumerate(texts):
        result = resolver.resolve(text)
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Progression: {i+1}/{len(texts)}")

    # Sauvegarder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Résultats sauvegardés: {output_file}")

    # Stats
    valid = sum(1 for r in results if r['valid'])
    print(f"\n📊 Statistiques:")
    print(f"  Total: {len(results)}")
    print(f"  Valides: {valid} ({valid/len(results)*100:.1f}%)")
    print(f"  Invalides: {len(results)-valid} ({(len(results)-valid)/len(results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Travel Order Resolver - Extraction d'itinéraires ferroviaires",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py                           # Mode interactif
  python main.py -t "De Paris à Lyon"      # Une seule phrase
  python main.py -i phrases.txt -o out.json # Mode batch
        """
    )

    parser.add_argument('-t', '--text', type=str, help='Phrase à analyser')
    parser.add_argument('-i', '--input', type=str, help='Fichier d\'entrée (batch mode)')
    parser.add_argument('-o', '--output', type=str, default='results.json', help='Fichier de sortie')
    parser.add_argument('-m', '--model', type=str, default='models/ner_md', help='Chemin du modèle NER')

    args = parser.parse_args()

    # Initialiser le resolver
    resolver = TravelOrderResolver(model_path=args.model)

    if args.text:
        # Mode une phrase
        result = resolver.resolve(args.text)
        print(f"\n📊 Résultat:")
        print(f"  Départ: {result['departure'] or 'INVALID'}")
        print(f"  Arrivée: {result['arrival'] or 'INVALID'}")
        print(f"  Valide: {'✅' if result['valid'] else '❌'}")

    elif args.input:
        # Mode batch
        batch_mode(resolver, args.input, args.output)

    else:
        # Mode interactif
        interactive_mode(resolver)


if __name__ == "__main__":
    main()
```

---

# ÉTAPE 7 : Requirements

### Fichier : `requirements.txt`

```
# NLP
spacy>=3.5.0
fr-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.5.0/fr_core_news_sm-3.5.0-py3-none-any.whl
fr-core-news-md @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.5.0/fr_core_news_md-3.5.0-py3-none-any.whl
fr-core-news-lg @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_lg-3.5.0/fr_core_news_lg-3.5.0-py3-none-any.whl

# Transformers (CamemBERT)
transformers>=4.30.0
datasets>=2.12.0
torch>=2.0.0
seqeval>=1.2.2

# Fuzzy matching
rapidfuzz>=3.0.0

# Graph
networkx>=3.0

# Évaluation
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0

# Visualisation
matplotlib>=3.7.0
```

---

# 📊 Tableau Comparatif Attendu

| Modèle | Dataset | Précision DEPART | Précision ARRIVEE | F1 Global | Temps/phrase | RAM |
|--------|---------|------------------|-------------------|-----------|--------------|-----|
| blank | 7000 | ~85% | ~85% | ~85% | ~5ms | 200MB |
| fr_core_news_sm | 7000 | ~90% | ~89% | ~89% | ~8ms | 400MB |
| fr_core_news_md | 7000 | ~93% | ~92% | ~92% | ~15ms | 600MB |
| fr_core_news_lg | 7000 | ~95% | ~94% | ~94% | ~25ms | 1GB |
| CamemBERT | 7000 | ~97% | ~96% | ~96% | ~50ms | 2GB |

---

# ✅ Checklist FU2

- [ ] Dataset avec split train/val/test documenté
- [ ] Modèle blank entraîné et évalué
- [ ] Modèle fr_core_news_sm entraîné et évalué
- [ ] Modèle fr_core_news_md entraîné et évalué
- [ ] Modèle fr_core_news_lg entraîné et évalué
- [ ] Modèle CamemBERT entraîné et évalué
- [ ] Métriques quantitatives (Precision, Recall, F1)
- [ ] Critique qualitative (performance vs ressources)
- [ ] CLI fonctionnel (pas de Jupyter)
- [ ] Fuzzy matching intégré
- [ ] (Bonus) Graph avec Dijkstra/A*

---

*Document créé le 22/01/2025 - Nouvelle approche légère pour FU2*
