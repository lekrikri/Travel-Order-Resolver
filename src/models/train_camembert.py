"""
ÉTAPE 6 : Entraînement CamemBERT pour NER
==========================================
Modèle transformer français pour extraire DEPART/ARRIVEE.

Exécuter avec : python3 src/models/train_camembert.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "camembert-base"
OUTPUT_DIR = "models/ner_camembert"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Labels BIO pour le NER
LABEL_LIST = ["O", "B-DEPART", "I-DEPART", "B-ARRIVEE", "I-ARRIVEE"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

def charger_donnees(json_path):
    """Charge les données depuis un fichier JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convertir_en_bio(exemple, tokenizer):
    """
    Convertit un exemple au format BIO pour les tokens.

    Input:
        {
            "text": "Je veux aller de Paris a Lyon",
            "entities": [(17, 22, "DEPART"), (25, 29, "ARRIVEE")]
        }

    Output:
        tokens: ["Je", "veux", "aller", "de", "Paris", "a", "Lyon"]
        labels: ["O", "O", "O", "O", "B-DEPART", "O", "B-ARRIVEE"]
    """
    text = exemple["text"]
    entities = exemple["entities"]

    # Tokenizer
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Créer les labels (par défaut "O")
    labels = ["O"] * len(tokenized["input_ids"])

    # Pour chaque entité, trouver les tokens correspondants
    for start_char, end_char, label in entities:
        for idx, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
            if token_start is None or token_end is None:
                continue
            if token_start == 0 and token_end == 0:
                continue

            # Le token chevauche l'entité ?
            if token_start >= start_char and token_end <= end_char:
                if token_start == start_char:
                    labels[idx] = f"B-{label}"  # Début
                else:
                    labels[idx] = f"I-{label}"  # Suite

    # Convertir en IDs
    label_ids = [LABEL2ID.get(l, 0) for l in labels]

    # Pour les tokens spéciaux, mettre -100 (ignoré par la loss)
    for idx, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
        if token_start == 0 and token_end == 0:
            label_ids[idx] = -100

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": label_ids
    }


def preparer_dataset(data, tokenizer):
    """Prépare un dataset pour l'entraînement."""
    processed = []

    for exemple in data:
        try:
            item = convertir_en_bio(exemple, tokenizer)
            processed.append(item)
        except Exception as e:
            print(f"Erreur: {e}")
            continue

    return Dataset.from_dict({
        "input_ids": [p["input_ids"] for p in processed],
        "attention_mask": [p["attention_mask"] for p in processed],
        "labels": [p["labels"] for p in processed]
    })


# =============================================================================
# MÉTRIQUES
# =============================================================================

def compute_metrics(eval_pred):
    """Calcule les métriques d'évaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Convertir en labels texte
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_clean = []

        for pred, label in zip(pred_seq, label_seq):
            if label != -100:  # Ignorer les tokens spéciaux
                true_seq.append(ID2LABEL[label])
                pred_seq_clean.append(ID2LABEL[pred])

        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    # Calculer les métriques
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ENTRAINEMENT CAMEMBERT POUR NER")
    print("=" * 60)

    # Vérifier GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Charger le tokenizer
    print(f"\nChargement du tokenizer {MODEL_NAME}...")
    tokenizer = CamembertTokenizerFast.from_pretrained(MODEL_NAME)

    # 2. Charger les données
    print("Chargement des donnees...")
    train_data = charger_donnees("data/processed/train.json")
    val_data = charger_donnees("data/processed/val.json")

    print(f"  Train: {len(train_data)} exemples")
    print(f"  Val: {len(val_data)} exemples")

    # 3. Préparer les datasets
    print("\nPreparation des datasets...")
    train_dataset = preparer_dataset(train_data, tokenizer)
    val_dataset = preparer_dataset(val_data, tokenizer)

    print(f"  Train dataset: {len(train_dataset)} exemples")
    print(f"  Val dataset: {len(val_dataset)} exemples")

    # 4. Charger le modèle
    print(f"\nChargement du modele {MODEL_NAME}...")
    model = CamembertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 5. Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Mixed precision si GPU
        report_to="none",  # Désactiver wandb
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Entraîner !
    print("\n" + "=" * 60)
    print("DEBUT DE L'ENTRAINEMENT")
    print("=" * 60)

    trainer.train()

    # 8. Sauvegarder
    print("\nSauvegarde du modele...")
    trainer.save_model(f"{OUTPUT_DIR}/best")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")

    # 9. Évaluation finale
    print("\n" + "=" * 60)
    print("EVALUATION FINALE")
    print("=" * 60)

    results = trainer.evaluate()
    print(f"  Precision: {results['eval_precision']*100:.2f}%")
    print(f"  Recall:    {results['eval_recall']*100:.2f}%")
    print(f"  F1-Score:  {results['eval_f1']*100:.2f}%")

    print("\n" + "=" * 60)
    print("ENTRAINEMENT TERMINE!")
    print("=" * 60)
    print(f"\nModele sauvegarde: {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()
