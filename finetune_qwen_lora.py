#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning de Qwen2.5-3B-Instruct avec LoRA
Optimisé pour RTX 3060 (6GB VRAM)
"""

import sys
import io
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import os

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("FINE-TUNING QWEN2.5 AVEC QLORA (4-BIT)")
print("="*70)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./models/qwen2.5-finetuned-lora"
TRAIN_FILE = "./data/finetune/train.jsonl"
VAL_FILE = "./data/finetune/val.jsonl"

# Vérifier GPU
print(f"\n✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if not torch.cuda.is_available():
    print("⚠️  ATTENTION: Pas de GPU détecté, le training sera très lent!")

# ÉTAPE 1: Charger le modèle et le tokenizer
print("\n[1/6] Chargement du modèle et tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side='right'
)

# Ajouter un pad token si manquant
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuration de la quantization 4-bit (QLoRA)
print("\n[1.5/6] Configuration de la quantization 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization pour plus d'économie
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.float16  # Compute en FP16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print(f"✅ Modèle chargé en 4-bit: {MODEL_NAME}")
print(f"✅ Quantization: NF4 + double quant")

# ÉTAPE 2: Configuration LoRA
print("\n[2/6] Configuration LoRA...")

lora_config = LoraConfig(
    r=16,  # Rank des matrices LoRA
    lora_alpha=32,  # Scaling factor
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],  # Modules à adapter
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Préparer le modèle pour l'entraînement avec quantization
model = prepare_model_for_kbit_training(model)

# Appliquer LoRA
model = get_peft_model(model, lora_config)

# S'assurer que les paramètres LoRA ont requires_grad=True
model.print_trainable_parameters()

# Afficher les paramètres entraînables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA configuré:")
print(f"   Paramètres entraînables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"   Paramètres totaux: {total_params:,}")

# ÉTAPE 3: Charger les datasets
print("\n[3/6] Chargement des datasets...")
dataset = load_dataset(
    'json',
    data_files={
        'train': TRAIN_FILE,
        'validation': VAL_FILE
    }
)

print(f"✅ Train: {len(dataset['train'])} exemples")
print(f"✅ Validation: {len(dataset['validation'])} exemples")

# ÉTAPE 4: Préparation des données
print("\n[4/6] Tokenization des données...")

def format_instruction(example):
    """Format instruction + input + output pour le training"""
    text = f"""<|im_start|>system
{example['instruction']}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
    return text

def tokenize_function(examples):
    """Tokenize les exemples"""
    texts = [format_instruction(ex) for ex in [
        {'instruction': inst, 'input': inp, 'output': out}
        for inst, inp, out in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        )
    ]]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors=None
    )

    # Labels = input_ids pour causal LM
    tokenized['labels'] = tokenized['input_ids'].copy()

    return tokenized

# Tokenize datasets
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing"
)

print("✅ Tokenization terminée")

# ÉTAPE 5: Configuration du training
print("\n[5/6] Configuration du training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Epochs et batch sizes
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Plus grand grâce à la quantization 4-bit
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # Simule batch size de 8

    # Optimisation mémoire
    optim="adamw_torch",

    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,

    # Logging et sauvegarde
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,

    # Performance
    fp16=True,
    dataloader_num_workers=2,

    # Autres
    remove_unused_columns=False,
    report_to="none",  # Pas de wandb/tensorboard
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

print("✅ Arguments configurés:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size effectif: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, pas masked LM
)

# ÉTAPE 6: Lancer le training
print("\n[6/6] Lancement du fine-tuning...")
print("⏱️  Cela peut prendre plusieurs heures sur RTX 3060...")
print("="*70 + "\n")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

# Training
train_result = trainer.train()

# Sauvegarder le modèle final
print("\n" + "="*70)
print("Sauvegarde du modèle fine-tuné...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modèle sauvegardé: {OUTPUT_DIR}")

# Afficher les métriques finales
print("\n" + "="*70)
print("RÉSULTATS DU FINE-TUNING")
print("="*70)
print(f"\n📊 Train Loss: {train_result.training_loss:.4f}")
print(f"⏱️  Temps total: {train_result.metrics['train_runtime']:.1f}s")
print(f"🚀 Samples/sec: {train_result.metrics['train_samples_per_second']:.2f}")

# Évaluation finale
print("\nÉvaluation finale sur validation set...")
eval_results = trainer.evaluate()
print(f"📊 Validation Loss: {eval_results['eval_loss']:.4f}")

print("\n" + "="*70)
print("✅ FINE-TUNING TERMINÉ")
print("="*70)
print(f"\n💾 Modèle disponible dans: {OUTPUT_DIR}")
print("\n📝 Pour utiliser le modèle fine-tuné:")
print("   from peft import PeftModel")
print(f"   model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")
