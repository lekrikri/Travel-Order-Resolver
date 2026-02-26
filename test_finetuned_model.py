#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du modèle Qwen2.5 fine-tuné avec LoRA
Évaluation sur 100 phrases de test
"""

import sys
import io
import csv
import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def flush_print(msg):
    """Print avec flush immédiat"""
    print(msg, flush=True)

flush_print("="*70)
flush_print("TEST MODÈLE QWEN2.5 FINE-TUNÉ")
flush_print("="*70)

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL = "./models/qwen2.5-finetuned-lora"

# Vérifier GPU
flush_print(f"\n✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ÉTAPE 1: Charger le modèle base
flush_print("\n[1/4] Chargement du modèle base...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

flush_print("✅ Modèle base chargé")

# ÉTAPE 2: Charger les adaptateurs LoRA
flush_print("\n[2/4] Chargement des adaptateurs LoRA...")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
model.eval()

flush_print("✅ Modèle fine-tuné chargé")

# ÉTAPE 3: Charger dataset de test
flush_print("\n[3/4] Chargement du dataset de test...")
test_sentences = []
with open('./data/annotated/train_sentences_10k.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        clean_sentence = re.sub(r'<[^>]+>', '', row['sentence'])
        test_sentences.append({
            'sentence': clean_sentence,
            'gold_dep': row['departure'],
            'gold_arr': row['arrival']
        })

# Prendre 100 phrases de test (différentes du training)
test_sentences = test_sentences[9000:9100]  # Les 100 dernières du 10k
flush_print(f"✅ {len(test_sentences)} phrases chargées")

# ÉTAPE 4: Extraction avec modèle fine-tuné
flush_print("\n[4/4] Extraction avec modèle fine-tuné...")
flush_print("   (Progression affichée toutes les 10 phrases)")

INSTRUCTION = """Tu es un assistant spécialisé dans l'extraction d'informations de voyage en train.

Ta tâche : extraire la ville de DÉPART et la ville d'ARRIVÉE d'une phrase en français.

Format de sortie (JSON strict) :
{"departure": "ville_de_depart", "arrival": "ville_d_arrivee"}

Si impossible à extraire, retourne :
{"departure": null, "arrival": null}"""

def extract_cities_finetuned(sentence):
    """Extrait les villes avec le modèle fine-tuné"""
    prompt = f"""<|im_start|>system
{INSTRUCTION}<|im_end|>
<|im_start|>user
{sentence}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire la partie après "assistant"
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()

    # Parser le JSON
    try:
        # Trouver le JSON dans la réponse
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {"departure": None, "arrival": None}
    except:
        return {"departure": None, "arrival": None}

results = []
start_time = time.time()

for i, test in enumerate(test_sentences):
    result = extract_cities_finetuned(test['sentence'])

    results.append({
        'sentence': test['sentence'],
        'gold_dep': test['gold_dep'],
        'gold_arr': test['gold_arr'],
        'pred_dep': result.get('departure'),
        'pred_arr': result.get('arrival')
    })

    # Afficher progression
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (len(test_sentences) - i - 1) / rate
        flush_print(f"   {i+1}/{len(test_sentences)} - {elapsed:.1f}s écoulées, ~{remaining:.0f}s restantes")

total_time = time.time() - start_time
flush_print(f"\n✅ Extraction terminée en {total_time:.1f}s ({total_time/len(test_sentences):.2f}s/phrase)")

# CALCUL DES MÉTRIQUES
flush_print("\nCalcul des métriques...")

correct_both = 0
correct_dep = 0
correct_arr = 0

for r in results:
    if r['pred_dep'] == r['gold_dep']:
        correct_dep += 1
    if r['pred_arr'] == r['gold_arr']:
        correct_arr += 1
    if r['pred_dep'] == r['gold_dep'] and r['pred_arr'] == r['gold_arr']:
        correct_both += 1

accuracy = correct_both / len(results) * 100
precision_dep = correct_dep / len(results) * 100
precision_arr = correct_arr / len(results) * 100

flush_print("\n" + "="*70)
flush_print("RÉSULTATS MODÈLE FINE-TUNÉ")
flush_print("="*70)
flush_print(f"\n📊 Phrases testées: {len(results)}")
flush_print(f"⏱️  Temps total: {total_time:.1f}s")
flush_print(f"🚀 Vitesse: {total_time/len(results):.2f}s/phrase")
flush_print(f"\n✅ Accuracy (les 2 correctes): {accuracy:.1f}%")
flush_print(f"🏙️  Départ correct: {precision_dep:.1f}%")
flush_print(f"🏙️  Arrivée correcte: {precision_arr:.1f}%")

# Comparaison avec résultats précédents
flush_print("\n" + "="*70)
flush_print("COMPARAISON AVEC APPROCHES PRÉCÉDENTES")
flush_print("="*70)
flush_print(f"\n📊 spaCy baseline:      32.2%")
flush_print(f"📊 Qwen2.5 seul:        75.0%")
flush_print(f"📊 Qwen2.5 + RAG:       93.0%")
flush_print(f"🏆 Qwen2.5 fine-tuné:   {accuracy:.1f}%")

if accuracy > 93.0:
    improvement = accuracy - 93.0
    flush_print(f"\n🎉 AMÉLIORATION: +{improvement:.1f} points vs RAG!")
elif accuracy == 93.0:
    flush_print(f"\n✅ Performance équivalente au RAG")
else:
    degradation = 93.0 - accuracy
    flush_print(f"\n⚠️  Performance inférieure au RAG: -{degradation:.1f} points")

# Sauvegarder résultats
output_file = './results/finetuned_100_results.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sentence', 'gold_dep', 'gold_arr', 'pred_dep', 'pred_arr'])
    writer.writeheader()
    writer.writerows(results)

flush_print(f"\n💾 Résultats sauvegardés: {output_file}")
flush_print("="*70)
