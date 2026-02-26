#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1.3 - Annotation: Générer et annoter 100 phrases de trains
Utilise Qwen2.5 pour générer automatiquement des phrases annotées
"""

import sys
import io
import csv
import re
import json

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("PHASE 1.3 - ANNOTATION : 100 phrases de trains")
print("="*70)

# ============================================================================
# ÉTAPE 1 : GÉNÉRER 100 PHRASES AVEC QWEN2.5 (optionnel)
# ============================================================================

print("\n[1/4] Génération de phrases avec Qwen2.5...")
print("       (Optionnel - sinon utilise template prédéfini)")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("       Chargement de Qwen2.5-3B-Instruct...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    print("       OK Modèle chargé")

    # Prompt pour générer des phrases de trains annotées
    prompt = """Génère 100 phrases françaises variées pour réserver des trains.

Format attendu (inline annotations):
Je veux aller de <Dep>Paris</Dep> à <Arr>Lyon</Arr>
Avec ma famille, on souhaite faire <Dep>Marseille</Dep>-<Arr>Nice</Arr>
Un billet pour <Arr>Bordeaux</Arr> depuis <Dep>Toulouse</Dep>

Règles:
- Varier les formulations (je veux, j'aimerais, réserver, etc.)
- Varier les villes françaises
- Inclure des contextes (famille, travail, vacances)
- Toujours annoter départ (<Dep>) et arrivée (<Arr>)
- Une phrase par ligne
- 100 phrases au total

Phrases:
"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("       Génération en cours (peut prendre 2-3 minutes)...")
    outputs = model.generate(**inputs, max_new_tokens=3000, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire les phrases du résultat
    generated_lines = response.split('\n')
    sentences = []
    for line in generated_lines:
        if '<Dep>' in line and '<Arr>' in line:
            sentences.append(line.strip())
            if len(sentences) >= 100:
                break

    print(f"       OK {len(sentences)} phrases générées avec Qwen2.5")
    use_qwen = True

except (ImportError, Exception) as e:
    print(f"       SKIP: {e}")
    print("       Utilisation de templates prédéfinis à la place")
    use_qwen = False
    sentences = []

# ============================================================================
# ÉTAPE 2 : CRÉER TEMPLATES SI QWEN NON DISPONIBLE
# ============================================================================

if not use_qwen or len(sentences) < 100:
    print("\n[2/4] Génération avec templates prédéfinis...")

    # Listes de villes françaises
    cities = [
        "Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes",
        "Strasbourg", "Montpellier", "Bordeaux", "Lille", "Rennes",
        "Reims", "Le Havre", "Saint-Étienne", "Toulon", "Grenoble",
        "Dijon", "Angers", "Nîmes", "Villeurbanne", "Le Mans",
        "Aix-en-Provence", "Clermont-Ferrand", "Brest", "Tours",
        "Amiens", "Limoges", "Annecy", "Perpignan", "Metz",
        "Besançon", "Boulogne-Billancourt", "Orléans", "Mulhouse",
        "Rouen", "Caen", "Nancy", "Argenteuil", "Saint-Denis",
        "Montreuil", "Roubaix", "Avignon", "Tourcoing", "Poitiers",
        "Nanterre", "Créteil", "Versailles", "Pau", "Courbevoie",
        "Vitry-sur-Seine", "Colombes", "Aulnay-sous-Bois", "Asnières-sur-Seine",
        "Rueil-Malmaison", "Antibes", "Saint-Maur-des-Fossés", "Champigny-sur-Marne",
        "La Rochelle", "Aubervilliers", "Calais", "Cannes", "Bourges",
        "Saint-Nazaire", "Colmar", "Quimper", "Valence", "Drancy",
        "Ajaccio", "Levallois-Perret", "Troyes", "Issy-les-Moulineaux",
        "Noisy-le-Grand", "Antony", "Villeneuve-d'Ascq", "Neuilly-sur-Seine",
        "Clichy", "Ivry-sur-Seine", "Chambéry", "Lorient", "Sarcelles",
        "Les Abymes", "Niort", "Pessac", "Montauban", "Vénissieux",
        "Cergy", "Beauvais", "Cholet", "Bayonne", "Vannes",
        "Laval", "Dunkerque", "Maisons-Alfort", "Blois", "Pantin",
        "Monaco", "Chamonix", "Lourdes", "Biarritz"
    ]

    # Templates variés
    templates = [
        "Je veux aller de <Dep>{dep}</Dep> à <Arr>{arr}</Arr>",
        "Un billet pour <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep> svp",
        "Réserver un train <Dep>{dep}</Dep>-<Arr>{arr}</Arr>",
        "Je souhaite partir de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",
        "Avec ma famille, on veut aller à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep>",
        "J'aimerais faire le trajet <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
        "Combien coûte un train de <Dep>{dep}</Dep> pour <Arr>{arr}</Arr>?",
        "Y a-t-il un TGV <Dep>{dep}</Dep> <Arr>{arr}</Arr>?",
        "Pour mes vacances, je pars de <Dep>{dep}</Dep> direction <Arr>{arr}</Arr>",
        "On doit aller depuis <Dep>{dep}</Dep> jusqu'à <Arr>{arr}</Arr>",
        "Départ <Dep>{dep}</Dep>, arrivée <Arr>{arr}</Arr> demain",
        "Besoin d'un train au départ de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",
        "Je rentre à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep>",
        "Pour le travail, je dois faire <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
        "Aller simple <Dep>{dep}</Dep> à <Arr>{arr}</Arr>",
        "Aller-retour <Dep>{dep}</Dep> <Arr>{arr}</Arr> ce week-end",
        "Un billet pour <Arr>{arr}</Arr>, je pars de <Dep>{dep}</Dep>",
        "Je vais à <Arr>{arr}</Arr> en partant de <Dep>{dep}</Dep>",
        "Trajet <Dep>{dep}</Dep> vers <Arr>{arr}</Arr> en TGV",
        "Direction <Arr>{arr}</Arr> au départ de <Dep>{dep}</Dep>",
    ]

    import random
    random.seed(42)  # Pour reproductibilité

    sentences = []
    used_combinations = set()

    while len(sentences) < 100:
        template = random.choice(templates)
        dep = random.choice(cities)
        arr = random.choice(cities)

        # Éviter départ = arrivée
        if dep == arr:
            continue

        # Éviter les doublons
        combination = (dep, arr)
        if combination in used_combinations:
            continue

        used_combinations.add(combination)
        sentence = template.format(dep=dep, arr=arr)
        sentences.append(sentence)

    print(f"       OK {len(sentences)} phrases générées avec templates")

# ============================================================================
# ÉTAPE 3 : VALIDER LES ANNOTATIONS
# ============================================================================

print(f"\n[3/4] Validation des annotations...")

valid_sentences = []
invalid_count = 0

for sentence in sentences:
    # Vérifier présence des tags
    if '<Dep>' not in sentence or '</Dep>' not in sentence:
        invalid_count += 1
        continue
    if '<Arr>' not in sentence or '</Arr>' not in sentence:
        invalid_count += 1
        continue

    # Extraire les entités
    dep_match = re.search(r'<Dep>([^<]+)</Dep>', sentence)
    arr_match = re.search(r'<Arr>([^<]+)</Arr>', sentence)

    if dep_match and arr_match:
        dep = dep_match.group(1).strip()
        arr = arr_match.group(1).strip()

        # Vérifier que ce ne sont pas des vides
        if dep and arr and dep != arr:
            valid_sentences.append(sentence)
        else:
            invalid_count += 1
    else:
        invalid_count += 1

print(f"       OK {len(valid_sentences)} phrases valides")
print(f"       X {invalid_count} phrases invalides ignorées")

# ============================================================================
# ÉTAPE 4 : SAUVEGARDER
# ============================================================================

print(f"\n[4/4] Sauvegarde des annotations...")

# Créer le dossier data/annotated s'il n'existe pas
import os
os.makedirs("./data/annotated", exist_ok=True)

# Sauvegarder en CSV
output_csv = "./data/annotated/train_sentences_100.csv"
with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'departure', 'arrival'])

    for sentence in valid_sentences[:100]:  # Limiter à 100
        dep_match = re.search(r'<Dep>([^<]+)</Dep>', sentence)
        arr_match = re.search(r'<Arr>([^<]+)</Arr>', sentence)

        if dep_match and arr_match:
            dep = dep_match.group(1).strip()
            arr = arr_match.group(1).strip()
            writer.writerow([sentence, dep, arr])

print(f"       OK Sauvegardé dans {output_csv}")

# Sauvegarder aussi en TXT avec annotations inline
output_txt = "./data/annotated/train_sentences_100.txt"
with open(output_txt, 'w', encoding='utf-8') as f:
    for sentence in valid_sentences[:100]:
        f.write(sentence + '\n')

print(f"       OK Sauvegardé dans {output_txt}")

# ============================================================================
# RÉSULTATS
# ============================================================================

print("\n" + "="*70)
print("RESULTATS - Phase 1.3 Annotation")
print("="*70)
print(f"Méthode:        {'Qwen2.5-3B-Instruct' if use_qwen else 'Templates prédéfinis'}")
print(f"Total généré:   {len(sentences)}")
print(f"Valide:         {len(valid_sentences)}")
print(f"Sauvegardé:     100 phrases")
print("-"*70)

print("\nExemples de phrases annotées:")
for i, sentence in enumerate(valid_sentences[:5]):
    print(f"  {i+1}. {sentence}")

print("\nFormat:")
print("  - Annotations inline: <Dep>ville</Dep> et <Arr>ville</Arr>")
print("  - CSV avec colonnes: sentence, departure, arrival")
print("  - TXT avec annotations brutes")

print("\n" + "="*70)
print("Statistiques:")

# Compter les villes les plus fréquentes
all_deps = []
all_arrs = []
for sentence in valid_sentences[:100]:
    dep_match = re.search(r'<Dep>([^<]+)</Dep>', sentence)
    arr_match = re.search(r'<Arr>([^<]+)</Arr>', sentence)
    if dep_match and arr_match:
        all_deps.append(dep_match.group(1).strip())
        all_arrs.append(arr_match.group(1).strip())

from collections import Counter
dep_counter = Counter(all_deps)
arr_counter = Counter(all_arrs)

print(f"\nVilles de départ les plus fréquentes:")
for city, count in dep_counter.most_common(5):
    print(f"  - {city}: {count} fois")

print(f"\nVilles d'arrivée les plus fréquentes:")
for city, count in arr_counter.most_common(5):
    print(f"  - {city}: {count} fois")

print(f"\nVilles uniques:")
print(f"  - Départs: {len(set(all_deps))}")
print(f"  - Arrivées: {len(set(all_arrs))}")

print("\n" + "="*70)
print("Conclusion Phase 1.3:")
print("- 100 phrases de trains annotées créées")
print("- Format compatible avec spaCy et Transformers")
print("- Dataset prêt pour entraînement/évaluation")
print("- Prochaine étape: Phase 2 - Qwen2.5 + ChromaDB")
print("="*70)
