#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amélioration du dataset de 100 phrases
Corrige les problèmes identifiés par l'analyse
"""

import csv
import re
import random

print("="*70)
print("AMÉLIORATION DU DATASET - 100 phrases")
print("="*70)

# Charger le dataset actuel
sentences = []
with open('./data/annotated/train_sentences_100.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row)

print(f"\nDataset actuel: {len(sentences)} phrases")

# ============================================================================
# 1. CORRIGER LES TRAJETS IRRÉALISTES
# ============================================================================

print("\n[1/4] Correction des trajets irréalistes...")

# Trajets problématiques (lignes 46, 60, 69 = indices 45, 59, 68)
problematic_indices = {
    45: ("Chamonix", "Strasbourg", "Marseille", "Strasbourg"),  # Ligne 46
    59: ("Ajaccio", "Paris", "Nantes", "Paris"),  # Ligne 60
    68: ("Ajaccio", "Chambéry", "Marseille", "Lyon"),  # Ligne 69
}

corrections_made = 0
for idx, (old_dep, old_arr, new_dep, new_arr) in problematic_indices.items():
    if idx < len(sentences):
        row = sentences[idx]
        if row['departure'] == old_dep and row['arrival'] == old_arr:
            # Remplacer dans la phrase
            new_sentence = row['sentence'].replace(f"<Dep>{old_dep}</Dep>", f"<Dep>{new_dep}</Dep>")
            new_sentence = new_sentence.replace(f"<Arr>{old_arr}</Arr>", f"<Arr>{new_arr}</Arr>")

            sentences[idx] = {
                'sentence': new_sentence,
                'departure': new_dep,
                'arrival': new_arr
            }
            corrections_made += 1
            print(f"   Corrigé: {old_dep}->{old_arr} devient {new_dep}->{new_arr}")

print(f"   OK {corrections_made} trajets corrigés")

# ============================================================================
# 2. AJOUTER PLUS DE VARIÉTÉ DANS LES FORMULATIONS
# ============================================================================

print("\n[2/4] Amélioration de la diversité des formulations...")

# Templates avec plus de variété
new_templates = [
    # Formulations supplémentaires
    "Est-ce qu'il y a des trains <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Un TGV direct de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>, c'est possible ?",
    "Cherche un billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour moi",
    "Faut que j'aille à <Arr>{arr}</Arr>, je pars de <Dep>{dep}</Dep>",
    "Train de <Dep>{dep}</Dep> à <Arr>{arr}</Arr>, s'il vous plaît",
    "Bonjour, je cherche <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "On part de <Dep>{dep}</Dep> pour rejoindre <Arr>{arr}</Arr>",
    "Quelle heure le prochain train <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Horaires des trains au départ de <Dep>{dep}</Dep> pour <Arr>{arr}</Arr>",
    "Je dois me rendre à <Arr>{arr}</Arr> en partant de <Dep>{dep}</Dep>",

    # Avec références temporelles
    "Train <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour demain matin",
    "Un billet pour <Arr>{arr}</Arr> lundi prochain, départ <Dep>{dep}</Dep>",
    "Je pars jeudi de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",
    "Réservation <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour ce week-end",
    "Départ <Dep>{dep}</Dep> direction <Arr>{arr}</Arr> samedi matin",
    "Train de nuit <Dep>{dep}</Dep> <Arr>{arr}</Arr> disponible ?",
    "Voyage <Dep>{dep}</Dep> <Arr>{arr}</Arr> prévu pour mardi",
    "Billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour la semaine prochaine",
    "Aller à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep> en fin d'après-midi",
    "Premier train du matin <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
]

# Villes principales à utiliser
main_cities = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes",
    "Strasbourg", "Montpellier", "Bordeaux", "Lille", "Rennes",
    "Reims", "Toulon", "Grenoble", "Dijon", "Angers"
]

# Remplacer les phrases les plus répétitives
# Identifier les phrases qui commencent pareil
phrase_starts = {}
for i, row in enumerate(sentences):
    sentence = row['sentence']
    clean = re.sub(r'<[^>]+>', '', sentence)
    start = ' '.join(clean.split()[:3])

    if start not in phrase_starts:
        phrase_starts[start] = []
    phrase_starts[start].append(i)

# Remplacer les doublons (garder seulement le premier de chaque type)
replacements = 0
used_combinations = set()

for start, indices in phrase_starts.items():
    if len(indices) > 2:  # Si plus de 2 phrases avec le même début
        # Garder les 2 premières, remplacer les autres
        for idx in indices[2:]:
            if replacements >= len(new_templates):
                break

            # Choisir une nouvelle combinaison de villes
            while True:
                dep = random.choice(main_cities)
                arr = random.choice(main_cities)
                if dep != arr and (dep, arr) not in used_combinations:
                    used_combinations.add((dep, arr))
                    break

            # Créer nouvelle phrase
            template = new_templates[replacements % len(new_templates)]
            new_sentence = template.format(dep=dep, arr=arr)

            sentences[idx] = {
                'sentence': new_sentence,
                'departure': dep,
                'arrival': arr
            }
            replacements += 1

print(f"   OK {replacements} phrases reformulées pour plus de diversité")

# ============================================================================
# 3. ASSURER PRÉSENCE DES GRANDES VILLES
# ============================================================================

print("\n[3/4] Vérification des grandes villes...")

# Compter les villes actuelles
all_deps = [row['departure'] for row in sentences]
all_arrs = [row['arrival'] for row in sentences]
all_cities = all_deps + all_arrs

# Villes qui doivent apparaître au moins 3 fois
required_cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Bordeaux", "Lille"]

cities_added = 0
for city in required_cities:
    count = all_cities.count(city)
    if count < 3:
        needed = 3 - count
        print(f"   Ajout de {needed} occurrences pour {city}")

        # Trouver des phrases à modifier
        for i in range(len(sentences)):
            if cities_added >= 10:  # Limite
                break

            # Remplacer aléatoirement départ ou arrivée
            if random.choice([True, False]):
                # Remplacer départ
                old_dep = sentences[i]['departure']
                if old_dep not in required_cities:
                    new_sentence = sentences[i]['sentence'].replace(
                        f"<Dep>{old_dep}</Dep>",
                        f"<Dep>{city}</Dep>"
                    )
                    sentences[i]['sentence'] = new_sentence
                    sentences[i]['departure'] = city
                    cities_added += 1
                    needed -= 1
            else:
                # Remplacer arrivée
                old_arr = sentences[i]['arrival']
                if old_arr not in required_cities:
                    new_sentence = sentences[i]['sentence'].replace(
                        f"<Arr>{old_arr}</Arr>",
                        f"<Arr>{city}</Arr>"
                    )
                    sentences[i]['sentence'] = new_sentence
                    sentences[i]['arrival'] = city
                    cities_added += 1
                    needed -= 1

            if needed <= 0:
                break

print(f"   OK {cities_added} modifications pour équilibrer les grandes villes")

# ============================================================================
# 4. SAUVEGARDER DATASET AMÉLIORÉ
# ============================================================================

print("\n[4/4] Sauvegarde du dataset amélioré...")

# Sauvegarder
output_csv = './data/annotated/train_sentences_100_improved.csv'
with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'departure', 'arrival'])

    for row in sentences:
        writer.writerow([row['sentence'], row['departure'], row['arrival']])

print(f"   OK Sauvegardé dans {output_csv}")

# Aussi en TXT
output_txt = './data/annotated/train_sentences_100_improved.txt'
with open(output_txt, 'w', encoding='utf-8') as f:
    for row in sentences:
        f.write(row['sentence'] + '\n')

print(f"   OK Sauvegardé dans {output_txt}")

# Remplacer l'original par la version améliorée
import shutil
shutil.copy('./data/annotated/train_sentences_100.csv', './data/annotated/train_sentences_100_backup.csv')
shutil.copy(output_csv, './data/annotated/train_sentences_100.csv')
shutil.copy(output_txt, './data/annotated/train_sentences_100.txt')

print("   OK Version originale sauvegardée en backup")
print("   OK Dataset amélioré est maintenant la version principale")

print("\n" + "="*70)
print("AMÉLIORATIONS TERMINÉES")
print("="*70)
print(f"\nModifications apportées:")
print(f"  - {corrections_made} trajets irréalistes corrigés")
print(f"  - {replacements} phrases reformulées")
print(f"  - {cities_added} ajustements de grandes villes")
print(f"\nFichiers:")
print(f"  - train_sentences_100.csv (version améliorée)")
print(f"  - train_sentences_100_backup.csv (version originale)")
print("="*70)
