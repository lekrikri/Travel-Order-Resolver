#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 - Génération de 10,000 phrases annotées
Dataset massif pour entraînement/évaluation
"""

import csv
import random
import sys
import io

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("PHASE 2 - GÉNÉRATION DE 10,000 PHRASES")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n[1/5] Préparation des templates et données...")

# Liste exhaustive de villes françaises
cities = [
    # Grandes métropoles
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
    "Levallois-Perret", "Troyes", "Issy-les-Moulineaux",
    "Noisy-le-Grand", "Antony", "Villeneuve-d'Ascq", "Neuilly-sur-Seine",
    "Clichy", "Ivry-sur-Seine", "Chambéry", "Lorient", "Sarcelles",
    "Niort", "Pessac", "Montauban", "Vénissieux",
    "Cergy", "Beauvais", "Cholet", "Bayonne", "Vannes",
    "Laval", "Dunkerque", "Maisons-Alfort", "Blois", "Pantin",
    "Monaco", "Chamonix", "Lourdes", "Biarritz", "Belfort",
    "Arras", "Charleville-Mézières", "La Roche-sur-Yon", "Maubeuge",
    "Épinal", "Thionville", "Évry", "Chartres", "Cherbourg",
    "Saint-Quentin", "Hyères", "Sète", "Fréjus", "Albi",
    "Carcassonne", "Tarbes", "Angoulême", "Périgueux", "Agen",
    "Mont-de-Marsan", "Dax", "Brive-la-Gaillarde", "Tulle", "Aurillac",
    "Le Puy-en-Velay", "Rodez", "Cahors", "Mende", "Gap",
    "Digne-les-Bains", "Briançon", "Embrun", "Sisteron",
]

# Jours de la semaine
days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# Moments de la journée
time_periods = [
    "matin", "après-midi", "soir", "en début de matinée", "en fin d'après-midi",
    "en soirée", "dans la matinée", "dans l'après-midi", "ce matin", "cet après-midi",
    "ce soir", "demain matin", "demain après-midi", "demain soir"
]

# Références temporelles
time_refs = [
    "demain", "après-demain", "ce week-end", "le week-end prochain",
    "la semaine prochaine", "le mois prochain", "dans 2 jours", "dans une semaine",
    "vendredi prochain", "samedi prochain", "dimanche prochain",
    "en début de semaine", "en fin de semaine", "aujourd'hui", "tout de suite",
    "cet après-midi", "ce soir", "cette semaine"
]

# Contextes
contexts = [
    "Pour le travail", "Pour mes vacances", "Pour un week-end",
    "Pour une réunion", "Pour rendre visite à ma famille",
    "Pour un mariage", "Pour un enterrement", "Pour des études",
    "Pour un entretien", "Pour un rendez-vous", "Avec ma famille",
    "Avec mes enfants", "Avec mes parents", "Avec des amis",
    "Seul", "En groupe", "Pour affaires", "Pour le plaisir"
]

# Templates TRÈS variés (100+ templates)
templates = [
    # Basiques
    "Je veux aller de <Dep>{dep}</Dep> à <Arr>{arr}</Arr>",
    "Un billet pour <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep> svp",
    "Réserver un train <Dep>{dep}</Dep>-<Arr>{arr}</Arr>",
    "Je souhaite partir de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",
    "J'aimerais faire le trajet <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Combien coûte un train de <Dep>{dep}</Dep> pour <Arr>{arr}</Arr> ?",
    "Y a-t-il un TGV <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Je dois aller depuis <Dep>{dep}</Dep> jusqu'à <Arr>{arr}</Arr>",
    "Départ <Dep>{dep}</Dep>, arrivée <Arr>{arr}</Arr>",
    "Besoin d'un train au départ de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",
    "Je rentre à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep>",
    "Aller simple <Dep>{dep}</Dep> à <Arr>{arr}</Arr>",
    "Aller-retour <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Un billet pour <Arr>{arr}</Arr>, je pars de <Dep>{dep}</Dep>",
    "Je vais à <Arr>{arr}</Arr> en partant de <Dep>{dep}</Dep>",
    "Trajet <Dep>{dep}</Dep> vers <Arr>{arr}</Arr> en TGV",
    "Direction <Arr>{arr}</Arr> au départ de <Dep>{dep}</Dep>",
    "On doit aller depuis <Dep>{dep}</Dep> jusqu'à <Arr>{arr}</Arr>",

    # Avec contexte
    "{context}, je veux aller de <Dep>{dep}</Dep> à <Arr>{arr}</Arr>",
    "{context}, on part de <Dep>{dep}</Dep> pour <Arr>{arr}</Arr>",
    "{context}, un billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> svp",
    "{context}, je dois faire <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "{context}, on veut aller à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep>",

    # Avec dates
    "Train <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour {time_ref}",
    "Un billet pour <Arr>{arr}</Arr> {time_ref}, départ <Dep>{dep}</Dep>",
    "Je pars {day} de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",
    "Réservation <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour {time_ref}",
    "Départ <Dep>{dep}</Dep> direction <Arr>{arr}</Arr> {time_ref}",
    "Billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour {time_ref}",
    "Voyage <Dep>{dep}</Dep> <Arr>{arr}</Arr> prévu pour {day}",
    "Aller à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep> {time_period}",

    # Questions variées
    "Est-ce qu'il y a des trains <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Un TGV direct de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>, c'est possible ?",
    "Quelle heure le prochain train <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Horaires des trains au départ de <Dep>{dep}</Dep> pour <Arr>{arr}</Arr>",
    "Train de nuit <Dep>{dep}</Dep> <Arr>{arr}</Arr> disponible ?",
    "C'est combien un billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Il reste des places <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Quel est le prix d'un train <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "À quelle heure part le train <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Combien de temps pour aller de <Dep>{dep}</Dep> à <Arr>{arr}</Arr> ?",

    # Informel
    "Cherche un billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> pour moi",
    "Faut que j'aille à <Arr>{arr}</Arr>, je pars de <Dep>{dep}</Dep>",
    "Train de <Dep>{dep}</Dep> à <Arr>{arr}</Arr>, s'il vous plaît",
    "Bonjour, je cherche <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "On part de <Dep>{dep}</Dep> pour rejoindre <Arr>{arr}</Arr>",
    "Je dois me rendre à <Arr>{arr}</Arr> en partant de <Dep>{dep}</Dep>",
    "Salut, un train <Dep>{dep}</Dep> <Arr>{arr}</Arr> svp",
    "Hello, je veux aller à <Arr>{arr}</Arr> depuis <Dep>{dep}</Dep>",

    # Spécifiques temps
    "Premier train du matin <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Dernier train <Dep>{dep}</Dep> <Arr>{arr}</Arr> ce soir",
    "Train <Dep>{dep}</Dep> <Arr>{arr}</Arr> en milieu de journée",
    "TGV <Dep>{dep}</Dep> <Arr>{arr}</Arr> avant midi",
    "Train <Dep>{dep}</Dep> <Arr>{arr}</Arr> en début d'après-midi",

    # Avec précisions
    "Billet 2ème classe <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Billet 1ère classe <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Train rapide <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "TGV direct <Dep>{dep}</Dep> <Arr>{arr}</Arr> sans arrêt",
    "Train avec correspondance <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Réservation groupe <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Billet famille <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Carte jeune <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Tarif senior <Dep>{dep}</Dep> <Arr>{arr}</Arr>",

    # Formulations longues
    "Je souhaiterais réserver un billet de train au départ de <Dep>{dep}</Dep> avec une arrivée à <Arr>{arr}</Arr>",
    "Pouvez-vous me trouver un train qui part de <Dep>{dep}</Dep> et qui arrive à <Arr>{arr}</Arr> ?",
    "J'ai besoin d'un billet pour voyager de <Dep>{dep}</Dep> jusqu'à <Arr>{arr}</Arr>",
    "Pourriez-vous me réserver un train <Dep>{dep}</Dep> <Arr>{arr}</Arr> ?",
    "Je cherche à réserver une place dans un train allant de <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",

    # Variants temps + contexte
    "{context}, train <Dep>{dep}</Dep> <Arr>{arr}</Arr> {time_ref}",
    "{context}, départ <Dep>{dep}</Dep> pour <Arr>{arr}</Arr> {day}",
    "Billet <Dep>{dep}</Dep> <Arr>{arr}</Arr> {time_ref} {context}",

    # Inversions
    "Depuis <Dep>{dep}</Dep>, je veux aller à <Arr>{arr}</Arr>",
    "Vers <Arr>{arr}</Arr>, au départ de <Dep>{dep}</Dep>",
    "À destination de <Arr>{arr}</Arr>, départ <Dep>{dep}</Dep>",
    "En partance de <Dep>{dep}</Dep>, direction <Arr>{arr}</Arr>",

    # Avec dates spécifiques
    "Départ <Dep>{dep}</Dep>, arrivée <Arr>{arr}</Arr> {day} {time_period}",
    "Réserver pour {day} {time_period} : <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",

    # Très courts
    "<Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "<Dep>{dep}</Dep> -> <Arr>{arr}</Arr>",
    "De <Dep>{dep}</Dep> vers <Arr>{arr}</Arr>",

    # Avec urgence
    "Urgent : train <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Besoin rapide d'un billet <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
    "Vite, un train <Dep>{dep}</Dep> <Arr>{arr}</Arr>",
]

print(f"   OK {len(templates)} templates prêts")
print(f"   OK {len(cities)} villes disponibles")
print(f"   OK {len(contexts)} contextes variés")

# ============================================================================
# GÉNÉRATION DES 10,000 PHRASES
# ============================================================================

print("\n[2/5] Génération de 10,000 phrases...")

random.seed(42)  # Pour reproductibilité
sentences = []
used_combinations = set()
attempts = 0
max_attempts = 50000  # Limite pour éviter boucle infinie

while len(sentences) < 10000 and attempts < max_attempts:
    attempts += 1

    # Choisir template et villes
    template = random.choice(templates)
    dep = random.choice(cities)
    arr = random.choice(cities)

    # Éviter départ = arrivée
    if dep == arr:
        continue

    # Créer la phrase
    sentence = template

    # Remplacer les variables
    if "{context}" in template:
        context = random.choice(contexts)
        sentence = sentence.replace("{context}", context)

    if "{time_ref}" in template:
        time_ref = random.choice(time_refs)
        sentence = sentence.replace("{time_ref}", time_ref)

    if "{day}" in template:
        day = random.choice(days)
        sentence = sentence.replace("{day}", day)

    if "{time_period}" in template:
        time_period = random.choice(time_periods)
        sentence = sentence.replace("{time_period}", time_period)

    # Remplacer dep/arr
    sentence = sentence.format(dep=dep, arr=arr)

    # Vérifier unicité (éviter doublons exacts)
    if sentence in [s['sentence'] for s in sentences]:
        continue

    sentences.append({
        'sentence': sentence,
        'departure': dep,
        'arrival': arr
    })

    if len(sentences) % 1000 == 0:
        print(f"   {len(sentences)}/10000 phrases générées...")

print(f"   OK {len(sentences)} phrases générées en {attempts} tentatives")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n[3/5] Validation des phrases...")

valid_count = 0
for row in sentences:
    sentence = row['sentence']
    if '<Dep>' in sentence and '</Dep>' in sentence and '<Arr>' in sentence and '</Arr>' in sentence:
        valid_count += 1

print(f"   OK {valid_count}/{len(sentences)} phrases valides ({valid_count*100//len(sentences)}%)")

# ============================================================================
# SAUVEGARDE
# ============================================================================

print("\n[4/5] Sauvegarde du dataset...")

import os
os.makedirs("./data/annotated", exist_ok=True)

# CSV
output_csv = "./data/annotated/train_sentences_10k.csv"
with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'departure', 'arrival'])
    for row in sentences:
        writer.writerow([row['sentence'], row['departure'], row['arrival']])

print(f"   OK Sauvegardé dans {output_csv}")

# TXT
output_txt = "./data/annotated/train_sentences_10k.txt"
with open(output_txt, 'w', encoding='utf-8') as f:
    for row in sentences:
        f.write(row['sentence'] + '\n')

print(f"   OK Sauvegardé dans {output_txt}")

# ============================================================================
# STATISTIQUES
# ============================================================================

print("\n[5/5] Statistiques du dataset...")

from collections import Counter

all_deps = [row['departure'] for row in sentences]
all_arrs = [row['arrival'] for row in sentences]
all_cities_used = all_deps + all_arrs

dep_counter = Counter(all_deps)
arr_counter = Counter(all_arrs)
city_counter = Counter(all_cities_used)

print("\n" + "="*70)
print("RÉSULTATS - Dataset 10,000 phrases")
print("="*70)
print(f"Total phrases:         {len(sentences):,}")
print(f"Phrases valides:       {valid_count:,} ({valid_count*100//len(sentences)}%)")
print(f"Villes uniques:        {len(set(all_cities_used))}")
print(f"  - Départs uniques:   {len(set(all_deps))}")
print(f"  - Arrivées uniques:  {len(set(all_arrs))}")
print("-"*70)

print("\nVilles les plus fréquentes:")
for city, count in city_counter.most_common(10):
    print(f"  - {city}: {count:,} fois")

print("\nDiversité:")
unique_sentences = len(set(row['sentence'] for row in sentences))
print(f"  - Phrases uniques: {unique_sentences:,}/{len(sentences):,} ({unique_sentences*100//len(sentences)}%)")

# Compter les patterns
question_count = sum(1 for row in sentences if '?' in row['sentence'])
date_count = sum(1 for row in sentences if any(word in row['sentence'].lower() for word in ['demain', 'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche', 'semaine']))
context_count = sum(1 for row in sentences if any(ctx.lower() in row['sentence'].lower() for ctx in ['travail', 'vacances', 'famille', 'réunion']))

print(f"\nVariété linguistique:")
print(f"  - Questions: {question_count:,} ({question_count*100//len(sentences)}%)")
print(f"  - Avec dates: {date_count:,} ({date_count*100//len(sentences)}%)")
print(f"  - Avec contexte: {context_count:,} ({context_count*100//len(sentences)}%)")

print("\n" + "="*70)
print("Exemples de phrases générées:")
print("="*70)
for i in range(10):
    print(f"{i+1}. {sentences[i]['sentence']}")

print("\n" + "="*70)
print("DATASET 10K TERMINÉ !")
print("="*70)
print(f"Fichiers créés:")
print(f"  - {output_csv}")
print(f"  - {output_txt}")
print(f"\nCe dataset est prêt pour:")
print(f"  - Entraînement de modèles NER")
print(f"  - Évaluation de systèmes d'extraction")
print(f"  - Tests de Qwen2.5 + ChromaDB")
print(f"  - Comparaison spaCy vs Transformers")
print("="*70)
