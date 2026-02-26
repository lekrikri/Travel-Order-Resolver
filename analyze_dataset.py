#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de qualité du dataset de 100 phrases
"""

import csv
import re
from collections import Counter

print("="*70)
print("ANALYSE DE QUALITÉ - Dataset 100 phrases")
print("="*70)

# Charger le dataset
sentences = []
with open('./data/annotated/train_sentences_100.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row)

print(f"\nTotal phrases: {len(sentences)}")

# ============================================================================
# 1. VÉRIFIER LES ANNOTATIONS
# ============================================================================

print("\n" + "-"*70)
print("1. VALIDATION DES ANNOTATIONS")
print("-"*70)

errors = []
for i, row in enumerate(sentences, 1):
    sentence = row['sentence']
    departure = row['departure']
    arrival = row['arrival']

    # Vérifier présence des tags
    if '<Dep>' not in sentence or '</Dep>' not in sentence:
        errors.append(f"Ligne {i}: Manque tag <Dep>")
    if '<Arr>' not in sentence or '</Arr>' not in sentence:
        errors.append(f"Ligne {i}: Manque tag <Arr>")

    # Vérifier cohérence
    dep_match = re.search(r'<Dep>([^<]+)</Dep>', sentence)
    arr_match = re.search(r'<Arr>([^<]+)</Arr>', sentence)

    if dep_match and dep_match.group(1) != departure:
        errors.append(f"Ligne {i}: Départ incohérent '{dep_match.group(1)}' vs '{departure}'")
    if arr_match and arr_match.group(1) != arrival:
        errors.append(f"Ligne {i}: Arrivée incohérente '{arr_match.group(1)}' vs '{arrival}'")

if errors:
    print(f"X {len(errors)} erreurs trouvées:")
    for error in errors[:10]:
        print(f"  - {error}")
else:
    print("OK Toutes les annotations sont correctes!")

# ============================================================================
# 2. DIVERSITÉ DES FORMULATIONS
# ============================================================================

print("\n" + "-"*70)
print("2. DIVERSITÉ DES FORMULATIONS")
print("-"*70)

# Extraire les débuts de phrases
phrase_starts = []
for row in sentences:
    sentence = row['sentence']
    # Enlever les annotations pour analyser le texte pur
    clean = re.sub(r'<[^>]+>', '', sentence)
    # Prendre les 3 premiers mots
    words = clean.split()[:3]
    start = ' '.join(words)
    phrase_starts.append(start)

start_counter = Counter(phrase_starts)
print(f"\nFormulations les plus fréquentes:")
for start, count in start_counter.most_common(10):
    print(f"  - '{start}...': {count} fois ({count}%)")

# Variété
unique_starts = len(set(phrase_starts))
print(f"\nFormulations uniques: {unique_starts}/100 ({unique_starts}%)")

if unique_starts < 30:
    print("X FAIBLE diversité - beaucoup de répétitions")
elif unique_starts < 60:
    print("~ MOYENNE diversité")
else:
    print("OK BONNE diversité")

# ============================================================================
# 3. DIVERSITÉ DES VILLES
# ============================================================================

print("\n" + "-"*70)
print("3. DIVERSITÉ DES VILLES")
print("-"*70)

departures = [row['departure'] for row in sentences]
arrivals = [row['arrival'] for row in sentences]
all_cities = departures + arrivals

dep_counter = Counter(departures)
arr_counter = Counter(arrivals)
city_counter = Counter(all_cities)

print(f"\nVilles uniques:")
print(f"  - Total: {len(set(all_cities))}")
print(f"  - Départs: {len(set(departures))}")
print(f"  - Arrivées: {len(set(arrivals))}")

print(f"\nVilles les plus fréquentes (toutes positions):")
for city, count in city_counter.most_common(10):
    print(f"  - {city}: {count} fois")

# Vérifier les grandes villes
major_cities = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes',
                'Strasbourg', 'Montpellier', 'Bordeaux', 'Lille']

print(f"\nPrésence des grandes villes françaises:")
for city in major_cities:
    count = all_cities.count(city)
    if count == 0:
        print(f"  X {city}: absente")
    elif count < 3:
        print(f"  ~ {city}: {count} fois (peu)")
    else:
        print(f"  OK {city}: {count} fois")

# ============================================================================
# 4. RÉALISME GÉOGRAPHIQUE
# ============================================================================

print("\n" + "-"*70)
print("4. RÉALISME GÉOGRAPHIQUE")
print("-"*70)

# Villes problématiques
overseas = ['Les Abymes', 'Ajaccio']  # Outre-mer ou nécessitant ferry
problematic = []

for i, row in enumerate(sentences, 1):
    dep = row['departure']
    arr = row['arrival']

    if dep in overseas or arr in overseas:
        problematic.append((i, f"{dep} -> {arr}"))

if problematic:
    print(f"\nX {len(problematic)} trajets peu réalistes (outre-mer/ferry):")
    for line, route in problematic:
        print(f"  - Ligne {line}: {route}")
else:
    print("OK Tous les trajets sont géographiquement réalistes")

# ============================================================================
# 5. VARIÉTÉ LINGUISTIQUE
# ============================================================================

print("\n" + "-"*70)
print("5. VARIÉTÉ LINGUISTIQUE")
print("-"*70)

# Patterns à détecter
patterns = {
    'Questions': r'\?',
    'Avec contexte famille': r'famille|enfants|parents',
    'Avec contexte travail': r'travail|bureau|réunion|professionnel',
    'Avec contexte vacances': r'vacances|week-end|loisir',
    'Formel (je souhaite)': r'je souhaite|j\'aimerais',
    'Informel (je veux)': r'je veux|on veut',
    'Dates/temps': r'demain|aujourd\'hui|semaine|lundi|mardi',
    'Types de billets': r'aller simple|aller-retour|allerretour',
}

for pattern_name, regex in patterns.items():
    count = sum(1 for row in sentences if re.search(regex, row['sentence'], re.IGNORECASE))
    percentage = (count / len(sentences)) * 100
    print(f"  - {pattern_name}: {count} ({percentage:.0f}%)")

# ============================================================================
# 6. RECOMMANDATIONS
# ============================================================================

print("\n" + "="*70)
print("RECOMMANDATIONS D'AMÉLIORATION")
print("="*70)

recommendations = []

# Check 1: Diversité formulations
if unique_starts < 50:
    recommendations.append("Augmenter la diversité des formulations (actuellement: {}/{})".format(unique_starts, len(sentences)))

# Check 2: Grandes villes
missing_major = [city for city in major_cities if city not in all_cities]
if missing_major:
    recommendations.append(f"Ajouter les grandes villes manquantes: {', '.join(missing_major)}")

# Check 3: Villes outre-mer
if problematic:
    recommendations.append(f"Remplacer {len(problematic)} trajets outre-mer/ferry par des trains continentaux")

# Check 4: Questions
question_count = sum(1 for row in sentences if '?' in row['sentence'])
if question_count < 10:
    recommendations.append(f"Ajouter plus de questions (actuellement: {question_count}%)")

# Check 5: Dates
date_count = sum(1 for row in sentences if re.search(r'demain|aujourd\'hui|semaine|lundi', row['sentence'], re.IGNORECASE))
if date_count < 10:
    recommendations.append(f"Ajouter plus de références temporelles (actuellement: {date_count}%)")

if recommendations:
    print("\nPoints à améliorer:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("\nOK Dataset de très bonne qualité!")

# ============================================================================
# 7. SCORE GLOBAL
# ============================================================================

print("\n" + "="*70)
print("SCORE GLOBAL DE QUALITÉ")
print("="*70)

score = 0
max_score = 0

# Critère 1: Annotations (30 pts)
max_score += 30
if len(errors) == 0:
    score += 30
    print("OK Annotations: 30/30")
else:
    partial = max(0, 30 - len(errors) * 5)
    score += partial
    print(f"~ Annotations: {partial}/30 ({len(errors)} erreurs)")

# Critère 2: Diversité formulations (25 pts)
max_score += 25
diversity_score = int((unique_starts / 100) * 25)
score += diversity_score
status = "OK" if diversity_score >= 20 else "~"
print(f"{status} Diversité formulations: {diversity_score}/25")

# Critère 3: Couverture villes (20 pts)
max_score += 20
city_coverage = len(set(all_cities)) / 100  # Idéal = 100 villes uniques
coverage_score = int(city_coverage * 20)
score += coverage_score
status = "OK" if coverage_score >= 15 else "~"
print(f"{status} Couverture villes: {coverage_score}/20")

# Critère 4: Réalisme (15 pts)
max_score += 15
if len(problematic) == 0:
    score += 15
    print("OK Réalisme géographique: 15/15")
else:
    realism_score = max(0, 15 - len(problematic) * 3)
    score += realism_score
    print(f"~ Réalisme géographique: {realism_score}/15")

# Critère 5: Variété linguistique (10 pts)
max_score += 10
linguistic_variety = (question_count + date_count) / 2  # Moyenne
variety_score = min(10, int(linguistic_variety))
score += variety_score
status = "OK" if variety_score >= 7 else "~"
print(f"{status} Variété linguistique: {variety_score}/10")

print("\n" + "-"*70)
print(f"SCORE FINAL: {score}/{max_score} ({int(score/max_score*100)}%)")
print("-"*70)

if score >= 90:
    print("EXCELLENT dataset, prêt pour le prof!")
elif score >= 75:
    print("BON dataset, quelques améliorations possibles")
elif score >= 60:
    print("CORRECT dataset, améliorations recommandées")
else:
    print("DATASET À AMÉLIORER avant le follow-up")

print("="*70)
