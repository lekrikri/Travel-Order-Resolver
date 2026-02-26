#!/usr/bin/env python3
"""Vérifier les gares SNCF disponibles pour les villes problématiques"""
import pandas as pd

df = pd.read_csv('data/sncf/stations.csv')
print('=== Recherche de gares pour villes problématiques ===')
print()

search_terms = ['saint-maur', 'champigny', 'clichy', 'boulogne', 'asnières', 'asniere',
                'issy', 'rueil', 'monaco', 'levallois', 'neuilly', 'villeneuve']

for term in search_terms:
    matches = df[df['nom_gare'].str.lower().str.contains(term, na=False) |
                 df['commune'].str.lower().str.contains(term, na=False)]
    if len(matches) > 0:
        print(f'\n{term.upper()} ({len(matches)} résultats):')
        for _, row in matches.head(5).iterrows():
            suggestable = '✓' if row['is_suggestable'] else '✗'
            print(f'  {suggestable} {row["nom_gare"]:40s} (commune: {row["commune"]})')
