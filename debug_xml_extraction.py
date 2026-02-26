#!/usr/bin/env python3
"""Debug extraction XML"""

import re

def extract_from_xml_tags(sentence):
    dep_match = re.search(r'<Dep>(.*?)</Dep>', sentence, re.IGNORECASE)
    arr_match = re.search(r'<Arr>(.*?)</Arr>', sentence, re.IGNORECASE)

    departure = dep_match.group(1).strip() if dep_match else None
    arrival = arr_match.group(1).strip() if arr_match else None

    return (departure, arrival)

# Test sur les phrases problématiques
test_phrases = {
    5038: "Je cherche à réserver une place dans un train allant de <Dep>Levallois-Perret</Dep> vers <Arr>Clichy</Arr>",
    5041: "Un billet pour <Arr>Chambéry</Arr> dans une semaine, départ <Dep>Vannes</Dep>",
    6014: "Faut que j'aille à <Arr>Clichy</Arr>, je pars de <Dep>Levallois-Perret</Dep>"
}

for id, sentence in test_phrases.items():
    dep, arr = extract_from_xml_tags(sentence)
    print(f"ID {id}:")
    print(f"  Phrase: {sentence}")
    print(f"  Tags XML: {dep} → {arr}")
    print()
