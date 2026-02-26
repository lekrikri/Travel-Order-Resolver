#!/usr/bin/env python3
"""Debug du problème Levallois-Perret / Clichy"""

import sys
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

sentences = [
    (5038, "Je cherche à réserver une place dans un train allant de <Dep>Levallois-Perret</Dep> vers <Arr>Clichy</Arr>"),
    (6014, "Faut que j'aille à <Arr>Clichy</Arr>, je pars de <Dep>Levallois-Perret</Dep>")
]

print("="*100)
print("DEBUG LEVALLOIS-PERRET / CLICHY")
print("="*100)

# Initialiser le pipeline
pipeline = TravelOrderPipeline(use_rag=True)

for sentence_id, sentence in sentences:
    print(f"\n[ID {sentence_id}]")
    print(f"Phrase: {sentence}")

    # Test extraction XML
    xml_dep, xml_arr = pipeline._extract_from_xml_tags(sentence)
    print(f"Extraction XML: {xml_dep} → {xml_arr}")

    # Traiter la phrase
    result = pipeline.process_sentence(sentence_id, sentence, metric="time")

    print(f"Résultat NLP:")
    print(f"  departure: {result['nlp_result']['departure']}")
    print(f"  arrival: {result['nlp_result']['arrival']}")

    if result["error"]:
        print(f"❌ ERREUR: {result['error']}")
    else:
        route = result["route_result"]
        print(f"Route trouvée:")
        print(f"  from_city: {route['from_city']}")
        print(f"  to_city: {route['to_city']}")
        print(f"  Stations: {' → '.join(route['stations'][:3])}")

        final_output = result["final_output"]
        parts = final_output.split(',')
        final_dep = parts[1] if len(parts) > 1 else ""
        final_arr = parts[-1] if len(parts) > 1 else ""
        print(f"Final output: {final_dep} → {final_arr}")

        print(f"Attendu: levallois-perret → clichy")
        print(f"Correct: {final_dep.lower() == 'levallois-perret' and final_arr.lower() == 'clichy'}")
