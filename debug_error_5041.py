#!/usr/bin/env python3
"""Debug de l'erreur 5041"""

import sys
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

sentence_id = 5041
sentence = "Un billet pour <Arr>Chambéry</Arr> dans une semaine, départ <Dep>Vannes</Dep>"

print("="*100)
print(f"DEBUG ERREUR {sentence_id}")
print("="*100)
print(f"Phrase: {sentence}")
print()

# Initialiser le pipeline
pipeline = TravelOrderPipeline(use_rag=True)

# Test extraction XML
xml_dep, xml_arr = pipeline._extract_from_xml_tags(sentence)
print(f"Extraction XML: {xml_dep} → {xml_arr}")

# Traiter la phrase
result = pipeline.process_sentence(sentence_id, sentence, metric="time")

print(f"\nRésultat NLP:")
print(f"  departure: {result['nlp_result']['departure']}")
print(f"  arrival: {result['nlp_result']['arrival']}")
print(f"  xml_extraction_used: {result['nlp_result'].get('xml_extraction_used', False)}")
print(f"  inversion_corrected: {result['nlp_result'].get('inversion_corrected', False)}")

if result["error"]:
    print(f"\n❌ ERREUR: {result['error']}")
else:
    final_output = result["final_output"]
    parts = final_output.split(',')
    final_dep = parts[1] if len(parts) > 1 else ""
    final_arr = parts[-1] if len(parts) > 1 else ""

    print(f"\nRésultat final:")
    print(f"  {final_dep} → {final_arr}")
    print(f"\nAttendu: vannes → chambéry")
    print(f"Correct: {final_dep.lower() == 'vannes' and final_arr.lower() == 'chambéry'}")
