#!/usr/bin/env python3
"""Identifier les 5 erreurs restantes"""

import sys
import pandas as pd
sys.path.insert(0, './src/2_qwen_chroma')
sys.path.insert(0, './src/3_pathfinder')

from complete_pipeline import TravelOrderPipeline

error_ids = [583, 673, 1370, 1372, 1466, 2752, 2869, 2876, 3008, 3176, 3645, 3875,
             4053, 4713, 4741, 5038, 5041, 5157, 6008, 6014, 6129, 6268, 7210, 7214,
             7535, 8477, 8578, 8613, 8636, 8917, 8916, 9268, 9387, 9656, 9689]

gt_df = pd.read_csv('./data/annotated/train_sentences_10k.csv').head(10000)

print("Initialisation...")
pipeline = TravelOrderPipeline(use_rag=True)

print("\n" + "="*100)
print("ERREURS RESTANTES")
print("="*100)

remaining = []

for error_id in error_ids:
    row = gt_df.iloc[error_id - 1]
    sentence = row['sentence']
    gt_dep = row['departure'].strip().lower()
    gt_arr = row['arrival'].strip().lower()

    result = pipeline.process_sentence(error_id, sentence, metric="time")

    if result["error"]:
        remaining.append({
            'id': error_id,
            'sentence': sentence,
            'error': result["error"]
        })
    else:
        final_output = result["final_output"]
        parts = final_output.split(',')
        final_dep = parts[1].strip().lower() if len(parts) > 1 else ""
        final_arr = parts[-1].strip().lower() if len(parts) > 1 else ""

        is_correct = (final_dep == gt_dep and final_arr == gt_arr)

        if not is_correct:
            remaining.append({
                'id': error_id,
                'sentence': sentence,
                'gt': f"{gt_dep} → {gt_arr}",
                'pred': f"{final_dep} → {final_arr}"
            })

print(f"\n{len(remaining)} ERREURS RESTANTES:\n")
for err in remaining:
    print(f"ID {err['id']}: {err['sentence'][:80]}")
    if 'error' in err:
        print(f"  ❌ ERREUR: {err['error']}")
    else:
        print(f"  GT:   {err['gt']}")
        print(f"  PRED: {err['pred']}")
    print()

print(f"\nTotal: {len(remaining)} / 35 erreurs restantes")
print(f"Taux de correction: {100 * (35 - len(remaining)) / 35:.1f}%")
