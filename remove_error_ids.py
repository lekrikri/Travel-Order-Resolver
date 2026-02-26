#!/usr/bin/env python3
"""Supprime les 35 IDs d'erreur du fichier output pour les recalculer"""

error_ids = {583, 673, 1370, 1372, 1466, 2752, 2869, 2876, 3008, 3176, 3645, 3875,
             4053, 4713, 4741, 5038, 5041, 5157, 6008, 6014, 6129, 6268, 7210, 7214,
             7535, 8477, 8578, 8613, 8636, 8917, 8916, 9268, 9387, 9656, 9689}

input_file = './results/output_pipeline_10k.txt'
output_file = './results/output_pipeline_10k_cleaned.txt'

print("="*100)
print("SUPPRESSION DES 35 IDs D'ERREUR")
print("="*100)

removed_count = 0
kept_count = 0

with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                sentence_id = int(line.split(',')[0])

                if sentence_id in error_ids:
                    print(f"Suppression ID {sentence_id}")
                    removed_count += 1
                else:
                    f_out.write(line + '\n')
                    kept_count += 1

print(f"\n✅ Terminé:")
print(f"   Lignes supprimées: {removed_count}")
print(f"   Lignes conservées: {kept_count}")
print(f"\n📁 Fichier nettoyé: {output_file}")
print("\n🔄 Renommer maintenant:")
print(f"   mv {output_file} {input_file}")
