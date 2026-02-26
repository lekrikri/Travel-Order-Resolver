#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération d'un rapport PDF professionnel avec graphiques
spaCy vs Qwen2.5 seul vs Qwen2.5 + ChromaDB RAG
"""

import sys
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("GÉNÉRATION DU RAPPORT PDF PROFESSIONNEL")
print("="*70)

# Configuration matplotlib pour meilleur rendu
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

# Données
approaches = ['spaCy\n(baseline)', 'Qwen2.5\nseul', 'Qwen2.5\n+ ChromaDB RAG']
accuracy = [32.2, 75.0, 93.0]
departure_correct = [43.8, 75.0, 93.0]
arrival_correct = [46.3, 75.0, 93.0]
time_per_phrase = [0.5, 6.57, 6.66]  # secondes

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Créer le PDF
pdf_filename = './RAPPORT_PROFESSIONNEL.pdf'
with PdfPages(pdf_filename) as pdf:

    # ========================================================================
    # PAGE 1: PAGE DE GARDE
    # ========================================================================
    print("\n[1/6] Génération de la page de garde...")

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 format
    fig.patch.set_facecolor('white')

    # Titre principal
    plt.text(0.5, 0.75, 'RAPPORT TECHNIQUE',
             ha='center', va='center', fontsize=32, fontweight='bold',
             color='#2C3E50')

    plt.text(0.5, 0.68, 'Travel Order Resolver NLP',
             ha='center', va='center', fontsize=20,
             color='#34495E')

    # Ligne de séparation
    plt.plot([0.2, 0.8], [0.62, 0.62], 'k-', linewidth=2, color='#3498DB')

    # Sous-titre
    plt.text(0.5, 0.55, 'Extraction de Villes pour Réservations de Train',
             ha='center', va='center', fontsize=14,
             color='#7F8C8D')

    plt.text(0.5, 0.50, 'Comparaison : spaCy vs Qwen2.5 vs Qwen2.5+ChromaDB RAG',
             ha='center', va='center', fontsize=12,
             color='#7F8C8D', style='italic')

    # Résultats clés
    plt.text(0.5, 0.38, 'RÉSULTATS CLÉS',
             ha='center', va='center', fontsize=16, fontweight='bold',
             color='#2C3E50')

    plt.text(0.5, 0.32, '🏆 Accuracy: 93.0% avec Qwen2.5 + ChromaDB RAG',
             ha='center', va='center', fontsize=13,
             color='#27AE60', fontweight='bold')

    plt.text(0.5, 0.27, '📈 Amélioration: +2.9x vs baseline spaCy',
             ha='center', va='center', fontsize=12,
             color='#2980B9')

    plt.text(0.5, 0.22, '🚀 Impact RAG: +18 points (75% → 93%)',
             ha='center', va='center', fontsize=12,
             color='#8E44AD')

    # Informations du projet
    plt.text(0.5, 0.12, f'Projet: T-AIA-911-PAR_14',
             ha='center', va='center', fontsize=10,
             color='#95A5A6')

    plt.text(0.5, 0.08, f'Date: {datetime.now().strftime("%d/%m/%Y")}',
             ha='center', va='center', fontsize=10,
             color='#95A5A6')

    plt.text(0.5, 0.04, 'Généré avec Qwen2.5 + ChromaDB + PyTorch CUDA',
             ha='center', va='center', fontsize=9,
             color='#BDC3C7', style='italic')

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 2: GRAPHIQUE COMPARAISON ACCURACY
    # ========================================================================
    print("[2/6] Génération du graphique d'accuracy...")

    fig, ax = plt.subplots(figsize=(11, 8))

    # Graphique en barres
    bars = ax.bar(approaches, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Ajouter les valeurs au-dessus des barres
    for i, (bar, val) in enumerate(zip(bars, accuracy)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Ajouter amélioration
        if i > 0:
            improvement = val - accuracy[0]
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'+{improvement:.1f} pts\n({val/accuracy[0]:.1f}x)',
                    ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comparaison de l\'Accuracy - Extraction de Villes',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Ligne de référence à 90%
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Objectif 90%')
    ax.legend(fontsize=11)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 3: GRAPHIQUE DÉPART vs ARRIVÉE
    # ========================================================================
    print("[3/6] Génération du graphique départ/arrivée...")

    fig, ax = plt.subplots(figsize=(11, 8))

    x = np.arange(len(approaches))
    width = 0.35

    bars1 = ax.bar(x - width/2, departure_correct, width, label='Départ Correct',
                   color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, arrival_correct, width, label='Arrivée Correcte',
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Précision (%)', fontsize=14, fontweight='bold')
    ax.set_title('Précision par Ville - Départ vs Arrivée',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 4: GRAPHIQUE ÉVOLUTION PROGRESSIVE
    # ========================================================================
    print("[4/6] Génération du graphique d'évolution...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique 1: Courbe d'évolution
    ax1.plot(range(3), accuracy, marker='o', markersize=12, linewidth=3,
             color='#2ECC71', label='Accuracy')
    ax1.fill_between(range(3), 0, accuracy, alpha=0.3, color='#2ECC71')

    for i, val in enumerate(accuracy):
        ax1.annotate(f'{val}%',
                    xy=(i, val),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2ECC71', linewidth=2))

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['Baseline\nspaCy', 'Qwen2.5\nseul', 'Qwen2.5\n+ RAG'])
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Évolution Progressive de l\'Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)

    # Graphique 2: Amélioration relative
    improvements = [0, accuracy[1] - accuracy[0], accuracy[2] - accuracy[0]]
    colors_imp = ['#95A5A6', '#F39C12', '#27AE60']

    bars = ax2.bar(range(3), improvements, color=colors_imp, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, improvements):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'+{val:.1f} pts',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Baseline', '+Qwen2.5', '+ChromaDB\nRAG'])
    ax2.set_ylabel('Amélioration (points)', fontsize=13, fontweight='bold')
    ax2.set_title('Gains Cumulatifs', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 70)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 5: ANALYSE PERFORMANCE
    # ========================================================================
    print("[5/6] Génération de l'analyse de performance...")

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Graphique 1: Temps par phrase
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(approaches, time_per_phrase, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, time_per_phrase):
        ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}s',
                va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Temps (secondes/phrase)', fontsize=11, fontweight='bold')
    ax1.set_title('Vitesse d\'Inférence', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Graphique 2: Ratio Accuracy/Temps
    ax2 = fig.add_subplot(gs[0, 1])
    efficiency = [acc/time if time > 0 else 0 for acc, time in zip(accuracy, time_per_phrase)]
    bars = ax2.bar(range(3), efficiency, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, efficiency):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['spaCy', 'Qwen2.5', 'Qwen2.5\n+RAG'], fontsize=9)
    ax2.set_ylabel('Accuracy / Temps', fontsize=11, fontweight='bold')
    ax2.set_title('Efficacité (Accuracy/Temps)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Tableau récapitulatif
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')

    table_data = [
        ['Approche', 'Accuracy', 'Départ', 'Arrivée', 'Temps/phrase', 'Amélioration'],
        ['spaCy (baseline)', '32.2%', '43.8%', '46.3%', '0.5s', '-'],
        ['Qwen2.5 seul', '75.0%', '75.0%', '75.0%', '6.6s', '+2.3x'],
        ['Qwen2.5 + ChromaDB RAG', '93.0%', '93.0%', '93.0%', '6.7s', '+2.9x'],
    ]

    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Mise en forme du tableau
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i == 3:  # Meilleur résultat en vert
                table[(i, j)].set_facecolor('#D5F4E6')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')

    ax3.set_title('Tableau Récapitulatif des Performances',
                 fontsize=14, fontweight='bold', pad=20)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # PAGE 6: CONCLUSIONS ET RECOMMANDATIONS
    # ========================================================================
    print("[6/6] Génération de la page de conclusions...")

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    plt.text(0.5, 0.95, 'CONCLUSIONS & RECOMMANDATIONS',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color='#2C3E50')

    plt.plot([0.1, 0.9], [0.92, 0.92], 'k-', linewidth=2, color='#3498DB')

    # Résultats clés
    y_pos = 0.88
    plt.text(0.1, y_pos, '📊 RÉSULTATS CLÉS', fontsize=14, fontweight='bold', color='#2C3E50')
    y_pos -= 0.05

    conclusions = [
        '• Qwen2.5 + ChromaDB RAG atteint 93% d\'accuracy (objectif dépassé)',
        '• Amélioration de +2.9x par rapport à spaCy baseline',
        '• Le RAG apporte un gain significatif de +18 points',
        '• Infrastructure GPU opérationnelle (RTX 3060)',
        '• Dataset de 10,000 phrases généré et validé',
    ]

    for conclusion in conclusions:
        plt.text(0.12, y_pos, conclusion, fontsize=11, va='top', color='#34495E')
        y_pos -= 0.04

    y_pos -= 0.03

    # Recommandations
    plt.text(0.1, y_pos, '💡 RECOMMANDATIONS', fontsize=14, fontweight='bold', color='#2C3E50')
    y_pos -= 0.05

    recommendations = [
        'Pour la Production:',
        '  → Utiliser Qwen2.5 + ChromaDB RAG (93% accuracy)',
        '  → Optimiser avec batch processing',
        '  → Déployer via API REST',
        '',
        'Améliorations Futures:',
        '  → Augmenter le dataset RAG à 1,000-5,000 phrases',
        '  → Fine-tuning de Qwen2.5 avec LoRA/QLoRA',
        '  → Optimisation GPU (quantization INT8)',
        '  → Intégrer Flash Attention pour accélérer',
    ]

    for rec in recommendations:
        if rec and not rec.startswith(' '):
            plt.text(0.12, y_pos, rec, fontsize=11, va='top', fontweight='bold', color='#2980B9')
        else:
            plt.text(0.12, y_pos, rec, fontsize=10, va='top', color='#34495E')
        y_pos -= 0.03

    y_pos -= 0.03

    # Points techniques
    plt.text(0.1, y_pos, '🔧 DÉTAILS TECHNIQUES', fontsize=14, fontweight='bold', color='#2C3E50')
    y_pos -= 0.05

    technical = [
        '• Modèle: Qwen2.5-3B-Instruct (Alibaba)',
        '• GPU: NVIDIA RTX 3060 Laptop (6GB VRAM)',
        '• Framework: PyTorch 2.5.1 + CUDA 12.1',
        '• Vector DB: ChromaDB avec embeddings multilingues',
        '• Vitesse: 6.6 secondes/phrase sur GPU',
    ]

    for tech in technical:
        plt.text(0.12, y_pos, tech, fontsize=10, va='top', color='#7F8C8D')
        y_pos -= 0.03

    # Footer
    plt.text(0.5, 0.05, f'Rapport généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}',
             ha='center', va='center', fontsize=9,
             color='#95A5A6', style='italic')

    plt.text(0.5, 0.02, 'Projet T-AIA-911-PAR_14 | Travel Order Resolver NLP',
             ha='center', va='center', fontsize=8,
             color='#BDC3C7')

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Métadonnées du PDF
    d = pdf.infodict()
    d['Title'] = 'Rapport Technique - Travel Order Resolver NLP'
    d['Author'] = 'T-AIA-911-PAR_14'
    d['Subject'] = 'Comparaison spaCy vs Qwen2.5 vs Qwen2.5+ChromaDB RAG'
    d['Keywords'] = 'NLP, Qwen2.5, ChromaDB, RAG, spaCy, GPU'
    d['CreationDate'] = datetime.now()

print(f"\n✅ Rapport PDF généré avec succès: {pdf_filename}")
print(f"   📄 6 pages avec graphiques professionnels")
print(f"   📊 Comparaisons visuelles complètes")
print(f"   📈 Analyses de performance détaillées")
print("="*70)
