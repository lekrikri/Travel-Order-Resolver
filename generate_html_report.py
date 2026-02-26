#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération d'un rapport HTML professionnel avec graphiques
spaCy vs Qwen2.5 seul vs Qwen2.5 + ChromaDB RAG
"""

import sys
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("GÉNÉRATION DU RAPPORT HTML PROFESSIONNEL")
print("="*70)

# Configuration matplotlib
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64 pour HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# Données
approaches = ['spaCy', 'Qwen2.5 seul', 'Qwen2.5 + ChromaDB RAG']
accuracy = [32.2, 75.0, 93.0]
departure_correct = [43.8, 75.0, 93.0]
arrival_correct = [46.3, 75.0, 93.0]
time_per_phrase = [0.5, 6.57, 6.66]

colors = ['#E74C3C', '#F39C12', '#27AE60']

print("\n[1/4] Génération des graphiques...")

# Graphique 1: Comparaison Accuracy
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(approaches, accuracy, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
for bar, val in zip(bars, accuracy):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{val}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Comparaison de l\'Accuracy', fontsize=15, fontweight='bold', pad=15)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Objectif 90%')
ax1.legend(fontsize=11)
chart1 = fig_to_base64(fig1)

# Graphique 2: Départ vs Arrivée
fig2, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(len(approaches))
width = 0.35
bars1 = ax2.bar(x - width/2, departure_correct, width, label='Départ Correct',
               color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, arrival_correct, width, label='Arrivée Correcte',
               color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.5)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Précision (%)', fontsize=13, fontweight='bold')
ax2.set_title('Précision Départ vs Arrivée', fontsize=15, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(approaches)
ax2.legend(fontsize=11, loc='upper left')
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
chart2 = fig_to_base64(fig2)

# Graphique 3: Évolution
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(range(3), accuracy, marker='o', markersize=15, linewidth=4,
        color='#2ECC71', label='Accuracy')
ax3.fill_between(range(3), 0, accuracy, alpha=0.3, color='#2ECC71')
for i, val in enumerate(accuracy):
    ax3.annotate(f'{val}%', xy=(i, val), xytext=(10, 10),
                textcoords='offset points', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2ECC71', linewidth=2))
ax3.set_xticks(range(3))
ax3.set_xticklabels(approaches)
ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax3.set_title('Évolution Progressive de l\'Accuracy', fontsize=15, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0, 100)
chart3 = fig_to_base64(fig3)

# Graphique 4: Performance
fig4, ax4 = plt.subplots(figsize=(10, 6))
efficiency = [acc/time if time > 0 else 0 for acc, time in zip(accuracy, time_per_phrase)]
bars = ax4.bar(approaches, efficiency, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
for bar, val in zip(bars, efficiency):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_ylabel('Efficacité (Accuracy/Temps)', fontsize=13, fontweight='bold')
ax4.set_title('Efficacité des Approches', fontsize=15, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
chart4 = fig_to_base64(fig4)

print("[2/4] Génération du HTML...")

# Template HTML
html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Technique - Travel Order Resolver NLP</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            color: #2C3E50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}

        .header h2 {{
            font-size: 1.5em;
            font-weight: 300;
            opacity: 0.95;
        }}

        .header .meta {{
            margin-top: 30px;
            font-size: 0.95em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section-title {{
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .subsection-title {{
            font-size: 1.4em;
            font-weight: 600;
            color: #764ba2;
            margin-top: 30px;
            margin-bottom: 15px;
        }}

        .highlight-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}

        .success-box {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 5px solid #28a745;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}

        .info-box {{
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border-left: 5px solid #17a2b8;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}

        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .chart-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #7F8C8D;
            font-size: 0.9em;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}

        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}

        tr:hover {{
            background-color: #e9ecef;
        }}

        .metric {{
            display: inline-block;
            background: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 200px;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
        }}

        .metric-label {{
            font-size: 0.9em;
            color: #7F8C8D;
            margin-top: 5px;
        }}

        .tech-stack {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .tech-item {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #ecf0f1;
            transition: all 0.3s ease;
        }}

        .tech-item:hover {{
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }}

        .tech-item h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}

        li {{
            margin: 8px 0;
        }}

        .footer {{
            background: #2C3E50;
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .print-button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            margin: 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}

        .print-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .print-button {{
                display: none;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Rapport Technique</h1>
            <h2>Travel Order Resolver NLP</h2>
            <p style="margin-top: 20px; font-size: 1.1em;">Extraction de Villes pour Réservations de Train en Français</p>
            <div class="meta">
                <p>Projet: T-AIA-911-PAR_14</p>
                <p>Date: {datetime.now().strftime("%d/%m/%Y")}</p>
            </div>
            <button class="print-button" onclick="window.print()">🖨️ Imprimer / Sauvegarder en PDF</button>
        </div>

        <div class="content">
            <!-- Résumé Exécutif -->
            <div class="section">
                <h2 class="section-title">📈 Résumé Exécutif</h2>

                <div class="success-box">
                    <h3 style="margin-bottom: 15px;">🏆 Résultats Exceptionnels</h3>
                    <div style="text-align: center;">
                        <div class="metric">
                            <div class="metric-value">93.0%</div>
                            <div class="metric-label">Accuracy Finale</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">+2.9x</div>
                            <div class="metric-label">vs Baseline</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">+18 pts</div>
                            <div class="metric-label">Gain du RAG</div>
                        </div>
                    </div>
                </div>

                <p style="font-size: 1.1em; line-height: 1.8;">
                    Ce projet implémente un système d'extraction intelligente de villes (départ/arrivée)
                    à partir de commandes en langage naturel pour des réservations de train en français.
                    Nous avons comparé trois approches : une baseline avec <strong>spaCy</strong>,
                    le modèle de langage <strong>Qwen2.5</strong> seul, et enfin <strong>Qwen2.5 combiné
                    avec ChromaDB RAG</strong> pour une amélioration contextuelle.
                </p>
            </div>

            <!-- Contexte et Problématique -->
            <div class="section">
                <h2 class="section-title">🎯 Contexte et Problématique</h2>

                <h3 class="subsection-title">Le Défi</h3>
                <p>
                    Les utilisateurs réservent des billets de train en utilisant des formulations variées :
                </p>
                <ul>
                    <li>"Je veux aller de Paris à Lyon demain"</li>
                    <li>"Un billet pour Marseille depuis Toulouse svp"</li>
                    <li>"Train Bordeaux Nice dans l'après-midi"</li>
                    <li>"Direction Lille au départ de Strasbourg"</li>
                </ul>

                <div class="info-box">
                    <h4>Objectif du Projet</h4>
                    <p>
                        Extraire automatiquement et avec précision la <strong>ville de départ</strong>
                        et la <strong>ville d'arrivée</strong> à partir de ces commandes en langage naturel,
                        quelle que soit la formulation utilisée.
                    </p>
                </div>

                <h3 class="subsection-title">Pourquoi c'est difficile ?</h3>
                <ul>
                    <li><strong>Variété de formulations</strong> : Des centaines de façons différentes d'exprimer la même chose</li>
                    <li><strong>Ordre inversé</strong> : "De Paris à Lyon" vs "Lyon depuis Paris"</li>
                    <li><strong>Contexte manquant</strong> : Parfois une seule ville est mentionnée</li>
                    <li><strong>Ambiguïtés</strong> : Distinguer les noms de villes d'autres entités</li>
                </ul>
            </div>

            <!-- Technologies Utilisées -->
            <div class="section">
                <h2 class="section-title">🛠️ Technologies et Outils</h2>

                <div class="tech-stack">
                    <div class="tech-item">
                        <h4>🤖 Qwen2.5-3B-Instruct</h4>
                        <p><strong>Qu'est-ce que c'est ?</strong></p>
                        <p>
                            Un modèle de langage (LLM) de 3 milliards de paramètres développé par <strong>Alibaba</strong>.
                            C'est une IA capable de comprendre et générer du texte en plusieurs langues dont le français.
                        </p>
                        <p><strong>Pourquoi l'utiliser ?</strong></p>
                        <ul>
                            <li>Excellente compréhension du contexte</li>
                            <li>Capacité de few-shot learning (apprend à partir d'exemples)</li>
                            <li>Multilingue natif (français supporté)</li>
                            <li>Taille optimale pour GPU (3B paramètres)</li>
                        </ul>
                    </div>

                    <div class="tech-item">
                        <h4>🗄️ ChromaDB RAG</h4>
                        <p><strong>Qu'est-ce que c'est ?</strong></p>
                        <p>
                            Une base de données vectorielle qui stocke des phrases avec leurs "embeddings"
                            (représentations mathématiques). Le <strong>RAG (Retrieval-Augmented Generation)</strong>
                            recherche les exemples les plus similaires pour améliorer les prédictions.
                        </p>
                        <p><strong>Pourquoi l'utiliser ?</strong></p>
                        <ul>
                            <li>Fournit des exemples contextuellement pertinents</li>
                            <li>Améliore la précision de +18 points !</li>
                            <li>Recherche vectorielle ultra-rapide</li>
                            <li>Adaptabilité dynamique aux requêtes</li>
                        </ul>
                    </div>

                    <div class="tech-item">
                        <h4>📚 spaCy (Baseline)</h4>
                        <p><strong>Qu'est-ce que c'est ?</strong></p>
                        <p>
                            Une bibliothèque NLP traditionnelle avec reconnaissance d'entités nommées (NER).
                            Utilisée comme baseline pour comparer les performances.
                        </p>
                        <p><strong>Limites identifiées :</strong></p>
                        <ul>
                            <li>Confond souvent départ et arrivée (19%)</li>
                            <li>Détecte une seule ville dans 55% des cas</li>
                            <li>Manque de compréhension contextuelle</li>
                        </ul>
                    </div>

                    <div class="tech-item">
                        <h4>🎮 GPU NVIDIA RTX 3060</h4>
                        <p><strong>Infrastructure</strong></p>
                        <ul>
                            <li>6GB de VRAM</li>
                            <li>PyTorch 2.5.1 + CUDA 12.1</li>
                            <li>Accélération 10-50x vs CPU</li>
                            <li>Temps: 6.6s/phrase sur GPU vs plusieurs minutes sur CPU</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Comment fonctionne le RAG -->
            <div class="section">
                <h2 class="section-title">🔍 Comment fonctionne ChromaDB RAG ?</h2>

                <div class="highlight-box">
                    <h3 style="margin-bottom: 15px;">Le Processus en 4 Étapes</h3>

                    <ol style="font-size: 1.05em; line-height: 1.8;">
                        <li>
                            <strong>Stockage :</strong> On charge 100 phrases annotées dans ChromaDB
                            (ex: "Je veux aller de Paris à Lyon" → départ=Paris, arrivée=Lyon)
                        </li>
                        <li>
                            <strong>Requête utilisateur :</strong> L'utilisateur demande
                            "Train pour Marseille depuis Nice"
                        </li>
                        <li>
                            <strong>Recherche vectorielle :</strong> ChromaDB trouve les 3 phrases
                            les plus similaires dans sa base (basé sur le sens, pas juste les mots)
                        </li>
                        <li>
                            <strong>Few-shot learning :</strong> Qwen2.5 utilise ces 3 exemples
                            similaires pour mieux comprendre et extraire départ=Nice, arrivée=Marseille
                        </li>
                    </ol>
                </div>

                <p style="margin-top: 20px; font-size: 1.05em;">
                    <strong>Résultat :</strong> Au lieu d'utiliser toujours les mêmes exemples génériques,
                    Qwen2.5 reçoit des exemples adaptés à chaque requête, ce qui améliore drastiquement
                    la précision (<strong>+18 points</strong> : de 75% à 93% !).
                </p>
            </div>

            <!-- Résultats Comparatifs -->
            <div class="section">
                <h2 class="section-title">📊 Résultats Comparatifs</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Approche</th>
                            <th>Accuracy</th>
                            <th>Départ Correct</th>
                            <th>Arrivée Correcte</th>
                            <th>Temps/phrase</th>
                            <th>Amélioration</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>spaCy (baseline)</strong></td>
                            <td>32.2%</td>
                            <td>43.8%</td>
                            <td>46.3%</td>
                            <td>0.5s</td>
                            <td>-</td>
                        </tr>
                        <tr style="background-color: #fff3cd;">
                            <td><strong>Qwen2.5 seul</strong></td>
                            <td>75.0%</td>
                            <td>75.0%</td>
                            <td>75.0%</td>
                            <td>6.6s</td>
                            <td><strong>+2.3x</strong> (+42.8 pts)</td>
                        </tr>
                        <tr style="background-color: #d4edda;">
                            <td><strong>Qwen2.5 + ChromaDB RAG</strong></td>
                            <td><strong>93.0%</strong></td>
                            <td><strong>93.0%</strong></td>
                            <td><strong>93.0%</strong></td>
                            <td>6.7s</td>
                            <td><strong>+2.9x</strong> (+60.8 pts)</td>
                        </tr>
                    </tbody>
                </table>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart1}" alt="Graphique Accuracy">
                    <p class="chart-caption">Figure 1 : Comparaison de l'accuracy entre les trois approches</p>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart2}" alt="Graphique Départ/Arrivée">
                    <p class="chart-caption">Figure 2 : Précision sur l'extraction du départ et de l'arrivée</p>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart3}" alt="Graphique Évolution">
                    <p class="chart-caption">Figure 3 : Évolution progressive de l'accuracy</p>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart4}" alt="Graphique Efficacité">
                    <p class="chart-caption">Figure 4 : Efficacité (ratio accuracy/temps)</p>
                </div>
            </div>

            <!-- Analyse Détaillée -->
            <div class="section">
                <h2 class="section-title">🔬 Analyse Détaillée</h2>

                <h3 class="subsection-title">Impact du RAG ChromaDB</h3>
                <div class="success-box">
                    <p style="font-size: 1.15em;">
                        Le RAG ChromaDB apporte un <strong>gain de +18 points</strong> d'accuracy
                        (de 75% à 93%), soit une <strong>amélioration relative de 24%</strong> !
                    </p>
                    <p style="margin-top: 10px;">
                        Cela démontre l'efficacité du few-shot learning adaptatif : en fournissant
                        des exemples contextuellement pertinents, Qwen2.5 comprend mieux la tâche
                        et génère des prédictions plus précises.
                    </p>
                </div>

                <h3 class="subsection-title">Pourquoi Qwen2.5 surpasse spaCy ?</h3>
                <ul>
                    <li><strong>Compréhension contextuelle :</strong> Qwen2.5 comprend le sens global de la phrase, pas seulement les mots</li>
                    <li><strong>Gestion de l'ordre :</strong> Pas de confusion entre "Paris à Lyon" et "Lyon depuis Paris"</li>
                    <li><strong>Robustesse :</strong> Fonctionne même avec des formulations inhabituelles</li>
                    <li><strong>Few-shot learning :</strong> S'adapte en voyant seulement quelques exemples</li>
                </ul>

                <h3 class="subsection-title">Limitations</h3>
                <ul>
                    <li><strong>Vitesse :</strong> 6.6s/phrase sur GPU vs 0.5s pour spaCy (13x plus lent)</li>
                    <li><strong>Ressources :</strong> Nécessite un GPU pour être performant</li>
                    <li><strong>Complexité :</strong> Architecture plus complexe que spaCy</li>
                </ul>

                <div class="info-box">
                    <h4>💡 Compromis Performance vs Vitesse</h4>
                    <p>
                        Pour un système de production, le choix dépend des besoins :
                    </p>
                    <ul>
                        <li><strong>Précision critique :</strong> Utiliser Qwen2.5 + RAG (93% accuracy)</li>
                        <li><strong>Volume élevé :</strong> Optimiser avec batch processing et GPU</li>
                        <li><strong>Contraintes temps réel :</strong> Envisager un modèle plus petit ou quantization</li>
                    </ul>
                </div>
            </div>

            <!-- Méthodologie -->
            <div class="section">
                <h2 class="section-title">🔧 Méthodologie</h2>

                <h3 class="subsection-title">1. Dataset Génération</h3>
                <p>
                    Nous avons généré automatiquement <strong>10,000 phrases</strong> avec :
                </p>
                <ul>
                    <li>131 villes françaises</li>
                    <li>83 templates de formulation variés</li>
                    <li>15% de questions, 10% avec dates, 4% avec contexte</li>
                    <li>100% de phrases uniques</li>
                </ul>

                <h3 class="subsection-title">2. Tests Effectués</h3>
                <ul>
                    <li><strong>spaCy :</strong> 1,000 phrases testées</li>
                    <li><strong>Qwen2.5 seul :</strong> 100 phrases testées</li>
                    <li><strong>Qwen2.5 + RAG :</strong> 100 phrases testées avec 100 exemples ChromaDB</li>
                </ul>

                <h3 class="subsection-title">3. Infrastructure GPU</h3>
                <ul>
                    <li>NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)</li>
                    <li>PyTorch 2.5.1 + CUDA 12.1</li>
                    <li>Python 3.12 dans environnement virtuel</li>
                    <li>Vitesse d'inférence : 6.6s/phrase</li>
                </ul>
            </div>

            <!-- Conclusions et Recommandations -->
            <div class="section">
                <h2 class="section-title">✅ Conclusions et Recommandations</h2>

                <h3 class="subsection-title">Conclusions Principales</h3>
                <div class="success-box">
                    <ol style="font-size: 1.05em; line-height: 1.8;">
                        <li><strong>Qwen2.5 + ChromaDB RAG</strong> atteint <strong>93% d'accuracy</strong>, dépassant largement l'objectif de 90%</li>
                        <li>Le <strong>RAG apporte un gain significatif de +18 points</strong>, démontrant l'efficacité du few-shot learning adaptatif</li>
                        <li>L'infrastructure GPU est <strong>opérationnelle et performante</strong> (RTX 3060)</li>
                        <li>Le dataset de <strong>10,000 phrases</strong> est prêt pour un éventuel fine-tuning futur</li>
                    </ol>
                </div>

                <h3 class="subsection-title">Recommandations pour la Production</h3>
                <div class="highlight-box">
                    <h4>Court Terme</h4>
                    <ul>
                        <li>Déployer <strong>Qwen2.5 + ChromaDB RAG</strong> comme solution principale</li>
                        <li>Implémenter <strong>batch processing</strong> pour traiter plusieurs requêtes simultanément</li>
                        <li>Créer une <strong>API REST</strong> pour faciliter l'intégration</li>
                        <li>Monitorer les performances en production</li>
                    </ul>

                    <h4>Moyen Terme</h4>
                    <ul>
                        <li><strong>Augmenter le dataset RAG</strong> à 1,000-5,000 phrases pour plus de couverture</li>
                        <li><strong>Fine-tuning de Qwen2.5</strong> avec LoRA/QLoRA sur le dataset 10k</li>
                        <li><strong>Optimisation GPU</strong> avec quantization INT8 pour réduire la VRAM</li>
                        <li>Implémenter <strong>Flash Attention</strong> pour accélérer l'inférence</li>
                    </ul>

                    <h4>Long Terme</h4>
                    <ul>
                        <li>Explorer des modèles plus grands (Qwen2.5-7B) si le matériel le permet</li>
                        <li>Étendre à d'autres langues (anglais, espagnol, etc.)</li>
                        <li>Intégrer des fonctionnalités supplémentaires (dates, horaires, préférences)</li>
                    </ul>
                </div>
            </div>

            <!-- Livrables -->
            <div class="section">
                <h2 class="section-title">📦 Livrables du Projet</h2>

                <h3 class="subsection-title">Datasets</h3>
                <ul>
                    <li><code>data/annotated/train_sentences_100.csv</code> - 100 phrases annotées (score 83/100)</li>
                    <li><code>data/annotated/train_sentences_10k.csv</code> - 10,000 phrases générées (885 KB)</li>
                </ul>

                <h3 class="subsection-title">Résultats de Tests</h3>
                <ul>
                    <li><code>results/spacy_10k_results.csv</code> - Résultats spaCy (32.2% accuracy)</li>
                    <li><code>results/qwen_only_100_results.csv</code> - Résultats Qwen2.5 seul (75.0%)</li>
                    <li><code>results/qwen_chromadb_100_results.csv</code> - Résultats Qwen2.5 + RAG (93.0%)</li>
                </ul>

                <h3 class="subsection-title">Code Source</h3>
                <ul>
                    <li><code>src/2_qwen_chroma/qwen_extractor.py</code> - Wrapper Qwen2.5</li>
                    <li><code>src/2_qwen_chroma/chroma_rag.py</code> - Système RAG ChromaDB</li>
                    <li><code>test_*.py</code> - Scripts de test pour chaque approche</li>
                </ul>

                <h3 class="subsection-title">Documentation</h3>
                <ul>
                    <li><code>RAPPORT_PROFESSIONNEL.html</code> - Ce rapport (format HTML)</li>
                    <li><code>RAPPORT_COMPARATIF_FINAL.md</code> - Rapport Markdown détaillé</li>
                    <li><code>README.md</code> - Documentation du projet</li>
                    <li><code>PRESENTATION_PROF.md</code> - Présentation pour le professeur</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            <p style="font-size: 1.1em; margin-bottom: 10px;">
                <strong>Projet T-AIA-911-PAR_14 | Travel Order Resolver NLP</strong>
            </p>
            <p style="opacity: 0.8;">
                Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")} |
                Technologies: Qwen2.5 + ChromaDB + PyTorch CUDA
            </p>
        </div>
    </div>
</body>
</html>"""

print("[3/4] Sauvegarde du fichier HTML...")

# Sauvegarder le fichier HTML
with open('./RAPPORT_PROFESSIONNEL.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("[4/4] Terminé !")
print("\n" + "="*70)
print("✅ RAPPORT HTML GÉNÉRÉ AVEC SUCCÈS")
print("="*70)
print(f"\n📄 Fichier: ./RAPPORT_PROFESSIONNEL.html")
print(f"📊 Contenu:")
print(f"   • Page de garde professionnelle")
print(f"   • Explications détaillées de Qwen2.5 et ChromaDB")
print(f"   • 4 graphiques avec légendes bien placées")
print(f"   • Analyse approfondie et recommandations")
print(f"   • Design professionnel responsive")
print(f"\n💡 Pour ouvrir: Double-cliquez sur le fichier")
print(f"📥 Pour sauvegarder en PDF: Ouvrir dans le navigateur → Imprimer → Enregistrer en PDF")
print("="*70)
