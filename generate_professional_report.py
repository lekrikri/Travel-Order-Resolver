import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

# ======================================================================================
# 1. CONFIGURATION ET CONSTANTES DU PROJET
# ======================================================================================

PROJECT_NAME = "Travel Order Resolver"
PROJECT_SUBTITLE = "Système NLP pour l'Extraction et la Résolution d'Itinéraires Ferroviaires"
PROJECT_ID = "T-AIA-911-PAR_14"
VERSION = "2.0.0"
DATE = "Janvier 2025"
AUTHOR = "Epitech"

COLORS = {
    'primary': '#667eea', 'secondary': '#764ba2', 'success': '#28a745',
    'error': '#dc3545', 'neutral': '#6c757d', 'text': '#333333',
    'background': '#f8f9fa', 'white': '#FFFFFF', 'code_bg': '#e9ecef'
}

FONT_FAMILY = 'Helvetica'
CHART_DIR = 'charts'

# ======================================================================================
# 2. FONCTIONS DE GÉNÉRATION DES GRAPHIQUES
# ======================================================================================

def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_FAMILY]
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['xtick.color'] = COLORS['text']
    plt.rcParams['ytick.color'] = COLORS['text']
    plt.rcParams['axes.titlecolor'] = COLORS['text']

def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    return path

# ... (Les fonctions de création de graphiques restent identiques)
def create_precision_comparison_chart():
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {'spaCy Baseline': 32.2, 'Qwen2.5 + RAG': 100.0}
    bars = ax.bar(data.keys(), data.values(), color=[COLORS['error'], COLORS['success']], width=0.6)
    ax.set_ylabel('Précision (%)', fontweight='bold')
    ax.set_title('Comparaison de Précision des Approches NLP', fontweight='bold', fontsize=16)
    ax.set_ylim(0, 115)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    return save_chart(fig, '1_precision_comparison.png')

def create_spacy_errors_pie_chart():
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    data = {'Une seule ville détectée': 55, 'Ordre départ/arrivée inversé': 19, 'Correspondance partielle': 17, 'Aucune ville détectée': 16}
    wedges, _, autotexts = ax.pie(data.values(), explode=(0.05, 0, 0, 0), labels=data.keys(), colors=[COLORS['primary'], '#a98ce2', '#d2bdf0', COLORS['error']], autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Répartition des Erreurs de la Baseline spaCy', fontweight='bold', fontsize=16)
    ax.axis('equal')
    return save_chart(fig, '2_spacy_errors.png')

def create_initial_errors_analysis_chart():
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {'Hallucinations NLP': 23, 'Inversions Départ/Arrivée': 7, 'Bug "Single-Station"': 2, 'Cas "INVALID" valides': 2, 'Bug XML/Inversion': 1}
    y_pos = np.arange(len(data))
    bars = ax.barh(y_pos, data.values(), align='center', color=COLORS['primary'], height=0.6)
    ax.set_yticks(y_pos, labels=data.keys())
    ax.invert_yaxis()
    ax.set_xlabel('Nombre d\'erreurs', fontweight='bold')
    ax.set_title('Analyse des 35 Erreurs Résiduelles (Phase 2)', fontweight='bold', fontsize=16)
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontweight='bold')
    return save_chart(fig, '3_initial_errors.png')

def create_precision_evolution_chart():
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    phases = ['Phase 1\n(spaCy)', 'Phase 2\n(Qwen2.5 initial)', 'Phase 3\n(Corrections)']
    precision = [32.2, 99.60, 100.00]
    ax.plot(phases, precision, color=COLORS['primary'], marker='o', linestyle='-', linewidth=3, markersize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Précision (%)', fontweight='bold')
    ax.set_title('Évolution de la Précision du Modèle', fontweight='bold', fontsize=16)
    for i, txt in enumerate(precision):
        ax.annotate(f'{txt:.2f}%', (phases[i], precision[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
    return save_chart(fig, '4_precision_evolution.png')

def create_dataset_stats_donut_chart():
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    data = {'Routes valides': 8745, 'Routes invalides': 1255}
    wedges, _, autotexts = ax.pie(data.values(), labels=data.keys(), colors=[COLORS['success'], COLORS['error']], autopct='%1.1f%%', startangle=90, pctdistance=0.85, wedgeprops={'edgecolor': 'white', 'linewidth': 3})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    centre_circle = plt.Circle((0,0),0.70,fc=COLORS['white'])
    fig.gca().add_artist(centre_circle)
    plt.text(0, 0.1, f'{sum(data.values()):,}'.replace(',', ' '), ha='center', va='center', fontsize=24, fontweight='bold', color=COLORS['text'])
    plt.text(0, -0.1, 'Phrases Totales', ha='center', va='center', fontsize=14, color=COLORS['neutral'])
    ax.set_title('Composition du Dataset (10k phrases)', fontweight='bold', fontsize=16, pad=20)
    ax.axis('equal')
    return save_chart(fig, '5_dataset_stats.png')

# ======================================================================================
# 3. CLASSE PDF PERSONNALISÉE
# ======================================================================================

def encode_text(text):
    """Encode text to latin-1, replacing unsupported characters."""
    return text.encode('latin-1', 'replace').decode('latin-1')

class ReportPDF(FPDF):
    def header(self):
        self.set_y(10)
        self.set_font(FONT_FAMILY, 'B', 9)
        self.set_text_color(*self.hex_to_rgb(COLORS['neutral']))
        self.cell(0, 10, encode_text(f'{PROJECT_NAME} | {PROJECT_ID}'), 0, 0, 'L')
        self.cell(0, 10, encode_text(DATE), 0, 0, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, 'I', 8)
        self.set_text_color(*self.hex_to_rgb(COLORS['neutral']))
        self.cell(0, 10, encode_text(f'Page {self.page_no()}/{{nb}}'), 0, 0, 'C')

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def chapter_title(self, num, title):
        self.set_font(FONT_FAMILY, 'B', 18)
        self.set_fill_color(*self.hex_to_rgb(COLORS['primary']))
        self.set_text_color(*self.hex_to_rgb(COLORS['white']))
        self.cell(0, 12, encode_text(f' {num}. {title}'), 0, 1, 'L', fill=True)
        self.ln(8)

    def chapter_body(self, text):
        self.set_font(FONT_FAMILY, '', 11)
        self.set_text_color(*self.hex_to_rgb(COLORS['text']))
        self.set_x(20)
        self.multi_cell(0, 7, encode_text(text), 0, 'J')
        self.ln()
        
    def add_table(self, header, data):
        self.set_font(FONT_FAMILY, 'B', 10)
        self.set_fill_color(*self.hex_to_rgb(COLORS['secondary']))
        self.set_text_color(*self.hex_to_rgb(COLORS['white']))
        col_widths = [45, 60, 60]
        for i, header_text in enumerate(header):
            self.cell(col_widths[i], 10, encode_text(header_text), 1, 0, 'C', fill=True)
        self.ln()
        self.set_font(FONT_FAMILY, '', 10)
        self.set_fill_color(*self.hex_to_rgb(COLORS['background']))
        self.set_text_color(*self.hex_to_rgb(COLORS['text']))
        fill = False
        for row in data:
            for i, text in enumerate(row):
                self.cell(col_widths[i], 10, encode_text(str(text)), 1, 0, 'C', fill=fill)
            self.ln()
            fill = not fill
            
    def add_code_snippet(self, code):
        self.set_font('Courier', '', 9)
        self.set_fill_color(*self.hex_to_rgb(COLORS['code_bg']))
        self.set_text_color(*self.hex_to_rgb(COLORS['text']))
        self.multi_cell(0, 5, encode_text(code), border=1, fill=True)
        self.ln(5)

    def cover_page(self):
        self.add_page()
        self.set_y(60)
        self.set_font(FONT_FAMILY, 'B', 36)
        self.set_text_color(*self.hex_to_rgb(COLORS['primary']))
        self.multi_cell(0, 15, encode_text(PROJECT_NAME), 0, 'C')
        self.ln(5)
        self.set_font(FONT_FAMILY, '', 18)
        self.set_text_color(*self.hex_to_rgb(COLORS['neutral']))
        self.multi_cell(0, 10, encode_text(PROJECT_SUBTITLE), 0, 'C')
        self.ln(20)
        self.set_y(self.h - 60)
        self.set_font(FONT_FAMILY, 'B', 12)
        self.set_text_color(*self.hex_to_rgb(COLORS['text']))
        self.cell(0, 10, encode_text(f"Version {VERSION} | {DATE}"), 0, 1, 'C')
        self.cell(0, 10, encode_text(f"Projet Académique: {PROJECT_ID}"), 0, 1, 'C')
        self.cell(0, 10, encode_text(AUTHOR), 0, 1, 'C')

# ======================================================================================
# 4. ASSEMBLAGE DU RAPPORT
# ======================================================================================
def create_report():
    print("Génération des graphiques...")
    if os.path.exists(CHART_DIR):
        shutil.rmtree(CHART_DIR)
    os.makedirs(CHART_DIR)

    chart_paths = {
        'precision_comparison': create_precision_comparison_chart(),
        'spacy_errors': create_spacy_errors_pie_chart(),
        'initial_errors': create_initial_errors_analysis_chart(),
        'precision_evolution': create_precision_evolution_chart(),
        'dataset_stats': create_dataset_stats_donut_chart(),
    }
    print("Graphiques générés.")

    pdf = ReportPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_margins(20, 15, 20)
    pdf.alias_nb_pages()
    
    pdf.cover_page()

    pdf.add_page()
    pdf.chapter_title(1, "Résumé Exécutif")
    pdf.chapter_body(
        "Ce rapport présente le projet 'Travel Order Resolver', un système de Traitement du Langage Naturel (NLP) "
        "conçu pour extraire les villes de départ et d'arrivée de phrases en français et calculer l'itinéraire "
        "ferroviaire optimal sur le réseau SNCF.\n\n"
        "L'objectif était de surpasser une approche classique (spaCy) en utilisant un grand modèle de langage (LLM) "
        "moderne (Qwen2.5) combiné à une architecture RAG (Retrieval-Augmented Generation).\n\n"
        "Le résultat final est un système atteignant 100% de précision sur un jeu de test de 10 000 phrases, "
        "soit une amélioration de 67.8 points par rapport à la baseline. Ce document détaille l'architecture, "
        "les phases d'amélioration, les métriques de performance et les conclusions du projet."
    )
    summary_header = ['Critère', 'Baseline (spaCy)', 'Modèle Final (Qwen2.5)']
    summary_data = [
        ['Précision', '32.2%', '100.0%'],
        ['Amélioration vs Baseline', '-', '+67.8 points'],
        ['GPU Requis', 'Non', 'Oui (pour inférence)'],
        ['Temps / phrase', '~0.01s', '~3.5s']
    ]
    pdf.add_table(summary_header, summary_data)

    pdf.add_page()
    pdf.chapter_title(2, "Architecture du Système")
    pdf.chapter_body(
        "Le système est conçu comme un pipeline modulaire qui traite une phrase en entrée pour produire un itinéraire ferroviaire en sortie. Chaque composant joue un rôle précis pour garantir la robustesse et la précision du résultat."
    )
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.cell(0, 10, "Diagramme du Pipeline", 0, 1, 'C')
    pdf.set_font('Courier', 'B', 11)
    pdf.multi_cell(0, 8, "INPUT (Phrase FR) --> [NLP: Qwen2.5] --> [RAG: ChromaDB] --> [Validation] --> [Pathfinder: NetworkX] --> OUTPUT (Itinéraire)", 0, 'C')
    pdf.ln(5)
    pdf.chapter_body(
        "1.  **NLP (Qwen2.5-3B-Instruct):** Le coeur du système. Le LLM est 'prompté' pour extraire les entités `DEPART` et `ARRIVEE` dans un format structuré (XML).\n\n"
        "2.  **RAG (ChromaDB):** Pour ancrer le LLM sur les données réelles du réseau, une base de données vectorielle ChromaDB est utilisée. Elle contient les noms des 2,785 gares SNCF, ce qui permet au modèle de corriger les coquilles et de ne pas halluciner des gares inexistantes.\n\n"
        "3.  **Validation:** Une étape cruciale qui vérifie la sortie du LLM. Elle s'assure que les deux villes sont présentes et qu'elles existent bien dans la liste des gares SNCF.\n\n"
        "4.  **Pathfinder (NetworkX):** Une fois les gares validées, l'algorithme de Dijkstra (implémenté avec NetworkX) est utilisé sur le graphe du réseau ferroviaire pour trouver le chemin le plus court entre le départ et l'arrivée.\n\n"
        "5.  **Output:** Le résultat est un itinéraire détaillé, incluant les gares intermédiaires."
    )
    
    # ... (Le reste du script pour les autres sections est identique)
    pdf.add_page()
    pdf.chapter_title(3, "Phase 1 - Baseline spaCy")
    pdf.chapter_body("La première phase consistait à établir une performance de référence en utilisant une approche NLP plus traditionnelle avec la bibliothèque spaCy...")
    pdf.image(chart_paths['precision_comparison'], w=pdf.w / 2, x=pdf.w / 4)
    pdf.chapter_body("L'analyse des 67.8% d'erreurs a révélé plusieurs faiblesses systémiques...")
    pdf.image(chart_paths['spacy_errors'], w=pdf.w - 40, x=20)
    pdf.chapter_body("**Conclusion de la Phase 1:** L'approche spaCy, bien que performante pour le NER général, peine à comprendre la relation contextuelle...")

    pdf.add_page()
    pdf.chapter_title(4, "Phase 2 - Qwen2.5 + ChromaDB RAG")
    pdf.chapter_body("La phase 2 a introduit un changement d'architecture majeur...")
    pdf.chapter_body("**Dataset de 10k Phrases:** Un dataset robuste a été généré...")
    pdf.image(chart_paths['dataset_stats'], w=pdf.w - 80, x=40)

    pdf.add_page()
    pdf.chapter_title(5, "Phase 3 - Analyse et Correction des Erreurs")
    pdf.chapter_body("Malgré l'excellente performance de 99.60%, 35 erreurs subsistaient...")
    pdf.image(chart_paths['initial_errors'], w=pdf.w - 40, x=20)
    pdf.chapter_body("Des corrections ciblées ont été apportées...")
    pdf.add_code_snippet(
"""# Exemple de correction: prompt plus directif
# Avant: "Trouve le départ et l'arrivée."
# Après:
prompt = f'''
Extrais DEPART et ARRIVEE de la phrase.
Retourne UNIQUEMENT au format XML: <trip><departure>...</departure><arrival>...</arrival></trip>.
Si une ville manque, laisse le champ vide.
Phrase: "{user_input}"
'''"""
    )
    pdf.chapter_body("Grâce à ces ajustements, le système a atteint une **précision de 100.00%** sur l'ensemble du jeu de test.")

    pdf.add_page()
    pdf.chapter_title(6, "Phase 4 - Production")
    pdf.chapter_body("Le modèle final a été intégré dans une architecture de production...")
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.cell(0, 10, "API REST (FastAPI)", 0, 1, 'L')
    pdf.chapter_body("Une API RESTful a été développée avec FastAPI...")
    api_header = ['Endpoint', 'Méthode', 'Description']
    api_data = [['/resolve_text', 'POST', 'Résout un itinéraire depuis un texte.'], ['/resolve_audio', 'POST', 'Transcrit un audio et résout l\'itinéraire.'], ['/stations', 'GET', 'Liste toutes les gares disponibles.']]
    pdf.add_table(api_header, api_data)
    pdf.ln(5)
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.cell(0, 10, "Frontend (React + Vite)", 0, 1, 'L')
    pdf.chapter_body("Une interface utilisateur simple et réactive a été créée...")
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 80, "[Mockup de l\'interface utilisateur Frontend]", 1, 1, 'C', fill=True)

    pdf.add_page()
    pdf.chapter_title(7, "Résultats et Métriques")
    pdf.chapter_body("Cette section consolide les métriques de performance et l\'évolution du projet.")
    pdf.image(chart_paths['precision_evolution'], w=pdf.w - 40, x=20)
    pdf.ln(5)
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.cell(0, 10, "Tableau Comparatif des Performances", 0, 1, 'C')
    perf_header = ['Critère', 'spaCy', 'Qwen2.5 + RAG']
    perf_data = [['Précision', '32.2%', '100%'], ['Temps/phrase', '0.01s', '3.5s'], ['Mémoire (RAM)', '500 MB', '4 GB'], ['GPU Requis', 'Non', 'Oui']]
    pdf.add_table(perf_header, perf_data)

    pdf.add_page()
    pdf.chapter_title(8, "Conclusion et Perspectives")
    pdf.chapter_body("Le projet 'Travel Order Resolver' a atteint avec succès son objectif principal...")

    output_filename = "RAPPORT_FINAL.pdf"
    pdf.output(output_filename)
    print(f"Le rapport a été généré avec succès : {output_filename}")
    
    shutil.rmtree(CHART_DIR)
    print("Répertoire des graphiques temporaires supprimé.")

if __name__ == '__main__':
    create_report()
