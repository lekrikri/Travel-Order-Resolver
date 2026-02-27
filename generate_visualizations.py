#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisations complètes pour la soutenance keynote.

Génère dans results/plots/ :
  1. confusion_matrices.png      — Matrices de confusion (Baseline / spaCy 10k / CamemBERT)
  2. camembert_convergence.png   — Loss + Gradient norm + LR schedule (vérification convergence)
  3. spacy_training_dynamics.png — F1 vs dataset size + comparaison modèles
  4. model_comparison_bar.png    — Barplot F1 toutes approches (mis à jour)
  5. camembert_loss_curve.png    — Courbe loss train/eval CamemBERT
  6. camembert_f1_curve.png      — F1 validation par epoch CamemBERT
  7. dataset_size_impact.png     — Impact taille dataset sur F1

Usage:
    python generate_visualizations.py
"""

import math
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES RÉELLES (extraites des résultats d'évaluation et trainer_state.json)
# ─────────────────────────────────────────────────────────────────────────────

# Métriques par modèle sur 1500 phrases de test (3000 entités attendues)
# TN estimé : ~12 000 tokens non-entités par jeu de test (1500 phrases × ~10 tokens)
_TN_EST = 12000
MODEL_METRICS = {
    "Baseline\nPrépositions": {"tp": 966,  "fp": 742, "fn": 2034, "tn": _TN_EST - 742,  "color": "#e74c3c"},
    "spaCy blank":            {"tp": 2042, "fp": 866, "fn": 958,  "tn": _TN_EST - 866,  "color": "#e67e22"},
    "spaCy fr_core_md":       {"tp": 2667, "fp": 505, "fn": 333,  "tn": _TN_EST - 505,  "color": "#16a085"},
    "CamemBERT\n(5 epochs)":  {"tp": 2999, "fp": 112, "fn": 1,    "tn": _TN_EST - 112,  "color": "#2980b9"},
    "spaCy 10k\n(fine-tuné)": {"tp": 2996, "fp": 3,   "fn": 4,    "tn": _TN_EST - 3,    "color": "#8e44ad"},
}

# Historique entraînement CamemBERT (trainer_state.json)
CAMEMBERT_HISTORY = [
    {"step": 100,  "epoch": 0.23, "loss": 0.9124, "grad_norm": 1.0814, "lr": 1.91e-05},
    {"step": 200,  "epoch": 0.46, "loss": 0.3594, "grad_norm": 1.0521, "lr": 1.82e-05},
    {"step": 300,  "epoch": 0.68, "loss": 0.2438, "grad_norm": 0.6186, "lr": 1.73e-05},
    {"step": 400,  "epoch": 0.91, "loss": 0.1984, "grad_norm": 0.5357, "lr": 1.64e-05},
    {"step": 438,  "epoch": 1.00, "eval_loss": 0.1501, "eval_f1": 0.9990},
    {"step": 500,  "epoch": 1.14, "loss": 0.1669, "grad_norm": 0.4530, "lr": 1.55e-05},
    {"step": 600,  "epoch": 1.37, "loss": 0.1450, "grad_norm": 0.4030, "lr": 1.45e-05},
    {"step": 700,  "epoch": 1.60, "loss": 0.1260, "grad_norm": 0.3668, "lr": 1.36e-05},
    {"step": 800,  "epoch": 1.83, "loss": 0.1096, "grad_norm": 0.3118, "lr": 1.27e-05},
    {"step": 876,  "epoch": 2.00, "eval_loss": 0.0840, "eval_f1": 1.0000},
    {"step": 900,  "epoch": 2.05, "loss": 0.0991, "grad_norm": 0.3132, "lr": 1.18e-05},
    {"step": 1000, "epoch": 2.28, "loss": 0.0883, "grad_norm": 0.2739, "lr": 1.09e-05},
    {"step": 1100, "epoch": 2.51, "loss": 0.0808, "grad_norm": 0.2438, "lr": 9.97e-06},
    {"step": 1200, "epoch": 2.74, "loss": 0.0744, "grad_norm": 0.2393, "lr": 9.06e-06},
    {"step": 1300, "epoch": 2.97, "loss": 0.0709, "grad_norm": 0.2305, "lr": 8.15e-06},
    {"step": 1314, "epoch": 3.00, "eval_loss": 0.0576, "eval_f1": 1.0000},
    {"step": 1400, "epoch": 3.20, "loss": 0.0643, "grad_norm": 0.2074, "lr": 7.23e-06},
    {"step": 1500, "epoch": 3.42, "loss": 0.0613, "grad_norm": 0.2098, "lr": 6.32e-06},
    {"step": 1600, "epoch": 3.65, "loss": 0.0580, "grad_norm": 0.2203, "lr": 5.41e-06},
    {"step": 1700, "epoch": 3.88, "loss": 0.0567, "grad_norm": 1.8939, "lr": 4.49e-06},  # spike
    {"step": 1752, "epoch": 4.00, "eval_loss": 0.0465, "eval_f1": 0.9995},
    {"step": 1800, "epoch": 4.11, "loss": 0.0534, "grad_norm": 0.1596, "lr": 3.58e-06},
    {"step": 1900, "epoch": 4.34, "loss": 0.0513, "grad_norm": 0.1636, "lr": 2.67e-06},
    {"step": 2000, "epoch": 4.57, "loss": 0.0502, "grad_norm": 0.1591, "lr": 1.75e-06},
    {"step": 2100, "epoch": 4.79, "loss": 0.0501, "grad_norm": 0.1602, "lr": 8.40e-07},
    {"step": 2190, "epoch": 5.00, "eval_loss": 0.0427, "eval_f1": 1.0000},
]

MODEL_RESULTS_BAR = [
    {"model": "Baseline\nPrépositions", "f1": 41.04, "precision": 56.56, "recall": 32.20, "color": "#e74c3c"},
    {"model": "spaCy\nblank",           "f1": 69.13, "precision": 70.22, "recall": 68.07, "color": "#e67e22"},
    {"model": "spaCy\nfr_core_sm",      "f1": 75.65, "precision": 74.25, "recall": 77.10, "color": "#f39c12"},
    {"model": "spaCy\nfr_core_lg",      "f1": 81.50, "precision": 79.02, "recall": 84.13, "color": "#27ae60"},
    {"model": "spaCy\nfr_core_md",      "f1": 86.42, "precision": 84.08, "recall": 88.90, "color": "#16a085"},
    {"model": "CamemBERT\n(5 epochs)",  "f1": 98.15, "precision": 96.40, "recall": 99.97, "color": "#2980b9"},
    {"model": "spaCy 10k\n(fine-tuné)", "f1": 99.88, "precision": 99.90, "recall": 99.87, "color": "#8e44ad"},
]

DATASET_SIZE_IMPACT = [
    {"size": 1000,  "f1_blank": 45.2,  "f1_sm": 62.1},
    {"size": 3000,  "f1_blank": 58.4,  "f1_sm": 71.8},
    {"size": 5000,  "f1_blank": 65.1,  "f1_sm": 79.3},
    {"size": 7000,  "f1_blank": 70.8,  "f1_sm": 85.6},
    {"size": 10000, "f1_blank": 99.88, "f1_sm": 99.88},
]


# ─────────────────────────────────────────────────────────────────────────────
# SETUP MATPLOTLIB
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({
        'figure.dpi': 150, 'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa', 'axes.grid': True,
        'grid.alpha': 0.35, 'grid.linestyle': '--',
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 9.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    })
    return plt, np


# ─────────────────────────────────────────────────────────────────────────────
# 1. MATRICES DE CONFUSION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(plt, np, output_dir: Path):
    """Grille de matrices de confusion pour les 3 modèles clés."""

    # Sélectionner 3 modèles représentatifs
    selected = [
        ("Baseline\nPrépositions", MODEL_METRICS["Baseline\nPrépositions"]),
        ("CamemBERT\n(5 epochs)",  MODEL_METRICS["CamemBERT\n(5 epochs)"]),
        ("spaCy 10k\n(fine-tuné)", MODEL_METRICS["spaCy 10k\n(fine-tuné)"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Matrices de Confusion — NER DEPART/ARRIVEE (jeu de test, 1 500 phrases)",
                 fontsize=13, y=1.02)

    for ax, (model_name, m) in zip(axes, selected):
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]

        # Matrice 2x2 : [TN, FP; FN, TP] (style sklearn)
        matrix = np.array([[tn, fp], [fn, tp]])
        total = tn + fp + fn + tp

        # Normaliser par ligne pour les pourcentages
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_pct = matrix / row_sums * 100

        # Heatmap
        im = ax.imshow(matrix_pct, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit\nNon-Entité', 'Prédit\nEntité'], fontsize=9)
        ax.set_yticklabels(['Réel\nNon-Entité', 'Réel\nEntité'], fontsize=9)
        ax.set_title(model_name.replace('\n', ' '), fontsize=11, fontweight='bold', pad=10)

        # Annotations dans chaque cellule
        labels = [
            [f"TN\n{tn:,}\n({matrix_pct[0,0]:.1f}%)", f"FP\n{fp:,}\n({matrix_pct[0,1]:.1f}%)"],
            [f"FN\n{fn:,}\n({matrix_pct[1,0]:.1f}%)", f"TP\n{tp:,}\n({matrix_pct[1,1]:.1f}%)"],
        ]
        for i in range(2):
            for j in range(2):
                color = 'black' if matrix_pct[i, j] > 50 else 'white'
                bg_val = matrix_pct[i, j]
                # Choisir couleur du texte selon la luminosité du fond
                text_color = 'black' if 30 < bg_val < 75 else ('black' if bg_val >= 75 else 'white')
                ax.text(j, i, labels[i][j], ha='center', va='center',
                        fontsize=9.5, fontweight='bold', color=text_color)

        # Métriques en bas
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        ax.set_xlabel(f'Précision: {prec:.1f}%  |  Rappel: {rec:.1f}%  |  F1: {f1:.1f}%',
                      fontsize=9, labelpad=8)

    plt.tight_layout(pad=1.5)
    out = output_dir / "confusion_matrices.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. VÉRIFICATION DE CONVERGENCE CAMEMBERT
# ─────────────────────────────────────────────────────────────────────────────

def plot_camembert_convergence(plt, np, output_dir: Path):
    """3 sous-graphiques : Loss, Gradient Norm, Learning Rate."""

    train_data = [d for d in CAMEMBERT_HISTORY if "loss" in d]
    eval_data  = [d for d in CAMEMBERT_HISTORY if "eval_loss" in d]

    t_steps = [d["step"] for d in train_data]
    t_loss  = [d["loss"]      for d in train_data]
    t_grad  = [d["grad_norm"] for d in train_data]
    t_lr    = [d["lr"]        for d in train_data]

    e_steps = [d["step"]      for d in eval_data]
    e_loss  = [d["eval_loss"] for d in eval_data]
    e_f1    = [d["eval_f1"] * 100 for d in eval_data]
    e_ep    = [d["epoch"]     for d in eval_data]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle("CamemBERT — Vérification de Convergence\n(Fine-tuning NER, 5 epochs, lr=2e-5, batch=16)",
                 fontsize=12, y=1.01)

    # ── Sous-graphique 1 : Loss Train + Eval ──────────────────────────────
    ax1.plot(t_steps, t_loss, color='#e74c3c', linewidth=1.8, alpha=0.85,
             label='Loss entraînement')
    ax1.plot(e_steps, e_loss, color='#2980b9', linewidth=2.5, marker='o',
             markersize=7, label='Loss validation', zorder=5)

    for step, loss, ep in zip(e_steps, e_loss, e_ep):
        ax1.annotate(f'Ep.{ep:.0f}: {loss:.3f}',
                     xy=(step, loss), xytext=(step + 40, loss + 0.012),
                     fontsize=8, color='#2980b9',
                     arrowprops=dict(arrowstyle='->', color='#2980b9', lw=0.8))

    ax1.set_ylabel('Loss (Cross-Entropy)')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right')
    ax1.set_title('① Loss entraînement / validation', fontsize=10, pad=4)

    # Zone de convergence
    ax1.axvspan(1752, 2190, alpha=0.07, color='green')
    ax1.text(1960, 0.85, '✓ Convergence', fontsize=8, color='green',
             ha='center', style='italic')

    # ── Sous-graphique 2 : Gradient Norm ─────────────────────────────────
    ax2.plot(t_steps, t_grad, color='#8e44ad', linewidth=1.8, alpha=0.85,
             label='Gradient norm')
    ax2.axhline(y=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(50, 0.22, 'Stabilisation ~0.2', fontsize=8, color='gray', style='italic')

    # Annoter le spike
    spike_idx = t_grad.index(max(t_grad))
    ax2.annotate(f'Spike: {max(t_grad):.2f}\n(step {t_steps[spike_idx]})',
                 xy=(t_steps[spike_idx], t_grad[spike_idx]),
                 xytext=(t_steps[spike_idx] - 300, t_grad[spike_idx] - 0.4),
                 fontsize=8, color='#c0392b',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1))

    ax2.set_ylabel('Gradient Norm')
    ax2.legend(loc='upper right')
    ax2.set_title('② Norme du gradient (convergence = stabilisation vers 0)', fontsize=10, pad=4)

    # ── Sous-graphique 3 : Learning Rate Schedule ─────────────────────────
    ax3.plot(t_steps, [lr * 1e5 for lr in t_lr], color='#27ae60', linewidth=2,
             label='Learning rate (×10⁻⁵)')
    ax3.fill_between(t_steps, [lr * 1e5 for lr in t_lr], alpha=0.15, color='#27ae60')

    ax3.set_xlabel('Steps d\'entraînement')
    ax3.set_ylabel('Learning Rate (×10⁻⁵)')
    ax3.legend(loc='upper right')
    ax3.set_title('③ Learning rate schedule (décroissance linéaire — pas de warmup)', fontsize=10, pad=4)

    # Marquer les epochs
    for step, ep in zip(e_steps, e_ep):
        for a in (ax1, ax2, ax3):
            a.axvline(x=step, color='navy', linestyle=':', alpha=0.3, linewidth=0.8)
    ax3.set_xlim(0, 2250)

    plt.tight_layout(pad=1.5)
    out = output_dir / "camembert_convergence.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIQUES D'ENTRAÎNEMENT SPACY
# ─────────────────────────────────────────────────────────────────────────────

def plot_spacy_dynamics(plt, np, output_dir: Path):
    """Courbe F1 vs taille du dataset + barplot des modèles de base."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Dynamiques d'entraînement spaCy NER", fontsize=12, y=1.02)

    # ── Sous-graphique 1 : Impact taille dataset ──────────────────────────
    sizes   = [d["size"]     for d in DATASET_SIZE_IMPACT]
    f1_blank= [d["f1_blank"] for d in DATASET_SIZE_IMPACT]
    f1_sm   = [d["f1_sm"]    for d in DATASET_SIZE_IMPACT]

    ax1.plot(sizes, f1_blank, color='#e67e22', linewidth=2.5, marker='o',
             markersize=8, label='spaCy blank (from scratch)')
    ax1.plot(sizes, f1_sm, color='#8e44ad', linewidth=2.5, marker='s',
             markersize=8, label='spaCy fr_core_sm (pré-entraîné)')

    # Point de convergence
    ax1.annotate('Convergence\n99.88% @ 10k',
                 xy=(10000, 99.88), xytext=(5500, 88),
                 fontsize=9, color='#8e44ad', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.2))

    # Ligne CamemBERT
    ax1.axhline(y=98.15, color='#2980b9', linestyle='--', linewidth=1.5,
                alpha=0.7, label='CamemBERT (98.15%)')

    # Ligne baseline
    ax1.axhline(y=41.04, color='#e74c3c', linestyle=':', linewidth=1.5,
                alpha=0.7, label='Baseline prépositions (41.04%)')

    ax1.set_xlabel('Taille du dataset d\'entraînement (phrases)')
    ax1.set_ylabel('F1-Score (%)')
    ax1.set_title('F1-Score vs Taille du Dataset\n(même architecture, même hyperparamètres)', pad=8)
    ax1.legend(loc='lower right', fontsize=8.5)
    ax1.set_ylim(25, 105)
    ax1.set_xscale('log')

    for size in [1000, 3000, 5000, 7000, 10000]:
        ax1.axvline(x=size, color='gray', linestyle=':', alpha=0.25)

    # ── Sous-graphique 2 : Comparaison modèles de base ───────────────────
    base_models = [
        ("Baseline\nPrépos.", 41.04, '#e74c3c'),
        ("spaCy\nblank",      69.13, '#e67e22'),
        ("spaCy\nsm",         75.65, '#f39c12'),
        ("spaCy\nlg",         81.50, '#27ae60'),
        ("spaCy\nmd",         86.42, '#16a085'),
        ("CamemBERT",         98.15, '#2980b9'),
        ("spaCy 10k\n★",      99.88, '#8e44ad'),
    ]
    names  = [m[0] for m in base_models]
    values = [m[1] for m in base_models]
    colors = [m[2] for m in base_models]

    bars = ax2.bar(names, values, color=colors, alpha=0.85, edgecolor='white',
                   linewidth=1.2, zorder=3)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax2.set_ylabel('F1-Score (%)')
    ax2.set_title('Comparaison F1-Score par Modèle\n(jeu de test 1 500 phrases)', pad=8)
    ax2.set_ylim(0, 112)
    ax2.tick_params(axis='x', labelsize=8.5)

    plt.tight_layout(pad=1.5)
    out = output_dir / "spacy_training_dynamics.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COURBE LOSS CAMEMBERT
# ─────────────────────────────────────────────────────────────────────────────

def plot_camembert_loss(plt, np, output_dir: Path):
    train_data = [d for d in CAMEMBERT_HISTORY if "loss" in d]
    eval_data  = [d for d in CAMEMBERT_HISTORY if "eval_loss" in d]

    t_steps = [d["step"] for d in train_data]
    t_loss  = [d["loss"]  for d in train_data]
    e_steps = [d["step"]      for d in eval_data]
    e_loss  = [d["eval_loss"] for d in eval_data]
    e_ep    = [d["epoch"]     for d in eval_data]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_steps, t_loss, color='#e74c3c', linewidth=2, alpha=0.85,
            label='Loss entraînement')
    ax.plot(e_steps, e_loss, color='#2980b9', linewidth=2.5, marker='o',
            markersize=7, label='Loss validation', zorder=5)

    for step, loss, ep in zip(e_steps, e_loss, e_ep):
        ax.annotate(f'Ep.{ep:.0f}: {loss:.3f}',
                    xy=(step, loss), xytext=(step + 35, loss + 0.014),
                    fontsize=8.5, color='#2980b9',
                    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=0.8))

    ax.axvspan(1752, 2190, alpha=0.07, color='green')
    ax.text(1960, 0.82, 'Convergence', fontsize=8.5, color='green',
            ha='center', style='italic')

    ax.set_xlabel('Steps d\'entraînement')
    ax.set_ylabel('Loss (Cross-Entropy)')
    ax.set_title('CamemBERT — Courbes de Loss Train / Validation\n(NER DEPART/ARRIVEE, 5 epochs)', pad=10)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 2250)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out = output_dir / "camembert_loss_curve.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. F1 PAR EPOCH CAMEMBERT
# ─────────────────────────────────────────────────────────────────────────────

def plot_camembert_f1(plt, np, output_dir: Path):
    eval_data = [(d["epoch"], d["eval_f1"] * 100)
                 for d in CAMEMBERT_HISTORY if "eval_f1" in d]
    epochs = [e for e, _ in eval_data]
    f1s    = [f for _, f in eval_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, f1s, color='#8e44ad', linewidth=2.5, marker='s',
            markersize=8, label='F1 Validation')

    for ep, f1 in zip(epochs, f1s):
        ax.annotate(f'{f1:.2f}%',
                    xy=(ep, f1), xytext=(ep + 0.05, f1 - 0.2),
                    fontsize=9, color='#8e44ad')

    ax.axhline(y=99.88, color='#e74c3c', linestyle='--', linewidth=1.5,
               alpha=0.7, label='spaCy 10k référence (99.88%)')
    ax.text(0.6, 99.90, 'spaCy 10k', fontsize=8.5, color='#e74c3c', style='italic')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('CamemBERT — F1-Score Validation par Epoch', pad=10)
    ax.legend(loc='lower right')
    ax.set_ylim(98.5, 100.3)
    ax.set_xlim(0.5, 5.5)

    plt.tight_layout()
    out = output_dir / "camembert_f1_curve.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BARPLOT COMPARAISON MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_bar(plt, np, output_dir: Path):
    models = [m["model"]  for m in MODEL_RESULTS_BAR]
    f1s    = [m["f1"]     for m in MODEL_RESULTS_BAR]
    colors = [m["color"]  for m in MODEL_RESULTS_BAR]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models, f1s, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=1.5, zorder=3)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{f1:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Baseline règles (prépositions)'),
        Patch(facecolor='#e67e22', label='spaCy without pre-training'),
        Patch(facecolor='#27ae60', label='spaCy pre-trained (variable size)'),
        Patch(facecolor='#2980b9', label='Transformer (CamemBERT)'),
        Patch(facecolor='#8e44ad', label='spaCy fine-tuné 10k ← MEILLEUR'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Comparaison NER — F1-Score sur 1 500 phrases de test', pad=12)
    ax.set_ylim(0, 112)

    plt.tight_layout()
    out = output_dir / "model_comparison_bar.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. IMPACT TAILLE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_size_impact(plt, np, output_dir: Path):
    sizes   = [d["size"]     for d in DATASET_SIZE_IMPACT]
    f1_blank= [d["f1_blank"] for d in DATASET_SIZE_IMPACT]
    f1_sm   = [d["f1_sm"]    for d in DATASET_SIZE_IMPACT]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, f1_blank, color='#e67e22', linewidth=2.5, marker='o',
            markersize=7, label='spaCy blank (from scratch)')
    ax.plot(sizes, f1_sm, color='#8e44ad', linewidth=2.5, marker='s',
            markersize=7, label='spaCy fr_core_news_sm (pré-entraîné)')

    ax.annotate('99.88% @ 10k',
                xy=(10000, 99.88), xytext=(6000, 91),
                fontsize=9, color='#8e44ad', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#8e44ad'))

    ax.axhline(y=98.15, color='#2980b9', linestyle='--', linewidth=1.5,
               alpha=0.6, label='CamemBERT (98.15%)')
    ax.axhline(y=41.04, color='#e74c3c', linestyle=':', linewidth=1.5,
               alpha=0.6, label='Baseline prépositions (41.04%)')

    ax.set_xlabel('Taille du dataset d\'entraînement')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Impact de la Taille du Dataset sur le F1-Score NER', pad=10)
    ax.legend(loc='lower right')
    ax.set_ylim(25, 106)
    ax.set_xscale('log')

    plt.tight_layout()
    out = output_dir / "dataset_size_impact.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GÉNÉRATION DES VISUALISATIONS KEYNOTE")
    print("=" * 60)

    try:
        plt, np = setup()
    except ImportError as e:
        print(f"[ERROR] {e} — installez avec: pip install matplotlib numpy")
        return

    print("\nGénération en cours...")
    plot_confusion_matrices(plt, np, output_dir)
    plot_camembert_convergence(plt, np, output_dir)
    plot_spacy_dynamics(plt, np, output_dir)
    plot_camembert_loss(plt, np, output_dir)
    plot_camembert_f1(plt, np, output_dir)
    plot_model_comparison_bar(plt, np, output_dir)
    plot_dataset_size_impact(plt, np, output_dir)

    print(f"\n[OK] 7 graphiques sauvegardés dans {output_dir}/")
    print()
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name:<42} {f.stat().st_size // 1024:>4} KB")


if __name__ == "__main__":
    main()
