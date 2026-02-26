#!/usr/bin/env python3
"""
Convertit RAPPORT_FINAL.md en PDF professionnel
"""
import subprocess
import sys

def generate_pdf():
    """Génère le PDF depuis le markdown"""

    print("="*70)
    print("GÉNÉRATION DU PDF PROFESSIONNEL")
    print("="*70)

    # Vérifier si pandoc est installé
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        print("[OK] Pandoc detecte")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERREUR] Pandoc n'est pas installe")
        print("\nInstallation:")
        print("  Ubuntu/Debian: sudo apt-get install pandoc texlive-latex-base")
        print("  Windows: choco install pandoc miktex")
        print("  Mac: brew install pandoc basictex")
        return False

    # Générer le PDF avec pandoc
    input_file = "RAPPORT_FINAL.md"
    output_file = "RAPPORT_FINAL.pdf"

    print(f"\n[PDF] Conversion {input_file} -> {output_file}")

    try:
        # Commande pandoc avec options pour un beau PDF professionnel
        # Utilisation de xelatex pour support Unicode complet (emojis, etc.)
        cmd = [
            "pandoc",
            input_file,
            "-o", output_file,
            "--pdf-engine=xelatex",
            "--toc",  # Table des matieres
            "--toc-depth=3",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "-V", "linkcolor=blue",
            "-V", "urlcolor=blue",
            "-V", "mainfont=DejaVu Sans",
            "--highlight-style=tango",
            "--metadata", "title=Rapport Final - Travel Order Resolver",
            "--metadata", "author=T-AIA-911-PAR_14",
            "--metadata", "date=28/12/2024"
        ]

        subprocess.run(cmd, check=True)
        print(f"[OK] PDF genere avec succes: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] Erreur lors de la generation: {e}")
        return False

if __name__ == "__main__":
    success = generate_pdf()
    sys.exit(0 if success else 1)
