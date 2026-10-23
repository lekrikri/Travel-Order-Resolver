"""Évalue les 8 moteurs NLP et enregistre les résultats dans MLflow.

Mêmes métriques que scripts/evaluate_8models.py (celles du README) :
précision, rappel, F1 et exact match au niveau des villes, sur data/processed/test.json.

Le modèle spaCy 10k est aussi enregistré dans le Model Registry.

Prérequis : un serveur MLflow lancé à la racine du projet
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Lancement (depuis la racine du projet) :
    python -m src.evaluation.mlflow_eval
"""

import hashlib
import os

import mlflow
import mlflow.spacy
from mlflow import MlflowClient

from scripts.evaluate_8models import DATA, eval_extract_fn, eval_spacy_model, load_test

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "travel-order-resolver-ner"
MODELE_REGISTRE = "tor-ner-spacy-10k"

# (nom du run, chemin du modèle, type)
MODELES = [
    ("rule_based", None, "regles"),
    ("spacy_blank", "models/ner_blank/model-best", "spacy"),
    ("spacy_sm", "models/ner_sm", "spacy"),
    ("spacy_md", "models/ner_md", "spacy"),
    ("spacy_lg", "models/ner_lg", "spacy"),
    ("spacy_10k", "models/ner_10k/model-best", "spacy"),
    ("crf_ner", None, "crf"),
    ("camembert_ner", "Jean-Baptiste/camembert-ner", "transformer"),
]


def empreinte_fichier(chemin):
    """Empreinte SHA-256 courte : prouve sur quelle version du jeu de test le run a été mesuré."""
    with open(chemin, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def evaluer(type_modele, chemin, test_data):
    """Lance l'évaluation adaptée au type de moteur."""
    if type_modele == "spacy":
        return eval_spacy_model(chemin, test_data)
    if type_modele == "regles":
        from src.nlp.rule_based import RuleBasedEngine
        return eval_extract_fn(RuleBasedEngine().extract, test_data)
    if type_modele == "crf":
        from src.nlp.crf_ner import CRFNEREngine
        return eval_extract_fn(CRFNEREngine().extract, test_data)
    if type_modele == "transformer":
        from src.nlp.transformer_ner import TransformerNEREngine
        return eval_extract_fn(TransformerNEREngine().extract, test_data)
    raise ValueError(f"Type de modèle inconnu : {type_modele!r}")


def enregistrer_modele_spacy(chemin):
    """Logue le modèle spaCy dans le run courant et le publie dans le Model Registry."""
    import spacy

    nlp = spacy.load(chemin)
    info = mlflow.spacy.log_model(nlp, name="model", registered_model_name=MODELE_REGISTRE)
    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODELE_REGISTRE, "candidate", version)
    print(f"     → registre : {MODELE_REGISTRE} v{version} (alias « candidate »)")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    test_data = load_test()
    empreinte = empreinte_fichier(DATA)
    print(f"Jeu de test : {len(test_data)} exemples (sha256 {empreinte})")

    resultats = {}
    for nom, chemin, type_modele in MODELES:
        print(f"- {nom} …", flush=True)
        with mlflow.start_run(run_name=nom):
            mlflow.log_param("type_modele", type_modele)
            mlflow.log_param("chemin", chemin)
            mlflow.log_param("nombre_d_exemples_de_test", len(test_data))
            mlflow.set_tags({"jeu_de_test": "hard_v2", "jeu_de_test_sha256": empreinte})

            metrics = evaluer(type_modele, chemin, test_data)
            if metrics is None:
                # Le modèle n'a pas pu être chargé : on garde la trace de l'échec dans le run.
                mlflow.set_tag("statut", "echec_chargement")
                print("     échec du chargement, run marqué en échec")
                continue

            mlflow.log_metrics(metrics)
            resultats[nom] = metrics
            print(f"     F1 {metrics['f1']} %, exact match {metrics['exact_match']} %")

            if nom == "spacy_10k":
                enregistrer_modele_spacy(chemin)

    meilleur = max(resultats, key=lambda n: resultats[n]["f1"])
    print(f"\nMeilleur F1 : {meilleur} ({resultats[meilleur]['f1']} %)")


if __name__ == "__main__":
    main()
