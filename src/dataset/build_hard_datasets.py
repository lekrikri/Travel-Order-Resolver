"""
Build hard datasets from hand-crafted sentences.

Outputs:
  - data/hard/train.json + val.json + test.json  (spaCy entity-span format)
  - data/hard/train_10k.csv                      (CSV with <Dep>/<Arr> tags)

Run: python3 src/dataset/build_hard_datasets.py
"""

import csv
import json
import random
from pathlib import Path

# =============================================================================
# HAND-CRAFTED SENTENCES
# =============================================================================
# Format: (sentence, dep_as_written, arr_as_written, dep_canonical, arr_canonical, category)
#
# dep_canonical / arr_canonical = real city name (for ground truth).
# dep_as_written / arr_as_written = how it appears in the sentence (may have typos).
# For invalid sentences: all fields are None except sentence and category.
# For non-existent cities: canonical is the written form (no SNCF match expected).

def _s(text, dep, arr, dep_real=None, arr_real=None, cat="standard"):
    """Shorthand to build a sentence tuple."""
    return (text, dep, arr, dep_real or dep, arr_real or arr, cat)

def _invalid(text):
    return (text, None, None, None, None, "invalid")


SENTENCES = [
    # =========================================================================
    # STANDARD — classic formulations
    # =========================================================================
    _s("Je veux aller de Nantes a Bordeaux", "Nantes", "Bordeaux"),
    _s("Un billet de Lille vers Marseille", "Lille", "Marseille"),
    _s("Je pars de Lyon pour Toulouse", "Lyon", "Toulouse"),
    _s("Train Strasbourg Montpellier", "Strasbourg", "Montpellier"),
    _s("Rennes - Dijon s'il vous plait", "Rennes", "Dijon"),
    _s("Aller simple de Metz a Grenoble", "Metz", "Grenoble"),
    _s("Un aller-retour Caen Angers", "Caen", "Angers"),
    _s("Je voudrais un billet de Pau a Tours", "Pau", "Tours"),
    _s("Reservation de Reims a Perpignan", "Reims", "Perpignan"),
    _s("Depuis Avignon jusqu'a Rouen", "Avignon", "Rouen"),
    _s("De Poitiers a Chambery", "Poitiers", "Chambery"),
    _s("Besoin d'un train au depart de Limoges vers Calais", "Limoges", "Calais"),
    _s("Brest -> Annecy", "Brest", "Annecy"),
    _s("Depart Blois arrivee Bourges", "Blois", "Bourges"),
    _s("Je cherche un train de La Rochelle a Colmar", "La Rochelle", "Colmar"),
    _s("Aller de Niort a Bayonne", "Niort", "Bayonne"),
    _s("Voyage de Dunkerque a Rodez", "Dunkerque", "Rodez"),
    _s("Trajet Vannes vers Cahors en TGV", "Vannes", "Cahors"),
    _s("Je dois me rendre a Gap depuis Lorient", "Lorient", "Gap"),
    _s("Prendre le train de Troyes a Dax", "Troyes", "Dax"),
    _s("Un billet pour Epinal depuis Angouleme svp", "Angouleme", "Epinal"),
    _s("Tarbes a Sete par le TER", "Tarbes", "Sete"),

    # =========================================================================
    # INVERSE — arrival mentioned before departure
    # =========================================================================
    _s("Direction Toulouse depuis Nantes", "Nantes", "Toulouse", cat="inverse"),
    _s("Vers Lyon en partant de Bordeaux", "Bordeaux", "Lyon", cat="inverse"),
    _s("Pour aller a Marseille je pars de Rennes", "Rennes", "Marseille", cat="inverse"),
    _s("A destination de Nice depuis Dijon", "Dijon", "Nice", cat="inverse"),
    _s("Grenoble au depart de Strasbourg", "Strasbourg", "Grenoble", cat="inverse"),
    _s("Je rentre a Lille depuis Tours", "Tours", "Lille", cat="inverse"),
    _s("Je vais a Montpellier en partant de Reims", "Reims", "Montpellier", cat="inverse"),
    _s("Un billet pour Perpignan je pars de Metz", "Metz", "Perpignan", cat="inverse"),
    _s("Direction Annecy au depart de Caen", "Caen", "Annecy", cat="inverse"),
    _s("Vers Brest depuis Le Havre", "Le Havre", "Brest", cat="inverse"),
    _s("Destination Avignon au depart de Clermont-Ferrand", "Clermont-Ferrand", "Avignon", cat="inverse"),
    _s("A Chambery en venant de Pau", "Pau", "Chambery", cat="inverse"),

    # =========================================================================
    # HABITE — departure signaled by habitation verb
    # =========================================================================
    _s("J'habite a Lyon et je veux aller a Nice", "Lyon", "Nice", cat="habite"),
    _s("On habite a Bordeaux et on va a Strasbourg", "Bordeaux", "Strasbourg", cat="habite"),
    _s("Je vis a Toulouse je dois aller a Nantes", "Toulouse", "Nantes", cat="habite"),
    _s("On habite sur Albi on part a Tours", "Albi", "Tours", cat="habite"),
    _s("On vit a Rennes et on veut rejoindre Marseille", "Rennes", "Marseille", cat="habite"),
    _s("Je reside a Dijon et je dois me rendre a Montpellier", "Dijon", "Montpellier", cat="habite"),
    _s("On habite a Metz mais on bosse a Strasbourg", "Metz", "Strasbourg", cat="habite"),
    _s("J'habite Clermont-Ferrand et je vais a Paris", "Clermont-Ferrand", "Paris", cat="habite"),
    # Destination before departure
    _s("On va a Angers on habite a Grenoble", "Grenoble", "Angers", cat="habite"),
    _s("Je pars a Lille j'habite a Perpignan", "Perpignan", "Lille", cat="habite"),
    _s("On part en vacances a Biarritz on habite a Rouen", "Rouen", "Biarritz", cat="habite"),
    _s("Je vais a Cannes je vis a Amiens", "Amiens", "Cannes", cat="habite"),

    # =========================================================================
    # MULTI-SENTENCE — several sentences, realistic user input
    # =========================================================================
    _s("Je pars en vacances a Marseille. On habite a Albi.", "Albi", "Marseille", cat="multi"),
    _s("On va a Lyon. On habite a Nantes.", "Nantes", "Lyon", cat="multi"),
    _s("Je vais a Bordeaux demain. Je suis a Lille.", "Lille", "Bordeaux", cat="multi"),
    _s("On part a Toulouse ce weekend. On vit a Reims.", "Reims", "Toulouse", cat="multi"),
    _s("Destination Grenoble. Depart depuis Poitiers.", "Poitiers", "Grenoble", cat="multi"),
    _s("On habite a Caen. On part a Nice demain matin.", "Caen", "Nice", cat="multi"),
    _s("Je vis a Rennes. Je veux aller a Montpellier.", "Rennes", "Montpellier", cat="multi"),
    _s("Je suis a Dijon actuellement. Je dois rejoindre Pau.", "Dijon", "Pau", cat="multi"),
    _s("Mon train part de Avignon. J'arrive a Annecy ce soir.", "Avignon", "Annecy", cat="multi"),
    _s("Je suis bloque a Metz. Il faut que j'aille a Tours.", "Metz", "Tours", cat="multi"),
    _s("Bonjour. Je cherche un train depuis Rouen vers Bayonne. Merci.", "Rouen", "Bayonne", cat="multi"),
    _s("Salut. On est a Strasbourg et on va a Perpignan. C'est pour 4 personnes.", "Strasbourg", "Perpignan", cat="multi"),

    # =========================================================================
    # PERSON DISTRACTOR — first names that are NOT cities
    # =========================================================================
    _s("Marie et moi on va de Lyon a Nice", "Lyon", "Nice", cat="person"),
    _s("Avec Jean on part de Bordeaux pour Toulouse", "Bordeaux", "Toulouse", cat="person"),
    _s("Sophie m'attend a Marseille je suis a Rennes", "Rennes", "Marseille", cat="person"),
    _s("Je rejoins Lucas a Nantes depuis Lille", "Lille", "Nantes", cat="person"),
    _s("Pierre veut qu'on parte de Dijon pour aller a Strasbourg", "Dijon", "Strasbourg", cat="person"),
    _s("Avec mon ami Thomas on va de Grenoble a Metz", "Grenoble", "Metz", cat="person"),
    _s("Emma et moi on prend le train de Caen a Tours", "Caen", "Tours", cat="person"),
    _s("On part avec Camille et Hugo de Reims a Pau", "Reims", "Pau", cat="person"),
    _s("Je voyage avec Antoine de Blois a Colmar", "Blois", "Colmar", cat="person"),
    _s("Clara m'a dit de venir a Annecy depuis Rouen", "Rouen", "Annecy", cat="person"),

    # =========================================================================
    # PERSON-CITY DISTRACTOR — city names used as person names (THE trap)
    # =========================================================================
    _s("Mon amie Paris et moi on va de Albi a Tours", "Albi", "Tours", cat="person_city"),
    _s("Avec Nancy on part de Lyon vers Marseille", "Lyon", "Marseille", cat="person_city"),
    _s("Mon pote Nice m'attend je pars de Bordeaux a Toulouse", "Bordeaux", "Toulouse", cat="person_city"),
    _s("Valence et moi on veut aller de Rennes a Nantes", "Rennes", "Nantes", cat="person_city"),
    _s("Je pars en vacances avec mon amie Paris de Dijon vers Grenoble", "Dijon", "Grenoble", cat="person_city"),
    _s("Mon amie Paris et moi on va a Angers. On habite sur Albi a 1h de Toulouse.", "Albi", "Angers", cat="person_city"),
    _s("Mon amie Paris et moi voulons aller en vacances a Tours. On habite a Albi.", "Albi", "Tours", cat="person_city"),
    _s("Avec Nancy on cherche un train de Strasbourg a Montpellier", "Strasbourg", "Montpellier", cat="person_city"),
    _s("Nice le chien et moi on part de Caen vers Reims", "Caen", "Reims", cat="person_city"),
    _s("Ma cousine Valence et moi on va de Metz a Perpignan", "Metz", "Perpignan", cat="person_city"),
    _s("Mon copain Paris m'accompagne de Poitiers a Rouen", "Poitiers", "Rouen", cat="person_city"),
    _s("Nancy ma soeur vient me chercher a Pau je suis a Limoges", "Limoges", "Pau", cat="person_city"),

    # =========================================================================
    # TYPOS — spelling mistakes on city names
    # =========================================================================
    _s("Je veux aller de Bordeau a Marseil", "Bordeau", "Marseil", "Bordeaux", "Marseille", "typo"),
    _s("Un billet Toulous Strasbour svp", "Toulous", "Strasbour", "Toulouse", "Strasbourg", "typo"),
    _s("De Monpellier a Grenobe", "Monpellier", "Grenobe", "Montpellier", "Grenoble", "typo"),
    _s("Train Perpinian Clermon-Ferrand", "Perpinian", "Clermon-Ferrand", "Perpignan", "Clermont-Ferrand", "typo"),
    _s("Billet Angoulem Carcasson", "Angoulem", "Carcasson", "Angouleme", "Carcassonne", "typo"),
    _s("Chamberi a Dunkerke", "Chamberi", "Dunkerke", "Chambery", "Dunkerque", "typo"),
    _s("De Lile a Nante", "Lile", "Nante", "Lille", "Nantes", "typo"),
    _s("Billet Lio Marsseille", "Lio", "Marsseille", "Lyon", "Marseille", "typo"),
    _s("Je par de Ren vers Toulon", "Ren", "Toulon", "Rennes", "Toulon", "typo"),
    _s("De Nansi a Mettz", "Nansi", "Mettz", "Nancy", "Metz", "typo"),
    _s("Depart Roueen arrivee Caenn", "Roueen", "Caenn", "Rouen", "Caen", "typo"),
    _s("Orleans a Chartre", "Orleans", "Chartre", "Orleans", "Chartres", "typo"),
    _s("De Mulhous a Bezancon", "Mulhous", "Bezancon", "Mulhouse", "Besancon", "typo"),
    _s("Billet Bayone Tarbbes", "Bayone", "Tarbbes", "Bayonne", "Tarbes", "typo"),
    _s("Je vais a Avinion depuis Nimmes", "Nimmes", "Avinion", "Nimes", "Avignon", "typo"),
    _s("Trin Brset Loriant", "Brset", "Loriant", "Brest", "Lorient", "typo"),

    # =========================================================================
    # SMS / INFORMAL — abbreviated, slang, no punctuation
    # =========================================================================
    _s("trin Lyon Marseille dem1", "Lyon", "Marseille", cat="sms"),
    _s("billet Nantes Bordeaux stp", "Nantes", "Bordeaux", cat="sms"),
    _s("Paris Lille vite", "Paris", "Lille", cat="sms"),
    _s("jsuis a Rennes jvais a Nice", "Rennes", "Nice", cat="sms"),
    _s("chui a Toulouse faut que jaille a Grenoble", "Toulouse", "Grenoble", cat="sms"),
    _s("train Dijon Strasbourg asap", "Dijon", "Strasbourg", cat="sms"),
    _s("Metz a Reims svp", "Metz", "Reims", cat="sms"),
    _s("go Rouen Caen", "Rouen", "Caen", cat="sms"),
    _s("billet Tours Angers urgent", "Tours", "Angers", cat="sms"),
    _s("de Lyon a Montpellier le + vite possible", "Lyon", "Montpellier", cat="sms"),
    _s("Perpignan Pau pk", "Perpignan", "Pau", cat="sms"),
    _s("lol jsuis a Blois et jdois aller a Bourges", "Blois", "Bourges", cat="sms"),
    _s("train bordx toulouse", "bordx", "toulouse", "Bordeaux", "Toulouse", "sms"),
    _s("besoin billet marseil nice rapidos", "marseil", "nice", "Marseille", "Nice", "sms"),

    # =========================================================================
    # QUESTIONS — interrogative forms
    # =========================================================================
    _s("Est-ce qu'il y a des trains de Nantes a Bordeaux ?", "Nantes", "Bordeaux", cat="question"),
    _s("Quelle heure le prochain train Lyon Marseille ?", "Lyon", "Marseille", cat="question"),
    _s("C'est combien un billet Toulouse Strasbourg ?", "Toulouse", "Strasbourg", cat="question"),
    _s("Il reste des places sur le Rennes Nice ?", "Rennes", "Nice", cat="question"),
    _s("Combien de temps pour aller de Dijon a Grenoble ?", "Dijon", "Grenoble", cat="question"),
    _s("A quelle heure part le train de Lille a Montpellier ?", "Lille", "Montpellier", cat="question"),
    _s("Quel est le prix d'un Metz Reims ?", "Metz", "Reims", cat="question"),
    _s("Y a un train direct Pau Perpignan ?", "Pau", "Perpignan", cat="question"),
    _s("On peut aller de Rouen a Annecy sans correspondance ?", "Rouen", "Annecy", cat="question"),
    _s("Ca existe un train de nuit Bordeaux Nice ?", "Bordeaux", "Nice", cat="question"),

    # =========================================================================
    # CONTEXT / DATES — with temporal or contextual noise
    # =========================================================================
    _s("Pour le travail je dois aller de Lyon a Paris lundi", "Lyon", "Paris", cat="context"),
    _s("Pour mes vacances un billet Nantes Marseille en aout", "Nantes", "Marseille", cat="context"),
    _s("Pour un weekend un train de Bordeaux a Nice vendredi soir", "Bordeaux", "Nice", cat="context"),
    _s("Pour une reunion je pars de Toulouse a Strasbourg mardi matin", "Toulouse", "Strasbourg", cat="context"),
    _s("Avec mes enfants on va de Rennes a Montpellier pendant les vacances", "Rennes", "Montpellier", cat="context"),
    _s("Billet Dijon Grenoble pour demain 8h", "Dijon", "Grenoble", cat="context"),
    _s("Reservation Lille Reims pour le 15 mars", "Lille", "Reims", cat="context"),
    _s("Premier train du matin Metz Tours", "Metz", "Tours", cat="context"),
    _s("Dernier train Caen Angers ce soir", "Caen", "Angers", cat="context"),
    _s("Train de nuit Rouen Perpignan samedi", "Rouen", "Perpignan", cat="context"),
    _s("Pour un mariage a Pau je pars de Limoges vendredi", "Limoges", "Pau", cat="context"),
    _s("Billet 1ere classe Lyon Nice pour 2 personnes", "Lyon", "Nice", cat="context"),

    # =========================================================================
    # LONG / CONVOLUTED — realistic verbose user input
    # =========================================================================
    _s("Bonjour je voudrais savoir s'il y a un train direct de Nantes a Bordeaux s'il vous plait merci", "Nantes", "Bordeaux", cat="long"),
    _s("Je souhaiterais reserver un billet de train au depart de Lille avec une arrivee a Toulouse", "Lille", "Toulouse", cat="long"),
    _s("Pouvez-vous me trouver un train qui part de Lyon et qui arrive a Marseille avant midi ?", "Lyon", "Marseille", cat="long"),
    _s("J'ai besoin d'un billet pour voyager de Strasbourg jusqu'a Montpellier si possible en TGV", "Strasbourg", "Montpellier", cat="long"),
    _s("Bonjour madame je cherche a reserver une place dans un train allant de Rennes vers Dijon pour moi et ma femme", "Rennes", "Dijon", cat="long"),
    _s("Excusez-moi est-ce que je peux avoir un billet de Grenoble pour aller a Annecy c'est pour mon fils qui est etudiant", "Grenoble", "Annecy", cat="long"),
    _s("En fait je suis a Rouen la et je dois absolument aller a Bordeaux avant demain soir c'est possible ?", "Rouen", "Bordeaux", cat="long"),
    _s("Alors voila ma situation j'habite a Clermont-Ferrand et je dois aller a Paris pour un entretien mardi", "Clermont-Ferrand", "Paris", cat="long"),

    # =========================================================================
    # NON-EXISTENT SNCF CITIES — foreign or very small / invented
    # =========================================================================
    _s("Je veux aller de Bruxelles a Lyon", "Bruxelles", "Lyon", cat="foreign"),
    _s("Un billet de Geneve a Paris", "Geneve", "Paris", cat="foreign"),
    _s("Train de Londres a Lille", "Londres", "Lille", cat="foreign"),
    _s("Billet Amsterdam Strasbourg", "Amsterdam", "Strasbourg", cat="foreign"),
    _s("Zurich a Mulhouse", "Zurich", "Mulhouse", cat="foreign"),
    _s("De Barcelona a Perpignan", "Barcelona", "Perpignan", cat="foreign"),
    _s("Luxembourg a Metz", "Luxembourg", "Metz", cat="foreign"),
    _s("Je pars de Turin pour aller a Nice", "Turin", "Nice", cat="foreign"),
    _s("Billet Francfort Strasbourg en ICE", "Francfort", "Strasbourg", cat="foreign"),
    _s("De Monaco a Marseille", "Monaco", "Marseille", cat="foreign"),
    # Completely made-up
    _s("Un train de Villemont a Bordeaux", "Villemont", "Bordeaux", cat="unknown"),
    _s("De Beaurivage a Toulouse", "Beaurivage", "Toulouse", cat="unknown"),
    _s("Billet Saint-Clairin Lyon", "Saint-Clairin", "Lyon", cat="unknown"),
    _s("Depart Montfleury arrivee Nantes", "Montfleury", "Nantes", cat="unknown"),
    _s("De Trifouillis-les-Oies a Paris", "Trifouillis-les-Oies", "Paris", cat="unknown"),
    _s("Je pars de Rennes pour Plougastel-du-Nord", "Rennes", "Plougastel-du-Nord", cat="unknown"),
    _s("Train Tataouine Lyon", "Tataouine", "Lyon", cat="unknown"),
    _s("De Poudlard a Marseille svp", "Poudlard", "Marseille", cat="unknown"),

    # =========================================================================
    # TRICKY AMBIGUITY — multiple city mentions, context cities, etc.
    # =========================================================================
    _s("Je suis a Paris mais je dois aller a Lyon en passant par Dijon", "Paris", "Lyon", cat="tricky"),
    _s("On vient de Marseille et on va a Bordeaux via Toulouse", "Marseille", "Bordeaux", cat="tricky"),
    _s("Mon amie de Nantes m'attend a Rennes", "Nantes", "Rennes", cat="tricky"),
    _s("Je travaille a Lyon mais j'habite a Grenoble et je dois aller a Paris", "Grenoble", "Paris", cat="tricky"),
    _s("On est parti de Nice ce matin on est a Marseille la et on va a Toulouse", "Marseille", "Toulouse", cat="tricky"),
    _s("Le train de Bordeaux arrive a Toulouse a 14h moi je veux Toulouse Montpellier", "Toulouse", "Montpellier", cat="tricky"),
    _s("J'ai rate mon train a Lyon il me faut un Lyon Lille rapide", "Lyon", "Lille", cat="tricky"),
    _s("Je pars en vacances avec mon amie Paris vers Angers. On habite sur Albi a 1h de Toulouse.", "Albi", "Angers", cat="tricky"),
    _s("De Tours a Blois c'est court mais de Tours a Lyon c'est ce que je veux", "Tours", "Lyon", cat="tricky"),
    _s("Mon pere habite a Nantes ma mere a Rennes moi je vais de Lyon a Marseille", "Lyon", "Marseille", cat="tricky"),

    # =========================================================================
    # TYPO + MULTI / TYPO + SMS — combined difficulties
    # =========================================================================
    _s("jsuis a Toulous faut que jaille a Bordeau", "Toulous", "Bordeau", "Toulouse", "Bordeaux", "typo_sms"),
    _s("billet Marseil Monpellier stp", "Marseil", "Monpellier", "Marseille", "Montpellier", "typo_sms"),
    _s("On habite a Grenobe. On va a Angoulem.", "Grenobe", "Angoulem", "Grenoble", "Angouleme", "typo_multi"),
    _s("Je vis a Chamberry et je vais a Anneci", "Chamberry", "Anneci", "Chambery", "Annecy", "typo"),
    _s("Lile Ren vite stp", "Lile", "Ren", "Lille", "Rennes", "typo_sms"),
    _s("Mon amie Paris et moi on va de Bordeau a Marseil", "Bordeau", "Marseil", "Bordeaux", "Marseille", "typo_person_city"),
    _s("On habite a Toulous. On part a Strasbour demain.", "Toulous", "Strasbour", "Toulouse", "Strasbourg", "typo_multi"),
    _s("Avec Nancy on part de Grenobe a Monpellier", "Grenobe", "Monpellier", "Grenoble", "Montpellier", "typo_person_city"),

    # =========================================================================
    # INVALID / NEGATIVE — no route, model must return nothing
    # =========================================================================
    _invalid("Bonjour comment allez-vous"),
    _invalid("Quel temps fait-il aujourd'hui"),
    _invalid("Je cherche un hotel pas cher"),
    _invalid("Ou sont les toilettes"),
    _invalid("Merci beaucoup au revoir"),
    _invalid("Un cafe s'il vous plait"),
    _invalid("Quelle heure est-il"),
    _invalid("Je voudrais un sandwich"),
    _invalid("Combien ca coute"),
    _invalid("Au revoir et bonne journee"),
    _invalid("J'ai perdu mon billet de train"),
    _invalid("Le train est en retard de 30 minutes"),
    _invalid("Ou est le quai numero 3"),
    _invalid("Je veux annuler ma reservation"),
    _invalid("Comment fonctionne l'appli SNCF"),
    _invalid("Mon train a ete supprime"),
    _invalid("Je cherche un taxi"),
    _invalid("Ou puis-je manger dans la gare"),
    _invalid("Y a du wifi dans le TGV"),
    _invalid("Comment joindre le service client SNCF"),
    _invalid("Je suis en retard"),
    _invalid("Mon bagage est perdu"),
    _invalid("C'est quoi le numero de mon train"),
    _invalid("La gare est fermee"),
    _invalid("J'ai oublie mon telephone dans le train"),
]


# =============================================================================
# CONVERSION
# =============================================================================

def find_span(text, needle):
    """Find (start, end) character span of needle in text."""
    idx = text.find(needle)
    if idx == -1:
        return None
    return (idx, idx + len(needle))


def to_spacy_format(sentences):
    """Convert to spaCy entity-span JSON format."""
    data = []
    for text, dep, arr, dep_real, arr_real, cat in sentences:
        if dep is None:  # invalid
            data.append({
                "text": text,
                "entities": [],
                "depart": "INVALID",
                "arrivee": "INVALID",
                "pattern_type": cat,
                "valid": False,
            })
            continue

        dep_span = find_span(text, dep)
        arr_span = find_span(text, arr)

        if dep_span is None or arr_span is None:
            print(f"  WARNING: span not found in '{text}' for dep='{dep}' arr='{arr}'")
            continue

        entities = [
            (dep_span[0], dep_span[1], "DEPART"),
            (arr_span[0], arr_span[1], "ARRIVEE"),
        ]
        entities.sort(key=lambda x: x[0])

        data.append({
            "text": text,
            "entities": entities,
            "depart": dep_real,
            "arrivee": arr_real,
            "pattern_type": cat,
            "valid": True,
        })
    return data


def to_csv_format(sentences):
    """Convert to CSV with <Dep>/<Arr> inline tags."""
    rows = []
    for text, dep, arr, dep_real, arr_real, cat in sentences:
        if dep is None:
            rows.append({"sentence": text, "departure": "INVALID", "arrival": "INVALID"})
            continue

        tagged = text.replace(dep, f"<Dep>{dep}</Dep>", 1)
        tagged = tagged.replace(arr, f"<Arr>{arr}</Arr>", 1)
        rows.append({
            "sentence": tagged,
            "departure": dep_real,
            "arrival": arr_real,
        })
    return rows


def split_data(data, train_ratio=0.70, val_ratio=0.15, seed=2026):
    """Split into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    return shuffled[:t], shuffled[t:v], shuffled[v:]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BUILD HARD DATASETS (hand-crafted)")
    print("=" * 60)

    output_dir = Path("data/hard")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    from collections import Counter
    cats = Counter(cat for _, _, _, _, _, cat in SENTENCES)
    valid = sum(1 for s in SENTENCES if s[1] is not None)
    print(f"\nTotal sentences: {len(SENTENCES)}")
    print(f"  Valid: {valid}, Invalid: {len(SENTENCES) - valid}")
    print(f"  Categories: {dict(cats)}")

    # --- spaCy format ---
    spacy_data = to_spacy_format(SENTENCES)
    train, val, test = split_data(spacy_data)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  spaCy: {path} ({len(split)} examples)")

    # --- CSV 10k format ---
    csv_data = to_csv_format(SENTENCES)
    csv_path = output_dir / "sentences_tagged.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "departure", "arrival"])
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"  CSV:   {csv_path} ({len(csv_data)} rows)")

    # --- Show samples per category ---
    print("\n--- Samples per category ---")
    shown = set()
    for text, dep, arr, dep_real, arr_real, cat in SENTENCES:
        if cat not in shown:
            shown.add(cat)
            if dep:
                print(f"  [{cat:14s}] {text[:75]}  ->  {dep_real} -> {arr_real}")
            else:
                print(f"  [{cat:14s}] {text[:75]}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("  data/hard/train.json + val.json + test.json  (spaCy)")
    print("  data/hard/sentences_tagged.csv               (CSV)")
    print("=" * 60)
