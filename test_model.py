"""
Test du modèle NER entraîné
"""
import spacy

print("Chargement du modele...")
nlp = spacy.load("models/ner_10k/model-best")
print("Modele charge!\n")

phrases = [
    "Je veux aller de Paris a Lyon",
    "Direction Marseille depuis Bordeaux",
    "Florence et moi voulons aller a Annecy depuis Paris",
    "Bonjour comment allez-vous",
    "Train Nantes Rennes pour demain",
    "Je pars de Lille pour Strasbourg",
]

print("=" * 60)
print("TEST DU MODELE NER")
print("=" * 60)

for phrase in phrases:
    doc = nlp(phrase)
    print(f"\nPhrase: \"{phrase}\"")

    depart = None
    arrivee = None

    for ent in doc.ents:
        print(f"  -> {ent.text:15} = {ent.label_}")
        if ent.label_ == "DEPART":
            depart = ent.text
        elif ent.label_ == "ARRIVEE":
            arrivee = ent.text

    if not doc.ents:
        print("  -> (aucune entite detectee)")
    else:
        print(f"  RESULTAT: {depart} --> {arrivee}")

print("\n" + "=" * 60)
print("TEST TERMINE!")
print("=" * 60)
