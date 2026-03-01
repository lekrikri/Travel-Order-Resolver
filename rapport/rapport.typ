// =============================================================================
// Travel Order Resolver -- Rapport d'Experimentation
// Format: Typst → PDF  |  Compile: typst compile rapport.typ
// =============================================================================

#set document(
  title: "Rapport d'Experimentation - Travel Order Resolver",
  author: ("Epitech MSc Pro 2026 - T-AIA-911-PAR_14",),
  date: datetime(year: 2026, month: 3, day: 1),
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2cm, right: 2cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: gray)
      Travel Order Resolver -- T-AIA-911-PAR_14
      #h(1fr)
      Epitech MSc Pro 2026
    ]
  },
)

#set text(font: "New Computer Modern", size: 11pt, lang: "fr")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(0.8em)
  block(text(14pt, weight: "bold", it))
  v(0.3em)
}

#show heading.where(level: 2): it => {
  v(0.5em)
  block(text(12pt, weight: "bold", it))
  v(0.2em)
}

// --- Page de garde -----------------------------------------------------------

#align(center)[
  #v(3cm)
  #text(24pt, weight: "bold")[Travel Order Resolver]
  #v(0.5cm)
  #text(14pt, fill: rgb("#555"))[Rapport d'Experimentation]
  #v(2cm)
  #text(12pt)[
    *Projet* : T-AIA-911-PAR_14 \
    *Equipe* : Epitech MSc Pro 2026 \
    *Date* : Mars 2026
  ]
  #v(3cm)
  #line(length: 60%, stroke: 0.5pt + gray)
  #v(0.5cm)
  #text(10pt, fill: gray)[
    Extraction d'entites NER, pathfinding ferroviaire \
    et benchmark multi-criteres
  ]
]

#pagebreak()

// --- Sommaire ----------------------------------------------------------------

#outline(title: "Sommaire", indent: 1.5em, depth: 2)

#pagebreak()

// =============================================================================
= Introduction
// =============================================================================

== Contexte

Le projet *Travel Order Resolver* vise a extraire automatiquement les villes de depart et d'arrivee a partir de phrases en langage naturel francais, puis a calculer un itineraire ferroviaire optimal.

#block(
  fill: rgb("#f0f4ff"),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  *Exemple :*
  - Entree : _"Je veux aller de Paris a Lyon"_
  - Sortie : `DEPART: Paris` | `ARRIVEE: Lyon` | Itineraire optimal via le reseau SNCF
]

== Objectifs

+ *Extraction NER* : Identifier les entites DEPART et ARRIVEE dans des phrases francaises
+ *Fuzzy Matching* : Corriger les fautes d'orthographe et valider les gares SNCF (2785 gares)
+ *Pathfinding* : Calculer l'itineraire optimal entre deux gares selon plusieurs criteres
+ *Benchmark* : Comparer les performances de plusieurs approches NLP et graph en temps reel

== Contraintes du jury

- Dataset avec split train/validation/test
- Comparaison de plusieurs modeles spaCy (sm, md, lg, blank)
- Au moins un modele transformer (CamemBERT)
- Metriques quantitatives d'evaluation
- Fuzzy matching pour la correction orthographique

// =============================================================================
= Description du probleme
// =============================================================================

== Tache NER (Named Entity Recognition)

La reconnaissance d'entites nommees consiste a identifier et classifier des segments de texte en categories predefinies. Notre tache definit deux entites :
- *DEPART* : ville ou gare de depart
- *ARRIVEE* : ville ou gare d'arrivee

== Defis identifies

#table(
  columns: (1fr, 2fr, 2fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header(
    [*Defi*], [*Exemple*], [*Solution*],
  ),
  [Ambiguite prenom/ville], [_"Mon amie Paris vers Angers"_], [Scoring contextuel + penalite],
  [Fautes d'orthographe], [_"Bordeau"_, _"Marsei"_], [Fuzzy matching (gazetteer)],
  [Formulations variees], [_"depuis X vers Y"_, _"de X a Y"_], [Dataset varie + patterns regex],
  [Gares multiples], [Paris Est, Paris Lyon], [Resolution via gazetteer SNCF],
)

== Format BIO

Nous utilisons le schema d'annotation *BIO* (Begin, Inside, Outside) :

#table(
  columns: (1fr, 1fr),
  align: (center, center),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Token*], [*Label*]),
  [Je], [O], [veux], [O], [aller], [O], [de], [O],
  [Paris], [*B-DEPART*], [a], [O], [Lyon], [*B-ARRIVEE*],
)

// =============================================================================
= Architecture du systeme
// =============================================================================

== Design Pattern : Strategy

L'architecture repose sur le *Strategy Pattern* : chaque etape du pipeline (NLP et Graph) est une interface abstraite avec plusieurs implementations interchangeables. Cela permet de comparer les algorithmes de maniere transparente.

#figure(
  ```
  Phrase utilisateur
        |
        v
  +-- NLPEngine (Strategy) --+     +-- GraphEngine (Strategy) --+
  | - rule-based             |     | - Dijkstra                 |
  | - spacy-ner              | --> | - BFS                      |
  | - crf-ner                |     | - A*                       |
  | - transformer-ner        |     +-----------------------------+
  +---------------------------+              |
                                             v
                                    Itineraire optimal
  ```,
  caption: [Pipeline Strategy Pattern avec engines interchangeables],
)

== Engines NLP disponibles

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Engine*], [*Description*], [*Dependance ML*]),
  [rule-based], [Regex + gazetteer + scoring contextuel], [Non],
  [spacy-ner], [Modele spaCy `fr_core_news_md` fine-tune], [spaCy],
  [crf-ner], [Champ aleatoire conditionnel (CRF) avec features manuelles], [sklearn-crfsuite],
  [transformer-ner], [CamemBERT fine-tune (`Jean-Baptiste/camembert-ner`)], [PyTorch + Transformers],
)

== Engines Graph disponibles

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Engine*], [*Description*], [*Complexite*]),
  [Dijkstra], [Plus court chemin avec poids configurables], [O((V+E) log V)],
  [BFS], [Minimise le nombre de correspondances (aretes)], [O(V+E)],
  [A\*], [Dijkstra + heuristique Haversine (distance GPS)], [< Dijkstra en pratique],
)

== Modes d'optimisation

Chaque engine graph peut recevoir une fonction de poids differente :

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Mode*], [*Fonction de poids*], [*Optimise*]),
  [time], [`edge.weight_minutes` (defaut)], [Duree totale du trajet],
  [pollution], [CO2 g/km × distance estimee], [Empreinte carbone],
  [hops], [`1.0` (poids constant)], [Nombre de correspondances],
)

#block(
  fill: rgb("#fff8e0"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  *Note :* BFS ignore la fonction de poids par design --- il minimise toujours le nombre d'aretes. Un warning est emis dans le benchmark quand le mode selectionne n'est pas "hops".
]

== Composants techniques

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Composant*], [*Technologie*], [*Role*]),
  [NLP engines], [Python / Strategy], [Extraction des entites DEPART/ARRIVEE],
  [Graph engines], [Python / Strategy], [Pathfinding multi-criteres],
  [Gazetteer], [JSON (2785 gares)], [Resolution et validation des noms de gare],
  [API], [FastAPI + SSE], [Interface REST + streaming benchmark],
  [Frontend], [React + TypeScript], [Interface utilisateur avec benchmark en temps reel],
)


// =============================================================================
= Dataset
// =============================================================================

== Generation des donnees

Le dataset a ete genere automatiquement a partir de templates varies couvrant differents registres de langue :
- Formel : _"Je souhaite me rendre de \{DEPART\} a \{ARRIVEE\}"_
- Informel : _"Train \{DEPART\} \{ARRIVEE\} pour demain"_
- SMS : _"Aller \{DEPART\} \{ARRIVEE\}"_
- Plus de 30 templates au total

== Repartition

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Split*], [*Exemples*], [*Pourcentage*]),
  [Train], [7 000], [70%],
  [Validation], [1 500], [15%],
  [Test], [1 500], [15%],
  [*Total*], [*10 000*], [*100%*],
)

== Caracteristiques

- *Villes utilisees* : 26 grandes villes francaises
- *Formulations* : 30+ templates differents
- *Distribution* : Equilibree entre les villes

== Impact de la taille du dataset

#figure(
  image("../results/plots/dataset_size_impact.png", width: 85%),
  caption: [Impact de la taille du dataset sur le F1-score],
)

Le gain en F1 est maximal entre 1k et 5k exemples, puis s'aplatit (rendements decroissants). A 10k exemples, les variantes blank et `fr_core_news_sm` atteignent toutes deux *99.88%*, montrant que la quantite de donnees compense l'avantage du modele pre-entraine.


// =============================================================================
= Modeles NER
// =============================================================================

== Modeles spaCy testes

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Modele*], [*Description*], [*Embeddings*]),
  [blank], [Modele vide, entraine from scratch], [Appris],
  [fr_core_news_sm], [Petit modele francais pre-entraine], [Statiques],
  [fr_core_news_md], [Modele moyen avec word vectors], [Word2Vec],
  [fr_core_news_lg], [Grand modele avec word vectors], [Word2Vec],
)

== CamemBERT

*CamemBERT* est un modele transformer base sur RoBERTa, entraine sur 138 Go de texte francais (110M parametres, vocabulaire de 32k tokens, contexte 512 tokens).

Avantages : embeddings contextuels (vs statiques pour spaCy), comprehension profonde du francais, transfer learning efficace.

== Configuration d'entrainement

*spaCy :* 30 epochs, batch 128, dropout 0.35, LR 0.001 \
*CamemBERT :* 5 epochs, batch 16, LR 2e-5, max length 128

== Courbes d'entrainement CamemBERT

#figure(
  image("../results/plots/camembert_loss_curve.png", width: 80%),
  caption: [Courbes de loss train/validation CamemBERT],
)

La loss d'entrainement descend de 0.91 a 0.05 sur 2190 steps. La loss de validation converge plus vite et reste inferieure, signe d'un bon fine-tuning sans sur-apprentissage.

#figure(
  image("../results/plots/camembert_f1_curve.png", width: 80%),
  caption: [F1-score par epoch CamemBERT],
)

Le F1 atteint 100% des l'epoch 2 sur l'ensemble de validation, confirmant la convergence rapide sur notre tache NER.


// =============================================================================
= Resultats quantitatifs
// =============================================================================

== Tableau comparatif

L'*Exact Match (EM)* est la metrique la plus stricte : une phrase est correcte uniquement si les deux entites (depart et arrivee) sont parfaitement identifiees.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt + gray,
  inset: 5pt,
  table.header([*Modele*], [*Precision*], [*Recall*], [*F1*], [*Exact Match*]),
  [Baseline prepositions], [56.56%], [32.20%], [41.04%], [29.53%],
  [spaCy blank], [70.22%], [68.07%], [69.13%], [~55%],
  [spaCy fr_core_news_sm], [74.25%], [77.10%], [75.65%], [~62%],
  [spaCy fr_core_news_md], [84.08%], [88.90%], [86.42%], [~76%],
  [spaCy fr_core_news_lg], [79.02%], [84.13%], [81.50%], [~70%],
  [CamemBERT], [96.40%], [99.97%], [98.15%], [~92.5%],
  [*spaCy 10k*], [*99.90%*], [*99.87%*], [*99.88%*], [*99.73%*],
)

== Analyse

Trois niveaux de performance se degagent :
+ *Baseline* (regles prepositions) : 41% F1 --- plancher de reference, confirme la necessite du NER
+ *spaCy non fine-tune* (blank, sm, md, lg) : 69--86% --- forte variabilite selon le modele de base
+ *Modeles fine-tunes* (CamemBERT et spaCy 10k) : 98--99% --- classe a part

*spaCy 10k* domine avec 7 erreurs sur 1500 tests (3 FP, 4 FN). CamemBERT a un excellent recall (99.97%) mais 112 faux positifs, affectant son Exact Match.

#figure(
  image("../results/plots/model_comparison_bar.png", width: 85%),
  caption: [Comparaison F1-score de tous les modeles],
)

== Matrices de confusion

#figure(
  image("../results/plots/confusion_matrices.png", width: 90%),
  caption: [Matrices de confusion normalisees : Baseline, CamemBERT, spaCy 10k],
)

Points remarquables : la baseline a un taux de FN de 67.8% (phrases sans prepositions explicites), CamemBERT a 3.7% de FP (sur-detection), spaCy 10k est quasi-parfait (0.1% FP, 0.1% FN).


// =============================================================================
= Analyse qualitative
// =============================================================================

== Comparaison des approches

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Critere*], [*spaCy 10k*], [*CamemBERT*]),
  [Performance], [99.88% F1], [98.15% F1],
  [Vitesse d'inference], [~1 ms (CPU)], [~50 ms (GPU)],
  [Memoire], [~50 Mo], [~500 Mo],
  [GPU requis], [Non], [Oui (recommande)],
  [Temps d'entrainement], [~5 min], [~30 min],
)

== Cas d'erreurs analyses

*Erreurs spaCy 10k* (7 / 1500 tests) : 3 FP (prenom detecte comme ville, ex: "Jean") et 4 FN (entite non detectee).

*Erreurs CamemBERT* (113 / 1500 tests) : 112 FP (sur-detection d'entites) et 1 FN.

== Engine rule-based : scoring contextuel

Le moteur rule-based utilise un systeme de scoring par mot pour desambiguiser les cas complexes :

#block(
  fill: rgb("#f5f5f5"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  *Phrase piege :* _"Je pars en vacances avec mon amie Paris vers Angers. On habite sur Albi a 1h de Toulouse"_

  - "Paris" : score 0.0 (penalite `PERSON_PENALTY` via "amie") #sym.arrow.r filtre
  - "Angers" : score 5.5 (boost "vers" + role destination) #sym.arrow.r *arrivee*
  - "Albi" : score 4.0 (boost "habite sur" + role depart) #sym.arrow.r *depart*
  - "Toulouse" : score 3.0 (boost "de") #sym.arrow.r contexte, ignore

  Resultat : *Albi #sym.arrow Angers* #sym.checkmark
]

== Recommandation

*Pour la production, nous recommandons spaCy 10k* : meilleure performance, pas de GPU, inference rapide, modele leger. CamemBERT reste pertinent pour la recherche et l'extension multi-langues.


// =============================================================================
= Pathfinding
// =============================================================================

== Donnees SNCF

#table(
  columns: (auto, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Donnee*], [*Valeur*]),
  [Gares], [2 785],
  [Connexions], [8 510],
  [Types de train], [11 (TGV, TER, Intercites, etc.)],
  [Composante connexe], [1 (graphe connecte)],
)

== Trois moteurs, trois criteres

Le benchmark lance automatiquement les *3 engines* $times$ *3 modes d'optimisation* = *9 combinaisons* :

#table(
  columns: (auto, 1fr, auto, auto),
  stroke: 0.5pt + gray,
  inset: 5pt,
  table.header([*Engine*], [*Algorithme*], [*Supporte weight_fn*], [*Optimal pour*]),
  [Dijkstra], [Plus court chemin (file de priorite)], [Oui], [time, pollution],
  [A\*], [Dijkstra + heuristique Haversine GPS], [Oui], [time, pollution],
  [BFS], [Parcours en largeur (min aretes)], [Non (ignore)], [hops],
)

En mode *hops*, Dijkstra et A\* utilisent `hops_weight(edge) = 1.0` ce qui les rend equivalents a BFS : tous minimisent le nombre d'aretes.

== Estimation CO2 et prix

Le mode *pollution* utilise des coefficients CO2 par type de train (source ADEME) :

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Type de train*], [*g CO2/km*], [*EUR/km*]),
  [TGV INOUI / OUIGO], [2.4], [0.06--0.12],
  [ICE], [3.0], [0.18],
  [Intercites], [5.0], [0.10],
  [TER], [25.0], [0.14],
  [Car TER], [68.0], [0.08],
)

== Metriques de benchmark

Chaque execution est instrumentee avec :
- *Wall time* et *CPU time* (ms) pour la performance
- *Memoire pic* (Ko) via `tracemalloc`
- *Energie estimee* ($mu$Wh) : `CPU_time × 15W TDP / 3600`

Ces metriques permettent de comparer l'efficacite energetique de chaque engine, pas seulement la qualite du resultat.


// =============================================================================
= Conclusion
// =============================================================================

== Objectifs atteints

- Dataset avec split train/val/test (10 000 phrases, 26 villes, 30+ templates)
- Baseline prepositions (reference plancher, 41.04% F1)
- Comparaison de modeles spaCy (blank, sm, md, lg, 10k)
- Modele CamemBERT fine-tune avec courbes de convergence
- *4 engines NLP* en Strategy Pattern (rule-based, spaCy, CRF, Transformer)
- *3 engines graph* (Dijkstra, BFS, A\*) avec 3 modes d'optimisation
- Benchmark multi-criteres avec metriques energie/memoire/CPU
- Interface React avec streaming SSE en temps reel
- Fuzzy matching avec 2 785 gares SNCF

== Performance finale

#table(
  columns: (auto, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  table.header([*Metrique*], [*Valeur*]),
  [Meilleur modele], [spaCy 10k],
  [F1-Score], [99.88%],
  [Precision], [99.90%],
  [Recall], [99.87%],
  [Exact Match], [99.73%],
)

== Lecons apprises

+ *Le fine-tuning est crucial* : un modele simple (spaCy) bien entraine sur la tache bat un transformer generaliste
+ *La qualite des donnees prime* : 10k exemples bien generes suffisent pour atteindre 99.88% F1
+ *L'architecture Strategy Pattern* permet de comparer les approches de maniere rigoureuse et extensible
+ *Le scoring contextuel* (rule-based) resout des ambiguites que meme les modeles ML manquent (prenom vs ville)

== Perspectives

- Integrer les horaires et prix SNCF en temps reel
- Etendre a d'autres langues avec mBERT (zero-shot)
- Ajouter la gestion des dates et horaires dans les phrases (NER multi-entites)
- Deployer en production avec monitoring et re-entrainement automatique
