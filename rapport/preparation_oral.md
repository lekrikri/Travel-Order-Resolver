# Preparation Oral - Notions Cles du Rapport

> Document complementaire au rapport d'experimentation.
> Objectif : comprendre et savoir expliquer chaque concept technique mentionne.

---

## 1. Introduction (Sections 1.1 - 1.3)

### NER - Named Entity Recognition (Reconnaissance d'Entites Nommees)

**Qu'est-ce que c'est ?**
Le NER est une sous-tache du NLP (Natural Language Processing) qui consiste a reperer dans un texte des "entites nommees" et a les classifier. Les categories classiques sont : personne, lieu, organisation, date, etc.

Dans notre projet, on definit **deux categories custom** :
- `DEPART` : la ville/gare d'ou part le voyageur
- `ARRIVEE` : la ville/gare ou il veut aller

**Pourquoi c'est difficile ?**
Le NER ne fait pas juste du "pattern matching". Il doit comprendre le **contexte** :
- "Paris" peut etre un prenom ou une ville
- "Lyon" peut etre "Lyon Part-Dieu" ou "Lyon Perrache"
- Les formulations varient enormement ("je veux aller de X a Y", "train X Y demain", "depuis X vers Y")

### Format BIO (Begin, Inside, Outside)

**Qu'est-ce que c'est ?**
C'est un schema d'annotation standard pour le NER. Chaque token (mot) du texte recoit un label :
- **B-XXX** (Begin) : premier mot d'une entite de type XXX
- **I-XXX** (Inside) : mot suivant a l'interieur de la meme entite
- **O** (Outside) : mot qui ne fait partie d'aucune entite

**Exemple :**
```
"Je veux aller de Paris Gare de Lyon a Marseille"
 O  O    O     O  B-DEP I-DEP I-DEP I-DEP O B-ARR
```

**Pourquoi B et I ?** Pour gerer les entites multi-mots. Sans le B/I, on ne saurait pas ou commence et finit une entite quand deux entites sont cote a cote.

### Fuzzy Matching

**Qu'est-ce que c'est ?**
Technique de correspondance approximative de chaines de caracteres. Quand l'utilisateur ecrit "Bordeau" au lieu de "Bordeaux", le fuzzy matching calcule une **distance d'edition** (Levenshtein) et trouve la gare la plus proche dans le gazetteer (dictionnaire des 2785 gares SNCF).

**Distance de Levenshtein** : nombre minimum d'operations (insertion, suppression, substitution) pour transformer une chaine en une autre.
- "Bordeau" -> "Bordeaux" = 1 insertion ('x') = distance 1

### Gazetteer

**Qu'est-ce que c'est ?**
Un dictionnaire/liste de reference contenant tous les noms valides. Ici, c'est la liste des 2785 gares SNCF. Il sert a :
1. Valider qu'une ville detectee par le NER est bien une gare existante
2. Corriger les fautes via fuzzy matching
3. Desambiguiser (ex: "Paris" -> quelle gare exactement ?)

---

## 2. Description du Probleme (Section 2)

### Ambiguite Prenom/Ville

C'est le defi le plus interessant du projet. Certains prenoms francais sont aussi des noms de ville :
- **Paris** (prenom feminin rare, mais existe)
- **Nancy**, **Valence**, **Nice** (prenoms courants)

Le systeme doit utiliser le **contexte** pour distinguer : "mon amie Paris" (prenom) vs "je pars de Paris" (ville).

### Tokenization

**Qu'est-ce que c'est ?**
Le decoupage d'un texte en unites elementaires (tokens). C'est la premiere etape de tout pipeline NLP.
- Tokenization simple : decoupage par espaces
- Tokenization spaCy : gere la ponctuation, les apostrophes, etc.
- Tokenization CamemBERT (SentencePiece) : sous-mots, peut couper "Marseille" en "Mars" + "eille"

---

## 3. Architecture - Strategy Pattern (Section 3)

### Strategy Pattern (Design Pattern)

**Qu'est-ce que c'est ?**
Un patron de conception (GoF) qui permet de definir une **famille d'algorithmes**, de les encapsuler chacun dans une classe, et de les rendre **interchangeables**. Le client utilise une interface commune sans connaitre l'implementation.

**Dans notre projet :**
```
NLPEngine (interface abstraite)
  |-- RuleBasedEngine      (regex + scoring)
  |-- SpacyNEREngine       (modele spaCy)
  |-- CRFEngine            (CRF sklearn)
  |-- TransformerEngine    (CamemBERT)

GraphEngine (interface abstraite)
  |-- DijkstraEngine
  |-- BFSEngine
  |-- AStarEngine
```

**Pourquoi c'est bien ?**
- On peut **comparer** les algorithmes sans changer le reste du code
- On peut **ajouter** un nouveau modele sans toucher aux existants (Open/Closed Principle)
- Le benchmark peut iterer sur toutes les strategies automatiquement

**Si le jury demande :** "Pourquoi Strategy et pas juste des if/else ?"
-> Reponse : extensibilite. Ajouter un 5e modele NLP = creer une classe, pas modifier du code existant. C'est le principe SOLID Open/Closed.

---

## 4. Modeles spaCy (Section 3.2 / 5.1) -- QUESTION CLE

### Qu'est-ce que spaCy ?

**spaCy** est une bibliotheque Python open-source pour le NLP industriel (pas pour la recherche comme NLTK). Elle fournit des pipelines pre-entraines pour plusieurs langues.

Un pipeline spaCy contient typiquement : tokenizer -> tagger (POS) -> parser (dependances) -> NER.

### Les 4 modeles spaCy et leurs differences

| Modele | Taille | Embeddings | Details |
|--------|--------|-----------|---------|
| **blank** | ~0 Mo | Aucun (appris from scratch) | Modele vide, juste la structure du pipeline. Pas de connaissance prealable. |
| **fr_core_news_sm** | ~15 Mo | Statiques (hash embeddings) | Pre-entraine sur des news francaises. Embeddings compresses via hashing trick. |
| **fr_core_news_md** | ~45 Mo | Word2Vec (50k vecteurs) | Inclut de vrais word vectors. Meilleure generalisation sur des mots inconnus. |
| **fr_core_news_lg** | ~560 Mo | Word2Vec (500k vecteurs) | Plus gros vocabulaire de vecteurs. Pas toujours meilleur que md (overfitting possible). |

### C'est quoi des "embeddings" ?

Des **representations vectorielles** des mots. Chaque mot est transforme en un vecteur de nombres (ex: 300 dimensions). Les mots semantiquement proches ont des vecteurs proches.

**Types d'embeddings mentionnes dans le rapport :**

1. **Pas d'embeddings (blank)** : le modele part de zero, apprend ses propres representations pendant l'entrainement sur notre dataset.

2. **Hash embeddings (sm)** : technique de compression. Au lieu de stocker un vecteur par mot, on utilise des fonctions de hashage pour partager les poids. Economise de la memoire mais perd en precision.

3. **Word2Vec statiques (md, lg)** : chaque mot a un vecteur fixe, appris sur un gros corpus (news). "Statique" = le vecteur de "Paris" est le meme que "Paris" soit un prenom ou une ville. C'est une limitation.

4. **Embeddings contextuels (CamemBERT)** : le vecteur d'un mot **depend de sa phrase**. "Paris" dans "mon amie Paris" et "je pars de Paris" auront des vecteurs differents. C'est la revolution des transformers.

### Pourquoi lg < md dans nos resultats ?

C'est contre-intuitif mais explicable :
- **lg** a 500k vecteurs (10x plus que md) mais ca inclut beaucoup de mots rares/bruit
- Sur une tache **specifique** avec 26 villes, le surplus de vocabulaire peut introduire du bruit
- Le fine-tuning sur notre petit dataset peut etre perturbe par des representations trop riches
- C'est un cas classique : **plus gros != meilleur** (biais-variance tradeoff)

### Pourquoi blank + 10k donnees bat tout ?

Parce qu'avec suffisamment de donnees d'entrainement **specifiques a la tache**, un modele vide peut apprendre exactement ce dont il a besoin, sans les connaissances "parasites" des modeles pre-entraines. C'est le point cle du rapport : **la qualite et la quantite des donnees prime sur la taille du modele**.

---

## 5. CamemBERT (Section 5.2)

### Qu'est-ce que CamemBERT ?

CamemBERT est le "BERT francais". C'est un modele de type **transformer** :

**Filiation :**
- **BERT** (Google, 2018) : modele bidirectionnel base sur l'architecture Transformer
- **RoBERTa** (Facebook, 2019) : version optimisee de BERT (plus de donnees, meilleur entrainement)
- **CamemBERT** (INRIA/Facebook, 2019) : RoBERTa entraine **specifiquement sur du francais** (138 Go de texte)

### Transformer vs spaCy : la difference fondamentale

| Aspect | spaCy (CNN-based) | CamemBERT (Transformer) |
|--------|-------------------|------------------------|
| Architecture | CNN + transitions | Self-Attention multi-head |
| Contexte | Fenetre locale | Phrase entiere (512 tokens) |
| Embeddings | Statiques (meme vecteur partout) | Contextuels (depend de la phrase) |
| Parametres | ~1-5M | 110M |
| Inference | ~1 ms CPU | ~50 ms GPU |

### Fine-tuning vs Entrainement from scratch

**Fine-tuning** : on prend un modele **deja entraine** sur une tache generale (comprendre le francais) et on l'adapte a notre tache specifique (NER depart/arrivee). On ne re-entraine que les dernieres couches.

**Avantage** : le modele sait deja "lire" le francais, il n'a plus qu'a apprendre "quoi chercher".

**Dans le rapport :**
- CamemBERT : fine-tune (5 epochs suffisent, le modele converge vite)
- spaCy blank : entrainement from scratch (30 epochs necessaires)

### Transfer Learning

C'est le principe general derriere le fine-tuning : transferer les connaissances d'une tache (comprendre le francais) vers une autre (detecter des villes). C'est ce qui rend CamemBERT performant meme avec peu de donnees.

---

## 6. CRF - Conditional Random Fields (Section 3.2)

### Qu'est-ce qu'un CRF ?

Un **Champ Aleatoire Conditionnel** est un modele probabiliste **discriminatif** pour l'etiquetage de sequences. Contrairement a un classifieur qui traite chaque mot independamment, le CRF considere les **dependances entre labels voisins**.

**Pourquoi c'est utile pour le NER ?**
- Si le mot precedent est B-DEPART, le mot suivant a plus de chances d'etre I-DEPART que B-ARRIVEE
- Le CRF modeleise ces transitions entre labels

**Features manuelles** : contrairement aux reseaux de neurones, le CRF necessite des features definies a la main (le mot, sa casse, le mot precedent/suivant, est-ce un chiffre, etc.)

---

## 7. Pathfinding - Algorithmes de Graphe (Section 7)

### Dijkstra

**Algorithme de plus court chemin** dans un graphe a poids positifs.
- Utilise une **file de priorite** (min-heap)
- Explore les noeuds par ordre de distance croissante
- **Garantit** le chemin optimal
- Complexite : O((V+E) log V)

**Dans le projet :** V = 2785 gares, E = 8510 connexions. Dijkstra trouve le trajet le plus rapide (poids = minutes) ou le moins polluant (poids = CO2).

### BFS (Breadth-First Search)

**Parcours en largeur** : explore tous les voisins d'un noeud avant de passer aux voisins des voisins.
- **Ignore les poids** des aretes
- Trouve le chemin avec le **minimum d'aretes** (correspondances)
- Complexite : O(V+E)

**Dans le projet :** minimise le nombre de correspondances (hops). Un Paris->Lyon direct sera prefere a Paris->Dijon->Lyon meme si ce dernier est plus rapide.

### A* (A-Star)

**Dijkstra + heuristique** : au lieu d'explorer "en cercle" autour du depart, A* utilise une estimation de la distance restante pour **orienter la recherche** vers la destination.

**Heuristique Haversine** : calcul de la distance a vol d'oiseau entre deux points GPS sur une sphere (la Terre). C'est une **sous-estimation** de la vraie distance (admissible), ce qui garantit l'optimalite.

**Avantage sur Dijkstra** : explore moins de noeuds, donc plus rapide en pratique, mais meme resultat optimal.

**Si le jury demande** : "Pourquoi A* n'est pas toujours meilleur ?"
-> L'heuristique a un cout de calcul (formule trigonometrique). Sur un petit graphe (2785 noeuds), l'overhead peut annuler le gain. A* brille sur de tres grands graphes.

### Modes d'optimisation

Les 3 modes changent la **fonction de poids** des aretes :
- **time** : poids = duree en minutes
- **pollution** : poids = CO2 en grammes (g/km x distance)
- **hops** : poids = 1 pour chaque arete (equivalent BFS)

---

## 8. Metriques d'Evaluation (Section 6)

### Precision, Recall, F1-Score

Imaginons 100 phrases de test :

| | Predit DEPART | Predit rien |
|---|---|---|
| **Vrai DEPART** | 90 (TP) | 5 (FN) |
| **Pas DEPART** | 3 (FP) | 2 (TN) |

- **Precision** = TP / (TP + FP) = 90/93 = 96.8% -> "Quand le modele dit DEPART, a-t-il raison ?"
- **Recall** = TP / (TP + FN) = 90/95 = 94.7% -> "Le modele trouve-t-il tous les vrais DEPART ?"
- **F1-Score** = 2 x (P x R) / (P + R) = moyenne harmonique -> equilibre les deux

### Exact Match (EM)

Metrique **plus stricte** : une phrase est comptee correcte uniquement si **les deux entites** (depart ET arrivee) sont parfaitement trouvees.

C'est pourquoi le F1 de CamemBERT est 98.15% mais son EM n'est que ~92.5% : il trouve presque toujours au moins une entite, mais sur-detecte parfois (faux positifs).

### Matrice de Confusion

Tableau 2D qui montre pour chaque classe reelle, ce que le modele a predit. Permet de voir les **types d'erreurs** :
- FP (False Positive) = sur-detection (le modele hallucine une entite)
- FN (False Negative) = sous-detection (le modele rate une entite)

---

## 9. Metriques de Benchmark (Section 7.5)

### tracemalloc

Module Python standard pour mesurer la **consommation memoire** d'un bloc de code. Donne le pic memoire en Ko.

### Wall time vs CPU time

- **Wall time** : temps reel ecoule (horloge murale). Inclut les attentes I/O, le scheduling OS.
- **CPU time** : temps effectivement passe a calculer sur le processeur. Toujours <= wall time.

### Estimation d'energie

Formule du rapport : `CPU_time x 15W TDP / 3600 = microWh`
- **TDP** (Thermal Design Power) : puissance maximale du CPU (~15W pour un laptop)
- C'est une estimation grossiere mais permet de **comparer** les engines entre eux

---

## 10. Technologies (Section 3.5)

### FastAPI + SSE

- **FastAPI** : framework Python moderne pour creer des API REST. Performant (base sur Starlette/ASGI), avec generation auto de documentation OpenAPI.
- **SSE** (Server-Sent Events) : protocole HTTP pour le streaming unidirectionnel serveur->client. Le benchmark envoie les resultats au fur et a mesure, sans attendre que toutes les 9 combinaisons soient finies.

### React + TypeScript

- **React** : bibliotheque JavaScript pour construire des interfaces utilisateur (composants, state, virtual DOM)
- **TypeScript** : sur-couche typee de JavaScript. Ajoute le typage statique pour attraper les erreurs a la compilation.

---

## Questions pieges possibles du jury

**Q: Pourquoi ne pas utiliser ChatGPT/GPT-4 directement ?**
-> Latence (secondes vs millisecondes), cout (API payante vs inference locale gratuite), controle (donnees sensibles restent locales), reproductibilite (modele fige vs modele qui evolue).

**Q: Pourquoi le modele lg est moins bon que md ?**
-> Voir section 4 ci-dessus. Biais-variance tradeoff. Plus de parametres != meilleure performance sur une tache specifique.

**Q: Comment gerer une ville inconnue du gazetteer ?**
-> Le fuzzy matching trouve la gare la plus proche. Si la distance est trop grande (seuil configurable), on rejette l'entite.

**Q: Pourquoi 10 000 exemples suffisent ?**
-> La tache est relativement simple (2 entites, 26 villes, patterns reguliers). Le graphe de la figure "dataset size impact" montre les rendements decroissants au-dela de 5k.

**Q: A* est-il toujours meilleur que Dijkstra ?**
-> En theorie oui (moins de noeuds explores). En pratique sur un petit graphe (2785 noeuds), le cout de l'heuristique peut egaliser ou depasser le gain. Les benchmarks montrent des temps tres proches.

**Q: Quelle est la difference entre votre rule-based et un simple regex ?**
-> Le rule-based utilise des regex pour les patterns prepositions, MAIS ajoute un **systeme de scoring contextuel** avec penalites (prenom, contexte irrelevant) et boosts (prepositions directionnelles). C'est un systeme expert, pas juste du regex.
