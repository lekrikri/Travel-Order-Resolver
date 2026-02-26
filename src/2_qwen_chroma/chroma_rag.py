#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB RAG pour Travel Orders
Stocke et recherche des exemples annotés pour améliorer l'extraction
"""

import csv
from typing import List, Dict, Optional


class ChromaRAG:
    """Système RAG avec ChromaDB pour few-shot learning dynamique"""

    def __init__(self, collection_name: str = "travel_orders"):
        """
        Initialiser ChromaDB

        Args:
            collection_name: Nom de la collection ChromaDB
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None

    def initialize(self):
        """Initialiser ChromaDB et créer la collection"""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            print("Initialisation de ChromaDB...")

            # Client ChromaDB (persiste dans ./chroma_db)
            self.client = chromadb.PersistentClient(path="./chroma_db")

            # Fonction d'embedding (sentence-transformers français)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

            # Créer ou récupérer la collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"Collection '{self.collection_name}' chargée ({self.collection.count()} documents)")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Travel order sentences with departure/arrival annotations"}
                )
                print(f"Collection '{self.collection_name}' créée")

            return True

        except ImportError as e:
            print(f"ERREUR: Dépendances manquantes: {e}")
            print("Installation: pip install chromadb sentence-transformers")
            return False
        except Exception as e:
            print(f"ERREUR lors de l'initialisation: {e}")
            return False

    def load_dataset(self, csv_path: str, limit: Optional[int] = None):
        """
        Charger un dataset CSV dans ChromaDB

        Args:
            csv_path: Chemin vers le fichier CSV
            limit: Limite de phrases à charger (None = toutes)
        """
        if self.collection is None:
            print("Collection non initialisée!")
            return False

        print(f"Chargement de {csv_path}...")

        documents = []
        metadatas = []
        ids = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break

                # Nettoyer la phrase (enlever les tags)
                import re
                sentence = row['sentence']
                clean_sentence = re.sub(r'<[^>]+>', '', sentence)

                documents.append(clean_sentence)
                metadatas.append({
                    'departure': row['departure'],
                    'arrival': row['arrival'],
                    'annotated_sentence': sentence
                })
                ids.append(f"train_{i}")

        # Ajouter à ChromaDB (par batches pour éviter les timeouts)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

            if (i + batch_size) % 1000 == 0:
                print(f"   {i+batch_size} documents chargés...")

        print(f"OK {len(documents)} documents chargés dans ChromaDB")
        return True

    def get_similar_examples(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Récupérer des exemples similaires pour few-shot learning

        Args:
            query: Phrase à analyser
            n_results: Nombre d'exemples à retourner

        Returns:
            Liste d'exemples avec input/output
        """
        if self.collection is None:
            print("Collection non initialisée!")
            return []

        try:
            # Recherche vectorielle
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            # Formater les résultats pour Qwen2.5
            examples = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    examples.append({
                        'input': results['documents'][0][i],
                        'output': {
                            'departure': metadata['departure'],
                            'arrival': metadata['arrival']
                        },
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })

            return examples

        except Exception as e:
            print(f"ERREUR lors de la recherche: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Obtenir les statistiques de la collection"""
        if self.collection is None:
            return {"error": "Collection non initialisée"}

        try:
            count = self.collection.count()

            # Récupérer quelques exemples
            sample = self.collection.peek(limit=5)

            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "sample": sample
            }
        except Exception as e:
            return {"error": str(e)}


# Fonction utilitaire pour test rapide
def test_chroma_rag():
    """Tester ChromaDB RAG"""
    print("="*70)
    print("TEST CHROMADB RAG")
    print("="*70)

    # Initialiser
    rag = ChromaRAG()
    if not rag.initialize():
        print("Impossible d'initialiser ChromaDB")
        return

    # Charger le dataset de 100 phrases (pour test rapide)
    csv_path = "./data/annotated/train_sentences_100.csv"
    rag.load_dataset(csv_path)

    # Tester la recherche
    print("\n" + "="*70)
    print("TEST DE RECHERCHE")
    print("="*70)

    test_queries = [
        "Je veux aller de Paris à Lyon",
        "Un train pour Marseille depuis Nice",
        "Direction Bordeaux au départ de Toulouse"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        examples = rag.get_similar_examples(query, n_results=3)

        print("Exemples similaires:")
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex['input']}")
            print(f"     {ex['output']['departure']} -> {ex['output']['arrival']}")
            if ex['distance']:
                print(f"     Distance: {ex['distance']:.3f}")

    # Stats
    print("\n" + "="*70)
    stats = rag.get_collection_stats()
    print(f"Total documents: {stats.get('total_documents')}")


if __name__ == "__main__":
    test_chroma_rag()
