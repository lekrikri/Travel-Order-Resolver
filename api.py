#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST pour l'extraction de villes de départ/arrivée
Qwen2.5 + ChromaDB RAG
"""

import sys
import io
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Imports locaux
sys.path.insert(0, './src/2_qwen_chroma')
from qwen_extractor import Qwen25Extractor
from chroma_rag import ChromaRAG

# Configuration
app = FastAPI(
    title="Travel Order Resolver API",
    description="API d'extraction de villes de départ/arrivée avec Qwen2.5 + ChromaDB RAG",
    version="1.0.0"
)

# Variables globales pour le modèle
extractor = None
rag = None
model_loaded = False

# Modèles de données
class ExtractionRequest(BaseModel):
    sentence: str
    use_rag: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "sentence": "Je voudrais un billet de train de Paris à Lyon pour demain",
                "use_rag": True
            }
        }

class ExtractionResponse(BaseModel):
    sentence: str
    departure: Optional[str]
    arrival: Optional[str]
    confidence: float
    method: str

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str]
    model_loaded: bool

# Initialisation au démarrage
@app.on_event("startup")
async def startup_event():
    """Chargement du modèle au démarrage"""
    global extractor, rag, model_loaded

    print("="*70)
    print("DÉMARRAGE DE L'API - CHARGEMENT DU MODÈLE")
    print("="*70)

    # Charger Qwen2.5
    print("\n[1/2] Chargement de Qwen2.5-3B-Instruct...")
    extractor = Qwen25Extractor(model_name="Qwen/Qwen2.5-3B-Instruct")

    if not extractor.load_model():
        print("❌ ERREUR: Impossible de charger Qwen2.5")
        return

    print("✅ Qwen2.5 chargé sur", "GPU" if torch.cuda.is_available() else "CPU")

    # Initialiser ChromaDB RAG
    print("\n[2/2] Initialisation de ChromaDB RAG...")
    rag = ChromaRAG()

    if not rag.initialize():
        print("❌ ERREUR: Impossible d'initialiser ChromaDB")
        return

    # Charger le dataset si la collection est vide
    if rag.collection.count() == 0:
        print("   Chargement des 100 phrases annotées...")
        rag.load_dataset("./data/annotated/train_sentences_100.csv")
        print(f"   ✅ {rag.collection.count()} documents chargés")
    else:
        print(f"   ✅ Collection déjà chargée: {rag.collection.count()} documents")

    model_loaded = True

    print("\n" + "="*70)
    print("✅ API PRÊTE - MODÈLE CHARGÉ")
    print("="*70)
    print(f"\n🚀 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"📊 ChromaDB: {rag.collection.count()} exemples")
    print("\n💡 Accédez à la documentation: http://localhost:8000/docs")
    print("="*70 + "\n")

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec documentation"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Travel Order Resolver API</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            .status {{
                background: {'#d4edda' if model_loaded else '#f8d7da'};
                border: 1px solid {'#c3e6cb' if model_loaded else '#f5c6cb'};
                color: {'#155724' if model_loaded else '#721c24'};
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .endpoint {{
                background: #ecf0f1;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #3498db;
                border-radius: 3px;
            }}
            code {{
                background: #2c3e50;
                color: #ecf0f1;
                padding: 2px 6px;
                border-radius: 3px;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚂 Travel Order Resolver API</h1>

            <div class="status">
                <strong>Status:</strong> {'✅ Modèle chargé et prêt' if model_loaded else '❌ Modèle non chargé'}
            </div>

            <h2>📚 Documentation</h2>
            <p>Cette API permet d'extraire automatiquement les villes de départ et d'arrivée à partir de commandes en langage naturel.</p>

            <h3>🔗 Endpoints disponibles</h3>

            <div class="endpoint">
                <strong>POST /extract</strong><br>
                Extrait les villes de départ et d'arrivée d'une phrase<br>
                <code>{{"sentence": "Je veux aller de Paris à Lyon", "use_rag": true}}</code>
            </div>

            <div class="endpoint">
                <strong>GET /health</strong><br>
                Vérifie l'état de l'API et du modèle
            </div>

            <div class="endpoint">
                <strong>GET /models</strong><br>
                Liste les modèles disponibles
            </div>

            <h3>🔧 Utilisation</h3>
            <p>Exemple avec curl:</p>
            <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:8000/extract" \\
  -H "Content-Type: application/json" \\
  -d '{{"sentence": "Je voudrais un billet Paris Lyon", "use_rag": true}}'
            </pre>

            <h3>📖 Documentation interactive</h3>
            <p>
                <a href="/docs">📄 Swagger UI</a> |
                <a href="/redoc">📘 ReDoc</a>
            </p>

            <h3>🎯 Performances</h3>
            <ul>
                <li><strong>Qwen2.5 seul:</strong> 75% accuracy</li>
                <li><strong>Qwen2.5 + RAG:</strong> 93% accuracy</li>
                <li><strong>GPU:</strong> {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU uniquement'}</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check de l'API"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        model_loaded=model_loaded
    )

@app.get("/models")
async def list_models():
    """Liste les modèles disponibles"""
    return {
        "models": [
            {
                "name": "qwen2.5-only",
                "description": "Qwen2.5-3B-Instruct seul",
                "accuracy": "75%",
                "speed": "~6.5s/phrase"
            },
            {
                "name": "qwen2.5-rag",
                "description": "Qwen2.5-3B-Instruct + ChromaDB RAG",
                "accuracy": "93%",
                "speed": "~6.7s/phrase",
                "recommended": True
            }
        ],
        "chromadb_examples": rag.collection.count() if rag else 0
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract_cities(request: ExtractionRequest):
    """
    Extrait les villes de départ et d'arrivée d'une phrase

    - **sentence**: La phrase à analyser
    - **use_rag**: Utiliser ChromaDB RAG pour améliorer l'extraction (recommandé)
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        if request.use_rag and rag:
            # Extraction avec RAG
            similar_examples = rag.get_similar_examples(request.sentence, n_results=3)
            result = extractor.extract_cities(request.sentence, examples=similar_examples)
            method = "Qwen2.5 + ChromaDB RAG"
        else:
            # Extraction sans RAG
            result = extractor.extract_cities(request.sentence)
            method = "Qwen2.5 seul"

        return ExtractionResponse(
            sentence=request.sentence,
            departure=result.get('departure'),
            arrival=result.get('arrival'),
            confidence=result.get('confidence', 0.0),
            method=method
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction: {str(e)}")

# Pour lancer l'API en développement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
