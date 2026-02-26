#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST pour Travel Order Resolver - Version spaCy
===================================================
Utilise spaCy 10k (F1: 99.88%) au lieu de Qwen2.5 + ChromaDB

Lancer avec: uvicorn api_spacy:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import io
import traceback
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Imports locaux
sys.path.insert(0, './src/inference')
sys.path.insert(0, './src/3_pathfinder')

# Variables globales
nlp = None
fuzzy_matcher = None
pathfinder = None
model_loaded = False


def load_models():
    """Charge tous les modeles"""
    global nlp, fuzzy_matcher, pathfinder, model_loaded

    import spacy
    from fuzzy_matcher import FuzzyStationMatcher
    from pathfinder import TrainPathfinder

    print("="*70)
    print("DEMARRAGE DE L'API - VERSION SPACY")
    print("="*70)

    # 1. Charger spaCy NER
    print("\n[1/3] Chargement du modele spaCy NER 10k...")
    nlp = spacy.load("models/ner_10k/model-best")
    print("      spaCy charge (F1: 99.88%)")

    # 2. Charger Fuzzy Matcher
    print("\n[2/3] Chargement du Fuzzy Matcher SNCF...")
    fuzzy_matcher = FuzzyStationMatcher()

    # 3. Charger Pathfinder
    print("\n[3/3] Initialisation du Pathfinder...")
    pathfinder = TrainPathfinder()
    pathfinder.initialize()

    model_loaded = True

    print("\n" + "="*70)
    print("API PRETE - SPACY + FUZZY + PATHFINDER")
    print("="*70)
    print("\nAccedez a la documentation: http://localhost:8000/docs")
    print("="*70 + "\n")


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    print("\nArret de l'API...")


# Configuration FastAPI
app = FastAPI(
    title="Travel Order Resolver API - spaCy Edition",
    description="API NER avec spaCy 10k (F1: 99.88%) + Fuzzy Matching + Pathfinder A*",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modeles Pydantic
class RouteRequest(BaseModel):
    sentence: str
    use_rag: bool = False  # Ignore, garde pour compatibilite
    metric: str = "time"
    algorithm: str = "astar"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sentence": "Je veux aller de Paris a Lyon",
                "metric": "time",
                "algorithm": "astar"
            }
        }
    )


class StationDetail(BaseModel):
    step_number: int
    code_uic: str
    nom_gare: str
    commune: Optional[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    next_time_minutes: Optional[float] = None
    next_distance_km: Optional[float] = None


class RouteResponse(BaseModel):
    sentence_id: int
    sentence: str
    departure: Optional[str]
    arrival: Optional[str]
    departure_matched: Optional[str]
    arrival_matched: Optional[str]
    confidence: float
    is_valid: bool
    route: Optional[List[str]]
    route_count: Optional[int]
    total_distance: Optional[float]
    total_time: Optional[float]
    formatted_output: str
    method: str
    details: Optional[List[StationDetail]] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    model_f1: str
    fuzzy_stations: int
    pathfinder_ready: bool


class StatsResponse(BaseModel):
    total_stations: int
    total_connections: int
    model: str
    model_f1: str


class ErrorResponse(BaseModel):
    """Modele pour les reponses d'erreur structurees"""
    error: bool = True
    error_type: str
    error_message: str
    error_details: Optional[str] = None
    suggestion: Optional[str] = None


# Gestionnaire d'exceptions global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Capture toutes les exceptions non gerees"""
    error_trace = traceback.format_exc()
    print(f"[ERREUR GLOBALE] {type(exc).__name__}: {str(exc)}")
    print(error_trace)

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_type": "internal_error",
            "error_message": "Une erreur interne s'est produite",
            "error_details": str(exc),
            "suggestion": "Verifiez les logs du serveur ou reessayez"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Gestion des HTTPException avec format structure"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_type": "http_error",
            "error_message": exc.detail,
            "error_details": None,
            "suggestion": get_suggestion_for_status(exc.status_code)
        }
    )


def get_suggestion_for_status(status_code: int) -> str:
    """Retourne une suggestion selon le code HTTP"""
    suggestions = {
        400: "Verifiez le format de votre requete",
        404: "La ressource demandee n'existe pas",
        422: "Verifiez les parametres de votre requete",
        500: "Une erreur serveur s'est produite, reessayez",
        503: "Le service n'est pas disponible, attendez le chargement du modele"
    }
    return suggestions.get(status_code, "Contactez l'administrateur")


# Fonctions d'extraction
def extract_with_spacy(text: str):
    """Extrait DEPART/ARRIVEE avec spaCy"""
    doc = nlp(text)

    depart = None
    arrivee = None

    for ent in doc.ents:
        if ent.label_ == "DEPART":
            depart = ent.text
        elif ent.label_ == "ARRIVEE":
            arrivee = ent.text

    return depart, arrivee


def validate_with_fuzzy(name: str):
    """Valide et corrige avec fuzzy matching"""
    if not name:
        return None, 0

    matched, score = fuzzy_matcher.match(name)
    return matched, score


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Travel Order Resolver API - spaCy Edition</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
            .status {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .endpoint {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #27ae60; border-radius: 3px; }}
            code {{ background: #2c3e50; color: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
            .badge {{ background: #27ae60; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Travel Order Resolver API - spaCy Edition</h1>

            <div class="status">
                <strong>Status:</strong> {'Pipeline spaCy charge et pret' if model_loaded else 'Non charge'}
            </div>

            <h2>Caracteristiques</h2>
            <ul>
                <li><strong>NER:</strong> spaCy 10k (F1: 99.88%)</li>
                <li><strong>Fuzzy:</strong> 2785 gares SNCF</li>
                <li><strong>Pathfinder:</strong> Dijkstra + A*</li>
                <li><strong>CPU:</strong> Pas de GPU necessaire</li>
            </ul>

            <h3>Endpoints</h3>

            <div class="endpoint">
                <strong>POST /route</strong> <span class="badge">PRINCIPAL</span><br>
                Extraction NER + Pathfinding complet<br>
                <code>{{"sentence": "Je veux aller de Paris a Lyon", "metric": "time"}}</code>
            </div>

            <div class="endpoint">
                <strong>POST /extract</strong><br>
                Extraction NER seule (sans pathfinding)
            </div>

            <div class="endpoint">
                <strong>GET /health</strong> | <strong>GET /stats</strong><br>
                Status et statistiques
            </div>

            <h3>Exemple</h3>
            <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px;">
curl -X POST "http://localhost:8000/route" \\
  -H "Content-Type: application/json" \\
  -d '{{"sentence": "Je veux aller de Paris a Lyon"}}'
            </pre>

            <p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model="spaCy NER 10k",
        model_f1="99.88%",
        fuzzy_stations=len(fuzzy_matcher.stations) if fuzzy_matcher else 0,
        pathfinder_ready=pathfinder is not None and pathfinder.initialized
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Statistiques"""
    return StatsResponse(
        total_stations=len(fuzzy_matcher.stations) if fuzzy_matcher else 0,
        total_connections=len(pathfinder.graph.graph.edges) if pathfinder else 0,
        model="spaCy NER 10k",
        model_f1="99.88%"
    )


@app.post("/extract")
async def extract_only(request: RouteRequest):
    """Extraction NER seule (sans pathfinding)"""
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "error_type": "service_unavailable",
                "error_message": "Le modele NER n'est pas encore charge",
                "suggestion": "Attendez quelques secondes et reessayez"
            }
        )

    # Validation de la phrase
    if not request.sentence or len(request.sentence.strip()) < 3:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_type": "invalid_input",
                "error_message": "La phrase est trop courte ou vide",
                "suggestion": "Entrez une phrase complete, ex: 'Je veux aller de Paris a Lyon'"
            }
        )

    try:
        # Extraction NER
        depart, arrivee = extract_with_spacy(request.sentence)

        # Fuzzy matching
        depart_matched, dep_score = validate_with_fuzzy(depart)
        arrivee_matched, arr_score = validate_with_fuzzy(arrivee)

        return {
            "error": False,
            "sentence": request.sentence,
            "departure": depart,
            "arrival": arrivee,
            "departure_matched": depart_matched,
            "arrival_matched": arrivee_matched,
            "confidence": min(dep_score, arr_score) / 100 if depart and arrivee else 0.0,
            "method": "spaCy NER 10k + Fuzzy Matching"
        }

    except Exception as e:
        print(f"[ERREUR /extract] {type(e).__name__}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_type": "extraction_error",
                "error_message": "Erreur lors de l'extraction NER",
                "error_details": str(e),
                "suggestion": "Verifiez la syntaxe de votre phrase"
            }
        )


@app.post("/route")
async def get_route(request: RouteRequest):
    """Pipeline complet: NER + Fuzzy + Pathfinding"""
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "error_type": "service_unavailable",
                "error_message": "Le pipeline n'est pas encore charge",
                "suggestion": "Attendez quelques secondes et reessayez"
            }
        )

    # Validation de la phrase
    if not request.sentence or len(request.sentence.strip()) < 3:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_type": "invalid_input",
                "error_message": "La phrase est trop courte ou vide",
                "suggestion": "Entrez une phrase complete, ex: 'Je veux aller de Paris a Lyon'"
            }
        )

    # Validation de la metrique
    if request.metric not in ["time", "distance"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_type": "invalid_metric",
                "error_message": f"Metrique invalide: {request.metric}",
                "suggestion": "Utilisez 'time' ou 'distance'"
            }
        )

    # Validation de l'algorithme
    if request.algorithm not in ["astar", "dijkstra"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_type": "invalid_algorithm",
                "error_message": f"Algorithme invalide: {request.algorithm}",
                "suggestion": "Utilisez 'astar' ou 'dijkstra'"
            }
        )

    try:
        # 1. Extraction NER
        try:
            depart, arrivee = extract_with_spacy(request.sentence)
        except Exception as e:
            print(f"[ERREUR NER] {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_type": "ner_error",
                    "error_message": "Erreur lors de l'extraction NER",
                    "error_details": str(e),
                    "suggestion": "Verifiez la syntaxe de votre phrase"
                }
            )

        # Verifier si on a trouve des villes
        if not depart and not arrivee:
            # Retourner un resultat invalide mais pas une erreur
            return {
                "error": False,
                "sentence_id": 1,
                "sentence": request.sentence,
                "departure": None,
                "arrival": None,
                "departure_matched": None,
                "arrival_matched": None,
                "confidence": 0.0,
                "is_valid": False,
                "route": None,
                "route_count": None,
                "total_distance": None,
                "total_time": None,
                "formatted_output": "1,INVALID",
                "method": f"spaCy NER + Fuzzy + {request.algorithm.upper()}",
                "details": None,
                "warning": "Aucune ville de depart ou d'arrivee detectee dans la phrase"
            }

        # 2. Fuzzy matching
        try:
            depart_matched, dep_score = validate_with_fuzzy(depart)
            arrivee_matched, arr_score = validate_with_fuzzy(arrivee)
        except Exception as e:
            print(f"[ERREUR FUZZY] {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_type": "fuzzy_error",
                    "error_message": "Erreur lors du fuzzy matching",
                    "error_details": str(e),
                    "suggestion": "Les noms de gares extraits sont peut-etre invalides"
                }
            )

        # 3. Pathfinding
        route_result = None
        details = None
        is_valid = False
        warning = None

        if depart_matched and arrivee_matched:
            try:
                route_result = pathfinder.find_route(
                    depart_matched,
                    arrivee_matched,
                    metric=request.metric,
                    algorithm=request.algorithm
                )

                if route_result and "error" not in route_result:
                    is_valid = True
                    details = route_result.get("details", [])
                elif route_result and "error" in route_result:
                    warning = route_result.get("error", "Aucun itineraire trouve")

            except Exception as e:
                print(f"[ERREUR PATHFINDING] {str(e)}")
                warning = f"Erreur pathfinding: {str(e)}"
        else:
            if not depart_matched:
                warning = f"Gare de depart '{depart}' non trouvee dans la base SNCF"
            elif not arrivee_matched:
                warning = f"Gare d'arrivee '{arrivee}' non trouvee dans la base SNCF"

        # 4. Formater la sortie
        if is_valid:
            formatted = f"1,{','.join([s.lower() for s in route_result['stations']])}"
        else:
            formatted = "1,INVALID"

        response = {
            "error": False,
            "sentence_id": 1,
            "sentence": request.sentence,
            "departure": depart,
            "arrival": arrivee,
            "departure_matched": depart_matched,
            "arrival_matched": arrivee_matched,
            "confidence": min(dep_score, arr_score) / 100 if depart and arrivee else 0.0,
            "is_valid": is_valid,
            "route": route_result.get("stations") if route_result and is_valid else None,
            "route_count": route_result.get("num_stops") if route_result and is_valid else None,
            "total_distance": route_result.get("total_distance_km") if route_result and is_valid else None,
            "total_time": route_result.get("total_time_minutes") if route_result and is_valid else None,
            "formatted_output": formatted,
            "method": f"spaCy NER + Fuzzy + {request.algorithm.upper()}",
            "details": details
        }

        if warning:
            response["warning"] = warning

        return response

    except Exception as e:
        print(f"[ERREUR GLOBALE /route] {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_type": "pipeline_error",
                "error_message": "Erreur lors du traitement de la requete",
                "error_details": str(e),
                "suggestion": "Verifiez votre phrase et reessayez"
            }
        )


@app.post("/speech-to-route")
async def speech_to_route(audio_file: UploadFile = File(...), metric: str = "time"):
    """Speech-to-Text + NER + Pathfinding"""
    try:
        import whisper
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Whisper non installe. pip install openai-whisper"
        )

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        import tempfile
        import os

        # Sauvegarder audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp:
            content = await audio_file.read()
            temp.write(content)
            temp_path = temp.name

        # Transcription
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(temp_path, language="fr")
        text = result["text"]

        os.unlink(temp_path)

        # NER + Route
        depart, arrivee = extract_with_spacy(text)
        depart_matched, _ = validate_with_fuzzy(depart)
        arrivee_matched, _ = validate_with_fuzzy(arrivee)

        route_result = None
        if depart_matched and arrivee_matched:
            route_result = pathfinder.find_route(depart_matched, arrivee_matched, metric=metric)

        return {
            "transcribed_text": text,
            "departure": depart_matched,
            "arrival": arrivee_matched,
            "route": route_result.get("stations") if route_result and "error" not in route_result else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
