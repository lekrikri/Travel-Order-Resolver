"""FastAPI server -- connects NLP engines and graph engines."""

from __future__ import annotations

import json
from dataclasses import asdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from src.graph.base import Edge, RouteSegment
from src.nlp.base import NLPResult
from src.nlp.rule_based import RuleBasedEngine
from src.nlp.spacy_ner import SpaCyNEREngine
from src.nlp.crf_ner import CRFNEREngine
from src.nlp.transformer_ner import TransformerNEREngine
from src.graph.base import GraphResult
from src.graph.dijkstra import DijkstraEngine
from src.graph.bfs import BFSEngine
from src.graph.astar import AStarEngine
from src import registry

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

app = FastAPI(title="Travel Order Resolver", version="0.1.0")

# CORS for React frontend on :3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# Register available engines at startup
registry.register_nlp(RuleBasedEngine())
registry.register_nlp(SpaCyNEREngine())
registry.register_nlp(CRFNEREngine())
registry.register_nlp(TransformerNEREngine())
registry.register_graph(DijkstraEngine())
registry.register_graph(BFSEngine())
registry.register_graph(AStarEngine())


# --- CO2 estimates (grams per passenger-km) by train type ---
# Sources: ADEME, SNCF environmental reports (approximate)
CO2_PER_KM: dict[str, float] = {
    "TGV INOUI": 2.4,
    "OUIGO": 2.4,
    "Lyria": 2.4,
    "ICE": 3.0,
    "INTERCITES": 5.0,
    "INTERCITES de nuit": 5.0,
    "TER": 25.0,
    "Car TER": 68.0,
    "Navette": 10.0,
    "TramTrain": 4.0,
    "Train": 8.0,
}
# Fallback for unknown types
_DEFAULT_CO2 = 10.0
# Rough km/min for converting weight_minutes to distance (used in pollution calc)
_AVG_KM_PER_MIN = 2.0


def pollution_weight(edge: Edge) -> float:
    """Weight an edge by estimated CO2 emissions (grams)."""
    co2_per_km = CO2_PER_KM.get(edge.train_type, _DEFAULT_CO2)
    estimated_km = edge.weight_minutes * _AVG_KM_PER_MIN
    return co2_per_km * estimated_km


# --- Price estimates (EUR per km) by train type ---
# Very rough averages based on SNCF typical fares
PRICE_PER_KM: dict[str, float] = {
    "TGV INOUI": 0.12,
    "OUIGO": 0.06,
    "Lyria": 0.20,
    "ICE": 0.18,
    "INTERCITES": 0.10,
    "INTERCITES de nuit": 0.08,
    "TER": 0.14,
    "Car TER": 0.08,
    "Navette": 0.05,
    "TramTrain": 0.10,
    "Train": 0.10,
}
_DEFAULT_PRICE = 0.10


def _estimate_co2(train_type: str, duration_min: float) -> float:
    """Estimate CO2 in grams for a segment."""
    co2_per_km = CO2_PER_KM.get(train_type, _DEFAULT_CO2)
    return co2_per_km * duration_min * _AVG_KM_PER_MIN


def _estimate_price(train_type: str, duration_min: float) -> float:
    """Estimate price in EUR for a segment."""
    price_per_km = PRICE_PER_KM.get(train_type, _DEFAULT_PRICE)
    return price_per_km * duration_min * _AVG_KM_PER_MIN


def group_into_billets(segments: list[SegmentInfo]) -> list[BilletInfo]:
    """Group consecutive segments with the same train type into billets."""
    if not segments:
        return []

    billets: list[BilletInfo] = []
    current_type = segments[0].train_type
    current_from = segments[0].from_station
    current_duration = 0.0
    stops: list[str] = []

    for i, seg in enumerate(segments):
        if seg.train_type != current_type:
            # Train type changed: close current billet
            billets.append(BilletInfo(
                from_station=current_from,
                to_station=segments[i - 1].to_station,
                train_type=current_type,
                duration_minutes=round(current_duration, 1),
                co2_grams=round(_estimate_co2(current_type, current_duration), 1),
                price_euros=round(_estimate_price(current_type, current_duration), 2),
                stops=stops,
            ))
            # Start new billet
            current_type = seg.train_type
            current_from = seg.from_station
            current_duration = seg.weight_minutes
            stops = []
        else:
            if i > 0 and seg.train_type == current_type:
                # Intermediate stop
                stops.append(seg.from_station)
            current_duration += seg.weight_minutes

    # Close last billet
    billets.append(BilletInfo(
        from_station=current_from,
        to_station=segments[-1].to_station,
        train_type=current_type,
        duration_minutes=round(current_duration, 1),
        co2_grams=round(_estimate_co2(current_type, current_duration), 1),
        price_euros=round(_estimate_price(current_type, current_duration), 2),
        stops=stops,
    ))

    return billets


# --- Request/Response models ---

class ResolveRequest(BaseModel):
    text: str
    nlp_engine: str = "rule-based"
    graph_engine: str = "dijkstra"
    train_types: list[str] | None = None
    optimization: str = "time"


class SegmentInfo(BaseModel):
    from_station: str
    to_station: str
    train_type: str
    weight_minutes: float


class BilletInfo(BaseModel):
    """One ticket = consecutive segments on the same train type."""
    from_station: str
    to_station: str
    train_type: str
    duration_minutes: float
    co2_grams: float
    price_euros: float
    stops: list[str]  # intermediate stations (excluding from/to)


class ResolveResponse(BaseModel):
    text: str
    departure: str | None
    destination: str | None
    route: list[str]
    segments: list[SegmentInfo]
    billets: list[BilletInfo]
    travel_time_min: float
    total_co2_grams: float
    total_price_euros: float
    is_valid: bool
    nlp_time_ms: float
    graph_time_ms: float
    total_time_ms: float


class BenchmarkNLPRequest(BaseModel):
    text: str


class BenchmarkGraphRequest(BaseModel):
    departure: str
    destination: str


class EngineInfo(BaseModel):
    name: str
    description: str
    type: str


# --- Endpoints ---

@app.post("/api/resolve")
def resolve(req: ResolveRequest) -> ResolveResponse:
    """Main endpoint: text -> route."""
    nlp = registry.get_nlp(req.nlp_engine)
    graph = registry.get_graph(req.graph_engine)

    nlp_result = nlp.run(req.text)

    route: list[str] = []
    segments: list[SegmentInfo] = []
    travel_time = 0.0
    graph_time = 0.0

    billets: list[BilletInfo] = []

    if nlp_result.is_valid:
        train_filter = set(req.train_types) if req.train_types else None
        weight_fn = pollution_weight if req.optimization == "pollution" else None
        graph_result = graph.run(
            nlp_result.departure, nlp_result.destination,  # type: ignore[arg-type]
            train_type_filter=train_filter,
            weight_fn=weight_fn,
        )
        route = graph_result.steps
        segments = [
            SegmentInfo(
                from_station=s.from_station,
                to_station=s.to_station,
                train_type=s.train_type,
                weight_minutes=s.weight_minutes,
            )
            for s in graph_result.segments
        ]
        billets = group_into_billets(segments)
        travel_time = graph_result.total_weight
        graph_time = graph_result.time_ms

    total_co2 = sum(b.co2_grams for b in billets)
    total_price = sum(b.price_euros for b in billets)

    return ResolveResponse(
        text=req.text,
        departure=nlp_result.departure,
        destination=nlp_result.destination,
        route=route,
        segments=segments,
        billets=billets,
        travel_time_min=travel_time,
        total_co2_grams=round(total_co2, 1),
        total_price_euros=round(total_price, 2),
        is_valid=nlp_result.is_valid and len(route) >= 2,
        nlp_time_ms=nlp_result.time_ms,
        graph_time_ms=graph_time,
        total_time_ms=round(nlp_result.time_ms + graph_time, 3),
    )


@app.post("/api/benchmark/nlp")
def benchmark_nlp(req: BenchmarkNLPRequest) -> list[dict]:
    """Run all NLP engines on the same text and compare."""
    results = []
    for engine in registry.list_nlp():
        r = engine.run(req.text)
        results.append(asdict(r))
    return results


@app.post("/api/benchmark/graph")
def benchmark_graph(req: BenchmarkGraphRequest) -> list[dict]:
    """Run all graph engines on the same route and compare."""
    results = []
    for engine in registry.list_graph():
        r = engine.run(req.departure, req.destination)
        d = asdict(r)
        results.append(d)
    return results


@app.get("/api/engines")
def list_engines() -> list[EngineInfo]:
    """List all available engines."""
    engines = []
    for e in registry.list_nlp():
        engines.append(EngineInfo(name=e.name, description=e.description, type="nlp"))
    for e in registry.list_graph():
        engines.append(EngineInfo(name=e.name, description=e.description, type="graph"))
    return engines


@app.get("/api/health")
def health_check():
    """Health check endpoint for frontend status bar."""
    stations_path = DATA_DIR / "stations.json"
    connections_path = DATA_DIR / "connections.json"
    n_stations = 0
    n_connections = 0
    if stations_path.exists():
        with open(stations_path) as f:
            n_stations = len(json.load(f))
    if connections_path.exists():
        with open(connections_path) as f:
            n_connections = len(json.load(f))
    return {
        "status": "ok",
        "nlp_engines": len(registry.list_nlp()),
        "graph_engines": len(registry.list_graph()),
        "stations": n_stations,
        "connections": n_connections,
    }


@app.get("/api/stats")
def stats():
    """Stats endpoint for frontend footer."""
    stations_path = DATA_DIR / "stations.json"
    connections_path = DATA_DIR / "connections.json"
    n_stations = 0
    n_connections = 0
    if stations_path.exists():
        with open(stations_path) as f:
            n_stations = len(json.load(f))
    if connections_path.exists():
        with open(connections_path) as f:
            n_connections = len(json.load(f))
    return {
        "total_stations": n_stations,
        "total_connections": n_connections,
        "nlp_engines": [e.name for e in registry.list_nlp()],
        "graph_engines": [e.name for e in registry.list_graph()],
    }


# Serve static frontend files
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
