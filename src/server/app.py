"""FastAPI server -- connects NLP engines and graph engines."""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path

from src.graph.base import Edge, RouteSegment
from src.nlp.base import NLPResult
from src.nlp.rule_based import RuleBasedEngine
from src.nlp.spacy_ner import SpaCyNEREngine
from src.nlp.crf_ner import CRFNEREngine
from src.nlp.transformer_ner import TransformerNEREngine
from src.graph.base import GraphResult, resolve_station
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


def hops_weight(edge: Edge) -> float:
    """Weight every edge equally — minimizes number of connections."""
    return 1.0


# Map optimization mode to weight function (None = default edge.weight_minutes)
_WEIGHT_FNS: dict[str, object] = {
    "time": None,
    "pollution": pollution_weight,
    "hops": hops_weight,
}


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


class BenchmarkStreamRequest(BaseModel):
    text: str
    expected_departure: str
    expected_destination: str


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
        weight_fn = _WEIGHT_FNS.get(req.optimization)
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


# --- Energy estimation ---
# Approximation: 15W TDP for a single CPU core
_CPU_TDP_W = 15.0


def _energy_uwh(cpu_time_ms: float) -> float:
    """Estimate energy in micro-watt-hours from CPU time."""
    return (cpu_time_ms / 1000) * _CPU_TDP_W / 3600 * 1e6


def _match_nlp(result, expected_dep: str, expected_dest: str) -> str:
    """Compare NLP result against ground truth.

    Resolves expected values through the gazetteer so that
    "Albi" matches "Albi Ville" (the canonical station name).
    """
    if not result.is_valid:
        return "missing"
    dep = (result.departure or "").strip().lower()
    dest = (result.destination or "").strip().lower()
    # Resolve expected values through the gazetteer
    exp_dep = resolve_station(expected_dep).strip().lower()
    exp_dest = resolve_station(expected_dest).strip().lower()
    if dep == exp_dep and dest == exp_dest:
        return "correct"
    if dep == exp_dest and dest == exp_dep:
        return "inverted"
    return "wrong"


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/benchmark/stream")
async def benchmark_stream(req: BenchmarkStreamRequest):
    """SSE endpoint: stream NLP then Graph benchmark results one by one."""

    async def generate():
        nlp_engines = registry.list_nlp()
        graph_engines = registry.list_graph()

        # Phase 1: NLP
        yield _sse_event({"type": "phase", "phase": "nlp", "total": len(nlp_engines)})

        for i, engine in enumerate(nlp_engines):
            try:
                result = await asyncio.to_thread(engine.run, req.text)
                d = asdict(result)
                d["type"] = "nlp_result"
                d["index"] = i
                d["total"] = len(nlp_engines)
                d["energy_uwh"] = round(_energy_uwh(result.cpu_time_ms), 2)
                d["match"] = _match_nlp(result, req.expected_departure, req.expected_destination)
            except Exception as exc:
                d = {
                    "type": "nlp_result",
                    "engine_name": engine.name,
                    "index": i,
                    "total": len(nlp_engines),
                    "error": str(exc),
                    "match": "missing",
                }
            yield _sse_event(d)

        # Phase 2: Graph — run all optimization modes × all engines
        opt_modes = list(_WEIGHT_FNS.keys())  # ["time", "pollution", "hops"]
        total_graph = len(graph_engines) * len(opt_modes)
        yield _sse_event({"type": "phase", "phase": "graph", "total": total_graph})

        idx = 0
        for opt_mode in opt_modes:
            weight_fn = _WEIGHT_FNS.get(opt_mode)
            for engine in graph_engines:
                try:
                    result = await asyncio.to_thread(
                        engine.run, req.expected_departure, req.expected_destination,
                        weight_fn=weight_fn,
                    )
                    d = asdict(result)
                    # Convert RouteSegment dataclasses to plain dicts
                    d["segments"] = [asdict(s) if hasattr(s, "__dataclass_fields__") else s for s in result.segments]
                    d["type"] = "graph_result"
                    d["index"] = idx
                    d["total"] = total_graph
                    d["optimization"] = opt_mode
                    d["energy_uwh"] = round(_energy_uwh(result.cpu_time_ms), 2)
                    # BFS always minimizes hops regardless of weight_fn
                    d["warning"] = ""
                    if engine.name == "bfs" and opt_mode != "hops":
                        d["warning"] = "BFS always minimizes hops regardless of optimization mode"
                except Exception as exc:
                    d = {
                        "type": "graph_result",
                        "engine_name": engine.name,
                        "index": idx,
                        "total": total_graph,
                        "error": str(exc),
                        "optimization": opt_mode,
                    }
                yield _sse_event(d)
                idx += 1

        yield _sse_event({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    """Speech-to-text only (no route resolution)."""
    try:
        import whisper
    except ImportError:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=501,
            content={"error": "whisper is not installed on this server"},
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp.flush()

        model = whisper.load_model("base")
        result = await asyncio.to_thread(model.transcribe, tmp.name, language="fr")

    return {"transcribed_text": result["text"].strip()}


# Serve static frontend files
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
