"""Base interface for pathfinding engines (Strategy pattern)."""

from __future__ import annotations

import json
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from unidecode import unidecode

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


@dataclass(frozen=True)
class Edge:
    """A single directed edge in the multigraph."""

    from_station: str
    to_station: str
    train_type: str
    weight_minutes: float


@dataclass
class RouteSegment:
    """One leg of a resolved route (station-to-station with train info)."""

    from_station: str
    to_station: str
    train_type: str
    weight_minutes: float


@dataclass
class GraphResult:
    """Result returned by any pathfinding engine."""

    steps: list[str] = field(default_factory=list)
    segments: list[RouteSegment] = field(default_factory=list)
    found: bool = False
    total_weight: float = 0.0
    nodes_explored: int = 0
    # Metrics (filled automatically by GraphEngine.run)
    time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_kb: float = 0.0
    engine_name: str = ""
    # Station resolution info (original → resolved via gazetteer)
    resolved_departure: str = ""
    resolved_destination: str = ""


# Lazy-loaded shared gazetteer for station name resolution
_STATION_GAZETTEER: dict[str, str] | None = None


def _load_gazetteer() -> dict[str, str]:
    global _STATION_GAZETTEER
    if _STATION_GAZETTEER is None:
        path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "gazetteer.json"
        with open(path, encoding="utf-8") as f:
            _STATION_GAZETTEER = json.load(f)
    return _STATION_GAZETTEER


def resolve_station(name: str) -> str:
    """Resolve a city/station name to its canonical station name via gazetteer.

    If the resolved name is not a node in the graph, falls back to the
    geographically nearest station (using GPS coordinates).
    """
    gaz = _load_gazetteer()
    key = name.strip().lower()
    resolved = None
    if key in gaz:
        resolved = gaz[key]
    else:
        key_ascii = unidecode(key)
        if key_ascii in gaz:
            resolved = gaz[key_ascii]

    if resolved is None:
        return name

    # Check if resolved station exists in the graph; if not, find nearest
    adj = load_adjacency()
    if resolved in adj:
        return resolved

    from src.graph.hub import find_nearest_station
    nearest = find_nearest_station(resolved, set(adj.keys()))
    return nearest if nearest else resolved


_ADJACENCY_CACHE: dict[str, list[Edge]] | None = None


def load_adjacency() -> dict[str, list[Edge]]:
    """Load the shared adjacency list (connections + intra-city hub transfers).

    Cached after first call so all engines share the same graph data.
    """
    global _ADJACENCY_CACHE
    if _ADJACENCY_CACHE is not None:
        return _ADJACENCY_CACHE

    from src.graph.hub import get_hub_edges

    adj: dict[str, list[Edge]] = defaultdict(list)
    with open(PROCESSED_DIR / "connections.json", encoding="utf-8") as f:
        edges = json.load(f)
    for e in edges:
        edge = Edge(e["from"], e["to"], e["train_type"], e["weight_minutes"])
        adj[edge.from_station].append(edge)
        rev = Edge(e["to"], e["from"], e["train_type"], e["weight_minutes"])
        adj[rev.from_station].append(rev)

    # Add intra-city hub transfer edges
    for edge in get_hub_edges():
        adj[edge.from_station].append(edge)

    _ADJACENCY_CACHE = dict(adj)
    return _ADJACENCY_CACHE


class GraphEngine(ABC):
    """Abstract base class for all pathfinding engines.

    Every engine must implement `find_route()` which takes a departure
    and destination station and returns the optimal path.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this engine (e.g. 'dijkstra', 'astar')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this algorithm."""

    @abstractmethod
    def find_route(
        self,
        departure: str,
        destination: str,
        train_type_filter: set[str] | None = None,
        weight_fn: Callable[[Edge], float] | None = None,
    ) -> tuple[list[str], list[RouteSegment], float, int]:
        """Find the best route between two stations.

        Args:
            departure: Start station name.
            destination: End station name.
            train_type_filter: If set, only use edges whose train_type is in this set.
            weight_fn: Custom weight function; defaults to edge.weight_minutes.

        Returns:
            (steps, segments, total_weight, nodes_explored).
            Return ([], [], 0.0, 0) if no route found.
        """

    def run(
        self,
        departure: str,
        destination: str,
        train_type_filter: set[str] | None = None,
        weight_fn: Callable[[Edge], float] | None = None,
    ) -> GraphResult:
        """Run pathfinding with automatic timing and memory measurement.

        When departure or destination resolves to a station that is part of
        a city hub (e.g. Paris), tries all hub stations and returns the best
        route (avoids unnecessary intra-city transfers).
        """
        from src.graph.hub import get_hub_siblings

        resolved_dep = resolve_station(departure)
        resolved_dest = resolve_station(destination)

        # Expand hub stations: try all siblings to find the best route
        dep_candidates = get_hub_siblings(resolved_dep)
        dest_candidates = get_hub_siblings(resolved_dest)

        tracemalloc.start()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        try:
            best = ([], [], float("inf"), 0)
            best_dep = resolved_dep
            best_dest = resolved_dest
            total_explored = 0

            for dep_c in dep_candidates:
                for dest_c in dest_candidates:
                    if dep_c == dest_c:
                        continue
                    steps, segments, weight, explored = self.find_route(
                        dep_c, dest_c, train_type_filter, weight_fn
                    )
                    total_explored += explored
                    if steps and weight < best[2]:
                        best = (steps, segments, weight, total_explored)
                        best_dep = dep_c
                        best_dest = dest_c

            steps, segments, weight, _ = best
            if weight == float("inf"):
                weight = 0.0
        finally:
            elapsed_ms = (time.perf_counter() - wall_start) * 1000
            cpu_elapsed_ms = (time.process_time() - cpu_start) * 1000
            _, peak_kb = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        return GraphResult(
            steps=steps,
            segments=segments,
            found=len(steps) >= 2,
            total_weight=round(weight, 2),
            nodes_explored=total_explored,
            time_ms=round(elapsed_ms, 3),
            cpu_time_ms=round(cpu_elapsed_ms, 3),
            memory_kb=round(peak_kb / 1024, 2),
            engine_name=self.name,
            resolved_departure=best_dep,
            resolved_destination=best_dest,
        )
