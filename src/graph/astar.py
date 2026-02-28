"""A* pathfinding engine -- Dijkstra with haversine heuristic."""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from src.graph.base import Edge, GraphEngine, RouteSegment

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# Average train speed (km/h) used to convert distance heuristic to time estimate.
# Slightly overestimates speed so the heuristic stays admissible (never overestimates cost).
AVERAGE_SPEED_KMH = 200


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class AStarEngine(GraphEngine):
    """A* algorithm with haversine distance heuristic (travel time in minutes)."""

    def __init__(self) -> None:
        self._adjacency: dict[str, list[Edge]] | None = None
        self._coords: dict[str, tuple[float, float]] | None = None

    @property
    def name(self) -> str:
        return "astar"

    @property
    def description(self) -> str:
        return "A* with haversine heuristic. Same optimal result as Dijkstra, fewer nodes explored."

    @property
    def adjacency(self) -> dict[str, list[Edge]]:
        if self._adjacency is None:
            self._adjacency = defaultdict(list)
            with open(PROCESSED_DIR / "connections.json", encoding="utf-8") as f:
                edges = json.load(f)
            for e in edges:
                edge = Edge(e["from"], e["to"], e["train_type"], e["weight_minutes"])
                self._adjacency[edge.from_station].append(edge)
                rev = Edge(e["to"], e["from"], e["train_type"], e["weight_minutes"])
                self._adjacency[rev.from_station].append(rev)
        return self._adjacency

    @property
    def coords(self) -> dict[str, tuple[float, float]]:
        if self._coords is None:
            with open(PROCESSED_DIR / "stations.json", encoding="utf-8") as f:
                stations = json.load(f)
            self._coords = {
                name: (info["lat"], info["lon"]) for name, info in stations.items()
            }
        return self._coords

    def _heuristic(self, node: str, goal: str) -> float:
        """Estimated travel time (minutes) from node to goal."""
        if node not in self.coords or goal not in self.coords:
            return 0.0
        lat1, lon1 = self.coords[node]
        lat2, lon2 = self.coords[goal]
        dist_km = haversine_km(lat1, lon1, lat2, lon2)
        # Convert to minutes using optimistic speed so heuristic stays admissible
        return (dist_km / AVERAGE_SPEED_KMH) * 60

    def find_route(
        self,
        departure: str,
        destination: str,
        train_type_filter: set[str] | None = None,
        weight_fn: Callable[[Edge], float] | None = None,
    ) -> tuple[list[str], list[RouteSegment], float, int]:
        adj = self.adjacency
        get_weight = weight_fn or (lambda e: e.weight_minutes)

        if departure not in adj or destination not in adj:
            return [], [], 0.0, 0

        g_score: dict[str, float] = {departure: 0}
        prev: dict[str, tuple[str | None, Edge | None]] = {departure: (None, None)}
        pq: list[tuple[float, float, str]] = [
            (self._heuristic(departure, destination), 0, departure)
        ]
        explored = 0

        while pq:
            f, g, node = heapq.heappop(pq)
            explored += 1

            if node == destination:
                path: list[str] = []
                segments: list[RouteSegment] = []
                current: str | None = destination
                while current is not None:
                    path.append(current)
                    prev_node, edge = prev[current]
                    if edge is not None:
                        segments.append(RouteSegment(
                            edge.from_station, edge.to_station,
                            edge.train_type, edge.weight_minutes,
                        ))
                    current = prev_node
                path.reverse()
                segments.reverse()
                return path, segments, g, explored

            if g > g_score.get(node, float("inf")):
                continue

            for edge in adj[node]:
                if train_type_filter and edge.train_type not in train_type_filter:
                    continue
                w = get_weight(edge)
                new_g = g + w
                if new_g < g_score.get(edge.to_station, float("inf")):
                    g_score[edge.to_station] = new_g
                    prev[edge.to_station] = (node, edge)
                    f_score = new_g + self._heuristic(edge.to_station, destination)
                    heapq.heappush(pq, (f_score, new_g, edge.to_station))

        return [], [], 0.0, explored
