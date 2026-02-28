"""Dijkstra pathfinding engine on the SNCF train network."""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from src.graph.base import Edge, GraphEngine, RouteSegment

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class DijkstraEngine(GraphEngine):
    """Shortest path using Dijkstra's algorithm (weighted by travel time)."""

    def __init__(self) -> None:
        self._adjacency: dict[str, list[Edge]] | None = None

    @property
    def name(self) -> str:
        return "dijkstra"

    @property
    def description(self) -> str:
        return "Dijkstra's algorithm. Optimal shortest path by travel time (minutes)."

    @property
    def adjacency(self) -> dict[str, list[Edge]]:
        if self._adjacency is None:
            self._adjacency = defaultdict(list)
            with open(PROCESSED_DIR / "connections.json", encoding="utf-8") as f:
                edges = json.load(f)
            for e in edges:
                edge = Edge(e["from"], e["to"], e["train_type"], e["weight_minutes"])
                self._adjacency[edge.from_station].append(edge)
                # Reverse edge (JSON is directional from GTFS trips)
                rev = Edge(e["to"], e["from"], e["train_type"], e["weight_minutes"])
                self._adjacency[rev.from_station].append(rev)
        return self._adjacency

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

        dist: dict[str, float] = {departure: 0}
        prev: dict[str, tuple[str | None, Edge | None]] = {departure: (None, None)}
        pq: list[tuple[float, str]] = [(0, departure)]
        explored = 0

        while pq:
            d, node = heapq.heappop(pq)
            explored += 1

            if node == destination:
                # Reconstruct path and segments
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
                return path, segments, d, explored

            if d > dist.get(node, float("inf")):
                continue

            for edge in adj[node]:
                if train_type_filter and edge.train_type not in train_type_filter:
                    continue
                w = get_weight(edge)
                new_dist = d + w
                if new_dist < dist.get(edge.to_station, float("inf")):
                    dist[edge.to_station] = new_dist
                    prev[edge.to_station] = (node, edge)
                    heapq.heappush(pq, (new_dist, edge.to_station))

        return [], [], 0.0, explored
