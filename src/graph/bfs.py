"""BFS pathfinding engine -- minimizes number of stops (unweighted)."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path

from src.graph.base import Edge, GraphEngine, RouteSegment

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class BFSEngine(GraphEngine):
    """Shortest path by number of stops (ignores travel time)."""

    def __init__(self) -> None:
        self._adjacency: dict[str, list[Edge]] | None = None

    @property
    def name(self) -> str:
        return "bfs"

    @property
    def description(self) -> str:
        return "Breadth-First Search. Fewest stops (ignores travel time weights)."

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

    def find_route(
        self,
        departure: str,
        destination: str,
        train_type_filter: set[str] | None = None,
        weight_fn: Callable[[Edge], float] | None = None,
    ) -> tuple[list[str], list[RouteSegment], float, int]:
        adj = self.adjacency
        # weight_fn is intentionally ignored: BFS minimizes stops, not weight

        if departure not in adj or destination not in adj:
            return [], [], 0.0, 0

        visited: set[str] = {departure}
        prev: dict[str, tuple[str | None, Edge | None]] = {departure: (None, None)}
        queue: deque[str] = deque([departure])
        explored = 0

        while queue:
            node = queue.popleft()
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
                weight = sum(s.weight_minutes for s in segments)
                return path, segments, float(len(path) - 1), explored

            for edge in adj[node]:
                if train_type_filter and edge.train_type not in train_type_filter:
                    continue
                # Deduplicate: only visit each station once (multiple edges may go there)
                if edge.to_station not in visited:
                    visited.add(edge.to_station)
                    prev[edge.to_station] = (node, edge)
                    queue.append(edge.to_station)

        return [], [], 0.0, explored
