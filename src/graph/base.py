"""Base interface for pathfinding engines (Strategy pattern)."""

from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


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
    memory_kb: float = 0.0
    engine_name: str = ""


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
        """Run pathfinding with automatic timing and memory measurement."""
        tracemalloc.start()
        start = time.perf_counter()

        steps, segments, weight, explored = self.find_route(
            departure, destination, train_type_filter, weight_fn
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return GraphResult(
            steps=steps,
            segments=segments,
            found=len(steps) >= 2,
            total_weight=round(weight, 2),
            nodes_explored=explored,
            time_ms=round(elapsed_ms, 3),
            memory_kb=round(peak_kb / 1024, 2),
            engine_name=self.name,
        )
