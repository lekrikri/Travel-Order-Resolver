"""Intra-city hub detection and transfer edge generation.

Detects stations belonging to the same city using a hybrid approach:
prefix-based grouping + GPS proximity clustering (30km threshold).
Generates virtual "transfer" edges between stations in the same hub.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from src.graph.base import Edge

STATIONS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "stations.json"

_HUB_CACHE: list[Edge] | None = None
_HUB_MEMBERS: dict[str, list[str]] | None = None
_STATIONS_CACHE: dict[str, dict] | None = None

# Stations within this distance are considered part of the same city hub.
HUB_RADIUS_KM = 30
# Transfer time estimation: minutes per km of intra-city travel.
TRANSFER_MIN_PER_KM = 3.0
TRANSFER_MIN_FLOOR = 5.0
TRANSFER_MIN_CAP = 45.0
TRANSFER_TRAIN_TYPE = "transfer"


def _load_stations() -> dict[str, dict]:
    global _STATIONS_CACHE
    if _STATIONS_CACHE is None:
        with open(STATIONS_PATH, encoding="utf-8") as f:
            _STATIONS_CACHE = json.load(f)
    return _STATIONS_CACHE


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two GPS points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _cluster_by_proximity(
    stations: list[tuple[str, dict]],
    threshold_km: float,
) -> list[list[tuple[str, dict]]]:
    """Cluster stations into groups where every member is within threshold_km
    of at least one other member (single-linkage clustering)."""
    n = len(stations)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            _, ci = stations[i]
            _, cj = stations[j]
            if _haversine(ci["lat"], ci["lon"], cj["lat"], cj["lon"]) < threshold_km:
                union(i, j)

    groups: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(stations[i])

    return [g for g in groups.values() if len(g) > 1]


def _transfer_weight(coords_a: dict, coords_b: dict) -> float:
    """Estimate transfer time (minutes) between two stations based on GPS distance."""
    dist_km = _haversine(coords_a["lat"], coords_a["lon"], coords_b["lat"], coords_b["lon"])
    minutes = dist_km * TRANSFER_MIN_PER_KM
    return max(TRANSFER_MIN_FLOOR, min(TRANSFER_MIN_CAP, round(minutes, 1)))


def get_hub_edges() -> list[Edge]:
    """Return virtual transfer edges connecting stations within the same city hub."""
    global _HUB_CACHE
    if _HUB_CACHE is not None:
        return _HUB_CACHE

    all_stations = _load_stations()

    # Group station names by first word (city prefix)
    prefix_groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, coords in all_stations.items():
        prefix = name.split()[0] if name.split() else name
        prefix_groups[prefix].append((name, coords))

    edges: list[Edge] = []

    for _prefix, stations in prefix_groups.items():
        if len(stations) < 2:
            continue

        # Cluster by GPS proximity within this prefix group
        clusters = _cluster_by_proximity(stations, HUB_RADIUS_KM)

        for cluster in clusters:
            # Connect all pairs in the cluster (full mesh)
            for i, (name_a, coords_a) in enumerate(cluster):
                for name_b, coords_b in cluster[i + 1 :]:
                    weight = _transfer_weight(coords_a, coords_b)
                    edges.append(Edge(name_a, name_b, TRANSFER_TRAIN_TYPE, weight))
                    edges.append(Edge(name_b, name_a, TRANSFER_TRAIN_TYPE, weight))

    _HUB_CACHE = edges
    return edges


def get_hub_siblings(station: str) -> list[str]:
    """Return all stations in the same city hub as the given station.

    Returns a list including the station itself. If the station is not
    part of any hub, returns a single-element list [station].
    """
    global _HUB_MEMBERS
    if _HUB_MEMBERS is None:
        # Build the membership map from hub edges
        get_hub_edges()  # ensure hubs are computed
        members: dict[str, set[str]] = defaultdict(set)
        for edge in _HUB_CACHE or []:
            if edge.train_type == TRANSFER_TRAIN_TYPE:
                members[edge.from_station].add(edge.from_station)
                members[edge.from_station].add(edge.to_station)
        _HUB_MEMBERS = {k: sorted(v) for k, v in members.items()}

    return _HUB_MEMBERS.get(station, [station])


def find_nearest_station(
    name: str,
    graph_nodes: set[str],
    max_km: float = 50.0,
) -> str | None:
    """Find the nearest station in the graph to a given station name.

    Uses GPS coordinates from stations.json. Returns None if no station
    is within max_km or if the name has no known coordinates.
    """
    all_stations = _load_stations()

    if name not in all_stations:
        return None

    target = all_stations[name]
    best_name: str | None = None
    best_dist = max_km

    for station_name, coords in all_stations.items():
        if station_name not in graph_nodes or station_name == name:
            continue
        dist = _haversine(target["lat"], target["lon"], coords["lat"], coords["lon"])
        if dist < best_dist:
            best_dist = dist
            best_name = station_name

    return best_name
