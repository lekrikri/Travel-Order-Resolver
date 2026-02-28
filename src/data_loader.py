"""Load and process SNCF GTFS data into a station gazetteer and connection graph."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from unidecode import unidecode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

GTFS_DIR = RAW_DIR / "sncf-gtfs"


def load_stops() -> dict[str, dict]:
    """Load station data from GTFS stops.txt.

    Returns dict of stop_id -> {name, lat, lon} for parent stations (StopArea) only.
    """
    stops = {}
    with open(GTFS_DIR / "stops.txt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only keep parent stations (location_type=1), not individual platforms
            if row["location_type"] == "1":
                stops[row["stop_id"]] = {
                    "name": row["stop_name"],
                    "lat": float(row["stop_lat"]),
                    "lon": float(row["stop_lon"]),
                }
    return stops


def load_stop_times() -> dict[str, list[dict]]:
    """Load stop_times.txt grouped by trip_id.

    Returns dict of trip_id -> sorted list of {stop_id, arrival, departure, sequence}.
    """
    trips: dict[str, list[dict]] = defaultdict(list)
    with open(GTFS_DIR / "stop_times.txt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trips[row["trip_id"]].append(
                {
                    "stop_id": row["stop_id"],
                    "arrival": row["arrival_time"],
                    "departure": row["departure_time"],
                    "sequence": int(row["stop_sequence"]),
                }
            )
    # Sort each trip's stops by sequence
    for trip_id in trips:
        trips[trip_id].sort(key=lambda x: x["sequence"])
    return trips


def extract_train_type(stop_id: str) -> str:
    """Extract the train type from a GTFS StopPoint ID.

    Examples:
        "StopPoint:OCETrain TER-87571000" -> "TER"
        "StopPoint:OCETGV INOUI-87686006" -> "TGV INOUI"
        "StopPoint:OCEOUIGO-87686006"     -> "OUIGO"
        "StopPoint:OCENavette-87571000"   -> "Navette"
        "StopPoint:OCECar TER-87571000"   -> "Car TER"
        "StopPoint:OCEICE-87212027"       -> "ICE"

    Known types in the dataset:
        Car TER, ICE, INTERCITES, INTERCITES de nuit, Lyria,
        Navette, OUIGO, TGV INOUI, Train, Train TER, TramTrain

    Returns the extracted train type string, or "Unknown" if parsing fails.
    """
    # The train type is between "OCE" and the last "-" in the stop_id
    oce_idx = stop_id.find("OCE")
    if oce_idx == -1:
        return "Unknown"
    after_oce = stop_id[oce_idx + 3:]
    dash_idx = after_oce.rfind("-")
    if dash_idx == -1:
        return "Unknown"
    raw = after_oce[:dash_idx]
    # Normalize: strip redundant "Train " prefix (but keep "Train" alone as-is)
    if raw.startswith("Train "):
        return raw[6:]
    return raw


def stoppoint_to_area(stop_id: str) -> str:
    """Convert StopPoint ID to its parent StopArea ID.

    StopPoint:OCETrain TER-87571000 -> StopArea:OCE87571000
    """
    # Extract the numeric code from the stop_id
    parts = stop_id.split("-")
    if len(parts) >= 2:
        code = parts[-1]
        return f"StopArea:OCE{code}"
    return stop_id


def time_to_minutes(t: str) -> int:
    """Convert HH:MM:SS to minutes since midnight."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def build_connections(
    stops: dict[str, dict], stop_times: dict[str, list[dict]]
) -> dict[tuple[str, str, str], int]:
    """Build direct connections between stations from trip data.

    For each trip, consecutive stops form a direct connection.
    We keep the minimum travel time per (from, to, train_type) combo.

    Returns dict of (from_name, to_name, train_type) -> min_travel_time_minutes.
    """
    connections: dict[tuple[str, str, str], int] = {}

    for trip_stops in stop_times.values():
        for i in range(len(trip_stops) - 1):
            from_stop = stoppoint_to_area(trip_stops[i]["stop_id"])
            to_stop = stoppoint_to_area(trip_stops[i + 1]["stop_id"])

            if from_stop not in stops or to_stop not in stops:
                continue
            if from_stop == to_stop:
                continue

            from_name = stops[from_stop]["name"]
            to_name = stops[to_stop]["name"]
            train_type = extract_train_type(trip_stops[i]["stop_id"])

            dep_time = time_to_minutes(trip_stops[i]["departure"])
            arr_time = time_to_minutes(trip_stops[i + 1]["arrival"])
            travel_time = arr_time - dep_time

            if travel_time <= 0:
                continue

            key = (from_name, to_name, train_type)
            if key not in connections or travel_time < connections[key]:
                connections[key] = travel_time

    return connections


def build_gazetteer(
    stops: dict[str, dict],
    connections: dict[tuple[str, str, str], int] | None = None,
) -> dict[str, str]:
    """Build a normalized station name -> canonical name lookup.

    Multiple normalized forms map to the same canonical name:
    - Full station name: "paris gare du nord" -> "Paris Gare du Nord"
    - Without accents: "beziers" -> station name
    - Without hyphens: "saint etienne" -> station name
    - City-only shortcut: "paris" -> most connected Paris station
    """
    gazetteer: dict[str, str] = {}

    # Count connections per station to determine "main" station per city
    conn_count: dict[str, int] = defaultdict(int)
    if connections:
        for from_name, to_name, _ in connections:
            conn_count[from_name] += 1
            conn_count[to_name] += 1

    for stop in stops.values():
        name = stop["name"]
        _add_variants(gazetteer, name)

    # Build city-level shortcuts: "paris" -> most connected Paris station
    # Group stations by their city prefix (first word, or first two for "Saint-X")
    city_stations: dict[str, list[str]] = defaultdict(list)
    for stop in stops.values():
        name = stop["name"]
        city = _extract_city(name)
        if city:
            city_stations[city.lower()].append(name)

    for city, station_names in city_stations.items():
        # Pick the station with the most connections
        best = max(station_names, key=lambda s: conn_count.get(s, 0))
        # Only add if the city name doesn't already map to something
        if city not in gazetteer:
            gazetteer[city] = best
        if unidecode(city) not in gazetteer:
            gazetteer[unidecode(city)] = best

    return gazetteer


def _extract_city(station_name: str) -> str | None:
    """Extract the city name from a station name.

    "Paris Gare du Nord" -> "paris"
    "Lyon Part Dieu" -> "lyon"
    "Saint-Étienne Châteaucreux" -> "saint-etienne"
    "Toulouse Matabiau" -> "toulouse"
    "Aix-en-Provence TGV" -> "aix-en-provence"
    """
    name = station_name.lower().strip()
    words = name.split()
    if not words:
        return None

    # Handle compound city names starting with saint/aix/etc.
    compound_prefixes = ["saint", "saint-", "aix", "aix-", "mont", "mont-",
                         "la", "le", "les", "port", "port-"]
    if len(words) >= 2 and words[0].rstrip("-") in [p.rstrip("-") for p in compound_prefixes]:
        return f"{words[0]}-{words[1]}".replace("--", "-")

    return words[0]


def _add_variants(gazetteer: dict[str, str], name: str) -> None:
    """Add all normalized variants of a station name to the gazetteer."""
    gazetteer[name.lower()] = name
    gazetteer[unidecode(name).lower()] = name
    gazetteer[name.lower().replace("-", " ")] = name
    gazetteer[unidecode(name).lower().replace("-", " ")] = name
    # Without "gare de" prefix
    for prefix in ["gare de ", "gare du ", "gare d'"]:
        lower = name.lower()
        if lower.startswith(prefix):
            short = lower[len(prefix):]
            gazetteer[short] = name
            gazetteer[unidecode(short)] = name


def process_and_save() -> None:
    """Process raw GTFS data and save processed files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stops...")
    stops = load_stops()
    print(f"  {len(stops)} parent stations loaded")

    print("Loading stop_times (this may take a moment)...")
    stop_times = load_stop_times()
    print(f"  {len(stop_times)} trips loaded")

    print("Building connections...")
    connections = build_connections(stops, stop_times)
    print(f"  {len(connections)} direct connections found")

    print("Building gazetteer...")
    gazetteer = build_gazetteer(stops, connections)
    print(f"  {len(gazetteer)} normalized forms")

    # Save gazetteer
    with open(PROCESSED_DIR / "gazetteer.json", "w", encoding="utf-8") as f:
        json.dump(gazetteer, f, ensure_ascii=False, indent=2)

    # Save stations with coordinates
    stations = {
        stop["name"]: {"lat": stop["lat"], "lon": stop["lon"]}
        for stop in stops.values()
    }
    with open(PROCESSED_DIR / "stations.json", "w", encoding="utf-8") as f:
        json.dump(stations, f, ensure_ascii=False, indent=2)

    # Save connections as edge list (with train_type for multigraph)
    edges = [
        {"from": k[0], "to": k[1], "train_type": k[2], "weight_minutes": v}
        for k, v in connections.items()
    ]
    with open(PROCESSED_DIR / "connections.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {PROCESSED_DIR}/:")
    print(f"  gazetteer.json  ({len(gazetteer)} entries)")
    print(f"  stations.json   ({len(stations)} stations)")
    print(f"  connections.json ({len(connections)} edges)")


if __name__ == "__main__":
    process_and_save()
