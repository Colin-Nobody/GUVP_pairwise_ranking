#!/usr/bin/env python3
"""Select a reproducible Mapillary street-image sample for 50 OECD-linked cities.

Mapillary image vector tiles are queried around each urban center. Candidate
images are restricted to non-panoramic images within a 6 km radius, deduplicated
by capture sequence, and sampled across 1 km grid cells. This script only writes
metadata; image thumbnails are downloaded separately through the Graph API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import requests


SEED = 20261024
ZOOM = 14
EXTENT = 4096
RADIUS_KM = 6.0
PER_CITY = 200
TOKEN_FILE = Path.home() / ".config/guvp/mapillary.env"
TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}/"
FIELDS = [
    "uuid", "source", "orig_id", "city", "city_ascii", "country", "continent",
    "oecd_region", "oecd_region_code", "lat", "lon", "sequence_id", "captured_at",
    "is_pano", "quality_score", "tile_z", "tile_x", "tile_y", "sample_seed",
    "selection_rank_within_city", "sampling_method",
]

# City centers are stable sample-frame anchors. OECD regions are the exact rows
# in the provided workbook; they are regional statistical units, not city data.
CITIES = [
    ("Amsterdam", "Netherlands", "North Holland", "NL32", 52.3676, 4.9041),
    ("Barcelona", "Spain", "Catalonia", "ES51", 41.3874, 2.1686),
    ("Berlin", "Germany", "Berlin", "DE3", 52.5200, 13.4050),
    ("Brussels", "Belgium", "Brussels Capital Region", "BE1", 50.8503, 4.3517),
    ("Budapest", "Hungary", "Budapest", "HU11", 47.4979, 19.0402),
    ("Copenhagen", "Denmark", "Capital Region", "DK01", 55.6761, 12.5683),
    ("Edinburgh", "United Kingdom", "Scotland", "UKM", 55.9533, -3.1883),
    ("Helsinki", "Finland", "Helsinki-Uusimaa", "FI1B", 60.1699, 24.9384),
    ("Krakow", "Poland", "Lesser Poland", "PL21", 50.0647, 19.9450),
    ("Madrid", "Spain", "Madrid", "ES30", 40.4168, -3.7038),
    ("Munich", "Germany", "Bavaria", "DE2", 48.1351, 11.5820),
    ("Oslo", "Norway", "Oslo and Viken", "NO08", 59.9139, 10.7522),
    ("Paris", "France", "Île-de-France", "FR1", 48.8566, 2.3522),
    ("Prague", "Czechia", "Prague", "CZ01", 50.0755, 14.4378),
    ("Rome", "Italy", "Lazio", "ITI4", 41.9028, 12.4964),
    ("Stockholm", "Sweden", "Stockholm", "SE11", 59.3293, 18.0686),
    ("Tallinn", "Estonia", "North Estonia", "EE001", 59.4370, 24.7536),
    ("Warsaw", "Poland", "Warsaw's capital city", "PL91", 52.2297, 21.0122),
    ("Zurich", "Switzerland", "Zurich", "CH04", 47.3769, 8.5417),
    ("Athens", "Greece", "Attica", "EL30", 37.9838, 23.7275),
    ("Bucharest", "Romania", "Bucharest - Ilfov", "RO32", 44.4268, 26.1025),
    ("Hamburg", "Germany", "Hamburg", "DE6", 53.5511, 9.9937),
    ("Lisbon", "Portugal", "Greater Lisbon", "PT1A", 38.7223, -9.1393),
    ("Lyon", "France", "Auvergne-Rhône-Alpes", "FRK", 45.7640, 4.8357),
    ("Milan", "Italy", "Lombardy", "ITC4", 45.4642, 9.1900),
    ("Riga", "Latvia", "Riga", "LV00A", 56.9496, 24.1052),
    ("Sofia", "Bulgaria", "South West", "BG41", 42.6977, 23.3219),
    ("Vienna", "Austria", "Vienna", "AT13", 48.2082, 16.3738),
    ("Vilnius", "Lithuania", "Vilnius", "LT011", 54.6872, 25.2797),
    ("Frankfurt", "Germany", "Hesse", "DE7", 50.1109, 8.6821),
    ("Cologne", "Germany", "North Rhine-Westphalia", "DEA", 50.9375, 6.9603),
    ("Marseille", "France", "Provence-Alpes-Côte d’Azur", "FRL", 43.2965, 5.3698),
    ("Toulouse", "France", "Occitanie", "FRJ", 43.6045, 1.4442),
    ("Naples", "Italy", "Campania", "ITF3", 40.8518, 14.2681),
    ("Turin", "Italy", "Piedmont", "ITC1", 45.0703, 7.6869),
    ("Valencia", "Spain", "Valencia", "ES52", 39.4699, -0.3763),
    ("Seville", "Spain", "Andalusia", "ES61", 37.3891, -5.9845),
    ("Wroclaw", "Poland", "Lower Silesia", "PL51", 51.1079, 17.0385),
    ("Gothenburg", "Sweden", "West Sweden", "SE23", 57.7089, 11.9746),
    ("Brno", "Czechia", "Southeast", "CZ06", 49.1951, 16.6068),
    ("Tampere", "Finland", "Western Finland", "FI19", 61.4978, 23.7610),
    ("Aarhus", "Denmark", "Central Jutland", "DK04", 56.1629, 10.2039),
    ("Porto", "Portugal", "North", "PT11", 41.1579, -8.6291),
    ("Antwerp", "Belgium", "Flemish Region", "BE2", 51.2194, 4.4025),
    ("Rotterdam", "Netherlands", "South Holland", "NL36", 51.9244, 4.4777),
    ("Dublin", "Ireland", "Eastern and Midland", "IE06", 53.3498, -6.2603),
    ("Geneva", "Switzerland", "Lake Geneva Region", "CH01", 46.2044, 6.1432),
    ("Zagreb", "Croatia", "City of Zagreb", "HR05", 45.8150, 15.9819),
    ("Bratislava", "Slovak Republic", "Bratislava Region", "SK01", 48.1486, 17.1077),
    ("Thessaloniki", "Greece", "Central Macedonia", "EL52", 40.6401, 22.9444),
]


def varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, pos
        shift += 7


def protobuf_fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    pos = 0
    while pos < len(buf):
        tag, pos = varint(buf, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 2:
            size, pos = varint(buf, pos)
            value = buf[pos : pos + size]
            pos += size
        elif wire == 1:
            value = buf[pos : pos + 8]
            pos += 8
        elif wire == 5:
            value = buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def packed_varints(buf: bytes) -> list[int]:
    result, pos = [], 0
    while pos < len(buf):
        value, pos = varint(buf, pos)
        result.append(value)
    return result


def mvt_value(buf: bytes) -> Any:
    import struct

    for number, wire, value in protobuf_fields(buf):
        if number == 1 and wire == 2:
            return value.decode("utf-8")
        if number == 2 and wire == 5:
            return struct.unpack("<f", value)[0]
        if number == 3 and wire == 1:
            return struct.unpack("<d", value)[0]
        if number in (4, 5) and wire == 0:
            return value
        if number == 6 and wire == 0:
            return (value >> 1) ^ -(value & 1)
        if number == 7 and wire == 0:
            return bool(value)
    return None


def tile_for(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_point_to_lonlat(x: int, y: int, zoom: int, px: float, py: float) -> tuple[float, float]:
    n = 2**zoom
    world_x = (x + px / EXTENT) / n
    world_y = (y + py / EXTENT) / n
    lon = world_x * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * world_y))))
    return lon, lat


def decode_image_layer(tile: bytes, x: int, y: int, zoom: int) -> list[dict[str, Any]]:
    for number, wire, layer in protobuf_fields(tile):
        if number != 3 or wire != 2:
            continue
        name, keys, values, features, extent = "", [], [], [], EXTENT
        for field, layer_wire, value in protobuf_fields(layer):
            if field == 1 and layer_wire == 2:
                name = value.decode("utf-8")
            elif field == 2 and layer_wire == 2:
                features.append(value)
            elif field == 3 and layer_wire == 2:
                keys.append(value.decode("utf-8"))
            elif field == 4 and layer_wire == 2:
                values.append(mvt_value(value))
            elif field == 5 and layer_wire == 0:
                extent = value
        if name != "image":
            continue
        result = []
        for feature in features:
            tags, geometry = [], []
            for field, feature_wire, value in protobuf_fields(feature):
                if field == 2 and feature_wire == 2:
                    tags = packed_varints(value)
                elif field == 4 and feature_wire == 2:
                    geometry = packed_varints(value)
            props = {keys[tags[i]]: values[tags[i + 1]] for i in range(0, len(tags), 2)}
            if props.get("is_pano") is True or not geometry:
                continue
            command = geometry[0]
            if (command & 7) != 1 or len(geometry) < 3:
                continue
            dx, dy = geometry[1], geometry[2]
            px, py = (dx >> 1) ^ -(dx & 1), (dy >> 1) ^ -(dy & 1)
            lon, lat = tile_point_to_lonlat(x, y, zoom, px * EXTENT / extent, py * EXTENT / extent)
            image_id = props.get("id") or props.get("image_id")
            if image_id is None:
                continue
            props.update(lon=lon, lat=lat)
            result.append(props)
        return result
    return []


def read_local_token() -> str:
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if token:
        return token
    if not TOKEN_FILE.is_file():
        raise RuntimeError(f"Local Mapillary token file is missing: {TOKEN_FILE}")
    for line in TOKEN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export MAPILLARY_ACCESS_TOKEN="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError(f"No non-empty MAPILLARY_ACCESS_TOKEN assignment found in {TOKEN_FILE}")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_tile(session: requests.Session, x: int, y: int) -> bytes:
    last_status = None
    for attempt in range(6):
        try:
            response = session.get(TILE_URL.format(z=ZOOM, x=x, y=y), timeout=(10, 60))
            if response.status_code == 429 or response.status_code >= 500:
                last_status = response.status_code
                response.close()
                time.sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 5:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Mapillary vector tile request failed after retries (last HTTP status: {last_status})")


def candidate_tiles(city: tuple[Any, ...], rings: int = 3) -> list[tuple[int, int]]:
    name, country, region, code, lat, lon = city
    x, y = tile_for(lon, lat, ZOOM)
    # Center first, then deterministic outward spiral. Radius filtering below
    # excludes neighboring city centers and distant suburb imagery.
    offsets = [(0, 0)]
    for ring in range(1, rings + 1):
        offsets.extend((dx, dy) for dy in range(-ring, ring + 1) for dx in range(-ring, ring + 1)
                       if max(abs(dx), abs(dy)) == ring)
    return [(x + dx, y + dy) for dx, dy in offsets]


def grid_key(lat: float, lon: float, size_degrees: float = 0.01) -> tuple[int, int]:
    # About 1.1 km north-south at European latitudes; longitude cell size
    # remains comparable after cosine adjustment in the grid-index step.
    lat_bin = math.floor(lat / size_degrees)
    lon_bin = math.floor(lon * math.cos(math.radians(lat)) / size_degrees)
    return lat_bin, lon_bin


def spatial_random_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_grid: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_grid[grid_key(row["lat"], row["lon"])].append(row)
    for items in by_grid.values():
        rng.shuffle(items)
    grid_order = list(by_grid)
    rng.shuffle(grid_order)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in grid_order:
            if by_grid[key]:
                selected.append(by_grid[key].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output candidate manifest CSV")
    parser.add_argument("--audit", type=Path, required=True, help="Output selection metadata JSON")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-city", type=int, default=PER_CITY)
    parser.add_argument("--cities", nargs="*", help="Optional city names to limit the run")
    parser.add_argument("--cities-csv", type=Path, help="Optional city roster CSV with city,country,oecd_region,oecd_region_code,lat,lon columns")
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude image IDs and capture sequences already present")
    parser.add_argument("--counts-json", type=Path, help="Optional JSON mapping city names to per-city selection counts")
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM)
    parser.add_argument("--tile-rings", type=int, default=3)
    args = parser.parse_args()

    token = read_local_token()
    if not token.startswith("MLY|"):
        raise RuntimeError("Local token does not have the expected Mapillary Client Token format")
    counts_by_city = json.loads(args.counts_json.read_text(encoding="utf-8")) if args.counts_json else {}
    cities = CITIES
    if args.cities_csv:
        with args.cities_csv.open(encoding="utf-8-sig", newline="") as handle:
            roster = list(csv.DictReader(handle))
        required_roster = {"city", "country", "oecd_region", "oecd_region_code", "lat", "lon"}
        if not roster or required_roster - set(roster[0]):
            raise ValueError(f"--cities-csv must contain columns: {sorted(required_roster)}")
        cities = []
        for row in roster:
            cities.append((row["city"], row["country"], row["oecd_region"], row["oecd_region_code"], float(row["lat"]), float(row["lon"])))
        if len({row[0] for row in cities}) != len(cities):
            raise ValueError("--cities-csv contains duplicate city names")
    if any(city not in {c[0] for c in cities} or int(count) < 1 for city, count in counts_by_city.items()):
        raise ValueError("--counts-json must map known city names to positive counts")
    if args.cities and args.cities_csv:
        raise ValueError("Use either --cities or --cities-csv, not both")
    cities = cities if not args.cities else [c for c in cities if c[0] in set(args.cities)]
    if args.cities and len(cities) != len(set(args.cities)):
        raise ValueError("Unknown city supplied to --cities")
    if args.counts_json and not args.cities:
        cities = [c for c in cities if c[0] in counts_by_city]

    session = requests.Session()
    session.params = {"access_token": token}
    # Lightweight auth probe. A valid token must get a Graph API response, even
    # though this collection endpoint has no unbounded no-bbox listing.
    probe = None
    for attempt in range(5):
        try:
            probe = requests.get("https://graph.mapillary.com/images", params={"fields": "id", "limit": 1, "access_token": token}, timeout=(10, 20))
            break
        except requests.RequestException:
            if attempt == 4:
                raise RuntimeError("Mapillary Graph API preflight failed after network retries") from None
            time.sleep(min(2**attempt, 15))
    if probe.status_code in (400, 401, 403):
        raise RuntimeError(f"Mapillary token rejected by Graph API (HTTP {probe.status_code})")
    probe.raise_for_status()

    all_rows, city_audits = [], []
    used_ids: set[int] = set()
    used_sequences: set[str] = set()
    if args.exclude_manifest:
        with args.exclude_manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("orig_id"):
                    used_ids.add(int(row["orig_id"]))
                if row.get("sequence_id"):
                    used_sequences.add(str(row["sequence_id"]))
    for city_index, city in enumerate(cities):
        name, country, region, code, center_lat, center_lon = city
        target_count = int(counts_by_city.get(name, args.per_city))
        unique_by_sequence: dict[str, dict[str, Any]] = {}
        used_tiles = 0
        for x, y in candidate_tiles(city, args.tile_rings):
            tile = fetch_tile(session, x, y)
            used_tiles += 1
            for row in decode_image_layer(tile, x, y, ZOOM):
                try:
                    image_id = int(row.get("id") or row.get("image_id"))
                    sequence_id = str(row.get("sequence_id") or "").strip()
                    lat, lon = float(row["lat"]), float(row["lon"])
                except (TypeError, ValueError):
                    continue
                if not sequence_id or haversine_km(center_lat, center_lon, lat, lon) > args.radius_km:
                    continue
                if image_id in used_ids or sequence_id in used_sequences:
                    continue
                if row.get("quality_score") is not None and float(row["quality_score"]) < 0.25:
                    continue
                candidate = {
                    "orig_id": image_id,
                    "sequence_id": sequence_id,
                    "lat": lat,
                    "lon": lon,
                    "captured_at": row.get("captured_at"),
                    "quality_score": row.get("quality_score"),
                    "tile_z": ZOOM,
                    "tile_x": x,
                    "tile_y": y,
                    "is_pano": False,
                }
                # If the same sequence appears in multiple tiles, retain the
                # clearest candidate while keeping one image per sequence.
                previous = unique_by_sequence.get(sequence_id)
                if previous is None or (candidate["quality_score"] or 0) > (previous["quality_score"] or 0):
                    unique_by_sequence[sequence_id] = candidate
            if len(unique_by_sequence) >= target_count:
                break
        candidates = list(unique_by_sequence.values())
        selected = spatial_random_sample(candidates, target_count, args.seed + city_index)
        used_ids.update(int(row["orig_id"]) for row in selected)
        used_sequences.update(str(row["sequence_id"]) for row in selected)
        city_audits.append({
            "city": name, "country": country, "oecd_region": region, "oecd_region_code": code,
            "center_lat": center_lat, "center_lon": center_lon, "radius_km": args.radius_km,
            "tiles_queried": used_tiles, "eligible_unique_sequences": len(candidates),
            "target_selected": target_count, "selected": len(selected),
            "status": "complete" if len(selected) == target_count else "insufficient_candidates",
        })
        for rank, item in enumerate(selected, start=1):
            all_rows.append({
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"guvp-mapillary-{args.seed}-{name}-{item['orig_id']}")),
                "source": "Mapillary", "orig_id": item["orig_id"], "city": name,
                "city_ascii": name, "country": country, "continent": "Europe",
                "oecd_region": region, "oecd_region_code": code, "lat": item["lat"],
                "lon": item["lon"], "sequence_id": item["sequence_id"],
                "captured_at": item["captured_at"], "is_pano": item["is_pano"],
                "quality_score": item["quality_score"], "tile_z": item["tile_z"],
                "tile_x": item["tile_x"], "tile_y": item["tile_y"],
                "sample_seed": args.seed, "selection_rank_within_city": rank,
                "sampling_method": f"random one-per-sequence; non-pano; quality>=0.25 when scored; within {args.radius_km:g}km of city center; round-robin 1km spatial grid",
            })
        print(f"{name}: candidates={len(candidates)} selected={len(selected)} tiles={used_tiles}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    audit = {
        "seed": args.seed, "per_city": args.per_city, "counts_by_city": counts_by_city or None,
        "mapillary_api": "mly1_public vector tiles + Graph API",
        "excluded_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else None,
        "sampled_cities": len(cities), "sampled_images": len(all_rows),
        "token_source": str(TOKEN_FILE) if not os.environ.get("MAPILLARY_ACCESS_TOKEN") else "MAPILLARY_ACCESS_TOKEN environment variable",
        "probe_http_status": probe.status_code,
        "candidate_rules": ["Europe cities matched by OECD region code", "non-panoramic image tiles", "at most one image per sequence", f"within {args.radius_km:g} km of city center", "quality_score >= 0.25 when score is available", "random draw round-robin over 1 km spatial cells"],
        "cities": city_audits,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    incomplete = [x for x in city_audits if x["selected"] != x["target_selected"]]
    if incomplete:
        raise SystemExit(f"Sample incomplete for {len(incomplete)} cities; inspect {args.audit}")


if __name__ == "__main__":
    main()
