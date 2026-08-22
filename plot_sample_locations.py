#!/usr/bin/env python3
"""Create label-free sample maps with buildings, roads, railways, and red points."""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageOps


TILE_SIZE = 256
RETINA_SCALE = 2
USER_AGENT = "GUVP-pairwise-ranking research map generator"
RAIL_TYPES = {"rail", "subway", "light_rail", "tram", "monorail", "narrow_gauge"}
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def lonlat_to_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * 2**zoom
    latitude = max(min(lat, 85.05112878), -85.05112878)
    sin_lat = math.sin(math.radians(latitude))
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def download_tile(
    session: requests.Session, zoom: int, tile_x: int, tile_y: int, cache: Path
) -> tuple[int, int, Image.Image]:
    path = cache / str(zoom) / str(tile_x) / f"{tile_y}.png"
    if path.exists():
        return tile_x, tile_y, Image.open(path).convert("RGB")
    url = (
        f"https://a.basemaps.cartocdn.com/light_nolabels/"
        f"{zoom}/{tile_x}/{tile_y}@2x.png"
    )
    response = session.get(url, timeout=45)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return tile_x, tile_y, Image.open(BytesIO(response.content)).convert("RGB")


def overpass_railways(
    session: requests.Session, south: float, west: float, north: float, east: float
) -> list[list[tuple[float, float]]] | None:
    types = "|".join(sorted(RAIL_TYPES))
    query = (
        f'[out:json][timeout:90];way["railway"~"{types}"]'
        f"({south},{west},{north},{east});out geom;"
    )
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = session.post(endpoint, data={"data": query}, timeout=120)
            response.raise_for_status()
            return [
                [(point["lon"], point["lat"]) for point in element["geometry"]]
                for element in response.json()["elements"]
                if len(element.get("geometry", [])) > 1
            ]
        except requests.RequestException:
            continue
    return None


def osm_api_railways(
    session: requests.Session,
    south: float,
    west: float,
    north: float,
    east: float,
    grid: int = 3,
) -> list[list[tuple[float, float]]]:
    """Fallback: split a dense area so the main OSM map API stays under its node limit."""
    ways: dict[str, list[tuple[float, float]]] = {}
    for row in range(grid):
        for column in range(grid):
            left = west + (east - west) * column / grid
            right = west + (east - west) * (column + 1) / grid
            bottom = south + (north - south) * row / grid
            top = south + (north - south) * (row + 1) / grid
            response = session.get(
                "https://api.openstreetmap.org/api/0.6/map",
                params={"bbox": f"{left},{bottom},{right},{top}"},
                timeout=120,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            nodes = {
                node.attrib["id"]: (float(node.attrib["lon"]), float(node.attrib["lat"]))
                for node in root.findall("node")
            }
            for way in root.findall("way"):
                tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
                if tags.get("railway") not in RAIL_TYPES:
                    continue
                geometry = [
                    nodes[item.attrib["ref"]]
                    for item in way.findall("nd")
                    if item.attrib["ref"] in nodes
                ]
                if len(geometry) > 1:
                    ways[way.attrib["id"]] = geometry
    return list(ways.values())


def make_city_map(
    city: str,
    rows: pd.DataFrame,
    output: Path,
    cache: Path,
    zoom: int,
) -> None:
    lon_padding = max(float(rows.lon.max() - rows.lon.min()) * 0.10, 0.0012)
    lat_padding = max(float(rows.lat.max() - rows.lat.min()) * 0.10, 0.0012)
    west = float(rows.lon.min() - lon_padding)
    east = float(rows.lon.max() + lon_padding)
    south = float(rows.lat.min() - lat_padding)
    north = float(rows.lat.max() + lat_padding)

    left, top = lonlat_to_pixel(west, north, zoom)
    right, bottom = lonlat_to_pixel(east, south, zoom)
    tile_x_min, tile_x_max = math.floor(left / TILE_SIZE), math.floor(right / TILE_SIZE)
    tile_y_min, tile_y_max = math.floor(top / TILE_SIZE), math.floor(bottom / TILE_SIZE)
    width = (tile_x_max - tile_x_min + 1) * TILE_SIZE * RETINA_SCALE
    height = (tile_y_max - tile_y_min + 1) * TILE_SIZE * RETINA_SCALE
    mosaic = Image.new("RGB", (width, height), "white")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    coordinates = [
        (x, y)
        for x in range(tile_x_min, tile_x_max + 1)
        for y in range(tile_y_min, tile_y_max + 1)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(download_tile, session, zoom, x, y, cache)
            for x, y in coordinates
        ]
        for future in as_completed(futures):
            tile_x, tile_y, tile = future.result()
            mosaic.paste(
                tile,
                (
                    (tile_x - tile_x_min) * TILE_SIZE * RETINA_SCALE,
                    (tile_y - tile_y_min) * TILE_SIZE * RETINA_SCALE,
                ),
            )

    origin_x = tile_x_min * TILE_SIZE
    origin_y = tile_y_min * TILE_SIZE
    crop_box = tuple(
        round(value)
        for value in (
            (left - origin_x) * RETINA_SCALE,
            (top - origin_y) * RETINA_SCALE,
            (right - origin_x) * RETINA_SCALE,
            (bottom - origin_y) * RETINA_SCALE,
        )
    )
    image = ImageOps.grayscale(mosaic.crop(crop_box)).convert("RGB")
    draw = ImageDraw.Draw(image)

    railways = overpass_railways(session, south, west, north, east)
    if railways is None:
        railways = osm_api_railways(session, south, west, north, east)
    for geometry in railways:
        line = [
            (
                (lonlat_to_pixel(lon, lat, zoom)[0] - left) * RETINA_SCALE,
                (lonlat_to_pixel(lon, lat, zoom)[1] - top) * RETINA_SCALE,
            )
            for lon, lat in geometry
        ]
        draw.line(line, fill=(15, 15, 15), width=7, joint="curve")
        draw.line(line, fill=(255, 255, 255), width=3, joint="curve")
        draw.line(line, fill=(15, 15, 15), width=1, joint="curve")

    for row in rows.itertuples():
        point_x, point_y = lonlat_to_pixel(row.lon, row.lat, zoom)
        x = (point_x - left) * RETINA_SCALE
        y = (point_y - top) * RETINA_SCALE
        radius = 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(220, 25, 32))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True, dpi=(240, 240))
    print(
        f"Created {output}: {image.width}x{image.height}, "
        f"{len(railways)} railway segments, {len(rows)} points"
    )
    time.sleep(0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/images_manifest.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/sample_location_maps")
    )
    parser.add_argument("--zoom", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.manifest)
    required = {"city", "lat", "lon"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing city/lat/lon values")

    cache = args.output_dir / ".carto_tile_cache"
    for city, rows in data.groupby("city", sort=True):
        make_city_map(
            city,
            rows,
            args.output_dir / f"{city.lower()}_sample_locations.png",
            cache,
            args.zoom,
        )


if __name__ == "__main__":
    main()
