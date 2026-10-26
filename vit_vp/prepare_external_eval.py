#!/usr/bin/env python3
"""Sample a reproducible, city-disjoint 200-image external evaluation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"
DEFAULT_CITIES = ("Chisinau", "Stuttgart", "Bern", "Hildesheim")
SEED = 20261019
GRID_SIZE_DEGREES = 0.005
IMAGE_FILTERS = {
    "source": "KartaView",
    "platform": ["driving surface", "walking surface", "cycling surface"],
    "quality": "good",
    "lighting_condition": "day",
    "glare": False,
    "reflection": False,
    "lat_lon_required": True,
    "sequence_id_required": True,
    "one_image_per_sequence_per_city": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streetscapes", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/expansion_750/images_manifest_1000.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/external_vp_4city_200")
    parser.add_argument("--cities", nargs=4, default=list(DEFAULT_CITIES))
    parser.add_argument("--per-city", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid-size-degrees", type=float, default=GRID_SIZE_DEGREES)
    args = parser.parse_args()
    if args.per_city != 50:
        raise ValueError("This external protocol is fixed at 50 images per city")

    columns = [
        "uuid", "source", "orig_id", "city", "city_ascii", "country", "continent", "lat", "lon",
        "datetime_local", "sequence_id", "sequence_index", "platform", "quality", "lighting_condition",
        "glare", "reflection", "weather", "pano_status", "projection_type", "heading", "highway",
        "type_highway", "name", "lanes", "maxspeed", "road_width", "green_view_index", "sky_view_index",
    ]
    schema_cols = set(pq.ParquetFile(args.streetscapes).schema_arrow.names)
    missing = set(columns) - schema_cols
    if missing:
        raise ValueError(f"Global Streetscapes parquet missing columns: {sorted(missing)}")
    frame = pq.read_table(args.streetscapes, columns=columns).to_pandas()
    trained = pd.read_csv(args.train_manifest)
    seen_cities = {str(value).casefold() for value in trained.city.dropna().unique()}
    if any(city.casefold() in seen_cities for city in args.cities):
        raise ValueError("All external cities must be absent from the 1,000-image training manifest")
    trained_ids = set(trained.image_id.astype(str)) if "image_id" in trained else set(trained.uuid.astype(str))
    trained_source_ids = set(zip(trained.source.astype(str), trained.orig_id.astype(str)))

    mask = (
        frame.city.isin(args.cities)
        & frame.source.eq("KartaView")
        & frame.platform.isin(IMAGE_FILTERS["platform"])
        & frame.quality.eq("good")
        & frame.lighting_condition.eq("day")
        & frame.glare.eq(False)
        & frame.reflection.eq(False)
        & frame.lat.notna()
        & frame.lon.notna()
        & frame.sequence_id.notna()
    )
    candidates = frame.loc[mask].copy()
    candidates = candidates.loc[
        ~candidates.uuid.astype(str).isin(trained_ids)
        & ~candidates.apply(lambda row: (str(row.source), str(row.orig_id)) in trained_source_ids, axis=1)
    ]
    candidates = candidates.sample(frac=1, random_state=args.seed).drop_duplicates(["city", "source", "sequence_id"])
    counts = candidates.groupby("city", sort=False).size().reindex(args.cities, fill_value=0)
    shortage = counts[counts < args.per_city]
    if not shortage.empty:
        raise RuntimeError("Insufficient eligible candidates: " + ", ".join(f"{city}={n}" for city, n in shortage.items()))

    selected = []
    candidate_pools = []
    for index, city in enumerate(args.cities):
        part = candidates.loc[candidates.city.eq(city)].copy()
        part["grid_lat"] = (part.lat / args.grid_size_degrees).floordiv(1).astype("int64")
        part["grid_lon"] = (part.lon / args.grid_size_degrees).floordiv(1).astype("int64")
        part["grid_cell"] = part.grid_lat.astype(str) + "_" + part.grid_lon.astype(str)
        part = part.sample(frac=1, random_state=args.seed + index)
        part["grid_draw_order"] = part.groupby("grid_cell", sort=False).cumcount()
        part = part.sort_values(["grid_draw_order", "grid_cell"], kind="stable").copy()
        part["selection_rank_within_city"] = range(1, len(part) + 1)
        candidate_pools.append(part)
        part = part.head(args.per_city).copy()
        selected.append(part)

    result = pd.concat(selected, ignore_index=True)
    if result.uuid.astype(str).duplicated().any() or result.groupby("city").size().to_dict() != {city: args.per_city for city in args.cities}:
        raise AssertionError("Sample uniqueness or city quota validation failed")
    if result.groupby("city").sequence_id.nunique().min() != args.per_city:
        raise AssertionError("A sampled city contains repeated capture sequences")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["image_id"] = result.uuid.astype(str)
    result["random_seed"] = args.seed
    result["grid_size_degrees"] = args.grid_size_degrees
    result["filter_rules"] = json.dumps(IMAGE_FILTERS, sort_keys=True)
    manifest_path = args.output_dir / "images_manifest.csv"
    result.to_csv(manifest_path, index=False)
    pd.concat(candidate_pools, ignore_index=True).to_csv(args.output_dir / "candidate_pool.csv", index=False)
    counts.rename("eligible_sequences").to_csv(args.output_dir / "candidate_counts.csv", header=True)
    metadata = {
        "purpose": "External city evaluation; do not add these images to training or hyperparameter selection.",
        "cities": list(args.cities),
        "images_per_city": args.per_city,
        "total_images": len(result),
        "source": "KartaView only; source differs from the predominantly Mapillary training sample.",
        "sampling": "seeded random sample after one-image-per-sequence deduplication, ordered by grid-cell round-robin",
        "seed": args.seed,
        "grid_size_degrees": args.grid_size_degrees,
        "filters": IMAGE_FILTERS,
        "excluded_training_city_count": len(seen_cities),
        "eligible_unique_sequences_by_city": {str(k): int(v) for k, v in counts.items()},
        "sampled_images_by_city": {str(k): int(v) for k, v in result.groupby("city").size().items()},
        "source_metadata_file": str(args.streetscapes.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    (args.output_dir / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
