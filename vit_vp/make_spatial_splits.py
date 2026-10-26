#!/usr/bin/env python3
"""Create repeated 250 m within-city grouped splits and 100 m buffer sensitivity splits."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"
SEEDS = (42, 137, 2026)
GRID_M = 250.0
BUFFER_M = 100.0
EARTH_M_PER_DEG_LAT = 110_574.0
EARTH_M_PER_DEG_LON = 111_320.0


def balanced_group_assignment(group_to_ids: dict[str, list[str]], k: int, seed: int) -> dict[str, int]:
    if len(group_to_ids) < k:
        raise ValueError(f"Need at least {k} spatial groups; found {len(group_to_ids)}")
    rng = np.random.default_rng(seed)
    groups = list(group_to_ids)
    rng.shuffle(groups)
    # Largest groups first; ties retain the seeded random order. Greedy load balancing
    # keeps the per-fold image count close while never splitting a 250 m cell.
    groups.sort(key=lambda group: len(group_to_ids[group]), reverse=True)
    loads = [0] * k
    assignment: dict[str, int] = {}
    for group in groups:
        smallest = min(loads)
        choices = [i for i, load in enumerate(loads) if load == smallest]
        fold = int(rng.choice(choices))
        assignment[group] = fold
        loads[fold] += len(group_to_ids[group])
    return assignment


def min_distances(coords: np.ndarray, query_idx: np.ndarray, reference_idx: np.ndarray) -> np.ndarray:
    if not len(query_idx) or not len(reference_idx):
        return np.full(len(query_idx), np.inf)
    out = np.empty(len(query_idx), dtype=np.float64)
    # Small city folds fit easily in memory; chunk for future larger manifests.
    for start in range(0, len(query_idx), 256):
        q = coords[query_idx[start:start + 256]]
        d2 = ((q[:, None, :] - coords[reference_idx][None, :, :]) ** 2).sum(axis=2)
        out[start:start + len(q)] = np.sqrt(d2.min(axis=1))
    return out


def write_split(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "splits")
    args = parser.parse_args()
    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    if manifest.image_id.duplicated().any() or manifest.lat.isna().any() or manifest.lon.isna().any():
        raise ValueError("Expected unique image IDs and complete coordinates")
    reviews = pd.read_csv(args.audit_dir / "near_duplicate_review.csv")
    if (reviews.decision == "pending_review").any():
        raise ValueError("Resolve near-duplicate review before creating spatial splits")
    dup_pairs = reviews.loc[reviews.decision == "confirmed_duplicate", ["image_id_a", "image_id_b"]].itertuples(index=False, name=None)
    dup_pairs = list(dup_pairs)

    coords = np.zeros((len(manifest), 2), dtype=np.float64)
    spatial_groups = np.empty(len(manifest), dtype=object)
    city_indices: dict[str, np.ndarray] = {}
    for city, frame in manifest.groupby("city", sort=True):
        idx = frame.index.to_numpy()
        lat0 = float(frame.lat.median())
        lon0 = float(frame.lon.median())
        cos_lat = math.cos(math.radians(lat0))
        coords[idx, 0] = (frame.lon.to_numpy(dtype=float) - lon0) * EARTH_M_PER_DEG_LON * cos_lat
        coords[idx, 1] = (frame.lat.to_numpy(dtype=float) - lat0) * EARTH_M_PER_DEG_LAT
        cells = [(math.floor(x / GRID_M), math.floor(y / GRID_M)) for x, y in coords[idx]]
        spatial_groups[idx] = [f"{city}|{cx}:{cy}" for cx, cy in cells]
        city_indices[str(city)] = idx

    image_row = {str(image_id): i for i, image_id in enumerate(manifest.image_id.astype(str))}
    for a, b in dup_pairs:
        ia, ib = image_row[str(a)], image_row[str(b)]
        if spatial_groups[ia] != spatial_groups[ib]:
            raise ValueError(f"Confirmed near duplicates do not share a 250 m group: {a}, {b}")

    outer_rows: list[dict[str, object]] = []
    outer_buffer_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    inner_buffer_rows: list[dict[str, object]] = []
    audit: dict[str, object] = {
        "grid_size_m": GRID_M,
        "buffer_sensitivity_m": BUFFER_M,
        "projection": "city-centered equirectangular approximation (x=delta_lon*111320*cos(median_lat), y=delta_lat*110574)",
        "seeds": list(SEEDS), "outer_folds": 5, "inner_folds": 4,
        "city_group_counts": {}, "outer_fold_summary": [], "inner_fold_summary": [],
    }
    for city, idx in city_indices.items():
        cells: dict[str, list[str]] = defaultdict(list)
        for i in idx:
            cells[str(spatial_groups[i])].append(str(manifest.iloc[i].image_id))
        audit["city_group_counts"][city] = {"images": int(len(idx)), "groups_250m": int(len(cells)), "max_group_size": int(max(map(len, cells.values())))}

    for repeat, seed in enumerate(SEEDS, start=1):
        outer_assignments: dict[str, int] = {}
        for city, idx in city_indices.items():
            cells: dict[str, list[str]] = defaultdict(list)
            for i in idx:
                cells[str(spatial_groups[i])].append(str(manifest.iloc[i].image_id))
            outer_assignments.update(balanced_group_assignment(cells, 5, seed * 1000 + sum(map(ord, city))))

        for outer_fold in range(5):
            test_idx = np.array([i for i in range(len(manifest)) if outer_assignments[str(spatial_groups[i])] == outer_fold], dtype=int)
            train_idx = np.array([i for i in range(len(manifest)) if outer_assignments[str(spatial_groups[i])] != outer_fold], dtype=int)
            excluded_train: set[int] = set()
            for city, city_idx in city_indices.items():
                city_test = np.intersect1d(test_idx, city_idx, assume_unique=False)
                city_train = np.intersect1d(train_idx, city_idx, assume_unique=False)
                dist_test = min_distances(coords, city_test, city_train)
                dist_train = min_distances(coords, city_train, city_test)
                for i, d in zip(city_test, dist_test):
                    outer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "city": city,
                                       "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                       "partition": "test", "nearest_opposite_m": float(d)})
                for i, d in zip(city_train, dist_train):
                    partition = "buffer_excluded" if d < BUFFER_M else "train"
                    if partition == "buffer_excluded":
                        excluded_train.add(int(i))
                    outer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "city": city,
                                       "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                       "partition": "train", "nearest_opposite_m": float(d)})
                    outer_buffer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "city": city,
                                              "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                              "partition": partition, "nearest_opposite_m": float(d)})
            # Keep held-out rows identical between primary and buffered protocols.
            for i in test_idx:
                city = str(manifest.iloc[i].city)
                city_train = np.intersect1d(train_idx, city_indices[city], assume_unique=False)
                d = float(min_distances(coords, np.array([i]), city_train)[0])
                outer_buffer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "city": city,
                                          "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                          "partition": "test", "nearest_opposite_m": d})
            if len({str(spatial_groups[i]) for i in train_idx} & {str(spatial_groups[i]) for i in test_idx}):
                raise AssertionError("Outer 250 m group leakage")
            if any(outer_assignments[str(spatial_groups[image_row[a]])] != outer_assignments[str(spatial_groups[image_row[b]])] for a, b in dup_pairs):
                raise AssertionError("Confirmed duplicate pair crosses spatial folds")
            outer_buffer_train = [r for r in outer_buffer_rows if r["repeat"] == repeat and r["outer_fold"] == outer_fold and r["partition"] == "train"]
            if outer_buffer_train and min(float(r["nearest_opposite_m"]) for r in outer_buffer_train) < BUFFER_M:
                raise AssertionError("Outer buffer split contains a training image inside the 100 m buffer")
            min_test_to_train_m = min(float(r["nearest_opposite_m"]) for r in outer_rows
                                      if r["repeat"] == repeat and r["outer_fold"] == outer_fold and r["partition"] == "test")
            audit["outer_fold_summary"].append({"repeat": repeat, "outer_fold": outer_fold,
                "train_images": int(len(train_idx)), "test_images": int(len(test_idx)),
                "buffer_train_images": int(len(train_idx) - len(excluded_train)), "buffer_excluded_images": int(len(excluded_train)),
                "min_test_to_train_m": min_test_to_train_m})

            inner_assignments: dict[str, int] = {}
            for city, city_idx in city_indices.items():
                outer_train_city = [i for i in city_idx if outer_assignments[str(spatial_groups[i])] != outer_fold]
                cells: dict[str, list[str]] = defaultdict(list)
                for i in outer_train_city:
                    cells[str(spatial_groups[i])].append(str(manifest.iloc[i].image_id))
                inner_assignments.update(balanced_group_assignment(cells, 4, seed * 100000 + outer_fold * 1000 + sum(map(ord, city))))
            for inner_fold in range(4):
                inner_train_idx = np.array([i for i in train_idx if inner_assignments[str(spatial_groups[i])] != inner_fold], dtype=int)
                val_idx = np.array([i for i in train_idx if inner_assignments[str(spatial_groups[i])] == inner_fold], dtype=int)
                excluded_inner: set[int] = set()
                for city, city_idx in city_indices.items():
                    city_val = np.intersect1d(val_idx, city_idx, assume_unique=False)
                    city_train = np.intersect1d(inner_train_idx, city_idx, assume_unique=False)
                    dist_val = min_distances(coords, city_val, city_train)
                    dist_train = min_distances(coords, city_train, city_val)
                    for i, d in zip(city_val, dist_val):
                        inner_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "inner_fold": inner_fold, "city": city,
                                           "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                           "partition": "validation", "nearest_opposite_m": float(d)})
                    for i, d in zip(city_train, dist_train):
                        partition = "buffer_excluded" if d < BUFFER_M else "train"
                        if partition == "buffer_excluded":
                            excluded_inner.add(int(i))
                        inner_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "inner_fold": inner_fold, "city": city,
                                           "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                           "partition": "train", "nearest_opposite_m": float(d)})
                        inner_buffer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "inner_fold": inner_fold, "city": city,
                                                  "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                                  "partition": partition, "nearest_opposite_m": float(d)})
                for i in val_idx:
                    city = str(manifest.iloc[i].city)
                    city_train = np.intersect1d(inner_train_idx, city_indices[city], assume_unique=False)
                    d = float(min_distances(coords, np.array([i]), city_train)[0])
                    inner_buffer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "inner_fold": inner_fold, "city": city,
                                              "image_id": str(manifest.iloc[i].image_id), "spatial_group": str(spatial_groups[i]),
                                              "partition": "validation", "nearest_opposite_m": d})
                if len({str(spatial_groups[i]) for i in inner_train_idx} & {str(spatial_groups[i]) for i in val_idx}):
                    raise AssertionError("Inner 250 m group leakage")
                inner_buffer_train = [r for r in inner_buffer_rows if r["repeat"] == repeat and r["outer_fold"] == outer_fold and r["inner_fold"] == inner_fold and r["partition"] == "train"]
                if inner_buffer_train and min(float(r["nearest_opposite_m"]) for r in inner_buffer_train) < BUFFER_M:
                    raise AssertionError("Inner buffer split contains a training image inside the 100 m buffer")
                audit["inner_fold_summary"].append({"repeat": repeat, "outer_fold": outer_fold, "inner_fold": inner_fold,
                    "train_images": int(len(inner_train_idx)), "validation_images": int(len(val_idx)),
                    "buffer_train_images": int(len(inner_train_idx) - len(excluded_inner)), "buffer_excluded_images": int(len(excluded_inner))})

    columns_outer = ["repeat", "seed", "outer_fold", "city", "image_id", "spatial_group", "partition", "nearest_opposite_m"]
    columns_inner = ["repeat", "seed", "outer_fold", "inner_fold", "city", "image_id", "spatial_group", "partition", "nearest_opposite_m"]
    write_split(outer_rows, args.output_dir / "within_city_spatial_250m_outer.csv")
    write_split(outer_buffer_rows, args.output_dir / "within_city_spatial_250m_buffer100_outer.csv")
    write_split(inner_rows, args.output_dir / "within_city_spatial_250m_inner.csv")
    write_split(inner_buffer_rows, args.output_dir / "within_city_spatial_250m_buffer100_inner.csv")
    (args.output_dir / "within_city_spatial_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"city_group_counts": audit["city_group_counts"],
                      "outer_fold_count": len(audit["outer_fold_summary"]),
                      "outer_train_images_range": [min(r["train_images"] for r in audit["outer_fold_summary"]), max(r["train_images"] for r in audit["outer_fold_summary"])],
                      "outer_buffer_excluded_range": [min(r["buffer_excluded_images"] for r in audit["outer_fold_summary"]), max(r["buffer_excluded_images"] for r in audit["outer_fold_summary"])],
                      "inner_buffer_excluded_range": [min(r["buffer_excluded_images"] for r in audit["inner_fold_summary"]), max(r["buffer_excluded_images"] for r in audit["inner_fold_summary"])],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
