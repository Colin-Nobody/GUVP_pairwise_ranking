#!/usr/bin/env python3
"""Make fixed 3x5 outer and 4-fold inner city splits for the V8 1,200-image study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 137, 2026)


def assign(groups: list[str], n_folds: int, seed: int) -> dict[str, int]:
    if len(groups) < n_folds:
        raise ValueError(f"Need {n_folds} city groups; got {len(groups)}")
    shuffled = np.random.default_rng(seed).permutation(sorted(groups)).tolist()
    return {city: fold for fold, chunk in enumerate(np.array_split(shuffled, n_folds)) for city in chunk}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "output/vit_vp_v8_1200_citycv")
    args = parser.parse_args()
    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv", dtype={"image_id": str})
    cities = sorted(manifest.city.astype(str).unique())
    counts = manifest.city.astype(str).value_counts().to_dict()
    if len(manifest) != 1200 or manifest.image_id.nunique() != 1200 or len(cities) != 24 or set(counts.values()) != {50}:
        raise ValueError("Expected 1,200 unique images, 24 cities, and 50 images per city")

    outer_rows, inner_rows, summary = [], [], {"seeds": list(SEEDS), "outer_folds": 5, "inner_folds": 4, "repeats": []}
    for repeat, seed in enumerate(SEEDS, start=1):
        outer_map = assign(cities, 5, seed)
        outer_record = {"repeat": repeat, "seed": seed, "folds": []}
        for fold in range(5):
            test_cities = {city for city, value in outer_map.items() if value == fold}
            train_cities = set(cities) - test_cities
            for city in cities:
                outer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": fold, "city": city,
                                   "partition": "test" if city in test_cities else "train"})
            inner_map = assign(sorted(train_cities), 4, seed * 100 + fold)
            inner_train_sizes = []
            for inner_fold in range(4):
                valid_cities = {city for city, value in inner_map.items() if value == inner_fold}
                fit_cities = train_cities - valid_cities
                inner_train_sizes.append(len(fit_cities))
                if fit_cities & valid_cities or valid_cities & test_cities or fit_cities & test_cities:
                    raise AssertionError("City leakage between nested partitions")
                for city in sorted(train_cities):
                    inner_rows.append({"repeat": repeat, "seed": seed, "outer_fold": fold,
                                       "inner_fold": inner_fold, "city": city,
                                       "partition": "validation" if city in valid_cities else "train"})
            outer_record["folds"].append({"outer_fold": fold, "train_cities": len(train_cities),
                                           "test_cities": len(test_cities), "train_images": 50 * len(train_cities),
                                           "test_images": 50 * len(test_cities),
                                           "inner_train_cities_per_fold": inner_train_sizes})
        summary["repeats"].append(outer_record)
    out = args.audit_dir / "splits"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outer_rows).to_csv(out / "unseen_city_outer.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(out / "unseen_city_inner.csv", index=False)
    summary["outer_city_partition_sha256"] = __import__("hashlib").sha256((out / "unseen_city_outer.csv").read_bytes()).hexdigest()
    summary["inner_city_partition_sha256"] = __import__("hashlib").sha256((out / "unseen_city_inner.csv").read_bytes()).hexdigest()
    (out / "split_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
