#!/usr/bin/env python3
"""Create reproducible city-held-out and within-city grouped CV manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "output/vit_vp_1000_v1"
SEEDS = (42, 137, 2026)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assign_folds(groups: list[str], n_splits: int, seed: int) -> dict[str, int]:
    if len(groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups, got {len(groups)}")
    shuffled = np.random.default_rng(seed).permutation(sorted(groups)).tolist()
    return {group: fold for fold, chunk in enumerate(np.array_split(shuffled, n_splits)) for group in chunk}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT / "splits")
    args = parser.parse_args()
    manifest = read_csv(args.audit_dir / "image_manifest.csv")
    reviews = read_csv(args.audit_dir / "near_duplicate_review.csv")
    if any(row["decision"] == "pending_review" for row in reviews):
        raise ValueError("Resolve all near-duplicate candidates before creating leakage-sensitive splits")
    if any(row["image_error"] for row in manifest):
        raise ValueError("Resolve image decoding errors before making splits")

    by_id = {row["image_id"]: row for row in manifest}
    cities = sorted({row["city"] for row in manifest})
    if len(cities) != 20 or any(sum(row["city"] == city for row in manifest) != 50 for city in cities):
        raise ValueError("Expected exactly 20 cities with 50 images each")
    duplicate_pairs = [row for row in reviews if row["decision"] == "confirmed_duplicate"]
    for pair in duplicate_pairs:
        if by_id[pair["image_id_a"]]["city"] != by_id[pair["image_id_b"]]["city"]:
            raise ValueError("Confirmed near-duplicate pair crosses cities; city split cannot isolate it")

    outer_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"seeds": list(SEEDS), "outer_fold_count": 5, "inner_fold_count": 4, "protocols": {}}
    for repeat, seed in enumerate(SEEDS, start=1):
        outer_assignment = assign_folds(cities, 5, seed)
        outer_counts = []
        image_outer_role: dict[tuple[int, str], str] = {}
        for outer_fold in range(5):
            test_cities = {city for city, fold in outer_assignment.items() if fold == outer_fold}
            train_cities = set(cities) - test_cities
            if len(test_cities) != 4 or len(train_cities) != 16:
                raise AssertionError("Unexpected outer city fold size")
            outer_counts.append({"train_images": 50 * len(train_cities), "test_images": 50 * len(test_cities)})
            for city in cities:
                role = "test" if city in test_cities else "train"
                outer_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "city": city, "partition": role})
                image_outer_role[(outer_fold, city)] = role
            inner_assignment = assign_folds(sorted(train_cities), 4, seed * 100 + outer_fold)
            for inner_fold in range(4):
                val_cities = {city for city, fold in inner_assignment.items() if fold == inner_fold}
                fit_cities = train_cities - val_cities
                if len(val_cities) != 4 or len(fit_cities) != 12:
                    raise AssertionError("Unexpected inner city fold size")
                for city in sorted(train_cities):
                    partition = "validation" if city in val_cities else "train"
                    inner_rows.append({"repeat": repeat, "seed": seed, "outer_fold": outer_fold, "inner_fold": inner_fold, "city": city, "partition": partition})

            for pair in duplicate_pairs:
                a_city = by_id[pair["image_id_a"]]["city"]
                b_city = by_id[pair["image_id_b"]]["city"]
                if image_outer_role[(outer_fold, a_city)] != image_outer_role[(outer_fold, b_city)]:
                    raise AssertionError("Near-duplicate pair crosses outer partition")
                for inner_fold in range(4):
                    local = [
                        row["partition"]
                        for row in inner_rows
                        if row["repeat"] == repeat and row["outer_fold"] == outer_fold and row["inner_fold"] == inner_fold and row["city"] in (a_city, b_city)
                    ]
                    if len(local) == 2 and local[0] != local[1]:
                        raise AssertionError("Near-duplicate pair crosses inner partition")

        heldout_counts = {city: 0 for city in cities}
        for row in outer_rows:
            if row["repeat"] == repeat and row["partition"] == "test":
                heldout_counts[str(row["city"])] += 1
        if set(heldout_counts.values()) != {1}:
            raise AssertionError("Each city must be held out once per repeat")
        summary["protocols"][f"unseen_city_repeat_{repeat}"] = {"seed": seed, "outer_folds": outer_counts, "heldout_city_counts": heldout_counts}

    if len(duplicate_pairs) != 3:
        raise ValueError(f"Expected the three visually confirmed duplicate pairs, found {len(duplicate_pairs)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "unseen_city_outer.csv", outer_rows, ["repeat", "seed", "outer_fold", "city", "partition"])
    write_csv(args.output_dir / "unseen_city_inner.csv", inner_rows, ["repeat", "seed", "outer_fold", "inner_fold", "city", "partition"])
    (args.output_dir / "split_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
