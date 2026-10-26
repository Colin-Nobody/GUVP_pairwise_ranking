#!/usr/bin/env python3
"""Build a connected, city-balanced 13-comparisons-per-image V6 graph."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/external_vp_4city_200"
SEED = 20261020


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "images_manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "pairs_200_13_per_image.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    id_col = "image_id" if "image_id" in manifest else "uuid"
    required = {id_col, "city"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest needs columns: {sorted(required)}")
    if len(manifest) != 200 or manifest[id_col].astype(str).nunique() != 200:
        raise ValueError("Expected exactly 200 unique images")
    groups = {str(city): list(group[id_col].astype(str)) for city, group in manifest.groupby("city", sort=True)}
    if len(groups) != 4 or any(len(items) != 50 for items in groups.values()):
        raise ValueError(f"Expected four cities with 50 images each; got { {k: len(v) for k,v in groups.items()} }")

    rng = np.random.default_rng(args.seed)
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    degree = Counter()
    adjacency = defaultdict(set)

    # Six within-city comparisons per image: a randomly relabelled circulant 6-regular graph.
    for city, image_ids in groups.items():
        permuted = list(np.asarray(image_ids, dtype=object)[rng.permutation(len(image_ids))])
        for offset in (1, 2, 3):
            for index, image_a in enumerate(permuted):
                image_b = permuted[(index + offset) % 50]
                key = tuple(sorted((str(image_a), str(image_b))))
                if key in pairs:
                    raise AssertionError("Duplicate within-city edge")
                pairs[key] = {"pair_type": "within_city", "city_a": city, "city_b": city}

    # Seven cross-city perfect matchings per image, rotating the three possible city pairings.
    cities = sorted(groups)
    matchings = [((cities[0], cities[1]), (cities[2], cities[3])),
                 ((cities[0], cities[2]), (cities[1], cities[3])),
                 ((cities[0], cities[3]), (cities[1], cities[2]))]
    for round_index in range(7):
        for city_a, city_b in matchings[round_index % 3]:
            left = groups[city_a]
            right = groups[city_b]
            for attempt in range(10000):
                shuffled_left = list(np.asarray(left, dtype=object)[rng.permutation(50)])
                shuffled_right = list(np.asarray(right, dtype=object)[rng.permutation(50)])
                proposed = [tuple(sorted((str(a), str(b)))) for a, b in zip(shuffled_left, shuffled_right)]
                if len(set(proposed)) == 50 and not any(key in pairs for key in proposed):
                    break
            else:
                raise RuntimeError(f"Unable to construct a unique matching between {city_a} and {city_b}")
            for image_a, image_b in zip(shuffled_left, shuffled_right):
                key = tuple(sorted((str(image_a), str(image_b))))
                pairs[key] = {"pair_type": "cross_city", "city_a": city_a, "city_b": city_b}

    rows = []
    keys = list(pairs)
    rng.shuffle(keys)
    for index, (image_a, image_b) in enumerate(keys, start=1):
        city_info = pairs[(image_a, image_b)]
        rows.append({"pair_id": f"extv6_{index:05d}", "image_a": f"{image_a}.jpg", "image_b": f"{image_b}.jpg", **city_info})
        degree[image_a] += 1
        degree[image_b] += 1
        adjacency[image_a].add(image_b)
        adjacency[image_b].add(image_a)
    if len(rows) != 1300 or set(degree.values()) != {13}:
        raise AssertionError(f"Pair count/degree mismatch: pairs={len(rows)}, degree_counts={Counter(degree.values())}")
    reached = set()
    queue = deque([next(iter(degree))])
    while queue:
        node = queue.popleft()
        if node not in reached:
            reached.add(node)
            queue.extend(adjacency[node] - reached)
    if len(reached) != 200:
        raise AssertionError(f"Comparison graph disconnected: reached {len(reached)} images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    info = {
        "n_images": 200,
        "n_cities": 4,
        "pairs": len(rows),
        "comparisons_per_image": 13,
        "within_city_pairs": sum(row["pair_type"] == "within_city" for row in rows),
        "cross_city_pairs": sum(row["pair_type"] == "cross_city" for row in rows),
        "graph_connected": True,
        "city_image_counts": {city: len(items) for city, items in groups.items()},
        "seed": args.seed,
        "pair_manifest": str(args.output.resolve()),
        "design": "6-regular within-city graph plus seven cross-city perfect matchings per image; shuffled pair order",
    }
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
