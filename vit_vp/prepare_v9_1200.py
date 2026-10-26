#!/usr/bin/env python3
"""Stage 1,200 existing images and make a connected 13-comparison graph for V9."""
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261020


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/vlm_v9_1200")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing V9 staging directory: {out}")

    core_path = ROOT / "data/expansion_750/images_manifest_1000.csv"
    ext_path = ROOT / "data/external_vp_4city_200/images_manifest.csv"
    core = pd.read_csv(core_path, dtype={"uuid": str})
    ext = pd.read_csv(ext_path, dtype={"uuid": str})
    if len(core) != 1000 or len(ext) != 200:
        raise ValueError(f"Expected 1,000 + 200 manifest rows; got {len(core)} + {len(ext)}")
    if core.uuid.duplicated().any() or ext.uuid.duplicated().any():
        raise ValueError("Duplicate UUID within an input manifest")
    if set(core.uuid) & set(ext.uuid):
        raise ValueError("Core and external image UUIDs overlap")

    core["cohort"] = "original_1000"
    ext["cohort"] = "external_4city_200"
    manifest = pd.concat([core, ext], ignore_index=True, sort=False)
    manifest["image_id"] = manifest.uuid.astype(str)
    if len(manifest) != 1200 or manifest.uuid.nunique() != 1200:
        raise ValueError("Combined manifest does not contain 1,200 unique images")

    image_dir = out / "images"
    image_dir.mkdir(parents=True)
    sources: dict[str, Path] = {}
    for row in manifest.itertuples(index=False):
        image_id = str(row.uuid)
        if row.cohort == "original_1000":
            candidates = [ROOT / "data/expansion_750/street_images_1000" / f"{image_id}.jpg",
                          ROOT / "data/expansion_750/street_images" / f"{image_id}.jpg"]
        else:
            candidates = [ROOT / "data/external_vp_4city_200/images" / f"{image_id}.jpg"]
        source = next((path.resolve() for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Could not resolve image {image_id} in {candidates}")
        try:
            with Image.open(source) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Image decode/verify failed for {image_id}: {exc}") from exc
        (image_dir / f"{image_id}.jpg").symlink_to(source)
        sources[image_id] = source

    ids = sorted(sources)
    rng = np.random.default_rng(args.seed)
    order = list(np.asarray(ids, dtype=object)[rng.permutation(len(ids))])
    cities = manifest.set_index("uuid").city.astype(str).to_dict()
    edges: set[tuple[str, str]] = set()
    for index in range(len(order)):
        for offset in range(1, 7):
            edges.add(tuple(sorted((str(order[index]), str(order[(index + offset) % len(order)])))))
        edges.add(tuple(sorted((str(order[index]), str(order[(index + 600) % len(order)])))))
    if len(edges) != 7800:
        raise AssertionError(f"Expected 7,800 unique pairs, found {len(edges)}")

    adjacency = {image_id: set() for image_id in ids}
    degree = Counter()
    rows = []
    shuffled_edges = list(edges)
    rng.shuffle(shuffled_edges)
    for index, (image_a, image_b) in enumerate(shuffled_edges, start=1):
        adjacency[image_a].add(image_b)
        adjacency[image_b].add(image_a)
        degree[image_a] += 1
        degree[image_b] += 1
        rows.append({
            "pair_id": f"v9_1200_{index:05d}",
            "image_a": f"{image_a}.jpg",
            "image_b": f"{image_b}.jpg",
            "pair_type": "within_city" if cities[image_a] == cities[image_b] else "cross_city",
            "city_a": cities[image_a],
            "city_b": cities[image_b],
        })
    if set(degree.values()) != {13}:
        raise AssertionError(f"Expected degree 13 for every image; got {Counter(degree.values())}")

    reached: set[str] = set()
    queue = deque([ids[0]])
    while queue:
        node = queue.popleft()
        if node not in reached:
            reached.add(node)
            queue.extend(adjacency[node] - reached)
    if len(reached) != 1200:
        raise AssertionError(f"Comparison graph is disconnected: reached {len(reached)}/1,200")

    manifest.to_csv(out / "images_manifest_1200.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "pairs_1200_13_per_image.csv", index=False)
    metadata = {
        "n_images": 1200,
        "cohorts": {str(k): int(v) for k, v in manifest.groupby("cohort").size().items()},
        "cities": {str(k): int(v) for k, v in manifest.groupby("city").size().items()},
        "source_counts": {str(k): int(v) for k, v in manifest.groupby("source").size().items()},
        "pairs": len(rows),
        "comparisons_per_image": 13,
        "within_city_pairs": sum(row["pair_type"] == "within_city" for row in rows),
        "cross_city_pairs": sum(row["pair_type"] == "cross_city" for row in rows),
        "graph_connected": True,
        "seed": args.seed,
        "design": "Seeded random relabeling of a 13-regular connected circulant graph: offsets 1..6 and antipodal matching at N/2.",
        "inputs": [str(core_path), str(ext_path)],
    }
    (out / "pair_design.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
