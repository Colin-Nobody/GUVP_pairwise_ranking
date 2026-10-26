#!/usr/bin/env python3
"""Freeze a city-disjoint 900/300 split for V9 ViT fine-tuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output/vit_vp_v9_holdout_900_300")
    args = ap.parse_args()

    manifest_path = ROOT / "data/vlm_v9_1200/images_manifest_1200.csv"
    score_path = ROOT / "output/vlm_v9_1200_gpuonly_optimized/perception_scores.csv"
    pair_audit_path = ROOT / "output/vlm_v9_1200_gpuonly_optimized/pair_audit.csv"
    cache_index_path = ROOT / "output/vit_vp_1000_v1/features/prefix_after_block10/edge512/index.csv"
    checkpoint_path = ROOT / "models/checkpoints/dinov3_vitb.safetensors"

    manifest = read_csv(manifest_path)
    score_rows = read_csv(score_path)
    score_by_id = {Path(row["image_id"]).stem: row for row in score_rows}
    if len(manifest) != 1200 or len({r["uuid"] for r in manifest}) != 1200:
        raise ValueError("Expected 1,200 unique V9 image-manifest rows")
    if len(score_by_id) != 1200 or set(score_by_id) != {r["uuid"] for r in manifest}:
        raise ValueError("V9 target IDs must match all 1,200 manifest rows")

    original_cities = sorted({r.get("city_ascii") or r.get("city") or "" for r in manifest
                              if r["cohort"] == "original_1000"})
    if len(original_cities) != 20 or "" in original_cities:
        raise ValueError(f"Expected 20 named original cities; got {original_cities}")
    held_original_cities = sorted(random.Random(20261020).sample(original_cities, 2))
    external_cities = sorted({r.get("city_ascii") or r.get("city") or "" for r in manifest
                              if r["cohort"] == "external_4city_200"})
    test_cities = set(held_original_cities + external_cities)

    train_rows, test_rows = [], []
    image_root = ROOT / "data/vlm_v9_1200/images"
    for r in manifest:
        image_id = r["uuid"]
        city = r.get("city_ascii") or r.get("city") or ""
        score = score_by_id[image_id]
        path = image_root / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        item = {
            "image_id": image_id,
            "image_path": str(path.resolve()),
            "city": city,
            "country": r.get("country", ""),
            "source": r.get("source", ""),
            "cohort": r["cohort"],
            "target_visual_pollution_score_0_10": score["visual_pollution_score_0_10"],
            "trueskill_mu": score["trueskill_mu"],
            "trueskill_sigma": score["trueskill_sigma"],
        }
        if city in test_cities:
            if r["cohort"] == "original_1000" and city in held_original_cities:
                item["split_group"] = "heldout_original_city"
            else:
                item["split_group"] = "external_4city"
            test_rows.append(item)
        elif r["cohort"] == "original_1000":
            item["split_group"] = "development_city"
            train_rows.append(item)
        else:
            raise ValueError(f"External city unexpectedly assigned to training: {city}")

    if len(train_rows) != 900 or len(test_rows) != 300:
        raise ValueError(f"Unexpected split sizes train={len(train_rows)}, test={len(test_rows)}")
    train_ids, test_ids = {r["image_id"] for r in train_rows}, {r["image_id"] for r in test_rows}
    train_city_set, test_city_set = {r["city"] for r in train_rows}, {r["city"] for r in test_rows}
    if train_ids & test_ids or train_city_set & test_city_set:
        raise AssertionError("Image or city overlap detected")

    cache_rows = read_csv(cache_index_path)
    cache_ids = {r["image_id"] for r in cache_rows}
    if not train_ids <= cache_ids:
        raise ValueError(f"Pretrained prefix cache is missing {len(train_ids-cache_ids)} training IDs")

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    write_csv(out / "training/image_manifest.csv", train_rows)
    write_csv(out / "test/test_manifest.csv", test_rows)

    pair_audit = read_csv(pair_audit_path)
    cross_edges = Counter()
    test_graph = defaultdict(set)
    for row in pair_audit:
        a, b = Path(row["image_a"]).stem, Path(row["image_b"]).stem
        if (a in train_ids) != (b in train_ids):
            cross_edges["all_audit_edges"] += 1
            if row.get("status") == "calibrated":
                cross_edges["calibrated_edges"] += 1
        if row.get("status") == "calibrated" and a in test_ids and b in test_ids:
            test_graph[a].add(b)
            test_graph[b].add(a)
    unseen = test_ids - set(test_graph)
    components = 0
    visited = set()
    for node in test_ids - unseen:
        if node in visited:
            continue
        components += 1
        stack = [node]
        visited.add(node)
        while stack:
            current = stack.pop()
            for neighbor in test_graph[current] - visited:
                visited.add(neighbor)
                stack.append(neighbor)

    audit = {
        "created_date": "2026-10-20",
        "protocol": "city-disjoint 900 train / 300 test; fixed epoch count, no test tuning or early stopping",
        "training_images": len(train_rows),
        "test_images": len(test_rows),
        "training_cities": sorted(train_city_set),
        "heldout_original_cities_seed_20261020": held_original_cities,
        "external_test_cities": external_cities,
        "test_cities": sorted(test_city_set),
        "training_ids_match_pretrained_prefix_cache": train_ids <= cache_ids,
        "image_ids_disjoint": True,
        "city_groups_disjoint": True,
        "test_subgroups": dict(Counter(r["split_group"] for r in test_rows)),
        "test_city_counts": dict(Counter(r["city"] for r in test_rows)),
        "cross_partition_v9_comparisons": dict(cross_edges),
        "test_internal_valid_edge_components": components,
        "test_images_without_internal_valid_edge": len(unseen),
        "v9_label_note": "All 1,200 targets were jointly estimated on the V9 pairwise comparison graph and globally min-max scaled to 0-10.",
        "interpretation_note": "ViT training and model selection exclude all 300 test images and all six test cities. Existing V9 labels remain jointly graph-derived, including cross-partition comparisons.",
        "input_sha256": {
            "manifest_1200": sha256(manifest_path),
            "v9_perception_scores": sha256(score_path),
            "v9_pair_audit": sha256(pair_audit_path),
            "pretrained_dinov3_checkpoint": sha256(checkpoint_path),
        },
    }
    (out / "partition_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
