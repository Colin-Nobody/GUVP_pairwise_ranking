#!/usr/bin/env python3
"""Freeze the V9 first-1000 development set and last-200 image holdout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/vit_vp_v9_holdout_1000_200")
    args = parser.parse_args()

    manifest_path = ROOT / "data/vlm_v9_1200/images_manifest_1200.csv"
    scores_path = ROOT / "output/vlm_v9_1200_gpuonly_optimized/perception_scores.csv"
    votes_path = ROOT / "output/vlm_v9_1200_gpuonly_optimized/vlm_pairwise_votes.csv"
    audit_path = ROOT / "output/vlm_v9_1200_gpuonly_optimized/pair_audit.csv"
    cache_index = ROOT / "output/vit_vp_1000_v1/features/prefix_after_block10/edge512/index.csv"
    base_checkpoint = ROOT / "models/checkpoints/dinov3_vitb.safetensors"

    manifest = read_csv(manifest_path)
    score_rows = read_csv(scores_path)
    score_by_id = {Path(row["image_id"]).stem: row for row in score_rows}
    if len(manifest) != 1200 or len({row["uuid"] for row in manifest}) != 1200:
        raise ValueError("Expected 1,200 unique manifest rows")
    if len(score_by_id) != 1200 or set(score_by_id) != {row["uuid"] for row in manifest}:
        raise ValueError("V9 score IDs do not exactly match the 1,200-image manifest")
    if [row["cohort"] for row in manifest[:1000]] != ["original_1000"] * 1000:
        raise ValueError("The first 1,000 manifest rows are not the original_1000 cohort")
    if [row["cohort"] for row in manifest[1000:]] != ["external_4city_200"] * 200:
        raise ValueError("The final 200 manifest rows are not the external_4city_200 cohort")

    train_rows, holdout_rows = [], []
    image_root = ROOT / "data/vlm_v9_1200/images"
    for row in manifest:
        image_id = row["uuid"]
        image_path = image_root / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        score = score_by_id[image_id]
        item = {
            "image_id": image_id,
            "image_path": str(image_path.resolve()),
            "city": row.get("city_ascii") or row.get("city") or "",
            "country": row.get("country", ""),
            "source": row.get("source", ""),
            "cohort": row["cohort"],
            "target_visual_pollution_score_0_10": score["visual_pollution_score_0_10"],
            "trueskill_mu": score["trueskill_mu"],
            "trueskill_sigma": score["trueskill_sigma"],
        }
        (train_rows if row["cohort"] == "original_1000" else holdout_rows).append(item)
    if len(train_rows) != 1000 or len(holdout_rows) != 200:
        raise AssertionError("Unexpected partition sizes")

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    write_csv(out / "training/image_manifest.csv", train_rows)
    write_csv(out / "holdout/holdout_manifest.csv", holdout_rows)
    split_source = ROOT / "output/vit_vp_1000_v1/splits"
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits/unseen_city_outer.csv").write_bytes((split_source / "unseen_city_outer.csv").read_bytes())
    (out / "splits/split_audit.json").write_bytes((split_source / "split_audit.json").read_bytes())

    train_ids = {row["image_id"] for row in train_rows}
    holdout_ids = {row["image_id"] for row in holdout_rows}
    cache_ids = {row["image_id"] for row in read_csv(cache_index)}
    if train_ids & holdout_ids or train_ids != cache_ids:
        raise ValueError("Partition overlap or training prefix-cache ID mismatch")

    audit_rows = read_csv(audit_path)
    votes = read_csv(votes_path)
    cross_edges = Counter()
    for row in audit_rows:
        left, right = Path(row["image_a"]).stem, Path(row["image_b"]).stem
        if (left in train_ids) != (right in train_ids):
            cross_edges["all_audit_edges"] += 1
            if row["status"] == "calibrated":
                cross_edges["calibrated_edges"] += 1
    holdout_internal = [
        row for row in votes
        if Path(row["image_a"]).stem in holdout_ids and Path(row["image_b"]).stem in holdout_ids
    ]
    audit = {
        "created_date": "2026-10-20",
        "training_images": len(train_rows),
        "holdout_images": len(holdout_rows),
        "training_cities": len({row["city"] for row in train_rows}),
        "holdout_cities": dict(Counter(row["city"] for row in holdout_rows)),
        "ids_disjoint": True,
        "training_image_ids_match_existing_pretrained_prefix_cache": True,
        "label": "V9 optimized prompt Qwen/TrueSkill visual_pollution_score_0_10",
        "label_direction": "higher_is_more_visual_pollution",
        "label_scale_note": "All 1,200 targets were min-max normalized from one joint V9 comparison graph.",
        "cross_partition_v9_comparisons": dict(cross_edges),
        "holdout_internal_valid_votes": len(holdout_internal),
        "strict_label_isolation_note": (
            "The holdout images and labels are excluded from ViT fitting, model selection, and early stopping. "
            "However, the precomputed V9 TrueSkill targets were jointly estimated from a graph containing "
            "cross-partition comparisons; these fixed labels cannot be made independent without new ratings."
        ),
        "input_sha256": {
            "manifest_1200": sha256(manifest_path),
            "v9_perception_scores": sha256(scores_path),
            "v9_calibrated_votes": sha256(votes_path),
            "v9_pair_audit": sha256(audit_path),
            "pretrained_dinov3_checkpoint": sha256(base_checkpoint),
        },
    }
    (out / "partition_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
