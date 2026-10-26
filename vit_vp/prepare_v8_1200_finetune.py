#!/usr/bin/env python3
"""Freeze and audit V8 labels and images for city-held-out ViT fine-tuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/vlm_v9_1200/images_manifest_1200.csv")
    parser.add_argument("--scores", type=Path, default=ROOT / "output/vlm_v8_1200_gpuonly/perception_scores.csv")
    parser.add_argument("--pairs", type=Path, default=ROOT / "output/vlm_v8_1200_gpuonly/pair_audit.csv")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/vlm_v9_1200/images")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/checkpoints/dinov3_vitb.safetensors")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/vit_vp_v8_1200_citycv")
    args = parser.parse_args()

    manifest, scores, pairs = map(read_csv, (args.manifest, args.scores, args.pairs))
    score_by_id = {Path(row["image_id"]).stem: row for row in scores}
    manifest_ids = [row["uuid"] for row in manifest]
    if len(manifest_ids) != 1200 or len(set(manifest_ids)) != 1200:
        raise ValueError(f"Expected 1,200 unique image IDs, got {len(manifest_ids)} rows")
    if len(score_by_id) != 1200 or set(score_by_id) != set(manifest_ids):
        raise ValueError("V8 score IDs do not match the fixed 1,200-image manifest")
    if len(pairs) != 7800:
        raise ValueError(f"Expected 7,800 V8 pair-audit rows, got {len(pairs)}")

    rows, image_hashes = [], Counter()
    cities = Counter()
    for item in manifest:
        image_id = item["uuid"]
        city = item.get("city_ascii") or item.get("city") or ""
        if not city:
            raise ValueError(f"Missing city for {image_id}")
        source_path = args.image_dir / f"{image_id}.jpg"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as image:
            image.verify()
        with Image.open(source_path) as image:
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            image_format = image.format or ""
        digest = sha256(source_path)
        image_hashes[digest] += 1
        target = float(score_by_id[image_id]["visual_pollution_score_0_10"])
        if not np.isfinite(target) or not 0 <= target <= 10:
            raise ValueError(f"Invalid V8 target for {image_id}: {target}")
        rows.append({
            "image_id": image_id,
            "image_path": str(source_path.resolve()),
            "city": city,
            "country": item.get("country", ""),
            "source": item.get("source", ""),
            "cohort": item.get("cohort", ""),
            "sequence_id": item.get("sequence_id", ""),
            "pano_status": item.get("pano_status", ""),
            "width": width,
            "height": height,
            "image_format": image_format,
            "sha256": digest,
            "target_visual_pollution_score_0_10": target,
            "trueskill_mu": float(score_by_id[image_id]["trueskill_mu"]),
            "trueskill_sigma": float(score_by_id[image_id]["trueskill_sigma"]),
        })
        cities[city] += 1
    if len(cities) != 24 or set(cities.values()) != {50}:
        raise ValueError(f"Expected 24 cities with 50 images each, got {dict(cities)}")
    if len(image_hashes) != 1200:
        raise ValueError(f"Found duplicate image content: {1200 - len(image_hashes)} extra copies")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_out = out / "image_manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pair_status = Counter(row.get("status", "") for row in pairs)
    audit = {
        "dataset": "V8 Qwen pairwise / TrueSkill labels for the fixed 1,200-image comparison",
        "images": len(rows), "unique_image_ids": len(manifest_ids),
        "unique_content_hashes": len(image_hashes), "cities": dict(sorted(cities.items())),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "pair_audit_rows": len(pairs), "pair_audit_status": dict(sorted(pair_status.items())),
        "target": "visual_pollution_score_0_10; larger means more visual pollution",
        "target_range": [min(row["target_visual_pollution_score_0_10"] for row in rows),
                         max(row["target_visual_pollution_score_0_10"] for row in rows)],
        "input_sha256": {"image_manifest": sha256(args.manifest), "v8_perception_scores": sha256(args.scores),
                         "v8_pair_audit": sha256(args.pairs), "original_dinov3_checkpoint": sha256(args.checkpoint)},
        "label_caveat": "V8 labels are VLM/TrueSkill-derived from a shared 1,200-image comparison graph; not human ground truth.",
    }
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {"prompt_version": "vp-definition-score-v8", "model": "Qwen2.5-VL-7B-Instruct",
                  "aggregation": "pairwise calibrated votes aggregated by TrueSkill and min-max scaled to [0,10]",
                  "scores_path": str(args.scores.resolve()), "scores_sha256": audit["input_sha256"]["v8_perception_scores"],
                  "pair_audit_path": str(args.pairs.resolve()), "pair_audit_sha256": audit["input_sha256"]["v8_pair_audit"]}
    (out / "label_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
