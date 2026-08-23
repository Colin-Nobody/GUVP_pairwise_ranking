#!/usr/bin/env python3
"""Verify that the archived GUVP experiment is internally complete."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/images_manifest.csv"
IMAGES = ROOT / "data/street_images"
VOTES = ROOT / "data/vlm_pairwise_formal_v2.csv"
SCORES = ROOT / "output/vlm_formal_v2/perception_scores.csv"
SUMMARY = ROOT / "output/vlm_formal_v2/ranking_summary.json"
VOTES_7B = ROOT / "data/vlm_pairwise_7b_formal_v2.csv"
SCORES_7B = ROOT / "output/vlm_7b_formal_v2/perception_scores.csv"
SUMMARY_7B = ROOT / "output/vlm_7b_formal_v2/ranking_summary.json"
CROSS_MODEL_SUMMARY = ROOT / "output/vlm_3b_vs_7b/comparison_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    votes = pd.read_csv(VOTES)
    scores = pd.read_csv(SCORES)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    votes_7b = pd.read_csv(VOTES_7B)
    scores_7b = pd.read_csv(SCORES_7B)
    summary_7b = json.loads(SUMMARY_7B.read_text(encoding="utf-8"))
    cross_model = json.loads(CROSS_MODEL_SUMMARY.read_text(encoding="utf-8"))

    manifest_ids = set(manifest["uuid"].astype(str))
    image_paths = sorted(IMAGES.glob("*.jpg"))
    image_ids = {path.stem for path in image_paths}
    compared_ids = {
        Path(value).stem
        for value in pd.concat([votes["image_a"], votes["image_b"]]).astype(str)
    }
    score_ids = {Path(value).stem for value in scores["image_id"].astype(str)}

    assert len(manifest) == len(manifest_ids) == 250
    assert len(image_paths) == 250
    assert manifest_ids == image_ids == compared_ids == score_ids
    assert len(votes) == 15_562
    assert votes["status"].eq("ok").all()
    assert summary["valid_vote_rows"] == 15_562
    assert summary["image_count"] == 250
    assert len(votes_7b) == 15_562
    assert votes_7b["status"].eq("ok").all()
    assert votes_7b["model"].eq("Qwen/Qwen2.5-VL-7B-Instruct").all()
    assert set(scores_7b["image_id"].map(lambda value: Path(value).stem)) == image_ids
    assert summary_7b["valid_vote_rows"] == 15_562
    assert summary_7b["image_count"] == 250
    assert cross_model["votes"]["matched_unordered_pairs"] == 15_562
    assert cross_model["scores"]["matched_images"] == 250
    assert manifest.groupby("city").size().eq(50).all()

    for path in image_paths:
        with Image.open(path) as image:
            image.verify()

    print("Bundle verification passed")
    print("  manifest/images/scored images: 250")
    print("  cities: 5 x 50 images")
    print("  valid 3B/7B VLM comparisons: 15,562 each")
    print(f"  3B votes SHA-256: {sha256(VOTES)}")
    print(f"  7B votes SHA-256: {sha256(VOTES_7B)}")
    print(f"  manifest SHA-256: {sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
