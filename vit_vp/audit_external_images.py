#!/usr/bin/env python3
"""Verify downloaded external images and flag exact/near duplicates vs development data."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/external_vp_4city_200"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(im.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "images_manifest.csv")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_DIR / "images")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/expansion_750/images_manifest_1000.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    external = pd.read_csv(args.manifest)
    train = pd.read_csv(args.train_manifest)
    if len(external) != 200 or external.uuid.astype(str).nunique() != 200:
        raise ValueError("Expected 200 unique external image records")
    rows, external_hashes, external_hash_by_id = [], {}, {}
    for row in external.itertuples(index=False):
        image_id = str(row.uuid)
        path = args.images_dir / f"{image_id}.jpg"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty external image: {path}")
        with Image.open(path) as im:
            im.verify()
        sha = digest(path)
        dh = dhash(path)
        if sha in external_hashes:
            raise ValueError(f"Exact duplicate external images: {image_id} and {external_hashes[sha]}")
        external_hashes[sha] = image_id
        external_hash_by_id[image_id] = (sha, dh)
        rows.append({"image_id": image_id, "city": str(row.city), "source": str(row.source), "path": str(path.resolve()),
                     "bytes": path.stat().st_size, "sha256": sha, "dhash_hex": f"{dh:016x}"})

    train_records = []
    for row in train.itertuples(index=False):
        image_id = str(row.uuid)
        path = ROOT / "data/expansion_750/street_images_1000" / f"{image_id}.jpg"
        if not path.is_file():
            path = ROOT / "data/street_images" / f"{image_id}.jpg"
        if path.is_file():
            train_records.append((image_id, str(row.city), dhash(path)))
    near = []
    threshold = args.near_duplicate_distance
    for a, b in combinations(rows, 2):
        distance = (int(a["dhash_hex"], 16) ^ int(b["dhash_hex"], 16)).bit_count()
        if distance <= threshold:
            near.append({"image_a": a["image_id"], "city_a": a["city"], "image_b": b["image_id"], "city_b": b["city"], "dhash_distance": distance, "comparison": "external_external"})
    for row in rows:
        dh = int(row["dhash_hex"], 16)
        for train_id, train_city, train_hash in train_records:
            distance = (dh ^ train_hash).bit_count()
            if distance <= threshold:
                near.append({"image_a": row["image_id"], "city_a": row["city"], "image_b": train_id, "city_b": train_city, "dhash_distance": distance, "comparison": "external_train_candidate"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "image_integrity.csv", index=False)
    pd.DataFrame(near, columns=["image_a", "city_a", "image_b", "city_b", "dhash_distance", "comparison"]).to_csv(args.output_dir / "near_duplicate_candidates.csv", index=False)
    result = {
        "image_count": len(rows),
        "cities": {str(k): int(v) for k, v in external.groupby("city").size().items()},
        "source_counts": {str(k): int(v) for k, v in external.groupby("source").size().items()},
        "all_images_decode": True,
        "unique_sha256_count": len(external_hashes),
        "train_images_checked_for_near_duplicates": len(train_records),
        "dhash_candidate_threshold": threshold,
        "near_duplicate_candidate_count": len(near),
        "manual_review_required_if_candidates_exist": bool(near),
        "outputs": ["image_integrity.csv", "near_duplicate_candidates.csv"],
    }
    (args.output_dir / "image_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
