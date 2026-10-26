#!/usr/bin/env python3
"""Replace failed/duplicate supplemental images and audit the nested 50x700 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG, got {image.format}: {path}")
        image.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--supplement-manifest", type=Path, required=True)
    parser.add_argument("--download-log", type=Path, required=True)
    parser.add_argument("--replacement-needed", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--candidate-image-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    image_dir = root / "images"
    base = pd.read_csv(root / "images_manifest.csv", dtype={"uuid": str, "orig_id": str, "sequence_id": str})
    extra = pd.read_csv(args.supplement_manifest, dtype={"uuid": str, "orig_id": str, "sequence_id": str})
    logs = pd.read_csv(args.download_log, dtype={"uuid": str})
    needed = pd.read_csv(args.replacement_needed, dtype={"uuid": str, "orig_id": str, "sequence_id": str})
    pool = pd.read_csv(args.candidate_manifest, dtype={"uuid": str, "orig_id": str, "sequence_id": str})
    pool_log = pd.read_csv(args.candidate_log, dtype={"uuid": str})
    if len(base) != 10_000 or len(extra) != 25_000 or len(logs) != 25_000:
        raise ValueError("Expected 10k base images and 25k supplemental selections/download log rows")
    if logs.uuid.nunique() != len(logs) or logs.status.eq("skipped_existing").any():
        raise ValueError("Supplemental download log is incomplete or has unexpected reused files")
    failed = set(logs.loc[logs.status.eq("failed"), "uuid"])
    if len(failed) + int((needed.reason == "duplicate_content").sum()) != len(needed):
        raise ValueError("Replacement-needed rows disagree with download failures")
    if pool_log.uuid.nunique() != len(pool_log) or set(pool.uuid) != set(pool_log.uuid):
        raise ValueError("Replacement-candidate downloads do not cover the candidate manifest")

    base_hash = pd.read_csv(root / "image_integrity.csv", dtype={"uuid": str})
    if len(base_hash) != 10_000 or base_hash.sha256.nunique() != 10_000:
        raise ValueError("Base 10k integrity record is not complete and unique")
    base_hash_by_uuid = dict(zip(base_hash.uuid, base_hash.sha256))
    base_hashes = set(base_hash_by_uuid.values())

    # Recompute hashes for the kept supplemental rows and ensure they still match the pre-audit.
    rejected = set(needed.uuid)
    kept = extra[~extra.uuid.isin(rejected)].copy()
    kept_hashes = {}
    for i, row in enumerate(kept.itertuples(index=False), start=1):
        path = image_dir / f"{row.uuid}.jpg"
        verify_image(path)
        digest = sha256(path)
        kept_hashes[row.uuid] = digest
        if digest in base_hashes:
            raise ValueError(f"Replacement audit missed a cross-batch duplicate: {row.uuid}")
        if i % 5000 == 0:
            print(f"verified kept supplement {i}/{len(kept)}", flush=True)
    if len(set(kept_hashes.values())) != len(kept_hashes):
        raise ValueError("Replacement audit missed a supplemental duplicate-content row")

    occupied_hashes = base_hashes | set(kept_hashes.values())
    candidate_hashes: dict[str, str] = {}
    candidate_valid = []
    candidate_failures = []
    pool_status = dict(zip(pool_log.uuid, pool_log.status))
    for i, row in enumerate(pool.itertuples(index=False), start=1):
        if pool_status.get(row.uuid) == "failed":
            candidate_failures.append(row.uuid)
            continue
        path = args.candidate_image_dir / f"{row.uuid}.jpg"
        verify_image(path)
        digest = sha256(path)
        candidate_hashes[row.uuid] = digest
        candidate_valid.append(row.uuid)
        if i % 500 == 0:
            print(f"verified replacement candidates {i}/{len(pool)}", flush=True)

    chosen = []
    used_candidate_hashes = set()
    needed_by_city = {city: group.copy() for city, group in needed.groupby("city", sort=False)}
    pool_by_city = {city: group.copy() for city, group in pool.groupby("city", sort=False)}
    replacement_rows = []
    for city, targets in needed_by_city.items():
        possible = pool_by_city.get(city, pool.iloc[0:0]).copy()
        possible = possible[possible.uuid.isin(candidate_valid)].copy()
        possible["sha256"] = possible.uuid.map(candidate_hashes)
        possible["quality_sort"] = pd.to_numeric(possible.quality_score, errors="coerce").fillna(-1)
        possible = possible.sort_values(["quality_sort", "uuid"], ascending=[False, True], kind="stable")
        available = possible[~possible.sha256.isin(occupied_hashes | used_candidate_hashes)]
        if len(available) < len(targets):
            raise RuntimeError(f"Insufficient unique decodable replacement candidates for {city}: need {len(targets)}, have {len(available)}")
        targets = targets.sort_values(["selection_rank_within_city", "uuid"], kind="stable").reset_index(drop=True)
        available = available.head(len(targets)).reset_index(drop=True)
        for target, candidate in zip(targets.itertuples(index=False), available.itertuples(index=False)):
            row = candidate._asdict()
            row["selection_rank_within_city"] = int(target.selection_rank_within_city)
            row["sampling_method"] = str(row.get("sampling_method", "")) + "; replacement for failed/duplicate supplemental candidate"
            row["replaces_uuid"] = str(target.uuid)
            row["replacement_reason"] = str(target.reason)
            row["replacement_sha256"] = str(candidate.sha256)
            replacement_rows.append(row)
            used_candidate_hashes.add(str(candidate.sha256))
            chosen.append(str(candidate.uuid))

    chosen_df = pd.DataFrame(replacement_rows)
    kept["replacement_for_uuid"] = ""
    kept["replacement_reason"] = ""
    final_extra = pd.concat([kept, chosen_df], ignore_index=True, sort=False)
    final_extra["mapillary_url"] = "https://www.mapillary.com/app/?pKey=" + final_extra.orig_id.astype(str)
    final = pd.concat([base, final_extra], ignore_index=True, sort=False)
    final = final.sort_values(["city", "selection_rank_within_city", "uuid"], kind="stable").reset_index(drop=True)

    if len(final) != 35_000 or final.city.nunique() != 50 or not final.groupby("city").size().eq(700).all():
        raise ValueError("Final combined manifest is not exactly 50 cities x 700 images")
    if final.uuid.nunique() != 35_000 or final.orig_id.astype(str).nunique() != 35_000 or final.sequence_id.astype(str).nunique() != 35_000:
        raise ValueError("Final manifest contains duplicate UUID, Mapillary ID, or capture sequence")

    # Keep rejected duplicate content outside the final images directory.
    quarantine = root / "supplement_500" / "rejected_images_700"
    quarantine.mkdir(parents=True, exist_ok=True)
    for row in needed.itertuples(index=False):
        source = image_dir / f"{row.uuid}.jpg"
        if source.exists():
            source.replace(quarantine / source.name)
    for uuid in chosen:
        source = args.candidate_image_dir / f"{uuid}.jpg"
        target = image_dir / source.name
        if target.exists():
            raise FileExistsError(target)
        source.replace(target)

    integrity = []
    for i, row in enumerate(final.itertuples(index=False), start=1):
        path = image_dir / f"{row.uuid}.jpg"
        width, height = verify_image(path)
        integrity.append({"uuid": row.uuid, "city": row.city, "mapillary_id": str(row.orig_id),
                          "bytes": path.stat().st_size, "width": width, "height": height,
                          "sha256": sha256(path), "decode": "ok"})
        if i % 5000 == 0:
            print(f"final integrity {i}/35000", flush=True)
    integrity_df = pd.DataFrame(integrity)
    if integrity_df.sha256.nunique() != 35_000:
        raise ValueError("Final 35k image content hashes are not unique")
    if len(list(image_dir.glob("*.jpg"))) != 35_000:
        raise ValueError("The final images/ directory does not contain exactly 35,000 JPEGs")

    final_manifest = root / "images_manifest_700.csv"
    final.to_csv(final_manifest, index=False, encoding="utf-8-sig")
    integrity_df.to_csv(root / "image_integrity_700.csv", index=False, encoding="utf-8-sig")
    replacement_log = needed.merge(chosen_df[["replaces_uuid", "uuid", "orig_id", "sequence_id", "replacement_reason", "replacement_sha256"]],
                                   left_on="uuid", right_on="replaces_uuid", suffixes=("_removed", "_replacement"),
                                   validate="one_to_one")
    replacement_log.to_csv(root / "supplement_500" / "replacement_log_700.csv", index=False, encoding="utf-8-sig")
    audit = {
        "final_images": 35_000, "cities": 50, "images_per_city": 700,
        "base_images_preserved": 10_000, "new_sample_target": 25_000,
        "unique_mapillary_ids": int(final.orig_id.astype(str).nunique()),
        "unique_sequences": int(final.sequence_id.astype(str).nunique()),
        "unique_content_sha256": int(integrity_df.sha256.nunique()),
        "full_decode_images": len(integrity_df), "replacement_slots": len(needed),
        "download_failures_replaced": int((needed.reason == "download_failed").sum()),
        "duplicate_content_rows_replaced": int((needed.reason == "duplicate_content").sum()),
        "replacement_candidate_download_failures": len(candidate_failures),
        "oecd_region_count": int(final.oecd_region_code.nunique()),
        "image_dir_jpg_count": len(list(image_dir.glob("*.jpg"))),
        "manifest": str(final_manifest.resolve()),
    }
    (root / "audit_700.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
