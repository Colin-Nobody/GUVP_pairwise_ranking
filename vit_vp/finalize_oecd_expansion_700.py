#!/usr/bin/env python3
"""Assemble complete, content-unique 700-image city samples and audit the result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"not JPEG: {path}")
        image.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        width, height = image.size
    return width, height, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--new-integrity", type=Path, required=True)
    parser.add_argument("--replacement-needed", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-integrity", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path)
    parser.add_argument("--replacement-log", type=Path)
    parser.add_argument("--replacement-image-dir", type=Path)
    parser.add_argument("--expected-per-city", type=int, default=700)
    args = parser.parse_args()

    root = args.root
    image_dir = root / "images"
    base_manifest = pd.read_csv(root / "images_manifest_700.csv", dtype={"uuid": str, "orig_id": str, "sequence_id": str, "oecd_region_code": str}, low_memory=False)
    base_integrity = pd.read_csv(root / "image_integrity_700.csv", dtype={"uuid": str, "sha256": str})
    new = pd.read_csv(args.new_manifest, dtype={"uuid": str, "orig_id": str, "sequence_id": str, "oecd_region_code": str})
    audit = pd.read_csv(args.new_integrity, dtype={"uuid": str, "sha256": str, "sequence_id": str})
    needed = pd.read_csv(args.replacement_needed, dtype={"uuid": str})
    if len(base_manifest) != 35_000 or base_manifest.city.nunique() != 50 or len(base_integrity) != 35_000:
        raise ValueError("The existing 50-city by 700-image base is incomplete")
    if len(new) != 700 * new.city.nunique() or new.groupby("city").size().ne(args.expected_per_city).any():
        raise ValueError("New manifest does not have exactly the expected number of rows per city")
    if set(audit.uuid) != set(new.uuid) or audit.uuid.duplicated().any():
        raise ValueError("New integrity audit does not cover the new manifest exactly once")
    if set(needed.uuid) != set(audit.loc[audit.reason.fillna("").ne(""), "uuid"]):
        raise ValueError("Replacement-needed rows disagree with integrity audit")

    candidates = pd.DataFrame()
    candidate_status: dict[str, str] = {}
    if args.replacement_manifest:
        if not args.replacement_log or not args.replacement_image_dir:
            raise ValueError("Replacement log and image directory are required with a replacement manifest")
        candidates = pd.read_csv(args.replacement_manifest, dtype={"uuid": str, "orig_id": str, "sequence_id": str, "oecd_region_code": str})
        clog = pd.read_csv(args.replacement_log, dtype={"uuid": str})
        if len(clog) != len(candidates) or clog.uuid.duplicated().any() or set(clog.uuid) != set(candidates.uuid):
            raise ValueError("Replacement download log is incomplete")
        candidate_status = dict(zip(clog.uuid, clog.status))

    occupied_ids = set(base_manifest.orig_id.astype(str)) | set(new.orig_id.astype(str))
    occupied_sequences = set(base_manifest.sequence_id.astype(str)) | set(new.sequence_id.astype(str))
    if candidates.size:
        if occupied_ids & set(candidates.orig_id.astype(str)) or occupied_sequences & set(candidates.sequence_id.astype(str)):
            raise ValueError("Replacement candidate repeats a selected Mapillary ID or capture sequence")

    base_hash = dict(zip(base_integrity.uuid.astype(str), base_integrity.sha256.astype(str)))
    if base_integrity.sha256.nunique() != len(base_integrity):
        raise ValueError("Existing base has duplicate content hashes")
    occupied_hashes = set(base_hash.values())
    audit_by_uuid = audit.set_index("uuid")
    candidate_groups = {c: g.copy() for c, g in candidates.groupby("city", sort=False)} if len(candidates) else {}
    final_new_rows, new_integrity_rows, chosen_candidates = [], [], set()
    dropped = []

    for city, rows in new.groupby("city", sort=False):
        rows = rows.sort_values(["selection_rank_within_city", "uuid"], kind="stable")
        valid_rows, city_hashes = [], set()
        for row in rows.itertuples(index=False):
            ar = audit_by_uuid.loc[str(row.uuid)]
            digest = "" if pd.isna(ar.sha256) else str(ar.sha256)
            reason = "" if pd.isna(ar.reason) else str(ar.reason)
            if reason or not digest or digest in occupied_hashes or digest in city_hashes:
                continue
            valid_rows.append((row, digest))
            city_hashes.add(digest)
        missing = args.expected_per_city - len(valid_rows)

        available_candidates = []
        candidate_digests = set()
        for candidate in candidate_groups.get(city, candidates.iloc[0:0]).itertuples(index=False):
            if candidate_status.get(str(candidate.uuid)) == "failed":
                continue
            path = args.replacement_image_dir / f"{candidate.uuid}.jpg"
            try:
                width, height, digest = inspect(path)
            except Exception:
                continue
            if digest in occupied_hashes or digest in city_hashes or digest in candidate_digests:
                continue
            quality = pd.to_numeric(pd.Series([getattr(candidate, "quality_score", None)]), errors="coerce").fillna(-1).iloc[0]
            available_candidates.append((float(quality), candidate, width, height, digest))
            candidate_digests.add(digest)
        available_candidates.sort(key=lambda x: (-x[0], str(x[1].uuid)))
        if missing > len(available_candidates):
            dropped.append({"city": city, "oecd_region_code": str(rows.iloc[0].oecd_region_code),
                            "downloaded_unique_valid": len(valid_rows), "replacement_needed": missing,
                            "replacement_available": len(available_candidates), "reason": "could_not_reach_700_valid_unique_images"})
            continue

        missing_targets = needed[needed.city.eq(city)].sort_values(["selection_rank_within_city", "uuid"], kind="stable")
        if len(missing_targets) != missing:
            raise RuntimeError(f"Replacement-needed rows do not match the missing unique images for {city}")
        used_for_city = [(row, digest, None) for row, digest in valid_rows]
        used_for_city.extend((candidate, digest, target) for (_, candidate, _, _, digest), target in zip(available_candidates[:missing], missing_targets.itertuples(index=False)))
        for row, digest, replacement_target in used_for_city:
            if replacement_target is None:
                ar = audit_by_uuid.loc[str(row.uuid)]
                width, height = int(ar.width), int(ar.height)
                image_path = image_dir / f"{row.uuid}.jpg"
                final_row = row._asdict()
                final_row["replacement_for_uuid"] = ""
                final_row["replacement_reason"] = ""
            else:
                source = args.replacement_image_dir / f"{row.uuid}.jpg"
                target = image_dir / source.name
                if target.exists():
                    raise FileExistsError(target)
                source.replace(target)
                chosen_candidates.add(str(row.uuid))
                width, height, checked_digest = inspect(target)
                if checked_digest != digest:
                    raise ValueError(f"Replacement hash changed during promotion: {row.uuid}")
                final_row = row._asdict()
                final_row["selection_rank_within_city"] = int(replacement_target.selection_rank_within_city)
                final_row["replacement_for_uuid"] = str(replacement_target.uuid)
                final_row["replacement_reason"] = str(replacement_target.reason)
                final_row["sampling_method"] = str(final_row.get("sampling_method", "")) + "; replacement for unavailable or duplicate content"
            occupied_hashes.add(digest)
            final_new_rows.append(final_row)
            new_integrity_rows.append({"uuid": str(row.uuid), "city": city, "mapillary_id": str(row.orig_id),
                                       "bytes": (image_dir / f"{row.uuid}.jpg").stat().st_size,
                                       "width": width, "height": height, "sha256": digest, "decode": "ok"})

    # Keep rejected generated samples out of the final directory; retain their CSV audits and logs.
    keep_file_ids = set(base_manifest.uuid.astype(str)) | {str(r["uuid"]) for r in final_new_rows}
    for row in new.itertuples(index=False):
        if str(row.uuid) not in keep_file_ids:
            (image_dir / f"{row.uuid}.jpg").unlink(missing_ok=True)
    if args.replacement_image_dir and args.replacement_image_dir.exists():
        for path in args.replacement_image_dir.glob("*.jpg"):
            if path.stem not in chosen_candidates:
                path.unlink(missing_ok=True)

    final_new = pd.DataFrame(final_new_rows, columns=new.columns)
    if len(final_new):
        final_new = final_new.sort_values(["city", "selection_rank_within_city", "uuid"], kind="stable")
    final = pd.concat([base_manifest, final_new], ignore_index=True, sort=False)
    final_integrity_new = pd.DataFrame(new_integrity_rows, columns=base_integrity.columns)
    final_integrity = pd.concat([base_integrity, final_integrity_new], ignore_index=True, sort=False)
    expected_cities = 50 + final_new.city.nunique()
    if final.city.nunique() != expected_cities or final.groupby("city").size().ne(args.expected_per_city).any():
        raise RuntimeError("Final manifest has a city with an unexpected image count")
    if final.uuid.duplicated().any() or final.orig_id.astype(str).duplicated().any() or final.sequence_id.astype(str).duplicated().any():
        raise RuntimeError("Final manifest contains duplicate UUID, Mapillary ID, or sequence")
    if final_integrity.sha256.nunique() != len(final_integrity) or len(final_integrity) != len(final):
        raise RuntimeError("Final integrity records are not complete and content-unique")
    if len(list(image_dir.glob("*.jpg"))) != len(final):
        raise RuntimeError("Image directory still contains files outside the finalized manifest")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_manifest, index=False, encoding="utf-8-sig")
    args.output_integrity.parent.mkdir(parents=True, exist_ok=True)
    final_integrity.to_csv(args.output_integrity, index=False, encoding="utf-8-sig")
    dropped_path = root / "excluded_cities_download_integrity.csv"
    pd.DataFrame(dropped).to_csv(dropped_path, index=False, encoding="utf-8-sig")
    summary = {
        "final_images": len(final), "final_cities": int(final.city.nunique()),
        "images_per_city": args.expected_per_city, "unique_oecd_regions": int(final.oecd_region_code.nunique()),
        "unique_mapillary_ids": int(final.orig_id.astype(str).nunique()),
        "unique_sequences": int(final.sequence_id.astype(str).nunique()),
        "unique_sha256": int(final_integrity.sha256.nunique()), "full_decode_images": len(final_integrity),
        "dropped_new_cities_for_integrity_or_replacement_shortfall": dropped,
        "image_dir_jpeg_count": len(list(image_dir.glob("*.jpg"))),
        "manifest": str(args.output_manifest.resolve()), "integrity": str(args.output_integrity.resolve()),
    }
    (root / "audit_expanded_700.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "dropped_new_cities_for_integrity_or_replacement_shortfall"}, ensure_ascii=False, indent=2))
    print(f"dropped new cities={len(dropped)}; details={dropped_path}")


if __name__ == "__main__":
    main()
