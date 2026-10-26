#!/usr/bin/env python3
"""Decode, hash, and detect duplicate content in a downloaded OECD city batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"expected JPEG, got {image.format}")
        image.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--download-log", type=Path, required=True)
    parser.add_argument("--base-integrity", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-integrity", type=Path, required=True)
    parser.add_argument("--replacement-needed", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.new_manifest, dtype={"uuid": str, "orig_id": str, "sequence_id": str, "oecd_region_code": str})
    log = pd.read_csv(args.download_log, dtype={"uuid": str})
    base = pd.read_csv(args.base_integrity, dtype={"uuid": str, "sha256": str})
    if manifest.uuid.duplicated().any() or manifest.orig_id.astype(str).duplicated().any() or manifest.sequence_id.astype(str).duplicated().any():
        raise ValueError("New manifest has duplicate UUID, Mapillary ID, or sequence")
    if len(log) != len(manifest) or log.uuid.duplicated().any() or set(log.uuid) != set(manifest.uuid):
        raise ValueError("Download log does not cover the new manifest exactly once")
    if base.uuid.nunique() != len(base) or base.sha256.nunique() != len(base):
        raise ValueError("Base integrity file is not complete and content-unique")

    status = dict(zip(log.uuid, log.status))
    occupied_hashes = set(base.sha256.astype(str))
    records, needed = [], []
    total = len(manifest)
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        record = {"uuid": str(row.uuid), "city": str(row.city), "oecd_region_code": str(row.oecd_region_code),
                  "orig_id": str(row.orig_id), "sequence_id": str(row.sequence_id),
                  "selection_rank_within_city": int(row.selection_rank_within_city),
                  "download_status": str(status[row.uuid]), "decode": "not_checked", "bytes": 0,
                  "width": 0, "height": 0, "sha256": "", "reason": ""}
        if status[row.uuid] == "failed":
            record["decode"] = "download_failed"
            record["reason"] = "download_failed"
        else:
            path = args.image_dir / f"{row.uuid}.jpg"
            try:
                width, height = decode(path)
                digest = sha256(path)
                record.update(decode="ok", bytes=path.stat().st_size, width=width, height=height, sha256=digest)
                if digest in occupied_hashes:
                    record["reason"] = "duplicate_content"
                else:
                    occupied_hashes.add(digest)
            except Exception as exc:
                record["decode"] = "invalid_or_missing"
                record["reason"] = "decode_or_file_error"
                record["detail"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if record["reason"]:
            needed.append(record)
        if i % 5000 == 0 or i == total:
            print(f"audited {i}/{total} supplemental images", flush=True)

    integrity = pd.DataFrame(records)
    needed_df = pd.DataFrame(needed, columns=["uuid", "city", "oecd_region_code", "orig_id", "sequence_id", "selection_rank_within_city", "download_status", "decode", "reason"])
    args.output_integrity.parent.mkdir(parents=True, exist_ok=True)
    integrity.to_csv(args.output_integrity, index=False, encoding="utf-8-sig")
    needed_df.to_csv(args.replacement_needed, index=False, encoding="utf-8-sig")
    summary = {
        "new_images": len(manifest), "new_cities": int(manifest.city.nunique()),
        "download_failed": int((integrity.download_status == "failed").sum()),
        "decode_or_file_errors": int((integrity.reason == "decode_or_file_error").sum()),
        "duplicate_content_rows": int((integrity.reason == "duplicate_content").sum()),
        "replacement_slots": len(needed_df), "unique_downloaded_sha256": int(integrity.loc[integrity.reason.eq(""), "sha256"].nunique()),
        "images_per_city_before_integrity": 700,
        "replacement_needed_by_city": {str(k): int(v) for k, v in needed_df.groupby("city").size().items()},
    }
    summary_path = args.output_integrity.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "replacement_needed_by_city"}, ensure_ascii=False, indent=2))
    print(f"Replacement counts: {summary['replacement_needed_by_city']}")


if __name__ == "__main__":
    main()
