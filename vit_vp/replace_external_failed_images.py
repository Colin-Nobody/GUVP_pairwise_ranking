#!/usr/bin/env python3
"""Replace inaccessible sampled images with next ranked candidates in the same city."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/external_vp_4city_200"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "images_manifest.csv")
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_DIR / "candidate_pool.csv")
    parser.add_argument("--download-log", type=Path, default=DEFAULT_DIR / "download_log.csv")
    parser.add_argument("--output-replacements", type=Path, default=DEFAULT_DIR / "download_replacements.csv")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    candidates = pd.read_csv(args.candidate_pool)
    log = pd.read_csv(args.download_log)
    current_ids = set(manifest.uuid.astype(str))
    all_failed_ids = set(log.loc[log.status.eq("failed"), "uuid"].astype(str))
    failed = log.loc[log.status.eq("failed") & log.uuid.astype(str).isin(current_ids), ["uuid", "city", "detail"]].drop_duplicates("uuid")
    if failed.empty:
        print("No failed image downloads found; no replacements needed.")
        return
    replacements = []
    for row in failed.itertuples(index=False):
        city_pool = candidates.loc[candidates.city.eq(row.city)].copy()
        city_pool = city_pool.loc[~city_pool.uuid.astype(str).isin(current_ids | all_failed_ids)]
        city_pool = city_pool.loc[~city_pool.sequence_id.astype(str).isin(set(manifest.loc[manifest.city.eq(row.city), "sequence_id"].astype(str)))]
        if city_pool.empty:
            raise RuntimeError(f"No unused same-city sequence candidates remain for failed image {row.uuid}")
        replacement = city_pool.sort_values("selection_rank_within_city", kind="stable").iloc[0].copy()
        replacement["replaced_uuid"] = str(row.uuid)
        replacement["replacement_reason"] = str(row.detail)
        index = manifest.index[manifest.uuid.astype(str).eq(str(row.uuid))]
        if len(index) != 1:
            raise ValueError(f"Expected one manifest row for failed UUID {row.uuid}")
        target_index = index[0]
        for column in replacement.index:
            if column in manifest.columns:
                manifest.at[target_index, column] = replacement[column]
        manifest.at[target_index, "image_id"] = str(replacement.uuid)
        manifest.at[target_index, "replaced_uuid"] = str(row.uuid)
        manifest.at[target_index, "replacement_reason"] = str(row.detail)
        current_ids.remove(str(row.uuid))
        current_ids.add(str(replacement.uuid))
        replacements.append(replacement)
    replacement_frame = pd.DataFrame(replacements)
    args.output_replacements.parent.mkdir(parents=True, exist_ok=True)
    replacement_frame.to_csv(args.output_replacements, index=False)
    manifest.to_csv(args.manifest, index=False)
    metadata = {
        "failed_images_replaced": len(replacements),
        "replacement_rule": "Next grid-round-robin candidate in the same city, with a unique sequence and not present in the training set.",
        "counts_per_city": {str(k): int(v) for k, v in manifest.groupby("city").size().items()},
        "replacement_file": str(args.output_replacements.resolve()),
        "updated_manifest": str(args.manifest.resolve()),
    }
    (args.manifest.parent / "replacement_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
