"""Replace failed Mapillary records only after checking their current API availability."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from prepare_global_streetscapes_sample import KEEP_COLUMNS, filter_candidates, make_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/images_manifest.csv"))
    parser.add_argument("--download-log", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("data/images_manifest_accessible_replacements.csv"))
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-api-checks", type=int, default=100)
    return parser.parse_args()


def available_image_id(session: requests.Session, image_id: int) -> bool:
    response = session.get(
        f"https://graph.mapillary.com/{image_id}", params={"fields": "id,thumb_2048_url"}, timeout=30
    )
    if not response.ok:
        return False
    return bool(response.json().get("thumb_2048_url"))


def main() -> None:
    args = parse_args()
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Set MAPILLARY_ACCESS_TOKEN before checking Mapillary candidates")
    manifest = pd.read_csv(args.manifest)
    failed = pd.read_csv(args.download_log).query("status == 'failed'")
    if len(failed) != 1 or failed.source.iloc[0] != "Mapillary":
        raise ValueError("This tool expects exactly one failed Mapillary row")
    old = manifest.loc[manifest.uuid.eq(failed.uuid.iloc[0])].iloc[0]
    source_set = sorted(manifest.source.dropna().unique().tolist())
    raw = pq.read_table(args.input, columns=KEEP_COLUMNS).to_pandas()
    candidates = make_grid(filter_candidates(raw, source_set), float(manifest.grid_size_degrees.iloc[0]))
    used_uuids = set(manifest.uuid)
    used_sequences = set(zip(manifest.source.astype(str), manifest.sequence_id.astype(str), strict=True))
    pool = candidates.loc[
        candidates.city.eq(old.city) & candidates.source.eq("Mapillary") & candidates.grid_cell.eq(old.grid_cell)
    ].copy()
    if pool.empty:
        pool = candidates.loc[candidates.city.eq(old.city) & candidates.source.eq("Mapillary")].copy()
    pool = pool.loc[~pool.uuid.isin(used_uuids)]
    pool = pool.loc[~pool.apply(lambda row: (str(row.source), str(row.sequence_id)) in used_sequences, axis=1)]
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    chosen = None
    checked = 0
    for _, candidate in pool.sort_values("uuid").sample(frac=1, random_state=args.seed).iterrows():
        checked += 1
        if available_image_id(session, int(candidate.orig_id)):
            chosen = candidate
            break
        if checked >= args.max_api_checks:
            break
    if chosen is None:
        raise RuntimeError(f"No accessible replacement after checking {checked} candidates")
    updated = manifest.copy()
    position = updated.index[updated.uuid.eq(old.uuid)][0]
    for column in KEEP_COLUMNS + ["grid_lat", "grid_lon", "grid_cell"]:
        value = chosen[column]
        if column == "datetime_local" and pd.notna(value):
            value = pd.Timestamp(value).isoformat(sep=" ")
        updated.loc[position, column] = value
    updated.loc[position, "replacement_of_uuid"] = old.uuid
    updated.loc[position, "replacement_reason"] = "original_source_unavailable_at_download; API-verified replacement"
    backup = args.manifest.with_name(f"{args.manifest.stem}_before_accessible_replacement.csv")
    manifest.to_csv(backup, index=False)
    updated.to_csv(args.manifest, index=False)
    pd.DataFrame([{"old_uuid": old.uuid, "new_uuid": chosen.uuid, "orig_id": chosen.orig_id, "city": chosen.city, "grid_cell": chosen.grid_cell, "api_candidates_checked": checked}]).to_csv(args.audit, index=False)
    print(f"Replaced {old.uuid} after checking {checked} API candidates; new UUID: {chosen.uuid}")


if __name__ == "__main__":
    main()
