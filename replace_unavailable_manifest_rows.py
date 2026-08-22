"""Replace failed image-download rows with equivalent unused manifest candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from prepare_global_streetscapes_sample import KEEP_COLUMNS, filter_candidates, make_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/images_manifest.csv"))
    parser.add_argument("--download-log", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="streetscapes.parquet")
    parser.add_argument("--output", type=Path, default=Path("data/images_manifest_replacements.csv"))
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    failures = pd.read_csv(args.download_log).query("status == 'failed'")
    if failures.empty:
        print("No failed rows to replace.")
        return
    if not set(failures.uuid).issubset(set(manifest.uuid)):
        raise ValueError("Download log contains failures not present in manifest")
    raw = pq.read_table(args.input, columns=KEEP_COLUMNS).to_pandas()
    source_set = sorted(manifest.source.dropna().unique().tolist())
    grid_size = float(manifest.grid_size_degrees.iloc[0])
    candidates = make_grid(filter_candidates(raw, source_set), grid_size)
    existing_sequences = set(zip(manifest.source.astype(str), manifest.sequence_id.astype(str), strict=True))
    existing_uuids = set(manifest.uuid)
    replacements: list[dict[str, object]] = []
    updated = manifest.copy()
    for index, failed in failures.reset_index(drop=True).iterrows():
        old = manifest.loc[manifest.uuid.eq(failed.uuid)].iloc[0]
        pool = candidates.loc[
            candidates.city.eq(old.city)
            & candidates.source.eq(old.source)
            & candidates.grid_cell.eq(old.grid_cell)
            & ~candidates.uuid.isin(existing_uuids)
        ].copy()
        pool = pool.loc[~pool.apply(lambda row: (str(row.source), str(row.sequence_id)) in existing_sequences, axis=1)]
        if pool.empty:
            pool = candidates.loc[
                candidates.city.eq(old.city) & candidates.source.eq(old.source) & ~candidates.uuid.isin(existing_uuids)
            ].copy()
            pool = pool.loc[~pool.apply(lambda row: (str(row.source), str(row.sequence_id)) in existing_sequences, axis=1)]
        if pool.empty:
            raise RuntimeError(f"No unused replacement candidate for {old.city}/{old.source}")
        choice = pool.sort_values("uuid").sample(n=1, random_state=args.seed + index).iloc[0]
        position = updated.index[updated.uuid.eq(failed.uuid)][0]
        for column in KEEP_COLUMNS + ["grid_lat", "grid_lon", "grid_cell"]:
            value = choice[column]
            if column == "datetime_local" and pd.notna(value):
                value = pd.Timestamp(value).isoformat(sep=" ")
            updated.loc[position, column] = value
        updated.loc[position, "replacement_of_uuid"] = failed.uuid
        updated.loc[position, "replacement_reason"] = "original_source_unavailable_at_download"
        existing_uuids.add(choice.uuid)
        existing_sequences.add((str(choice.source), str(choice.sequence_id)))
        replacements.append({"old_uuid": failed.uuid, "new_uuid": choice.uuid, "source": choice.source, "orig_id": choice.orig_id, "city": choice.city, "grid_cell": choice.grid_cell})
    backup = args.manifest.with_name(f"{args.manifest.stem}_before_replacement.csv")
    manifest.to_csv(backup, index=False)
    updated.to_csv(args.manifest, index=False)
    pd.DataFrame(replacements).to_csv(args.output, index=False)
    print(f"Replaced {len(replacements)} row(s); backup: {backup}; audit: {args.output}")


if __name__ == "__main__":
    main()
