"""Replace unavailable downloads in the 750-image European expansion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from extend_global_streetscapes_sample import ALLOWED_PLATFORMS, KEEP_COLUMNS, NEW_CITIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Global Streetscapes streetscapes.parquet")
    parser.add_argument("--metadata-kv", type=Path, required=True, help="Global Streetscapes metadata_kv.parquet")
    parser.add_argument("--manifest", type=Path, default=Path("data/expansion_750/images_manifest_new_750.csv"))
    parser.add_argument("--combined", type=Path, default=Path("data/expansion_750/images_manifest_1000.csv"))
    parser.add_argument("--download-log", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("data/expansion_750/replacements.csv"))
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--max-api-checks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    failures = pd.read_csv(args.download_log).query("status == 'failed'")
    if failures.empty:
        print("No failed rows to replace.")
        return
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if manifest.source.eq("Mapillary").any() and not token:
        raise RuntimeError("Set MAPILLARY_ACCESS_TOKEN before checking Mapillary replacements")
    missing = set(KEEP_COLUMNS) - set(pq.ParquetFile(args.input).schema_arrow.names)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    extra = [column for column in KEEP_COLUMNS if column not in {"city"}]
    read_columns = ["city", "city_ascii", "country", "continent", *[c for c in extra if c not in {"city_ascii", "country", "continent"}]]
    candidates = pq.read_table(args.input, columns=read_columns).to_pandas()
    candidates = candidates.loc[
        candidates.city_ascii.isin(NEW_CITIES)
        & candidates.continent.eq("Europe")
        & candidates.source.isin(("KartaView", "Mapillary"))
        & candidates.platform.isin(ALLOWED_PLATFORMS)
        & candidates.quality.eq("good")
        & candidates.lighting_condition.eq("day")
        & candidates.glare.eq(False)
        & candidates.reflection.eq(False)
        & candidates.lat.notna()
        & candidates.lon.notna()
        & candidates.sequence_id.notna()
    ].copy()
    size = float(manifest.grid_size_degrees.iloc[0])
    candidates["grid_lat"] = (candidates.lat / size).floordiv(1).astype("int64")
    candidates["grid_lon"] = (candidates.lon / size).floordiv(1).astype("int64")
    candidates["grid_cell"] = candidates.grid_lat.astype(str) + "_" + candidates.grid_lon.astype(str)
    candidates = candidates.sample(frac=1, random_state=args.seed).drop_duplicates(["source", "sequence_id"])
    used_uuids = set(manifest.uuid.astype(str))
    used_sequences = set(zip(manifest.source.astype(str), manifest.sequence_id.astype(str), strict=True))
    for column in ("replacement_of_uuid", "replacement_reason"):
        if column in manifest.columns:
            manifest[column] = manifest[column].astype("object")
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"OAuth {token}"
    metadata = pd.read_parquet(args.metadata_kv, columns=["orig_id", "kv_fileurlProc"])
    kv_urls = dict(zip(metadata.orig_id.astype("int64"), metadata.kv_fileurlProc, strict=False))
    updated = manifest.copy()
    audit: list[dict[str, object]] = []

    for index, failed in failures.reset_index(drop=True).iterrows():
        old = manifest.loc[manifest.uuid.eq(failed.uuid)].iloc[0]
        pool = candidates.loc[
            candidates.city_ascii.eq(old.city)
            & candidates.source.eq(old.source)
            & ~candidates.uuid.astype(str).isin(used_uuids)
            & ~candidates.sequence_id.astype(str).map(lambda seq: (str(old.source), seq) in used_sequences)
        ].copy()
        if pool.empty:
            raise RuntimeError(f"No unused {old.source} replacement candidate for {old.city}")
        pool = pool.sort_values("uuid").sample(frac=1, random_state=args.seed + index)
        chosen = None
        checked = 0
        for _, candidate in pool.iterrows():
            checked += 1
            if candidate.source == "Mapillary":
                response = session.get(
                    f"https://graph.mapillary.com/{int(candidate.orig_id)}",
                    params={"fields": "id,thumb_2048_url"}, timeout=30,
                )
                accessible = response.ok and bool(response.json().get("thumb_2048_url"))
            else:
                accessible = pd.notna(kv_urls.get(int(candidate.orig_id)))
            if accessible:
                chosen = candidate
                break
            if checked >= args.max_api_checks:
                break
        if chosen is None:
            raise RuntimeError(f"No accessible replacement after checking {checked} candidates for {old.city}")

        position = updated.index[updated.uuid.eq(old.uuid)][0]
        for column in KEEP_COLUMNS:
            if column not in updated.columns or column == "city":
                continue
            value = chosen[column]
            if column == "datetime_local" and pd.notna(value):
                value = pd.Timestamp(value).isoformat(sep=" ")
            updated.loc[position, column] = value
        for column in ("city_ascii", "country", "continent", "grid_lat", "grid_lon", "grid_cell"):
            if column in updated.columns:
                updated.loc[position, column] = chosen[column]
        updated.loc[position, "replacement_of_uuid"] = old.uuid
        updated.loc[position, "replacement_reason"] = "source_unavailable; same-city, same-source candidate API-verified"
        used_uuids.add(str(chosen.uuid))
        used_sequences.add((str(chosen.source), str(chosen.sequence_id)))
        audit.append({
            "old_uuid": old.uuid, "new_uuid": chosen.uuid, "source": chosen.source,
            "orig_id": chosen.orig_id, "city": old.city, "grid_cell": chosen.grid_cell,
            "candidate_checks": checked,
        })

    combined = pd.concat([pd.read_csv(Path("data/images_manifest.csv")), updated], ignore_index=True, sort=False)
    if len(updated) != 750 or updated.uuid.nunique() != 750 or len(combined) != 1000 or combined.uuid.nunique() != 1000:
        raise ValueError("Replacement would violate the 750-new / 1,000-combined unique-image counts")
    backup = args.manifest.with_name(f"{args.manifest.stem}_before_replacements.csv")
    manifest.to_csv(backup, index=False)
    updated.to_csv(args.manifest, index=False)
    combined.to_csv(args.combined, index=False)
    pd.DataFrame(audit).to_csv(args.audit, index=False)
    print(f"Replaced {len(audit)} rows; backup: {backup}; audit: {args.audit}")


if __name__ == "__main__":
    main()
