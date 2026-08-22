"""Download the GUVP Global Streetscapes manifest with an auditable log.

Mapillary downloads require MAPILLARY_ACCESS_TOKEN in the environment. KartaView
uses the public processed-image URLs held in metadata_kv.parquet.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
import requests


MAPILLARY_URL = "https://graph.mapillary.com/{image_id}?fields=id,thumb_2048_url"
LOG_COLUMNS = ["uuid", "source", "orig_id", "city", "filename", "status", "bytes", "detail"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/images_manifest.csv"))
    parser.add_argument("--metadata-kv", type=Path, help="Global Streetscapes metadata_kv.parquet; required if KartaView rows are selected")
    parser.add_argument("--output-dir", type=Path, default=Path("data/street_images"))
    parser.add_argument("--log", type=Path, default=Path("data/image_download_log.csv"))
    parser.add_argument("--limit", type=int, help="Optional number of manifest rows for a smoke test")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def kartaview_urls(manifest: pd.DataFrame, metadata_path: Path | None) -> dict[int, str]:
    karta_ids = manifest.loc[manifest.source.eq("KartaView"), "orig_id"].astype("int64").tolist()
    if not karta_ids:
        return {}
    if not metadata_path:
        raise ValueError("--metadata-kv is required for selected KartaView rows")
    metadata = pd.read_parquet(metadata_path, columns=["orig_id", "kv_fileurlProc"])
    urls = metadata[metadata.orig_id.isin(karta_ids)].dropna(subset=["kv_fileurlProc"])
    result = dict(zip(urls.orig_id.astype("int64"), urls.kv_fileurlProc, strict=True))
    missing = sorted(set(karta_ids) - set(result))
    if missing:
        raise ValueError(f"No processed KartaView URL for {len(missing)} selected images")
    return result


def request_with_retries(session: requests.Session, url: str, timeout: float, retries: int, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


def mapillary_image_url(session: requests.Session, image_id: int, timeout: float, retries: int) -> str:
    response = request_with_retries(session, MAPILLARY_URL.format(image_id=image_id), timeout, retries)
    url = response.json().get("thumb_2048_url")
    if not url:
        raise RuntimeError("Mapillary response did not contain thumb_2048_url")
    return url


def download(session: requests.Session, url: str, target: Path, timeout: float, retries: int) -> int:
    response = request_with_retries(session, url, timeout, retries, stream=True)
    temporary = target.with_suffix(".part")
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded empty file")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.stat().st_size


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"uuid", "source", "orig_id", "city"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if manifest.source.eq("Mapillary").any() and not token:
        raise RuntimeError("Set MAPILLARY_ACCESS_TOKEN before downloading Mapillary images")
    kv_urls = kartaview_urls(manifest, args.metadata_kv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"OAuth {token}"

    results: list[dict[str, object]] = []
    for index, row in manifest.reset_index(drop=True).iterrows():
        filename = f"{row.uuid}.jpg"
        target = args.output_dir / filename
        result = {"uuid": row.uuid, "source": row.source, "orig_id": int(row.orig_id), "city": row.city, "filename": filename}
        try:
            if target.is_file() and target.stat().st_size > 0:
                result.update(status="skipped_existing", bytes=target.stat().st_size, detail="")
            else:
                url = mapillary_image_url(session, int(row.orig_id), args.timeout, args.retries) if row.source == "Mapillary" else kv_urls[int(row.orig_id)]
                size = download(session, url, target, args.timeout, args.retries)
                result.update(status="downloaded", bytes=size, detail="")
            print(f"[{index + 1}/{len(manifest)}] {result['status']}: {filename}")
        except Exception as error:  # keep the remaining manifest auditable and resumable
            result.update(status="failed", bytes=0, detail=str(error))
            print(f"[{index + 1}/{len(manifest)}] failed: {filename}: {error}")
        results.append(result)
    with args.log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    summary = pd.DataFrame(results).groupby(["source", "status"], dropna=False).size()
    print("Download summary:")
    print(summary.to_string())
    if any(row["status"] == "failed" for row in results):
        raise SystemExit("One or more downloads failed; rerun the same command to resume.")


if __name__ == "__main__":
    main()
