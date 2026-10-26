#!/usr/bin/env python3
"""Resumable, concurrent Mapillary thumbnail downloader for the OECD 50-city manifest."""

from __future__ import annotations

import argparse
import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps


TOKEN_FILE = Path.home() / ".config/guvp/mapillary.env"
API_ROOT = "https://graph.mapillary.com/{}"
LOG_FIELDS = ["uuid", "city", "orig_id", "filename", "status", "bytes", "detail"]
_local = threading.local()


def load_token() -> tuple[str, str]:
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if token:
        return token, "MAPILLARY_ACCESS_TOKEN environment variable"
    if not TOKEN_FILE.is_file():
        raise RuntimeError(f"Local Mapillary token file is missing: {TOKEN_FILE}")
    for line in TOKEN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export MAPILLARY_ACCESS_TOKEN="):
            token = line.split("=", 1)[1].strip().strip("\"'")
            if token:
                return token, str(TOKEN_FILE)
    raise RuntimeError(f"No non-empty token assignment found in {TOKEN_FILE}")


def session(token: str) -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({"Authorization": f"OAuth {token}", "User-Agent": "GUVP-OECD-city-image-sample/1.0"})
        _local.session = s
    return _local.session


def get_with_retry(s: requests.Session, url: str, *, params: dict | None = None, stream: bool = False) -> requests.Response:
    last_error = None
    for attempt in range(6):
        try:
            response = s.get(url, params=params, timeout=(15, 90), stream=stream)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 60) if retry_after and retry_after.isdigit() else min(2**attempt, 30)
                response.close()
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code < 500 and exc.response.status_code != 429:
                raise
            if attempt < 5:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(str(last_error or "Mapillary API retry limit reached"))


def verify_jpeg(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise RuntimeError(f"Expected JPEG, got {image.format}")
        image.verify()
    with Image.open(path) as image:
        ImageOps.exif_transpose(image).convert("RGB").load()


def fetch_jpeg(s: requests.Session, url: str, target: Path) -> int:
    temporary = target.with_suffix(".part")
    response = get_with_retry(s, url, stream=True)
    total = 0
    try:
        with temporary.open("wb") as out:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    if total == 0 and not chunk.startswith(b"\xff\xd8"):
                        raise RuntimeError("Thumbnail is not a JPEG")
                    out.write(chunk)
                    total += len(chunk)
        if total < 1000:
            raise RuntimeError(f"Thumbnail is too small ({total} bytes)")
        verify_jpeg(temporary)
        temporary.replace(target)
        return total
    finally:
        response.close()
        temporary.unlink(missing_ok=True)


def download_one(row: dict, image_dir: Path, token: str) -> dict:
    filename = f"{row['uuid']}.jpg"
    target = image_dir / filename
    result = {"uuid": row["uuid"], "city": row["city"], "orig_id": int(row["orig_id"]), "filename": filename}
    if target.is_file() and target.stat().st_size > 1000:
        try:
            verify_jpeg(target)
            result.update(status="skipped_existing", bytes=target.stat().st_size, detail="")
            return result
        except Exception:
            target.unlink(missing_ok=True)
    s = session(token)
    try:
        meta = get_with_retry(s, API_ROOT.format(int(row["orig_id"])), params={"fields": "id,thumb_1024_url,thumb_original_url"})
        payload = meta.json()
        if str(payload.get("id")) != str(int(row["orig_id"])) or not payload.get("thumb_1024_url"):
            raise RuntimeError("Graph API response did not contain the requested image thumbnail")
        resolution = "1024"
        try:
            total = fetch_jpeg(s, payload["thumb_1024_url"], target)
        except Exception:
            # Some Mapillary records expose stale/invalid transformed thumbnail
            # links while their original image URL remains live. The same V8
            # long-edge transform is applied later, so original-resolution
            # fallbacks remain input-size compatible.
            fallback = payload.get("thumb_original_url")
            if not fallback:
                raise
            total = fetch_jpeg(s, fallback, target)
            resolution = "original_fallback"
        result.update(status="downloaded", bytes=total, detail="", resolution_used=resolution)
    except Exception as exc:
        target.with_suffix(".part").unlink(missing_ok=True)
        safe = re.sub(r"https?://\S+", "<redacted-url>", str(exc))
        result.update(status="failed", bytes=0, detail=f"{type(exc).__name__}: {safe}", resolution_used="")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expected-count", type=int, default=10_000,
                        help="Expected number of selected images; set for a supplemental batch")
    parser.add_argument("--limit", type=int, help="Optional initial subset for API preflight")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("--workers must be from 1 through 12")
    token, token_source = load_token()
    if not token.startswith("MLY|"):
        raise RuntimeError("Local token does not have the expected Mapillary Client Token format")

    manifest = pd.read_csv(args.manifest)
    required = {"uuid", "source", "orig_id", "city", "lat", "lon", "oecd_region_code"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    if len(manifest) != args.expected_count and args.limit is None:
        raise ValueError(f"Expected {args.expected_count:,} selected images; found {len(manifest):,}")
    if manifest["orig_id"].nunique() != len(manifest) and args.limit is None:
        raise ValueError("Manifest contains duplicate Mapillary image IDs")
    if set(manifest["source"].dropna()) != {"Mapillary"}:
        raise ValueError("Manifest is not Mapillary-only")
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    probe_session = requests.Session()
    probe_session.headers["Authorization"] = f"OAuth {token}"
    try:
        probe = get_with_retry(probe_session, "https://graph.mapillary.com/images",
                               params={"fields": "id", "limit": 1})
        probe.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("Mapillary Graph API preflight did not validate the configured local token") from None

    rows = manifest.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, args.image_dir, token) for row in rows]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 50 == 0 or done == len(futures):
                interim = pd.DataFrame(results)
                interim.to_csv(args.log, index=False)
                counts = interim.status.value_counts().to_dict()
                print(f"completed={done}/{len(futures)} downloaded={counts.get('downloaded',0)} skipped={counts.get('skipped_existing',0)} failed={counts.get('failed',0)}", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(args.log, index=False)
    print(f"token_source={token_source}")
    print(out.groupby(["city", "status"]).size().unstack(fill_value=0).to_string())
    print(f"total_bytes={int(out['bytes'].sum())}")
    if out.status.eq("failed").any():
        raise SystemExit(f"{int(out.status.eq('failed').sum())} downloads failed; rerun to resume from existing JPEGs")


if __name__ == "__main__":
    main()
