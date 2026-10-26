#!/usr/bin/env python3
"""Resumable, low-concurrency download of selected public KartaView images."""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/external_vp_4city_200"
thread_local = threading.local()
log_lock = threading.Lock()


def session() -> requests.Session:
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers["User-Agent"] = "GUVP-research-image-fetch/1.0"
    return thread_local.session


def append_log(path: Path, row: dict) -> None:
    with log_lock:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["uuid", "source", "orig_id", "city", "filename", "status", "bytes", "detail"])
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


def download_one(row: dict, url: str, out_dir: Path, log_path: Path, timeout: float, retries: int) -> dict:
    filename = f"{row['uuid']}.jpg"
    target = out_dir / filename
    result = {"uuid": str(row["uuid"]), "source": "KartaView", "orig_id": int(row["orig_id"]), "city": str(row["city"]), "filename": filename}
    if target.is_file() and target.stat().st_size > 0:
        try:
            with Image.open(target) as image:
                image.verify()
            result.update(status="skipped_existing_valid", bytes=target.stat().st_size, detail="")
            append_log(log_path, result)
            return result
        except Exception:
            target.unlink(missing_ok=True)
    last_error = "unknown error"
    for attempt in range(retries):
        temporary = target.with_suffix(".part")
        temporary.unlink(missing_ok=True)
        try:
            with session().get(url, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                if not response.headers.get("content-type", "").lower().startswith("image/"):
                    raise RuntimeError(f"Unexpected content type: {response.headers.get('content-type')}")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded empty image")
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(target)
            result.update(status="downloaded", bytes=target.stat().st_size, detail="")
            append_log(log_path, result)
            return result
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    result.update(status="failed", bytes=0, detail=last_error)
    append_log(log_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "images_manifest.csv")
    parser.add_argument("--metadata-kv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "images")
    parser.add_argument("--log", type=Path, default=DEFAULT_DIR / "download_log.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 200 or manifest.uuid.astype(str).nunique() != 200:
        raise ValueError("Expected 200 unique selected KartaView images")
    ids = set(manifest.orig_id.astype("int64"))
    kv = pd.read_parquet(args.metadata_kv, columns=["orig_id", "kv_fileurlProc"])
    kv = kv.loc[kv.orig_id.astype("int64").isin(ids)].dropna(subset=["kv_fileurlProc"])
    url_map = dict(zip(kv.orig_id.astype("int64"), kv.kv_fileurlProc.astype(str)))
    missing = ids - set(url_map)
    if missing:
        raise ValueError(f"No public processed-image URL for {len(missing)} selected KartaView IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if not args.log.exists():
        args.log.write_text("uuid,source,orig_id,city,filename,status,bytes,detail\n", encoding="utf-8")
    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, row._asdict(), url_map[int(row.orig_id)], args.output_dir, args.log, args.timeout, args.retries)
                   for row in manifest.itertuples(index=False)]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result["status"] == "failed":
                failed.append(result)
            if done % 10 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} resolved; downloaded/skipped={done-len(failed)}, failed={len(failed)}", flush=True)
    summary = pd.read_csv(args.log).tail(len(manifest)).groupby(["city", "status"]).size().to_dict()
    print(json.dumps({"images_dir": str(args.output_dir.resolve()), "log": str(args.log.resolve()), "status_counts": {str(k): int(v) for k,v in summary.items()}, "failed": failed}, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(f"{len(failed)} image downloads failed; use candidate_pool.csv to replace failures and rerun.")


if __name__ == "__main__":
    main()
