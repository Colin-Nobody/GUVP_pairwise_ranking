#!/usr/bin/env python3
"""Freeze and audit the 1,000-image DINOv3 VP experiment inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/expansion_750/images_manifest_1000.csv"
DEFAULT_SCORES = ROOT / "output/vlm_expansion_1000/perception_scores.csv"
DEFAULT_PAIRS = ROOT / "output/vlm_expansion_1000/pair_audit.csv"
DEFAULT_VOTES = ROOT / "data/expansion_750/vlm_pairwise_votes.csv"
DEFAULT_IMAGES = ROOT / "data/expansion_750/street_images_1000"
DEFAULT_OUTPUT = ROOT / "output/vit_vp_1000_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    image = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(image.get_flattened_data())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--votes", type=Path, default=DEFAULT_VOTES)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    scores = read_csv(args.scores)
    pairs = read_csv(args.pairs)
    votes = read_csv(args.votes)
    score_by_uuid: dict[str, dict[str, str]] = {}
    for row in scores:
        image_id = Path(row["image_id"]).stem
        if image_id in score_by_uuid:
            raise ValueError(f"Duplicate score for {image_id}")
        score_by_uuid[image_id] = row

    manifest_ids = [row["uuid"] for row in manifest]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("Duplicate UUID in image manifest")
    if set(manifest_ids) != set(score_by_uuid):
        raise ValueError(
            f"Manifest/score mismatch: missing={len(set(manifest_ids)-set(score_by_uuid))}, "
            f"extra={len(set(score_by_uuid)-set(manifest_ids))}"
        )
    if len(manifest) != 1000:
        raise ValueError(f"Expected 1000 images, found {len(manifest)}")

    errors: list[dict[str, str]] = []
    inventory: list[dict[str, object]] = []
    same_hash: dict[str, list[str]] = defaultdict(list)
    dhashes: dict[str, int] = {}
    for item in manifest:
        image_id = item["uuid"]
        path = args.images / f"{image_id}.jpg"
        record: dict[str, object] = {
            "image_id": image_id,
            "image_path": str(path.resolve()),
            "city": item["city"],
            "country": item.get("country", ""),
            "source": item["source"],
            "sequence_id": item["sequence_id"],
            "grid_cell": item.get("grid_cell", ""),
            "lat": item.get("lat", ""),
            "lon": item.get("lon", ""),
            "pano_status": item.get("pano_status", ""),
            "projection_type": item.get("projection_type", ""),
            "target_visual_pollution_score_0_10": score_by_uuid[image_id]["visual_pollution_score_0_10"],
            "trueskill_mu": score_by_uuid[image_id]["trueskill_mu"],
            "trueskill_sigma": score_by_uuid[image_id]["trueskill_sigma"],
            "image_error": "",
        }
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            record["file_bytes"] = path.stat().st_size
            record["sha256"] = sha256(path)
            same_hash[str(record["sha256"])].append(image_id)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                oriented = ImageOps.exif_transpose(im)
                record["format"] = im.format or ""
                record["width"] = oriented.width
                record["height"] = oriented.height
                record["mode"] = im.mode
                dhashes[image_id] = difference_hash(oriented)
        except Exception as exc:  # Inventory every bad/missing image before stopping.
            record["image_error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"image_id": image_id, "image_path": str(path), "error": record["image_error"]})
        inventory.append(record)

    exact_groups = [ids for ids in same_hash.values() if len(ids) > 1]
    exact_candidates = [
        {"sha256": digest, "image_ids": ";".join(ids)}
        for digest, ids in same_hash.items()
        if len(ids) > 1
    ]
    near_candidates: list[dict[str, object]] = []
    valid_ids = list(dhashes)
    for i, a in enumerate(valid_ids):
        for b in valid_ids[i + 1 :]:
            distance = (dhashes[a] ^ dhashes[b]).bit_count()
            if distance <= 4:
                near_candidates.append({"image_id_a": a, "image_id_b": b, "dhash_distance": distance})
    near_candidates.sort(key=lambda row: (int(row["dhash_distance"]), str(row["image_id_a"]), str(row["image_id_b"])))
    confirmed_near_duplicates = {
        frozenset(("0beb4032-9766-43da-a46c-7ffe61dcec1f", "2cf560a4-7631-4d10-a819-f307fa2d76e7")),
        frozenset(("40e1821e-bf85-45b2-aa66-e1d3a28170ef", "e27078ff-1faf-4f34-8c85-ad410a6cceae")),
        frozenset(("c2ba8e12-fdeb-4459-897d-43e18e76fe8c", "f2facae8-3840-4932-b186-477d79056759")),
    }
    rejected_candidates = {
        frozenset(("67a42126-b63b-4ffc-a8a0-8498546944b3", "678dd26a-09e6-4363-ae72-2271bece7ac0")),
    }
    reviewed_pairs = []
    for row in near_candidates:
        pair = frozenset((str(row["image_id_a"]), str(row["image_id_b"])))
        if pair in confirmed_near_duplicates:
            decision, rationale = "confirmed_duplicate", "Visual review: same street scene and near-identical frame across sources"
        elif pair in rejected_candidates:
            decision, rationale = "false_positive", "Visual review: different cities, scenes, and capture dates"
        else:
            decision, rationale = "pending_review", ""
        reviewed_pairs.append({**row, "decision": decision, "rationale": rationale})

    image_ids = set(manifest_ids)
    degree: Counter[str] = Counter()
    for vote in votes:
        if vote.get("status") == "calibrated" and vote.get("winner"):
            a, b = Path(vote["image_a"]).stem, Path(vote["image_b"]).stem
            if a not in image_ids or b not in image_ids:
                raise ValueError(f"Vote endpoint outside manifest: {a}, {b}")
            degree[a] += 1
            degree[b] += 1

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "image_manifest.csv", inventory, list(inventory[0]))
    write_csv(output / "image_errors.csv", errors, ["image_id", "image_path", "error"])
    write_csv(output / "exact_duplicate_groups.csv", exact_candidates, ["sha256", "image_ids"])
    write_csv(output / "near_duplicate_candidates.csv", near_candidates, ["image_id_a", "image_id_b", "dhash_distance"])
    write_csv(output / "near_duplicate_review.csv", reviewed_pairs, ["image_id_a", "image_id_b", "dhash_distance", "decision", "rationale"])
    near_groups = [
        {"duplicate_group": f"near-{index:03d}", "image_id": image_id}
        for index, pair in enumerate(sorted(confirmed_near_duplicates, key=lambda p: sorted(p)), 1)
        for image_id in sorted(pair)
    ]
    write_csv(output / "near_duplicate_groups.csv", near_groups, ["duplicate_group", "image_id"])

    score_values = [float(row["visual_pollution_score_0_10"]) for row in scores]
    city_counts = Counter(row["city"] for row in manifest)
    source_counts = Counter(row["source"] for row in manifest)
    pano_counts = Counter(row.get("pano_status", "") for row in manifest)
    vote_status = Counter(row.get("status", "") for row in pairs)
    if len(pairs) != 6500 or len(scores) != 1000:
        raise ValueError(f"Unexpected pair/score sizes: {len(pairs)} / {len(scores)}")
    summary = {
        "audit_date": "2026-10-19",
        "expected_images": 1000,
        "manifest_images": len(manifest),
        "unique_image_ids": len(image_ids),
        "scores": len(scores),
        "score_range": [min(score_values), max(score_values)],
        "invalid_or_unreadable_images": len(errors),
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_extra_image_count": sum(len(g) - 1 for g in exact_groups),
        "near_duplicate_candidate_count_dhash_hamming_le_4": len(near_candidates),
        "near_duplicate_confirmed_pair_count": len(confirmed_near_duplicates),
        "near_duplicate_false_positive_count": len(rejected_candidates),
        "near_duplicate_pending_review_count": sum(row["decision"] == "pending_review" for row in reviewed_pairs),
        "cities": dict(sorted(city_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "panorama_flags": dict(sorted(pano_counts.items())),
        "score_target": "visual_pollution_score_0_10; larger means more visual pollution",
        "label_audit_model_prompt": sorted({(r.get("model", ""), r.get("prompt_version", "")) for r in pairs}),
        "pair_audit_status": dict(sorted(vote_status.items())),
        "valid_non_tie_vote_count": sum(1 for r in pairs if r.get("status") == "calibrated" and r.get("winner")),
        "vote_endpoint_degree_min": min(degree.values()) if degree else None,
        "vote_endpoint_degree_max": max(degree.values()) if degree else None,
        "input_sha256": {
            "manifest": sha256(args.manifest),
            "scores": sha256(args.scores),
            "pair_audit": sha256(args.pairs),
            "votes": sha256(args.votes),
        },
        "image_file_count": len(inventory) - len(errors),
    }
    (output / "dataset_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "target_column": "visual_pollution_score_0_10",
        "target_direction": "larger values mean more visual pollution",
        "teacher_model": sorted({r.get("model", "") for r in pairs}),
        "prompt_versions": sorted({r.get("prompt_version", "") for r in pairs}),
        "prompt_sha256": sorted({r.get("prompt_sha256", "") for r in pairs}),
        "comparison_edge_count": len(pairs),
        "valid_non_tie_edges": summary["valid_non_tie_vote_count"],
        "ties": vote_status.get("calibrated_tie", 0),
        "invalid_edges": sum(count for status, count in vote_status.items() if status.startswith("invalid")),
        "derivation": "10 * (1 - minmax(TrueSkill mu)); TrueSkill winner means cleaner, so this target is reversed",
        "scale_scope": "relative to the frozen 1000-image ranking; keep fixed for inference",
        "input_sha256": summary["input_sha256"],
    }
    (output / "label_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(f"Found {len(errors)} missing/unreadable images; see {output / 'image_errors.csv'}")


if __name__ == "__main__":
    main()
