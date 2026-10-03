#!/usr/bin/env python3
"""Run baseline and overlapping-logit sliding inference on the audit set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mmcv
import mmengine
import mmseg
import numpy as np
import pandas as pd
import torch
from mmseg.apis import inference_model, init_model
from mmseg.apis.utils import _preprare_data


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "segmentation/vendor/mmsegmentation/configs/deeplabv3plus/"
    "deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py"
)
CHECKPOINT = (
    ROOT
    / "segmentation/checkpoints/"
    "deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth"
)
TARGET_CLASSES = [
    "building",
    "wall",
    "road",
    "sidewalk",
    "tree",
    "grass",
    "plant",
    "sky",
    "car",
    "bus",
    "truck",
    "person",
    "fence",
    "signboard",
    "poster",
    "pole",
    "streetlight",
    "awning",
    "bench",
    "ashcan",
]
MAPILLARY_TARGET_CLASSES = [
    "Building",
    "Wall",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Terrain",
    "Sky",
    "Car",
    "Bus",
    "Truck",
    "Person",
    "Fence",
    "Signage - Advertisement",
    "Signage - Store",
    "Signage - Information",
    "Pole",
    "Utility Pole",
    "Street Light",
    "Bench",
    "Trash Can",
    "Pothole",
    "Temporary Barrier",
    "Traffic Cone",
    "Traffic Sign (Front)",
    "Traffic Light - General (Single)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "output/segmentation_audit/audit_manifest.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output/segmentation_audit"
    )
    parser.add_argument("--mode", choices=["baseline", "sliding", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--expected-images", type=int, default=30)
    return parser.parse_args()


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def baseline_inference(model, image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Official preprocessing/inference, with categorical restore on CPU."""
    data, _ = _preprare_data(str(image_path), model)
    sample = data["data_samples"][0]
    sample.set_metainfo({"ori_shape": sample.metainfo["img_shape"]})
    with torch.no_grad():
        result = model.test_step(data)[0]
    small_mask = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.uint8)
    height, width = shape
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    del result, data, small_mask
    return mask


def sliding_logit_inference(
    model, image: np.ndarray, tile_size: int, overlap: float, num_classes: int
) -> tuple[np.ndarray, int]:
    """Merge overlapping 512px tiles by equal-weight logit averaging.

    Only an active horizontal band of logits is held in RAM. Rows that can no
    longer receive contributions from future tiles are immediately finalized.
    Float16 accumulation keeps the maximum 4032x3024 case below 0.7 GiB RAM.
    """
    height, width = image.shape[:2]
    stride = int(round(tile_size * (1.0 - overlap)))
    if stride <= 0:
        raise ValueError("overlap must be less than 1")
    ys = tile_positions(height, tile_size, stride)
    xs = tile_positions(width, tile_size, stride)
    output = np.zeros((height, width), dtype=np.uint8)
    base_y = 0
    accum = np.zeros((num_classes, 0, width), dtype=np.float16)
    coverage = np.zeros((0, width), dtype=np.uint8)
    tile_count = 0

    for y in ys:
        if y > base_y:
            finalized = y - base_y
            if np.any(coverage[:finalized] == 0):
                raise RuntimeError("Sliding inference left uncovered pixels")
            output[base_y:y] = np.argmax(accum[:, :finalized], axis=0).astype(np.uint8)
            accum = accum[:, finalized:, :]
            coverage = coverage[finalized:, :]
            base_y = y

        band_end = min(y + tile_size, height)
        needed = band_end - base_y
        if needed > accum.shape[1]:
            extra = needed - accum.shape[1]
            accum = np.pad(accum, ((0, 0), (0, extra), (0, 0)))
            coverage = np.pad(coverage, ((0, extra), (0, 0)))

        for x in xs:
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            with torch.no_grad():
                result = inference_model(model, tile)
            logits = result.seg_logits.data.detach().cpu().numpy().astype(np.float16)
            tile_h, tile_w = tile.shape[:2]
            if logits.shape != (num_classes, tile_h, tile_w):
                raise RuntimeError(
                    f"Unexpected tile logits {logits.shape}; expected {(num_classes, tile_h, tile_w)}"
                )
            offset_y = y - base_y
            accum[:, offset_y : offset_y + tile_h, x : x + tile_w] += logits
            coverage[offset_y : offset_y + tile_h, x : x + tile_w] += 1
            tile_count += 1
            del result, logits

    if np.any(coverage == 0):
        raise RuntimeError("Sliding inference left uncovered pixels at finalization")
    output[base_y:height] = np.argmax(accum[:, : height - base_y], axis=0).astype(np.uint8)
    return output, tile_count


def save_prediction(
    mode_dir: Path,
    row: pd.Series,
    image: np.ndarray,
    mask: np.ndarray,
    classes: list[str],
    palette: np.ndarray,
    opacity: float,
    elapsed: float,
    peak_mib: float,
    tile_count: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    image_id = row["image_id"]
    (mode_dir / "masks").mkdir(parents=True, exist_ok=True)
    (mode_dir / "overlays").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mode_dir / "masks" / f"{image_id}.png"), mask)
    overlay = cv2.addWeighted(image, 1.0 - opacity, palette[mask][..., ::-1], opacity, 0)
    cv2.imwrite(str(mode_dir / "overlays" / f"{image_id}.jpg"), overlay)

    counts = np.bincount(mask.ravel(), minlength=len(classes))
    total = int(mask.size)
    long_rows: list[dict[str, object]] = []
    for class_id, (class_name, count) in enumerate(zip(classes, counts)):
        long_rows.append(
            {
                "image_id": image_id,
                "city": row["city"],
                "vp_stratum": row["vp_stratum"],
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": int(count),
                "pixel_ratio": float(count / total),
            }
        )
    summary: dict[str, object] = {
        "image_id": image_id,
        "city": row["city"],
        "vp_stratum": row["vp_stratum"],
        "vp_score_v8": float(row["vp_score_v8"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "classes_present": int(np.count_nonzero(counts)),
        "inference_seconds": elapsed,
        "peak_gpu_memory_mib": peak_mib,
        "tile_count": tile_count,
    }
    summary.update({f"ratio__{name}": float(counts[i] / total) for i, name in enumerate(classes)})
    return long_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_mode(
    mode: str,
    model,
    manifest: pd.DataFrame,
    output_dir: Path,
    classes: list[str],
    palette: np.ndarray,
    args: argparse.Namespace,
) -> None:
    mode_dir = output_dir / mode
    long_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for index, row in manifest.iterrows():
        path = ROOT / row["image_path"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        expected_shape = (int(row["height"]), int(row["width"]))
        if image.shape[:2] != expected_shape:
            raise RuntimeError(f"Manifest/image shape mismatch for {path}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if mode == "baseline":
            mask = baseline_inference(model, path, expected_shape)
            tile_count = None
        else:
            mask, tile_count = sliding_logit_inference(
                model, image, args.tile_size, args.overlap, len(classes)
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        if mask.shape != expected_shape or int(mask.max()) >= len(classes):
            raise RuntimeError(f"Invalid mask for {path}: {mask.shape}, max={mask.max()}")
        rows, summary = save_prediction(
            mode_dir,
            row,
            image,
            mask,
            classes,
            palette,
            args.opacity,
            elapsed,
            peak_mib,
            tile_count,
        )
        long_rows.extend(rows)
        summaries.append(summary)
        print(
            f"[{mode} {index + 1:02d}/{len(manifest)}] {row['city']} {row['image_id']} "
            f"{expected_shape[1]}x{expected_shape[0]} {elapsed:.2f}s"
            + (f" {tile_count} tiles" if tile_count is not None else ""),
            flush=True,
        )
        del image, mask
        torch.cuda.empty_cache()
    write_csv(mode_dir / "class_counts_and_ratios.csv", long_rows)
    write_csv(mode_dir / "semantic_features_raw.csv", summaries)


def compare_modes(manifest: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        image_id = row["image_id"]
        baseline = cv2.imread(
            str(output_dir / "baseline/masks" / f"{image_id}.png"), cv2.IMREAD_UNCHANGED
        )
        sliding = cv2.imread(
            str(output_dir / "sliding_512_o25/masks" / f"{image_id}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if baseline is None or sliding is None or baseline.shape != sliding.shape:
            raise RuntimeError(f"Missing or mismatched comparison masks for {image_id}")
        union_classes = np.union1d(np.unique(baseline), np.unique(sliding))
        ious = []
        for class_id in union_classes:
            b = baseline == class_id
            s = sliding == class_id
            union = np.count_nonzero(b | s)
            if union:
                ious.append(np.count_nonzero(b & s) / union)
        rows.append(
            {
                "image_id": image_id,
                "city": row["city"],
                "vp_stratum": row["vp_stratum"],
                "pixel_agreement": float(np.mean(baseline == sliding)),
                "mean_iou_between_methods": float(np.mean(ious)),
                "classes_in_union": int(len(union_classes)),
                "baseline_classes": int(len(np.unique(baseline))),
                "sliding_classes": int(len(np.unique(sliding))),
            }
        )
    write_csv(output_dir / "baseline_vs_sliding.csv", rows)


def create_rating_template(
    manifest: pd.DataFrame, output_dir: Path, target_classes: list[str]
) -> None:
    rows = []
    for _, row in manifest.iterrows():
        for class_name in target_classes:
            rows.append(
                {
                    "audit_order": int(row["audit_order"]),
                    "image_id": row["image_id"],
                    "city": row["city"],
                    "vp_stratum": row["vp_stratum"],
                    "class_name": class_name,
                    "baseline_rating_0_1_2_or_NA": "",
                    "sliding_rating_0_1_2_or_NA": "",
                    "preferred_method": "",
                    "failure_modes": "",
                    "notes": "",
                }
            )
    write_csv(output_dir / "audit_ratings.csv", rows)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest).sort_values("audit_order").reset_index(drop=True)
    if len(manifest) != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} manifest images, got {len(manifest)}"
        )
    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    classes = list(model.dataset_meta["classes"])
    palette = np.asarray(model.dataset_meta["palette"], dtype=np.uint8)
    target_candidates = (
        MAPILLARY_TARGET_CLASSES
        if "Signage - Advertisement" in classes
        else TARGET_CLASSES
    )
    target_classes = [name for name in target_candidates if name in classes]
    modes = ["baseline", "sliding_512_o25"] if args.mode == "both" else [args.mode]
    if args.mode == "sliding":
        modes = ["sliding_512_o25"]
    for mode in modes:
        run_mode(mode, model, manifest, args.output_dir, classes, palette, args)
    if (args.output_dir / "baseline/masks").is_dir() and (
        args.output_dir / "sliding_512_o25/masks"
    ).is_dir():
        compare_modes(manifest, args.output_dir)
    create_rating_template(manifest, args.output_dir, target_classes)
    config = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "torch": torch.__version__,
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "mmsegmentation": mmseg.__version__,
        "classes": classes,
        "palette_rgb": palette.tolist(),
        "target_classes": target_classes,
        "baseline_postprocess": "argmax at official test resolution; CPU nearest-neighbor categorical restore",
        "sliding": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": int(round(args.tile_size * (1.0 - args.overlap))),
            "merge": "equal-weight average of overlapping logits; argmax after accumulation",
            "accumulator_dtype": "float16",
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audit inference complete: {args.output_dir}")


if __name__ == "__main__":
    main()
