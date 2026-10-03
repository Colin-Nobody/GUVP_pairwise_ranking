#!/usr/bin/env python3
"""Run the pinned DeepLabv3+ ADE20K model on representative local images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmseg.apis import init_model
from mmseg.apis.utils import _preprare_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "segmentation/vendor/mmsegmentation/configs/deeplabv3plus/"
    "deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "segmentation/checkpoints/"
    "deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth"
)
DEFAULT_IMAGES = [
    ROOT / "data/street_images/011464cb-85a8-4216-9b27-9d0f50f89719.jpg",
    ROOT / "data/street_images/0044f495-22e7-42b2-a483-fef7b1e4fa8c.jpg",
    ROOT / "data/street_images/0a8e6990-68df-48df-8f4d-baa1ad07b297.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/segmentation_smoke",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opacity", type=float, default=0.45)
    return parser.parse_args()


def low_memory_inference(model, image_path: Path, original_shape: tuple[int, int]):
    """Run the official test pipeline, restoring the categorical mask on CPU.

    MMSegmentation normally upsamples all 150 logits to the original image size
    on the GPU before argmax. That needs about 6.8 GiB just for a 4032x3024
    image. Here inference and preprocessing are unchanged, but argmax happens at
    the resized test resolution and the integer mask is restored with nearest-
    neighbor interpolation on the CPU.
    """
    data, _ = _preprare_data(str(image_path), model)
    sample = data["data_samples"][0]
    sample.set_metainfo({"ori_shape": sample.metainfo["img_shape"]})
    with torch.no_grad():
        result = model.test_step(data)[0]
    resized_mask = (
        result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.uint8)
    )
    height, width = original_shape
    return cv2.resize(resized_mask, (width, height), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    args = parse_args()
    paths = [p.resolve() for p in args.images]
    for path in [args.config, args.checkpoint, *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    classes = list(model.dataset_meta["classes"])
    palette = np.asarray(model.dataset_meta["palette"], dtype=np.uint8)
    if len(classes) != 150 or palette.shape != (150, 3):
        raise RuntimeError("Expected the 150-class ADE20K metadata.")

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        height, width = image.shape[:2]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        mask = low_memory_inference(model, path, (height, width))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if mask.shape != (height, width):
            raise RuntimeError(
                f"Mask shape {mask.shape} does not match source {(height, width)}"
            )
        if int(mask.max()) >= len(classes):
            raise RuntimeError(f"Unexpected class index {int(mask.max())}")

        image_id = path.stem
        cv2.imwrite(str(masks_dir / f"{image_id}.png"), mask)
        color_rgb = palette[mask]
        color_bgr = color_rgb[..., ::-1]
        overlay = cv2.addWeighted(
            image, 1.0 - args.opacity, color_bgr, args.opacity, 0.0
        )
        cv2.imwrite(str(overlays_dir / f"{image_id}.jpg"), overlay)

        counts = np.bincount(mask.ravel(), minlength=len(classes))
        total = int(mask.size)
        for class_id, (class_name, count) in enumerate(zip(classes, counts)):
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(path.relative_to(ROOT)),
                    "width": width,
                    "height": height,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixel_count": int(count),
                    "pixel_ratio": float(count / total),
                }
            )
        summaries.append(
            {
                "image_id": image_id,
                "image_path": str(path.relative_to(ROOT)),
                "width": width,
                "height": height,
                "mask_width": int(mask.shape[1]),
                "mask_height": int(mask.shape[0]),
                "classes_present": int(np.count_nonzero(counts)),
                "min_class_id": int(mask.min()),
                "max_class_id": int(mask.max()),
                "inference_seconds": elapsed,
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
        print(
            f"OK {path.name}: {width}x{height}, mask={mask.shape[1]}x{mask.shape[0]}, "
            f"classes={np.count_nonzero(counts)}, time={elapsed:.2f}s"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (args.output_dir / "class_counts_and_ratios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "config": str(args.config.resolve().relative_to(ROOT)),
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "postprocess": "argmax at official test resolution; nearest-neighbor categorical mask restore on CPU",
        "classes": classes,
        "palette_rgb": palette.tolist(),
        "images": summaries,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote smoke-test outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
