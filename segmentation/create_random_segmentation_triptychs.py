#!/usr/bin/env python3
"""Create reproducible original/mask/overlay triptychs from completed masks."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def fit_to_canvas(image: np.ndarray, width: int, height: int, nearest: bool = False) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w, new_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    resized = cv2.resize(
        image, (new_w, new_h),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    y, x = (height - new_h) // 2, (width - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def build_element_legend(
    counts: np.ndarray, classes: list[str], palette_bgr: np.ndarray, width: int
) -> np.ndarray:
    """Render the colour/name/percentage legend for classes present in one mask."""
    present = np.flatnonzero(counts)
    columns, row_height = 4, 30
    heading_height, padding = 38, 12
    rows = max(1, int(np.ceil(len(present) / columns)))
    legend = np.full(
        (heading_height + rows * row_height + padding, width, 3), 245, dtype=np.uint8
    )
    cv2.putText(
        legend,
        "Legend: colour = predicted Mapillary element (percentage of all image pixels)",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    total = int(counts.sum())
    column_width = width // columns
    for index, class_id in enumerate(present):
        row, column = divmod(index, columns)
        x = 16 + column * column_width
        y = heading_height + 20 + row * row_height
        colour = tuple(int(value) for value in palette_bgr[class_id])
        cv2.rectangle(legend, (x, y - 14), (x + 16, y + 2), colour, thickness=-1)
        cv2.rectangle(legend, (x, y - 14), (x + 16, y + 2), (90, 90, 90), thickness=1)
        label = f"{classes[class_id]} {counts[class_id] / total * 100:.2f}%"
        cv2.putText(
            legend,
            label,
            (x + 23, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    return legend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("/home/colin/GUVP_pairwise_ranking"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Reuse image IDs from a prior selected_30.csv instead of drawing a new sample.",
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--panel-width", type=int, default=720)
    parser.add_argument("--panel-height", type=int, default=500)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo_dir = args.repo_dir.resolve()
    output_dir = args.output_dir.resolve()
    triptychs_dir = output_dir / "triptychs"
    triptychs_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest or run_dir / "segmentation_manifest.csv")
    masks_dir = run_dir / "baseline/masks" if (run_dir / "baseline/masks").is_dir() else run_dir / "sliding_512_o25/masks"
    available = manifest[manifest["image_id"].map(lambda value: (masks_dir / f"{value}.png").is_file())]
    if len(available) < args.count:
        raise SystemExit(f"Only {len(available)} masks are available; requested {args.count}")
    if args.selection_csv is not None:
        selection = pd.read_csv(args.selection_csv)
        if len(selection) != args.count:
            raise SystemExit(f"Selection contains {len(selection)} rows; expected {args.count}")
        sample = selection[["sample_number", "image_id"]].merge(
            available, on="image_id", how="left", validate="one_to_one")
        if sample["image_path"].isnull().any():
            raise SystemExit("At least one reused image ID is absent from the available masks")
        sample = sample.sort_values("sample_number").reset_index(drop=True)
    else:
        selected = random.Random(args.seed).sample(list(available.index), args.count)
        sample = available.loc[selected].copy().reset_index(drop=True)
        sample.insert(0, "sample_number", range(1, len(sample) + 1))
    config = json.loads((run_dir / "run_config.json").read_text())
    palette_bgr = np.asarray(config["palette_rgb"], dtype=np.uint8)[:, ::-1]
    title_h, gap = 42, 8
    records: list[dict[str, object]] = []

    for index, row in sample.iterrows():
        image_id = str(row["image_id"])
        original = cv2.imread(str(repo_dir / row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks_dir / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
        if original is None or mask is None:
            raise RuntimeError(f"Could not read original or mask for {image_id}")
        if original.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Original/mask size mismatch for {image_id}")
        colored_mask = palette_bgr[np.clip(mask, 0, len(palette_bgr) - 1)]
        overlay = cv2.addWeighted(original, 1.0 - args.opacity, colored_mask, args.opacity, 0)
        panels = [
            fit_to_canvas(original, args.panel_width, args.panel_height),
            fit_to_canvas(colored_mask, args.panel_width, args.panel_height, nearest=True),
            fit_to_canvas(overlay, args.panel_width, args.panel_height),
        ]
        body = cv2.hconcat(panels)
        header = np.full((title_h, body.shape[1], 3), 245, dtype=np.uint8)
        labels = ["Original", "Colour semantic mask", f"Overlay ({args.opacity:.0%})"]
        for panel, label in enumerate(labels):
            origin_x = panel * args.panel_width + 18
            cv2.putText(header, label, (origin_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        counts = np.bincount(mask.ravel(), minlength=len(config["classes"]))
        legend = build_element_legend(counts, config["classes"], palette_bgr, body.shape[1])
        image = cv2.vconcat([header, body, legend])
        filename = f"{int(row['sample_number']):02d}_{image_id}.jpg"
        destination = triptychs_dir / filename
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 93])
        records.append({
            "sample_number": int(row["sample_number"]),
            "image_id": image_id,
            "city": row.get("city", ""),
            "source_image": row["image_path"],
            "triptych": str(destination.relative_to(output_dir)),
        })

    sample_csv = output_dir / "selected_30.csv"
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Created {len(records)} triptychs in {triptychs_dir}")
    print(f"Selection manifest: {sample_csv}")


if __name__ == "__main__":
    main()
