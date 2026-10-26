#!/usr/bin/env python3
"""Normalize the independent Qwen-V6 and frozen ViT scores and report association."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/external_vp_4city_200"
DEFAULT_VLM = ROOT / "output/vlm_external_4city_200_v6/perception_scores.csv"
DEFAULT_VIT = ROOT / "output/external_vp_4city_200/vit_predictions.csv"
DEFAULT_OUT = ROOT / "output/external_vp_4city_200"


def normalize_0_10(values: pd.Series) -> pd.Series:
    low, high = float(values.min()), float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Cannot min-max normalize constant or invalid scores: min={low}, max={high}")
    return (values.astype(float) - low) / (high - low) * 10.0


def metrics(frame: pd.DataFrame, x_col: str, y_col: str, scope: str) -> dict:
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    return {
        "scope": scope,
        "n_images": len(frame),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
        "mae_on_reported_scale": float(np.mean(np.abs(x - y))),
        "rmse_on_reported_scale": float(np.sqrt(np.mean((x - y) ** 2))),
        "mean_vit_minus_qwen_on_reported_scale": float(np.mean(y - x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "images_manifest.csv")
    parser.add_argument("--qwen-scores", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--vit-predictions", type=Path, default=DEFAULT_VIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    qwen = pd.read_csv(args.qwen_scores)
    vit = pd.read_csv(args.vit_predictions)
    for frame, name in ((manifest, "manifest"), (qwen, "Qwen scores"), (vit, "ViT predictions")):
        if "image_id" not in frame and "uuid" not in frame:
            raise ValueError(f"{name} has no image_id/uuid key")
        if "image_id" not in frame:
            frame["image_id"] = frame.uuid.astype(str)
        frame["image_id"] = frame.image_id.astype(str).str.removesuffix(".jpg")
        if frame.image_id.duplicated().any():
            raise ValueError(f"{name} contains duplicate image IDs")
    qwen_col = "visual_pollution_score_0_10"
    vit_col = "predicted_vp_score"
    if qwen_col not in qwen or vit_col not in vit:
        raise ValueError(f"Need Qwen `{qwen_col}` and ViT `{vit_col}` score columns")

    merged = manifest[["image_id", "city", "source"]].merge(
        qwen[["image_id", qwen_col]], on="image_id", how="inner", validate="one_to_one"
    ).merge(vit[["image_id", vit_col]], on="image_id", how="inner", validate="one_to_one")
    if len(merged) != 200 or set(merged.image_id) != set(manifest.image_id):
        raise ValueError(f"Expected full one-to-one coverage of 200 images; got {len(merged)}")
    merged["qwen_minmax_0_10"] = normalize_0_10(merged[qwen_col])
    merged["vit_minmax_0_10"] = normalize_0_10(merged[vit_col])
    merged["qwen_zscore"] = (merged[qwen_col] - merged[qwen_col].mean()) / merged[qwen_col].std(ddof=0)
    merged["vit_zscore"] = (merged[vit_col] - merged[vit_col].mean()) / merged[vit_col].std(ddof=0)

    rows = [metrics(merged, "qwen_minmax_0_10", "vit_minmax_0_10", "all_200_images")]
    for city, group in merged.groupby("city", sort=True):
        rows.append(metrics(group, "qwen_minmax_0_10", "vit_minmax_0_10", f"city:{city}"))
    centered = merged.copy()
    centered["qwen_centered"] = centered.qwen_minmax_0_10 - centered.groupby("city").qwen_minmax_0_10.transform("mean")
    centered["vit_centered"] = centered.vit_minmax_0_10 - centered.groupby("city").vit_minmax_0_10.transform("mean")
    rows.append(metrics(centered, "qwen_centered", "vit_centered", "within_city_centered_pooled"))
    metrics_frame = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "external_qwen_vit_scores.csv", index=False)
    metrics_frame.to_csv(args.output_dir / "external_correlations.csv", index=False)
    metadata = {
        "n_images": len(merged),
        "cities": {str(city): int(n) for city, n in merged.groupby("city").size().items()},
        "source_counts": {str(source): int(n) for source, n in merged.groupby("source").size().items()},
        "qwen_target": "Qwen2.5-VL-7B-Instruct, vp-faireval-mec-bpc-score-v6, TrueSkill then min-max to 0-10 on this 200-image cohort",
        "vit_model": "Three-seed DINOv3 ViT-B/16 last-two-block fine-tuned ensemble, trained only on original 1,000 images",
        "normalization": "Separately min-max scale Qwen and ViT cohort predictions to [0,10]; positive affine scaling does not change Pearson or Spearman correlation.",
        "interpretation": "Association of independent Qwen V6 labels with external-city ViT predictions; separate cohort scaling evaluates association/ranking, not absolute score calibration.",
        "dependence_caveat": "These Qwen scores are derived from a shared pairwise comparison graph, and images are nested in only four cities; report correlations descriptively and do not interpret ordinary image-independent p-values as city-generalization evidence.",
        "outputs": ["external_qwen_vit_scores.csv", "external_correlations.csv"],
    }
    (args.output_dir / "external_analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(metrics_frame.to_string(index=False))
    print(f"Saved analysis under {args.output_dir}")


if __name__ == "__main__":
    main()
