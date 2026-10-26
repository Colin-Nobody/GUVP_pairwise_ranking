#!/usr/bin/env python3
"""Evaluate a finalized V9-trained ViT ensemble on the untouched 200-image holdout."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, require_cuda
from predict_finetuned import load_seed_model, predict_one


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
        "mean_prediction_minus_target": float(np.mean(pred-y)),
        "prediction_below_0_count": int(np.sum(pred < 0)),
        "prediction_above_10_count": int(np.sum(pred > 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_v9_holdout_1000_200")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.experiment_dir
    holdout = read_csv(root / "holdout/holdout_manifest.csv")
    if len(holdout) != 200 or len({row["image_id"] for row in holdout}) != 200:
        raise ValueError("Holdout must contain exactly 200 unique images")
    if {row["cohort"] for row in holdout} != {"external_4city_200"}:
        raise ValueError("Only the sealed external_4city_200 holdout can be evaluated here")

    model_dir = root / "deployment/finetuned_last2_v9"
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if checkpoint_sha256(args.checkpoint) != meta["source_checkpoint_sha256"]:
        raise ValueError("Pretrained DINOv3 checkpoint hash does not match the fitted model metadata")
    device = require_cuda(torch.device(args.device))
    ensemble = []
    for item in meta["artifacts"]:
        weights = model_dir / item["weights"]
        if checkpoint_sha256(weights) != item["sha256"]:
            raise ValueError(f"Fine-tuned weight hash mismatch: {weights}")
        encoder, head = load_seed_model(args.checkpoint, weights, device)
        ensemble.append((encoder, head, item))

    prediction_rows = []
    for index, row in enumerate(holdout, start=1):
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        prediction_rows.append({
            "image_id": row["image_id"], "image_path": str(path), "city": row["city"],
            "country": row["country"], "source": row["source"],
            "y_true_v9_0_10": float(row["target_visual_pollution_score_0_10"]),
            "y_pred_vit_0_10": predict_one(path, ensemble, device, meta),
        })
        if index % 25 == 0 or index == len(holdout):
            print(f"holdout {index}/{len(holdout)} predicted", flush=True)

    out_path = root / "holdout_predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    y = np.asarray([row["y_true_v9_0_10"] for row in prediction_rows], dtype=np.float64)
    pred = np.asarray([row["y_pred_vit_0_10"] for row in prediction_rows], dtype=np.float64)
    overall = metrics(y, pred)
    clipped = np.clip(pred, 0, 10)
    overall["clipped_0_10_mae_diagnostic"] = float(mean_absolute_error(y, clipped))
    overall["clipped_0_10_rmse_diagnostic"] = float(np.sqrt(mean_squared_error(y, clipped)))

    city_rows = []
    for city in sorted({row["city"] for row in prediction_rows}):
        rows = [row for row in prediction_rows if row["city"] == city]
        cy = np.asarray([row["y_true_v9_0_10"] for row in rows], dtype=np.float64)
        cp = np.asarray([row["y_pred_vit_0_10"] for row in rows], dtype=np.float64)
        city_rows.append({"city": city, "country": rows[0]["country"], **metrics(cy, cp)})
    with (root / "holdout_metrics_by_city.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(city_rows[0]))
        writer.writeheader()
        writer.writerows(city_rows)

    audit = json.loads((root / "partition_audit.json").read_text(encoding="utf-8"))
    summary = {
        "evaluation": "sealed 200-image holdout after fixed model selection and final fit on the original 1,000 images",
        "overall": overall,
        "city_count": len(city_rows),
        "city_macro_mae": float(statistics.mean(row["mae"] for row in city_rows)),
        "city_macro_spearman": float(statistics.mean(row["spearman"] for row in city_rows)),
        "city_metrics": city_rows,
        "target_label_version": meta["label_version"],
        "label_graph_caveat": audit["strict_label_isolation_note"],
        "holdout_used_for_training_selection_or_early_stopping": False,
        "ordinary_image_level_p_values_reported": False,
    }
    (root / "holdout_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    oof_path = root / "internal_oof_metrics_by_repeat.csv"
    report = [
        "# DINOv3 trained on V9 labels: 1,000 development images / 200-image holdout", "",
        "## Protocol", "",
        "The ViT starts from the original DINOv3 ViT-B/16 LVD-1689M pretrained checkpoint. No previously fine-tuned V6 weights are loaded. Only the final two Transformer blocks and the patch-mean MLP regressor are trained. The last 200 images were excluded from training, internal split selection, and early stopping. Hyperparameters were locked to the existing selected recipe: 512-pixel long edge, backbone LR 1e-5, head LR 1e-4, weight decay 0.01, 16 epochs, and seeds 42/137/2026.", "",
        "The first 1,000 images have 3-repeat, 5-fold city-held-out OOF evaluation. The final 3-seed ensemble is fitted on all 1,000 development images and evaluated once on the four-city 200-image holdout. Predictions are evaluated on the fixed V9 0-10 target scale without holdout recalibration.", "",
        "## Internal development OOF", "",
        f"Metrics by repeat are in {oof_path.name}.", "",
        "## Sealed 200-image holdout", "",
        f"- MAE: {overall['mae']:.4f}", f"- RMSE: {overall['rmse']:.4f}", f"- R2: {overall['r2']:.4f}",
        f"- Spearman rho: {overall['spearman']:.4f}", f"- Pearson r: {overall['pearson']:.4f}",
        f"- Mean prediction minus target: {overall['mean_prediction_minus_target']:.4f}",
        f"- City macro MAE: {summary['city_macro_mae']:.4f}", f"- City macro Spearman rho: {summary['city_macro_spearman']:.4f}", "",
        "City-by-city results are in holdout_metrics_by_city.csv. Unclipped predictions are the primary metrics; clipped metrics are a display-range diagnostic only.", "",
        "## Interpretation limit", "",
        audit["strict_label_isolation_note"],
        "The holdout images and their scores were never read by the ViT fitting script. However, V9's labels were produced from one joint comparison graph, so the existing train and holdout targets share cross-partition comparison information. This evaluates image-level holdout generalization to the fixed V9 labels, not an independently generated label set or human perception.", "",
        "No independent-image p-values are reported because target labels are pairwise-graph-derived and dependent."
    ]
    (root / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
