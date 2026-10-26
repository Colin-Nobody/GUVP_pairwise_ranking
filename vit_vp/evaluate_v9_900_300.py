#!/usr/bin/env python3
"""Evaluate the locked 900-image V9 model on the city-held-out 300-image set."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, require_cuda
from predict_finetuned import load_seed_model, predict_one


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
        "mean_prediction_minus_target": float(np.mean(pred-y)),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    y = np.asarray([float(r["y_true_v9_0_10"]) for r in rows], dtype=np.float64)
    pred = np.asarray([float(r["y_pred_vit_0_10"]) for r in rows], dtype=np.float64)
    return metrics(y, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_v9_holdout_900_300")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = args.experiment_dir
    test = read_csv(root / "test/test_manifest.csv")
    if len(test) != 300 or len({r["image_id"] for r in test}) != 300:
        raise ValueError("Test manifest must contain exactly 300 unique images")
    if len({r["city"] for r in test}) != 6:
        raise ValueError("Test set must contain six completely held-out cities")
    model_dir = root / "deployment/finetuned_last2_v9"
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if checkpoint_sha256(args.checkpoint) != meta["source_checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint hash does not match training metadata")

    device = require_cuda(torch.device(args.device))
    ensemble = []
    for artifact in meta["artifacts"]:
        path = model_dir / artifact["weights"]
        if checkpoint_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Fine-tuned artifact hash mismatch: {path}")
        encoder, head = load_seed_model(args.checkpoint, path, device)
        ensemble.append((encoder, head, artifact))

    predictions = []
    for i, row in enumerate(test, start=1):
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        predictions.append({
            "image_id": row["image_id"], "image_path": str(path), "city": row["city"],
            "country": row["country"], "source": row["source"], "cohort": row["cohort"],
            "split_group": row["split_group"],
            "y_true_v9_0_10": float(row["target_visual_pollution_score_0_10"]),
            "y_pred_vit_0_10": predict_one(path, ensemble, device, meta),
        })
        if i % 25 == 0 or i == len(test):
            print(f"test {i}/{len(test)} predicted", flush=True)

    with (root / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    overall = summarize(predictions)
    group_rows = []
    for group in sorted({r["split_group"] for r in predictions}):
        subset = [r for r in predictions if r["split_group"] == group]
        group_rows.append({"split_group": group, **summarize(subset)})
    city_rows = []
    for city in sorted({r["city"] for r in predictions}):
        subset = [r for r in predictions if r["city"] == city]
        city_rows.append({"city": city, "country": subset[0]["country"], **summarize(subset)})

    for filename, rows in [("test_metrics_by_group.csv", group_rows), ("test_metrics_by_city.csv", city_rows)]:
        with (root / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    audit = json.loads((root / "partition_audit.json").read_text(encoding="utf-8"))
    summary = {
        "protocol": "city-disjoint 900-image fine-tuning; one locked 300-image six-city test",
        "overall": overall,
        "test_group_count": len(group_rows),
        "group_macro_mae": float(statistics.mean(r["mae"] for r in group_rows)),
        "group_macro_spearman": float(statistics.mean(r["spearman"] for r in group_rows)),
        "test_city_count": len(city_rows),
        "city_macro_mae": float(statistics.mean(r["mae"] for r in city_rows)),
        "city_macro_spearman": float(statistics.mean(r["spearman"] for r in city_rows)),
        "group_metrics": group_rows,
        "city_metrics": city_rows,
        "test_used_for_training_selection_or_early_stopping": False,
        "test_set_previously_evaluated_by_same_configuration": True,
        "interpretation": "This is a city-held-out evaluation for this run, but the same 300-image set was reported in an earlier same-configuration run; treat as a reused test set, not a new blind confirmation.",
        "label_graph_caveat": audit["interpretation_note"],
        "ordinary_image_level_p_values_reported": False,
    }
    (root / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# V9 DINOv3 ViT: city-disjoint 900/300 evaluation", "",
        "## Protocol", "",
        "The model starts from the original DINOv3 ViT-B/16 LVD-1689M pretrained checkpoint; no previous fine-tuned V6/V9 weights are loaded. The final two Transformer blocks and patch-mean MLP head are fine-tuned. Six entire cities (the four external cities plus two original-cohort cities selected by fixed seed 20261020) form a 300-image test set. The remaining 18 original cities (900 images) are used for training.", "",
        "The configuration and 16-epoch duration are fixed in advance. There is no hyperparameter selection or early stopping on the test set. Three seed models (42, 137, 2026), each reset from the same original pretrained checkpoint, are averaged for the test prediction. V9 targets were generated and globally min-max scaled from the existing joint 1,200-image pairwise graph; no test-specific recalibration is applied.", "",
        "## Overall test metrics", "",
        f"- N: {overall['n']}", f"- MAE: {overall['mae']:.4f}", f"- RMSE: {overall['rmse']:.4f}",
        f"- R²: {overall['r2']:.4f}", f"- Spearman rho: {overall['spearman']:.4f}",
        f"- Pearson r: {overall['pearson']:.4f}", f"- Mean prediction minus target: {overall['mean_prediction_minus_target']:.4f}", "",
        "See test_metrics_by_group.csv for external-vs-original-city test breakdown and test_metrics_by_city.csv for each city.", "",
        "## Interpretation limits", "",
        "The 300-image set was not read by this run's training, model-selection, or early-stopping code. However, it was already evaluated and reported in an earlier same-configuration run, so this result is a repeated evaluation on a reused test set, not a new blind confirmation.", "",
        audit["interpretation_note"],
        "These scores evaluate image-level and city-held-out generalization to the fixed V9 targets. They are not independent human ratings. The four external cities are all KartaView, while most training images are Mapillary, so the external subgroup also includes a source shift.", "",
        "No ordinary image-independent p-values are reported because V9 labels come from a shared pairwise comparison graph and images are clustered by city.",
    ]
    (root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
