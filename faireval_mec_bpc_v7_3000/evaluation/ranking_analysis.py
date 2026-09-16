"""Summary statistics and plots for pairwise-ranking outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


def write_evaluation(scores: pd.DataFrame, output_dir: Path, vote_count: int, dimension: str) -> None:
    """Write reproducible summary statistics and two diagnostic charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    q = scores["q_score_0_10"]
    ts = scores["trueskill_preference_0_1"]
    if len(scores) >= 2 and q.nunique() > 1 and ts.nunique() > 1:
        pearson = pearsonr(q, ts)
        spearman = spearmanr(q, ts)
        correlations = {
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
        }
    else:
        correlations = {"pearson_r": None, "pearson_p_value": None, "spearman_rho": None, "spearman_p_value": None}

    summary = {
        "dimension": dimension,
        "valid_vote_rows": vote_count,
        "image_count": int(len(scores)),
        "comparisons_per_image": {
            "minimum": int(scores["comparison_count"].min()),
            "median": float(scores["comparison_count"].median()),
            "maximum": int(scores["comparison_count"].max()),
        },
        "q_score_vs_trueskill": correlations,
    }
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, column, title in zip(
        axes,
        ["q_score_0_10", "trueskill_preference_0_1", "visual_pollution_score_0_1"],
        ["Q-score (0–10)", "TrueSkill preference (0–1)", "Visual pollution (0–1)"],
    ):
        axis.hist(scores[column], bins=min(20, max(3, len(scores))), color="#3578a8", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel("Score")
        axis.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(q, ts, color="#3578a8")
    axis.set_xlabel("Q-score (0–10)")
    axis.set_ylabel("TrueSkill preference (0–1)")
    axis.set_title("Agreement between ranking methods")
    fig.tight_layout()
    fig.savefig(output_dir / "score_comparison.png", dpi=180)
    plt.close(fig)

def write_ground_truth_validation(scores: pd.DataFrame, ground_truth_path: Path, output_dir: Path) -> None:
    """Write correlations and rank errors against a known VP ground truth."""
    truth = pd.read_csv(ground_truth_path)
    if "image_id" not in truth.columns:
        raise ValueError("Ground-truth CSV must contain an image_id column")
    if truth["image_id"].duplicated().any():
        raise ValueError("Ground-truth CSV contains duplicate image_id values")
    has_score = "true_visual_pollution_score_0_1" in truth.columns
    has_rank = "true_visual_pollution_rank" in truth.columns
    if not (has_score or has_rank):
        raise ValueError("Ground-truth CSV must contain true_visual_pollution_score_0_1 or true_visual_pollution_rank")

    validation = scores.merge(truth, on="image_id", how="inner", validate="one_to_one")
    if len(validation) < 2:
        raise ValueError("At least two overlapping image_id values are required for ground-truth validation")
    if not has_score:
        ranks = validation["true_visual_pollution_rank"].astype(float)
        validation["true_visual_pollution_score_0_1"] = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    if not has_rank:
        validation["true_visual_pollution_rank"] = validation["true_visual_pollution_score_0_1"].rank(method="average")

    validation["expected_rank_by_visual_pollution"] = len(validation) + 1 - validation["true_visual_pollution_rank"]
    validation["rank_error"] = validation["rank_by_visual_pollution"] - validation["expected_rank_by_visual_pollution"]
    validation["absolute_rank_error"] = validation["rank_error"].abs()
    validation = validation.sort_values("rank_by_visual_pollution").round(6)
    validation.to_csv(output_dir / "ground_truth_validation.csv", index=False)
    estimated = validation["visual_pollution_score_0_1"]
    expected = validation["true_visual_pollution_score_0_1"]
    pearson = pearsonr(estimated, expected)
    spearman = spearmanr(estimated, expected)
    kendall = kendalltau(estimated, expected)
    summary = {
        "ground_truth_file": str(ground_truth_path),
        "overlapping_images": int(len(validation)),
        "missing_from_ground_truth": int(len(scores) - len(validation)),
        "missing_from_scores": int(len(truth) - len(validation)),
        "score_correlations": {
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
            "kendall_tau": float(kendall.statistic),
            "kendall_p_value": float(kendall.pvalue),
        },
        "rank_error": {
            "mean_absolute_error": float(validation["absolute_rank_error"].mean()),
            "root_mean_squared_error": float((validation["rank_error"] ** 2).mean() ** 0.5),
            "maximum_absolute_error": float(validation["absolute_rank_error"].max()),
            "exact_rank_match_count": int((validation["rank_error"] == 0).sum()),
        },
    }
    (output_dir / "ground_truth_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
