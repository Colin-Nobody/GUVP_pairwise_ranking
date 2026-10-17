#!/usr/bin/env python3
"""Analyze agreement between two runs over the same image collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


SCORE = "visual_pollution_score_0_10"
RANK = "rank_by_visual_pollution"
SUCCESS = {"calibrated", "calibrated_tie"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--scores-a", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--audit-b", type=Path, required=True)
    parser.add_argument("--scores-b", type=Path, required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def load(path: Path, columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    p = pearsonr(x, y)
    s = spearmanr(x, y)
    k = kendalltau(x, y)
    return {
        "pearson_r": float(p.statistic),
        "pearson_p": float(p.pvalue),
        "spearman_rho": float(s.statistic),
        "spearman_p": float(s.pvalue),
        "kendall_tau": float(k.statistic),
        "kendall_p": float(k.pvalue),
    }


def bootstrap_corr(x: np.ndarray, y: np.ndarray, repetitions: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(20260911)
    pearsons = np.empty(repetitions)
    spearmans = np.empty(repetitions)
    n = len(x)
    for i in range(repetitions):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        pearsons[i] = np.corrcoef(xb, yb)[0, 1]
        spearmans[i] = np.corrcoef(rankdata(xb), rankdata(yb))[0, 1]
    return {
        "pearson_r_95_ci": np.quantile(pearsons, [0.025, 0.975]).tolist(),
        "spearman_rho_95_ci": np.quantile(spearmans, [0.025, 0.975]).tolist(),
    }


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * covariance / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def cohens_kappa(a: pd.Series, b: pd.Series, quadratic: bool = False) -> float:
    """Compute nominal or quadratic-weighted Cohen's kappa."""
    labels = sorted(set(a.dropna()) | set(b.dropna()))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for left, right in zip(a, b):
        matrix[index[left], index[right]] += 1
    observed = matrix / matrix.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if quadratic:
        denominator = max(len(labels) - 1, 1) ** 2
        weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / denominator, matrix.shape)
    else:
        weights = 1 - np.eye(len(labels))
    expected_disagreement = float((weights * expected).sum())
    return float(1 - (weights * observed).sum() / expected_disagreement)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return [float(center - half_width), float(center + half_width)]


def prepare_audit(path: Path, label: str) -> pd.DataFrame:
    required = {
        "image_a", "image_b", "winner", "status", "calibrated_score_a",
        "calibrated_score_b", "model", "prompt_sha256", "mec_samples",
        "temperature", "max_pixels", "decode_mode",
    }
    frame = load(path, required).copy()
    frame["pair_low"] = frame[["image_a", "image_b"]].min(axis=1)
    frame["pair_high"] = frame[["image_a", "image_b"]].max(axis=1)
    if frame.duplicated(["pair_low", "pair_high"]).any():
        raise ValueError(f"{path}: duplicate unordered pairs")
    frame["success"] = frame["status"].isin(SUCCESS)
    frame["decisive"] = frame["winner"].notna()
    frame["outcome"] = np.where(
        frame["winner"].eq(frame["pair_low"]), "low",
        np.where(frame["winner"].eq(frame["pair_high"]), "high", "tie_or_invalid"),
    )
    low_is_a = frame["image_a"].eq(frame["pair_low"])
    frame["score_low"] = np.where(low_is_a, frame["calibrated_score_a"], frame["calibrated_score_b"])
    frame["score_high"] = np.where(low_is_a, frame["calibrated_score_b"], frame["calibrated_score_a"])
    frame["canonical_margin_high_minus_low"] = frame["score_high"] - frame["score_low"]
    keep = [
        "pair_low", "pair_high", "winner", "status", "success", "decisive",
        "outcome", "canonical_margin_high_minus_low",
    ]
    return frame[keep].rename(columns={c: f"{c}_{label}" for c in keep if c not in {"pair_low", "pair_high"}})


def overlap(ids_a: set[str], ids_b: set[str]) -> dict[str, float | int]:
    intersection = len(ids_a & ids_b)
    union = len(ids_a | ids_b)
    return {
        "count": intersection,
        "rate_of_k": intersection / len(ids_a),
        "jaccard": intersection / union,
    }


def configuration(audit: pd.DataFrame, path: Path) -> dict[str, object]:
    raw = pd.read_csv(path)
    fields = ["model", "prompt_sha256", "mec_samples", "temperature", "max_pixels", "decode_mode"]
    return {field: sorted(raw[field].dropna().astype(str).unique().tolist()) for field in fields}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_scores = {"image_id", SCORE, RANK, "q_score_0_10", "trueskill_sigma", "comparison_count"}
    scores_a = load(args.scores_a, required_scores)
    scores_b = load(args.scores_b, required_scores)
    scores = scores_a.merge(
        scores_b, on="image_id", how="inner", suffixes=(f"_{args.label_a}", f"_{args.label_b}"), validate="one_to_one"
    )
    if len(scores) != len(scores_a) or len(scores) != len(scores_b):
        raise ValueError("Score files do not contain exactly the same image IDs")

    if args.manifest:
        manifest = load(args.manifest, {"uuid", "city"})[["uuid", "city"]].copy()
        manifest["image_id"] = manifest["uuid"].astype(str) + ".jpg"
        scores = scores.merge(manifest[["image_id", "city"]], on="image_id", how="left", validate="one_to_one")

    score_a, score_b = f"{SCORE}_{args.label_a}", f"{SCORE}_{args.label_b}"
    q_a, q_b = f"q_score_0_10_{args.label_a}", f"q_score_0_10_{args.label_b}"
    rank_a, rank_b = f"{RANK}_{args.label_a}", f"{RANK}_{args.label_b}"
    sigma_a, sigma_b = f"trueskill_sigma_{args.label_a}", f"trueskill_sigma_{args.label_b}"
    scores["score_delta_b_minus_a"] = scores[score_b] - scores[score_a]
    scores["absolute_score_difference"] = scores["score_delta_b_minus_a"].abs()
    scores["rank_delta_b_minus_a"] = scores[rank_b] - scores[rank_a]
    scores["absolute_rank_difference"] = scores["rank_delta_b_minus_a"].abs()

    n = len(scores)
    scores["quartile_a"] = np.minimum(4, np.ceil(scores[rank_a] * 4 / n)).astype(int)
    scores["quartile_b"] = np.minimum(4, np.ceil(scores[rank_b] * 4 / n)).astype(int)
    correlations = corr(scores[score_a], scores[score_b])
    correlations.update(bootstrap_corr(scores[score_a].to_numpy(), scores[score_b].to_numpy(), args.bootstrap))

    top_bottom: dict[str, object] = {}
    for k in (10, 25, 50):
        top_a = set(scores.nsmallest(k, rank_a)["image_id"])
        top_b = set(scores.nsmallest(k, rank_b)["image_id"])
        bottom_a = set(scores.nlargest(k, rank_a)["image_id"])
        bottom_b = set(scores.nlargest(k, rank_b)["image_id"])
        top_bottom[str(k)] = {
            "most_polluted": overlap(top_a, top_b),
            "least_polluted": overlap(bottom_a, bottom_b),
        }

    audit_a = prepare_audit(args.audit_a, args.label_a)
    audit_b = prepare_audit(args.audit_b, args.label_b)
    pairs = audit_a.merge(audit_b, on=["pair_low", "pair_high"], how="inner", validate="one_to_one")
    success_a, success_b = f"success_{args.label_a}", f"success_{args.label_b}"
    decisive_a, decisive_b = f"decisive_{args.label_a}", f"decisive_{args.label_b}"
    outcome_a, outcome_b = f"outcome_{args.label_a}", f"outcome_{args.label_b}"
    common_success = pairs[success_a] & pairs[success_b]
    common_decisive = pairs[decisive_a] & pairs[decisive_b]
    decisive_pairs = pairs[common_decisive].copy()
    successful_pairs = pairs[common_success].copy()
    decisive_pairs["same_decisive_winner"] = decisive_pairs[outcome_a].eq(decisive_pairs[outcome_b])
    successful_pairs["outcome_success_a"] = np.where(
        successful_pairs[decisive_a], successful_pairs[outcome_a], "tie"
    )
    successful_pairs["outcome_success_b"] = np.where(
        successful_pairs[decisive_b], successful_pairs[outcome_b], "tie"
    )
    margin_a = f"canonical_margin_high_minus_low_{args.label_a}"
    margin_b = f"canonical_margin_high_minus_low_{args.label_b}"

    pair_metrics = {
        "pairs_a": int(len(audit_a)),
        "pairs_b": int(len(audit_b)),
        "matched_pairs": int(len(pairs)),
        "matched_pair_coverage_of_a": float(len(pairs) / len(audit_a)),
        "both_success_including_ties": int(common_success.sum()),
        "both_decisive": int(common_decisive.sum()),
        "same_decisive_winner": int(decisive_pairs["same_decisive_winner"].sum()),
        "decisive_winner_agreement_rate": float(decisive_pairs["same_decisive_winner"].mean()),
        "decisive_winner_agreement_95_ci_wilson": wilson_interval(
            int(decisive_pairs["same_decisive_winner"].sum()), len(decisive_pairs)
        ),
        "decisive_winner_cohens_kappa": cohens_kappa(decisive_pairs[outcome_a], decisive_pairs[outcome_b]),
        "three_way_outcome_agreement_rate": float(
            successful_pairs["outcome_success_a"].eq(successful_pairs["outcome_success_b"]).mean()
        ),
        "three_way_outcome_cohens_kappa": float(
            cohens_kappa(successful_pairs["outcome_success_a"], successful_pairs["outcome_success_b"])
        ),
        "canonical_margin_correlations_on_both_successful": corr(successful_pairs[margin_a], successful_pairs[margin_b]),
    }

    bias = scores["score_delta_b_minus_a"]
    score_metrics = {
        "matched_images": n,
        "direct_q_score_agreement": {
            "correlations": corr(scores[q_a], scores[q_b]),
            "mean_bias_b_minus_a": float((scores[q_b] - scores[q_a]).mean()),
            "mean_absolute_difference": float((scores[q_b] - scores[q_a]).abs().mean()),
            "root_mean_square_difference": float(np.sqrt(np.mean((scores[q_b] - scores[q_a]) ** 2))),
        },
        "correlations": correlations,
        "lins_concordance_correlation": lins_ccc(scores[score_a].to_numpy(), scores[score_b].to_numpy()),
        "mean_bias_b_minus_a": float(bias.mean()),
        "mean_absolute_difference": float(bias.abs().mean()),
        "root_mean_square_difference": float(np.sqrt(np.mean(bias**2))),
        "median_absolute_difference": float(bias.abs().median()),
        "bland_altman_95_limits": [float(bias.mean() - 1.96 * bias.std(ddof=1)), float(bias.mean() + 1.96 * bias.std(ddof=1))],
        "score_within_0_5": float(bias.abs().le(0.5).mean()),
        "score_within_1_0": float(bias.abs().le(1.0).mean()),
        "mean_absolute_rank_difference": float(scores["absolute_rank_difference"].mean()),
        "median_absolute_rank_difference": float(scores["absolute_rank_difference"].median()),
        "rank_difference_95th_percentile": float(scores["absolute_rank_difference"].quantile(0.95)),
        "rank_within_10": float(scores["absolute_rank_difference"].le(10).mean()),
        "rank_within_25": float(scores["absolute_rank_difference"].le(25).mean()),
        "exact_quartile_agreement": float(scores["quartile_a"].eq(scores["quartile_b"]).mean()),
        "within_one_quartile": float((scores["quartile_a"] - scores["quartile_b"]).abs().le(1).mean()),
        "quadratic_weighted_quartile_kappa": cohens_kappa(scores["quartile_a"], scores["quartile_b"], quadratic=True),
        "top_bottom_overlap": top_bottom,
        "trueskill_uncertainty": {
            f"mean_sigma_{args.label_a}": float(scores[sigma_a].mean()),
            f"mean_sigma_{args.label_b}": float(scores[sigma_b].mean()),
            "mean_sigma_reduction_rate": float(1 - scores[sigma_b].mean() / scores[sigma_a].mean()),
            "images_with_lower_sigma_in_b": int(scores[sigma_b].lt(scores[sigma_a]).sum()),
        },
    }

    city_rows: list[dict[str, object]] = []
    if "city" in scores:
        city = scores.groupby("city").agg(
            image_count=("image_id", "size"), mean_score_a=(score_a, "mean"), mean_score_b=(score_b, "mean"),
            mean_absolute_score_difference=("absolute_score_difference", "mean"),
        ).reset_index()
        city["mean_delta_b_minus_a"] = city["mean_score_b"] - city["mean_score_a"]
        city_rows = city.round(6).to_dict(orient="records")

    config_a = configuration(audit_a, args.audit_a)
    config_b = configuration(audit_b, args.audit_b)
    summary = {
        "comparison": f"{args.label_a}_vs_{args.label_b}",
        "configuration_identical": config_a == config_b,
        "configuration_a": config_a,
        "configuration_b": config_b,
        "score_and_rank_agreement": score_metrics,
        "repeated_pair_agreement": pair_metrics,
        "city_summary": city_rows,
        "interpretation_note": "The larger run contains all pairs from the smaller run plus additional pairs. It is not an independent sample; paired agreement is descriptive, not external validation.",
    }
    (args.output_dir / "consistency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    scores.sort_values("absolute_rank_difference", ascending=False).round(6).to_csv(
        args.output_dir / "per_image_differences.csv", index=False
    )
    pairs.to_csv(args.output_dir / "matched_pair_outcomes.csv", index=False)
    if city_rows:
        pd.DataFrame(city_rows).to_csv(args.output_dir / "city_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].scatter(scores[score_a], scores[score_b], alpha=0.7, s=22)
    axes[0, 0].plot([0, 10], [0, 10], "k--", lw=1)
    axes[0, 0].set(xlabel=f"VP score ({args.label_a})", ylabel=f"VP score ({args.label_b})", title=f"Scores: Spearman rho={correlations['spearman_rho']:.3f}")
    axes[0, 1].scatter(scores[rank_a], scores[rank_b], alpha=0.7, s=22)
    axes[0, 1].plot([1, n], [1, n], "k--", lw=1)
    axes[0, 1].invert_xaxis(); axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel=f"Rank ({args.label_a})", ylabel=f"Rank ({args.label_b})", title=f"Ranks: Kendall tau={correlations['kendall_tau']:.3f}")
    means = (scores[score_a] + scores[score_b]) / 2
    axes[1, 0].scatter(means, bias, alpha=0.7, s=22)
    axes[1, 0].axhline(bias.mean(), color="tab:red", lw=1)
    for limit in score_metrics["bland_altman_95_limits"]:
        axes[1, 0].axhline(limit, color="gray", ls="--", lw=1)
    axes[1, 0].set(xlabel="Mean VP score", ylabel=f"{args.label_b} - {args.label_a}", title="Bland-Altman score differences")
    ks = [10, 25, 50]
    x = np.arange(len(ks)); width = 0.36
    axes[1, 1].bar(x - width/2, [top_bottom[str(k)]["most_polluted"]["rate_of_k"] for k in ks], width, label="Most polluted")
    axes[1, 1].bar(x + width/2, [top_bottom[str(k)]["least_polluted"]["rate_of_k"] for k in ks], width, label="Least polluted")
    axes[1, 1].set(xticks=x, xticklabels=[f"Top {k}" for k in ks], ylim=(0, 1), ylabel="Overlap / k", title="Extreme-set overlap")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "consistency_analysis.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
