#!/usr/bin/env python3
"""Compare two VLM pairwise-vote runs and their aggregated VP scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest, kendalltau, pearsonr, spearmanr


VOTE_COLUMNS = {"image_a", "image_b", "winner"}
SCORE_COLUMNS = {
    "image_id",
    "rank_by_visual_pollution",
    "visual_pollution_score_0_10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-a", type=Path, required=True)
    parser.add_argument("--scores-a", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--votes-b", type=Path, required=True)
    parser.add_argument("--scores-b", type=Path, required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    return frame


def prepare_votes(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, object]]:
    votes = load_csv(path, VOTE_COLUMNS).copy()
    valid = votes["winner"].eq(votes["image_a"]) | votes["winner"].eq(votes["image_b"])
    votes["pair_low"] = votes[["image_a", "image_b"]].min(axis=1)
    votes["pair_high"] = votes[["image_a", "image_b"]].max(axis=1)
    duplicate_count = int(votes.duplicated(["pair_low", "pair_high"]).sum())
    if duplicate_count:
        raise ValueError(f"{path} contains {duplicate_count} duplicate unordered image pair(s)")

    a_selected = votes["winner"].eq(votes["image_a"])
    a_count = int(a_selected.sum())
    valid_count = int(valid.sum())
    time_summary: dict[str, object] | None = None
    if "timestamp_utc" in votes and votes["timestamp_utc"].notna().any():
        timestamps = pd.to_datetime(votes["timestamp_utc"], utc=True)
        elapsed = float((timestamps.max() - timestamps.min()).total_seconds())
        time_summary = {
            "first_timestamp_utc": timestamps.min().isoformat(),
            "last_timestamp_utc": timestamps.max().isoformat(),
            "elapsed_seconds": elapsed,
            "seconds_per_pair": elapsed / max(len(votes) - 1, 1),
        }

    summary = {
        "label": label,
        "source": str(path),
        "rows": int(len(votes)),
        "valid_winners": valid_count,
        "invalid_winners": int((~valid).sum()),
        "unique_images": int(pd.unique(pd.concat([votes["image_a"], votes["image_b"]])).size),
        "side_selection": {
            "a_count": a_count,
            "b_count": valid_count - a_count,
            "a_rate": a_count / valid_count,
            "two_sided_binomial_p_value_vs_0_5": float(binomtest(a_count, valid_count, 0.5).pvalue),
        },
        "model_values": sorted(votes["model"].dropna().astype(str).unique().tolist()) if "model" in votes else [],
        "prompt_versions": sorted(votes["prompt_version"].dropna().astype(str).unique().tolist())
        if "prompt_version" in votes
        else [],
        "max_pixels_values": sorted(votes["max_pixels"].dropna().astype(int).unique().tolist())
        if "max_pixels" in votes
        else [],
        "timing": time_summary,
    }
    return votes, summary


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    kendall = kendalltau(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p_value": float(kendall.pvalue),
    }


def overlap_at_k(scores: pd.DataFrame, rank_a: str, rank_b: str, k: int, most_polluted: bool) -> int:
    selector = pd.DataFrame.nsmallest if most_polluted else pd.DataFrame.nlargest
    ids_a = set(selector(scores, k, rank_a)["image_id"])
    ids_b = set(selector(scores, k, rank_b)["image_id"])
    return len(ids_a & ids_b)


def write_plot(scores: pd.DataFrame, vote_summaries: list[dict[str, object]], labels: list[str], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    rates = [float(item["side_selection"]["a_rate"]) for item in vote_summaries]
    axes[0].bar(labels, rates, color=["#7b68a6", "#3578a8"])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction choosing image A")
    axes[0].set_title("A/B position selection")
    for index, rate in enumerate(rates):
        axes[0].text(index, rate + 0.025, f"{rate:.1%}", ha="center")

    x = scores[f"visual_pollution_score_0_10_{labels[0]}"]
    y = scores[f"visual_pollution_score_0_10_{labels[1]}"]
    axes[1].scatter(x, y, color="#3578a8", alpha=0.75)
    axes[1].plot([0, 10], [0, 10], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].set_xlabel(f"VP score — {labels[0]}")
    axes[1].set_ylabel(f"VP score — {labels[1]}")
    axes[1].set_title("Cross-model VP score agreement")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.label_a == args.label_b:
        raise ValueError("--label-a and --label-b must differ")

    votes_a, vote_summary_a = prepare_votes(args.votes_a, args.label_a)
    votes_b, vote_summary_b = prepare_votes(args.votes_b, args.label_b)
    pair_columns = ["pair_low", "pair_high", "winner"]
    paired = votes_a[pair_columns].merge(
        votes_b[pair_columns],
        on=["pair_low", "pair_high"],
        how="inner",
        suffixes=(f"_{args.label_a}", f"_{args.label_b}"),
        validate="one_to_one",
    )
    winner_a = f"winner_{args.label_a}"
    winner_b = f"winner_{args.label_b}"
    paired["same_physical_winner"] = paired[winner_a].eq(paired[winner_b])

    scores_a = load_csv(args.scores_a, SCORE_COLUMNS)
    scores_b = load_csv(args.scores_b, SCORE_COLUMNS)
    scores = scores_a.merge(
        scores_b,
        on="image_id",
        how="inner",
        suffixes=(f"_{args.label_a}", f"_{args.label_b}"),
        validate="one_to_one",
    )
    score_a = f"visual_pollution_score_0_10_{args.label_a}"
    score_b = f"visual_pollution_score_0_10_{args.label_b}"
    rank_a = f"rank_by_visual_pollution_{args.label_a}"
    rank_b = f"rank_by_visual_pollution_{args.label_b}"
    scores["vp_score_delta_b_minus_a"] = scores[score_b] - scores[score_a]
    scores["absolute_rank_difference"] = (scores[rank_b] - scores[rank_a]).abs()

    city_summary: list[dict[str, object]] = []
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        if not {"uuid", "city"}.issubset(manifest.columns):
            raise ValueError("Manifest must contain uuid and city columns")
        city_map = dict(zip(manifest["uuid"].astype(str) + ".jpg", manifest["city"].astype(str)))
        scores["city"] = scores["image_id"].map(city_map)
        city_frame = (
            scores.groupby("city", dropna=False)
            .agg(
                image_count=("image_id", "size"),
                **{
                    f"mean_vp_{args.label_a}": (score_a, "mean"),
                    f"mean_vp_{args.label_b}": (score_b, "mean"),
                },
            )
            .reset_index()
        )
        city_frame["mean_vp_delta_b_minus_a"] = (
            city_frame[f"mean_vp_{args.label_b}"] - city_frame[f"mean_vp_{args.label_a}"]
        )
        city_summary = city_frame.round(6).to_dict(orient="records")

    summary = {
        "comparison": f"{args.label_a}_vs_{args.label_b}",
        "votes": {
            args.label_a: vote_summary_a,
            args.label_b: vote_summary_b,
            "matched_unordered_pairs": int(len(paired)),
            "unmatched_pairs_a": int(len(votes_a) - len(paired)),
            "unmatched_pairs_b": int(len(votes_b) - len(paired)),
            "same_physical_winner_count": int(paired["same_physical_winner"].sum()),
            "different_physical_winner_count": int((~paired["same_physical_winner"]).sum()),
            "winner_agreement_rate": float(paired["same_physical_winner"].mean()),
        },
        "scores": {
            "source_a": str(args.scores_a),
            "source_b": str(args.scores_b),
            "matched_images": int(len(scores)),
            "visual_pollution_score_0_10_correlations": correlation(scores[score_a], scores[score_b]),
            "mean_absolute_vp_score_difference": float(scores["vp_score_delta_b_minus_a"].abs().mean()),
            "mean_absolute_rank_difference": float(scores["absolute_rank_difference"].mean()),
            "top_bottom_overlap": {
                str(k): {
                    "most_polluted": overlap_at_k(scores, rank_a, rank_b, k, True),
                    "least_polluted": overlap_at_k(scores, rank_a, rank_b, k, False),
                }
                for k in (10, 25, 50)
            },
        },
        "city_mean_vp": city_summary,
        "quality_flags": {
            args.label_a: ["A selection rate differs from 50% by at least 5 percentage points"]
            if abs(float(vote_summary_a["side_selection"]["a_rate"]) - 0.5) >= 0.05
            else [],
            args.label_b: ["A selection rate differs from 50% by at least 5 percentage points"]
            if abs(float(vote_summary_b["side_selection"]["a_rate"]) - 0.5) >= 0.05
            else [],
            "interpretation": (
                "Q-score/TrueSkill agreement within a run does not validate the VLM judgments. "
                "A/B swap auditing is required before treating either run as ground truth."
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    scores.round(6).sort_values(rank_b).to_csv(args.output_dir / "per_image_score_comparison.csv", index=False)
    paired.to_csv(args.output_dir / "per_pair_winner_comparison.csv", index=False)
    write_plot(
        scores,
        [vote_summary_a, vote_summary_b],
        [args.label_a, args.label_b],
        args.output_dir / "cross_model_comparison.png",
    )

    print(f"Compared {len(paired)} matched image pairs and {len(scores)} matched images.")
    print(f"Summary: {args.output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
