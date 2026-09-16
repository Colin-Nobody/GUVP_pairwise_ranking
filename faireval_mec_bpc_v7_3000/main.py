"""CLI for converting pairwise street-image preferences into VP scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algorithms.q_score import calculate_q_scores
from algorithms.trueskill_rank import calculate_trueskill
from evaluation.ranking_analysis import write_evaluation, write_ground_truth_validation

REQUIRED_COLUMNS = {"image_a", "image_b", "winner"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with image_a,image_b,winner columns")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for score files and charts")
    parser.add_argument("--dimension", default="visual_pollution", help="Label saved in the evaluation summary")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional CSV with image_id and true_visual_pollution_score_0_1 or true_visual_pollution_rank",
    )
    return parser.parse_args()


def load_votes(path: Path) -> list[tuple[str, str, str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV column(s): {', '.join(sorted(missing))}")
    frame = frame.loc[:, ["image_a", "image_b", "winner"]].apply(lambda col: col.str.strip())
    if frame.eq("").any().any():
        raise ValueError("image_a, image_b, and winner cannot be empty")
    same_image = frame["image_a"] == frame["image_b"]
    if same_image.any():
        raise ValueError(f"Found {int(same_image.sum())} comparison(s) where image_a equals image_b")
    valid_winner = frame["winner"].eq(frame["image_a"]) | frame["winner"].eq(frame["image_b"])
    if not valid_winner.all():
        bad_rows = (frame.index[~valid_winner] + 2).tolist()[:10]
        raise ValueError(f"winner must equal image_a or image_b. Invalid CSV row(s): {bad_rows}")
    return list(frame.itertuples(index=False, name=None))


def min_max_normalize(values: pd.Series) -> pd.Series:
    low, high = values.min(), values.max()
    if high == low:
        return pd.Series(0.5, index=values.index, dtype=float)
    return (values - low) / (high - low)


def main() -> None:
    args = parse_args()
    votes = load_votes(args.input)
    if not votes:
        raise ValueError("The input CSV has no comparison rows")

    q_scores = pd.DataFrame.from_dict(calculate_q_scores(votes), orient="index")
    trueskill_scores = pd.DataFrame.from_dict(calculate_trueskill(votes), orient="index")
    scores = q_scores.join(trueskill_scores).rename_axis("image_id").reset_index()
    scores["trueskill_preference_0_1"] = min_max_normalize(scores["trueskill_mu"])
    scores["visual_pollution_score_0_1"] = 1.0 - scores["trueskill_preference_0_1"]
    scores["visual_pollution_score_0_10"] = 10.0 * scores["visual_pollution_score_0_1"]
    scores = scores.sort_values("visual_pollution_score_0_1", ascending=False).reset_index(drop=True)
    scores.insert(1, "dimension", args.dimension)
    scores.insert(2, "rank_by_visual_pollution", range(1, len(scores) + 1))
    scores = scores.round(6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "perception_scores.csv", index=False)
    write_evaluation(scores, args.output_dir, len(votes), args.dimension)
    if args.ground_truth:
        write_ground_truth_validation(scores, args.ground_truth, args.output_dir)
    print(f"Processed {len(votes)} valid comparisons across {len(scores)} images.")
    print(f"Scores: {args.output_dir / 'perception_scores.csv'}")
    if args.ground_truth:
        print(f"Ground-truth validation: {args.output_dir / 'ground_truth_validation_summary.json'}")


if __name__ == "__main__":
    main()
