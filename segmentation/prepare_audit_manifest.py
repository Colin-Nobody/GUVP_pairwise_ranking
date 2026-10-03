#!/usr/bin/env python3
"""Build the deterministic 30-image target-domain segmentation audit set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/images_manifest.csv")
    parser.add_argument(
        "--v8-scores",
        type=Path,
        default=ROOT / "output/vlm_7b_definition_v8_3000_prefix_cache/perception_scores.csv",
    )
    parser.add_argument(
        "--v7-scores",
        type=Path,
        default=ROOT / "output/vlm_7b_definition_v7_3000_prefix_cache/perception_scores.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/segmentation_audit/audit_manifest.csv",
    )
    return parser.parse_args()


def normalize_scores(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["uuid"] = frame["image_id"].str.replace(r"\.[^.]+$", "", regex=True)
    return frame[["uuid", "visual_pollution_score_0_10", "rank_by_visual_pollution"]].rename(
        columns={
            "visual_pollution_score_0_10": f"vp_score_{suffix}",
            "rank_by_visual_pollution": f"global_rank_{suffix}",
        }
    )


def resolution_bucket(width: int, height: int) -> str:
    pixels = width * height
    if pixels >= 10_000_000:
        return "very_high_10mp_plus"
    if pixels >= 6_000_000:
        return "high_6_to_10mp"
    if pixels >= 3_000_000:
        return "medium_3_to_6mp"
    return "low_under_3mp"


def aspect_bucket(aspect: float) -> str:
    if aspect >= 1.8:
        return "panoramic"
    if aspect >= 1.2:
        return "landscape"
    if aspect <= 0.8:
        return "portrait"
    return "near_square"


def diversity_score(row: pd.Series, selected: pd.DataFrame, disagreement_max: float) -> float:
    """Reward metadata states not yet represented in this city's six images."""
    score = 0.0
    for column in [
        "source",
        "platform",
        "pano_status",
        "projection_type",
        "resolution_bucket",
        "aspect_bucket",
    ]:
        if selected.empty or row[column] not in set(selected[column]):
            score += 2.0
    if disagreement_max > 0:
        score += float(row["vp_v8_v7_abs_delta"] / disagreement_max)
    # Prefer larger images as useful stress tests when diversity is tied.
    score += min(float(row["megapixels"]) / 12.0, 1.0) * 0.25
    return score


def select_city(city_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = city_frame.sort_values(["vp_score_v8", "uuid"]).reset_index(drop=True)
    if len(ordered) != 50:
        raise ValueError(f"Expected 50 images for {ordered.city.iloc[0]}, got {len(ordered)}")

    # Fixed city-relative thirds: 17 low, 16 middle, 17 high.
    ordered["vp_stratum"] = "mid"
    ordered.loc[:16, "vp_stratum"] = "low"
    ordered.loc[33:, "vp_stratum"] = "high"
    ordered["city_score_rank_ascending"] = range(1, 51)

    chosen: list[pd.Series] = []
    disagreement_max = max(float(ordered["vp_v8_v7_abs_delta"].max()), 1e-12)
    for stratum in ["high", "mid", "low"]:
        candidates = ordered[ordered["vp_stratum"] == stratum].copy()
        if stratum == "high":
            anchor = candidates.sort_values(["vp_score_v8", "uuid"], ascending=[False, True]).iloc[0]
        elif stratum == "low":
            anchor = candidates.sort_values(["vp_score_v8", "uuid"]).iloc[0]
        else:
            median = float(ordered["vp_score_v8"].median())
            candidates["median_distance"] = (candidates["vp_score_v8"] - median).abs()
            anchor = candidates.sort_values(["median_distance", "uuid"]).iloc[0]
        anchor = anchor.copy()
        anchor["selection_role"] = "score_anchor"
        chosen.append(anchor)

        selected = pd.DataFrame(chosen)
        remaining = candidates[candidates["uuid"] != anchor["uuid"]].copy()
        remaining["diversity_objective"] = remaining.apply(
            diversity_score,
            axis=1,
            selected=selected,
            disagreement_max=disagreement_max,
        )
        # Deterministic tie-breaking: diversity, v7/v8 disagreement, megapixels, UUID.
        diverse = remaining.sort_values(
            ["diversity_objective", "vp_v8_v7_abs_delta", "megapixels", "uuid"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()
        diverse["selection_role"] = "coverage_candidate"
        chosen.append(diverse)
    return pd.DataFrame(chosen)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    v8 = normalize_scores(args.v8_scores, "v8")
    v7 = normalize_scores(args.v7_scores, "v7")
    frame = manifest.merge(v8, on="uuid", validate="one_to_one").merge(
        v7, on="uuid", validate="one_to_one"
    )
    if len(frame) != 250:
        raise ValueError(f"Expected 250 joined images, got {len(frame)}")

    image_paths: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    for uuid in frame["uuid"]:
        path = ROOT / "data/street_images" / f"{uuid}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        image_paths.append(str(path.relative_to(ROOT)))
        widths.append(width)
        heights.append(height)
    frame["image_id"] = frame["uuid"]
    frame["image_path"] = image_paths
    frame["width"] = widths
    frame["height"] = heights
    frame["megapixels"] = frame["width"] * frame["height"] / 1_000_000
    frame["aspect_ratio"] = frame["width"] / frame["height"]
    frame["resolution_bucket"] = [
        resolution_bucket(w, h) for w, h in zip(frame["width"], frame["height"])
    ]
    frame["aspect_bucket"] = frame["aspect_ratio"].map(aspect_bucket)
    frame["vp_v8_v7_delta"] = frame["vp_score_v8"] - frame["vp_score_v7"]
    frame["vp_v8_v7_abs_delta"] = frame["vp_v8_v7_delta"].abs()

    selected = pd.concat(
        [select_city(group) for _, group in frame.groupby("city", sort=True)],
        ignore_index=True,
    )
    selected = selected.sort_values(["city", "vp_stratum", "selection_role", "uuid"])
    selected["audit_order"] = range(1, len(selected) + 1)
    selected["selection_reason"] = selected.apply(
        lambda row: (
            f"{row.vp_stratum} city-relative VP stratum; "
            + (
                "extreme/median score anchor"
                if row.selection_role == "score_anchor"
                else "greedy metadata-resolution-v7/v8 coverage"
            )
        ),
        axis=1,
    )

    columns = [
        "audit_order",
        "image_id",
        "image_path",
        "city",
        "vp_stratum",
        "selection_role",
        "selection_reason",
        "vp_score_v8",
        "vp_score_v7",
        "vp_v8_v7_delta",
        "vp_v8_v7_abs_delta",
        "city_score_rank_ascending",
        "global_rank_v8",
        "global_rank_v7",
        "source",
        "platform",
        "pano_status",
        "projection_type",
        "width",
        "height",
        "megapixels",
        "aspect_ratio",
        "resolution_bucket",
        "aspect_bucket",
        "quality",
        "lighting_condition",
        "weather",
        "lat",
        "lon",
    ]
    output = selected[columns]
    if len(output) != 30:
        raise RuntimeError(f"Expected 30 selections, got {len(output)}")
    counts = output.groupby(["city", "vp_stratum"]).size()
    if not (counts == 2).all() or len(counts) != 15:
        raise RuntimeError(f"Invalid city/stratum counts:\n{counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")
    print(output.groupby(["city", "vp_stratum"]).size().unstack().to_string())
    print("\nCoverage:")
    for column in ["source", "platform", "pano_status", "projection_type", "resolution_bucket", "aspect_bucket"]:
        print(f"{column}: {output[column].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
