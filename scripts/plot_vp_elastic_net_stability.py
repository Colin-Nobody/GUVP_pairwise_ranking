#!/usr/bin/env python3
"""Plot strict and relaxed Elastic Net semantic-element associations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/colin/GUVP_pairwise_ranking")


def draw(table: pd.DataFrame, prevalence: pd.Series, output: Path, title: str, subtitle: str) -> None:
    table = table.sort_values("standardised_beta").reset_index(drop=True)
    labels = table["class_name"].astype(str).tolist()
    beta = table["standardised_beta"].to_numpy(dtype=float)
    counts = np.array([prevalence.get(name, 0) for name in table["feature_column"]])
    colors = np.where(beta >= 0, "#bf3f3f", "#2774a6")
    y = np.arange(len(table))

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, (ax, ax_prev) = plt.subplots(
        ncols=2,
        figsize=(15.5, max(8.5, 0.36 * len(table) + 3.0)),
        gridspec_kw={"width_ratios": [4.8, 1.3], "wspace": 0.06},
        sharey=True,
    )
    ax.barh(y, beta, color=colors, alpha=0.92, height=0.72)
    ax.axvline(0, color="#252525", linewidth=1)
    max_abs = max(float(np.max(np.abs(beta))), 0.001)
    ax.set_xlim(-1.20 * max_abs, 1.20 * max_abs)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Standardised Elastic Net coefficient")
    ax.set_title("Association with VP score")
    pad = max(np.max(np.abs(beta)) * 0.035, 0.0005)
    for index, value in enumerate(beta):
        ax.text(
            value + (pad if value >= 0 else -pad), index, f"{value:+.3f}",
            va="center", ha="left" if value >= 0 else "right",
            fontsize=8, color="#333333",
        )

    ax_prev.scatter(counts, y, s=42, color="#5b8c5a", edgecolor="white", linewidth=0.5, zorder=3)
    ax_prev.set_xlim(0, 1050)
    ax_prev.set_xticks([0, 250, 500, 750, 1000])
    ax_prev.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax_prev.set_axisbelow(True)
    ax_prev.set_xlabel("Images present\n(out of 1000)")
    ax_prev.set_title("Occurrence")
    ax_prev.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=10.5, color="#404040")
    fig.text(
        0.5, 0.012,
        "Red: larger predicted share is associated with higher VP score; blue: lower VP score. "
        "Within-city conditional associations; compositional shares sum to 100%; not causal effects.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.92, bottom=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{output} ({len(table)} elements)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path,
        default=ROOT / "output/vp_pollution_elastic_net_expansion_1000/semantic_element_weights.csv",
    )
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "output/segmentation_expansion_1000/elastic_net_features.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output/vp_pollution_elastic_net_expansion_1000",
    )
    args = parser.parse_args()

    weights = pd.read_csv(args.weights)
    features = pd.read_csv(args.features)
    prevalence = {
        column: int((features[column].fillna(0) > 0).sum())
        for column in weights["feature_column"]
        if column in features
    }

    strict = weights[
        (weights["outer_selection_frequency"] >= 0.70)
        & (weights["bootstrap_selection_frequency"] >= 0.70)
        & (weights["outer_sign_consistency"] >= 0.90)
        & (weights["bootstrap_sign_consistency"] >= 0.90)
    ].copy()
    relaxed = weights[
        (weights["outer_selection_frequency"] >= 0.30)
        & (weights["bootstrap_selection_frequency"] >= 0.50)
        & (weights["outer_sign_consistency"] >= 0.70)
        & (weights["bootstrap_sign_consistency"] >= 0.70)
    ].copy()

    common = "1000 images • 20 cities • city-adjusted Elastic Net • standardized coefficients"
    draw(
        strict,
        pd.Series(prevalence),
        args.output_dir / "vp_semantic_weights_strict_28.png",
        f"VP score associations — strict stability screen ({len(strict)} elements)",
        common + " • strict: outer and bootstrap selection ≥70%; sign consistency ≥90%",
    )
    draw(
        relaxed,
        pd.Series(prevalence),
        args.output_dir / "vp_semantic_weights_relaxed_38.png",
        f"VP score associations — relaxed stability screen ({len(relaxed)} elements)",
        common + " • relaxed: outer selection ≥30%; bootstrap ≥50%; sign consistency ≥70%",
    )


if __name__ == "__main__":
    main()
