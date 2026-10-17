#!/usr/bin/env python3
"""Visualise the pooled continuous VP semantic-element weight ranking."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path("/home/colin/GUVP_pairwise_ranking")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=root / "output/vp_pollution_pooled_ridge_136k_nonrare_elements/continuous_semantic_element_weight_ranking.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "output/vp_pollution_pooled_ridge_136k_nonrare_elements/continuous_semantic_element_weight_ranking.png",
    )
    args = parser.parse_args()

    table = pd.read_csv(args.input).sort_values("standardised_beta").reset_index(drop=True)
    labels = table["class_name"].to_list()
    beta = table["standardised_beta"].to_numpy()
    prevalence = table["n_images_present"].to_numpy()
    colors = np.where(beta >= 0, "#bf3f3f", "#2774a6")
    y = np.arange(len(table))

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, (ax, ax_prev) = plt.subplots(
        ncols=2, figsize=(15.5, 15),
        gridspec_kw={"width_ratios": [4.8, 1.3], "wspace": 0.06},
        sharey=True,
    )
    ax.barh(y, beta, color=colors, alpha=0.92, height=0.72)
    ax.axvline(0, color="#252525", linewidth=1)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Standardised Ridge coefficient")
    ax.set_title("Association with VP score")

    pad = max(np.max(np.abs(beta)) * 0.03, 0.0002)
    for index, value in enumerate(beta):
        x = value + pad if value >= 0 else value - pad
        ax.text(x, index, f"{value:+.3f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=8, color="#333333")

    ax_prev.scatter(prevalence, y, s=42, color="#5b8c5a", edgecolor="white", linewidth=0.5, zorder=3)
    ax_prev.set_xlim(0, 255)
    ax_prev.set_xticks([0, 50, 100, 150, 200, 250])
    ax_prev.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax_prev.set_axisbelow(True)
    ax_prev.set_xlabel("Images present\n(out of 250)")
    ax_prev.set_title("Occurrence")
    ax_prev.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(
        "VP pollution score — continuous semantic-element weight ranking",
        fontsize=18, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.958,
        "Pooled 250-image analysis • 42 elements occurring in ≥25 images • old fine-tuned DeepLabV3+ semantic shares\n"
        "Blue: larger share associated with lower VP score; red: larger share associated with higher VP score",
        ha="center", va="top", fontsize=10.5, color="#404040",
    )
    fig.text(
        0.5, 0.012,
        "Descriptive association ranking only. Shares sum to 100%, so an element's increase implies decreases in other elements; coefficients are not causal effects.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.92, bottom=0.045)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
