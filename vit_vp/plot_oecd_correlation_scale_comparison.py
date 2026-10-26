#!/usr/bin/env python3
"""Plot OECD correlation results across four Mapillary sample-scale experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THEMES = ["收入", "就业", "住房", "教育", "健康", "环境", "安全", "公民参与", "服务可及性", "社区", "生活满意度"]
DERIVED = "oecd_11_theme_equal_mean_derived"
EXPERIMENTS = [
    {
        "key": "50city_200",
        "name": "50 城 × 200 图/城",
        "short": "50 城\n200 图/城\n10,000 图",
        "file": "oecd_analysis/correlations.csv",
        "color": "#2878B5",
        "marker": "o",
    },
    {
        "key": "50city_700",
        "name": "50 城 × 700 图/城",
        "short": "50 城\n700 图/城\n35,000 图",
        "file": "oecd_analysis_700/correlations.csv",
        "color": "#E07A22",
        "marker": "s",
    },
    {
        "key": "135city_700",
        "name": "135 城 × 700 图/城",
        "short": "135 城\n700 图/城\n94,500 图",
        "file": "oecd_analysis_expanded_700/correlations.csv",
        "color": "#3A946B",
        "marker": "D",
    },
    {
        "key": "135city_1000",
        "name": "135 城 × 1,000 图/城",
        "short": "135 城\n1,000 图/城\n135,000 图",
        "file": "oecd_analysis_expanded_1000/correlations.csv",
        "color": "#8064A2",
        "marker": "^",
    },
]


def load_result(root: Path, experiment: dict) -> pd.DataFrame:
    path = root / experiment["file"]
    data = pd.read_csv(path)
    if "vp_aggregation" in data.columns:
        data = data[data.vp_aggregation.eq("primary_equal_grid_then_equal_city")].copy()
    required = {
        "oecd_theme",
        "spearman_rho",
        "spearman_country_cluster_bootstrap_ci95_low",
        "spearman_country_cluster_bootstrap_ci95_high",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    n_column = "n_regions" if "n_regions" in data.columns else "n_cities"
    data = data[["oecd_theme", n_column, "spearman_rho",
                 "spearman_country_cluster_bootstrap_ci95_low",
                 "spearman_country_cluster_bootstrap_ci95_high"]].rename(columns={n_column: "n_observations"})
    data["experiment"] = experiment["name"]
    data["experiment_key"] = experiment["key"]
    return data


def add_ci_point(ax, x: float, y: float, low: float, high: float, *, color: str, marker: str,
                 yerr: float | None = None, label: str | None = None, markersize: float = 7) -> None:
    ypos = y if yerr is None else y + yerr
    ax.errorbar(
        x, ypos,
        xerr=np.array([[max(0.0, y - low)], [max(0.0, high - y)]]),
        fmt=marker,
        color=color,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=markersize,
        elinewidth=1.8,
        capsize=3.5,
        label=label,
        zorder=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {e["key"]: load_result(args.data_root, e) for e in EXPERIMENTS}
    long_data = pd.concat(results.values(), ignore_index=True)
    long_data.to_csv(args.output_dir / "correlation_plot_data.csv", index=False, encoding="utf-8-sig")

    # Figure 1: one compact summary for the derived (non-official) mean of all
    # 11 OECD topics. The confidence intervals are country-cluster bootstraps.
    fig, ax = plt.subplots(figsize=(10.5, 6.1), constrained_layout=True)
    x_positions = np.arange(len(EXPERIMENTS))
    for x, experiment in zip(x_positions, EXPERIMENTS):
        row = results[experiment["key"]].set_index("oecd_theme").loc[DERIVED]
        add_ci_point(
            ax, x, row.spearman_rho,
            row.spearman_country_cluster_bootstrap_ci95_low,
            row.spearman_country_cluster_bootstrap_ci95_high,
            color=experiment["color"], marker=experiment["marker"], markersize=9,
        )
        unit = "城市" if experiment["key"].startswith("50city") else "OECD 地区"
        ax.annotate(
            f"ρ={row.spearman_rho:.3f}\nN={int(row.n_observations)} {unit}",
            (x, row.spearman_rho), xytext=(0, 15), textcoords="offset points",
            ha="center", va="bottom", color=experiment["color"], fontsize=10,
        )
    ax.axhline(0, color="#555555", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks(x_positions, [e["short"] for e in EXPERIMENTS])
    ax.set_xlim(-0.55, len(EXPERIMENTS) - 0.45)
    ax.set_ylim(-0.68, 0.36)
    ax.set_ylabel("Spearman ρ（视觉污染分数 vs OECD 分数）")
    ax.set_title("四组样本规模实验：11 主题等权派生均值的相关性", loc="left", fontsize=15, weight="bold", pad=15)
    ax.text(
        0, 1.015,
        "圆点/方块/菱形为相关系数；横线为按国家聚类 bootstrap 95% 区间。派生均值不是 OECD 官方综合指标。",
        transform=ax.transAxes, ha="left", va="bottom", color="#444444", fontsize=9.5,
    )
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01, -0.005,
        "注意：前两组为 50 个城市（同一城市集合）；扩展组为 135 个城市并按唯一 OECD 地区聚合。组间差异同时包含样本量、城市覆盖和观测单位变化，不能单独归因于图片数。",
        ha="left", va="top", fontsize=9, color="#555555", wrap=True,
    )
    for ext in ("png", "svg"):
        fig.savefig(args.output_dir / f"01_derived_index_four_scales.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: all 11 official themes, with one estimate and interval per sample.
    fig, ax = plt.subplots(figsize=(15.0, 9.4))
    base_y = np.arange(len(THEMES))[::-1]
    offsets = np.linspace(0.27, -0.27, len(EXPERIMENTS))
    for experiment, offset in zip(EXPERIMENTS, offsets):
        data = results[experiment["key"]].set_index("oecd_theme")
        for theme, y in zip(THEMES, base_y):
            row = data.loc[theme]
            ax.errorbar(
                row.spearman_rho, y + offset,
                xerr=np.array([[
                    max(0.0, row.spearman_rho - row.spearman_country_cluster_bootstrap_ci95_low)
                ], [
                    max(0.0, row.spearman_country_cluster_bootstrap_ci95_high - row.spearman_rho)
                ]]),
                fmt=experiment["marker"], color=experiment["color"], ecolor=experiment["color"],
                markerfacecolor=experiment["color"], markeredgecolor="white", markeredgewidth=0.7,
                markersize=6.3, elinewidth=1.5, capsize=2.5, zorder=3,
                label=experiment["name"] if theme == THEMES[0] else None,
            )
    ax.axvline(0, color="#555555", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(base_y, THEMES)
    ax.set_xlim(-0.72, 0.48)
    ax.set_ylim(-0.65, len(THEMES) - 0.35)
    ax.set_xlabel("Spearman ρ（视觉污染分数 vs OECD 主题分数）")
    # N is shown in the y-axis labels in the same order as the legend.
    labels = []
    for theme in THEMES:
        ns = [int(results[e["key"]].set_index("oecd_theme").loc[theme, "n_observations"]) for e in EXPERIMENTS]
        labels.append(f"{theme}   ({' / '.join(map(str, ns))})")
    ax.set_yticks(base_y, labels)
    ax.legend(loc="lower right", frameon=False, ncol=1, title="城市数 × 每城图片数")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.12, top=0.88)
    fig.suptitle("11 个 OECD 福祉主题：四组样本规模的相关性对比", x=0.02, y=0.985,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.02, 0.945,
             "点为 Spearman ρ，横线为按国家聚类 bootstrap 95% 区间；每行括号为各实验有效 N（顺序见图例）。",
             ha="left", va="top", color="#444444", fontsize=9.5)
    fig.text(
        0.02, 0.025,
        "N 顺序：50 城×200 图/城 / 50 城×700 图/城 / 135 城×700 图/城 / 135 城×1,000 图/城。前两项 N 是城市数，扩展组 N 是 OECD 地区数。区间重叠与否不等同于组间差异检验；11 项为探索性比较，未作多重检验校正。",
        ha="left", va="top", fontsize=9, color="#555555", wrap=True,
    )
    for ext in ("png", "svg"):
        fig.savefig(args.output_dir / f"02_oecd_11_themes_four_scales.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figures and plot data to {args.output_dir}")


if __name__ == "__main__":
    main()
