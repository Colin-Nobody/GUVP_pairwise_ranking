"""Create a compact, Chinese-language results dashboard for supervisor meetings.

Usage:
    python create_presentation_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output" / "synthetic_test"


def add_metric(ax: plt.Axes, x: float, title: str, value: str, note: str) -> None:
    ax.text(x, 0.74, title, ha="center", va="center", fontsize=13, color="#475569")
    ax.text(x, 0.43, value, ha="center", va="center", fontsize=25, fontweight="bold", color="#0f4c5c")
    ax.text(x, 0.14, note, ha="center", va="center", fontsize=10, color="#64748b")


def main() -> None:
    mpl.rcParams.update({
        "font.family": "Noto Sans CJK SC",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.titleweight": "bold",
    })
    scores = pd.read_csv(OUTPUT_DIR / "perception_scores.csv")
    validation = pd.read_csv(OUTPUT_DIR / "ground_truth_validation.csv")
    with (OUTPUT_DIR / "ranking_summary.json").open(encoding="utf-8") as handle:
        ranking_summary = json.load(handle)
    with (OUTPUT_DIR / "ground_truth_validation_summary.json").open(encoding="utf-8") as handle:
        validation_summary = json.load(handle)

    correlations = validation_summary["score_correlations"]
    errors = validation_summary["rank_error"]
    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.68, 1.6, 1.55])
    figure.suptitle("GUVP 两两比较排序：合成数据验证结果", fontsize=25, fontweight="bold", color="#0f172a")

    ax_metrics = figure.add_subplot(grid[0, :])
    ax_metrics.axis("off")
    add_metric(ax_metrics, 0.12, "街景图像", f"{ranking_summary['image_count']}", "合成验证样本")
    add_metric(ax_metrics, 0.37, "有效两两比较", f"{ranking_summary['valid_vote_rows']:,}", "每张图 25 次")
    add_metric(ax_metrics, 0.62, "Pearson 相关", f"r = {correlations['pearson_r']:.3f}", "估计 VP 与真值")
    add_metric(ax_metrics, 0.87, "平均排名误差", f"{errors['mean_absolute_error']:.2f}", "100 张图中的名次")

    ax_scatter = figure.add_subplot(grid[1, 0])
    x = validation["true_visual_pollution_score_0_1"]
    y = validation["visual_pollution_score_0_1"]
    ax_scatter.scatter(x, y, s=46, color="#0f766e", alpha=0.78, edgecolor="white", linewidth=0.5)
    line = linregress(x, y)
    line_x = np.array([0, 1])
    ax_scatter.plot(line_x, line_x, linestyle="--", color="#94a3b8", label="理想一致线")
    ax_scatter.plot(line_x, line.intercept + line.slope * line_x, color="#e76f51", linewidth=2, label="拟合趋势")
    ax_scatter.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="真值视觉污染指数", ylabel="TrueSkill 估计 VP 指数")
    ax_scatter.set_title("1. 估计分数与真值高度一致", loc="left", fontsize=15)
    ax_scatter.text(0.04, 0.93, f"Pearson r = {correlations['pearson_r']:.3f}\nSpearman ρ = {correlations['spearman_rho']:.3f}\nKendall τ = {correlations['kendall_tau']:.3f}", transform=ax_scatter.transAxes, va="top", fontsize=11,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f0fdfa", "edgecolor": "#99f6e4"})
    ax_scatter.legend(frameon=False, loc="lower right")
    ax_scatter.grid(alpha=0.2)

    ax_error = figure.add_subplot(grid[1, 1])
    ax_error.hist(validation["absolute_rank_error"], bins=np.arange(-0.5, 18.5, 1), color="#457b9d", edgecolor="white")
    ax_error.axvline(errors["mean_absolute_error"], color="#e63946", linewidth=2, label=f"平均值 = {errors['mean_absolute_error']:.2f}")
    ax_error.set(xlabel="绝对排名误差（位）", ylabel="图像数量")
    ax_error.set_title("2. 排名误差集中在较小范围", loc="left", fontsize=15)
    ax_error.text(0.98, 0.93, f"RMSE = {errors['root_mean_squared_error']:.2f}\n最大误差 = {errors['maximum_absolute_error']:.0f}\n完全命中 = {errors['exact_rank_match_count']} 张", transform=ax_error.transAxes, ha="right", va="top", fontsize=11,
                  bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fff7ed", "edgecolor": "#fed7aa"})
    ax_error.legend(frameon=False)
    ax_error.grid(axis="y", alpha=0.2)

    ax_table = figure.add_subplot(grid[2, :])
    ax_table.axis("off")
    most_polluted = scores.nsmallest(5, "rank_by_visual_pollution").copy()
    cleanest = scores.nlargest(5, "rank_by_visual_pollution").sort_values("rank_by_visual_pollution").copy()
    selection = pd.concat([most_polluted, cleanest], ignore_index=True)
    selection["类别"] = ["污染最高"] * 5 + ["污染最低"] * 5
    selection["图像"] = selection["image_id"]
    selection["VP 指数（0–10）"] = selection["visual_pollution_score_0_10"].map("{:.2f}".format)
    selection["估计排名"] = selection["rank_by_visual_pollution"].astype(int)
    selection["胜/负"] = selection["wins"].astype(int).astype(str) + "/" + selection["losses"].astype(int).astype(str)
    shown = selection[["类别", "图像", "VP 指数（0–10）", "估计排名", "胜/负"]]
    table = ax_table.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", colLoc="center", loc="center", bbox=[0.08, 0.03, 0.84, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#0f4c5c")
            cell.set_text_props(color="white", weight="bold")
        elif row <= 5:
            cell.set_facecolor("#fff1f2")
        else:
            cell.set_facecolor("#eff6ff")
    ax_table.set_title("3. 可直接核查的极端样本（VP 越高 = 视觉污染越严重）", loc="left", fontsize=15, pad=8)

    figure.savefig(OUTPUT_DIR / "presentation_dashboard.png", dpi=220, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "presentation_dashboard.pdf", bbox_inches="tight")
    plt.close(figure)

    shown.to_csv(OUTPUT_DIR / "presentation_key_results.csv", index=False, encoding="utf-8-sig")
    notes = """# GUVP 阶段性结果（导师汇报版）

## 结论

在 100 张合成街景、1,250 条两两比较（每张 25 次比较）上，TrueSkill 聚合得到的视觉污染（VP）指数与已知真值高度一致：Pearson r=0.974，Spearman ρ=0.975，Kendall τ=0.867。平均绝对排名误差为 5.24 位（满量程 100 位）。

## 方法说明

两两比较题目固定为“哪张街景更干净、视觉污染更低？”。TrueSkill 较高表示更干净；报告的 VP 指数取归一化偏好分的反向值，因此 VP 越高表示视觉污染越严重。Q-score 作为 Place Pulse 风格基线，与 TrueSkill 偏好分的 Pearson r=0.996、Spearman ρ=0.998，说明两种既有聚合方法给出高度一致的排序。

## 展示时可说

“当前结果验证的是排序与聚合模块本身：在可控真值下，它能从两两比较中稳定恢复图像层面的 VP 排序。下一步将以人工或 VLM 产生真实街景的比较结果，并开展人工一致性和跨城市泛化验证。”

## 边界

这批数据是合成验证，不等同于已证明 VLM 与人类感知一致；后续仍需小规模人工标注验证。
"""
    (OUTPUT_DIR / "presentation_brief.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
