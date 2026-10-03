#!/usr/bin/env python3
"""Export and plot complete validation-mIoU histories for the Mapillary runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(r"Iter\(train\) \[\s*(\d+)/")
VAL_RE = re.compile(r"Iter\(val\) \[2000/2000\].*?mIoU: ([\d.]+)")


def read_points(logs: list[Path]) -> dict[int, float]:
    points: dict[int, float] = {}
    for path in logs:
        current_iter: int | None = None
        for line in path.read_text(errors="replace").splitlines():
            train = TRAIN_RE.search(line)
            if train:
                current_iter = int(train.group(1))
                continue
            val = VAL_RE.search(line)
            if val and current_iter is not None:
                points[current_iter] = float(val.group(1))
    return points


def export_csv(points: dict[int, float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "validation_miou_percent"])
        writer.writeheader()
        writer.writerows(
            {"iteration": iteration, "validation_miou_percent": points[iteration]}
            for iteration in sorted(points)
        )


def draw(axis: plt.Axes, points: dict[int, float], title: str, xlabel: str) -> None:
    iterations = sorted(points)
    values = [points[iteration] for iteration in iterations]
    best_index = max(range(len(values)), key=values.__getitem__)
    axis.plot(iterations, values, "o-", color="#238b45", linewidth=2, markersize=4.5)
    axis.scatter(iterations[best_index], values[best_index], color="#d94801", s=65, zorder=3)
    axis.annotate(
        f"best {values[best_index]:.2f}%\n@ {iterations[best_index]:,}",
        xy=(iterations[best_index], values[best_index]),
        xytext=(10, -38),
        textcoords="offset points",
        color="#a63603",
        arrowprops={"arrowstyle": "->", "color": "#a63603"},
    )
    axis.set(title=title, xlabel=xlabel, ylabel="Validation mIoU (%)")
    axis.grid(alpha=0.25)
    axis.set_xlim(0, max(iterations))
    axis.set_ylim(min(values) - 0.8, max(values) + 0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--finetune-csv", type=Path, required=True)
    parser.add_argument("--main-logs", type=Path, nargs="+", required=True)
    parser.add_argument("--finetune-log", type=Path, required=True)
    args = parser.parse_args()

    main_points = read_points(args.main_logs)
    finetune_points = read_points([args.finetune_log])
    if not main_points or not finetune_points:
        raise SystemExit("Could not find validation mIoU points in the supplied logs")
    export_csv(main_points, args.main_csv)
    export_csv(finetune_points, args.finetune_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 complete validation history", fontsize=16, fontweight="bold")
    draw(axes[0], main_points, "Main training: 8k–450k", "Global iteration")
    draw(axes[1], finetune_points, "Fine-tuning: LR 3e-5 → 1e-6, 2k–140k", "Fine-tuning iteration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"main points={len(main_points)}, fine-tune points={len(finetune_points)}")


if __name__ == "__main__":
    main()
