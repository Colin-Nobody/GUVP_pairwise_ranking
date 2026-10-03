#!/usr/bin/env python3
"""Plot logged validation mIoU alongside post-hoc training-subset mIoU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_validation(path: Path) -> dict[int, float]:
    with path.open(newline="") as handle:
        return {int(row["iteration"]): float(row["validation_miou_percent"])
                for row in csv.DictReader(handle)}


def read_train(path: Path, run: str) -> dict[int, float]:
    with path.open(newline="") as handle:
        return {int(row["iteration"]): float(row["train_eval_miou_percent"])
                for row in csv.DictReader(handle) if row["run"] == run}


def train_segments(points: dict[int, float]) -> list[list[int]]:
    steps = sorted(points)
    groups: list[list[int]] = []
    for step in steps:
        if not groups or step - groups[-1][-1] > 10000:
            groups.append([step])
        else:
            groups[-1].append(step)
    return groups


def draw(axis: plt.Axes, val: dict[int, float], train: dict[int, float], title: str, xlabel: str) -> None:
    val_steps = sorted(val)
    axis.plot(val_steps, [val[step] for step in val_steps], "o-", color="#238b45", linewidth=1.8, markersize=3.8, label="validation mIoU (full log)")
    for index, group in enumerate(train_segments(train)):
        axis.plot(group, [train[step] for step in group], "s--", color="#d95f0e", linewidth=1.8, markersize=5, label="train mIoU (fixed 2k subset; retained checkpoints)" if index == 0 else None)
    axis.set(title=title, xlabel=xlabel, ylabel="mIoU (%)")
    axis.grid(alpha=0.25)
    axis.set_xlim(0, max(val_steps))
    axis.legend(loc="best", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-validation", type=Path, required=True)
    parser.add_argument("--finetune-validation", type=Path, required=True)
    parser.add_argument("--train-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_val = read_validation(args.main_validation)
    fine_val = read_validation(args.finetune_validation)
    main_train = read_train(args.train_eval, "main_450k")
    fine_train = read_train(args.train_eval, "finetune_140k")
    if not main_train or not fine_train:
        raise SystemExit("Training-subset results for both runs are required")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), constrained_layout=True)
    fig.suptitle("Mapillary mIoU: validation history and retained-checkpoint train evaluation", fontsize=15, fontweight="bold")
    draw(axes[0], main_val, main_train, "Main training 8k–450k", "Global iteration")
    draw(axes[1], fine_val, fine_train, "Fine-tuning 2k–140k", "Fine-tuning iteration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
