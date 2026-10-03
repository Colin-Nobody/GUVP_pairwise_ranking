#!/usr/bin/env python3
"""Plot the complete Mapillary DeepLabV3+ training history from MMEngine logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE = re.compile(
    r"Iter\(train\) \[\s*(?P<iter>\d+)/\d+\].*?"
    r"lr: (?P<lr>[\deE+.-]+).*?loss: (?P<loss>[\deE+.-]+).*?"
    r"decode\.acc_seg: (?P<acc>[\deE+.-]+)"
)
VAL_RE = re.compile(
    r"Iter\(val\) \[2000/2000\].*?aAcc: (?P<aacc>[\deE+.-]+)\s+"
    r"mIoU: (?P<miou>[\deE+.-]+)\s+mAcc: (?P<macc>[\deE+.-]+)\s+"
    r"mDice: (?P<mdice>[\deE+.-]+)"
)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def read_logs(paths: list[Path]):
    train, val = {}, []
    pending_iteration = None
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            match = TRAIN_RE.search(line)
            if match:
                item = {key: float(value) for key, value in match.groupdict().items()}
                iteration = int(item.pop("iter"))
                train[iteration] = item
                pending_iteration = iteration
                continue
            match = VAL_RE.search(line)
            if match and pending_iteration is not None:
                item = {key: float(value) for key, value in match.groupdict().items()}
                item["iter"] = pending_iteration
                val.append(item)
    return train, {item["iter"]: item for item in val}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("dashboard", "convergence"), default="dashboard")
    args = parser.parse_args()

    train_map, val_map = read_logs(args.logs)
    if not train_map or not val_map:
        raise SystemExit("No train or validation records parsed from the supplied logs.")

    train_steps = np.array(sorted(train_map))
    losses = np.array([train_map[i]["loss"] for i in train_steps])
    lrs = np.array([train_map[i]["lr"] for i in train_steps])
    accs = np.array([train_map[i]["acc"] for i in train_steps])
    val_steps = np.array(sorted(val_map))

    if args.layout == "convergence":
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
        fig.suptitle("DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 complete convergence", fontsize=16, fontweight="bold")
        smooth = rolling_mean(losses, min(101, len(losses)))
        axes[0].plot(train_steps, losses, color="#9ecae1", linewidth=0.45, alpha=0.55, label="train loss (50-iter log mean)")
        axes[0].plot(train_steps, smooth, color="#08519c", linewidth=1.8, label="rolling mean (101 logs)")
        axes[0].set(title="Training loss", xlabel="Iteration", ylabel="Cross-entropy loss", xlim=(0, 450000))
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=10)

        for key, label, color in (
            ("miou", "mIoU", "#238b45"),
            ("mdice", "mDice", "#756bb1"),
            ("macc", "mAcc", "#e6550d"),
        ):
            axes[1].plot(val_steps, [val_map[i][key] for i in val_steps], "o-", markersize=4.5, linewidth=2, label=label, color=color)
        best_iter = max(val_steps, key=lambda i: val_map[i]["miou"])
        best = val_map[best_iter]["miou"]
        final_iter = val_steps[-1]
        final = val_map[final_iter]["miou"]
        axes[1].annotate(f"best mIoU = {best:.2f}% @ {best_iter:,}", xy=(best_iter, best), xytext=(0.56, 0.19), textcoords="axes fraction", color="#238b45", arrowprops={"arrowstyle": "->", "color": "#238b45"})
        axes[1].annotate(f"final mIoU = {final:.2f}%", xy=(final_iter, final), xytext=(0.56, 0.10), textcoords="axes fraction", color="#238b45", arrowprops={"arrowstyle": "->", "color": "#238b45"})
        axes[1].set(title="Validation metrics (2,000 images each)", xlabel="Iteration", ylabel="Percent (%)", xlim=(0, 450000), ylim=(0, 35))
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 complete training", fontsize=16, fontweight="bold")

    smooth = rolling_mean(losses, min(101, len(losses)))
    axes[0, 0].plot(train_steps, losses, color="#9ecae1", linewidth=0.45, alpha=0.55, label="50-iter logged loss")
    axes[0, 0].plot(train_steps, smooth, color="#08519c", linewidth=1.8, label="rolling mean (101 logs)")
    axes[0, 0].set(title="Training loss", xlabel="Iteration", ylabel="Cross-entropy loss")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(train_steps, accs, color="#d95f0e", linewidth=0.7, alpha=0.75)
    axes[0, 1].plot(train_steps, rolling_mean(accs, min(101, len(accs))), color="#a63603", linewidth=1.8)
    axes[0, 1].set(title="Decode-head training accuracy", xlabel="Iteration", ylabel="Accuracy (%)", ylim=(0, 100))

    axes[1, 0].plot(train_steps, lrs, color="#54278f", linewidth=1.7)
    axes[1, 0].set(title="Learning rate", xlabel="Iteration", ylabel="LR")
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for key, label, color in (
        ("miou", "mIoU", "#238b45"),
        ("mdice", "mDice", "#756bb1"),
        ("macc", "mAcc", "#e6550d"),
        ("aacc", "aAcc", "#3182bd"),
    ):
        axes[1, 1].plot(val_steps, [val_map[i][key] for i in val_steps], "o-", markersize=3.5, linewidth=1.5, label=label, color=color)
    best_iter = max(val_steps, key=lambda i: val_map[i]["miou"])
    best = val_map[best_iter]["miou"]
    axes[1, 1].annotate(f"best mIoU {best:.2f}% @ {best_iter:,}", xy=(best_iter, best), xytext=(0.03, 0.08), textcoords="axes fraction", color="#238b45", arrowprops={"arrowstyle": "->", "color": "#238b45"})
    axes[1, 1].set(title="Validation metrics (2,000 images)", xlabel="Iteration", ylabel="Percent (%)", ylim=(0, 100))
    axes[1, 1].legend(ncol=2, fontsize=9)

    for axis in axes.flat:
        axis.set_xlim(0, 450000)
        axis.grid(alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
