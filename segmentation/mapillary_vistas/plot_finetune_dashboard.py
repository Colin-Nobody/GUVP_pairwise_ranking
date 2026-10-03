#!/usr/bin/env python3
"""Plot learning rate, training loss, and validation mIoU for fine-tuning."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE = re.compile(
    r"Iter\(train\) \[\s*(?P<iter>\d+)/\d+\].*?lr: (?P<lr>[\deE+.-]+).*?loss: (?P<loss>[\deE+.-]+)"
)
VAL_RE = re.compile(r"Iter\(val\) \[2000/2000\].*?mIoU: (?P<miou>[\d.]+)")


def rolling_mean(values: np.ndarray, window: int = 101) -> np.ndarray:
    window = min(window, len(values))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train: dict[int, tuple[float, float]] = {}
    val: dict[int, float] = {}
    current_iter = None
    for line in args.log.read_text(errors="replace").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            current_iter = int(match["iter"])
            train[current_iter] = (float(match["lr"]), float(match["loss"]))
            continue
        match = VAL_RE.search(line)
        if match and current_iter is not None:
            val[current_iter] = float(match["miou"])
    if not train or not val:
        raise SystemExit("No train loss/LR or validation mIoU records found")

    train_steps = np.array(sorted(train))
    lrs = np.array([train[i][0] for i in train_steps])
    losses = np.array([train[i][1] for i in train_steps])
    val_steps = np.array(sorted(val))
    miou = np.array([val[i] for i in val_steps])
    best_index = int(np.argmax(miou))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    fig.suptitle("DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 fine-tuning (PolyLR)", fontsize=16, fontweight="bold")

    axes[0].plot(train_steps, lrs, color="#54278f", linewidth=2)
    axes[0].set(title="Learning rate: PolyLR (3e-5 → 1e-6, power=0.9)", ylabel="Learning rate")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[1].plot(train_steps, losses, color="#9ecae1", alpha=0.6, linewidth=0.5, label="50-iteration logged loss")
    axes[1].plot(train_steps, rolling_mean(losses), color="#08519c", linewidth=1.8, label="rolling mean (101 logs)")
    axes[1].set(title="Training loss", ylabel="Cross-entropy loss")
    axes[1].legend(loc="upper right")

    axes[2].plot(val_steps, miou, "o-", color="#238b45", linewidth=2, markersize=5, label="validation mIoU")
    axes[2].scatter([val_steps[best_index]], [miou[best_index]], color="#d94801", s=75, zorder=3, label=f"best {miou[best_index]:.2f}% @ {val_steps[best_index]:,}")
    axes[2].set(title="Validation mIoU (2,000 images per evaluation)", xlabel="Fine-tuning iteration", ylabel="mIoU (%)")
    axes[2].legend(loc="lower right")

    for axis in axes:
        axis.set_xlim(0, train_steps.max())
        axis.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
