"""Render convergence charts from an MMSegmentation JSON-lines scalar log."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def rolling_mean(values, window=25):
    result = []
    for index in range(len(values)):
        left = max(0, index - window + 1)
        result.append(sum(values[left:index + 1]) / (index - left + 1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scalars", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.scalars.read_text().splitlines() if line]
    train = [item for item in records if "loss" in item]
    val = [item for item in records if "mIoU" in item]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle("DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 convergence", fontsize=15, fontweight="bold")

    train_steps = [item["step"] for item in train]
    losses = [item["loss"] for item in train]
    axes[0].plot(train_steps, losses, color="#9ecae1", linewidth=0.8, alpha=0.7, label="train loss (50-iter log mean)")
    axes[0].plot(train_steps, rolling_mean(losses), color="#08519c", linewidth=2, label="rolling mean (25 logs)")
    axes[0].set(title="Training loss", xlabel="Iteration", ylabel="Cross-entropy loss", xlim=(0, 80000))
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    val_steps = [item["step"] for item in val]
    for key, label, color in (("mIoU", "mIoU", "#238b45"), ("mDice", "mDice", "#756bb1"), ("mAcc", "mAcc", "#e6550d")):
        values = [item[key] for item in val]
        axes[1].plot(val_steps, values, "o-", linewidth=2, markersize=5, label=label, color=color)
    axes[1].annotate("final mIoU = 20.60%", xy=(80000, val[-1]["mIoU"]), xytext=(47000, 13),
                     arrowprops={"arrowstyle": "->", "color": "#238b45"}, color="#238b45")
    axes[1].set(title="Validation metrics (2,000 images each)", xlabel="Iteration", ylabel="Percent (%)", xlim=(0, 80000), ylim=(0, 30))
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
