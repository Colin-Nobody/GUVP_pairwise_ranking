"""Render loss, validation mIoU and learning-rate curves for the first 450k run."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"Iter\(train\) \[\s*(?P<step>\d+)/(?P<total>\d+)\].*?"
    r"lr: (?P<lr>[\deE+\-.]+).*? loss: (?P<loss>[\d.]+)"
)
VAL_RE = re.compile(r"aAcc: [\d.]+\s+mIoU: (?P<mIoU>[\d.]+)")


def rolling_mean(values, window=101):
    return [
        sum(values[max(0, index - window + 1):index + 1]) / min(index + 1, window)
        for index in range(len(values))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=450_000)
    args = parser.parse_args()

    train, validation = [], []
    last_train_step = None
    for line in args.log.read_text(errors="replace").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            last_train_step = int(match["step"])
            train.append((last_train_step, float(match["loss"]), float(match["lr"])))
            continue
        match = VAL_RE.search(line)
        if match and last_train_step is not None:
            validation.append((last_train_step, float(match["mIoU"])))

    if not train or not validation:
        raise SystemExit("No complete train/validation records found in the log.")

    steps, losses, lrs = zip(*train)
    val_steps, mious = zip(*validation)
    best_index = max(range(len(mious)), key=mious.__getitem__)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), constrained_layout=True)
    fig.suptitle(
        "DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 baseline-resize: first 450k iterations",
        fontsize=15,
        fontweight="bold",
    )

    axes[0].plot(steps, losses, color="#9ecae1", linewidth=0.7, alpha=0.65,
                 label="train loss (50-iter log mean)")
    axes[0].plot(steps, rolling_mean(losses), color="#08519c", linewidth=2,
                 label="rolling mean (101 logs)")
    axes[0].set(title="Training loss", xlabel="Iteration", ylabel="Cross-entropy loss",
                xlim=(0, args.max_iters))
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(val_steps, mious, "o-", color="#238b45", linewidth=2, markersize=4,
                 label="validation mIoU")
    axes[1].scatter([val_steps[best_index]], [mious[best_index]], color="#e6550d", zorder=3,
                    label=f"best: {mious[best_index]:.2f}% @ {val_steps[best_index]:,}")
    axes[1].annotate(
        f"final = {mious[-1]:.2f}%",
        xy=(val_steps[-1], mious[-1]),
        xytext=(max(0, val_steps[-1] - 135000), max(0.5, mious[-1] - 4)),
        arrowprops={"arrowstyle": "->", "color": "#238b45"}, color="#238b45",
    )
    axes[1].set(title="Validation mIoU (2,000 images)", xlabel="Iteration", ylabel="mIoU (%)",
                xlim=(0, args.max_iters), ylim=(0, max(25, max(mious) + 3)))
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    axes[2].plot(steps, lrs, color="#756bb1", linewidth=2, label="actual learning rate")
    axes[2].set(title="PolyLR schedule", xlabel="Iteration", ylabel="Learning rate",
                xlim=(0, args.max_iters))
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
