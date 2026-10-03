"""Plot the live convergence curve from an MMSegmentation plain-text log."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"Iter\(train\) \[\s*(?P<step>\d+)/(?P<total>\d+)\].*? loss: (?P<loss>[\d.]+)"
)
VAL_RE = re.compile(
    r"aAcc: (?P<aAcc>[\d.]+)\s+mIoU: (?P<mIoU>[\d.]+)\s+"
    r"mAcc: (?P<mAcc>[\d.]+)\s+mDice: (?P<mDice>[\d.]+)"
)


def rolling_mean(values, window=25):
    return [
        sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
        for i in range(len(values))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=450_000)
    args = parser.parse_args()

    train, val = [], []
    last_train_step = None
    for line in args.log.read_text(errors="replace").splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            last_train_step = int(train_match["step"])
            train.append((last_train_step, float(train_match["loss"])))
            continue
        val_match = VAL_RE.search(line)
        if val_match and last_train_step is not None:
            val.append((
                last_train_step,
                float(val_match["mIoU"]),
                float(val_match["mDice"]),
                float(val_match["mAcc"]),
            ))

    if not train or not val:
        raise SystemExit("No complete train/validation records found in the log.")

    train_steps, losses = zip(*train)
    val_steps, mious, mdices, maccs = zip(*val)
    final_step, final_miou = val_steps[-1], mious[-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle(
        "DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 baseline-resize convergence",
        fontsize=15,
        fontweight="bold",
    )

    axes[0].plot(train_steps, losses, color="#9ecae1", linewidth=0.8, alpha=0.7,
                 label="train loss (50-iter log mean)")
    axes[0].plot(train_steps, rolling_mean(losses), color="#08519c", linewidth=2,
                 label="rolling mean (25 logs)")
    axes[0].set(title="Training loss", xlabel="Iteration", ylabel="Cross-entropy loss",
                xlim=(0, args.max_iters))
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for values, label, color in ((mious, "mIoU", "#238b45"),
                                 (mdices, "mDice", "#756bb1"),
                                 (maccs, "mAcc", "#e6550d")):
        axes[1].plot(val_steps, values, "o-", linewidth=2, markersize=5,
                     label=label, color=color)
    text_step = max(0, final_step - args.max_iters * 0.20)
    axes[1].annotate(
        f"latest mIoU = {final_miou:.2f}%",
        xy=(final_step, final_miou),
        xytext=(text_step, max(0.5, final_miou - 5)),
        arrowprops={"arrowstyle": "->", "color": "#238b45"},
        color="#238b45",
    )
    metric_max = max(max(mious), max(mdices), max(maccs))
    axes[1].set(title="Validation metrics (2,000 images each)", xlabel="Iteration",
                ylabel="Percent (%)", xlim=(0, args.max_iters), ylim=(0, max(20, metric_max + 5)))
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
