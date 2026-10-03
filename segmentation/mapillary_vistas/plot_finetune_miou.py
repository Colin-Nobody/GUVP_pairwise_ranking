#!/usr/bin/env python3
"""Plot validation mIoU over a fine-tuning run from an MMEngine text log."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(r"Iter\(train\) \[\s*(\d+)/")
VAL_RE = re.compile(r"Iter\(val\) \[2000/2000\].*?mIoU: ([\d.]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=float, default=None)
    args = parser.parse_args()

    current_iter = None
    points: dict[int, float] = {}
    for line in args.log.read_text(errors="replace").splitlines():
        train = TRAIN_RE.search(line)
        if train:
            current_iter = int(train.group(1))
            continue
        val = VAL_RE.search(line)
        if val and current_iter is not None:
            points[current_iter] = float(val.group(1))

    if not points:
        raise SystemExit("No complete validation mIoU records were found.")

    iterations = sorted(points)
    values = [points[i] for i in iterations]
    best_iter = max(iterations, key=points.get)
    best = points[best_iter]
    final_iter, final = iterations[-1], values[-1]

    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    fig.suptitle("DeepLabV3+ R50-D8 — Mapillary Vistas v2.0 fine-tuning (LR 3e-5)", fontsize=15, fontweight="bold")
    axis.plot(iterations, values, "o-", color="#238b45", linewidth=2.2, markersize=5.5, label="validation mIoU")
    if args.baseline is not None:
        axis.axhline(args.baseline, color="#6a51a3", linestyle="--", linewidth=1.6, label=f"fine-tune start: {args.baseline:.2f}%")
    axis.scatter([best_iter], [best], color="#d94801", s=80, zorder=3, label=f"best: {best:.2f}% @ {best_iter:,}")
    axis.scatter([final_iter], [final], color="#08519c", s=70, zorder=3, label=f"final: {final:.2f}% @ {final_iter:,}")
    axis.annotate(f"+{best - args.baseline:.2f} mIoU" if args.baseline is not None else f"{best:.2f}%", xy=(best_iter, best), xytext=(best_iter - 33000, best - 0.55), arrowprops={"arrowstyle": "->", "color": "#d94801"}, color="#d94801", fontsize=11)
    axis.set(xlabel="Fine-tuning iteration", ylabel="mIoU (%)", xlim=(0, max(iterations)), ylim=(min(values + ([args.baseline] if args.baseline is not None else [])) - 0.5, max(values) + 0.7))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
