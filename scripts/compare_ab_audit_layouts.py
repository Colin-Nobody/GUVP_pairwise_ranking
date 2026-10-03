#!/usr/bin/env python3
"""Compare label-after and label-before A/B audit summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "natural_a_selection": "Natural-image A selection",
    "natural_swap_consistency": "Natural-image swap consistency",
    "visual_access_accuracy": "Visual-access accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-after", type=Path, required=True)
    parser.add_argument("--labels-before", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layouts = {
        "labels_after": json.loads(args.labels_after.read_text(encoding="utf-8")),
        "labels_before": json.loads(args.labels_before.read_text(encoding="utf-8")),
    }
    rows: list[dict[str, object]] = []
    for layout, payload in layouts.items():
        for model, summary in payload["runs"].items():
            row: dict[str, object] = {"layout": layout, "model": model}
            for metric in METRICS:
                row[metric] = summary[metric]["rate"]
                row[f"{metric}_ci_low"] = summary[metric]["clopper_pearson_95pct"][0]
                row[f"{metric}_ci_high"] = summary[metric]["clopper_pearson_95pct"][1]
            rows.append(row)
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "layout_comparison.csv", index=False)

    models = list(layouts["labels_after"]["runs"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    width = 0.36
    x = list(range(len(models)))
    for axis, (metric, title) in zip(axes, METRICS.items()):
        for offset, layout in ((-width / 2, "labels_after"), (width / 2, "labels_before")):
            subset = frame[frame["layout"].eq(layout)].set_index("model").loc[models]
            positions = [value + offset for value in x]
            axis.bar(positions, subset[metric], width=width, label=layout.replace("_", " "))
        axis.set_title(title)
        axis.set_xticks(x, [name.replace("qwen2_5_", "Q2.5-").replace("qwen3_", "Q3-").upper() for name in models])
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[-1].legend(loc="lower right")
    fig.suptitle("A/B audit: effect of placing labels before images")
    fig.tight_layout()
    fig.savefig(args.output_dir / "layout_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote layout comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
