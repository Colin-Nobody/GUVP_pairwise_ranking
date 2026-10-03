#!/usr/bin/env python3
"""Analyze A/B swap consistency, null controls, and visual-access controls."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=CSV (repeat for each model)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/vlm_ab_position_audit"))
    return parser.parse_args()


def interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [low, high]


def proportion(successes: int, total: int) -> dict[str, object]:
    return {"count": successes, "total": total, "rate": successes / total if total else None,
            "clopper_pearson_95pct": interval(successes, total)}


def run_summary(label: str, frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    valid = frame[frame["status"].eq("ok")].copy()
    natural = valid[valid["case_type"].eq("natural_vp")].copy()
    paired = natural.pivot(index="case_id", columns="order", values=["choice", "chosen_image"])
    paired.columns = [f"{kind}_{order}" for kind, order in paired.columns]
    paired["both_decisive"] = paired["choice_AB"].isin(["A", "B"]) & paired["choice_BA"].isin(["A", "B"])
    paired["any_tie"] = paired["choice_AB"].eq("T") | paired["choice_BA"].eq("T")
    paired["swap_consistent"] = paired["both_decisive"] & paired["chosen_image_AB"].eq(paired["chosen_image_BA"])

    identical = valid[valid["case_type"].eq("identical_vp")]
    access = valid[valid["case_type"].eq("visual_access")].copy()
    access["correct"] = access["chosen_image"].eq(access["expected_image"])
    access_paired = access.pivot(index="case_id", columns="order", values="chosen_image")
    access_consistent = access_paired["AB"].eq(access_paired["BA"])

    natural_a = int(natural["choice"].eq("A").sum())
    natural_t = int(natural["choice"].eq("T").sum())
    identical_a = int(identical["choice"].eq("A").sum())
    identical_t = int(identical["choice"].eq("T").sum())
    access_correct = int(access["correct"].sum())
    swap_ok = int(paired["swap_consistent"].sum())
    both_decisive = int(paired["both_decisive"].sum())
    abstained_pairs = int(paired["any_tie"].sum())
    access_swap_ok = int(access_consistent.sum())
    access_rate = access_correct / len(access)
    natural_swap_rate = swap_ok / len(paired)
    identical_a_rate = identical_a / len(identical)
    natural_a_rate = natural_a / len(natural)
    if access_rate < 0.9:
        diagnosis = "visual_access_or_label_mapping_failure"
    elif natural_swap_rate < 0.7 and abs(identical_a_rate - 0.5) >= 0.25 and abs(natural_a_rate - 0.5) >= 0.15:
        diagnosis = "position_prior_dominates_natural_judgment"
    elif natural_swap_rate < 0.7:
        diagnosis = "natural_vp_judgment_unstable_or_ambiguous"
    else:
        diagnosis = "natural_vp_judgment_substantially_swap_consistent"

    summary = {
        "label": label,
        "model_values": sorted(valid["model"].unique().tolist()),
        "rows": len(frame),
        "valid_rows": len(valid),
        "invalid_rows": len(frame) - len(valid),
        "natural_a_selection": proportion(natural_a, len(natural)),
        "natural_tie_selection": proportion(natural_t, len(natural)),
        "natural_swap_consistency": proportion(swap_ok, len(paired)),
        "natural_decisive_pair_coverage": proportion(both_decisive, len(paired)),
        "natural_swap_consistency_given_decisive": proportion(swap_ok, both_decisive),
        "natural_pair_abstention": proportion(abstained_pairs, len(paired)),
        "identical_image_a_selection": proportion(identical_a, len(identical)),
        "identical_image_tie_selection": proportion(identical_t, len(identical)),
        "visual_access_accuracy": proportion(access_correct, len(access)),
        "visual_access_swap_consistency": proportion(access_swap_ok, len(access_consistent)),
        "diagnostic_classification": diagnosis,
    }
    paired = paired.reset_index()
    paired["model_label"] = label
    return summary, paired


def main() -> None:
    args = parse_args()
    runs: dict[str, Path] = {}
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            raise ValueError("Each --run must be LABEL=CSV")
        runs[label] = Path(path)

    summaries: dict[str, object] = {}
    paired_frames: list[pd.DataFrame] = []
    choices: dict[str, pd.Series] = {}
    for label, path in runs.items():
        frame = pd.read_csv(path)
        summary, paired = run_summary(label, frame)
        summaries[label] = summary
        paired_frames.append(paired)
        consistent = paired[paired["swap_consistent"]].set_index("case_id")["chosen_image_AB"]
        choices[label] = consistent

    cross_model: dict[str, object] = {}
    for first, second in combinations(runs, 2):
        matched = pd.concat([choices[first].rename(first), choices[second].rename(second)], axis=1, join="inner").dropna()
        agreements = int(matched[first].eq(matched[second]).sum())
        cross_model[f"{first}_vs_{second}"] = {
            "jointly_swap_consistent_cases": len(matched),
            "same_physical_winner": proportion(agreements, len(matched)),
        }

    result = {"runs": summaries, "cross_model_on_jointly_swap_consistent_natural_cases": cross_model}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.concat(paired_frames, ignore_index=True).to_csv(args.output_dir / "natural_pair_diagnostics.csv", index=False)

    lines = ["# 三模型 A/B 位置与视觉访问诊断", ""]
    for label, summary in summaries.items():
        lines.extend([
            f"## {label}", "",
            f"- 自然图 A 选择率：{summary['natural_a_selection']['rate']:.1%}",
            f"- 自然图交换一致率：{summary['natural_swap_consistency']['rate']:.1%}",
            f"- 自然图 T 选择率：{summary['natural_tie_selection']['rate']:.1%}",
            f"- 同图对照 A 选择率：{summary['identical_image_a_selection']['rate']:.1%}",
            f"- 同图对照 T 选择率：{summary['identical_image_tie_selection']['rate']:.1%}",
            f"- 视觉访问对照准确率：{summary['visual_access_accuracy']['rate']:.1%}",
            f"- 诊断分类：`{summary['diagnostic_classification']}`", "",
        ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary: {args.output_dir / 'audit_summary.json'}")


if __name__ == "__main__":
    main()
