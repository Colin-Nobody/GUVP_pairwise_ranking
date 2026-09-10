#!/usr/bin/env python3
"""Summarize one model's formal MEC+BPC audit without mixing configurations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rate(count: int, total: int) -> float | None:
    return count / total if total else None


def numbers(row: dict[str, str], field: str) -> list[float]:
    return [float(value) for value in json.loads(row[field] or "[]")]


def main() -> None:
    args = parse_args()
    with args.audit.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Audit contains no completed pairs: {args.audit}")

    identity_fields = [
        "model", "prompt_version", "prompt_sha256", "decode_mode", "mec_samples", "temperature",
        "max_new_tokens", "seed",
    ]
    identities = {field: sorted({row[field] for row in rows}) for field in identity_fields}
    mixed = {field: values for field, values in identities.items() if len(values) != 1}
    if mixed:
        raise ValueError(f"Audit mixes experiment configurations: {mixed}")

    statuses = Counter(row["status"] for row in rows)
    accepted = statuses["calibrated"]
    ties = statuses["calibrated_tie"]
    mec_samples = int(identities["mec_samples"][0])
    margins = [abs(float(row["calibrated_margin"])) for row in rows if row["calibrated_margin"]]
    entropies = [float(row["bpde"]) for row in rows if row["bpde"]]

    first_position_scores: list[float] = []
    second_position_scores: list[float] = []
    for row in rows:
        # AB: physical A is first, physical B second. BA: physical B is first, physical A second.
        first_position_scores += numbers(row, "scores_ab_image_a") + numbers(row, "scores_ba_image_b")
        second_position_scores += numbers(row, "scores_ab_image_b") + numbers(row, "scores_ba_image_a")
    lower_score_is_cleaner = "lower_score_is_cleaner" in identities["decode_mode"][0]
    mean_position_advantage = None
    if first_position_scores and second_position_scores:
        raw_position_difference = (
            statistics.fmean(first_position_scores) - statistics.fmean(second_position_scores)
        )
        mean_position_advantage = (
            -raw_position_difference if lower_score_is_cleaner else raw_position_difference
        )

    summary = {
        "model": identities["model"][0],
        "prompt_version": identities["prompt_version"][0],
        "prompt_sha256": identities["prompt_sha256"][0],
        "score_direction": (
            "lower_is_cleaner" if lower_score_is_cleaner else "higher_is_cleaner"
        ),
        "mec_samples_per_position": mec_samples,
        "temperature": float(identities["temperature"][0]),
        "completed_unordered_pairs": len(rows),
        "mec_directional_samples": 2 * mec_samples * len(rows),
        "calibrated_votes": accepted,
        "calibrated_vote_rate": rate(accepted, len(rows)),
        "calibrated_ties": ties,
        "invalid_pairs": len(rows) - accepted - ties,
        "status_counts": dict(sorted(statuses.items())),
        "mean_absolute_calibrated_margin": statistics.fmean(margins) if margins else None,
        "mean_bpde": statistics.fmean(entropies) if entropies else None,
        "mean_first_position_score_advantage": mean_position_advantage,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pair_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    def decimal(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "# 正式 MEC+BPC 审计摘要", "",
        f"- 模型：`{summary['model']}`",
        f"- 提示词版本：`{summary['prompt_version']}`",
        f"- MEC：每个位置 k={mec_samples}，temperature={summary['temperature']}",
        f"- 已完成无序图片对：{summary['completed_unordered_pairs']}",
        f"- MEC 方向采样次数：{summary['mec_directional_samples']}",
        f"- 有效校准票：{accepted}（{percent(summary['calibrated_vote_rate'])}）",
        f"- 校准平局：{ties}",
        f"- 无效对：{summary['invalid_pairs']}",
        f"- 平均绝对校准分差：{decimal(summary['mean_absolute_calibrated_margin'])}",
        f"- 平均 BPDE：{decimal(summary['mean_bpde'])}",
        f"- 第一位置平均分优势：{decimal(summary['mean_first_position_score_advantage'])}", "",
        "## 状态计数", "",
    ]
    lines.extend(f"- `{status}`：{count}" for status, count in sorted(statuses.items()))
    (args.output_dir / "PAIR_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit summary: {args.output_dir / 'pair_audit_summary.json'}")


if __name__ == "__main__":
    main()
