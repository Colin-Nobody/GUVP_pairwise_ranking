#!/usr/bin/env python3
"""Project full MEC-batch runtime from matched batch-1 and higher-batch HPC pilots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"completed_pairs", "wall_seconds", "model_load_seconds"}
    if not required.issubset(data):
        raise ValueError(f"{path} missing {sorted(required - set(data))}")
    if data["completed_pairs"] <= 0:
        raise ValueError(f"{path} has no completed pilot pairs")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch1", type=Path, required=True)
    parser.add_argument("--batch-n", type=Path, required=True,
                        help="Timing JSON for the target MEC batch size")
    parser.add_argument("--pair-batch-size", type=int, default=10)
    parser.add_argument("--total-pairs", type=int, default=65_000)
    parser.add_argument("--max-hours", type=float, default=70.0,
                        help="Do not start the full run if projected runtime exceeds this safety limit")
    args = parser.parse_args()
    one, target = load(args.batch1), load(args.batch_n)

    def projected(data: dict[str, float]) -> float:
        measured = max(0.0, data["wall_seconds"] - data["model_load_seconds"])
        return data["model_load_seconds"] + measured / data["completed_pairs"] * args.total_pairs

    seconds1, seconds_target = projected(one), projected(target)
    body1 = max(1.0, one["wall_seconds"] - one["model_load_seconds"]) / one["completed_pairs"]
    body_target = max(1.0, target["wall_seconds"] - target["model_load_seconds"]) / target["completed_pairs"]
    speedup = body1 / body_target
    report = {
        "pilot_pairs_batch1": int(one["completed_pairs"]),
        "pilot_pairs_target_batch": int(target["completed_pairs"]),
        "pair_batch_size": args.pair_batch_size,
        "measured_target_batch_speedup": round(speedup, 3),
        "estimated_full_hours_batch1": round(seconds1 / 3600, 2),
        "estimated_full_hours_target_batch": round(seconds_target / 3600, 2),
        "target_pilot_peak_cuda_allocated_gib": target.get("peak_cuda_allocated_gib"),
        "full_pairs": args.total_pairs,
        "safety_limit_hours": args.max_hours,
        "full_run_allowed": seconds_target <= args.max_hours * 3600,
        "note": "Estimate uses this allocated GPU and excludes possible queue, filesystem, and model-download delays.",
    }
    print(json.dumps(report, indent=2))
    if seconds_target > args.max_hours * 3600:
        print("Projected runtime exceeds the safety limit; full inference was not started.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
