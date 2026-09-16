#!/usr/bin/env python3
"""Monitor one model's formal MEC+BPC audit CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--mec-samples", type=int, default=3)
    parser.add_argument("--model-fragment", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() or result.stderr.strip() or "unavailable"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def snapshot(audit: Path, total: int, model_fragment: str, mec_samples: int) -> str:
    rows = read_rows(audit)
    completed = len(rows)
    percent = completed / total * 100 if total else 0.0
    statuses = Counter(row.get("status", "") for row in rows)
    rate_text = "warming up"
    eta_text = "unknown"
    if completed >= 2:
        first = datetime.fromisoformat(rows[0]["timestamp_utc"])
        last = datetime.fromisoformat(rows[-1]["timestamp_utc"])
        seconds = (last - first).total_seconds()
        if seconds > 0:
            rate = (completed - 1) / seconds
            rate_text = f"{rate:.3f} pairs/s ({3600 * rate:.0f} pairs/h)"
            eta_text = str(timedelta(seconds=round(max(0, total - completed) / rate)))

    gpu = command_output([
        "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ])
    processes = command_output(["pgrep", "-af", "vlm_pairwise.py"])
    model_processes = "\n".join(
        line for line in processes.splitlines() if model_fragment in line
    ) or "not running"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    invalid = sum(count for status, count in statuses.items() if status.startswith("invalid"))
    return "\n".join([
        f"Updated: {now}",
        f"Completed unordered pairs: {completed}/{total} ({percent:.2f}%)",
        f"MEC directional samples: {2 * mec_samples * completed}/{2 * mec_samples * total}",
        f"Calibrated={statuses['calibrated']} ties={statuses['calibrated_tie']} invalid={invalid}",
        f"Accepted rate: {statuses['calibrated'] / completed:.1%}" if completed else "Accepted rate: n/a",
        f"Rate: {rate_text}",
        f"ETA: {eta_text}",
        f"GPU (used, free, util, temp): {gpu}",
        f"Process: {model_processes}",
    ])


def main() -> None:
    args = parse_args()
    while True:
        print("\033[2J\033[H" + snapshot(
            args.audit, args.total, args.model_fragment, args.mec_samples
        ), flush=True)
        rows = read_rows(args.audit)
        if args.once or len(rows) >= args.total:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
