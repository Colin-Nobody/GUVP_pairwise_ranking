#!/usr/bin/env python3
"""Monitor an actively appended VLM pairwise-vote CSV."""

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
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() or result.stderr.strip() or "unavailable"


def snapshot(votes: Path, total: int) -> str:
    rows: list[dict[str, str]] = []
    if votes.exists():
        with votes.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    completed = len(rows)
    percent = completed / total * 100 if total else 0.0
    statuses = Counter(row.get("status", "") for row in rows)
    choices = Counter(row.get("raw_response", "").strip().upper() for row in rows)
    rate_text = "warming up"
    eta_text = "unknown"
    if completed >= 2:
        first = datetime.fromisoformat(rows[0]["timestamp_utc"])
        last = datetime.fromisoformat(rows[-1]["timestamp_utc"])
        seconds = (last - first).total_seconds()
        if seconds > 0:
            rate = (completed - 1) / seconds
            rate_text = f"{rate:.3f} pairs/s ({3600 * rate:.0f} pairs/h)"
            eta_text = str(timedelta(seconds=round((total - completed) / rate)))

    gpu = command_output([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ])
    processes = command_output(["pgrep", "-af", "vlm_pairwise.py"])
    qwen3_processes = "\n".join(line for line in processes.splitlines() if "Qwen3-VL-8B" in line) or "not running"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return "\n".join([
        f"Updated: {now}",
        f"Progress: {completed}/{total} ({percent:.2f}%)",
        f"Status: ok={statuses['ok']} invalid={completed - statuses['ok']}",
        f"Choices: A={choices['A']} B={choices['B']}",
        f"Rate: {rate_text}",
        f"ETA: {eta_text}",
        f"GPU (used, free, util, temp): {gpu}",
        f"Process: {qwen3_processes}",
    ])


def main() -> None:
    args = parse_args()
    while True:
        print("\033[2J\033[H" + snapshot(args.votes, args.total), flush=True)
        if args.once:
            return
        if args.votes.exists():
            with args.votes.open(newline="", encoding="utf-8") as handle:
                completed = sum(1 for _ in csv.DictReader(handle))
            if completed >= args.total:
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
