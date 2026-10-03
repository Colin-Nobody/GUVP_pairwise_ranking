#!/usr/bin/env python3
"""Evaluate retained Mapillary checkpoints on the fixed training subset.

This intentionally uses the deterministic, augmentation-free 2,000-image
training subset defined by ``make_train_eval_subset.py``.  It is comparable
between checkpoints but is not the stochastic optimisation-batch accuracy.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


METRIC_RE = re.compile(r"Iter\((?:val|test)\) \[2000/2000\].*?mIoU: ([\d.]+)")


def checkpoints(repo: Path) -> list[tuple[str, int, Path]]:
    output = repo / "output"
    main_80 = output / "deeplabv3plus_r50_mapillary_v2_80k"
    main_450 = output / "deeplabv3plus_r50_mapillary_v2_450k"
    finetune = output / "deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k"
    return [
        *( ("main_450k", step, main_80 / f"iter_{step}.pth") for step in (80000, 84000, 88000, 92000, 96000) ),
        ("main_450k", 432000, main_450 / "best_mIoU_iter_432000.pth"),
        *( ("main_450k", step, main_450 / f"iter_{step}.pth") for step in (436000, 440000, 444000, 448000, 450000) ),
        *( ("finetune_140k", step, finetune / f"iter_{step}.pth") for step in (132000, 134000, 136000, 138000, 140000) ),
    ]


def load_done(output_csv: Path) -> set[tuple[str, int]]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="") as handle:
        return {(row["run"], int(row["iteration"])) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/colin/GUVP_pairwise_ranking"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    mmseg = repo / "segmentation/vendor/mmsegmentation"
    config = repo / "segmentation/mapillary_vistas/configs/deeplabv3plus_r50-d8_mapillary-v2-train-eval-2000.py"
    python = Path("/home/colin/miniconda3/envs/guvp-seg/bin/python")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output.parent / "train_eval_2000_logs"
    logs_dir.mkdir(exist_ok=True)
    done = load_done(args.output)
    write_header = not args.output.exists()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo / 'segmentation/mapillary_vistas'}:{mmseg}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    with args.output.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run", "iteration", "checkpoint", "train_eval_miou_percent"],
        )
        if write_header:
            writer.writeheader()
        for run, iteration, checkpoint in checkpoints(repo):
            key = (run, iteration)
            if key in done:
                print(f"[skip] {run} {iteration:,}", flush=True)
                continue
            if not checkpoint.is_file():
                print(f"[missing] {checkpoint}", flush=True)
                continue
            log = logs_dir / f"{run}_iter_{iteration}.log"
            matches = METRIC_RE.findall(log.read_text(errors="replace")) if log.exists() else []
            if matches:
                print(f"[reuse] {run} {iteration:,}: existing evaluation log", flush=True)
            else:
                command = [
                    str(python), "tools/test.py", str(config), str(checkpoint),
                    "--work-dir", str(logs_dir / f"{run}_iter_{iteration}"),
                ]
                print(f"[evaluate] {run} {iteration:,}: {checkpoint.name}", flush=True)
                with log.open("w") as handle:
                    subprocess.run(command, cwd=mmseg, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
                matches = METRIC_RE.findall(log.read_text(errors="replace"))
            if not matches:
                raise RuntimeError(f"No final mIoU in {log}")
            miou = float(matches[-1])
            writer.writerow({
                "run": run,
                "iteration": iteration,
                "checkpoint": str(checkpoint),
                "train_eval_miou_percent": f"{miou:.4f}",
            })
            csv_file.flush()
            print(f"[done] {run} {iteration:,}: train-subset mIoU={miou:.2f}%", flush=True)


if __name__ == "__main__":
    main()
