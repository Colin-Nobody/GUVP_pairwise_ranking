#!/usr/bin/env python3
"""Stage the HPC Qwen3 10k results in a separate Git worktree, without GPU."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

root = Path("/share/home/sgxcw3/GUVP_pairwise_ranking")
worktree = Path("/share/home/sgxcw3/guvp-qwen3-10k-upload")
branch = "qwen3-10k-comparison"
source = root / "output/vlm_qwen3_10k_v8_pairbatch50_mec3"
comparison = root / "output/qwen_compare/result"
run = lambda *args, **kw: subprocess.run(args, check=True, **kw)
if worktree.exists():
    raise SystemExit("Worktree already exists; inspect it before proceeding")
for f in ("pair_audit.csv", "vlm_pairwise_votes.csv", "perception_scores.csv", "ranking_summary.json", "timing.json"):
    if not (source / f).is_file():
        raise SystemExit("Missing source: " + str(source / f))
for f in ("comparison_summary.json", "per_image_score_comparison.csv", "per_pair_winner_comparison.csv"):
    if not (comparison / f).is_file():
        raise SystemExit("Missing comparison: " + str(comparison / f))
run("git", "-C", str(root), "fetch", "origin", branch)
env = dict(os.environ)
env["GIT_LFS_SKIP_SMUDGE"] = "1"
run("git", "-C", str(root), "worktree", "add", "--detach", str(worktree), "origin/" + branch, env=env)
target = worktree / "output/vlm_qwen3_10k_v8_pairbatch50_mec3"
if target.exists():
    raise SystemExit("Target directory already exists; no source files were overwritten")
target.mkdir(parents=True)
files = ("pair_audit.csv", "vlm_pairwise_votes.csv", "perception_scores.csv", "ranking_summary.json", "timing.json", "progress.log", "run.log", "score_comparison.png", "score_distributions.png")
for name in files:
    if (source / name).is_file():
        shutil.copy2(source / name, target / name)
for name in ("pair_batch1_mec3", "pair_batch50_mec3"):
    if (source / "benchmark" / name).is_dir():
        shutil.copytree(source / "benchmark" / name, target / "benchmark" / name)
dest = worktree / "output/qwen_compare/result"
if dest.exists():
    raise SystemExit("Comparison destination already exists; inspect worktree")
dest.mkdir(parents=True)
for name in ("comparison_summary.json", "per_image_score_comparison.csv", "per_pair_winner_comparison.csv"):
    shutil.copy2(comparison / name, dest / name)
large_paths = [
    "output/vlm_qwen3_10k_v8_pairbatch50_mec3/pair_audit.csv",
    "output/vlm_qwen3_10k_v8_pairbatch50_mec3/vlm_pairwise_votes.csv",
]
run("git", "-C", str(worktree), "lfs", "track", *large_paths)
all_files = sorted([p for base in (target, dest) for p in base.rglob("*") if p.is_file()])
manifest = worktree / "docs/QWEN3_10K_SHA256.txt"
with manifest.open("w") as out:
    for path in all_files:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        out.write(h.hexdigest() + "  " + str(path.relative_to(worktree)) + "\n")
run("git", "-C", str(worktree), "add", ".gitattributes", "docs/QWEN3_10K_SHA256.txt")
run("git", "-C", str(worktree), "add", "-f", "output/vlm_qwen3_10k_v8_pairbatch50_mec3", "output/qwen_compare/result")
run("git", "-C", str(worktree), "status", "--short")
run("git", "-C", str(worktree), "commit", "-m", "Add Qwen3 10k results and cross-model analysis")
run("git", "-C", str(worktree), "show", "--stat", "--oneline", "HEAD")
print("READY TO PUSH FROM", worktree, flush=True)
