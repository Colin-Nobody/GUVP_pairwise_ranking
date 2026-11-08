#!/usr/bin/env python3
"""Prepare separate Qwen3 pilot/full scripts based on the existing HPC batch-50 pipeline."""
from pathlib import Path
import py_compile
root = Path(__file__).resolve().parents[1]
runner_path = root / "hpc_vlm_10k/run_b50_5d.sh"
job_path = root / "hpc_vlm_10k/submit_b50_5d.sbatch"
runner = runner_path.read_text()
job = job_path.read_text()
changes = [
 ('model="Qwen/Qwen2.5-VL-7B-Instruct"', 'model="Qwen/Qwen3-VL-8B-Instruct"'),
 ('output/vlm_hpc_10k_v8_pairbatch50_5day_mec3', 'output/vlm_qwen3_10k_v8_pairbatch50_mec3'),
 ('"$python_bin" vlm_pairwise.py \\', '"$python_bin" hpc_vlm_10k/vlm_qwen3_local.py \\'),
]
for old, new in changes:
    if runner.count(old) != 1:
        raise RuntimeError(f"Expected exactly one source occurrence: {old!r}; found {runner.count(old)}")
    runner = runner.replace(old, new, 1)
marker = 'full_audit="$output_dir/pair_audit.csv"'
if runner.count(marker) != 1:
    raise RuntimeError("Cannot locate pilot/full boundary")
pilot = runner.split(marker)[0] + """
run_vlm "qwen3 serial pilot" 1 1 "$pilot_pairs" "$output_dir/benchmark/pair_batch1_mec3"
run_vlm "qwen3 batch50 pilot" 1 50 "$pilot_pairs" "$output_dir/benchmark/pair_batch50_mec3"
"$python_bin" hpc_vlm_10k/estimate_runtime.py \\
  --batch1 "$output_dir/benchmark/pair_batch1_mec3/timing.json" \\
  --batch-n "$output_dir/benchmark/pair_batch50_mec3/timing.json" \\
  --pair-batch-size 50 --total-pairs 65000 --max-hours 110 \\
  | tee "$output_dir/benchmark/runtime_projection.json"
"""
needle = "bash hpc_vlm_10k/run_b50_5d.sh"
if job.count(needle) != 1:
    raise RuntimeError("Cannot locate Slurm runner")
job = job.replace("#SBATCH --job-name=guvp-vlm10k-v8", "#SBATCH --job-name=qwen3-10k")
job = job.replace("#SBATCH --output=guvp-vlm10k-v8-pair10-%j.out", "#SBATCH --output=qwen3-10k-%j.out")
job = job.replace("#SBATCH --time=3-00:00:00", "#SBATCH --time=5-00:00:00")
full_job = job.replace(needle, "bash hpc_vlm_10k/run_qwen3_b50_5d.sh")
pilot_job = job.replace(needle, "bash hpc_vlm_10k/run_qwen3_b50_pilot.sh").replace("#SBATCH --time=5-00:00:00", "#SBATCH --time=12:00:00").replace("#SBATCH --job-name=qwen3-10k", "#SBATCH --job-name=qwen3-pilot").replace("qwen3-10k-%j.out", "qwen3-pilot-%j.out")
targets = {"run_qwen3_b50_5d.sh":runner, "run_qwen3_b50_pilot.sh":pilot, "submit_qwen3_b50_5d.sbatch":full_job, "submit_qwen3_b50_pilot.sbatch":pilot_job}
for name in targets:
    p=root/"hpc_vlm_10k"/name
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite {p}")
for name,content in targets.items():
    p=root/"hpc_vlm_10k"/name
    p.write_text(content)
    print("Created",p)
py_compile.compile(str(root/"hpc_vlm_10k/vlm_qwen3_local.py"), doraise=True)
