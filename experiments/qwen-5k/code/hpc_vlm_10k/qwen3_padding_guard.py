#!/usr/bin/env python3
"""Apply Qwen3 left padding and gate full inference on the pilot."""
import csv
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if len(sys.argv) == 3 and sys.argv[1] == "--check":
    bench = Path(sys.argv[2])
    def read(folder):
        with (bench / folder / "pair_audit.csv").open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    baseline, batch = read("pair_batch1_mec3"), read("pair_batch50_mec3")
    def pairs(rows):
        return [(r["image_a"], r["image_b"]) for r in rows]
    if len(baseline) != 100 or len(batch) != 100 or pairs(baseline) != pairs(batch):
        raise SystemExit("PILOT GATE FAILED: expected identical 100 pairs in each audit")
    valid1 = sum(r["status"] == "calibrated" and bool(r["winner"]) for r in baseline)
    valid50 = sum(r["status"] == "calibrated" and bool(r["winner"]) for r in batch)
    print(f"Pilot gate: batch1={valid1}/100, batch50={valid50}/100; minimum batch50=80% of baseline", flush=True)
    if valid1 == 0 or valid50 * 5 < valid1 * 4:
        raise SystemExit("PILOT GATE FAILED: batched valid votes are below 80% of baseline")
    raise SystemExit(0)

wrapper = root / "hpc_vlm_10k/vlm_qwen3_local.py"
runner = root / "hpc_vlm_10k/run_qwen3_b50_5d.sh"
backup_wrapper = wrapper.with_name("vlm_qwen3_local.pre_leftpad_388562.py")
backup_runner = runner.with_name("run_qwen3_b50_5d.pre_gate_388562.sh")
output = root / "output/vlm_qwen3_10k_v8_pairbatch50_mec3"
archive = output.with_name(output.name + "_pre_leftpad_388562")
if backup_wrapper.exists() or backup_runner.exists() or archive.exists():
    raise SystemExit("Backup exists; inspect manually before repeating")
w = wrapper.read_text()
r = runner.read_text()
needle = '    print(f"Loaded pre-quantized model locally from {model_dir}; GPU placement {device_map}", flush=True)'
if needle not in w or 'processor.tokenizer.padding_side = "left"' in w:
    raise SystemExit("Unexpected Qwen3 loader; no changes made")
gate_point = '  "$python_bin" hpc_vlm_10k/estimate_runtime.py'
if gate_point not in r or "qwen3_padding_guard.py" in r:
    raise SystemExit("Unexpected runner; no changes made")
new_w = w.replace(needle, '    processor.tokenizer.padding_side = "left"\n' + needle, 1)
gate = '  "$python_bin" hpc_vlm_10k/qwen3_padding_guard.py --check "$output_dir/benchmark"\n'
new_r = r.replace(gate_point, gate + gate_point, 1)
compile(new_w, str(wrapper), "exec")
shutil.copy2(wrapper, backup_wrapper)
shutil.copy2(runner, backup_runner)
wrapper.write_text(new_w)
runner.write_text(new_r)
if output.exists():
    output.rename(archive)
    print("Previous output preserved:", archive)
print("Installed left padding and pilot quality gate")
