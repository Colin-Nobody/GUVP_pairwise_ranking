
## About me

**Chongliang Wang** — PhD researcher in Electrical and Electronic Engineering (EEE) at the University of Nottingham Ningbo China.

## About this project

GUVP is my ongoing research project on visual pollution in urban street-view imagery. This repository turns pairwise image preferences into image-level rankings using Q-score and TrueSkill, and explores vision-language models as scalable evaluators. The VLM workflow uses evidence-based scoring and calibration across swapped image positions to help reduce A/B position bias.

For the project workflow and experiment archive, see [`DGX_SPARK_MIGRATION.md`](DGX_SPARK_MIGRATION.md).

### Qwen3-VL 8B formal pairwise run on school HPC Q10

The Qwen3-VL 8B formal runner follows the Qwen2.5-VL 7B optimized MEC+BPC
pipeline: the same v6 prompt, fixed 3,000-pair manifest, three samples per
image position, serial decoding with a reused visual/prompt KV prefix, formal
audit summary, and Q-score/TrueSkill aggregation. Project model inference and
training are GPU-only: CPU offload is rejected, and the model must fit in GPU
memory. Configure Q10 as described in `jobs/Q10_SETUP.md`, then
validate and submit it from the repository root:

```bash
jobs/submit_q10.sh --check jobs/run_qwen3_vl_8b_formal_3000.sbatch
jobs/submit_q10.sh --submit jobs/run_qwen3_vl_8b_formal_3000.sbatch
```

To watch progress in another terminal:

```bash
bash scripts/monitor_8b_formal_abba.sh
```

The Q10 probe and workstation-to-HPC sync steps are documented in
[`jobs/Q10_SETUP.md`](jobs/Q10_SETUP.md). The current Q10 limits have not yet
been queried from the cluster; starter resource requests below are provisional.

This pipeline writes to `data/vlm_pairwise_qwen3_vl_8b_formal_mec3_bpc_v6_3000_prefix_cache.csv`
and `output/vlm_qwen3_vl_8b_formal_mec3_bpc_v6_3000_prefix_cache/`. The prior
non-prefix-cache Qwen3 run files are left intact; the new run starts in its own
output namespace. CPU is still used for image decoding, preprocessing, file I/O,
and non-model statistics; neural model forward/generation runs on CUDA only.

### External data and checkpoint archive

The Mapillary Vistas source dataset and the archived segmentation checkpoints
remain on the external drive `白` because of their size. The archive inventory
and SHA-256 checksums are tracked under
[`archive_manifests/GUVP_pairwise_ranking_2026-10-05/`](archive_manifests/GUVP_pairwise_ranking_2026-10-05/).
The inventory describes the October 5, 2026 archive; it does not include later
experiment outputs.

### Qwen2.5-VL 7B V8 run on 1,000 images on school HPC Q10

The Slurm job `jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch` runs the V8 prompt
on the tracked 1,000-image set and 6,500-pair manifest, then writes the audit,
ranking, and timing outputs under `output/vlm_expansion_1000_definition_v8/`.
The job stages portable image links because the checked-in 1,000-image directory
contains links created on the original workstation. The staging logic is embedded
in the job file, so there is no separate helper script to upload.

From the repository root on the HPC, run the Q10 probe and configure
`jobs/q10.env` first, then validate and submit:

```bash
jobs/submit_q10.sh --check jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch
jobs/submit_q10.sh --submit jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch
```

Both jobs target Q10. Starter requests are one GPU, 8 CPUs, 32 GB host memory,
and 24 hours, copied from the prior Q5 job as provisional values. They are not
verified Q10 limits: use the Q10 probe and `sbatch --test-only` validation in
`jobs/Q10_SETUP.md` before submission.

---
