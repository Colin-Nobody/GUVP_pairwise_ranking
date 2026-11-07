# 10k V8 deployment handoff for Ubuntu Codex CLI

Status recorded 2026-11-07. The Slurm job 384732 was submitted in Q10 and was pending at the last check. It trains deployment weights on HPC; do not assume it completed or that its outputs have reached GitHub.

## What the model is

DINOv3 ViT-B/16 initialized from the original pretrained checkpoint, with transformer blocks 10 and 11 (zero-based) and an MLP regression head fitted on 10,000 V8 VLM-derived scores. It is not a LoRA or text-conditioned attention model. The intended output is an ensemble of seeds 42, 137 and 2026, averaged at inference. Cross-validation estimates are not final deployment checkpoints.

## HPC completion and upload

From the project root on HPC:
```bash
squeue -j 384732
tail -f slurm-384732.out
```
If the job has left the queue, check `sacct -j 384732 --format=JobID,State,ExitCode,Elapsed` and the log. A successful fit must produce `output/vit_vp_10k_v8/deployment/finetuned_last2_v8/model.json`, `fit_config.json`, and all three `seed_*.safetensors`. Verify their SHA-256 values against model.json and run a prediction smoke test on a GPU before describing them as ready.

The 10k scores and CV results exist on HPC; an isolated worktree at `/share/home/sgxcw3/guvp-10k-upload` contains commit `5843efc` with those results. Its push previously failed because GitHub rejected authentication. The current branch `hpc-10k-deploy` contains the training and Slurm scripts but, as of this status, does not contain the 10k final weights or HPC results. Never put a token into a file, repository, shell command line, or chat.

After successful training and verification, upload the deployment files using Git LFS and a properly authenticated Git client; separately push or integrate the isolated results commit without overwriting unrelated changes. Check the repository's LFS quota and verify the remote LFS objects, not merely their pointer files. Preserve the exact model.json and pretrained checkpoint paired with the trained artifacts.

## Ubuntu RTX 5060: download and prepare

First wait until the 10k artifacts are confirmed uploaded to the branch/release identified by the person completing the upload.

```bash
git clone --branch hpc-10k-deploy https://github.com/Colin-Nobody/GUVP_pairwise_ranking.git
cd GUVP_pairwise_ranking
git lfs install
git lfs pull
nvidia-smi
```

A fresh clone requires Git LFS installed first to fetch large files automatically. If LFS is not installed, install it from the system's trusted package source, run `git lfs install` and `git lfs pull`. The original checkpoint is `models/checkpoints/dinov3_vitb.safetensors`. Confirm its SHA-256 is `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`, and confirm every seed file is an actual downloaded LFS object and matches model.json.

Set up a dedicated Python environment with a CUDA-enabled PyTorch build that supports the installed RTX 5060 and driver, plus the repository's required packages including transformers, safetensors, pillow, numpy and pandas. The HPC run used Python 3.11, PyTorch 2.10 with CUDA 12.8, and Transformers 5.20.1; verify local GPU support and version compatibility rather than assuming the HPC environment is portable. Read project documentation and `AGENTS.md`. Do not fall back silently to CPU.

After verifying the files and CUDA, run from the repository root:
```bash
python vit_vp/predict_finetuned.py /path/to/image_or_directory \
  --model-dir output/vit_vp_10k_v8/deployment/finetuned_last2_v8 \
  --checkpoint models/checkpoints/dinov3_vitb.safetensors \
  --output output/local_10k_predictions.csv
```
The existing predictor checks SHA-256 hashes and averages the three seeds. It loads all three encoders at once; peak GPU memory and throughput on this particular RTX 5060 have not been measured. If it runs out of memory, implement and test sequential seed inference with the same preprocessing and arithmetic averaging, then compare against the original on a GPU where both fit. Avoid changing resolution or scores merely to fit memory.

## Prompt for local Codex CLI

Inspect AGENTS.md and this handoff. Determine whether the 10k model.json and three matching LFS seed weights actually exist and verify hashes. Inspect nvidia-smi, GPU memory, driver and CUDA/PyTorch support. Create an isolated environment, run a one-image GPU smoke test and a small folder test, report outputs, elapsed time and peak VRAM. If necessary implement a sequential ensemble without changing the model's preprocessing or prediction semantics and validate numerical agreement. Do not mistake the 1200-image deployment for the 10k model. If 10k files are absent, stop and report exactly which are missing; do not substitute other weights.
