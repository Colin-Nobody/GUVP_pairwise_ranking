# UNNC HPC Q10 setup

## Known information and current limitation

The local HPC user guide lists `deeplight.nottingham.edu.cn` (`10.179.1.20`),
requires the campus network/VPN/VDI, and documents Slurm commands such as
`sinfo`, `squeue`, `scontrol`, and `sbatch`. It does not document Q10's current
GPU model, node memory, partition time limit, or account/QOS limits. An SSH
probe from the workstation timed out, so those current values must be queried
from the university network or an HPC login session.

## Step 1 — connect to the login node

Connect to the campus network or VDI/VPN first, then run:

```bash
ssh sgxcw3@10.179.1.20
```

The username is the UNNC HPC account. If your account name differs from
`sgxcw3`, substitute it here.

## Step 2 — locate and update the existing HPC project

Do not clone another copy. After connecting to the school network, log in and
locate the existing project under your HPC home directory:

```bash
ssh sgxcw3@10.179.1.20
find "/share/home/$USER" -maxdepth 6 -type f -name vlm_pairwise.py -print 2>/dev/null
```

Use the result to set the real project path. For example, if it is
`/share/home/sgxcw3/GUVP_pairwise_ranking`:

```bash
export PROJECT_DIR="/share/home/$USER/GUVP_pairwise_ranking"
test -f "$PROJECT_DIR/vlm_pairwise.py" && cd "$PROJECT_DIR"
```

Before replacing any scripts, make a backup on HPC storage:

```bash
backup="$HOME/guvp_q10_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$backup" -C "$PROJECT_DIR" jobs scripts README.md .gitignore
echo "Backup: $backup"
```

The Q10 edits are in the current workstation copy. From a **second local
terminal** in `/home/colin/GUVP_pairwise_ranking`, copy only the job/config and
related runner files to the existing HPC project (replace the user or path if
the `find` result differs):

```bash
rsync -av --exclude=q10.env jobs/ README.md .gitignore CODEX_CONTEXT.md \
  vlm_pairwise.py \
  scripts/run_qwen3_vl_8b_formal_and_score.sh \
  scripts/run_qwen3_vl_8b_smoke.sh scripts/monitor_8b_formal_abba.sh \
  scripts/monitor_qwen3_vl_8b.sh \
  sgxcw3@10.179.1.20:/share/home/sgxcw3/GUVP_pairwise_ranking/
```

This does not use GitHub and does not transfer image datasets or experiment
outputs. `jobs/q10.env` is excluded so an existing HPC-specific config is
preserved. Do not run VLM inference on the login node.

## Step 3 — query the current Q10 partition and account limits

Run:

```bash
bash jobs/probe_q10.sh
```

It saves and prints a timestamped report under `output/hpc_q10/`. The report
contains Q10 availability, wall-time limit, nodes, CPU count, memory, GRES/GPU
description, partition configuration, user account/QOS associations, and
relevant software modules. If the account association command is denied, keep
that message; the other output is still useful.

For a short GPU-node probe, first confirm from the report that Q10 advertises
the `gpu:1` GRES syntax, then run:

```bash
srun --partition=Q10 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:03:00 \
  bash -lc 'hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total --format=csv'
```

This requests a short allocation on Q10; it is not a production job.

## Step 4 — configure the environment and resources

On the HPC, create the private config file:

```bash
cp jobs/q10.env.example jobs/q10.env
nano jobs/q10.env
```

Set `Q10_GRES`, `Q10_CPUS`, `Q10_MEM`, `Q10_TIME`, and optional
`Q10_ACCOUNT` according to the probe output and the allowed limits. The
example's resource values are provisional values copied from the old Q5 job;
they are not confirmed Q10 limits. Set `HPC_MODULE` to the module shown by the
probe, set `CONDA_ENV_NAME=guvp-vlm` if that environment exists on HPC, or set
`PYTHON_BIN` to its Python executable. Check the environment on a login node:

```bash
module load anaconda
conda activate guvp-vlm
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

If the HPC uses another module or environment manager, update the config to
match it. The job helper reports Python, PyTorch, Transformers, and CUDA
availability before launching inference.

## Step 5 — validate jobs without submitting

From the repository root:

```bash
jobs/submit_q10.sh --check jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch
jobs/submit_q10.sh --check jobs/run_qwen3_vl_8b_formal_3000.sbatch
```

`--check` calls `sbatch --test-only`: it checks whether Slurm accepts the
partition/resource request but does not enqueue the job. If it rejects a GRES,
account, memory, CPU, or time request, adjust `jobs/q10.env` using the probe
output and repeat the check.

## Step 6 — submit one job to Q10

After validation, submit the desired workflow:

```bash
jobs/submit_q10.sh --submit jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch
# or
jobs/submit_q10.sh --submit jobs/run_qwen3_vl_8b_formal_3000.sbatch
```

The submit helper runs `sbatch --test-only` again before actual submission. It
passes the shared Q10 resources as command-line options, while the job files
also declare Q10 starter defaults for clarity.

## Step 7 — monitor

```bash
squeue -u "$USER"
sinfo -p Q10
tail -f guvp-v8-1000-JOBID.out
# or
tail -f guvp-qwen3-8b-3000-JOBID.out
```

The Qwen3-VL formal workflow can also be monitored with:

```bash
bash scripts/monitor_8b_formal_abba.sh 30
```

The VLM runners write resumable progress and audit files under their respective
`output/` directories. Re-submit only after checking the run logs and audit
state; the runner lock prevents duplicate runs of the same pipeline.
