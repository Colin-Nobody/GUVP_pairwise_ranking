#!/usr/bin/env bash

# Shared environment setup for the Q10 Slurm jobs and their runner scripts.
q10_load_config() {
  local project_dir="${1:?usage: q10_load_config PROJECT_DIR}"
  local config_file="${Q10_CONFIG:-$project_dir/jobs/q10.env}"
  if [[ -f "$config_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$config_file"
    set +a
  fi
}

q10_resolve_python() {
  if [[ -n "${HPC_MODULE:-}" && "${HPC_MODULE}" != "none" ]]; then
    if ! type module >/dev/null 2>&1; then
      for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$init" ]]; then
          # shellcheck disable=SC1090
          source "$init"
          break
        fi
      done
    fi
    if ! type module >/dev/null 2>&1; then
      echo "HPC_MODULE is set to '$HPC_MODULE' but the module command is unavailable." >&2
      return 1
    fi
    module load "$HPC_MODULE"
  fi

  if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -r "$base/etc/profile.d/conda.sh" ]]; then
          # shellcheck disable=SC1090
          source "$base/etc/profile.d/conda.sh"
          break
        fi
      done
    fi
    if ! command -v conda >/dev/null 2>&1; then
      echo "Cannot find conda. Set PYTHON_BIN to the guvp-vlm Python executable or load the correct module." >&2
      return 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
  fi

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
      echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
      return 1
    fi
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
    if [[ -z "$PYTHON_BIN" ]]; then
      echo "Python was not found. Configure PYTHON_BIN or CONDA_ENV_NAME in jobs/q10.env." >&2
      return 1
    fi
  fi
  export PYTHON_BIN
  "$PYTHON_BIN" -c 'import torch, transformers; print("Python", __import__("sys").version.split()[0], "PyTorch", torch.__version__, "Transformers", transformers.__version__, "CUDA", torch.cuda.is_available())'
}
