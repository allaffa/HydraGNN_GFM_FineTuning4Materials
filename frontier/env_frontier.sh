#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared Frontier (OLCF) environment setup for the fine-tuning / benchmark
# pipeline.  Source this from the SLURM batch scripts (or on a login node
# before an interactive `salloc`).
#
#   source frontier/env_frontier.sh
#
# All site-specific paths are overridable via environment variables so you do
# not need to edit this file:
#   HYDRAGNN_VENV        conda/venv prefix to activate (REQUIRED)
#   MODULE_LOAD_SCRIPT   optional site module-load script to `source`
#   FRONTIER_ROCM        ROCm module version (default 7.1.1)
#   REPO_ROOT            repo checkout (default: $SLURM_SUBMIT_DIR or $PWD)
# ---------------------------------------------------------------------------
# NOTE: do NOT `set -e` here - module/conda commands often return nonzero
# harmlessly.  Strict mode is enabled by the caller after this script returns.

# --- Repo root -------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export REPO_ROOT
if [[ ! -d "${REPO_ROOT}/examples" ]]; then
    echo "ERROR: REPO_ROOT='${REPO_ROOT}' does not look like the repo root" >&2
    echo "       Submit from the repo root or export REPO_ROOT explicitly."   >&2
    return 1 2>/dev/null || exit 1
fi

# --- OLCF outbound proxy (needed for any on-node HTTP; harmless otherwise) --
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# --- Modules ---------------------------------------------------------------
FRONTIER_ROCM="${FRONTIER_ROCM:-7.1.1}"
if [[ -n "${MODULE_LOAD_SCRIPT:-}" && -f "${MODULE_LOAD_SCRIPT}" ]]; then
    # Site-provided module loader (e.g. a world-shared module-to-load script).
    source "${MODULE_LOAD_SCRIPT}"
else
    # Fall back to the loader shipped with HydraGNN.
    source "${REPO_ROOT}/HydraGNN/installation_DOE_supercomputers/module_loads_frontier.sh"
    load_frontier_modules "${FRONTIER_ROCM}" "${FRONTIER_ROCM}"
fi

# --- Python environment ----------------------------------------------------
if [[ -z "${HYDRAGNN_VENV:-}" ]]; then
    echo "ERROR: set HYDRAGNN_VENV to your conda/venv prefix before sourcing" >&2
    return 1 2>/dev/null || exit 1
fi
# `source activate` works for both conda envs and venvs created by miniforge.
source activate "${HYDRAGNN_VENV}"

# Make the repo (and HydraGNN) importable.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/HydraGNN:${PYTHONPATH:-}"

# --- Runtime / ROCm env ----------------------------------------------------
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export HYDRAGNN_NUM_WORKERS=0
# Model caches (MACE + HuggingFace/fairchem) live under $HOME by default;
# override to a project space if $HOME quota is tight.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MACE_CACHE="${MACE_CACHE:-$HOME/.cache/mace}"

echo "== Frontier environment =="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda(HIP):', torch.cuda.is_available())"
echo "REPO_ROOT=${REPO_ROOT}"
echo "=========================="
