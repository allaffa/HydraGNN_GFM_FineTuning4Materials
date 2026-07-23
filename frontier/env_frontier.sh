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
# NOTE: proxy.ccs.ornl.gov:3128 is an HTTP (squid) proxy.  all_proxy MUST use
# the http:// scheme — httpx (used by huggingface_hub / fairchem) rejects a
# bare "socks://" scheme with "Unknown scheme for proxy URL", which breaks UMA
# checkpoint loading even when the weights are already cached locally.
export all_proxy=http://proxy.ccs.ornl.gov:3128/
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
# Activate the env.  Prefer sourcing conda.sh then `conda activate` (works in
# non-interactive / batch shells where the bare `source activate` shim is not
# on PATH); fall back to `source activate` for plain venvs.
if [[ -n "${CONDA_EXE:-}" ]]; then
    _conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
    if [[ -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
        source "${_conda_base}/etc/profile.d/conda.sh"
    fi
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "${HYDRAGNN_VENV}"
else
    source activate "${HYDRAGNN_VENV}"
fi

# Make the repo (and HydraGNN) importable.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/HydraGNN:${PYTHONPATH:-}"

# --- Runtime / ROCm env ----------------------------------------------------
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export HYDRAGNN_NUM_WORKERS=0
# Model caches live inside the repo (world-shared, visible to compute nodes),
# at the same level as pretrained_model_ensemble/, instead of $HOME/.cache.
#   UMA / fairchem download via HuggingFace  -> honours HF_HOME  (uma_cache/)
#   fairchem checkpoint store (uma-s-1p2.pt) -> honours FAIRCHEM_CACHE_DIR
#   MACE (mace-torch get_cache_dir)          -> honours XDG_CACHE_HOME -> <dir>/mace
export HF_HOME="${HF_HOME:-$REPO_ROOT/uma_cache}"
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$REPO_ROOT/uma_cache/fairchem}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/mace_cache}"
export MACE_CACHE="$XDG_CACHE_HOME/mace"

echo "== Frontier environment =="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda(HIP):', torch.cuda.is_available())"
echo "REPO_ROOT=${REPO_ROOT}"
echo "=========================="
