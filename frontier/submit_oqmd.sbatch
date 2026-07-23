#!/bin/bash
# ---------------------------------------------------------------------------
# Frontier batch job: OQMD (formation-energy) fine-tune comparison pipeline.
#   HydraGNN fine-tune -> MACE fine-tune -> UMA fine-tune.
#
# OQMD targets formation energy per atom, so this is a *fine-tune only*
# comparison (no meaningful zero-shot: foundation models are not referenced
# to DFT formation enthalpies).
#
# Submit from the repo root:
#   export HYDRAGNN_VENV=/lustre/orion/<proj>/.../hydragnn_venv
#   sbatch frontier/submit_oqmd.sbatch
#
# Prefetch model weights on a login node FIRST:  bash frontier/prefetch_models.sh
# The dataset (dataset/oqmd.pickle) must be built first via
#   python examples/oqmd/oqmd_getData.py
# (rebuild if it predates the data.cell patch).
# ---------------------------------------------------------------------------
#SBATCH -A LRN070
#SBATCH -J oqmd_gfm
#SBATCH -o logs/frontier/oqmd-%j.out
#SBATCH -e logs/frontier/oqmd-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${REPO_ROOT}/logs/frontier"

# --- Environment selection (ROCm 7.2 base venv + dedicated MACE venv) -------
export FRONTIER_ROCM="${FRONTIER_ROCM:-7.2.0}"
export HYDRAGNN_VENV="${HYDRAGNN_VENV:-${REPO_ROOT}/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
export MACE_VENV="${MACE_VENV:-${REPO_ROOT}/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72}"

source "${REPO_ROOT}/frontier/env_frontier.sh"

set -euo pipefail

SRUN="srun -N1 -n1 -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest"

# Compute nodes have no outbound internet: force offline model loading.
export HF_HUB_OFFLINE=1
# MACE foundation .model files must deserialize with weights_only=False.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# MACE fine-tune runs in the dedicated e3nn-0.4.4 venv.
MACE_PY="${MACE_VENV}/bin/python"

echo "### [1/3] HydraGNN fine-tune"
cd "${REPO_ROOT}"
$SRUN python -u examples/oqmd/ensemble_fine_tune.py

echo "### [2/3] MACE fine-tune"
cd "${REPO_ROOT}/examples/oqmd"
$SRUN "${MACE_PY}" -u run_mace_finetune.py --device cuda

echo "### [3/3] UMA fine-tune"
$SRUN python -u run_uma_finetune.py --device cuda --freeze-backbone

echo "### OQMD pipeline complete. Results in examples/oqmd/benchmark_results/"
