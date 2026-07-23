#!/bin/bash
# ---------------------------------------------------------------------------
# Frontier batch job: MS25 (MLIP) comparison pipeline.
#   HydraGNN fine-tune (per system) -> MACE fine-tune -> UMA fine-tune ->
#   zero-shot MACE/UMA benchmarks (accuracy + wall-clock, every system).
#
# The MACE/UMA scripts iterate over all MS25 systems internally, so each is
# launched once.  HydraGNN fine-tuning loops systems inside ensemble_fine_tune.py.
#
# Submit from the repo root:
#   export HYDRAGNN_VENV=/lustre/orion/<proj>/.../hydragnn_venv
#   sbatch frontier/submit_ms25.sbatch
#
# Prefetch model weights on a login node FIRST:  bash frontier/prefetch_models.sh
# ---------------------------------------------------------------------------
#SBATCH -A LRN070
#SBATCH -J ms25_gfm
#SBATCH -o logs/frontier/ms25-%j.out
#SBATCH -e logs/frontier/ms25-%j.out
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
# MACE fine-tune/benchmark run in the dedicated e3nn-0.4.4 venv.
MACE_PY="${MACE_VENV}/bin/python"

echo "### [1/4] HydraGNN fine-tune (all MS25 systems)"
cd "${REPO_ROOT}/examples/ms25"
$SRUN python -u ensemble_fine_tune.py

echo "### [2/4] MACE fine-tune (all systems)"
$SRUN "${MACE_PY}" -u run_mace_finetune.py --device cuda

echo "### [3/4] UMA fine-tune (all systems)"
$SRUN python -u run_uma_finetune.py --device cuda --freeze-backbone

echo "### [4/4] Zero-shot MACE + UMA benchmarks"
$SRUN python -u run_uma_benchmark.py --device cuda
$SRUN "${MACE_PY}" -u run_mace_benchmark.py --device cuda

echo "### MS25 pipeline complete. Results in examples/ms25/benchmark_results/"
