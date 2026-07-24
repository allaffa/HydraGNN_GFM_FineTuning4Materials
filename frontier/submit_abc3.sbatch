#!/bin/bash
# ---------------------------------------------------------------------------
# Frontier batch job: ABC3 (formation-energy) fine-tune comparison pipeline.
#   HydraGNN fine-tune -> MACE fine-tune -> UMA fine-tune.
#
# ABC3 targets formation energy per atom, so this is a *fine-tune only*
# comparison (no meaningful zero-shot: foundation models are not referenced
# to DFT formation enthalpies).
#
# Submit from the repo root:
#   export HYDRAGNN_VENV=/lustre/orion/<proj>/.../hydragnn_venv
#   sbatch frontier/submit_abc3.sbatch
#
# Prefetch model weights on a login node FIRST:  bash frontier/prefetch_models.sh
# The dataset (dataset/abc3.pickle) must be built first via
#   python examples/abc3/abc3_getData_API.py
# (rebuild if it predates the data.cell patch).
# ---------------------------------------------------------------------------
#SBATCH -A LRN070
#SBATCH -J abc3_gfm
#SBATCH -o logs/frontier/abc3-%j.out
#SBATCH -e logs/frontier/abc3-%j.out
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
$SRUN python -u examples/abc3/ensemble_fine_tune.py

echo "### [2/3] MACE fine-tune"
cd "${REPO_ROOT}/examples/abc3"
$SRUN "${MACE_PY}" -u run_mace_finetune.py --device cuda

echo "### [3/3] UMA fine-tune  (full-FT 2 epochs + frozen + LoRA, concurrent on 3 GCDs)"
# abc3 has 3185 train samples; energy-only but ~45 min/epoch -> 2 epochs fits within 2h.
rm -f benchmark_results/uma_finetuned_full_summary.json \
      benchmark_results/uma_finetuned_frozen_summary.json \
      benchmark_results/uma_finetuned_lora_summary.json
UMA_SRUN="srun -N1 -n1 -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --overlap"
$UMA_SRUN python -u run_uma_finetune.py --device cuda --tag full --epochs 2 &
uma_full_pid=$!
$UMA_SRUN python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone &
uma_frozen_pid=$!
$UMA_SRUN python -u run_uma_finetune.py --device cuda --tag lora --lora &
uma_lora_pid=$!
uma_rc=0
wait "${uma_full_pid}"   || uma_rc=$?
wait "${uma_frozen_pid}" || uma_rc=$?
wait "${uma_lora_pid}"   || uma_rc=$?
if [ "${uma_rc}" -ne 0 ]; then
    echo "UMA fine-tune step failed (rc=${uma_rc})" >&2
    exit "${uma_rc}"
fi

echo "### ABC3 pipeline complete. Results in examples/abc3/benchmark_results/"
