#!/bin/bash
# ---------------------------------------------------------------------------
# Frontier batch job: run ALL 5 dataset pipelines in parallel, one node each.
#
# debug QOS has no MaxNodes limit (only 2h wall + 1 submitted job at a time),
# so requesting 5 nodes + --qos=debug is valid.  Total wall time = max over
# all datasets ≈ 2h (each individual pipeline is already timed to fit 2h).
#
# Node assignment:
#   Node 0: QM9        Node 1: MD17       Node 2: ABC3
#   Node 3: Wiggle150  Node 4: MS25
# (OQMD omitted until dataset download completes.)
#
# Each dataset's srun calls are pinned to their assigned node via -w $NODE.
# --overlap is required on every srun so all 5 pipelines run concurrently
# rather than queuing behind each other in the same allocation.
#
# Per-dataset logs land in logs/frontier/all-<jobid>-<dataset>.log.
# The main log (all-<jobid>.out) shows launch/finish lines only.
#
# Submit from the repo root:
#   export REPO_ROOT=$PWD FRONTIER_ROCM=7.2.0 \
#     HYDRAGNN_VENV=$PWD/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72 \
#     MACE_VENV=$PWD/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72
#   sbatch --qos=debug frontier/submit_all.sbatch
#
# Prefetch model weights on a login node FIRST: bash frontier/prefetch_models.sh
# ---------------------------------------------------------------------------
#SBATCH -A LRN070
#SBATCH -J all_gfm
#SBATCH -o logs/frontier/all-%j.out
#SBATCH -e logs/frontier/all-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 5

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${REPO_ROOT}/logs/frontier"

# --- Shared environment -------------------------------------------------------
export FRONTIER_ROCM="${FRONTIER_ROCM:-7.2.0}"
export HYDRAGNN_VENV="${HYDRAGNN_VENV:-${REPO_ROOT}/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
export MACE_VENV="${MACE_VENV:-${REPO_ROOT}/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72}"
source "${REPO_ROOT}/frontier/env_frontier.sh"

export HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MACE_PY="${MACE_VENV}/bin/python"

# --- Assign one node per dataset ----------------------------------------------
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
QM9_NODE="${NODES[0]}"
MD17_NODE="${NODES[1]}"
ABC3_NODE="${NODES[2]}"
WIG_NODE="${NODES[3]}"
MS25_NODE="${NODES[4]}"

echo "Node assignments:"
echo "  QM9=${QM9_NODE}  MD17=${MD17_NODE}  ABC3=${ABC3_NODE}  WIG=${WIG_NODE}  MS25=${MS25_NODE}"

# --- Helper: srun pinned to one node, overlapping with all other pipelines ----
# All srun calls use --overlap so the 5 dataset pipelines (and the concurrent
# UMA steps within each) can all run simultaneously.
srun_on() {
    local node=$1; shift
    srun -N1 -n1 -c"${OMP_NUM_THREADS}" --gpus-per-task=1 --gpu-bind=closest \
         --overlap -w "$node" "$@"
}

# ===========================================================================
# QM9 pipeline (energy-only; full-FT result already saved, run frozen + LoRA)
# ===========================================================================
pipeline_qm9() {
    local NODE=$1
    local RES="${REPO_ROOT}/examples/qm9/benchmark_results"
    set -euo pipefail
    cd "${REPO_ROOT}/examples/qm9"

    if [ ! -f "${RES}/benchmark_summary.json" ]; then
        echo "### QM9 [1/5] HydraGNN benchmark"
        srun_on "$NODE" python -u run_benchmark.py
    else
        echo "### QM9 [1/5] HydraGNN benchmark — SKIPPED (results exist)"
    fi

    if [ ! -f "${RES}/mace_finetuned_summary.json" ]; then
        echo "### QM9 [2/5] MACE fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --lr 1e-4
    else
        echo "### QM9 [2/5] MACE fine-tune — SKIPPED (results exist)"
    fi

    echo "### QM9 [3/5] UMA fine-tune (frozen + LoRA concurrent)"
    rm -f "${RES}/uma_finetuned_frozen_summary.json" \
          "${RES}/uma_finetuned_lora_summary.json"
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone &
    local p1=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora &
    local p2=$!
    local rc=0
    wait "$p1" || rc=$?
    wait "$p2" || rc=$?
    [ "$rc" -eq 0 ] || { echo "QM9 UMA step failed (rc=$rc)" >&2; exit "$rc"; }

    echo "### QM9 [4/5] UMA zero-shot"
    srun_on "$NODE" python -u run_uma_benchmark.py --device cuda \
        --hydragnn-summary "${RES}/benchmark_summary.json"

    echo "### QM9 [5/5] MACE zero-shot"
    srun_on "$NODE" "${MACE_PY}" -u run_mace_benchmark.py --device cuda \
        --hydragnn-summary "${RES}/benchmark_summary.json" \
        --uma-summary "${RES}/uma_benchmark_summary.json"
    echo "### QM9 pipeline complete."
}

# ===========================================================================
# MD17 pipeline (MLIP: energy + forces; full-FT 4 epochs + frozen + LoRA)
# ===========================================================================
pipeline_md17() {
    local NODE=$1
    local RES="${REPO_ROOT}/examples/md17/benchmark_results"
    set -euo pipefail
    cd "${REPO_ROOT}/examples/md17"

    if [ ! -f "${RES}/benchmark_summary.json" ]; then
        echo "### MD17 [1/5] HydraGNN benchmark"
        srun_on "$NODE" python -u run_benchmark.py
    else
        echo "### MD17 [1/5] HydraGNN benchmark — SKIPPED (results exist)"
    fi

    if [ ! -f "${RES}/mace_finetuned_summary.json" ]; then
        echo "### MD17 [2/5] MACE fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --lr 1e-4
    else
        echo "### MD17 [2/5] MACE fine-tune — SKIPPED (results exist)"
    fi

    echo "### MD17 [3/5] UMA fine-tune (full 4ep + frozen + LoRA concurrent)"
    rm -f "${RES}/uma_finetuned_full_summary.json" \
          "${RES}/uma_finetuned_frozen_summary.json" \
          "${RES}/uma_finetuned_lora_summary.json"
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag full --epochs 4 &
    local p1=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone &
    local p2=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora &
    local p3=$!
    local rc=0
    wait "$p1" || rc=$?
    wait "$p2" || rc=$?
    wait "$p3" || rc=$?
    [ "$rc" -eq 0 ] || { echo "MD17 UMA step failed (rc=$rc)" >&2; exit "$rc"; }

    echo "### MD17 [4/5] UMA zero-shot"
    srun_on "$NODE" python -u run_uma_benchmark.py --device cuda \
        --hydragnn-summary "${RES}/benchmark_summary.json"

    echo "### MD17 [5/5] MACE zero-shot"
    srun_on "$NODE" "${MACE_PY}" -u run_mace_benchmark.py --device cuda \
        --hydragnn-summary "${RES}/benchmark_summary.json" \
        --uma-summary "${RES}/uma_benchmark_summary.json"
    echo "### MD17 pipeline complete."
}

# ===========================================================================
# ABC3 pipeline (formation energy fine-tune only; no zero-shot step)
# ===========================================================================
pipeline_abc3() {
    local NODE=$1
    local RES="${REPO_ROOT}/examples/abc3/benchmark_results"
    set -euo pipefail

    echo "### ABC3 [1/3] HydraGNN fine-tune"
    cd "${REPO_ROOT}"
    srun_on "$NODE" python -u examples/abc3/ensemble_fine_tune.py

    echo "### ABC3 [2/3] MACE fine-tune"
    cd "${REPO_ROOT}/examples/abc3"
    srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda

    echo "### ABC3 [3/3] UMA fine-tune (full 2ep + frozen + LoRA concurrent)"
    rm -f "${RES}/uma_finetuned_full_summary.json" \
          "${RES}/uma_finetuned_frozen_summary.json" \
          "${RES}/uma_finetuned_lora_summary.json"
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag full --epochs 2 &
    local p1=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone &
    local p2=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora &
    local p3=$!
    local rc=0
    wait "$p1" || rc=$?
    wait "$p2" || rc=$?
    wait "$p3" || rc=$?
    [ "$rc" -eq 0 ] || { echo "ABC3 UMA step failed (rc=$rc)" >&2; exit "$rc"; }
    echo "### ABC3 pipeline complete."
}

# ===========================================================================
# Wiggle150 pipeline (conformational energy; full + frozen + LoRA)
# ===========================================================================
pipeline_wiggle150() {
    local NODE=$1
    local RES="${REPO_ROOT}/examples/wiggle150/benchmark_results"
    set -euo pipefail
    cd "${REPO_ROOT}/examples/wiggle150"

    if [ ! -f "${RES}/benchmark_summary.json" ]; then
        echo "### Wiggle150 [1/5] HydraGNN benchmark"
        srun_on "$NODE" python -u run_benchmark.py
    else
        echo "### Wiggle150 [1/5] HydraGNN benchmark — SKIPPED (results exist)"
    fi

    if [ ! -f "${RES}/mace_finetuned_summary.json" ]; then
        echo "### Wiggle150 [2/5] MACE fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --lr 1e-4
    else
        echo "### Wiggle150 [2/5] MACE fine-tune — SKIPPED (results exist)"
    fi

    echo "### Wiggle150 [3/5] UMA fine-tune (full + frozen + LoRA concurrent)"
    rm -f "${RES}/uma_finetuned_full_summary.json" \
          "${RES}/uma_finetuned_frozen_summary.json" \
          "${RES}/uma_finetuned_lora_summary.json"
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag full &
    local p1=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone &
    local p2=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora &
    local p3=$!
    local rc=0
    wait "$p1" || rc=$?
    wait "$p2" || rc=$?
    wait "$p3" || rc=$?
    [ "$rc" -eq 0 ] || { echo "Wiggle150 UMA step failed (rc=$rc)" >&2; exit "$rc"; }

    echo "### Wiggle150 [4/5] UMA zero-shot"
    srun_on "$NODE" python -u run_uma_benchmark.py --device cuda \
        --hydragnn-summary "${RES}/benchmark_summary.json"

    echo "### Wiggle150 [5/5] MACE zero-shot"
    srun_on "$NODE" "${MACE_PY}" -u run_mace_benchmark.py --device cuda \
        --hydragnn-summary "${RES}/benchmark_summary.json" \
        --uma-summary "${RES}/uma_benchmark_summary.json"
    echo "### Wiggle150 pipeline complete."
}

# ===========================================================================
# MS25 pipeline (MLIP multi-system; frozen + LoRA only — full-FT too slow)
# ===========================================================================
pipeline_ms25() {
    local NODE=$1
    local RES="${REPO_ROOT}/examples/ms25/benchmark_results"
    set -euo pipefail
    cd "${REPO_ROOT}/examples/ms25"

    echo "### MS25 [1/4] HydraGNN fine-tune (all systems)"
    srun_on "$NODE" python -u ensemble_fine_tune.py

    echo "### MS25 [2/4] MACE fine-tune"
    srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --lr 1e-4

    echo "### MS25 [3/4] UMA fine-tune (frozen + LoRA concurrent)"
    rm -f "${RES}/uma_finetuned_frozen_summary.json" \
          "${RES}/uma_finetuned_lora_summary.json"
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone &
    local p1=$!
    srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora &
    local p2=$!
    local rc=0
    wait "$p1" || rc=$?
    wait "$p2" || rc=$?
    [ "$rc" -eq 0 ] || { echo "MS25 UMA step failed (rc=$rc)" >&2; exit "$rc"; }

    echo "### MS25 [4/4] Zero-shot benchmarks"
    srun_on "$NODE" python -u run_uma_benchmark.py --device cuda
    srun_on "$NODE" "${MACE_PY}" -u run_mace_benchmark.py --device cuda
    echo "### MS25 pipeline complete."
}

# ===========================================================================
# Launch all 5 pipelines in parallel — each writes its own log file
# ===========================================================================
LOG="${REPO_ROOT}/logs/frontier/all-${SLURM_JOB_ID}"
echo "### Launching 5 dataset pipelines in parallel at $(date)"
pipeline_qm9       "$QM9_NODE"  > "${LOG}-qm9.log"      2>&1 &
pipeline_md17      "$MD17_NODE" > "${LOG}-md17.log"     2>&1 &
pipeline_abc3      "$ABC3_NODE" > "${LOG}-abc3.log"     2>&1 &
pipeline_wiggle150 "$WIG_NODE"  > "${LOG}-wiggle150.log" 2>&1 &
pipeline_ms25      "$MS25_NODE" > "${LOG}-ms25.log"     2>&1 &

wait
echo "### All 5 pipelines finished at $(date)."
echo "### Per-dataset logs: ${LOG}-{qm9,md17,abc3,wiggle150,ms25}.log"
echo "### Results in examples/*/benchmark_results/"
