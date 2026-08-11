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

# --- Ensure the HydraGNN ensemble checkpoint has the base-name that ------------
# hydragnn.utils.model.load_existing_model expects.  The distributed pretraining
# saved it as "<name>_epoch_97.pk"; load_existing_model looks for "<name>.pk".
# abc3/ms25 (ensemble_fine_tune.py) use load_existing_model directly, so create a
# symlink if the base-named file is missing (idempotent, no-op if it exists).
GFM_DIR="${REPO_ROOT}/pretrained_model_ensemble/multidataset_hpo-BEST6-fp64"
GFM_BASE="${GFM_DIR}/multidataset_hpo-BEST6-fp64.pk"
if [ ! -e "${GFM_BASE}" ]; then
    GFM_EPOCH=$(ls -1 "${GFM_DIR}"/multidataset_hpo-BEST6-fp64_epoch_*.pk 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "${GFM_EPOCH}" ]; then
        ln -s "$(basename "${GFM_EPOCH}")" "${GFM_BASE}"
        echo "Linked checkpoint: $(basename "${GFM_BASE}") -> $(basename "${GFM_EPOCH}")"
    else
        echo "WARNING: no GFM checkpoint found in ${GFM_DIR}" >&2
    fi
fi

# --- Helper: srun pinned to one node, overlapping with all other pipelines ----
# All srun calls use --overlap so the 5 dataset pipelines (and the concurrent
# UMA steps within each) can all run simultaneously.
srun_on() {
    local node=$1; shift
    srun -N1 -n1 -c"${OMP_NUM_THREADS}" --gpus-per-task=1 --gpu-bind=closest \
         --overlap -w "$node" "$@"
}

# --- Helper: srun pinned to one node AND one specific GCD ---------------------
# Used for the concurrent UMA fine-tune variants (full/frozen/lora) so they run
# on SEPARATE GCDs instead of contending on GCD 0.  On Frontier/MI250X the
# correct env var is HIP_VISIBLE_DEVICES (ROCR_VISIBLE_DEVICES is ignored by
# the ROCm 7.x runtime under SLURM).  Each process sees exactly one device (cuda:0).
srun_gcd() {
    local node=$1 gcd=$2; shift 2
    srun -N1 -n1 -c"${OMP_NUM_THREADS}" --gpus-per-node=8 --overlap \
         -w "$node" --export=ALL,HIP_VISIBLE_DEVICES="$gcd",ROCR_VISIBLE_DEVICES="$gcd" "$@"
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

    if [ ! -f "${RES}/mace_finetuned_lora_summary.json" ]; then
        echo "### QM9 [2b/5] MACE LoRA fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --tag lora --lora
    else
        echo "### QM9 [2b/5] MACE LoRA fine-tune — SKIPPED (results exist)"
    fi

    echo "### QM9 [3/5] UMA fine-tune (frozen then LoRA — sequential to avoid GCD contention)"
    if [ ! -f "${RES}/uma_finetuned_frozen_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone \
            || { echo "QM9 UMA frozen failed" >&2; exit 1; }
    else
        echo "### QM9 [3/5] UMA frozen — SKIPPED (results exist)"
    fi
    if [ ! -f "${RES}/uma_finetuned_lora_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora \
            || { echo "QM9 UMA lora failed" >&2; exit 1; }
    else
        echo "### QM9 [3/5] UMA lora — SKIPPED (results exist)"
    fi

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

    if [ ! -f "${RES}/mace_finetuned_lora_summary.json" ]; then
        echo "### MD17 [2b/5] MACE LoRA fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --tag lora --lora
    else
        echo "### MD17 [2b/5] MACE LoRA fine-tune — SKIPPED (results exist)"
    fi

    echo "### MD17 [3/5] UMA fine-tune (full then frozen then LoRA — sequential)"
    if [ ! -f "${RES}/uma_finetuned_full_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag full --epochs 4 \
            || { echo "MD17 UMA full failed" >&2; exit 1; }
    else
        echo "### MD17 [3/5] UMA full — SKIPPED (results exist)"
    fi
    if [ ! -f "${RES}/uma_finetuned_frozen_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone \
            || { echo "MD17 UMA frozen failed" >&2; exit 1; }
    else
        echo "### MD17 [3/5] UMA frozen — SKIPPED (results exist)"
    fi
    if [ ! -f "${RES}/uma_finetuned_lora_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora \
            || { echo "MD17 UMA lora failed" >&2; exit 1; }
    else
        echo "### MD17 [3/5] UMA lora — SKIPPED (results exist)"
    fi

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
# ABC3 pipeline (formation energy; HydraGNN FT + MACE FT + UMA/MACE zero-shot)
# ===========================================================================
pipeline_abc3() {
    local NODE=$1
    local RES="${REPO_ROOT}/examples/abc3/benchmark_results"
    set -euo pipefail

    cd "${REPO_ROOT}"
    if [ ! -f "${RES}/.hydragnn_done" ]; then
        echo "### ABC3 [1/5] HydraGNN fine-tune"
        srun_on "$NODE" python -u examples/abc3/ensemble_fine_tune.py \
            && touch "${RES}/.hydragnn_done"
    else
        echo "### ABC3 [1/5] HydraGNN fine-tune — SKIPPED (results exist)"
    fi

    cd "${REPO_ROOT}/examples/abc3"
    if [ ! -f "${RES}/mace_finetuned_summary.json" ]; then
        echo "### ABC3 [2/5] MACE fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda
    else
        echo "### ABC3 [2/5] MACE fine-tune — SKIPPED (results exist)"
    fi

    if [ ! -f "${RES}/mace_finetuned_lora_summary.json" ]; then
        echo "### ABC3 [2b] MACE LoRA fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --tag lora --lora
    else
        echo "### ABC3 [2b] MACE LoRA fine-tune — SKIPPED (results exist)"
    fi

    echo "### ABC3 [3/5] UMA fine-tune (full only; frozen/lora too slow at 55 min/epoch)"
    if [ ! -f "${RES}/uma_finetuned_full_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag full --epochs 2 \
            || { echo "ABC3 UMA full failed" >&2; exit 1; }
    else
        echo "### ABC3 [3/5] UMA full — SKIPPED (results exist)"
    fi
    # ABC3 has 3185 training samples; UMA takes ~55 min/epoch → frozen/lora
    # at 10 epochs each ≈ 9h: not feasible in 2h debug QOS.  Skip.
    echo "### ABC3 [3/5] UMA frozen/lora — SKIPPED (too slow: ~55 min/epoch × 10ep)"

    echo "### ABC3 [4/5] UMA zero-shot"
    if [ ! -f "${RES}/uma_benchmark_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_benchmark.py --device cuda \
            --hydragnn-summary "${RES}/benchmark_summary.json"
    else
        echo "### ABC3 [4/5] UMA zero-shot — SKIPPED (results exist)"
    fi

    echo "### ABC3 [5/5] MACE zero-shot"
    if [ ! -f "${RES}/mace_benchmark_summary.json" ]; then
        srun_on "$NODE" "${MACE_PY}" -u run_mace_benchmark.py --device cuda \
            --hydragnn-summary "${RES}/benchmark_summary.json" \
            --uma-summary "${RES}/uma_benchmark_summary.json"
    else
        echo "### ABC3 [5/5] MACE zero-shot — SKIPPED (results exist)"
    fi
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

    if [ ! -f "${RES}/mace_finetuned_lora_summary.json" ]; then
        echo "### Wiggle150 [2b/5] MACE LoRA fine-tune"
        srun_on "$NODE" "${MACE_PY}" -u run_mace_finetune.py --device cuda --tag lora --lora
    else
        echo "### Wiggle150 [2b/5] MACE LoRA fine-tune — SKIPPED (results exist)"
    fi

    echo "### Wiggle150 [3/5] UMA fine-tune (full then frozen then LoRA — sequential)"
    if [ ! -f "${RES}/uma_finetuned_full_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag full --epochs 10 \
            || { echo "Wiggle150 UMA full failed" >&2; exit 1; }
    else
        echo "### Wiggle150 [3/5] UMA full — SKIPPED (results exist)"
    fi
    if [ ! -f "${RES}/uma_finetuned_frozen_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag frozen --freeze-backbone \
            || { echo "Wiggle150 UMA frozen failed" >&2; exit 1; }
    else
        echo "### Wiggle150 [3/5] UMA frozen — SKIPPED (results exist)"
    fi
    if [ ! -f "${RES}/uma_finetuned_lora_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_finetune.py --device cuda --tag lora --lora \
            || { echo "Wiggle150 UMA lora failed" >&2; exit 1; }
    else
        echo "### Wiggle150 [3/5] UMA lora — SKIPPED (results exist)"
    fi

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

    if [ ! -f "${RES}/hydragnn_benchmark_summary.json" ]; then
        echo "### MS25 [1/4] HydraGNN fine-tune (all 8 systems in parallel, one per GCD)"
        TRAIN_PIDS=()
        gcd=0
        for system in MgO-2x2 MgO-4x4 H2O-64 H2O-192 CHA HEA Reaction Zr-O; do
            srun_gcd "$NODE" $gcd python -u ensemble_fine_tune.py --system "$system" &
            TRAIN_PIDS+=($!)
            gcd=$((gcd+1))
        done
        # watchdog: kill remaining training after 90 min using explicit PIDs
        ( sleep 90m && echo "[watchdog] 90 min elapsed — killing remaining training" \
            && kill "${TRAIN_PIDS[@]}" 2>/dev/null ) &
        WATCHDOG=$!
        wait "${TRAIN_PIDS[@]}" || true   # || true: some may be killed by watchdog
        kill "$WATCHDOG" 2>/dev/null || true
        python3 -u -c "
import json, os, sys
systems = ['MgO-2x2','MgO-4x4','H2O-64','H2O-192','CHA','HEA','Reaction','Zr-O']
results = {}
for s in systems:
    fp = os.path.join('logs', s + '_mlip_seed0', 'benchmark_results', 'benchmark_summary.json')
    if os.path.isfile(fp):
        with open(fp) as f:
            results[s] = json.load(f)
    else:
        print(f'[WARN] no result for {s}', file=sys.stderr)
if not results:
    print('[ERROR] no per-system results found — not writing summary', file=sys.stderr)
    sys.exit(1)
out = '${RES}/hydragnn_benchmark_summary.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'HydraGNN MS25 results saved ({len(results)}/8 systems): {out}')
"
    else
        echo "### MS25 [1/4] HydraGNN fine-tune — SKIPPED (results exist)"
    fi

    # MS25 MACE/UMA FT: infeasible (large systems, ~2h per system).
    echo "### MS25 [2/4] MACE fine-tune — SKIPPED (8 systems × ~2h each, not feasible in 2h debug QOS)"
    echo "### MS25 [3/4] UMA frozen/lora — SKIPPED (large dataset, not feasible in 2h debug QOS)"

    echo "### MS25 [4/4] Zero-shot benchmarks"
    if [ ! -f "${RES}/uma_benchmark_summary.json" ]; then
        srun_on "$NODE" python -u run_uma_benchmark.py --device cuda
    else
        echo "### MS25 [4/4] UMA zero-shot — SKIPPED (results exist)"
    fi
    if [ ! -f "${RES}/mace_benchmark_summary.json" ]; then
        srun_on "$NODE" "${MACE_PY}" -u run_mace_benchmark.py --device cuda
    else
        echo "### MS25 [4/4] MACE zero-shot — SKIPPED (results exist)"
    fi

    echo ""
    echo "============================================================"
    echo "  MS25 — energy MAE (meV/atom)"
    echo "============================================================"
    python3 -c "
import json, os
res_dir = '${RES}'
systems = ['MgO-2x2','MgO-4x4','H2O-64','H2O-192','CHA','HEA','Reaction','Zr-O']
hgnn_f = os.path.join(res_dir, 'hydragnn_benchmark_summary.json')
mace_f = os.path.join(res_dir, 'mace_benchmark_summary.json')
uma_f  = os.path.join(res_dir, 'uma_benchmark_summary.json')
hgnn = json.load(open(hgnn_f)) if os.path.isfile(hgnn_f) else {}
mace_root = json.load(open(mace_f)) if os.path.isfile(mace_f) else {}
mace = next(iter(mace_root.values()), {}) if mace_root else {}
uma  = json.load(open(uma_f))  if os.path.isfile(uma_f)  else {}
print(f'  {\"System\":<14} {\"HydraGNN FT\":>14} {\"MACE-MP-0 ZS\":>14} {\"UMA ZS\":>10}')
print('  ' + '-'*56)
for s in systems:
    h = hgnn.get(s, {})
    hv = h.get('test_mae_eV_per_atom', h.get('test_mae_eV', float('nan'))) * 1000 if h else float('nan')
    mv = mace.get(s, {}).get('testset', {}).get('energy_mae_meV_per_atom', float('nan'))
    uv = uma.get(s, {}).get('testset', {}).get('energy_mae_meV_per_atom', float('nan'))
    print(f'  {s:<14} {hv:>14.2f} {mv:>14.2f} {uv:>10.2f}')
" 2>/dev/null || true
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
