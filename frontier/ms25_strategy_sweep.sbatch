#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J ms25_ft_sweep
#SBATCH -o logs/frontier/ms25-sweep-%j.out
#SBATCH -e logs/frontier/ms25-sweep-%j.out
#SBATCH -t 3-00:00:00
#SBATCH -p batch
#SBATCH -N 3

set -euo pipefail

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export FRONTIER_ROCM="${FRONTIER_ROCM:-7.2.0}"
export HYDRAGNN_VENV="${HYDRAGNN_VENV:-${REPO_ROOT}/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
export MACE_VENV="${MACE_VENV:-${REPO_ROOT}/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72}"
export HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export UMA_EPOCHS="${UMA_EPOCHS:-2}"
export MACE_EPOCHS="${MACE_EPOCHS:-5}"

mkdir -p "${REPO_ROOT}/logs/frontier"
source "${REPO_ROOT}/frontier/env_frontier.sh"
MACE_PY="${MACE_VENV}/bin/python"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HYDRA_NODE="${NODES[0]}"
UMA_NODE="${NODES[1]}"
MACE_NODE="${NODES[2]}"
SYSTEMS=(MgO-2x2 MgO-4x4 H2O-64 H2O-192 CHA HEA Reaction Zr-O)
MS25="${REPO_ROOT}/examples/ms25"
RUNS="${MS25}/benchmark_results/strategy_runs"

srun_gcd() {
    local node=$1 gcd=$2; shift 2
    srun -N1 -n1 -c7 --gpus-per-node=8 --overlap -w "$node" \
         --export=ALL,HIP_VISIBLE_DEVICES="$gcd",ROCR_VISIBLE_DEVICES="$gcd" "$@"
}

result_ok() {
    local path=$1
    [[ -f "$path" ]] && python3 -c \
        "import json; d=json.load(open('$path')); raise SystemExit(0 if any(isinstance(v, dict) and 'testset' in v for v in d.values()) else 1)"
}

run_hydragnn() {
    cd "$MS25"
    for strategy in frozen_message_passing frozen_decoder; do
        pids=()
        gcd=0
        for system in "${SYSTEMS[@]}"; do
            result="${MS25}/logs/${system}_${strategy}_seed0/benchmark_results/checkpoint_benchmark_summary.json"
            if [[ -f "$result" ]]; then
                echo "[HydraGNN/$strategy/$system] skip: result exists"
            else
                srun_gcd "$HYDRA_NODE" "$gcd" python -u ensemble_fine_tune.py \
                    --system "$system" --strategy "$strategy" &
                pids+=($!)
            fi
            gcd=$((gcd + 1))
        done
        if (( ${#pids[@]} )); then wait "${pids[@]}"; fi

        pids=()
        gcd=0
        for system in "${SYSTEMS[@]}"; do
            result="${MS25}/logs/${system}_${strategy}_seed0/benchmark_results/checkpoint_benchmark_summary.json"
            if [[ ! -f "$result" ]]; then
                srun_gcd "$HYDRA_NODE" "$gcd" python -u evaluate_hydragnn_checkpoint.py \
                    --system "$system" --strategy "$strategy" &
                pids+=($!)
            fi
            gcd=$((gcd + 1))
        done
        if (( ${#pids[@]} )); then wait "${pids[@]}"; fi
    done
}

run_uma() {
    cd "$MS25"
    for strategy in full frozen_backbone lora; do
        mkdir -p "$RUNS/uma/$strategy"
        pids=()
        gcd=0
        for system in "${SYSTEMS[@]}"; do
            output="$RUNS/uma/$strategy/${system}.json"
            if result_ok "$output"; then
                echo "[UMA/$strategy/$system] skip: result exists"
            else
                extra=()
                [[ "$strategy" == frozen_backbone ]] && extra+=(--freeze-backbone)
                [[ "$strategy" == lora ]] && extra+=(--lora)
                srun_gcd "$UMA_NODE" "$gcd" python -u run_uma_finetune.py \
                    --systems "$system" --device cuda --epochs "$UMA_EPOCHS" \
                    --tag "$strategy" --output-path "$output" "${extra[@]}" &
                pids+=($!)
            fi
            gcd=$((gcd + 1))
        done
        if (( ${#pids[@]} )); then wait "${pids[@]}"; fi
    done
}

run_mace() {
    cd "$MS25"
    for strategy in full lora; do
        mkdir -p "$RUNS/mace/$strategy"
        pids=()
        gcd=0
        for system in "${SYSTEMS[@]}"; do
            output="$RUNS/mace/$strategy/${system}.json"
            if result_ok "$output"; then
                echo "[MACE/$strategy/$system] skip: result exists"
            else
                extra=()
                [[ "$strategy" == lora ]] && extra+=(--lora)
                srun_gcd "$MACE_NODE" "$gcd" "$MACE_PY" -u run_mace_finetune.py \
                    --systems "$system" --device cuda --epochs "$MACE_EPOCHS" \
                    --tag "$strategy" --output-path "$output" "${extra[@]}" &
                pids+=($!)
            fi
            gcd=$((gcd + 1))
        done
        if (( ${#pids[@]} )); then wait "${pids[@]}"; fi
    done
}

run_hydragnn > "${REPO_ROOT}/logs/frontier/ms25-sweep-${SLURM_JOB_ID}-hydragnn.log" 2>&1 &
HYDRA_PID=$!
run_uma > "${REPO_ROOT}/logs/frontier/ms25-sweep-${SLURM_JOB_ID}-uma.log" 2>&1 &
UMA_PID=$!
run_mace > "${REPO_ROOT}/logs/frontier/ms25-sweep-${SLURM_JOB_ID}-mace.log" 2>&1 &
MACE_PID=$!

status=0
wait "$HYDRA_PID" || status=1
wait "$UMA_PID" || status=1
wait "$MACE_PID" || status=1
cd "$MS25"
python3 collect_strategy_results.py
exit "$status"
