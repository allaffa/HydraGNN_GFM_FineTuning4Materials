#!/bin/bash
# ---------------------------------------------------------------------------
# Frontier batch job: full Wiggle150 (conformational energy) comparison
# pipeline.  HydraGNN benchmark -> MACE fine-tune -> UMA fine-tune ->
# zero-shot MACE/UMA benchmarks (accuracy + wall-clock timing per model).
#
# Submit from the repo root:
#   export HYDRAGNN_VENV=/lustre/orion/<proj>/.../hydragnn_venv
#   sbatch frontier/submit_wiggle150.sbatch     # or: sbatch -A <PROJ> ...
#
# Prefetch model weights on a login node FIRST:  bash frontier/prefetch_models.sh
# ---------------------------------------------------------------------------
#SBATCH -A LRN070
#SBATCH -J wig150_gfm
#SBATCH -o logs/frontier/wiggle150-%j.out
#SBATCH -e logs/frontier/wiggle150-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1

export REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${REPO_ROOT}/logs/frontier"
source "${REPO_ROOT}/frontier/env_frontier.sh"

# Abort the pipeline if any step below fails.
set -euo pipefail

SRUN="srun -N1 -n1 -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest"

cd "${REPO_ROOT}/examples/wiggle150"
RESULTS=benchmark_results

echo "### [1/5] HydraGNN benchmark"
$SRUN python -u run_benchmark.py

echo "### [2/5] MACE fine-tune"
$SRUN python -u run_mace_finetune.py --device cuda

echo "### [3/5] UMA fine-tune"
$SRUN python -u run_uma_finetune.py --device cuda

echo "### [4/5] UMA zero-shot benchmark"
$SRUN python -u run_uma_benchmark.py --device cuda \
    --hydragnn-summary "${RESULTS}/benchmark_summary.json"

echo "### [5/5] MACE zero-shot benchmark (final comparison table)"
$SRUN python -u run_mace_benchmark.py --device cuda \
    --hydragnn-summary "${RESULTS}/benchmark_summary.json" \
    --uma-summary "${RESULTS}/uma_benchmark_summary.json"

echo "### Wiggle150 pipeline complete. Results in examples/wiggle150/${RESULTS}/"
