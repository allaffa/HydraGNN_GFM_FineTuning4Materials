#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-download all foundation-model checkpoints on a Frontier LOGIN node.
#
# Frontier compute nodes have no outbound internet, so the MACE and UMA
# (fairchem) weights MUST be cached before submitting the batch jobs.
# Run this once on a login node:
#
#   export HYDRAGNN_VENV=/path/to/your/env
#   bash frontier/prefetch_models.sh
#
# It populates $MACE_CACHE and $HF_HOME, which the batch jobs reuse.
# UMA (gated on Hugging Face) requires a valid HF token:  `huggingface-cli login`.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
source "${REPO_ROOT}/frontier/env_frontier.sh"

echo "== Prefetching MACE-OFF checkpoints =="
python - <<'PY'
from utils.mace_calculator import MACE_MODELS, build_mace_calculator
for mid in MACE_MODELS:
    print(f"  fetching {mid} ...", flush=True)
    try:
        build_mace_calculator(mid, device="cpu")
    except Exception as exc:
        print(f"    WARN: {mid} failed: {exc}")
print("MACE prefetch done.")
PY

echo "== Prefetching UMA checkpoint (uma-s-1p2) =="
python - <<'PY'
try:
    from fairchem.core import pretrained_mlip
    # load_predict_unit expects a *path*; resolve the registered model name to
    # its cached checkpoint first (this triggers the HuggingFace download into
    # $HF_HOME), exactly like utils/uma_calculator.py / utils/uma_finetune.py.
    ckpt = pretrained_mlip.pretrained_checkpoint_path_from_name("uma-s-1p2")
    pretrained_mlip.load_predict_unit(ckpt, device="cpu",
                                      inference_settings="default")
    print(f"UMA prefetch done. checkpoint={ckpt}")
except Exception as exc:
    print(f"WARN: UMA prefetch failed: {exc}")
    print("If this is an auth error run: huggingface-cli login")
PY

echo "All prefetch steps complete. Caches:"
echo "  MACE_CACHE=${MACE_CACHE}"
echo "  HF_HOME=${HF_HOME}"
