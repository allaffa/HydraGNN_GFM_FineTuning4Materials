#!/usr/bin/env bash
# Build a dedicated MACE virtualenv that shadows e3nn with the 0.4.4 release
# required to deserialize the MACE foundation-model checkpoints.
#
# It layers on top of the main ROCm 7.2 venv via --system-site-packages, so it
# REUSES torch/torchvision/PyG/HydraGNN (no multi-GB re-download) and only adds
# e3nn==0.4.4 + mace-torch into its own site-packages, which take precedence.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
BASE_VENV="$REPO_ROOT/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72"
MACE_VENV="$REPO_ROOT/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72"

# Correct CA bundle for this SUSE system (pip/requests default path is wrong).
export REQUESTS_CA_BUNDLE=/etc/ssl/ca-bundle.pem
export CURL_CA_BUNDLE=/etc/ssl/ca-bundle.pem
export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem
unset PIP_CONSTRAINT

echo "== Creating MACE venv (--system-site-packages from base) =="
"$BASE_VENV/bin/python" -m venv --system-site-packages "$MACE_VENV"

PIP="$MACE_VENV/bin/pip"
"$PIP" install --upgrade pip >/dev/null 2>&1 || true

# Protect the inherited core so pip never re-pulls torch/numpy/scipy.
CONS=$(mktemp)
cat > "$CONS" <<EOF
torch==2.13.0+rocm7.2
torchvision==0.28.0+rocm7.2
numpy==1.26.4
scipy==1.14.1
EOF

echo "== Installing e3nn==0.4.4 into MACE venv (shadows base 0.6.0) =="
"$PIP" install --no-cache-dir -c "$CONS" "e3nn==0.4.4"
RC1=$?

echo "== Installing mace-torch==0.3.16 (deps satisfied by inherited base) =="
"$PIP" install --no-cache-dir --no-deps "mace-torch==0.3.16"
RC2=$?

echo "== Verifying MACE venv =="
"$MACE_VENV/bin/python" - <<'PY'
import torch, e3nn
print("torch", torch.__version__, "HIP", torch.cuda.is_available())
print("e3nn", e3nn.__version__)
import mace
print("mace", mace.__version__)
PY
RC3=$?

echo "BUILD_RC e3nn=$RC1 mace=$RC2 verify=$RC3"
