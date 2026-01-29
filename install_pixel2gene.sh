#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pixel2Gene one-shot installer (method 3 / plan B)
# - conda/mamba: only core + ABI-sensitive packages
# - pip: most extras (wheels) to avoid slow/conflicting solves
# ============================================================

ENV_NAME="pixel2gene"
CORE_YML="environment.core.yml"
CUDA_OVERRIDE="${CUDA_OVERRIDE:-12.4}"     # important for __cuda virtual package

# -------- helpers --------
info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

need_cmd conda

# Init conda in non-interactive shell
# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh || die "Failed to source conda.sh"

# Use mamba if available; fallback to conda
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
else
  SOLVER="conda"
  warn "mamba not found; using conda (may be slower). Recommend:"
  warn "  conda install -n base -c conda-forge mamba"
fi

# Speed-ups
export CONDA_REPODATA_FNS="${CONDA_REPODATA_FNS:-current_repodata.json}"
export CONDA_OVERRIDE_CUDA="${CUDA_OVERRIDE}"

# Use classic solver to avoid libmamba plugin weirdness in conda
conda config --set solver classic >/dev/null 2>&1 || true

# -------- create env if missing --------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  info "Conda env '${ENV_NAME}' already exists. Skipping core creation."
else
  [[ -f "${CORE_YML}" ]] || die "Cannot find ${CORE_YML} in current directory."
  info "Creating conda env '${ENV_NAME}' from ${CORE_YML} (solver=${SOLVER}) ..."
  # NOTE: mamba env create sometimes doesn't like -y; keep it clean.
  ${SOLVER} env create -f "${CORE_YML}"
fi

info "Setting channel priority=strict (env-local)"
conda config --env --set channel_priority strict

# Helper: run inside env without relying on conda activate
run_in_env() {
  conda run -n "${ENV_NAME}" "$@"
}

# ============================================================
# Group A: single-cell / ST stack (conda-forge)  [ABI-heavy]
# ============================================================
info "Installing single-cell / ST stack (conda-forge)..."
${SOLVER} install -y -n "${ENV_NAME}" -c conda-forge --strict-channel-priority \
  anndata=0.9.2 \
  scanpy=1.9.3 \
  h5py=3.10.0 \
  zarr=2.16.1 \
  natsort=8.4.0 \
  statsmodels=0.14.1 \
  umap-learn=0.5.6

# ============================================================
# Group B: pyarrow  [try conda CPU build; fallback to pip wheel]
# ============================================================
info "Installing pyarrow..."
set +e
${SOLVER} install -y -n "${ENV_NAME}" -c conda-forge --strict-channel-priority \
  pyarrow=17.0.0
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  warn "conda/mamba pyarrow solve failed; falling back to pip pyarrow==17.0.0"
  run_in_env python -m pip install --no-cache-dir pyarrow==17.0.0
fi

# ============================================================
# Group C: Imaging / IO (pip wheels — fast & py38-friendly)
# NOTE: imagecodecs wheels are flaky on py38; install without hard pin.
# ============================================================
info "Installing imaging / IO stack (pip)..."
run_in_env python -m pip install --no-cache-dir \
  pillow==9.4.0 \
  tifffile==2023.7.10 \
  imageio==2.34.0 \
  opencv-python==4.9.0.80 \
  scikit-image==0.19.3 \
  brokenaxes==0.6.2

# Optional: try imagecodecs (best-effort)
set +e
run_in_env python -m pip install --no-cache-dir imagecodecs
set -e

# ============================================================
# Group D: DL utilities (pip-only)
# ============================================================
info "Installing DL / training utilities (pip)..."
run_in_env python -m pip install --no-cache-dir \
  timm==1.0.11 \
  einops==0.7.0 \
  lightning==2.3.3 \
  pytorch-lightning==2.0.8 \
  torchmetrics==1.4.1 \
  torch-geometric==2.6.1 \
  huggingface-hub==0.26.2 \
  webdataset==0.2.86 \
  tqdm-joblib==0.0.4 \
  jsonpickle==4.0.0 \
  safetensors==0.4.5

# ============================================================
# Sanity checks
# ============================================================
info "Running sanity checks..."
run_in_env python - << 'EOF'
import sys
print("python:", sys.version.split()[0])

import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())

import cv2
print("opencv:", cv2.__version__)

import tifffile
print("tifffile:", tifffile.__version__)

import imageio
print("imageio:", imageio.__version__)

import scanpy as sc
print("scanpy:", sc.__version__)

import anndata as ad
print("anndata:", ad.__version__)

try:
    import pyarrow as pa
    print("pyarrow:", pa.__version__)
except Exception as e:
    print("pyarrow import failed:", repr(e))

try:
    import skimage
    print("skimage:", skimage.__version__)
except Exception as e:
    print("skimage import failed:", repr(e))
EOF

info "✅ Done. Activate with: conda activate ${ENV_NAME}"