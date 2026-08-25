#!/bin/bash
# Serve the pi0.5 LIBERO checkpoint (fp16/bf16) for the RoboCerebra / LIBERO evals.
#
# Usage: ./run_libero_serve.sh
#   LIBERO_GPU=6  LIBERO_PORT=8001  ./run_libero_serve.sh
set -e

source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate openpi

export CUDA_VISIBLE_DEVICES=${LIBERO_GPU:-6}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}

CKPT=${LIBERO_CKPT:-/home1/gyy/vla/QuantVLA/code/pi05/checkpoints/pi05_libero_pytorch}
PORT=${LIBERO_PORT:-8001}

cd /home1/gyy/vla/QuantVLA/code/pi05/openpi
exec python scripts/serve_policy.py \
  --env LIBERO \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$CKPT"
