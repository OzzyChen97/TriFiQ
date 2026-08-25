#!/bin/bash
# Serve the RoboCasa365 pi0.5 checkpoint (fp16/bf16) for eval.
#
# Usage: ./run_robocasa_serve.sh
#   ROBOCASA_GPU=5  ROBOCASA_PORT=8003  ./run_robocasa_serve.sh
set -e

source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate openpi

export CUDA_VISIBLE_DEVICES=${ROBOCASA_GPU:-5}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}

CKPT=${ROBOCASA_CKPT:-/home1/gyy/vla/QuantVLA/checkpoints/robocasa/pi05_pretrain_human300_pytorch}
PORT=${ROBOCASA_PORT:-8003}

cd /home1/gyy/vla/QuantVLA/code/pi05/openpi
exec python scripts/serve_policy.py \
  --env LIBERO \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config pi05_pretrain_human300 \
  --policy.dir "$CKPT"
