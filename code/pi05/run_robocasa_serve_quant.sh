#!/bin/bash
# Serve the RoboCasa365 pi0.5 checkpoint with QuantVLA quantization
# (DuQuant W4A8 + ATM + OHB) for eval.
#
# Usage: ./run_robocasa_serve_quant.sh
#   ROBOCASA_QUANT_GPU=6  ROBOCASA_QUANT_PORT=8004  ./run_robocasa_serve_quant.sh
set -e

source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate openpi

export CUDA_VISIBLE_DEVICES=${ROBOCASA_QUANT_GPU:-6}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}

# QuantVLA quantization (same recipe as the LIBERO pi05 servers, see
# pi05/table2_results_local.md). torch.compile must stay disabled or the
# first quantized forward hangs for hours (max-autotune over 180 layers).
export TORCHDYNAMO_DISABLE=1
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=64
export OPENPI_DUQUANT_LS=0.15
export OPENPI_DUQUANT_PERMUTE=0
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=160
export OPENPI_DUQUANT_PACKDIR=${ROBOCASA_PACKDIR:-/home1/gyy/vla/QuantVLA/code/pi05/packs/pi05_robocasa_w4a8_b64c160ls015}
export OPENPI_ATM_ENABLE=1
export OPENPI_ATM_ALPHA_PATH=${ROBOCASA_ATM_PATH:-/home1/gyy/vla/QuantVLA/code/pi05/packs/atm_alpha_beta_pi05_robocasa.json}
export OPENPI_ATM_SCOPE=expert
export OPENPI_OHB_ENABLE=1
export OPENPI_OHB_SCOPE=expert

CKPT=${ROBOCASA_CKPT:-/home1/gyy/vla/QuantVLA/checkpoints/robocasa/pi05_pretrain_human300_pytorch}
PORT=${ROBOCASA_QUANT_PORT:-8004}

cd /home1/gyy/vla/QuantVLA/code/pi05/openpi
exec python scripts/serve_pi05_quant_policy.py \
  --env LIBERO \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config pi05_pretrain_human300 \
  --policy.dir "$CKPT"
