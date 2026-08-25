#!/bin/bash
# Serve the pi0.5 LIBERO checkpoint with QuantVLA quantization
# (DuQuant W4A8 + ATM + OHB) for the RoboCerebra / LIBERO evals.
#
# Usage: ./run_libero_serve_quant.sh
#   LIBERO_QUANT_GPU=1  LIBERO_QUANT_PORT=8002  ./run_libero_serve_quant.sh
set -e

source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate openpi

export CUDA_VISIBLE_DEVICES=${LIBERO_QUANT_GPU:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}

# QuantVLA quantization (same recipe as pi05/table2_results_local.md).
# torch.compile must stay disabled or the first quantized forward hangs.
export TORCHDYNAMO_DISABLE=1
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=64
export OPENPI_DUQUANT_LS=0.15
export OPENPI_DUQUANT_PERMUTE=0
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=160
export OPENPI_DUQUANT_PACKDIR=${LIBERO_PACKDIR:-/home1/gyy/vla/QuantVLA/code/pi05/packs/pi05_w4a8_b64c160ls015}
export OPENPI_ATM_ENABLE=1
export OPENPI_ATM_ALPHA_PATH=${LIBERO_ATM_PATH:-/home1/gyy/vla/QuantVLA/code/pi05/packs/atm_alpha_beta_pi05.json}
export OPENPI_ATM_SCOPE=expert
export OPENPI_OHB_ENABLE=1
export OPENPI_OHB_SCOPE=expert

CKPT=${LIBERO_CKPT:-/home1/gyy/vla/QuantVLA/code/pi05/checkpoints/pi05_libero_pytorch}
PORT=${LIBERO_QUANT_PORT:-8002}

cd /home1/gyy/vla/QuantVLA/code/pi05/openpi
exec python scripts/serve_pi05_quant_policy.py \
  --env LIBERO \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$CKPT"
