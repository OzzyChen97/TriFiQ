#!/bin/bash
# Activate pi0.5 (openpi) env
source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate openpi
export PYTHONPATH=/home1/gyy/vla/QuantVLA/code/pi05/openpi:$PYTHONPATH
echo "openpi activated (pi0.5)"
