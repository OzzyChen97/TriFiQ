#!/bin/bash
# Activate LIBERO eval client env
source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate libero_test
export PYTHONPATH=/home1/gyy/vla/QuantVLA:/home1/gyy/vla/QuantVLA/code/LIBERO:$PYTHONPATH
echo "libero_test activated (LIBERO eval client)"
