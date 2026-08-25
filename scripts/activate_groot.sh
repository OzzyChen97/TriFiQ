#!/bin/bash
# Activate GR00T N1.5 inference server env
source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate groot_test
export LD_LIBRARY_PATH=/home1/gyy/probe/miniforge3/envs/groot_test/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
echo "groot_test activated (GR00T N1.5)"
