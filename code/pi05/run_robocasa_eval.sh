#!/bin/bash
# RoboCasa365 pi0.5 eval client. Talks to a policy server started with
# run_robocasa_serve.sh (fp16, :8003) or run_robocasa_serve_quant.sh (W4A8, :8004).
#
# Usage:
#   ./run_robocasa_eval.sh --args.port 8003 --args.task_set atomic_seen \
#       --args.num_trials 3 --args.task_filter CloseFridge,TurnOnElectricKettle
# Full benchmark (50 rollouts/task):
#   ./run_robocasa_eval.sh --args.port 8003 \
#       --args.task_set atomic_seen composite_seen composite_unseen --args.num_trials 50
set -e

source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate robocasa365

export PYTHONPATH=/home1/gyy/vla/QuantVLA/code/pi05/openpi/packages/openpi-client/src:$PYTHONPATH
export MUJOCO_GL=egl
export ROBOCASA_LOG_DIR=${ROBOCASA_LOG_DIR:-/home1/gyy/vla/QuantVLA/checkpoints/robocasa/pi05_pretrain_human300/multitask_learning/75000}

cd /home1/gyy/vla/QuantVLA/code/pi05/openpi/examples/robocasa
exec python main.py --args.log_dir "$ROBOCASA_LOG_DIR" "$@"
