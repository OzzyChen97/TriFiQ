#!/usr/bin/env bash
set -euo pipefail

# Crash-safe worker for preregistered pi0.5 week-1 controls.  SCOPE is part of
# the frozen schedule: all50 evaluates every official task; heldout46 excludes
# the four development tasks; dev4 evaluates only those four tasks.

if [[ $# -ne 10 ]]; then
    echo "usage: $0 CONFIG PORT EGL_GPU SHARD_INDEX SHARD_COUNT TRIAL_SEEDS WORKER_ID METADATA_SHA256 RUN_DIR SCOPE" >&2
    exit 2
fi

CONFIG="$1"
PORT="$2"
EGL_GPU="$3"
SHARD_INDEX="$4"
SHARD_COUNT="$5"
TRIAL_SEEDS="$6"
WORKER_ID="$7"
METADATA_SHA256="$8"
RUN_DIR="$9"
SCOPE="${10}"

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
DEV4="CoffeeSetupMug,OpenCabinet,OpenStandMixerHead,PickPlaceDrawerToCounter"
HELDOUT_ATOMIC14="CloseBlenderLid,CloseFridge,CloseToasterOvenDoor,NavigateKitchen,OpenDrawer,PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceSinkToCounter,PickPlaceToasterToCounter,SlideDishwasherRack,TurnOffStove,TurnOnElectricKettle,TurnOnMicrowave,TurnOnSinkFaucet"

for numeric in "$PORT" "$EGL_GPU" "$SHARD_INDEX" "$SHARD_COUNT"; do
    if [[ ! "$numeric" =~ ^[0-9]+$ ]]; then
        echo "invalid numeric argument: $numeric" >&2
        exit 2
    fi
done
if [[ ! "$TRIAL_SEEDS" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "week-1 worker requires one inclusive seed range" >&2
    exit 2
fi
if [[ ! "$WORKER_ID" =~ ^[A-Za-z0-9_.-]+$ || ! "$CONFIG" =~ ^[A-Za-z0-9_.-]+$ || ! "$METADATA_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid config, worker id, or metadata hash" >&2
    exit 2
fi
case "$SCOPE" in
    all50) TASK_SETS=(atomic_seen composite_seen composite_unseen) ;;
    heldout46) TASK_SETS=(atomic_seen composite_seen composite_unseen) ;;
    dev4) TASK_SETS=(atomic_seen) ;;
    *) echo "invalid scope: $SCOPE" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR/results/$CONFIG"
export PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

for task_set in "${TASK_SETS[@]}"; do
    task_args=()
    if [[ "$SCOPE" == "dev4" ]]; then
        task_args+=(--tasks "$DEV4")
    elif [[ "$SCOPE" == "heldout46" && "$task_set" == "atomic_seen" ]]; then
        task_args+=(--tasks "$HELDOUT_ATOMIC14")
    fi
    output="$RUN_DIR/results/$CONFIG/${task_set}_${WORKER_ID}.jsonl"
    "$PYTHON" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --config-id "$CONFIG" \
        --task-set "$task_set" \
        "${task_args[@]}" \
        --task-shard-index "$SHARD_INDEX" \
        --task-shard-count "$SHARD_COUNT" \
        --trial-seeds "$TRIAL_SEEDS" \
        --split target \
        --replan-steps 16 \
        --egl-device "$EGL_GPU" \
        --expected-server-metadata-sha256 "$METADATA_SHA256" \
        --resume-dir "$RUN_DIR/results/$CONFIG" \
        --out "$output"
done

echo "pi05 week1 worker complete: $WORKER_ID scope=$SCOPE seeds=$TRIAL_SEEDS config=$CONFIG shard=$SHARD_INDEX/$SHARD_COUNT"
