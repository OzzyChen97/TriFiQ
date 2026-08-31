#!/usr/bin/env bash
set -u

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
CONTROL=$RUN_ROOT/control/gr00t_unattended
CAPTURE_MIN_FREE_MIB=${GR00T_DYPAC_CAPTURE_MIN_FREE_MIB:-18000}
PIPELINE_MIN_FREE_MIB=${GR00T_DYPAC_PIPELINE_MIN_FREE_MIB:-18000}
POLL_SECONDS=${GR00T_DYPAC_RESUME_POLL_SECONDS:-30}
SCORE_SHARDS_BY_SUITE=${GR00T_DYPAC_SHARDS_BY_SUITE:-2,6,6,7}
SCORE_SLOT_MIB=${GR00T_DYPAC_SCORE_SLOT_MIB:-8100}
SCORE_RESERVE_MIB=${GR00T_DYPAC_SCORE_RESERVE_MIB:-1500}
SCORE_MAX_SLOTS_PER_GPU=${GR00T_DYPAC_SCORE_MAX_SLOTS_PER_GPU:-5}
IFS=, read -r -a SCORE_SHARD_COUNTS <<<"$SCORE_SHARDS_BY_SUITE"
(( ${#SCORE_SHARD_COUNTS[@]} == 4 )) || { echo "GR00T_DYPAC_SHARDS_BY_SUITE needs four values" >&2; exit 2; }
SCORE_REQUIRED=0
for count in "${SCORE_SHARD_COUNTS[@]}"; do
    (( count >= 1 )) || { echo "invalid GR00T suite shard count" >&2; exit 2; }
    SCORE_REQUIRED=$((SCORE_REQUIRED + count))
done
mkdir -p "$CONTROL"
exec 8>"$RUN_ROOT/control/gr00t_resume.lock"
flock -n 8 || { echo "GR00T resume controller is already active" >&2; exit 1; }

phase() {
    printf '%s phase=%s detail=%s\n' "$(date --iso-8601=seconds)" "$1" "${2:-}"
}

best_capture_gpu() {
    local index free best_index= best_free=-1
    while IFS=, read -r index free; do
        index=${index//[[:space:]]/}
        free=${free//[[:space:]]/}
        [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        (( index == 0 )) && continue
        if (( free > best_free )); then
            best_index=$index
            best_free=$free
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    printf '%s %s\n' "$best_index" "$best_free"
}

pipeline_gpus() {
    local index free selected=()
    while IFS=, read -r index free; do
        index=${index//[[:space:]]/}
        free=${free//[[:space:]]/}
        [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        (( index == 0 || free < PIPELINE_MIN_FREE_MIB )) && continue
        selected+=("$index")
        (( ${#selected[@]} == 4 )) && break
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    local IFS=,
    printf '%s\n' "${selected[*]}"
}

pipeline_gpu_slots() {
    local index free capacity round position
    local indices=() capacities=() selected=()
    while IFS=, read -r index free; do
        index=${index//[[:space:]]/}
        free=${free//[[:space:]]/}
        [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        (( index == 0 )) && continue
        capacity=$(( (free - SCORE_RESERVE_MIB) / SCORE_SLOT_MIB ))
        (( capacity < 0 )) && capacity=0
        (( capacity > SCORE_MAX_SLOTS_PER_GPU )) && capacity=$SCORE_MAX_SLOTS_PER_GPU
        if (( capacity > 0 )); then
            indices+=("$index")
            capacities+=("$capacity")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    for ((round=0; round<SCORE_MAX_SLOTS_PER_GPU; round++)); do
        for position in "${!indices[@]}"; do
            if (( capacities[position] > round )); then
                selected+=("${indices[position]}")
                if (( ${#selected[@]} == SCORE_REQUIRED )); then
                    local IFS=,
                    printf '%s\n' "${selected[*]}"
                    return 0
                fi
            fi
        done
    done
    local IFS=,
    printf '%s\n' "${selected[*]}"
    return 1
}

cd "$ROOT" || exit 1
export LIBERO_DYPAC_RUN_ROOT="$RUN_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GR00T_DYPAC_CAPTURE_BATCH_SIZE=1
export GR00T_DYPAC_HESSIAN_BATCH_SIZE="${GR00T_DYPAC_HESSIAN_BATCH_SIZE:-8}"

hessian_gpus() {
    local index free candidates=() selected=() cursor=0
    while IFS=, read -r index free; do
        index=${index//[[:space:]]/}
        free=${free//[[:space:]]/}
        [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        # Hessian builders stream one layer at a time and use under 1 GiB on
        # the current checkpoints.  Admit every visible card with a margin.
        (( free >= 2000 )) && candidates+=("$index")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    (( ${#candidates[@]} > 0 )) || return 1
    while (( ${#selected[@]} < 8 )); do
        selected+=("${candidates[$((cursor % ${#candidates[@]}))]}")
        cursor=$((cursor + 1))
    done
    local IFS=,
    printf '%s\n' "${selected[*]}"
}

while true; do
    read -r gpu free_mib < <(best_capture_gpu)
    if [[ -z "${gpu:-}" ]] || (( free_mib < CAPTURE_MIN_FREE_MIB )); then
        phase wait_capture_gpu "max_free_mib=${free_mib:-0}"
        sleep "$POLL_SECONDS"
        continue
    fi
    four="$gpu,$gpu,$gpu,$gpu"
    hessian_list=$(hessian_gpus) || {
        phase wait_hessian_gpu "no card has 2 GiB free"
        sleep "$POLL_SECONDS"
        continue
    }
    export GR00T_DYPAC_CAPTURE_GPUS="$four"
    export GR00T_DYPAC_HESSIAN_GPUS="$hessian_list"
    phase resume_artifacts "capture_gpu=$gpu hessian_gpus=$hessian_list free_mib=$free_mib"
    if bash "$ROOT/scripts/run_libero_dypac_gr00t_artifacts.sh" all; then
        break
    fi
    phase artifact_retry "previous attempt failed"
    sleep 60
done

while true; do
    score_list=$(pipeline_gpu_slots || true)
    IFS=, read -r -a score_gpus <<<"$score_list"
    if (( ${#score_gpus[@]} < SCORE_REQUIRED )); then
        phase wait_pipeline_gpus "slots=${#score_gpus[@]}/$SCORE_REQUIRED slot_mib=$SCORE_SLOT_MIB"
        sleep "$POLL_SECONDS"
        continue
    fi
    audit_list=$(pipeline_gpus)
    IFS=, read -r -a audit_gpus <<<"$audit_list"
    if (( ${#audit_gpus[@]} < 4 )); then
        phase wait_audit_gpus "ready=${#audit_gpus[@]}/4"
        sleep "$POLL_SECONDS"
        continue
    fi
    export GR00T_DYPAC_SCORE_GPUS="$score_list"
    export GR00T_DYPAC_AUDIT_GPUS="$audit_list"
    export GR00T_DYPAC_SHARDS_BY_SUITE="$SCORE_SHARDS_BY_SUITE"
    export GR00T_DYPAC_START_EVAL=1
    phase resume_pipeline "shards_by_suite=$SCORE_SHARDS_BY_SUITE score_gpus=$score_list audit_gpus=$audit_list"
    exec bash "$ROOT/scripts/run_libero_dypac_gr00t_pipeline.sh"
done
