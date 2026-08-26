#!/usr/bin/env bash
set -euo pipefail

REPO="/home1/gyy/vla/QuantVLA"
ROOT="$REPO/runs/dpac_softfold_15x20_v1"
LOG="$ROOT/orchestrator.log"
PID_FILE="$ROOT/orchestrator.pid"
PHASE_FILE="$ROOT/phase.txt"
MANIFEST="$REPO/scripts/dpac_softfold_15x20_manifest.json"
GROOT_PY="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
PI_CLIENT="$REPO/code/pi05/openpi/packages/openpi-client/src"
PI_BUFFER="$REPO/runs/pi05_gdsq_gr00t_aligned/diagnostics/fp16_onpolicy_probe/fp16_onpolicy_target_4tasks_s0-1_r4_n32.npz"
PI_ARTIFACT_BUFFER="$REPO/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
PI_RAW="$REPO/runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"
PI_PLAN="$REPO/runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
PI_CONTROL="$ROOT/closed_loop/pi05/control"
PI_SERVER="$REPO/scripts/run_pi05_formal_server.sh"
PI_EVAL="$REPO/scripts/run_robocasa365_pi05_eval.py"
GPU_A=1
GPU_B=2
GPU_C=3
GPU_D=4
GPU_E=5
GPU_F=6
GPU_G=7
ACTIVE_GPUS=("$GPU_A" "$GPU_B" "$GPU_C" "$GPU_D" "$GPU_E" "$GPU_F" "$GPU_G")
CALIBRATION_GPUS=(1 2 4 4 5 6 7)
ALLOW_SHARED="${DPAC_ALLOW_SHARED:-0}"
MIN_FREE_MIB="${DPAC_MIN_FREE_MIB:-20000}"
SEED_SHARDS_PER_TASK="${DPAC_SEED_SHARDS_PER_TASK:-4}"
QUEUE_CHILDREN=()
PI_INSTANCES=()

usage() {
    echo "usage: $0 start | run | status | stop" >&2
}

phase() {
    mkdir -p "$ROOT"
    printf '%s\n' "$1" >"$PHASE_FILE"
    echo "[dpac-softfold] phase=$1"
}

live_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

gpu_free() {
    local gpu="$1" output
    output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${output//[[:space:]]/}" ]]
}

gpu_free_mib() {
    nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits \
        2>/dev/null | tr -d '[:space:]'
}

gpu_has_headroom() {
    local gpu="$1" threshold="${2:-$MIN_FREE_MIB}" available
    if [[ "$ALLOW_SHARED" != "1" ]]; then
        gpu_free "$gpu"
        return
    fi
    available="$(gpu_free_mib "$gpu")"
    [[ "$available" =~ ^[0-9]+$ ]] && (( available >= threshold ))
}

wait_for_available_gpus() {
    local threshold="${1:-$MIN_FREE_MIB}" waiting=0 gpu ready
    while true; do
        ready=1
        for gpu in "${ACTIVE_GPUS[@]}"; do
            gpu_has_headroom "$gpu" "$threshold" || ready=0
        done
        [[ "$ready" == 1 ]] && break
        if [[ "$waiting" == 0 ]]; then
            phase "waiting_for_shared_gpu_headroom"
            waiting=1
        fi
        sleep 30
    done
    if [[ "$ALLOW_SHARED" == "1" ]]; then
        echo "[dpac-softfold] GPUs ${ACTIVE_GPUS[*]} each have >=${threshold} MiB free; shared execution enabled"
    else
        echo "[dpac-softfold] GPUs ${ACTIVE_GPUS[*]} are idle"
    fi
}

wait_for_predecessor() {
    local pid="${DPAC_WAIT_PID:-118063}" command=""
    if [[ "$ALLOW_SHARED" == "1" ]]; then
        echo "[dpac-softfold] shared execution authorized; predecessor wait bypassed"
        return
    fi
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "invalid DPAC_WAIT_PID: $pid" >&2
        return 2
    fi
    while kill -0 "$pid" 2>/dev/null; do
        command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$command" != *run_gdsq_week1_queue.sh* ]]; then
            echo "[dpac-softfold] PID $pid no longer identifies the predecessor; continuing"
            break
        fi
        phase "waiting_for_predecessor_pid_${pid}"
        sleep 30
    done
}

calibration_complete() {
    local task_set directory
    for task_set in atomic_seen composite_seen composite_unseen; do
        directory="$ROOT/calibration/gr00t/$task_set"
        [[ -s "$directory/scores_merged.json" ]] || return 1
        [[ -s "$directory/softfold_dfunc.json" ]] || return 1
        [[ -s "$directory/softfold_dpac.json" ]] || return 1
    done
    directory="$ROOT/calibration/pi05"
    [[ -s "$directory/scores_merged.json" ]] || return 1
    [[ -s "$directory/softfold_dfunc.json" ]] || return 1
    [[ -s "$directory/softfold_dpac.json" ]] || return 1
}

wait_pair() {
    local first="$1" second="$2" failed=0
    wait "$first" || failed=1
    wait "$second" || failed=1
    QUEUE_CHILDREN=()
    [[ "$failed" == 0 ]]
}

wait_many() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    QUEUE_CHILDREN=()
    [[ "$failed" == 0 ]]
}

gr00t_grid() {
    local task_set="$1" checkpoint="$2" plan="$3" pack="$4" a8="$5" raw="$6"
    local directory="$ROOT/calibration/gr00t/$task_set"
    if [[ -s "$directory/scores_merged.json" && -s "$directory/softfold_dfunc.json" && -s "$directory/softfold_dpac.json" ]]; then
        echo "[dpac-softfold] reuse complete GR00T grid: $task_set"
        return
    fi
    mkdir -p "$directory/grid_shard0" "$directory/grid_shard1"
    phase "calibration_gr00t_${task_set}"
    CUDA_VISIBLE_DEVICES="$GPU_A" "$GROOT_PY" "$REPO/scripts/tools/gr00t_score_softfold_grid.py" \
        --checkpoint "$checkpoint" --plan "$plan" --pack-dir "$pack" --a8 "$a8" \
        --raw-correction "$raw" --n-obs 32 --batch-size 8 \
        --shard-index 0 --shard-count 2 --grid-dir "$directory/grid_shard0" \
        --out "$directory/scores_shard0.json" >"$directory/shard0.log" 2>&1 &
    local first=$!
    CUDA_VISIBLE_DEVICES="$GPU_B" "$GROOT_PY" "$REPO/scripts/tools/gr00t_score_softfold_grid.py" \
        --checkpoint "$checkpoint" --plan "$plan" --pack-dir "$pack" --a8 "$a8" \
        --raw-correction "$raw" --n-obs 32 --batch-size 8 \
        --shard-index 1 --shard-count 2 --grid-dir "$directory/grid_shard1" \
        --out "$directory/scores_shard1.json" >"$directory/shard1.log" 2>&1 &
    local second=$!
    QUEUE_CHILDREN=("$first" "$second")
    wait_pair "$first" "$second"
    "$GROOT_PY" "$REPO/scripts/tools/merge_softfold_grid_scores.py" \
        --input "$directory/scores_shard0.json" --input "$directory/scores_shard1.json" \
        --out "$directory/scores_merged.json" >"$directory/merge.log" 2>&1
    local checkpoint_sha plan_sha buffer_sha
    checkpoint_sha="$(jq -r '.checkpoint_sha256' "$directory/scores_merged.json")"
    plan_sha="$(jq -r '.plan_sha256' "$directory/scores_merged.json")"
    buffer_sha="$(jq -r '.artifact_calibration_buffer_sha256' "$directory/scores_merged.json")"
    for metric in dfunc dpac; do
        local metric_name="d_func_v1"
        [[ "$metric" == "dpac" ]] && metric_name="d_pac_v1"
        "$GROOT_PY" "$REPO/scripts/tools/fit_softfold_compensation.py" \
            --raw-correction "$raw" --validation-scores "$directory/scores_merged.json" \
            --metric "$metric_name" --teacher-checkpoint-sha256 "$checkpoint_sha" \
            --quant-plan-sha256 "$plan_sha" --buffer-sha256 "$buffer_sha" \
            --out "$directory/softfold_${metric}.json" >"$directory/fit_${metric}.log" 2>&1
    done
}

gr00t_grid4() {
    local task_set="$1" checkpoint="$2" plan="$3" pack="$4" a8="$5" raw="$6"
    local directory="$ROOT/calibration/gr00t/$task_set" shard gpu pid
    local pids=() merge_args=()
    if [[ -s "$directory/scores_merged.json" && -s "$directory/softfold_dfunc.json" && -s "$directory/softfold_dpac.json" ]]; then
        echo "[dpac-softfold] reuse complete GR00T grid: $task_set"
        return
    fi
    phase "calibration_gr00t_${task_set}_four_gpu"
    for shard in 0 1 2 3; do
        gpu="${ACTIVE_GPUS[$shard]}"
        mkdir -p "$directory/grid_shard${shard}"
        CUDA_VISIBLE_DEVICES="$gpu" "$GROOT_PY" "$REPO/scripts/tools/gr00t_score_softfold_grid.py" \
            --checkpoint "$checkpoint" --plan "$plan" --pack-dir "$pack" --a8 "$a8" \
            --raw-correction "$raw" --n-obs 32 --batch-size 8 \
            --shard-index "$shard" --shard-count 4 --grid-dir "$directory/grid_shard${shard}" \
            --out "$directory/scores_shard${shard}.json" >"$directory/shard${shard}.log" 2>&1 &
        pid=$!
        pids+=("$pid")
        merge_args+=(--input "$directory/scores_shard${shard}.json")
    done
    QUEUE_CHILDREN=("${pids[@]}")
    wait_many "${pids[@]}"
    "$GROOT_PY" "$REPO/scripts/tools/merge_softfold_grid_scores.py" \
        "${merge_args[@]}" --out "$directory/scores_merged.json" >"$directory/merge.log" 2>&1
    local checkpoint_sha plan_sha buffer_sha metric metric_name
    checkpoint_sha="$(jq -r '.checkpoint_sha256' "$directory/scores_merged.json")"
    plan_sha="$(jq -r '.plan_sha256' "$directory/scores_merged.json")"
    buffer_sha="$(jq -r '.artifact_calibration_buffer_sha256' "$directory/scores_merged.json")"
    for metric in dfunc dpac; do
        metric_name="d_func_v1"
        [[ "$metric" == "dpac" ]] && metric_name="d_pac_v1"
        "$GROOT_PY" "$REPO/scripts/tools/fit_softfold_compensation.py" \
            --raw-correction "$raw" --validation-scores "$directory/scores_merged.json" \
            --metric "$metric_name" --teacher-checkpoint-sha256 "$checkpoint_sha" \
            --quant-plan-sha256 "$plan_sha" --buffer-sha256 "$buffer_sha" \
            --out "$directory/softfold_${metric}.json" >"$directory/fit_${metric}.log" 2>&1
    done
}

pi05_grid() {
    local directory="$ROOT/calibration/pi05"
    local shard gpu pid pids=() merge_args=()
    if [[ -s "$directory/scores_merged.json" && -s "$directory/softfold_dfunc.json" && -s "$directory/softfold_dpac.json" ]]; then
        echo "[dpac-softfold] reuse complete pi0.5 grid"
        return
    fi
    phase "calibration_pi05_seven_gpu"
    for shard in 0 1 2 3 4 5 6; do
        gpu="${CALIBRATION_GPUS[$shard]}"
        mkdir -p "$directory/grid7_shard${shard}"
        CUDA_VISIBLE_DEVICES="$gpu" "$OPENPI_PY" "$REPO/scripts/tools/pi05_score_configs_against_fp16.py" \
            --buffer "$PI_BUFFER" --artifact-calibration-buffer "$PI_ARTIFACT_BUFFER" --n-obs 32 \
            --selection-metric d_pac --strict-artifacts --softfold-grid --softfold-raw "$PI_RAW" \
            --softfold-grid-shard-index "$shard" --softfold-grid-shard-count 7 \
            --softfold-grid-dir "$directory/grid7_shard${shard}" --out "$directory/scores_7way_shard${shard}.json" \
            >"$directory/shard7_${shard}.log" 2>&1 &
        pid=$!
        pids+=("$pid")
        merge_args+=(--input "$directory/scores_7way_shard${shard}.json")
    done
    QUEUE_CHILDREN=("${pids[@]}")
    wait_many "${pids[@]}"
    "$OPENPI_PY" "$REPO/scripts/tools/merge_softfold_grid_scores.py" \
        "${merge_args[@]}" --out "$directory/scores_merged.json" >"$directory/merge.log" 2>&1
    local checkpoint_sha plan_sha buffer_sha
    checkpoint_sha="$(jq -r '.checkpoint_sha256' "$directory/scores_merged.json")"
    plan_sha="$(sha256sum "$PI_PLAN" | cut -d' ' -f1)"
    buffer_sha="$(jq -r '.artifact_calibration_buffer_sha256' "$directory/scores_merged.json")"
    for metric in dfunc dpac; do
        local metric_name="d_func_v1"
        [[ "$metric" == "dpac" ]] && metric_name="d_pac_v1"
        "$OPENPI_PY" "$REPO/scripts/tools/fit_softfold_compensation.py" \
            --raw-correction "$PI_RAW" --validation-scores "$directory/scores_merged.json" \
            --metric "$metric_name" --teacher-checkpoint-sha256 "$checkpoint_sha" \
            --quant-plan-sha256 "$plan_sha" --buffer-sha256 "$buffer_sha" \
            --out "$directory/softfold_${metric}.json" >"$directory/fit_${metric}.log" 2>&1
    done
}

gr00t_closed_loop_complete() {
    local run_dir="$1" config shard file
    for config in softfold_dfunc softfold_dpac; do
        for shard in $(seq 0 19); do
            file="$run_dir/${config}_s${shard}.jsonl"
            [[ -f "$file" && "$(wc -l <"$file")" == 5 ]] || return 1
        done
    done
}

gr00t_closed_loop() {
    local task_set="$1" tasks="$2" checkpoint="$3" spec="$4"
    local run_dir="$ROOT/closed_loop/gr00t/$task_set"
    if gr00t_closed_loop_complete "$run_dir"; then
        echo "[dpac-softfold] reuse complete GR00T closed-loop matrix: $task_set"
        return
    fi
    phase "closed_loop_gr00t_${task_set}"
    local shared_args=()
    [[ "$ALLOW_SHARED" == "1" ]] && shared_args+=(--allow-shared-gpus)
    local attempt=0
    until "$GROOT_PY" "$REPO/scripts/tools/run_robocasa_atomic_matrix.py" \
            --spec "$spec" --run-dir "$run_dir" --phase formal --task-set "$task_set" \
            --tasks "$tasks" --seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
            --checkpoint "$checkpoint" --n-shards 5 \
            --seed-shards-per-task "$SEED_SHARDS_PER_TASK" \
            --egl-device-pool "1,2,3,4,5,6,7" \
            --trial-timeout 7200 --trial-batch-size 5 --action-noise paired \
            --formal-provenance-v2 "${shared_args[@]}"; do
        attempt=$((attempt + 1))
        echo "[dpac-softfold] matrix retry task_set=$task_set attempt=$attempt in 15s"
        sleep 15
    done
}

pi_runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

start_pi_server() {
    local config="$1" artifact="$2" gpu="$3" port="$4" instance="$5"
    PI05_CONTROL_DIR="$PI_CONTROL" PI05_GDSQ_ATM="$artifact" \
        bash "$PI_SERVER" start "$config" "$gpu" "$port" "$instance"
    PI_INSTANCES+=("$instance")
    "$ROBOCASA_PY" - "$PI_CONTROL/$instance.runtime.json" "$artifact" "$config" <<'PY'
import hashlib, json, pathlib, sys
runtime_path, artifact_path, config = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
runtime = json.load(runtime_path.open(encoding="utf-8"))["openpi_runtime"]
atm = runtime["atm_ohb"]
checks = {
    "config": runtime.get("config_id") == config,
    "wrapped": int(runtime["duquant"]["wrapped_layers"]) == 80,
    "artifact": atm.get("artifact_sha256") == hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    "atm_application": atm.get("atm_application") == "fold_q_weight",
    "ohb_application": atm.get("ohb_application") == "fold_o_weight_perhead",
    "selector_free": not runtime.get("runtime_selector", {}).get("enabled", False),
}
failed = [name for name, value in checks.items() if not value]
if failed:
    raise SystemExit(f"pi0.5 SoftFold runtime attestation failed: {failed}")
PY
}

run_pi_lane() {
    local config="$1" port="$2" egl_gpu="$3" metadata_sha="$4" lane="$5" seed_group="$6"
    local seed_start=$((seed_group * 5))
    local seed_spec="${seed_start}-$((seed_start + 4))"
    local task_set tasks output directory="$ROOT/closed_loop/pi05/results/$config"
    mkdir -p "$directory"
    for task_set in atomic_seen composite_seen composite_unseen; do
        case "$task_set/$lane" in
            atomic_seen/0) tasks="CloseBlenderLid" ;;
            atomic_seen/1) tasks="CloseFridge" ;;
            atomic_seen/2) tasks="CloseToasterOvenDoor" ;;
            atomic_seen/3) tasks="NavigateKitchen" ;;
            atomic_seen/4) tasks="OpenDrawer" ;;
            composite_seen/0) tasks="DeliverStraw" ;;
            composite_seen/1) tasks="KettleBoiling" ;;
            composite_seen/2) tasks="LoadDishwasher" ;;
            composite_seen/3) tasks="PrepareCoffee" ;;
            composite_seen/4) tasks="WashLettuce" ;;
            composite_unseen/0) tasks="WeighIngredients" ;;
            composite_unseen/1) tasks="WaffleReheat" ;;
            composite_unseen/2) tasks="WashFruitColander" ;;
            composite_unseen/3) tasks="MakeIceLemonade" ;;
            composite_unseen/4) tasks="ArrangeBreadBasket" ;;
            *) return 2 ;;
        esac
        output="$directory/${task_set}_lane${lane}_seed${seed_group}.jsonl"
        PYTHONPATH="$PI_CLIENT${PYTHONPATH:+:$PYTHONPATH}" "$ROBOCASA_PY" "$PI_EVAL" \
            --host 127.0.0.1 --port "$port" --config-id "$config" --task-set "$task_set" \
            --tasks "$tasks" --trial-seeds "$seed_spec" --split target --replan-steps 16 \
            --egl-device "$egl_gpu" --expected-server-metadata-sha256 "$metadata_sha" \
            --resume-dir "$directory" --out "$output"
    done
}

stop_pi_servers() {
    local instance
    for instance in "${PI_INSTANCES[@]:-}"; do
        [[ -n "$instance" ]] || continue
        PI05_CONTROL_DIR="$PI_CONTROL" bash "$PI_SERVER" stop "$instance" || true
    done
    PI_INSTANCES=()
}

pi05_closed_loop() {
    local dfunc="$ROOT/calibration/pi05/softfold_dfunc.json"
    local dpac="$ROOT/calibration/pi05/softfold_dpac.json"
    local config_a="gdsq_vla_softfold_dfunc" config_b="gdsq_vla_softfold_dpac"
    local instance_a1="dpac15x20_dfunc_g${GPU_A}" instance_a2="dpac15x20_dfunc_g${GPU_B}"
    local instance_a3="dpac15x20_dfunc_g${GPU_C}"
    local instance_b1="dpac15x20_dpac_g${GPU_D}" instance_b2="dpac15x20_dpac_g${GPU_E}"
    local instance_b3="dpac15x20_dpac_g${GPU_F}" instance_b4="dpac15x20_dpac_g${GPU_G}"
    local port_a1=21501 port_a2=21502 port_a3=21503
    local port_b1=21504 port_b2=21505 port_b3=21506 port_b4=21507
    phase "closed_loop_pi05"
    mkdir -p "$PI_CONTROL"
    PI_INSTANCES=(
        "$instance_a1" "$instance_a2" "$instance_a3"
        "$instance_b1" "$instance_b2" "$instance_b3" "$instance_b4"
    )
    local server_start_pids=()
    start_pi_server "$config_a" "$dfunc" "$GPU_A" "$port_a1" "$instance_a1" & server_start_pids+=("$!")
    start_pi_server "$config_a" "$dfunc" "$GPU_B" "$port_a2" "$instance_a2" & server_start_pids+=("$!")
    start_pi_server "$config_a" "$dfunc" "$GPU_C" "$port_a3" "$instance_a3" & server_start_pids+=("$!")
    start_pi_server "$config_b" "$dpac" "$GPU_D" "$port_b1" "$instance_b1" & server_start_pids+=("$!")
    start_pi_server "$config_b" "$dpac" "$GPU_E" "$port_b2" "$instance_b2" & server_start_pids+=("$!")
    start_pi_server "$config_b" "$dpac" "$GPU_F" "$port_b3" "$instance_b3" & server_start_pids+=("$!")
    start_pi_server "$config_b" "$dpac" "$GPU_G" "$port_b4" "$instance_b4" & server_start_pids+=("$!")
    QUEUE_CHILDREN=("${server_start_pids[@]}")
    wait_many "${server_start_pids[@]}"
    local hash_a1 hash_a2 hash_a3 hash_b1 hash_b2 hash_b3 hash_b4
    local pids=() pid failed=0 lane seed_group slot server_index port hash egl
    hash_a1="$(pi_runtime_hash "$PI_CONTROL/$instance_a1.runtime.json")"
    hash_a2="$(pi_runtime_hash "$PI_CONTROL/$instance_a2.runtime.json")"
    hash_a3="$(pi_runtime_hash "$PI_CONTROL/$instance_a3.runtime.json")"
    hash_b1="$(pi_runtime_hash "$PI_CONTROL/$instance_b1.runtime.json")"
    hash_b2="$(pi_runtime_hash "$PI_CONTROL/$instance_b2.runtime.json")"
    hash_b3="$(pi_runtime_hash "$PI_CONTROL/$instance_b3.runtime.json")"
    hash_b4="$(pi_runtime_hash "$PI_CONTROL/$instance_b4.runtime.json")"
    local ports_a=("$port_a1" "$port_a2" "$port_a3") hashes_a=("$hash_a1" "$hash_a2" "$hash_a3")
    local ports_b=("$port_b1" "$port_b2" "$port_b3" "$port_b4") hashes_b=("$hash_b1" "$hash_b2" "$hash_b3" "$hash_b4")
    for lane in 0 1 2 3 4; do
        for seed_group in 0 1 2 3; do
            slot=$((lane * 4 + seed_group))
            server_index=$((slot % 3)); port="${ports_a[$server_index]}"; hash="${hashes_a[$server_index]}"
            egl="${ACTIVE_GPUS[$((slot % 7))]}"
            run_pi_lane "$config_a" "$port" "$egl" "$hash" "$lane" "$seed_group" \
                >"$ROOT/closed_loop/pi05/${config_a}_lane${lane}_seed${seed_group}.log" 2>&1 &
            pids+=("$!")
            server_index=$((slot % 4)); port="${ports_b[$server_index]}"; hash="${hashes_b[$server_index]}"
            egl="${ACTIVE_GPUS[$(((slot + 3) % 7))]}"
            run_pi_lane "$config_b" "$port" "$egl" "$hash" "$lane" "$seed_group" \
                >"$ROOT/closed_loop/pi05/${config_b}_lane${lane}_seed${seed_group}.log" 2>&1 &
            pids+=("$!")
        done
    done
    QUEUE_CHILDREN=("${pids[@]}")
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    QUEUE_CHILDREN=()
    stop_pi_servers
    [[ "$failed" == 0 ]]
}

cleanup() {
    local status=$? child current=""
    for child in "${QUEUE_CHILDREN[@]:-}"; do
        [[ -n "$child" ]] && kill "$child" 2>/dev/null || true
    done
    stop_pi_servers
    [[ -f "$PID_FILE" ]] && current="$(<"$PID_FILE")"
    [[ "$current" == "$$" ]] && rm -f "$PID_FILE"
    if [[ "$status" != 0 ]]; then
        phase "failed_exit_${status}"
    fi
    exit "$status"
}

run_all() {
    trap cleanup EXIT INT TERM
    cd "$REPO"
    mkdir -p "$ROOT"
    printf '%s\n' "$$" >"$PID_FILE"
    if [[ -f "$ROOT/preregistered_manifest.json" ]]; then
        cmp -s "$MANIFEST" "$ROOT/preregistered_manifest.json" || {
            echo "frozen preregistration drift: $ROOT/preregistered_manifest.json" >&2
            return 1
        }
    else
        cp "$MANIFEST" "$ROOT/preregistered_manifest.json"
    fi
    wait_for_predecessor
    if calibration_complete; then
        echo "[dpac-softfold] all calibration artifacts complete; skip redundant startup GPU gate"
    else
        wait_for_available_gpus
    fi

    local pack_root="$REPO/checkpoints/packs/robocasa365"
    local checkpoint_root="$REPO/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining"
    gr00t_grid atomic_seen \
        "$checkpoint_root/atomic_seen/checkpoint-60000" \
        "$pack_root/gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json" \
        "$pack_root/duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015" \
        "$pack_root/a8_scales_cscka_16to1_protocolfix_d4.npz" \
        "$pack_root/atm_alpha_beta_static_cscka_16to1_protocolfix_d4.json"
    gr00t_grid composite_seen \
        "$checkpoint_root/composite_seen/checkpoint-60000" \
        "$pack_root/gr00t_quant_plan_robocasa365_cscka_16to1_composite_seen.final_plan.json" \
        "$pack_root/duquant_packed_robocasa365_composite_seen_d4_w4a8_b64c32ls015" \
        "$pack_root/a8_scales_cscka_16to1_composite_seen_d4.npz" \
        "$pack_root/atm_alpha_beta_static_cscka_16to1_composite_seen_d4.json"
    gr00t_grid4 composite_unseen \
        "$checkpoint_root/composite_unseen/checkpoint-60000" \
        "$pack_root/gr00t_quant_plan_robocasa365_cscka_16to1_composite_unseen.final_plan.json" \
        "$pack_root/duquant_packed_robocasa365_composite_unseen_d4_w4a8_b64c32ls015" \
        "$pack_root/a8_scales_cscka_16to1_composite_unseen_d4.npz" \
        "$pack_root/atm_alpha_beta_static_cscka_16to1_composite_unseen_d4.json"
    pi05_grid

    gr00t_closed_loop atomic_seen \
        "CloseBlenderLid,CloseFridge,CloseToasterOvenDoor,NavigateKitchen,OpenDrawer" \
        "$checkpoint_root/atomic_seen/checkpoint-60000" \
        "$REPO/scripts/dpac_softfold_gr00t_atomic_spec.json"
    gr00t_closed_loop composite_seen \
        "DeliverStraw,KettleBoiling,LoadDishwasher,PrepareCoffee,WashLettuce" \
        "$checkpoint_root/composite_seen/checkpoint-60000" \
        "$REPO/scripts/dpac_softfold_gr00t_composite_seen_spec.json"
    gr00t_closed_loop composite_unseen \
        "ArrangeBreadBasket,MakeIceLemonade,WaffleReheat,WashFruitColander,WeighIngredients" \
        "$checkpoint_root/composite_unseen/checkpoint-60000" \
        "$REPO/scripts/dpac_softfold_gr00t_composite_unseen_spec.json"
    wait_for_available_gpus
    pi05_closed_loop
    phase "aggregate"
    "$GROOT_PY" "$REPO/scripts/tools/aggregate_dpac_softfold_15x20.py" --root "$ROOT" \
        >"$ROOT/aggregate.log" 2>&1
    phase "complete"
}

start_detached() {
    mkdir -p "$ROOT"
    if live_pid; then
        echo "already running pid=$(<"$PID_FILE") phase=$(<"$PHASE_FILE" 2>/dev/null || true)"
        return
    fi
    nohup setsid bash "$0" run >"$LOG" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "queue failed during startup" >&2
        tail -n 120 "$LOG" >&2
        return 1
    fi
    echo "started pid=$pid log=$LOG"
}

status_queue() {
    local state="stopped" pid="" phase_value="not_started"
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ -f "$PHASE_FILE" ]] && phase_value="$(<"$PHASE_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state="running"; fi
    echo "state=$state pid=${pid:-none} phase=$phase_value"
    for gpu in "${ACTIVE_GPUS[@]}"; do
        echo "gpu=$gpu free_mib=$(gpu_free_mib "$gpu") shared=$ALLOW_SHARED min_free_mib=$MIN_FREE_MIB"
        nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null || true
    done
    [[ -f "$LOG" ]] && tail -n 20 "$LOG"
    return 0
}

stop_queue() {
    if ! live_pid; then
        echo "queue is not running"
        return
    fi
    local pid command group
    pid="$(<"$PID_FILE")"
    command="$(ps -p "$pid" -o args=)"
    if [[ "$command" != *run_dpac_softfold_15x20.sh*run* ]]; then
        echo "refusing to stop unrelated pid=$pid command=$command" >&2
        return 1
    fi
    group="$(ps -p "$pid" -o pgid= | tr -d ' ')"
    if [[ "$group" == "$pid" ]]; then kill -- "-$pid"; else kill "$pid"; fi
    echo "stop requested pid=$pid"
}

case "${1:-}" in
    start) start_detached ;;
    run) run_all ;;
    status) status_queue ;;
    stop) stop_queue ;;
    *) usage; exit 2 ;;
esac
