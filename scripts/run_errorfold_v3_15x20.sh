#!/usr/bin/env bash
set -euo pipefail

# End-to-end, crash-resumable runner for the frozen ErrorFold-v3 protocol.
# The outer workflow is identical for GR00T N1.5 and pi0.5; model-specific
# checkpoint loading and observation/action conversion live only in adapters.

REPO="/home1/gyy/vla/QuantVLA"
ROOT="${ERRORFOLD_V3_ROOT:-$REPO/runs/errorfold_v3_15x20}"
MANIFEST="$REPO/scripts/quantvla_cross_model_protocol.json"
GROOT_PY="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
PI_CLIENT="$REPO/code/pi05/openpi/packages/openpi-client/src"
PI_SERVER="$REPO/scripts/run_pi05_formal_server.sh"
PI_EVAL="$REPO/scripts/run_robocasa365_pi05_eval.py"
PI_CONTROL="$ROOT/closed_loop/pi05/control"
LOG="$ROOT/orchestrator.log"
PID_FILE="$ROOT/orchestrator.pid"
PHASE_FILE="$ROOT/phase.txt"

CALIBRATION_BUFFER="$REPO/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
SELECTION_BUFFER="$REPO/runs/pi05_gdsq_gr00t_aligned/diagnostics/fp16_onpolicy_probe/fp16_onpolicy_target_4tasks_s0-1_r4_n32.npz"
PACK_ROOT="$REPO/checkpoints/packs/robocasa365"
CHECKPOINT_ROOT="$REPO/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining"
PI_CHECKPOINT="$REPO/checkpoints/robocasa/pi05_pretrain_human300_pytorch"
PI_PLAN="$REPO/runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"

# Shared GPUs are allowed.  The only launch gate is enough currently-free
# memory for one model process; no exclusive-idle or predecessor gate exists.
MIN_FREE_MIB="${ERRORFOLD_V3_MIN_FREE_MIB:-18000}"
GPU_POLL_SECONDS="${ERRORFOLD_V3_GPU_POLL_SECONDS:-20}"
SEED_SHARDS_PER_TASK="${ERRORFOLD_V3_SEED_SHARDS_PER_TASK:-2}"
CALIBRATION_GPUS_TEXT="${ERRORFOLD_V3_CALIBRATION_GPUS:-1,4,6,7}"
IFS=',' read -r -a CALIBRATION_GPUS <<<"$CALIBRATION_GPUS_TEXT"
ALL_GPUS=(0 1 2 3 4 5 6 7)
CONFIGS=(fp16 quantvla_w4a8_paper errorfold_dfunc errorfold_dpac_v2)
TASK_SETS=(atomic_seen composite_seen composite_unseen)
QUEUE_CHILDREN=()
PI_INSTANCES=()

declare -A GR_PLAN=(
    [atomic_seen]="$PACK_ROOT/quantvla_v1_uniform_w4a8.json"
    [composite_seen]="$PACK_ROOT/quantvla_v1_uniform_w4a8_composite_seen.json"
    [composite_unseen]="$PACK_ROOT/quantvla_v1_uniform_w4a8_composite_unseen.json"
)
declare -A GR_CHECKPOINT=(
    [atomic_seen]="$CHECKPOINT_ROOT/atomic_seen/checkpoint-60000"
    [composite_seen]="$CHECKPOINT_ROOT/composite_seen/checkpoint-60000"
    [composite_unseen]="$CHECKPOINT_ROOT/composite_unseen/checkpoint-60000"
)
declare -A TASKS=(
    [atomic_seen]="CloseBlenderLid,CloseFridge,CloseToasterOvenDoor,NavigateKitchen,OpenDrawer"
    [composite_seen]="DeliverStraw,KettleBoiling,LoadDishwasher,PrepareCoffee,WashLettuce"
    [composite_unseen]="ArrangeBreadBasket,MakeIceLemonade,WaffleReheat,WashFruitColander,WeighIngredients"
)

usage() {
    echo "usage: $0 start | run | status | stop | audit" >&2
}

phase() {
    mkdir -p "$ROOT"
    printf '%s\n' "$1" >"$PHASE_FILE"
    echo "[errorfold-v3] phase=$1"
}

gpu_free_mib() {
    nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits \
        2>/dev/null | tr -d '[:space:]'
}

wait_gpu_headroom() {
    local gpu="$1" threshold="${2:-$MIN_FREE_MIB}" free=""
    while true; do
        free="$(gpu_free_mib "$gpu")"
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= threshold )); then
            echo "[errorfold-v3] gpu=$gpu free_mib=$free launch_threshold=$threshold shared=1"
            return
        fi
        echo "[errorfold-v3] waiting gpu=$gpu free_mib=${free:-unknown} threshold=$threshold" >&2
        sleep "$GPU_POLL_SECONDS"
    done
}

wait_many() {
    local failed=0 pid
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    QUEUE_CHILDREN=()
    [[ "$failed" == 0 ]]
}

freeze_manifest() {
    mkdir -p "$ROOT"
    if [[ -f "$ROOT/preregistered_manifest.json" ]]; then
        cmp -s "$MANIFEST" "$ROOT/preregistered_manifest.json" || {
            echo "frozen v3 manifest drift: $ROOT/preregistered_manifest.json" >&2
            return 1
        }
    else
        cp "$MANIFEST" "$ROOT/preregistered_manifest.json"
    fi
}

calibration_complete() {
    local directory="$1" sidecar="$2"
    [[ -s "$directory/calibration_manifest.json" ]] &&
        [[ -s "$directory/fp16_capture.npz" ]] &&
        [[ -s "$directory/fp16_attention.npz" ]] &&
        [[ -s "$directory/a8_scales.npz" ]] &&
        [[ -s "$directory/a8_scales.npz$sidecar" ]] &&
        [[ -s "$directory/hessian_w4.npz" ]] &&
        [[ -s "$directory/hessian_w4.npz.json" ]] &&
        [[ -s "$directory/quant_capture.npz" ]] &&
        [[ -s "$directory/quant_attention.npz" ]] &&
        [[ -s "$directory/paired_errorfold_capture.npz" ]] &&
        [[ -s "$directory/raw_errorfold.json" ]] &&
        [[ -s "$directory/identity_pack/manifest.json" ]]
}

run_calibration_job() {
    local model="$1" label="$2" checkpoint="$3" plan="$4" gpu="$5"
    local directory="$ROOT/calibration/$model"
    local python="$OPENPI_PY" sidecar=".json"
    [[ "$model" == "gr00t" ]] && directory="$directory/$label" && python="$GROOT_PY" && sidecar=".meta.json"
    if calibration_complete "$directory" "$sidecar"; then
        echo "[errorfold-v3] reuse calibration model=$model label=$label"
        return
    fi
    mkdir -p "$directory"
    local force=()
    if find "$directory" -maxdepth 1 -type f -print -quit | grep -q .; then
        force=(--force)
    fi
    wait_gpu_headroom "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" \
        "$REPO/scripts/tools/calibrate_errorfold_v3.py" \
        --model "$model" --checkpoint "$checkpoint" --plan "$plan" \
        --buffer "$CALIBRATION_BUFFER" --out-dir "$directory" \
        --pack-dir "$directory/identity_pack" --batch-size 8 \
        --device cuda --hessian-device cuda "${force[@]}" \
        >"$directory/calibration.log" 2>&1
}

calibrate_all() {
    phase calibration_four_adapter_bindings
    local pids=()
    run_calibration_job gr00t atomic_seen "${GR_CHECKPOINT[atomic_seen]}" \
        "${GR_PLAN[atomic_seen]}" "${CALIBRATION_GPUS[0]}" & pids+=("$!")
    run_calibration_job gr00t composite_seen "${GR_CHECKPOINT[composite_seen]}" \
        "${GR_PLAN[composite_seen]}" "${CALIBRATION_GPUS[1]}" & pids+=("$!")
    run_calibration_job gr00t composite_unseen "${GR_CHECKPOINT[composite_unseen]}" \
        "${GR_PLAN[composite_unseen]}" "${CALIBRATION_GPUS[2]}" & pids+=("$!")
    run_calibration_job pi05 shared "$PI_CHECKPOINT" "$PI_PLAN" \
        "${CALIBRATION_GPUS[3]}" & pids+=("$!")
    QUEUE_CHILDREN=("${pids[@]}")
    wait_many "${pids[@]}"
}

run_grid_job() {
    local model="$1" label="$2" shard="$3" gpu="$4"
    local directory="$ROOT/calibration/$model"
    local python="$OPENPI_PY"
    [[ "$model" == "gr00t" ]] && directory="$directory/$label" && python="$GROOT_PY"
    mkdir -p "$directory/grid_shard${shard}"
    wait_gpu_headroom "$gpu"
    if [[ "$model" == "gr00t" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 GR00T_DUQUANT_FUSED=1 \
            "$python" "$REPO/scripts/tools/gr00t_score_softfold_grid.py" \
            --checkpoint "${GR_CHECKPOINT[$label]}" --plan "${GR_PLAN[$label]}" \
            --pack-dir "$directory/identity_pack" --a8 "$directory/a8_scales.npz" \
            --hessian-w4 "$directory/hessian_w4.npz" \
            --raw-correction "$directory/raw_errorfold.json" \
            --buffer "$SELECTION_BUFFER" \
            --artifact-calibration-buffer "$CALIBRATION_BUFFER" \
            --n-obs 32 --batch-size 8 --denoising-steps 4 \
            --shard-index "$shard" --shard-count 2 \
            --grid-dir "$directory/grid_shard${shard}" \
            --out "$directory/scores_shard${shard}.json" \
            >"$directory/grid_shard${shard}.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
            "$python" "$REPO/scripts/tools/pi05_score_configs_against_fp16.py" \
            --checkpoint-dir "$PI_CHECKPOINT" --pack-dir "$directory/identity_pack" \
            --buffer "$SELECTION_BUFFER" \
            --artifact-calibration-buffer "$CALIBRATION_BUFFER" \
            --n-obs 32 --selection-metric d_pac --strict-artifacts \
            --softfold-grid --softfold-raw "$directory/raw_errorfold.json" \
            --hessian-w4 "$directory/hessian_w4.npz" \
            --v3-plan "$PI_PLAN" --v3-a8 "$directory/a8_scales.npz" \
            --softfold-grid-shard-index "$shard" \
            --softfold-grid-shard-count 2 \
            --softfold-grid-dir "$directory/grid_shard${shard}" \
            --out "$directory/scores_shard${shard}.json" \
            >"$directory/grid_shard${shard}.log" 2>&1
    fi
}

score_all_grids() {
    phase softfold_complete_9x9_four_bindings_all_gpus
    local pids=() gpu_index=0 model label shard
    for label in "${TASK_SETS[@]}"; do
        for shard in 0 1; do
            run_grid_job gr00t "$label" "$shard" "${ALL_GPUS[$gpu_index]}" &
            pids+=("$!")
            gpu_index=$((gpu_index + 1))
        done
    done
    for shard in 0 1; do
        run_grid_job pi05 shared "$shard" "${ALL_GPUS[$gpu_index]}" &
        pids+=("$!")
        gpu_index=$((gpu_index + 1))
    done
    QUEUE_CHILDREN=("${pids[@]}")
    wait_many "${pids[@]}"
}

merge_and_fit() {
    local model="$1" label="$2"
    local directory="$ROOT/calibration/$model" python="$OPENPI_PY"
    [[ "$model" == "gr00t" ]] && directory="$directory/$label" && python="$GROOT_PY"
    "$python" "$REPO/scripts/tools/merge_softfold_grid_scores.py" \
        --input "$directory/scores_shard0.json" \
        --input "$directory/scores_shard1.json" \
        --out "$directory/scores_merged.json" >"$directory/grid_merge.log" 2>&1
    local checkpoint_sha plan_sha buffer_sha
    checkpoint_sha="$(jq -er '.checkpoint_sha256' "$directory/scores_merged.json")"
    plan_sha="$(jq -er '.plan_sha256' "$directory/scores_merged.json")"
    buffer_sha="$(jq -er '.artifact_calibration_buffer_sha256' "$directory/scores_merged.json")"
    "$python" "$REPO/scripts/tools/fit_softfold_compensation.py" \
        --raw-correction "$directory/raw_errorfold.json" \
        --validation-scores "$directory/scores_merged.json" --metric d_func_v1 \
        --teacher-checkpoint-sha256 "$checkpoint_sha" \
        --quant-plan-sha256 "$plan_sha" --buffer-sha256 "$buffer_sha" \
        --out "$directory/errorfold_dfunc.json" >"$directory/fit_dfunc.log" 2>&1
    "$python" "$REPO/scripts/tools/fit_softfold_compensation.py" \
        --raw-correction "$directory/raw_errorfold.json" \
        --validation-scores "$directory/scores_merged.json" --metric d_pac_v2 \
        --teacher-checkpoint-sha256 "$checkpoint_sha" \
        --quant-plan-sha256 "$plan_sha" --buffer-sha256 "$buffer_sha" \
        --out "$directory/errorfold_dpac_v2.json" >"$directory/fit_dpac_v2.log" 2>&1
}

select_all() {
    phase paired_one_se_selection
    local label
    for label in "${TASK_SETS[@]}"; do merge_and_fit gr00t "$label"; done
    merge_and_fit pi05 shared
}

run_noise_b_job() {
    local model="$1" label="$2" gpu="$3"
    local directory="$ROOT/calibration/$model" python="$OPENPI_PY" checkpoint="$PI_CHECKPOINT" plan="$PI_PLAN"
    [[ "$model" == "gr00t" ]] && directory="$directory/$label" && python="$GROOT_PY" && checkpoint="${GR_CHECKPOINT[$label]}" && plan="${GR_PLAN[$label]}"
    if [[ -s "$directory/noise_b_audit.json" ]]; then
        echo "[errorfold-v3] reuse noise-B model=$model label=$label"
        return
    fi
    wait_gpu_headroom "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" \
        "$REPO/scripts/tools/audit_errorfold_v3_noise_b.py" \
        --model "$model" --checkpoint "$checkpoint" --plan "$plan" \
        --pack-dir "$directory/identity_pack" --a8 "$directory/a8_scales.npz" \
        --hessian-w4 "$directory/hessian_w4.npz" \
        --selected "$directory/errorfold_dfunc.json" \
        --selected "$directory/errorfold_dpac_v2.json" \
        --buffer "$SELECTION_BUFFER" --n-obs 32 --batch-size 8 \
        --out "$directory/noise_b_audit.json" >"$directory/noise_b_audit.log" 2>&1
}

audit_noise_b_all() {
    phase heldout_noise_b_frozen_audit
    local pids=()
    run_noise_b_job gr00t atomic_seen "${CALIBRATION_GPUS[0]}" & pids+=("$!")
    run_noise_b_job gr00t composite_seen "${CALIBRATION_GPUS[1]}" & pids+=("$!")
    run_noise_b_job gr00t composite_unseen "${CALIBRATION_GPUS[2]}" & pids+=("$!")
    run_noise_b_job pi05 shared "${CALIBRATION_GPUS[3]}" & pids+=("$!")
    QUEUE_CHILDREN=("${pids[@]}")
    wait_many "${pids[@]}"
}

matrix_complete() {
    local model="$1" directory="$2" task_set="${3:-all}"
    "$GROOT_PY" - "$MANIFEST" "$model" "$directory" "$task_set" <<'PY' >/dev/null
import json, pathlib, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
model, root = sys.argv[2], pathlib.Path(sys.argv[3])
scope = sys.argv[4]
groups = ([manifest["closed_loop"]["tasks"][scope]] if scope != "all"
          else manifest["closed_loop"]["tasks"].values())
expected = {(task, int(seed)) for tasks in groups
            for task in tasks for seed in manifest["closed_loop"]["seeds"]}
for config in [row["id"] for row in manifest["evaluation_matrix"]["configs"]]:
    paths = root.glob(f"{config}_s*.jsonl") if model == "gr00t" else (root / config).glob("*.jsonl")
    actual = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("success") is not None and (row.get("config") or row.get("config_id")) == config:
                    actual.add((str(row["task"]), int(row["seed"])))
    if actual != expected:
        raise SystemExit(1)
PY
}

run_gr00t_matrix() {
    local task_set="$1" spec="$ROOT/specs/gr00t_${task_set}.json"
    local run_dir="$ROOT/closed_loop/gr00t/$task_set"
    if [[ -s "$run_dir/runtime_info.json" ]] && matrix_complete gr00t "$run_dir" "$task_set"; then
        echo "[errorfold-v3] reuse complete GR00T matrix task_set=$task_set"
        return
    fi
    phase "closed_loop_gr00t_${task_set}"
    mkdir -p "$run_dir"
    local attempt=0
    while true; do
        if "$GROOT_PY" "$REPO/scripts/tools/run_robocasa_atomic_matrix.py" \
            --spec "$spec" --run-dir "$run_dir" --phase formal \
            --task-set "$task_set" --tasks "${TASKS[$task_set]}" \
            --seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
            --checkpoint "${GR_CHECKPOINT[$task_set]}" --n-shards 5 \
            --seed-shards-per-task "$SEED_SHARDS_PER_TASK" \
            --egl-device-pool "0,1,2,3,4,5,6,7" \
            --trial-timeout 7200 --trial-batch-size 5 --action-noise paired \
            --formal-provenance-v2 --allow-shared-gpus; then
            return
        fi
        attempt=$((attempt + 1))
        if (( attempt >= 20 )); then
            echo "GR00T matrix failed after $attempt resumable attempts: $task_set" >&2
            return 1
        fi
        echo "[errorfold-v3] retry GR00T task_set=$task_set attempt=$attempt in 20s" >&2
        sleep 20
    done
}

gr00t_closed_loop() {
    phase materialize_gr00t_specs
    "$GROOT_PY" "$REPO/scripts/tools/make_errorfold_v3_gr00t_specs.py" \
        --run-root "$ROOT" --out-dir "$ROOT/specs" >"$ROOT/specs.log" 2>&1
    local task_set
    for task_set in "${TASK_SETS[@]}"; do run_gr00t_matrix "$task_set"; done
}

pi_runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

start_pi_server() {
    local config="$1" gpu="$2" port="$3" instance="$4"
    PI05_CONTROL_DIR="$PI_CONTROL" \
    PI05_V3_ROOT="$ROOT/calibration/pi05" \
    PI05_V3_PACK_DIR="$ROOT/calibration/pi05/identity_pack" \
    PI05_V3_A8="$ROOT/calibration/pi05/a8_scales.npz" \
    PI05_V3_HESSIAN_W4="$ROOT/calibration/pi05/hessian_w4.npz" \
    PI05_V3_ERRORFOLD_DFUNC="$ROOT/calibration/pi05/errorfold_dfunc.json" \
    PI05_V3_ERRORFOLD_DPAC="$ROOT/calibration/pi05/errorfold_dpac_v2.json" \
        bash "$PI_SERVER" start "$config" "$gpu" "$port" "$instance"
}

stop_pi_servers() {
    local instance
    for instance in "${PI_INSTANCES[@]:-}"; do
        [[ -n "$instance" ]] || continue
        PI05_CONTROL_DIR="$PI_CONTROL" bash "$PI_SERVER" stop "$instance" || true
    done
    PI_INSTANCES=()
}

run_pi_lane() {
    local config="$1" port="$2" egl="$3" metadata_sha="$4" lane="$5"
    local task_set task output directory="$ROOT/closed_loop/pi05/results/$config"
    mkdir -p "$directory"
    for task_set in "${TASK_SETS[@]}"; do
        IFS=',' read -r -a task_array <<<"${TASKS[$task_set]}"
        task="${task_array[$lane]}"
        output="$directory/${task_set}_lane${lane}.jsonl"
        PYTHONPATH="$PI_CLIENT${PYTHONPATH:+:$PYTHONPATH}" "$ROBOCASA_PY" "$PI_EVAL" \
            --host 127.0.0.1 --port "$port" --config-id "$config" \
            --task-set "$task_set" --tasks "$task" --trial-seeds 0-19 \
            --split target --replan-steps 16 --action-noise-mode paired \
            --egl-device "$egl" --expected-server-metadata-sha256 "$metadata_sha" \
            --resume-dir "$directory" --out "$output"
    done
}

write_pi_runtime_index() {
    "$GROOT_PY" - "$PI_CONTROL" "$ROOT/closed_loop/pi05/runtime_info.json" "${PI_INSTANCES[@]}" <<'PY'
import json, pathlib, sys
control, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
instances = sys.argv[3:]
result = {}
for instance in instances:
    config, replica = instance.rsplit("_r", 1)
    key = config if replica == "0" else f"{config}/r{replica}"
    result[key] = json.loads((control / f"{instance}.runtime.json").read_text(encoding="utf-8"))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

pi05_closed_loop() {
    local result_root="$ROOT/closed_loop/pi05/results"
    if [[ -s "$ROOT/closed_loop/pi05/runtime_info.json" ]] && matrix_complete pi05 "$result_root"; then
        echo "[errorfold-v3] reuse complete pi0.5 matrix"
        return
    fi
    phase closed_loop_pi05_four_config_all_gpus
    mkdir -p "$PI_CONTROL" "$result_root"
    local ports=(22200 22201 22202 22203 22204 22205 22206 22207)
    local pids=() index=0 config replica instance
    PI_INSTANCES=()
    for config in "${CONFIGS[@]}"; do
        for replica in 0 1; do
            instance="${config}_r${replica}"
            PI_INSTANCES+=("$instance")
            wait_gpu_headroom "${ALL_GPUS[$index]}"
            start_pi_server "$config" "${ALL_GPUS[$index]}" "${ports[$index]}" "$instance" &
            pids+=("$!")
            index=$((index + 1))
        done
    done
    QUEUE_CHILDREN=("${pids[@]}")
    wait_many "${pids[@]}"
    write_pi_runtime_index

    local lane server_index port metadata_sha egl failed=0
    pids=()
    for index in 0 1 2 3; do
        config="${CONFIGS[$index]}"
        for lane in 0 1 2 3 4; do
            replica=$((lane % 2))
            server_index=$((index * 2 + replica))
            port="${ports[$server_index]}"
            instance="${config}_r${replica}"
            metadata_sha="$(pi_runtime_hash "$PI_CONTROL/$instance.runtime.json")"
            egl="${ALL_GPUS[$(((index * 5 + lane) % 8))]}"
            run_pi_lane "$config" "$port" "$egl" "$metadata_sha" "$lane" \
                >"$ROOT/closed_loop/pi05/${config}_lane${lane}.log" 2>&1 &
            pids+=("$!")
        done
    done
    QUEUE_CHILDREN=("${pids[@]}")
    for index in "${pids[@]}"; do wait "$index" || failed=1; done
    QUEUE_CHILDREN=()
    stop_pi_servers
    [[ "$failed" == 0 ]]
}

final_reports() {
    phase deployment_statistics_and_final_audit
    "$GROOT_PY" "$REPO/scripts/tools/collect_errorfold_v3_deployment.py" \
        --run-root "$ROOT" >"$ROOT/deployment.log" 2>&1
    "$GROOT_PY" "$REPO/scripts/tools/aggregate_dpac_softfold_15x20.py" \
        --root "$ROOT" >"$ROOT/aggregate.log" 2>&1
    "$GROOT_PY" "$REPO/scripts/tools/audit_errorfold_v3.py" \
        --run-root "$ROOT" --out "$ROOT/final_audit.json" --strict \
        >"$ROOT/final_audit.log" 2>&1
}

cleanup() {
    local status=$? child current=""
    for child in "${QUEUE_CHILDREN[@]:-}"; do
        [[ -n "$child" ]] && kill "$child" 2>/dev/null || true
    done
    stop_pi_servers
    [[ -f "$PID_FILE" ]] && current="$(<"$PID_FILE")"
    [[ "$current" == "$$" ]] && rm -f "$PID_FILE"
    if [[ "$status" != 0 ]]; then phase "failed_exit_${status}"; fi
    exit "$status"
}

run_all() {
    trap cleanup EXIT INT TERM
    cd "$REPO"
    mkdir -p "$ROOT"
    printf '%s\n' "$$" >"$PID_FILE"
    freeze_manifest
    calibrate_all
    score_all_grids
    select_all
    audit_noise_b_all
    gr00t_closed_loop
    pi05_closed_loop
    final_reports
    phase complete
}

live_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
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
        echo "ErrorFold-v3 runner failed during startup" >&2
        tail -n 120 "$LOG" >&2
        return 1
    fi
    echo "started pid=$pid log=$LOG"
}

status_queue() {
    local state="stopped" pid="" phase_value="not_started" gpu
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ -f "$PHASE_FILE" ]] && phase_value="$(<"$PHASE_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state="running"; fi
    echo "state=$state pid=${pid:-none} phase=$phase_value root=$ROOT"
    for gpu in "${ALL_GPUS[@]}"; do
        echo "gpu=$gpu free_mib=$(gpu_free_mib "$gpu") shared=1"
    done
    [[ -f "$LOG" ]] && tail -n 30 "$LOG"
}

stop_queue() {
    if ! live_pid; then echo "runner is not active"; return; fi
    local pid command group
    pid="$(<"$PID_FILE")"
    command="$(ps -p "$pid" -o args=)"
    if [[ "$command" != *run_errorfold_v3_15x20.sh*run* ]]; then
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
    audit) "$GROOT_PY" "$REPO/scripts/tools/audit_errorfold_v3.py" --run-root "$ROOT" ;;
    *) usage; exit 2 ;;
esac
