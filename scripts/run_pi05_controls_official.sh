#!/usr/bin/env bash
set -euo pipefail

# Result-blind pi0.5 random-mask/action-only chain:
# functional screen -> dev4 selection -> heldout46 formal evaluation.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
WEEK1_ROOT="$REPO_ROOT/runs/gdsq_week1_preregistered_v1"
EXECUTION="$WEEK1_ROOT/execution"
SCREEN="$REPO_ROOT/scripts/tools/gdsq_week1_pi05_screen.py"
SELECTION="$REPO_ROOT/scripts/tools/gdsq_week1_pi05_selection.py"
SERVER="$REPO_ROOT/scripts/run_pi05_week1_server.sh"
WORKER="$REPO_ROOT/scripts/run_pi05_week1_worker.sh"
EVALUATOR="$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py"
PREFLIGHT_AUDITOR="$REPO_ROOT/scripts/tools/audit_pi05_week1_preflight.py"
MANIFEST_TOOL="$REPO_ROOT/scripts/tools/pi05_make_week1_manifest.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_pi05_week1.py"
GPU_MONITOR="$REPO_ROOT/scripts/tools/monitor_pi05_formal_gpu.py"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SELECTOR_MANIFEST="$WEEK1_ROOT/pi05_selector_official50/manifest.json"
STATIC_MANIFEST="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/official_target_paired50/manifest.json"

usage() {
    echo "usage: $0 screen | run-dev | select | run-formal | run-all | status" >&2
}

gpu_free() {
    local gpu="$1" output
    output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${output//[[:space:]]/}" ]]
}

require_gpus_free() {
    local gpu
    for gpu in {0..7}; do
        if ! gpu_free "$gpu"; then
            echo "refusing to start pi0.5 controls: GPU $gpu is occupied" >&2
            return 1
        fi
    done
}

runtime_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["server_metadata_sha256"])
PY
}

run_preflight() {
    local run_dir="$1" control="$2" config="$3" instance="$4" port="$5" gpu="$6" tasks="$7"
    local directory="$run_dir/preflight" output="$run_dir/preflight/two_task_two_seed.jsonl"
    local summary="$run_dir/preflight/summary.json" audit="$control/$instance.audit.json" hash
    mkdir -p "$directory"
    hash="$(runtime_hash "$audit")"
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$EVALUATOR" \
        --host 127.0.0.1 \
        --port "$port" \
        --config-id "$config" \
        --task-set atomic_seen \
        --tasks "$tasks" \
        --trial-seeds 0-1 \
        --split target \
        --replan-steps 16 \
        --egl-device "$gpu" \
        --expected-server-metadata-sha256 "$hash" \
        --resume-dir "$directory" \
        --out "$output"
    "$ROBOCASA_PY" "$PREFLIGHT_AUDITOR" \
        --results "$output" --runtime-audit "$audit" --config "$config" \
        --tasks "$tasks" --out "$summary" >/dev/null
}

aggregate_run() {
    local run_dir="$1"
    PYTHONPATH="$REPO_ROOT/scripts/tools:$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$AGGREGATOR" \
        --run-dir "$run_dir" --out-dir "$run_dir/aggregate" --bootstrap 10000
}

run_dev_config() (
    set -euo pipefail
    local config="$1" plan="$2" a8="$3" wrapped="$4" gpu="$5" port="$6"
    local run_dir="$EXECUTION/runs/pi05_controls_dev4/$config"
    local control="$run_dir/control" instance="pi05_dev_${config}_g${gpu}"
    local audit="$control/$instance.audit.json" runtime="$control/$instance.runtime.json"
    export PI05_WEEK1_CONTROL_DIR="$control"
    mkdir -p "$control/workers"
    trap '"$SERVER" stop "$instance" >/dev/null 2>&1 || true' EXIT
    "$SERVER" start "$config" "$gpu" "$port" "$instance" "$plan" "$a8" "$wrapped"
    run_preflight "$run_dir" "$control" "$config" "$instance" "$port" "$gpu" \
        "CoffeeSetupMug,OpenCabinet"
    local manifest="$run_dir/manifest.json" mode=create hash
    [[ -f "$manifest" ]] && mode=verify
    args=(
        --run-dir "$run_dir" --out "$manifest" --config "$config" --scope dev4
        --plan "$plan" --a8 "$a8" --expected-wrapped "$wrapped"
        --orchestrator "$REPO_ROOT/scripts/run_pi05_controls_official.sh"
        --preflight-results "$run_dir/preflight/two_task_two_seed.jsonl"
        --preflight-audit "$run_dir/preflight/summary.json"
        --server "$instance,$config,$gpu,$port,$runtime,$audit"
        --worker "${config}_dev_s0_lo,$instance,0,2,0-24"
        --worker "${config}_dev_s0_hi,$instance,0,2,25-49"
        --worker "${config}_dev_s1_lo,$instance,1,2,0-24"
        --worker "${config}_dev_s1_hi,$instance,1,2,25-49"
    )
    [[ "$mode" == verify ]] && args+=(--verify)
    "$ROBOCASA_PY" "$MANIFEST_TOOL" "${args[@]}"
    hash="$(runtime_hash "$audit")"
    local failed pid
    while true; do
        failed=0
        pids=()
        for worker_spec in "s0_lo,0,0-24" "s0_hi,0,25-49" "s1_lo,1,0-24" "s1_hi,1,25-49"; do
            IFS=',' read -r suffix shard seeds <<<"$worker_spec"
            "$WORKER" "$config" "$port" "$gpu" "$shard" 2 "$seeds" \
                "${config}_dev_${suffix}" "$hash" "$run_dir" dev4 \
                >"$control/workers/${config}_dev_${suffix}.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" == 0 ]] && break
        echo "[pi05 dev] retry config=$config" >&2
        sleep 30
    done
    aggregate_run "$run_dir"
    "$ROBOCASA_PY" - "$run_dir/aggregate/summary.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); assert m["complete"] and m["completed_episodes"] == 200
PY
)

screen_controls() {
    require_gpus_free
    mkdir -p "$EXECUTION/screen"
    CUDA_VISIBLE_DEVICES=0 "$OPENPI_PY" "$SCREEN" \
        --family random --out "$EXECUTION/screen/pi05_random.json"
    CUDA_VISIBLE_DEVICES=0 "$OPENPI_PY" "$SCREEN" \
        --family action_only --out "$EXECUTION/screen/pi05_action_only.json"
    "$ROBOCASA_PY" "$SELECTION" materialize-dev
}

wave_rows() {
    local wave="$1"
    "$ROBOCASA_PY" - "$EXECUTION/selection/pi05_control_dev_schedule.json" "$wave" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); wanted=int(sys.argv[2])
for wave in m["waves"]:
    if int(wave["wave"]) != wanted: continue
    for row in wave["configs"]:
        print("\t".join(map(str, [row["config_id"], row["plan_path"], row["a8_path"], row["wrapped_layers"], row["gpu"], row["port"]])))
PY
}

run_dev() {
    "$ROBOCASA_PY" "$SELECTION" materialize-dev
    local wave config plan a8 wrapped gpu port pid failed
    while read -r wave; do
        require_gpus_free
        failed=0
        pids=()
        while IFS=$'\t' read -r config plan a8 wrapped gpu port; do
            [[ -n "$config" ]] || continue
            run_dev_config "$config" "$plan" "$a8" "$wrapped" "$gpu" "$port" &
            pids+=("$!")
        done < <(wave_rows "$wave")
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" == 0 ]] || return 1
    done < <("$ROBOCASA_PY" - "$EXECUTION/selection/pi05_control_dev_schedule.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["waves"]:
    print(int(row["wave"]))
PY
    )
}

select_controls() {
    "$ROBOCASA_PY" "$SELECTION" select
    "$ROBOCASA_PY" "$SELECTION" formal-spec
}

run_formal_config() (
    set -euo pipefail
    local config="$1" plan="$2" a8="$3" wrapped="$4" gpu_csv="$5" port_csv="$6"
    local run_dir="$EXECUTION/runs/pi05_controls_heldout46/$config" control="$run_dir/control"
    local gpus=() ports=() instances=() server_pids=() hashes=()
    IFS=',' read -r -a gpus <<<"$gpu_csv"
    IFS=',' read -r -a ports <<<"$port_csv"
    [[ "${#gpus[@]}" == 4 && "${#ports[@]}" == 4 ]] || {
        echo "formal config requires four GPU replicas: $config" >&2
        return 1
    }
    export PI05_WEEK1_CONTROL_DIR="$control"
    mkdir -p "$control/workers"
    cleanup() {
        local instance
        for instance in "${instances[@]}"; do
            "$SERVER" stop "$instance" >/dev/null 2>&1 || true
        done
    }
    trap cleanup EXIT
    local index instance pid failed=0
    for index in {0..3}; do
        instance="pi05_${config}_g${gpus[$index]}"
        instances+=("$instance")
        "$SERVER" start "$config" "${gpus[$index]}" "${ports[$index]}" \
            "$instance" "$plan" "$a8" "$wrapped" &
        server_pids+=("$!")
    done
    for pid in "${server_pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || return 1
    run_preflight "$run_dir" "$control" "$config" "${instances[0]}" "${ports[0]}" "${gpus[0]}" \
        "OpenDrawer,TurnOnMicrowave"
    local manifest="$run_dir/manifest.json" mode=create
    [[ -f "$manifest" ]] && mode=verify
    args=(
        --run-dir "$run_dir" --out "$manifest" --config "$config" --scope heldout46
        --plan "$plan" --a8 "$a8" --expected-wrapped "$wrapped"
        --orchestrator "$REPO_ROOT/scripts/run_pi05_controls_official.sh"
        --preflight-results "$run_dir/preflight/two_task_two_seed.jsonl"
        --preflight-audit "$run_dir/preflight/summary.json"
        --reference "gdsq_vla_selector,$SELECTOR_MANIFEST,gdsq_vla_runtime_selector"
        --reference "static_gdsq,$STATIC_MANIFEST,gdsq_vla"
    )
    for index in {0..3}; do
        instance="${instances[$index]}"
        args+=(--server "$instance,$config,${gpus[$index]},${ports[$index]},$control/$instance.runtime.json,$control/$instance.audit.json")
    done
    local quarter_spec quarter seeds shard
    for shard in {0..3}; do
        for quarter_spec in "q0,0-12" "q1,13-24" "q2,25-37" "q3,38-49"; do
            IFS=',' read -r quarter seeds <<<"$quarter_spec"
            args+=(--worker "${config}_s${shard}_${quarter},${instances[$shard]},$shard,4,$seeds")
        done
    done
    [[ "$mode" == verify ]] && args+=(--verify)
    "$ROBOCASA_PY" "$MANIFEST_TOOL" "${args[@]}"
    for index in {0..3}; do
        hashes+=("$(runtime_hash "$control/${instances[$index]}.audit.json")")
    done
    nohup "$ROBOCASA_PY" "$GPU_MONITOR" --run-dir "$run_dir" --interval 30 \
        >"$control/gpu_monitor.log" 2>&1 & monitor_pid=$!
    while true; do
        failed=0
        pids=()
        for shard in {0..3}; do
            for quarter_spec in "q0,0-12" "q1,13-24" "q2,25-37" "q3,38-49"; do
                IFS=',' read -r quarter seeds <<<"$quarter_spec"
                worker="${config}_s${shard}_${quarter}"
                "$WORKER" "$config" "${ports[$shard]}" "${gpus[$shard]}" "$shard" 4 "$seeds" \
                    "$worker" "${hashes[$shard]}" "$run_dir" heldout46 \
                    >"$control/workers/$worker.log" 2>&1 &
                pids+=("$!"); echo "$!" >"$control/workers/$worker.pid"
            done
        done
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        for pid_file in "$control/workers/${config}"_s{0,1,2,3}_q{0,1,2,3}.pid; do
            printf 'done\n' >"$pid_file"
        done
        [[ "$failed" == 0 ]] && break
        echo "[pi05 heldout46] retry config=$config" >&2
        sleep 30
    done
    wait "$monitor_pid" || true
    aggregate_run "$run_dir"
    "$ROBOCASA_PY" - "$run_dir/aggregate/summary.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); assert m["complete"] and m["completed_episodes"] == 2300
PY
)

formal_rows() {
    "$ROBOCASA_PY" - "$EXECUTION/specs/pi05_same_budget_controls_heldout46.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
for row in m["configs"]:
    print("\t".join(map(str, [row["formal_config_id"], row["plan_path"], row["a8_path"], row["wrapped_layers"], ",".join(map(str, row["gpus"])), ",".join(map(str, row["ports"]))])))
PY
}

run_formal() {
    select_controls
    require_gpus_free
    local config plan a8 wrapped gpus ports pid failed=0
    pids=()
    while IFS=$'\t' read -r config plan a8 wrapped gpus ports; do
        run_formal_config "$config" "$plan" "$a8" "$wrapped" "$gpus" "$ports" &
        pids+=("$!")
    done < <(formal_rows)
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}

status() {
    "$ROBOCASA_PY" - "$EXECUTION" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
reported=False
for path in sorted((root / "runs").glob("pi05_controls_*/*")):
    if not path.is_dir(): continue
    rows=[]
    for result in path.glob("results/**/*.jsonl"):
        rows.extend(json.loads(line) for line in result.read_text().splitlines() if line.strip())
    expected=200 if "dev4" in str(path) else 2300
    print(f"{path.parent.name}/{path.name}: rows={len(rows)}/{expected} (coverage only)")
    reported=True
for path in sorted((root / "screen").glob("pi05_*.json")):
    value=json.load(open(path)); print(f"{path.name}: complete={value.get('complete')} scores={len(value.get('scores', []))}")
    reported=True
if not reported:
    print("pi05 controls: state=not_started")
PY
}

cd "$REPO_ROOT"
case "${1:-}" in
    screen) screen_controls ;;
    run-dev) run_dev ;;
    select) select_controls ;;
    run-formal) run_formal ;;
    run-all) screen_controls; run_dev; run_formal ;;
    status) status ;;
    *) usage; exit 2 ;;
esac
