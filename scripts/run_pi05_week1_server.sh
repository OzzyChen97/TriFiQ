#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OPENPI_ROOT="$REPO_ROOT/code/pi05/openpi"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CHECKPOINT="$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch"
CHECKPOINT_SHA256="4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
PACK_DIR="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
CONTROL_DIR="${PI05_WEEK1_CONTROL_DIR:-$REPO_ROOT/runs/gdsq_week1_preregistered_v1/execution/control/pi05}"
AUDITOR="$REPO_ROOT/scripts/tools/audit_pi05_week1_runtime.py"

usage() {
    echo "usage: $0 start CONFIG GPU PORT INSTANCE PLAN A8 EXPECTED_WRAPPED | stop INSTANCE | status" >&2
}

sha256_file() {
    sha256sum "$1" | cut -d' ' -f1
}

clear_quant_environment() {
    local variable
    while IFS='=' read -r variable _; do
        case "$variable" in
            OPENPI_DUQUANT_*|OPENPI_ATM_*|OPENPI_OHB_*|OPENPI_RUNTIME_SELECTOR_*|OPENPI_FORMAL_*) unset "$variable" ;;
        esac
    done < <(env)
}

start_server() {
    if [[ $# -ne 7 ]]; then usage; exit 2; fi
    local config="$1" gpu="$2" port="$3" instance="$4" plan="$5" a8="$6" wrapped="$7"
    if [[ ! "$config" =~ ^[A-Za-z0-9_.-]+$ || ! "$instance" =~ ^[A-Za-z0-9_.-]+$ || ! "$gpu" =~ ^[0-7]$ || ! "$port" =~ ^[0-9]+$ || ! "$wrapped" =~ ^[0-9]+$ ]]; then
        echo "invalid week-1 server argument" >&2; exit 2
    fi
    for path in "$CHECKPOINT/model.safetensors" "$PACK_DIR/manifest.json" "$BUFFER" "$plan" "$a8" "$a8.json"; do
        if [[ ! -f "$path" ]]; then echo "missing artifact: $path" >&2; exit 1; fi
    done
    if [[ "$(sha256_file "$CHECKPOINT/model.safetensors")" != "$CHECKPOINT_SHA256" ]]; then
        echo "checkpoint hash mismatch" >&2; exit 1
    fi
    mkdir -p "$CONTROL_DIR"
    local pid_file="$CONTROL_DIR/$instance.pid"
    local runtime_file="$CONTROL_DIR/$instance.runtime.json"
    local audit_file="$CONTROL_DIR/$instance.audit.json"
    local log_file="$CONTROL_DIR/$instance.server.log"
    if [[ -f "$pid_file" ]]; then
        local previous
        previous="$(<"$pid_file")"
        if [[ "$previous" =~ ^[0-9]+$ ]] && kill -0 "$previous" 2>/dev/null; then
            echo "instance already running: $instance pid=$previous" >&2; exit 1
        fi
    fi
    (
        clear_quant_environment
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONUNBUFFERED=1
        export TORCHDYNAMO_DISABLE=1
        export OPENPI_MODEL_DTYPE=float16
        export OPENPI_CHECKPOINT_SHA256="$CHECKPOINT_SHA256"
        export OPENPI_FORMAL_MODE=1
        export OPENPI_FORMAL_WEEK1_CONTROL=1
        export OPENPI_FORMAL_EXPECT_WRAPPED="$wrapped"
        export OPENPI_CONFIG_ID="$config"
        export OPENPI_RUNTIME_INFO_PATH="$runtime_file"
        export OPENPI_DUQUANT_PLAN="$plan"
        export OPENPI_DUQUANT_PLAN_STRICT=1
        export OPENPI_DUQUANT_WBITS_DEFAULT=4
        export OPENPI_DUQUANT_ABITS=8
        export OPENPI_DUQUANT_BLOCK=64
        export OPENPI_DUQUANT_BLOCK_OUT=64
        export OPENPI_DUQUANT_EXPECT_BLOCK=64
        export OPENPI_DUQUANT_EXPECT_WRAPPED="$wrapped"
        export OPENPI_DUQUANT_LS=0.15
        export OPENPI_DUQUANT_PERMUTE=0
        export OPENPI_DUQUANT_ROW_ROT=restore
        export OPENPI_DUQUANT_ACT_PCT=99.9
        export OPENPI_DUQUANT_CALIB_STEPS=32
        export OPENPI_DUQUANT_DENOISING_STEPS=4
        export OPENPI_DUQUANT_PACKDIR="$PACK_DIR"
        export OPENPI_DUQUANT_ACT_SCALE_PATH="$a8"
        export OPENPI_DUQUANT_REQUIRE_ACT_SCALE=1
        export OPENPI_DUQUANT_CALIB_BUFFER_SHA256
        OPENPI_DUQUANT_CALIB_BUFFER_SHA256="$(sha256_file "$BUFFER")"
        export OPENPI_DUQUANT_STRICT_ARTIFACTS=1
        export OPENPI_DUQUANT_PRECACHE_WEIGHTS=1
        export OPENPI_DUQUANT_TRITON=0
        export OPENPI_DUQUANT_QUIET=1
        export OPENPI_ATM_ENABLE=0
        export OPENPI_OHB_ENABLE=0
        cd "$OPENPI_ROOT"
        exec nohup setsid "$OPENPI_PY" scripts/serve_pi05_quant_policy.py \
            --env ROBOCASA --port "$port" --denoising-steps 4 \
            policy:checkpoint --policy.config pi05_pretrain_human300 \
            --policy.dir "$CHECKPOINT"
    ) >"$log_file" 2>&1 </dev/null &
    local pid=$!
    echo "$pid" >"$pid_file"
    local attempt
    for attempt in $(seq 1 180); do
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 120 "$log_file" >&2; exit 1
        fi
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            "$ROBOCASA_PY" "$AUDITOR" --runtime "$runtime_file" --plan "$plan" \
                --a8 "$a8" --config "$config" --expected-wrapped "$wrapped" \
                --out "$audit_file" >/dev/null
            echo "started instance=$instance config=$config gpu=$gpu port=$port pid=$pid"
            return
        fi
        sleep 2
    done
    echo "server startup timed out: $instance" >&2; exit 1
}

stop_server() {
    if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9_.-]+$ ]]; then usage; exit 2; fi
    local instance="$1" pid_file="$CONTROL_DIR/$1.pid"
    if [[ ! -f "$pid_file" ]]; then return; fi
    local pid command
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        command="$(ps -p "$pid" -o args=)"
        if [[ "$command" != *serve_pi05_quant_policy.py* ]]; then
            echo "refusing unrelated pid=$pid command=$command" >&2; exit 1
        fi
        kill -- "-$pid" 2>/dev/null || kill "$pid"
    fi
}

status_all() {
    local pid_file pid state
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR"/*.pid; do
        pid="$(<"$pid_file")"; state=stale
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state=running; fi
        echo "$(basename "$pid_file" .pid) pid=$pid state=$state"
    done
}

case "${1:-}" in
    start) shift; start_server "$@" ;;
    stop) shift; stop_server "$@" ;;
    status) status_all ;;
    *) usage; exit 2 ;;
esac
