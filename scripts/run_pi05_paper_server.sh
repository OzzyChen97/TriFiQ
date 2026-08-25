#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OPENPI_ROOT="$REPO_ROOT/code/pi05/openpi"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CHECKPOINT="${PI05_CHECKPOINT:-$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch}"
PAPER_ROOT="${PI05_PAPER_ROOT:-$REPO_ROOT/runs/pi05_quantvla_paper}"
PAPER_PACK="${PI05_PAPER_PACK:-$PAPER_ROOT/packs/pi05_robocasa_block64_permute_w4a8_ls015}"
PAPER_BUFFER="${PI05_PAPER_BUFFER:-$PAPER_ROOT/calibration/robocasa_real_observations_128.npz}"
PAPER_PLAN="${PI05_PAPER_PLAN:-$PAPER_ROOT/plans/pi05_quantvla_paper_w4a8.plan.json}"
PAPER_A8="${PI05_PAPER_A8:-$PAPER_ROOT/a8/pi05_quantvla_paper_real32_p999_b32.npz}"
PAPER_ATM="${PI05_PAPER_ATM:-$PAPER_ROOT/atm_ohb/pi05_quantvla_paper_real128_scalar.json}"
GDSQ_PLAN="${PI05_PAPER_GDSQ_PLAN:-$PAPER_ROOT/plans/pi05_gdsq_vla_final_paper.plan.json}"
GDSQ_A8="${PI05_PAPER_GDSQ_A8:-$PAPER_ROOT/a8/pi05_gdsq_vla_final_real32_p999_b32.npz}"
GDSQ_ATM="${PI05_PAPER_GDSQ_ATM:-$PAPER_ROOT/atm_ohb/pi05_gdsq_vla_final_real128_scalar.json}"
CUSTOM_PACK="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
CUSTOM_BUFFER="$REPO_ROOT/runs/pi05_gdsq_port/probes/pi05_robocasa_official_buffer32_state16.npz"
CUSTOM_PLAN="$REPO_ROOT/runs/pi05_gdsq_port/plans/pi05_quantvla_full_w4a8.plan.json"
CUSTOM_A8="$REPO_ROOT/runs/pi05_gdsq_port/a8_scales/pi05_full_w4a8_state16_p999_b32.npz"
CUSTOM_ATM="$REPO_ROOT/runs/pi05_gdsq_port/atm_ohb/pi05_full_w4a8_state16_static_expert.json"
CONTROL_DIR="${PI05_PAPER_CONTROL_DIR:-$PAPER_ROOT/control}"
CHECKPOINT_SHA256="4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"

usage() {
    echo "usage: $0 start CONFIG GPU PORT INSTANCE | stop INSTANCE | status" >&2
    echo "CONFIG: fp16 | quantvla_w4a8_atmohb | gdsq_vla_atmohb | gdsq_vla | paper_fp16 | quantvla_paper_w4a8 | quantvla_paper_w4a8_atmohb | gdsq_vla_paper | gdsq_vla_paper_atmohb | quantvla_custom_w4a8_atmohb" >&2
}

sha256_file() {
    sha256sum "$1" | cut -d' ' -f1
}

require_file() {
    [[ -f "$1" ]] || { echo "missing required artifact: $1" >&2; exit 1; }
}

clear_quant_environment() {
    local variable
    while IFS='=' read -r variable _; do
        case "$variable" in
            OPENPI_DUQUANT_*|OPENPI_ATM_*|OPENPI_OHB_*) unset "$variable" ;;
        esac
    done < <(env)
}

configure_quant() {
    local plan="$1" scale="$2" pack="$3" buffer="$4" permute="$5" wrapped="$6"
    export OPENPI_DUQUANT_PLAN="$plan"
    export OPENPI_DUQUANT_PLAN_STRICT=1
    export OPENPI_DUQUANT_WBITS_DEFAULT=4
    export OPENPI_DUQUANT_ABITS=8
    export OPENPI_DUQUANT_BLOCK=64
    export OPENPI_DUQUANT_BLOCK_OUT=64
    export OPENPI_DUQUANT_EXPECT_BLOCK=64
    export OPENPI_DUQUANT_EXPECT_WRAPPED="$wrapped"
    export OPENPI_DUQUANT_LS=0.15
    export OPENPI_DUQUANT_PERMUTE="$permute"
    export OPENPI_DUQUANT_ROW_ROT=restore
    export OPENPI_DUQUANT_ACT_PCT=99.9
    export OPENPI_DUQUANT_CALIB_STEPS=32
    export OPENPI_DUQUANT_DENOISING_STEPS=10
    export OPENPI_DUQUANT_PACKDIR="$pack"
    export OPENPI_DUQUANT_PACK_MANIFEST_SHA256
    OPENPI_DUQUANT_PACK_MANIFEST_SHA256="$(sha256_file "$pack/manifest.json")"
    export OPENPI_DUQUANT_ACT_SCALE_PATH="$scale"
    export OPENPI_DUQUANT_REQUIRE_ACT_SCALE=1
    export OPENPI_DUQUANT_CALIB_BUFFER_SHA256
    OPENPI_DUQUANT_CALIB_BUFFER_SHA256="$(sha256_file "$buffer")"
    export OPENPI_DUQUANT_STRICT_ARTIFACTS=1
    export OPENPI_DUQUANT_PRECACHE_WEIGHTS=1
    export OPENPI_DUQUANT_TRITON=0
    export OPENPI_DUQUANT_QUIET=1
    export OPENPI_CHECKPOINT_SHA256="$CHECKPOINT_SHA256"
    if [[ "$permute" == 1 ]]; then
        export OPENPI_DUQUANT_ATTEST_PERMUTE=1
        export OPENPI_DUQUANT_EXPECT_PERMUTE_METADATA=1
    fi
}

configure_atm() {
    local artifact="$1" plan="$2" buffer="$3" paper="$4"
    export OPENPI_ATM_ENABLE=1
    export OPENPI_OHB_ENABLE=1
    export OPENPI_ATM_ALPHA_PATH="$artifact"
    export OPENPI_ATM_SCOPE=expert
    export OPENPI_OHB_SCOPE=expert
    export OPENPI_ATM_STRICT=1
    export OPENPI_ATM_EXPECT_LAYERS=18
    export OPENPI_ATM_EXPECT_PLAN_SHA256
    OPENPI_ATM_EXPECT_PLAN_SHA256="$(sha256_file "$plan")"
    export OPENPI_ATM_EXPECT_BUFFER_SHA256
    OPENPI_ATM_EXPECT_BUFFER_SHA256="$(sha256_file "$buffer")"
    if [[ "$paper" == 1 ]]; then
        export OPENPI_ATM_APPLICATION=fold_q_weight
        export OPENPI_OHB_EXPECT_MODE=per_layer_post_projection
        export OPENPI_OHB_APPLICATION=fold_o_weight
    fi
}

validate_runtime() {
    local runtime="$1" config="$2" permute="$3" atm="$4" paper="$5"
    "$OPENPI_PY" - "$runtime" "$config" "$permute" "$atm" "$paper" <<'PY'
import json
import sys

path, config, permute_text, atm_text, paper_text = sys.argv[1:]
root = json.load(open(path, encoding="utf-8"))["openpi_runtime"]
duquant = root.get("duquant") or {}
scaling = root.get("atm_ohb") or {}
dtype = root.get("model_dtype") or {}
if root.get("config_id") != config:
    raise SystemExit(f"runtime config mismatch: {root.get('config_id')!r} != {config!r}")
if dtype.get("resolved") != "float16":
    raise SystemExit(f"runtime is not strict FP16: {dtype}")
wanted_wrapped = 0 if config in ("fp16", "paper_fp16") else (
    69 if config in ("gdsq_vla", "gdsq_vla_atmohb", "gdsq_vla_paper", "gdsq_vla_paper_atmohb") else 180
)
if int(duquant.get("wrapped_layers", 0)) != wanted_wrapped:
    raise SystemExit(f"wrapped-layer mismatch: {duquant}")
if wanted_wrapped and bool(duquant.get("enable_permute")) != bool(int(permute_text)):
    raise SystemExit(f"permutation mismatch: {duquant}")
if wanted_wrapped and not duquant.get("pack_manifest_sha256"):
    raise SystemExit(f"pack-manifest hash missing from runtime: {duquant}")
if wanted_wrapped and (
    duquant.get("execution_backend") != "fake_quant_fp16_gemm"
    or duquant.get("integer_gemm") is not False
    or duquant.get("packed_low_bit_residency") is not False
):
    raise SystemExit(f"unexpected quant execution backend: {duquant}")
if bool(scaling.get("enabled")) != bool(int(atm_text)):
    raise SystemExit(f"ATM/OHB enablement mismatch: {scaling}")
if int(paper_text) and (
    scaling.get("atm_application") != "fold_q_weight"
    or scaling.get("ohb_mode") != "per_layer_post_projection"
    or scaling.get("ohb_application") != "fold_o_weight"
):
    raise SystemExit(f"paper ATM/OHB mode mismatch: {scaling}")
print(json.dumps({
    "config": config,
    "wrapped_layers": wanted_wrapped,
    "enable_permute": bool(int(permute_text)) if wanted_wrapped else False,
    "atm_ohb": bool(int(atm_text)),
    "paper_faithful_scaling": bool(int(paper_text)),
}, sort_keys=True))
PY
}

start_server() {
    [[ $# -eq 4 ]] || { usage; exit 2; }
    local config="$1" gpu="$2" port="$3" instance="$4"
    [[ "$gpu" =~ ^[0-7]$ && "$port" =~ ^[0-9]+$ && "$instance" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        echo "invalid GPU, port, or instance" >&2; exit 2;
    }
    local permute=0 atm=0 paper=0
    require_file "$CHECKPOINT/model.safetensors"
    case "$config" in
        fp16|paper_fp16) ;;
        quantvla_paper_w4a8)
            permute=1
            for path in "$PAPER_PACK/manifest.json" "$PAPER_BUFFER" "$PAPER_PLAN" "$PAPER_A8" "$PAPER_A8.json"; do require_file "$path"; done
            ;;
        quantvla_w4a8_atmohb|quantvla_paper_w4a8_atmohb)
            permute=1; atm=1; paper=1
            for path in "$PAPER_PACK/manifest.json" "$PAPER_BUFFER" "$PAPER_PLAN" "$PAPER_A8" "$PAPER_A8.json" "$PAPER_ATM"; do require_file "$path"; done
            ;;
        gdsq_vla|gdsq_vla_paper)
            permute=1
            for path in "$PAPER_PACK/manifest.json" "$PAPER_BUFFER" "$GDSQ_PLAN" "$GDSQ_A8" "$GDSQ_A8.json"; do require_file "$path"; done
            ;;
        gdsq_vla_atmohb|gdsq_vla_paper_atmohb)
            permute=1; atm=1; paper=1
            for path in "$PAPER_PACK/manifest.json" "$PAPER_BUFFER" "$GDSQ_PLAN" "$GDSQ_A8" "$GDSQ_A8.json" "$GDSQ_ATM"; do require_file "$path"; done
            ;;
        quantvla_custom_w4a8_atmohb)
            atm=1
            for path in "$CUSTOM_PACK/manifest.json" "$CUSTOM_BUFFER" "$CUSTOM_PLAN" "$CUSTOM_A8" "$CUSTOM_A8.json" "$CUSTOM_ATM"; do require_file "$path"; done
            ;;
        *) usage; exit 2 ;;
    esac
    mkdir -p "$CONTROL_DIR"
    local pid_file="$CONTROL_DIR/$instance.pid" log_file="$CONTROL_DIR/$instance.server.log"
    local runtime_file="$CONTROL_DIR/$instance.runtime.json"
    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid="$(<"$pid_file")"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "instance already running: $instance pid=$old_pid" >&2; exit 1
        fi
    fi
    (
        clear_quant_environment
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 OPENPI_MODEL_DTYPE=float16
        export OPENPI_FORMAL_MODE=0 OPENPI_CONFIG_ID="$config" OPENPI_RUNTIME_INFO_PATH="$runtime_file"
        case "$config" in
            fp16|quantvla_w4a8_atmohb|gdsq_vla_atmohb|gdsq_vla)
                export OPENPI_FORMAL_MODE=1 OPENPI_FORMAL_EXPECT_WRAPPED=69
                ;;
        esac
        case "$config" in
            fp16|paper_fp16) export OPENPI_NATIVE_RNG_SEED="${PI05_NATIVE_RNG_SEED:-51001}" ;;
            quantvla_paper_w4a8) export OPENPI_NATIVE_RNG_SEED="${PI05_NATIVE_RNG_SEED:-51002}" ;;
            quantvla_w4a8_atmohb|quantvla_paper_w4a8_atmohb) export OPENPI_NATIVE_RNG_SEED="${PI05_NATIVE_RNG_SEED:-51003}" ;;
            quantvla_custom_w4a8_atmohb) export OPENPI_NATIVE_RNG_SEED="${PI05_NATIVE_RNG_SEED:-51004}" ;;
            gdsq_vla|gdsq_vla_paper) export OPENPI_NATIVE_RNG_SEED="${PI05_NATIVE_RNG_SEED:-51005}" ;;
            gdsq_vla_atmohb|gdsq_vla_paper_atmohb) export OPENPI_NATIVE_RNG_SEED="${PI05_NATIVE_RNG_SEED:-51006}" ;;
        esac
        case "$config" in
            fp16|paper_fp16) ;;
            quantvla_paper_w4a8)
                configure_quant "$PAPER_PLAN" "$PAPER_A8" "$PAPER_PACK" "$PAPER_BUFFER" 1 180
                ;;
            quantvla_w4a8_atmohb|quantvla_paper_w4a8_atmohb)
                configure_quant "$PAPER_PLAN" "$PAPER_A8" "$PAPER_PACK" "$PAPER_BUFFER" 1 180
                configure_atm "$PAPER_ATM" "$PAPER_PLAN" "$PAPER_BUFFER" 1
                ;;
            gdsq_vla|gdsq_vla_paper)
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$PAPER_PACK" "$PAPER_BUFFER" 1 69
                ;;
            gdsq_vla_atmohb|gdsq_vla_paper_atmohb)
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$PAPER_PACK" "$PAPER_BUFFER" 1 69
                configure_atm "$GDSQ_ATM" "$GDSQ_PLAN" "$PAPER_BUFFER" 1
                ;;
            quantvla_custom_w4a8_atmohb)
                configure_quant "$CUSTOM_PLAN" "$CUSTOM_A8" "$CUSTOM_PACK" "$CUSTOM_BUFFER" 0 180
                configure_atm "$CUSTOM_ATM" "$CUSTOM_PLAN" "$CUSTOM_BUFFER" 0
                ;;
        esac
        cd "$OPENPI_ROOT"
        exec nohup setsid "$OPENPI_PY" scripts/serve_pi05_quant_policy.py \
            --env ROBOCASA --port "$port" policy:checkpoint \
            --policy.config pi05_pretrain_human300 --policy.dir "$CHECKPOINT"
    ) >"$log_file" 2>&1 </dev/null &
    local server_pid=$!
    echo "$server_pid" >"$pid_file"
    local attempt
    for attempt in $(seq 1 180); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -n 120 "$log_file" >&2; exit 1
        fi
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            validate_runtime "$runtime_file" "$config" "$permute" "$atm" "$paper"
            echo "started instance=$instance config=$config gpu=$gpu port=$port pid=$server_pid"
            return
        fi
        sleep 2
    done
    echo "paper server startup timed out: $instance" >&2; exit 1
}

stop_server() {
    [[ $# -eq 1 ]] || { usage; exit 2; }
    local instance="$1" pid_file="$CONTROL_DIR/$1.pid"
    [[ -f "$pid_file" ]] || { echo "instance not found: $instance"; return; }
    local pid
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
    echo "stopped instance=$instance"
}

status() {
    mkdir -p "$CONTROL_DIR"
    local pid_file pid instance
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR"/*.pid; do
        pid="$(<"$pid_file")"; instance="$(basename "$pid_file" .pid)"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$instance running pid=$pid"
        else
            echo "$instance stopped"
        fi
    done
}

case "${1:-}" in
    start) shift; start_server "$@" ;;
    stop) shift; stop_server "$@" ;;
    status) status ;;
    *) usage; exit 2 ;;
esac
