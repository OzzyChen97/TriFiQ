#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OPENPI_ROOT="$REPO_ROOT/code/pi05/openpi"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
CHECKPOINT="${PI05_CHECKPOINT:-$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch}"
CHECKPOINT_SHA256="${PI05_CHECKPOINT_SHA256:-4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c}"
PACK_DIR="${PI05_PACK_DIR:-$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015}"
ALIGNED_ROOT="${PI05_GR00T_ALIGNED_ROOT:-$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned}"
BUFFER="${PI05_CALIBRATION_BUFFER:-$ALIGNED_ROOT/calibration/pi05_robocasa365_seed0_n256.npz}"
FULL_PLAN="${PI05_FULL_PLAN:-$ALIGNED_ROOT/plans/pi05_quantvla_uniform_w4a8_d4.plan.json}"
GDSQ_PLAN="${PI05_GDSQ_PLAN:-$ALIGNED_ROOT/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json}"
FULL_A8="${PI05_FULL_A8:-$ALIGNED_ROOT/a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz}"
GDSQ_A8="${PI05_GDSQ_A8:-$ALIGNED_ROOT/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz}"
FULL_ATM="${PI05_FULL_ATM:-$ALIGNED_ROOT/atm_ohb/pi05_quantvla_uniform_w4a8_d4_static_perhead.json}"
GDSQ_ATM="${PI05_GDSQ_ATM:-$ALIGNED_ROOT/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json}"
SOFTFOLD_ATM="${PI05_SOFTFOLD_ATM:-$GDSQ_ATM}"
V3_ROOT="${PI05_V3_ROOT:-$REPO_ROOT/runs/errorfold_v3_15x20/calibration/pi05}"
V3_PACK_DIR="${PI05_V3_PACK_DIR:-$V3_ROOT/identity_pack}"
V3_A8="${PI05_V3_A8:-$V3_ROOT/a8_scales.npz}"
V3_HESSIAN_W4="${PI05_V3_HESSIAN_W4:-$V3_ROOT/hessian_w4.npz}"
V3_ERRORFOLD_DFUNC="${PI05_V3_ERRORFOLD_DFUNC:-$V3_ROOT/errorfold_dfunc.json}"
V3_ERRORFOLD_DPAC="${PI05_V3_ERRORFOLD_DPAC:-$V3_ROOT/errorfold_dpac_v2.json}"
V5_ROOT="${PI05_V5_ROOT:-$REPO_ROOT/runs/errorfold_v4_iter/calibration/pi05_full_w4}"
V5_PACK_DIR="${PI05_V5_PACK_DIR:-$V5_ROOT/identity_pack}"
V5_HESSIAN_W4="${PI05_V5_HESSIAN_W4:-$V5_ROOT/hessian_w4.npz}"
V5_ERRORFOLD="${PI05_V5_ERRORFOLD:-}"
FULL_CONTEXT_ROOT="${PI05_FULL_CONTEXT_ROOT:-$REPO_ROOT/runs/full_context_v1/pi05}"
FULL_CONTEXT_PLAN="${PI05_FULL_CONTEXT_PLAN:-$FULL_CONTEXT_ROOT/round1/pi05_full_context_round1_frozen.json}"
FULL_CONTEXT_PACK_DIR="${PI05_FULL_CONTEXT_PACK_DIR:-$FULL_CONTEXT_ROOT/calibration_flow10/identity_pack}"
FULL_CONTEXT_HESSIAN_W4="${PI05_FULL_CONTEXT_HESSIAN_W4:-$FULL_CONTEXT_ROOT/deployment/hessian_w4.npz}"
FULL_CONTEXT_BUFFER="${PI05_FULL_CONTEXT_BUFFER:-$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz}"
FLOW_STEPS="${PI05_FLOW_STEPS:-4}"
RUNTIME_SELECTOR="${PI05_RUNTIME_SELECTOR:-$REPO_ROOT/runs/atmohb_dynamic_selector_v8/selector.json}"
OMEGA_ROOT="${PI05_OMEGA_ROOT:-$REPO_ROOT/external/Omega-QVLA}"
OMEGA_PACK="${PI05_OMEGA_PACK:-}"
OMEGA_CALIBRATION_MANIFEST="${PI05_OMEGA_CALIBRATION_MANIFEST:-}"
OMEGA_ATTESTATION="${OMEGA_PACK%.pt}.attestation.json"
OMEGA_INCLUDE='.*paligemma_with_expert\.(?:paligemma\.model\.language_model|gemma_expert\.model)\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
CONTROL_DIR="${PI05_CONTROL_DIR:-$ALIGNED_ROOT/official_target_paired50/control}"
CONTROL_DIR="$(mkdir -p "$CONTROL_DIR" && cd "$CONTROL_DIR" && pwd)"

usage() {
    echo "usage: $0 start CONFIG GPU PORT INSTANCE | stop INSTANCE | status" >&2
    echo "CONFIG: fp16 | omega_qvla_w4a4 | quantvla_w4a8_paper | quantvla_w4a8_dynamic | quantvla_w4a8_dynamic_profile | full_context_w4a8_dynamic_profile | quantvla_w4a8_dynamic_profile_errorfold | errorfold_dfunc | errorfold_dpac_v2 | legacy configs" >&2
}

sha256_file() {
    local artifact="$1"
    local resolved signature cache_key cache_dir cache_file lock_file cached_signature cached_digest digest temporary fd
    resolved="$(readlink -f -- "$artifact")"
    signature="$(stat -c '%d:%i:%s:%y:%z' -- "$resolved")"
    cache_dir="${QUANTVLA_HASH_CACHE_DIR:-$REPO_ROOT/runs/.artifact_sha256_cache}"
    mkdir -p "$cache_dir"
    cache_key="$(printf '%s' "$resolved" | sha256sum | cut -d' ' -f1)"
    cache_file="$cache_dir/$cache_key.tsv"
    lock_file="$cache_dir/$cache_key.lock"
    exec {fd}>"$lock_file"
    flock "$fd"
    if [[ -s "$cache_file" ]]; then
        IFS=$'\t' read -r cached_signature cached_digest <"$cache_file" || true
        if [[ "$cached_signature" == "$signature" && "$cached_digest" =~ ^[0-9a-f]{64}$ ]]; then
            flock -u "$fd"
            exec {fd}>&-
            printf '%s\n' "$cached_digest"
            return
        fi
    fi
    digest="$(sha256sum "$resolved" | cut -d' ' -f1)"
    temporary="$cache_file.tmp.$$"
    printf '%s\t%s\n' "$signature" "$digest" >"$temporary"
    mv -f "$temporary" "$cache_file"
    flock -u "$fd"
    exec {fd}>&-
    printf '%s\n' "$digest"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "missing required artifact: $1" >&2
        exit 1
    fi
}

wrapped_layers() {
    "$OPENPI_PY" - "$1" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
layers = payload.get("layers") or {}
selected = sum(
    not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
    and int(row.get("bits", 0) or 0) > 0
    for row in layers.values()
)
if selected <= 0:
    raise SystemExit("quant plan selects zero layers")
print(selected)
PY
}

clear_quant_environment() {
    local variable
    while IFS='=' read -r variable _; do
        case "$variable" in
            OPENPI_DUQUANT_*|OPENPI_ATM_*|OPENPI_OHB_*|OPENPI_RUNTIME_SELECTOR_*|OPENPI_ERRORFOLD_PATH|OPENPI_OMEGA_*|GR00T_GPTQ*|QUANTVLA_ADAPTER_ONLY) unset "$variable" ;;
        esac
    done < <(env)
}

configure_quant() {
    local plan="$1"
    local scale="$2"
    local wrapped="$3"
    local pack_dir="${4:-$PACK_DIR}"
    local row_rotation="${5:-restore}"
    local activation_mode="${6:-static_a8}"
    local calibration_buffer="${7:-$BUFFER}"
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
    export OPENPI_DUQUANT_ROW_ROT="$row_rotation"
    export OPENPI_DUQUANT_ACT_PCT=99.9
    export OPENPI_DUQUANT_CALIB_STEPS=32
    export OPENPI_DUQUANT_DENOISING_STEPS="$FLOW_STEPS"
    export OPENPI_DUQUANT_PACKDIR="$pack_dir"
    if [[ "$activation_mode" == "dynamic_a8" ]]; then
        unset OPENPI_DUQUANT_ACT_SCALE_PATH
        export OPENPI_DUQUANT_ACT_DYNAMIC=1
        export OPENPI_DUQUANT_REQUIRE_ACT_SCALE=0
    else
        export OPENPI_DUQUANT_ACT_SCALE_PATH="$scale"
        export OPENPI_DUQUANT_ACT_DYNAMIC=0
        export OPENPI_DUQUANT_REQUIRE_ACT_SCALE=1
    fi
    export OPENPI_DUQUANT_CALIB_BUFFER_SHA256
    OPENPI_DUQUANT_CALIB_BUFFER_SHA256="$(sha256_file "$calibration_buffer")"
    export OPENPI_DUQUANT_STRICT_ARTIFACTS=1
    export OPENPI_DUQUANT_PRECACHE_WEIGHTS=1
    export OPENPI_DUQUANT_TRITON=1
    export OPENPI_DUQUANT_QUIET=1
    export OPENPI_CHECKPOINT_SHA256="$CHECKPOINT_SHA256"
}

configure_errorfold_v3() {
    local artifact="$1"
    configure_errorfold "$artifact" "$V3_HESSIAN_W4" "$FULL_PLAN"
}

configure_errorfold() {
    local artifact="$1"
    local hessian_w4="$2"
    local plan="$3"
    export OPENPI_DUQUANT_HESSIAN_W4_PATH="$hessian_w4"
    export OPENPI_ERRORFOLD_PATH="$artifact"
    configure_atm "$artifact" "$plan" 1 1
    unset OPENPI_OHB_EXPECT_MODE
    export OPENPI_ATM_APPLICATION=fold_q_weight
    export OPENPI_OHB_APPLICATION=fold_o_weight_perhead
}

configure_atm() {
    local artifact="$1"
    local plan="$2"
    local atm_enable="${3:-1}"
    local ohb_enable="${4:-1}"
    export OPENPI_ATM_ENABLE="$atm_enable"
    export OPENPI_OHB_ENABLE="$ohb_enable"
    export OPENPI_ATM_ALPHA_PATH="$artifact"
    export OPENPI_ATM_SCOPE=expert
    export OPENPI_OHB_SCOPE=expert
    export OPENPI_ATM_STRICT=1
    export OPENPI_ATM_EXPECT_LAYERS=18
    export OPENPI_ATM_APPLICATION="${PI05_ATM_APPLICATION:-runtime_query}"
    export OPENPI_OHB_EXPECT_MODE=per_head_pre_projection
    export OPENPI_OHB_APPLICATION="${PI05_OHB_APPLICATION:-runtime_output}"
    export OPENPI_ATM_EXPECT_PLAN_SHA256
    OPENPI_ATM_EXPECT_PLAN_SHA256="$(sha256_file "$plan")"
    export OPENPI_ATM_EXPECT_BUFFER_SHA256
    OPENPI_ATM_EXPECT_BUFFER_SHA256="$(sha256_file "$BUFFER")"
}

start_server() {
    if [[ $# -ne 4 ]]; then
        usage
        exit 2
    fi
    local config="$1"
    local gpu="$2"
    local port="$3"
    local instance="$4"
    if [[ ! "$gpu" =~ ^[0-7]$ || ! "$port" =~ ^[0-9]+$ || ! "$instance" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "invalid GPU, port, or instance" >&2
        exit 2
    fi
    mkdir -p "$CONTROL_DIR"
    local pid_file="$CONTROL_DIR/$instance.pid"
    local log_file="$CONTROL_DIR/$instance.server.log"
    local runtime_file="$CONTROL_DIR/$instance.runtime.json"
    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid="$(<"$pid_file")"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "instance already running: $instance pid=$old_pid" >&2
            exit 1
        fi
    fi

    for artifact in "$CHECKPOINT/model.safetensors" "$BUFFER"; do
        require_file "$artifact"
    done
    local actual_checkpoint_sha256
    actual_checkpoint_sha256="$(sha256_file "$CHECKPOINT/model.safetensors")"
    if [[ "$actual_checkpoint_sha256" != "$CHECKPOINT_SHA256" ]]; then
        echo "checkpoint hash mismatch: $CHECKPOINT/model.safetensors" >&2
        echo "  actual:   $actual_checkpoint_sha256" >&2
        echo "  expected: $CHECKPOINT_SHA256" >&2
        exit 1
    fi
    local gdsq_wrapped=""
    case "$config" in
        fp16) ;;
        omega_qvla_w4a4)
            for artifact in "$OMEGA_PACK" "$OMEGA_ATTESTATION" "$OMEGA_CALIBRATION_MANIFEST"; do
                require_file "$artifact"
            done
            ;;
        quantvla_w4a8_atmohb|quantvla_w4a8_paper)
            for artifact in "$PACK_DIR/manifest.json" "$FULL_PLAN" "$FULL_A8" "$FULL_A8.json" "$FULL_ATM"; do
                require_file "$artifact"
            done
            ;;
        errorfold_dfunc|errorfold_dpac_v2)
            local selected="$V3_ERRORFOLD_DFUNC"
            [[ "$config" == "errorfold_dpac_v2" ]] && selected="$V3_ERRORFOLD_DPAC"
            for artifact in "$FULL_PLAN" "$V3_A8" "$V3_A8.json" \
                "$V3_HESSIAN_W4" "$V3_HESSIAN_W4.json" "$selected"; do
                require_file "$artifact"
            done
            if [[ ! -d "$V3_PACK_DIR" ]]; then
                echo "missing v3 identity pack directory: $V3_PACK_DIR" >&2
                exit 1
            fi
            ;;
        quantvla_w4a8_dynamic)
            for artifact in "$FULL_PLAN" "$V5_HESSIAN_W4" "$V5_HESSIAN_W4.json"; do
                require_file "$artifact"
            done
            if [[ ! -d "$V5_PACK_DIR" ]]; then
                echo "missing dynamic-A8 identity pack directory: $V5_PACK_DIR" >&2
                exit 1
            fi
            ;;
        quantvla_w4a8_dynamic_profile)
            for artifact in "$FULL_PLAN" "$V5_HESSIAN_W4" "$V5_HESSIAN_W4.json"; do
                require_file "$artifact"
            done
            if [[ ! -d "$V5_PACK_DIR" ]]; then
                echo "missing dynamic-A8 profile pack directory: $V5_PACK_DIR" >&2
                exit 1
            fi
            gdsq_wrapped="$(wrapped_layers "$FULL_PLAN")"
            ;;
        full_context_w4a8_dynamic_profile)
            for artifact in "$FULL_CONTEXT_PLAN" "$FULL_CONTEXT_HESSIAN_W4" "$FULL_CONTEXT_HESSIAN_W4.json" "$FULL_CONTEXT_BUFFER"; do
                require_file "$artifact"
            done
            if [[ ! -d "$FULL_CONTEXT_PACK_DIR" ]]; then
                echo "missing full-context identity pack directory: $FULL_CONTEXT_PACK_DIR" >&2
                exit 1
            fi
            gdsq_wrapped="$(wrapped_layers "$FULL_CONTEXT_PLAN")"
            ;;
        quantvla_w4a8_dynamic_profile_errorfold)
            for artifact in "$FULL_PLAN" "$V5_HESSIAN_W4" "$V5_HESSIAN_W4.json" "$V5_ERRORFOLD"; do
                require_file "$artifact"
            done
            if [[ ! -d "$V5_PACK_DIR" ]]; then
                echo "missing dynamic-A8 ErrorFold profile pack directory: $V5_PACK_DIR" >&2
                exit 1
            fi
            gdsq_wrapped="$(wrapped_layers "$FULL_PLAN")"
            ;;
        quantvla_w4a8_softfold_dfunc|quantvla_w4a8_softfold_dpac)
            for artifact in "$PACK_DIR/manifest.json" "$FULL_PLAN" "$FULL_A8" "$FULL_A8.json" "$SOFTFOLD_ATM"; do
                require_file "$artifact"
            done
            ;;
        gdsq_vla_atmohb|gdsq_vla_atm_only|gdsq_vla_ohb_only|gdsq_vla_runtime_selector|gdsq_vla_softfold_dfunc|gdsq_vla_softfold_dpac)
            for artifact in "$PACK_DIR/manifest.json" "$GDSQ_PLAN" "$GDSQ_A8" "$GDSQ_A8.json" "$GDSQ_ATM"; do
                require_file "$artifact"
            done
            if [[ "$config" == "gdsq_vla_runtime_selector" ]]; then
                require_file "$RUNTIME_SELECTOR"
            fi
            gdsq_wrapped="$(wrapped_layers "$GDSQ_PLAN")"
            ;;
        gdsq_vla)
            for artifact in "$PACK_DIR/manifest.json" "$GDSQ_PLAN" "$GDSQ_A8" "$GDSQ_A8.json"; do
                require_file "$artifact"
            done
            gdsq_wrapped="$(wrapped_layers "$GDSQ_PLAN")"
            ;;
        *) usage; exit 2 ;;
    esac

    (
        clear_quant_environment
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONUNBUFFERED=1
        export TORCHDYNAMO_DISABLE=1
        export OPENPI_MODEL_DTYPE=float16
        export OPENPI_CHECKPOINT_SHA256="$CHECKPOINT_SHA256"
        export OPENPI_FORMAL_MODE=1
        export OPENPI_FORMAL_FLOW_STEPS="$FLOW_STEPS"
        if [[ "$FLOW_STEPS" == 10 ]]; then
            export OPENPI_FULL_CONTEXT_PROTOCOL=1
        fi
        export OPENPI_CONFIG_ID="$config"
        export OPENPI_RUNTIME_INFO_PATH="$runtime_file"
        # Every DuQuant formal row, including the historical GDSQ baseline,
        # must use the same packed-W4 deployment path.  The formal runtime
        # validator already requires packed residency and zero FP-sized W4
        # buffers; setting this once prevents a fake-quant baseline.
        case "$config" in
            fp16|omega_qvla_w4a4) ;;
            *) export QUANTVLA_ADAPTER_ONLY=1 ;;
        esac
        case "$config" in
            fp16) ;;
            omega_qvla_w4a4)
                export OPENPI_OMEGA_QVLA=1
                export OPENPI_OMEGA_ROOT="$OMEGA_ROOT"
                export OPENPI_OMEGA_PACK="$OMEGA_PACK"
                export OPENPI_OMEGA_PACK_SHA256
                OPENPI_OMEGA_PACK_SHA256="$(sha256_file "$OMEGA_PACK")"
                export OPENPI_OMEGA_CALIBRATION_MANIFEST="$OMEGA_CALIBRATION_MANIFEST"
                export OPENPI_FORMAL_EXPECT_WRAPPED=252
                export GR00T_GPTQ=1
                export GR00T_GPTQ_PATH="$OMEGA_PACK"
                export GR00T_GPTQ_INCLUDE="$OMEGA_INCLUDE"
                export GR00T_GPTQ_EXCLUDE='(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)'
                export GR00T_GPTQ_WBITS_DEFAULT=4
                export GR00T_GPTQ_ABITS=4
                export GR00T_GPTQ_MISSING=error
                export GR00T_GPTQ_KEEP_FP=0
                ;;
            quantvla_w4a8_atmohb|quantvla_w4a8_paper)
                export QUANTVLA_ADAPTER_ONLY=1
                configure_quant "$FULL_PLAN" "$FULL_A8" 180
                configure_atm "$FULL_ATM" "$FULL_PLAN"
                ;;
            errorfold_dfunc|errorfold_dpac_v2)
                local selected="$V3_ERRORFOLD_DFUNC"
                [[ "$config" == "errorfold_dpac_v2" ]] && selected="$V3_ERRORFOLD_DPAC"
                export OPENPI_FORMAL_EXPECT_WRAPPED=180
                export QUANTVLA_ADAPTER_ONLY=1
                configure_quant "$FULL_PLAN" "$V3_A8" 180 "$V3_PACK_DIR" 0
                configure_errorfold_v3 "$selected"
                ;;
            quantvla_w4a8_dynamic)
                export OPENPI_FORMAL_EXPECT_WRAPPED=180
                export QUANTVLA_ADAPTER_ONLY=1
                export OPENPI_DUQUANT_HESSIAN_W4_PATH="$V5_HESSIAN_W4"
                configure_quant "$FULL_PLAN" "" 180 "$V5_PACK_DIR" 0 dynamic_a8
                ;;
            quantvla_w4a8_dynamic_profile)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                export QUANTVLA_ADAPTER_ONLY=1
                export OPENPI_DUQUANT_HESSIAN_W4_PATH="$V5_HESSIAN_W4"
                configure_quant "$FULL_PLAN" "" "$gdsq_wrapped" "$V5_PACK_DIR" 0 dynamic_a8
                ;;
            full_context_w4a8_dynamic_profile)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                export QUANTVLA_ADAPTER_ONLY=1
                export OPENPI_DUQUANT_HESSIAN_W4_PATH="$FULL_CONTEXT_HESSIAN_W4"
                configure_quant "$FULL_CONTEXT_PLAN" "" "$gdsq_wrapped" \
                    "$FULL_CONTEXT_PACK_DIR" 0 dynamic_a8 "$FULL_CONTEXT_BUFFER"
                ;;
            quantvla_w4a8_dynamic_profile_errorfold)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                export QUANTVLA_ADAPTER_ONLY=1
                configure_quant "$FULL_PLAN" "" "$gdsq_wrapped" "$V5_PACK_DIR" 0 dynamic_a8
                configure_errorfold "$V5_ERRORFOLD" "$V5_HESSIAN_W4" "$FULL_PLAN"
                ;;
            quantvla_w4a8_softfold_dfunc|quantvla_w4a8_softfold_dpac)
                export OPENPI_FORMAL_EXPECT_WRAPPED=180
                export QUANTVLA_ADAPTER_ONLY=1
                configure_quant "$FULL_PLAN" "$FULL_A8" 180
                configure_atm "$SOFTFOLD_ATM" "$FULL_PLAN" 1 1
                export OPENPI_ATM_APPLICATION=fold_q_weight
                export OPENPI_OHB_APPLICATION=fold_o_weight_perhead
                ;;
            gdsq_vla_atmohb)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$gdsq_wrapped"
                configure_atm "$GDSQ_ATM" "$GDSQ_PLAN" 1 1
                ;;
            gdsq_vla_atm_only)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$gdsq_wrapped"
                configure_atm "$GDSQ_ATM" "$GDSQ_PLAN" 1 0
                ;;
            gdsq_vla_ohb_only)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$gdsq_wrapped"
                configure_atm "$GDSQ_ATM" "$GDSQ_PLAN" 0 1
                ;;
            gdsq_vla_runtime_selector)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                export OPENPI_RUNTIME_SELECTOR_PATH="$RUNTIME_SELECTOR"
                export OPENPI_RUNTIME_SELECTOR_MODEL=pi05
                export OPENPI_RUNTIME_SELECTOR_STRICT=1
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$gdsq_wrapped"
                configure_atm "$GDSQ_ATM" "$GDSQ_PLAN" 1 1
                export OPENPI_ATM_APPLICATION=runtime_query
                export OPENPI_OHB_APPLICATION=runtime_output
                ;;
            gdsq_vla_softfold_dfunc|gdsq_vla_softfold_dpac)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$gdsq_wrapped"
                configure_atm "$GDSQ_ATM" "$GDSQ_PLAN" 1 1
                export OPENPI_ATM_APPLICATION=fold_q_weight
                export OPENPI_OHB_APPLICATION=fold_o_weight_perhead
                ;;
            gdsq_vla)
                export OPENPI_FORMAL_EXPECT_WRAPPED="$gdsq_wrapped"
                configure_quant "$GDSQ_PLAN" "$GDSQ_A8" "$gdsq_wrapped"
                ;;
        esac
        cd "$OPENPI_ROOT"
        exec nohup setsid "$OPENPI_PY" scripts/serve_pi05_quant_policy.py \
            --env ROBOCASA \
            --port "$port" \
            --denoising-steps "$FLOW_STEPS" \
            policy:checkpoint \
            --policy.config pi05_pretrain_human300 \
            --policy.dir "$CHECKPOINT"
    ) >"$log_file" 2>&1 </dev/null &
    local server_pid=$!
    echo "$server_pid" >"$pid_file"

    local attempt
    for attempt in $(seq 1 180); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited during startup: $instance" >&2
            tail -n 120 "$log_file" >&2
            exit 1
        fi
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            if [[ ! -s "$runtime_file" ]]; then
                echo "health endpoint ready but runtime attestation missing: $runtime_file" >&2
                exit 1
            fi
            echo "started instance=$instance config=$config gpu=$gpu port=$port pid=$server_pid"
            return
        fi
        sleep 2
    done
    echo "server startup timed out: $instance" >&2
    tail -n 120 "$log_file" >&2
    exit 1
}

stop_server() {
    if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        usage
        exit 2
    fi
    local instance="$1"
    local pid_file="$CONTROL_DIR/$instance.pid"
    if [[ ! -f "$pid_file" ]]; then
        echo "instance has no pid file: $instance"
        return
    fi
    local pid
    pid="$(<"$pid_file")"
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "invalid pid file: $pid_file" >&2
        exit 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
        local command
        command="$(ps -p "$pid" -o args=)"
        if [[ "$command" != *serve_pi05_quant_policy.py* ]]; then
            echo "refusing to stop unrelated pid=$pid command=$command" >&2
            exit 1
        fi
        local process_group
        process_group="$(ps -p "$pid" -o pgid= | tr -d ' ')"
        if [[ "$process_group" == "$pid" ]]; then
            kill -- "-$pid"
        else
            kill "$pid"
        fi
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            command="$(ps -p "$pid" -o args=)"
            if [[ "$command" != *serve_pi05_quant_policy.py* ]]; then
                echo "refusing KILL escalation for changed pid=$pid command=$command" >&2
                exit 1
            fi
            if [[ "$process_group" == "$pid" ]]; then
                kill -KILL -- "-$pid"
            else
                kill -KILL "$pid"
            fi
            for _ in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
        fi
        if kill -0 "$pid" 2>/dev/null; then
            echo "failed to stop instance=$instance pid=$pid" >&2
            exit 1
        fi
    fi
    rm -f "$pid_file"
    echo "stopped instance=$instance pid=$pid"
}

status_servers() {
    mkdir -p "$CONTROL_DIR"
    local pid_file pid state command
    shopt -s nullglob
    for pid_file in "$CONTROL_DIR"/*.pid; do
        pid="$(<"$pid_file")"
        state="stale"
        command=""
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            state="running"
            command="$(ps -p "$pid" -o args=)"
        fi
        echo "$(basename "$pid_file" .pid) pid=$pid state=$state $command"
    done
}

case "${1:-}" in
    start) shift; start_server "$@" ;;
    stop) shift; stop_server "$@" ;;
    status) shift; status_servers "$@" ;;
    *) usage; exit 2 ;;
esac
