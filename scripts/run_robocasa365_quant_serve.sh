#!/bin/bash
# RoboCasa365 GR00T quantized inference server (v1.4, D-031 criterion-4).
#
# Launches scripts/inference_service.py directly (not the LIBERO-specific
# run_quantvla.sh) with the RoboCasa365 data config + DuQuant plan + pack dir.
#
# Usage:
#   GR00T_GPU=4 GR00T_PORT=5571 GR00T_DUQUANT_PLAN=<plan.json> \
#     ./scripts/run_robocasa365_quant_serve.sh
#   # plan must reference (or be accompanied by) the robocasa365 pack dir:
#   export GR00T_DUQUANT_PACKDIR=checkpoints/packs/robocasa365/duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015
set -euo pipefail

REPO=/home1/gyy/vla/QuantVLA
cd "$REPO"
export PYTHONPATH="$REPO/code:$REPO/scripts/tools:${PYTHONPATH:-}"
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
export CUDA_VISIBLE_DEVICES=${GR00T_GPU:-4}
PORT=${GR00T_PORT:-5571}
DENOISING_STEPS=${GR00T_DENOISING_STEPS:-4}
export GR00T_DENOISING_STEPS="$DENOISING_STEPS"

MODEL_PATH=${GR00T_MODEL_PATH:-$REPO/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000}
DATA_CONFIG=${GR00T_DATA_CONFIG:-examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig}
PACKDIR=${GR00T_DUQUANT_PACKDIR:-$REPO/checkpoints/packs/robocasa365/duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015}
PLAN=${GR00T_DUQUANT_PLAN:-}
ACT_SCALE=${GR00T_DUQUANT_ACT_SCALE_PATH:-}
CONFIG_ID=${GR00T_CONFIG_ID:-}
RUNTIME_SELECTOR=${GR00T_RUNTIME_SELECTOR_PATH:-}

export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE=".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*"
export GR00T_DUQUANT_EXCLUDE="(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_BLOCK_OUT=64
export GR00T_DUQUANT_PERMUTE=0
if [[ -n "${GR00T_DUQUANT_HESSIAN_W4_PATH:-}" ]]; then
    export GR00T_DUQUANT_ROW_ROT=0
else
    export GR00T_DUQUANT_ROW_ROT=${GR00T_DUQUANT_ROW_ROT:-restore}
fi
export GR00T_DUQUANT_ACT_PCT=99.9
export GR00T_DUQUANT_CALIB_STEPS=32
export GR00T_DUQUANT_LS=0.15
export GR00T_DUQUANT_PACKDIR="$PACKDIR"
export GR00T_DUQUANT_ACT_DYNAMIC=0
export GR00T_DUQUANT_FUSED=1
export GR00T_DUQUANT_PRECACHE_WEIGHTS=1
export GR00T_DUQUANT_DEBUG=0
export GR00T_OBS_FORMAT=robocasa365
if [[ -n "$PLAN" ]]; then
    export GR00T_DUQUANT_PLAN="$PLAN"
fi
if [[ -n "$ACT_SCALE" ]]; then
    export GR00T_DUQUANT_ACT_SCALE_PATH="$ACT_SCALE"
fi

if [[ -n "$RUNTIME_SELECTOR" && -n "$CONFIG_ID" && "$CONFIG_ID" != "gdsq_vla_runtime_selector" && "$CONFIG_ID" != "cscka_final_runtime_selector" ]]; then
    echo "static GR00T config $CONFIG_ID cannot inherit GR00T_RUNTIME_SELECTOR_PATH" >&2
    exit 1
fi
if [[ -n "$RUNTIME_SELECTOR" && -z "$CONFIG_ID" ]]; then
    CONFIG_ID=cscka_final_runtime_selector
fi

if [[ "$CONFIG_ID" == "gdsq_vla_runtime_selector" || "$CONFIG_ID" == "cscka_final_runtime_selector" ]]; then
    : "${CONFIG_ID:=cscka_final_runtime_selector}"
    if [[ "$DENOISING_STEPS" != "4" ]]; then
        echo "aligned runtime selector requires GR00T_DENOISING_STEPS=4" >&2
        exit 1
    fi
    export GR00T_CONFIG_ID="$CONFIG_ID"
    : "${GR00T_RUNTIME_SELECTOR_PATH:=$REPO/runs/atmohb_dynamic_selector_v8/selector.json}"
    : "${PLAN:=$REPO/checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json}"
    : "${ACT_SCALE:=$REPO/checkpoints/packs/robocasa365/a8_scales_cscka_16to1_protocolfix_d4.npz}"
    ATM_ARTIFACT=${GR00T_ATM_ALPHA_PATH:-$REPO/checkpoints/packs/robocasa365/atm_alpha_beta_static_cscka_16to1_protocolfix_d4.json}
    for artifact in "$GR00T_RUNTIME_SELECTOR_PATH" "$PLAN" "$ACT_SCALE" "$ATM_ARTIFACT"; do
        if [[ ! -f "$artifact" ]]; then
            echo "missing aligned runtime artifact: $artifact" >&2
            exit 1
        fi
    done
    export GR00T_DUQUANT_PLAN="$PLAN"
    export GR00T_DUQUANT_ACT_SCALE_PATH="$ACT_SCALE"
    export GR00T_RUNTIME_SELECTOR_PATH
    export GR00T_RUNTIME_SELECTOR_MODEL=${GR00T_RUNTIME_SELECTOR_MODEL:-gr00t}
    export GR00T_RUNTIME_SELECTOR_STRICT=${GR00T_RUNTIME_SELECTOR_STRICT:-1}
    export GR00T_ATM_ENABLE=1
    export GR00T_OHB_ENABLE=1
    export GR00T_ATM_ALPHA_PATH="$ATM_ARTIFACT"
    export GR00T_ATM_PER_STEP=0
    export GR00T_ATM_APPLICATION=runtime_query
    export GR00T_OHB_APPLICATION=runtime_output
else
    unset GR00T_RUNTIME_SELECTOR_PATH GR00T_RUNTIME_SELECTOR_MODEL GR00T_RUNTIME_SELECTOR_STRICT
    if [[ -n "$CONFIG_ID" ]]; then
        export GR00T_CONFIG_ID="$CONFIG_ID"
    fi
fi

exec "$PY" scripts/inference_service.py --server \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --embodiment-tag new_embodiment \
    --port "$PORT" \
    --denoising-steps "$DENOISING_STEPS"
