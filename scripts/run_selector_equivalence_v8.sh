#!/usr/bin/env bash
set -euo pipefail

REPO="/home1/gyy/vla/QuantVLA"
OUT_ROOT="${SELECTOR_EQ_OUT_ROOT:-$REPO/runs/gdsq_week1_preregistered_v1/selector_equivalence}"
MANIFEST="$OUT_ROOT/manifest.json"
SELECTOR="$REPO/runs/atmohb_dynamic_selector_v8/selector.json"
EXPECTED_SELECTOR_SHA="0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
GROOT_PY="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"

GR_STATIC_GPU="${GR_STATIC_GPU:-0}"
GR_SELECTOR_GPU="${GR_SELECTOR_GPU:-1}"
PI_STATIC_GPU="${PI_STATIC_GPU:-2}"
PI_SELECTOR_GPU="${PI_SELECTOR_GPU:-3}"
GR_STATIC_PORT="${GR_STATIC_PORT:-19601}"
GR_SELECTOR_PORT="${GR_SELECTOR_PORT:-19602}"
PI_STATIC_PORT="${PI_STATIC_PORT:-19603}"
PI_SELECTOR_PORT="${PI_SELECTOR_PORT:-19604}"

GR_PLAN="$REPO/checkpoints/packs/robocasa365/gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json"
GR_A8="$REPO/checkpoints/packs/robocasa365/a8_scales_cscka_16to1_protocolfix_d4.npz"
GR_PACK="$REPO/checkpoints/packs/robocasa365/duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
PI_CONTROL="$OUT_ROOT/control/pi05"

mkdir -p "$OUT_ROOT/control/gr00t" "$PI_CONTROL" "$OUT_ROOT/logs"
cd "$REPO"

if [[ "$(sha256sum "$SELECTOR" | cut -d' ' -f1)" != "$EXPECTED_SELECTOR_SHA" ]]; then
    echo "frozen v8 selector SHA mismatch" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "freeze the selector-equivalence manifest before server start: $MANIFEST" >&2
    exit 1
fi
for result in "$OUT_ROOT/gr00t.json" "$OUT_ROOT/pi05.json" "$OUT_ROOT/summary.json"; do
    if [[ -e "$result" ]]; then
        echo "refusing to overwrite equivalence result: $result" >&2
        exit 1
    fi
done

gr_pids=()
cleanup_gr00t() {
    local pid command pgid
    for pid in "${gr_pids[@]:-}"; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        kill -0 "$pid" 2>/dev/null || continue
        command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$command" != *run_robocasa365_quant_serve.sh* && "$command" != *inference_service.py* ]]; then
            echo "refusing to stop unrelated GR00T pid=$pid command=$command" >&2
            continue
        fi
        pgid="$(ps -p "$pid" -o pgid= | tr -d ' ')"
        if [[ "$pgid" == "$pid" ]]; then
            kill -- "-$pid" 2>/dev/null || true
        else
            kill "$pid" 2>/dev/null || true
        fi
    done
}

cleanup() {
    set +e
    if [[ "${KEEP_SERVERS:-0}" != "1" ]]; then
        PI05_CONTROL_DIR="$PI_CONTROL" bash scripts/run_pi05_formal_server.sh stop selector_eq_pi_static
        PI05_CONTROL_DIR="$PI_CONTROL" bash scripts/run_pi05_formal_server.sh stop selector_eq_pi_runtime
        cleanup_gr00t
    fi
}
trap cleanup EXIT INT TERM

start_gr00t() {
    local role="$1" gpu="$2" port="$3" config="$4" selector_mode="$5"
    local log="$OUT_ROOT/logs/gr00t_${role}.log"
    (
        export GR00T_GPU="$gpu"
        export GR00T_PORT="$port"
        export GR00T_CONFIG_ID="$config"
        export GR00T_DENOISING_STEPS=4
        export GR00T_DUQUANT_PLAN="$GR_PLAN"
        export GR00T_DUQUANT_ACT_SCALE_PATH="$GR_A8"
        export GR00T_DUQUANT_PACKDIR="$GR_PACK"
        export GR00T_ATM_ENABLE=0
        export GR00T_OHB_ENABLE=0
        if [[ "$selector_mode" == "1" ]]; then
            export GR00T_RUNTIME_SELECTOR_PATH="$SELECTOR"
            export GR00T_RUNTIME_SELECTOR_MODEL=gr00t
            export GR00T_RUNTIME_SELECTOR_STRICT=1
        else
            unset GR00T_RUNTIME_SELECTOR_PATH GR00T_RUNTIME_SELECTOR_MODEL GR00T_RUNTIME_SELECTOR_STRICT
        fi
        exec setsid bash scripts/run_robocasa365_quant_serve.sh
    ) >"$log" 2>&1 </dev/null &
    gr_pids+=("$!")
}

start_gr00t static "$GR_STATIC_GPU" "$GR_STATIC_PORT" cscka_final 0
start_gr00t selector "$GR_SELECTOR_GPU" "$GR_SELECTOR_PORT" cscka_final_runtime_selector 1

PI05_CONTROL_DIR="$PI_CONTROL" bash scripts/run_pi05_formal_server.sh \
    start gdsq_vla_ohb_only "$PI_STATIC_GPU" "$PI_STATIC_PORT" selector_eq_pi_static &
pi_static_start=$!
PI05_CONTROL_DIR="$PI_CONTROL" bash scripts/run_pi05_formal_server.sh \
    start gdsq_vla_runtime_selector "$PI_SELECTOR_GPU" "$PI_SELECTOR_PORT" selector_eq_pi_runtime &
pi_selector_start=$!
wait "$pi_static_start"
wait "$pi_selector_start"

ready=0
for _ in $(seq 1 180); do
    if "$GROOT_PY" scripts/tools/gdsq_selector_equivalence.py \
        --model gr00t --static-port "$GR_STATIC_PORT" --selector-port "$GR_SELECTOR_PORT" \
        --ready-only >/dev/null 2>&1; then
        ready=1
        break
    fi
    for pid in "${gr_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "GR00T server exited during startup; inspect $OUT_ROOT/logs" >&2
            exit 1
        fi
    done
    sleep 2
done
if [[ "$ready" != "1" ]]; then
    echo "GR00T selector-equivalence servers did not become ready" >&2
    exit 1
fi

manifest_sha="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
"$GROOT_PY" scripts/tools/gdsq_selector_equivalence.py \
    --model gr00t --static-port "$GR_STATIC_PORT" --selector-port "$GR_SELECTOR_PORT" \
    --manifest-sha256 "$manifest_sha" --out "$OUT_ROOT/gr00t.json" \
    >"$OUT_ROOT/logs/gr00t_client.log" 2>&1 &
gr_client=$!
"$OPENPI_PY" scripts/tools/gdsq_selector_equivalence.py \
    --model pi05 --static-port "$PI_STATIC_PORT" --selector-port "$PI_SELECTOR_PORT" \
    --manifest-sha256 "$manifest_sha" --out "$OUT_ROOT/pi05.json" \
    >"$OUT_ROOT/logs/pi05_client.log" 2>&1 &
pi_client=$!
wait "$gr_client"
wait "$pi_client"

"$GROOT_PY" scripts/tools/gdsq_selector_equivalence_summary.py \
    --manifest "$MANIFEST" --gr00t "$OUT_ROOT/gr00t.json" --pi05 "$OUT_ROOT/pi05.json" \
    --out "$OUT_ROOT/summary.json"

echo "selector equivalence passed: $OUT_ROOT/summary.json"
