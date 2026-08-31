#!/usr/bin/env bash
set -euo pipefail

# Resumable QuantVLA / Uniform-W6 / compression-anchor portion of Table 6.
# FP16 and Omega-QVLA are handled by run_table6_libero_omega_subset.sh.

ROOT="/home1/gyy/vla/QuantVLA"
OFFICIAL="$ROOT/external/Omega-QVLA"
RUN_ROOT="${TABLE6_RUN_ROOT:-$ROOT/runs/table6_libero_v1}"
RESULTS="$RUN_ROOT/results"
STAGING="$RUN_ROOT/staging"
ARTIFACTS="$RUN_ROOT/artifacts"
CONTROL="$RUN_ROOT/control"
VALIDATOR="$ROOT/scripts/tools/validate_table6_libero_cell.py"
CONDA="/home1/gyy/probe/miniforge3"
DRIVER_PY="$CONDA/envs/groot_test/bin/python"
EVAL_PY="$CONDA/envs/libero_test/bin/python"
OPENPI_PY="$CONDA/envs/openpi/bin/python"
AUDIT_PY="$CONDA/envs/robocasa365/bin/python"
LIBERO_ROOT="$ROOT/code/LIBERO"
OPENPI_ROOT="$ROOT/code/pi05/openpi"
OPENPI_CLIENT="$OPENPI_ROOT/packages/openpi-client/src"
PI_CHECKPOINT="$ROOT/code/pi05/checkpoints/pi05_libero_pytorch"
PI_CHECKPOINT_SHA="0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74"
GPU_OVERRIDE="${TABLE6_GPU_LIST:-auto}"
GPU_UTIL_MAX="${TABLE6_GPU_UTIL_MAX:-100}"
GR00T_W4_PROCS="${TABLE6_GR00T_W4_PROCS_PER_GPU:-2}"
GR00T_W6_PROCS="${TABLE6_GR00T_W6_PROCS_PER_GPU:-2}"
GR00T_SELECTOR_PROCS="${TABLE6_GR00T_SELECTOR_PROCS_PER_GPU:-2}"
PI05_W4_PROCS="${TABLE6_PI05_W4_PROCS_PER_GPU:-2}"
PI05_W6_PROCS="${TABLE6_PI05_W6_PROCS_PER_GPU:-1}"
PI05_SELECTOR_PROCS="${TABLE6_PI05_SELECTOR_PROCS_PER_GPU:-2}"

usage() {
    echo "usage: $0 run | run-cell MODEL CONFIG SUITE | smoke-cell MODEL CONFIG SUITE | status" >&2
}

available_gpu_list() {
    local replicas="$1"
    if [[ "$GPU_OVERRIDE" != auto ]]; then echo "$GPU_OVERRIDE"; return; fi
    [[ "$replicas" =~ ^[1-4]$ ]] || return 2
    local process_mib reserve_mib=3000 cards=() capacities=() values=()
    local index free util copy capacity card_index
    case "$replicas" in
        1) process_mib=30000 ;;
        2) process_mib=18000 ;;
        3) process_mib=13500 ;;
        4) process_mib=10500 ;;
    esac
    while IFS=',' read -r index free util; do
        index="${index//[[:space:]]/}"
        free="${free//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        capacity=$(( (free - reserve_mib) / process_mib ))
        (( capacity > replicas )) && capacity="$replicas"
        if (( capacity > 0 && util <= GPU_UTIL_MAX )); then
            cards+=("$index")
            capacities+=("$capacity")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
    for ((copy = 0; copy < replicas; copy++)); do
        for card_index in "${!cards[@]}"; do
            if (( copy < capacities[card_index] )); then values+=("${cards[$card_index]}"); fi
        done
    done
    [[ ${#values[@]} -gt 0 ]] || {
        echo "no safely free GPU for selected concurrency=$replicas" >&2
        return 1
    }
    local IFS=,
    echo "${values[*]}"
}

replicas_per_gpu() {
    local model="$1" config="$2" value
    case "$model:$config" in
        gr00t:quantvla_w4a8) value="$GR00T_W4_PROCS" ;;
        gr00t:uniform_w6) value="$GR00T_W6_PROCS" ;;
        gr00t:gdsq_vla_selector) value="$GR00T_SELECTOR_PROCS" ;;
        pi05:quantvla_w4a8) value="$PI05_W4_PROCS" ;;
        pi05:uniform_w6) value="$PI05_W6_PROCS" ;;
        pi05:gdsq_vla_selector) value="$PI05_SELECTOR_PROCS" ;;
        *) return 2 ;;
    esac
    [[ "$value" =~ ^[1-4]$ ]] || {
        echo "invalid per-GPU process count for $model/$config: $value" >&2
        return 2
    }
    echo "$value"
}

suite_name() {
    case "$1" in
        goal) echo libero_goal ;;
        spatial) echo libero_spatial ;;
        object) echo libero_object ;;
        long) echo libero_10 ;;
        *) return 2 ;;
    esac
}

gr00t_checkpoint() {
    case "$1" in
        goal) echo "$ROOT/checkpoints/gr00t/libero-goal" ;;
        spatial) echo "$ROOT/checkpoints/gr00t/libero-spatial" ;;
        object) echo "$ROOT/checkpoints/gr00t/libero-object" ;;
        long) echo "$ROOT/checkpoints/gr00t/libero-long" ;;
        *) return 2 ;;
    esac
}

gr00t_pack() {
    local suffix="$1"; [[ "$suffix" == long ]] && suffix=10
    echo "$ROOT/checkpoints/packs/gr00t/duquant_packed_libero_${suffix}_w4a8_b64c32ls015"
}

gr00t_plan() {
    local config="$1" suite="$2"
    case "$config" in
        quantvla_w4a8) echo "$ARTIFACTS/gr00t/$suite/quantvla_w4a8.plan.json" ;;
        uniform_w6)
            if [[ "$suite" == long ]]; then
                echo "$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_long_transfer_w6.json"
            else
                echo "$ROOT/checkpoints/packs/gr00t/baselines_${suite}/uniform_w6.json"
            fi ;;
        gdsq_vla_selector)
            if [[ "$suite" == long ]]; then
                echo "$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_long_transfer_v14.json"
            else
                echo "$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${suite}_v14_adjudicated.final_plan.json"
            fi ;;
        *) return 2 ;;
    esac
}

pi_plan() {
    echo "$ARTIFACTS/pi05/$2/$1.plan.json"
}

valid_cell() {
    "$AUDIT_PY" "$VALIDATOR" "$1/merged_summary.json" >/dev/null 2>&1
}

base_environment() {
    export CONDA_ROOT="$CONDA" GROOT_PY="$DRIVER_PY"
    export LIBERO_CONDA_ENV=libero_test LIBERO_PY="$EVAL_PY" EVAL_PY="$EVAL_PY"
    export LIBERO_ROOT LIBERO_CONFIG_PATH=/home1/gyy/.libero
    export QUANTVLA_CACHE_ROOT="$RUN_ROOT/cache" NUMBA_CACHE_DIR="$RUN_ROOT/numba_cache"
    # Prefer the repository GR00T implementation.  It contains the static-A8
    # runtime helpers used by inference_service.py; the bundled Omega tree is
    # retained later on PYTHONPATH for its evaluator and compatibility code.
    export PYTHONPATH="$ROOT/code:$ROOT/scripts/tools:$ROOT:$OFFICIAL:$OPENPI_CLIENT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export EVAL_SCRIPT="$OFFICIAL/examples/Libero/eval/run_libero_eval.py"
    export EVAL_CWD="$OFFICIAL/examples/Libero/eval"
    export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
    export NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
    export NUM_TRIALS_PER_TASK="${TABLE6_NUM_TRIALS:-10}" GR00T_EVAL_INIT_OFFSET=10
    export DENOISING_STEPS=8 NUM_STEPS_WAIT=10 REPLAN_STEPS=5
    export GR00T_SHARD_STAGGER_MODE=wait_eval_batch GR00T_SHARD_WAIT_EVAL_TIMEOUT_S=300
    export GR00T_SAVE_VIDEOS=0 HEADLESS=1 WBITS=4 ABITS=8 METHOD=duquant
    export GR00T_DYNAMIC_EPISODE_QUEUE=1 GR00T_DYNAMIC_RESUME_QUEUE=1
    export GR00T_DYNAMIC_RESTART_FAILED_WORKERS=1
    export GR00T_DYNAMIC_AUTO_JOIN="${TABLE6_DYNAMIC_AUTO_JOIN:-1}"
    export GR00T_DYNAMIC_GPU_POOL="${TABLE6_DYNAMIC_GPU_POOL:-auto}"
    export GR00T_DYNAMIC_JOIN_UTIL_MAX="$GPU_UTIL_MAX"
    export GR00T_DYNAMIC_JOIN_POLL_S="${TABLE6_DYNAMIC_JOIN_POLL_S:-10}"
    export GR00T_DYNAMIC_STARTS_PER_GPU_SCAN=1
    export GR00T_DYNAMIC_CLAIM_TIMEOUT_S="${TABLE6_DYNAMIC_CLAIM_TIMEOUT_S:-1800}"
    export GR00T_DYNAMIC_QUEUE_DRAIN_TIMEOUT_S=86400
    if [[ -n "${TABLE6_TASK_IDS_OVERRIDE:-}" ]]; then
        export GR00T_TASK_IDS_OVERRIDE="$TABLE6_TASK_IDS_OVERRIDE"
    else
        unset GR00T_TASK_IDS_OVERRIDE 2>/dev/null || true
    fi
}

invoke_gr00t() {
    local config="$1" suite="$2" output="$3" gpu_list="$4"
    local plan pack scale buffer buffer_sha replicas dynamic=0
    plan="$(gr00t_plan "$config" "$suite")"
    pack="$(gr00t_pack "$suite")"
    [[ -f "$plan" && -d "$pack" ]] || { echo "missing GR00T artifacts: $plan $pack" >&2; return 1; }
    if [[ "$config" == gdsq_vla_selector ]]; then dynamic=1; fi
    scale="$ARTIFACTS/gr00t/$suite/${config}.a8.npz"
    buffer="$ARTIFACTS/calibration/$suite.npz"
    if [[ "$dynamic" == 0 && ( ! -f "$scale" || ! -f "$buffer" ) ]]; then
        echo "missing GR00T A8 artifact or calibration buffer: $scale $buffer" >&2
        return 1
    fi
    buffer_sha="$(sha256sum "$buffer" | cut -d' ' -f1)"
    (
        base_environment
        replicas="$(replicas_per_gpu gr00t "$config")"
        export GR00T_DYNAMIC_MAX_PROCS_PER_GPU="$replicas"
        export GR00T_DYNAMIC_PROCESS_MIB=17000 GR00T_DYNAMIC_RESERVE_MIB=2000
        export QUANTVLA_ROOT="$ROOT" QUANTVLA_CONDA_ENV=groot_test
        export GR00T_SHARD_STARTS_PER_GPU="${TABLE6_GR00T_QUANT_STARTS_PER_GPU:-2}"
        export INFERENCE_PY="$DRIVER_PY" INFERENCE_SCRIPT="$ROOT/scripts/inference_service.py"
        export INFERENCE_CWD="$ROOT" OUTPUT_ROOT="$output"
        export TASK_SUITE="$(suite_name "$suite")" SUITE="$suite" GPU_LIST="$gpu_list"
        export PORT_BASE="${TABLE6_PORT_BASE:-19400}"
        export MODEL_PATH="$(gr00t_checkpoint "$suite")"
        export DATA_CONFIG=examples.Libero.custom_data_config:LiberoDataConfig
        [[ "$suite" == goal ]] && export DATA_CONFIG=examples.Libero.custom_data_config:LiberoDataConfigMeanStd
        export PACKDIR="$pack" BENCHMARK_LABEL="${config}_gr00t_${suite}"
        export GR00T_DUQUANT_PLAN="$plan" GR00T_DUQUANT_PACKDIR="$pack"
        export GR00T_DUQUANT_PLAN_STRICT=1 GR00T_DUQUANT_WBITS_DEFAULT=4
        export GR00T_DUQUANT_ABITS=8 GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64
        export GR00T_DUQUANT_PERMUTE=0 GR00T_DUQUANT_ROW_ROT=restore
        export GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32
        export GR00T_DUQUANT_LS=0.15 GR00T_DUQUANT_ACT_DYNAMIC="$dynamic"
        export GR00T_DUQUANT_CALIB_BUFFER_PATH="$buffer"
        export GR00T_DUQUANT_CALIB_BUFFER_SHA256="$buffer_sha"
        if [[ "$dynamic" == 1 ]]; then
            unset GR00T_DUQUANT_ACT_SCALE_PATH 2>/dev/null || true
        else
            export GR00T_DUQUANT_ACT_SCALE_PATH="$scale"
        fi
        "$DRIVER_PY" -u "$OFFICIAL/scripts/run_libero_duquant_benchmark_multi_gpu.py" \
            > "$output/logs/run.log" 2>&1
    )
}

invoke_pi05() {
    local config="$1" suite="$2" output="$3" gpu_list="$4"
    local plan scale buffer pack manifest replicas process_mib dynamic=0 fast=1 adapter=1
    plan="$(pi_plan "$config" "$suite")"
    buffer="$ARTIFACTS/calibration/${suite}.npz"
    pack="$ARTIFACTS/pi05/pack_block64"
    manifest="$pack/manifest.json"
    [[ -f "$plan" && -f "$buffer" && -f "$manifest" ]] || {
        echo "missing pi0.5 artifacts: $plan $buffer $manifest" >&2; return 1
    }
    scale="$ARTIFACTS/pi05/$suite/${config}.a8.npz"
    if [[ "$config" == gdsq_vla_selector ]]; then dynamic=1; fi
    if [[ "$config" == uniform_w6 ]]; then fast=0; adapter=0; fi
    if [[ "$dynamic" == 0 && ! -f "$scale" ]]; then
        echo "missing pi0.5 A8 artifact: $scale" >&2; return 1
    fi
    local buffer_sha manifest_sha
    buffer_sha="$(sha256sum "$buffer" | cut -d' ' -f1)"
    manifest_sha="$(sha256sum "$manifest" | cut -d' ' -f1)"
    (
        base_environment
        replicas="$(replicas_per_gpu pi05 "$config")"
        process_mib=9500
        # Uniform W6 disables the Triton/adapter-only fast path and peaks at
        # roughly 23.2 GiB during layer wrapping.  Admit only one instance on
        # a 46-GiB card with a real safety margin.
        [[ "$config" == uniform_w6 ]] && process_mib=26000
        export GR00T_DYNAMIC_MAX_PROCS_PER_GPU="$replicas"
        export GR00T_DYNAMIC_PROCESS_MIB="$process_mib" GR00T_DYNAMIC_RESERVE_MIB=2000
        export QUANTVLA_ROOT="$ROOT" QUANTVLA_CONDA_ENV=groot_test
        if [[ "$config" == uniform_w6 ]]; then
            export GR00T_SHARD_STARTS_PER_GPU="${TABLE6_PI05_W6_STARTS_PER_GPU:-1}"
        else
            export GR00T_SHARD_STARTS_PER_GPU="${TABLE6_PI05_QUANT_STARTS_PER_GPU:-2}"
        fi
        export INFERENCE_PY="$OPENPI_PY" INFERENCE_SCRIPT="$ROOT/scripts/table6_pi05_inference_service.py"
        export INFERENCE_CWD="$OPENPI_ROOT"
        export INFERENCE_EXTRA_ARGS_JSON='["--device","cuda"]'
        export EVAL_EXTRA_ARGS_JSON='["--policy-backend","openpi_ws","--max-steps-profile","openpi","--replan-steps","5","--paired-action-noise","--action-noise-stream","A"]'
        export OUTPUT_ROOT="$output" TASK_SUITE="$(suite_name "$suite")" GPU_LIST="$gpu_list"
        export PORT_BASE="${TABLE6_PORT_BASE:-19500}"
        export MODEL_PATH="$PI_CHECKPOINT" DATA_CONFIG=pi05_libero
        export PACKDIR="$pack" BENCHMARK_LABEL="${config}_pi05_${suite}"
        export OPENPI_MODEL_DTYPE=float16 OPENPI_CONFIG_ID="$config"
        export OPENPI_CHECKPOINT_SHA256="$PI_CHECKPOINT_SHA"
        export OPENPI_DUQUANT_PLAN="$plan" OPENPI_DUQUANT_PLAN_STRICT=1
        export OPENPI_DUQUANT_WBITS_DEFAULT=4 OPENPI_DUQUANT_ABITS=8
        export OPENPI_DUQUANT_BLOCK=64 OPENPI_DUQUANT_BLOCK_OUT=64
        export OPENPI_DUQUANT_EXPECT_BLOCK=64 OPENPI_DUQUANT_LS=0.15
        export OPENPI_DUQUANT_PERMUTE=0 OPENPI_DUQUANT_ROW_ROT=restore
        export OPENPI_DUQUANT_ACT_PCT=99.9 OPENPI_DUQUANT_CALIB_STEPS=32
        export OPENPI_DUQUANT_DENOISING_STEPS=8 OPENPI_DUQUANT_PACKDIR="$pack"
        export OPENPI_DUQUANT_ACT_DYNAMIC="$dynamic" OPENPI_DUQUANT_TRITON="$fast"
        export OPENPI_DUQUANT_PRECACHE_WEIGHTS=1 OPENPI_DUQUANT_QUIET=1
        export OPENPI_DUQUANT_STRICT_ARTIFACTS=1
        export OPENPI_DUQUANT_CALIB_BUFFER_SHA256="$buffer_sha"
        export OPENPI_DUQUANT_PACK_MANIFEST_SHA256="$manifest_sha"
        export QUANTVLA_ADAPTER_ONLY="$adapter"
        if [[ "$dynamic" == 1 ]]; then
            export OPENPI_DUQUANT_REQUIRE_ACT_SCALE=0
            unset OPENPI_DUQUANT_ACT_SCALE_PATH 2>/dev/null || true
        else
            export OPENPI_DUQUANT_REQUIRE_ACT_SCALE=1
            export OPENPI_DUQUANT_ACT_SCALE_PATH="$scale"
        fi
        "$DRIVER_PY" -u "$OFFICIAL/scripts/run_libero_duquant_benchmark_multi_gpu.py" \
            > "$output/logs/run.log" 2>&1
    )
}

run_cell() {
    local model="$1" config="$2" suite="$3" replicas gpu_list final stage cell_lock cell_lock_fd final_in_place=0
    [[ "$model" == gr00t || "$model" == pi05 ]] || return 2
    [[ "$config" == quantvla_w4a8 || "$config" == uniform_w6 || "$config" == gdsq_vla_selector ]] || return 2
    suite_name "$suite" >/dev/null
    # An overlap runner may start paper-method cells while the Omega subset is
    # still draining.  Serialize each formal cell independently so the later
    # unattended pass waits and reuses its validated result instead of
    # launching a duplicate evaluation.
    mkdir -p "$CONTROL/cell_locks"
    cell_lock="$CONTROL/cell_locks/${model}_${config}_${suite}.lock"
    exec {cell_lock_fd}>"$cell_lock"
    flock "$cell_lock_fd"
    final="$RESULTS/$model/$config/$suite"
    if valid_cell "$final"; then echo "[table6] reuse $model/$config/$suite"; return; fi
    replicas="$(replicas_per_gpu "$model" "$config")"
    mkdir -p "$STAGING/$model/$config"
    stage=""
    if [[ -d "$final" ]]; then
        # A prior supervisor may have been interrupted after creating the
        # destination.  Resume its exact episode queue in place so absolute
        # committed-summary paths remain valid.
        stage="$final"
        final_in_place=1
        echo "[table6] resume incomplete final cell $stage"
    else
        while IFS= read -r candidate; do
            if valid_cell "$candidate"; then
                mkdir -p "$(dirname "$final")"
                mv "$candidate" "$final"
                echo "[table6] promoted recovered staging cell $model/$config/$suite"
                return
            fi
            if [[ -d "$candidate" ]]; then
                stage="$candidate"
                break
            fi
        done < <(
            find "$STAGING/$model/$config" -mindepth 1 -maxdepth 1 \
                -type d -name "${suite}.*" -printf '%T@ %p\n' 2>/dev/null \
                | sort -k1,1nr | cut -d' ' -f2-
        )
    fi
    if [[ -n "$stage" ]]; then
        echo "[table6] resume staging cell $stage"
    else
        stage="$(mktemp -d "$STAGING/$model/$config/${suite}.XXXXXX")"
    fi
    mkdir -p "$stage/logs"
    gpu_list="$(available_gpu_list "$replicas")"
    echo "[table6] $model/$config/$suite GPUs=$gpu_list"
    if [[ "$model" == gr00t ]]; then
        invoke_gr00t "$config" "$suite" "$stage" "$gpu_list"
    else
        invoke_pi05 "$config" "$suite" "$stage" "$gpu_list"
    fi
    "$AUDIT_PY" "$VALIDATOR" "$stage/merged_summary.json"
    if (( final_in_place == 0 )); then
        mkdir -p "$(dirname "$final")"
        mv "$stage" "$final"
        echo "[table6] promoted $model/$config/$suite"
    else
        echo "[table6] completed resumed final $model/$config/$suite"
    fi
}

run_all() {
    mkdir -p "$CONTROL"
    exec 9>"$CONTROL/quant_subset.lock"
    flock -n 9 || { echo "Table 6 quant subset already running" >&2; return 1; }
    local model config suite
    # DyPAC-VLA is completed by the dedicated cross-benchmark pipeline first.
    # Run the remaining Table-5 baselines in the requested order, with pi0.5
    # preceding GR00T for each configuration.
    for config in uniform_w6 quantvla_w4a8; do
        for model in pi05 gr00t; do
            for suite in goal spatial object long; do
                run_cell "$model" "$config" "$suite"
            done
        done
    done
}

smoke_cell() {
    local model="$1" config="$2" suite="$3" tasks="${TABLE6_SMOKE_TASKS:-2}"
    local trials="${TABLE6_NUM_TRIALS:-1}" tag="${TABLE6_SMOKE_TAG:-default}"
    local output="$RUN_ROOT/preflight/quant_${model}_${config}_${suite}_${tag}"
    local gpu_list
    [[ "$model" == gr00t || "$model" == pi05 ]] || return 2
    [[ "$config" == quantvla_w4a8 || "$config" == uniform_w6 || "$config" == gdsq_vla_selector ]] || return 2
    if "$AUDIT_PY" "$VALIDATOR" "$output/merged_summary.json" \
        --tasks "$tasks" --trials "$trials" --offset 10 >/dev/null 2>&1; then
        echo "[table6 smoke] reuse $model/$config/$suite"
        return
    fi
    if [[ -e "$output" ]]; then
        echo "incomplete smoke retained: $output" >&2
        return 1
    fi
    gpu_list="$(available_gpu_list 1)"
    mkdir -p "$output/logs"
    if [[ "$model" == gr00t ]]; then
        invoke_gr00t "$config" "$suite" "$output" "$gpu_list"
    else
        invoke_pi05 "$config" "$suite" "$output" "$gpu_list"
    fi
    "$AUDIT_PY" "$VALIDATOR" "$output/merged_summary.json" \
        --tasks "$tasks" --trials "$trials" --offset 10
}

status_all() {
    local complete=0 total=0 episodes=0 model config suite
    for model in gr00t pi05; do
        for config in quantvla_w4a8 uniform_w6 gdsq_vla_selector; do
            for suite in goal spatial object long; do
                total=$((total + 1))
                if valid_cell "$RESULTS/$model/$config/$suite"; then
                    complete=$((complete + 1)); episodes=$((episodes + 100))
                fi
            done
        done
    done
    echo "table6_quant_subset cells=$complete/$total episodes=$episodes/2400 gpu_policy=$GPU_OVERRIDE gr00t_procs=$GR00T_W4_PROCS/$GR00T_W6_PROCS/$GR00T_SELECTOR_PROCS pi05_procs=$PI05_W4_PROCS/$PI05_W6_PROCS/$PI05_SELECTOR_PROCS"
}

cd "$ROOT"
case "${1:-}" in
    run) run_all ;;
    run-cell) [[ $# -eq 4 ]] || { usage; exit 2; }; run_cell "$2" "$3" "$4" ;;
    smoke-cell) [[ $# -eq 4 ]] || { usage; exit 2; }; smoke_cell "$2" "$3" "$4" ;;
    status) status_all ;;
    *) usage; exit 2 ;;
esac
