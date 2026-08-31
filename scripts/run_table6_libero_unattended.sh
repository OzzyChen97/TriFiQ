#!/usr/bin/env bash
set -euo pipefail

# Unattended completion pipeline for the preregistered 4,000-episode LIBERO
# Table 6.  It prepares result-blind artifacts while the FP16/Omega subset is
# running, gates each quantized runtime with the repository's fast multi-GPU
# evaluator, then aggregates, builds, and syncs the audited paper.

ROOT="/home1/gyy/vla/QuantVLA"
RUN_ROOT="${TABLE6_RUN_ROOT:-$ROOT/runs/table6_libero_v1}"
ARTIFACTS="$RUN_ROOT/artifacts"
CONTROL="$RUN_ROOT/control"
LOGS="$CONTROL/unattended_logs"
RESULTS="$RUN_ROOT/results"
OFFICIAL="$ROOT/external/Omega-QVLA"
LIBERO_ROOT="$ROOT/code/LIBERO"
OPENPI_ROOT="$ROOT/code/pi05/openpi"
CONDA="/home1/gyy/probe/miniforge3"
LIBERO_PY="$CONDA/envs/libero_test/bin/python"
OPENPI_PY="$CONDA/envs/openpi/bin/python"
GR00T_PY="$CONDA/envs/groot_test/bin/python"
AUDIT_PY="$CONDA/envs/robocasa365/bin/python"
PI_CHECKPOINT="$ROOT/code/pi05/checkpoints/pi05_libero_pytorch"
PI_CHECKPOINT_SHA="0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74"
PI_PACK="$ARTIFACTS/pi05/pack_block64"
VALIDATOR="$ROOT/scripts/tools/validate_table6_libero_cell.py"
STATUS="$CONTROL/unattended_status.json"
PACK_PIDS=()
FREE_GPUS=()
GPU_UTIL_MAX="${TABLE6_GPU_UTIL_MAX:-100}"

mkdir -p "$CONTROL" "$LOGS" "$ARTIFACTS"
cd "$ROOT"

phase() {
    local name="$1" detail="${2:-}"
    "$AUDIT_PY" - "$STATUS" "$name" "$detail" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "phase": sys.argv[2],
    "detail": sys.argv[3],
    "pid": os.getppid(),
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
temporary = pathlib.Path(str(path) + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
    echo "[table6 unattended] phase=$name detail=$detail"
}

on_error() {
    local code="$1" line="$2"
    set +e
    phase failed "exit=$code line=$line"
    exit "$code"
}
trap 'on_error $? $LINENO' ERR

wait_for_pids() {
    local failed=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then failed=1; fi
    done
    (( failed == 0 ))
}

pack_manifest_valid() {
    "$OPENPI_PY" - "$PI_PACK/manifest.json" "$PI_CHECKPOINT_SHA" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
assert value["complete"] is True
assert value["checkpoint_sha256"] == sys.argv[2]
assert value["wrapped_layer_count"] == 180
assert value["block_in"] == value["block_out"] == 64
assert value["enable_permute"] is False
assert len(value["files"]) == 180
assert all((path.parent / row["file"]).is_file() for row in value["files"])
PY
}

start_pi_pack() {
    if pack_manifest_valid; then
        echo "[table6 unattended] reuse complete pi0.5 block64 pack"
        return
    fi
    phase prepare_pi_pack "six CPU shards"
    mkdir -p "$PI_PACK"
    local shard
    for shard in 0 1 2 3 4 5; do
        (
            export CUDA_VISIBLE_DEVICES=""
            export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
            "$OPENPI_PY" "$ROOT/scripts/tools/pi05_build_block64_pack.py" \
                --checkpoint "$PI_CHECKPOINT/model.safetensors" \
                --checkpoint-sha256 "$PI_CHECKPOINT_SHA" \
                --out "$PI_PACK" --shard-index "$shard" --num-shards 6 \
                --block 64 --lambda-smooth 0.15 --no-permute \
                >"$LOGS/pi_pack_shard_${shard}.log" 2>&1
        ) &
        PACK_PIDS+=("$!")
    done
}

finish_pi_pack() {
    if pack_manifest_valid; then return; fi
    phase finish_pi_pack "waiting for six CPU shards"
    wait_for_pids "${PACK_PIDS[@]}"
    "$OPENPI_PY" "$ROOT/scripts/tools/pi05_build_block64_pack.py" \
        --checkpoint "$PI_CHECKPOINT/model.safetensors" \
        --checkpoint-sha256 "$PI_CHECKPOINT_SHA" \
        --out "$PI_PACK" --num-shards 6 --block 64 \
        --lambda-smooth 0.15 --no-permute --finalize \
        >"$LOGS/pi_pack_finalize.log" 2>&1
    pack_manifest_valid
}

wait_for_omega_subset() {
    local ticks=0
    phase wait_omega_subset "formal FP16 and Omega-QVLA cells"
    while tmux has-session -t table6_libero_omega 2>/dev/null; do
        sleep 60
        ticks=$((ticks + 1))
        if (( ticks % 10 == 0 )); then
            bash "$ROOT/scripts/run_table6_libero_omega_subset.sh" status \
                >>"$LOGS/omega_wait_status.log" 2>&1 || true
        fi
    done
    local model config suite
    for model in gr00t pi05; do
        for config in fp16 omega_qvla_w4a4; do
            for suite in goal spatial object long; do
                "$AUDIT_PY" "$VALIDATOR" \
                    "$RESULTS/$model/$config/$suite/merged_summary.json" \
                    >>"$LOGS/omega_exact_validation.log" 2>&1
            done
        done
    done
    phase omega_subset_validated "16 cells and 1600 episodes"
}

refresh_free_gpus() {
    local minimum_free="$1" index free util
    FREE_GPUS=()
    while IFS=',' read -r index free util; do
        index="${index//[[:space:]]/}"
        free="${free//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        if (( free >= minimum_free && util <= GPU_UTIL_MAX )); then FREE_GPUS+=("$index"); fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
}

wait_for_free_gpus() {
    local minimum_free="$1"
    while true; do
        refresh_free_gpus "$minimum_free"
        if (( ${#FREE_GPUS[@]} > 0 )); then
            echo "[table6 unattended] free GPUs: ${FREE_GPUS[*]}"
            return
        fi
        phase wait_free_gpu "need one card with ${minimum_free} MiB free"
        sleep 60
    done
}

calibration_valid() {
    "$AUDIT_PY" - "$1" <<'PY' >/dev/null 2>&1
import hashlib
import json
import numpy as np
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sidecar = json.loads(pathlib.Path(str(path) + ".json").read_text())
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert sidecar["sha256"] == digest
assert sidecar["rows"] == 256 and sidecar["tasks"] == 10
assert sidecar["initial_state_indices"] == list(range(5))
assert sidecar["held_out_initial_state_indices"] == list(range(10, 20))
assert sidecar["overlap_with_held_out"] is False
assert sidecar["policy_queries"] == 0
assert sidecar["test_rollout_feedback_used"] is False
with np.load(path, allow_pickle=False) as archive:
    assert archive["states"].shape == (256, 8)
    assert archive["action_noises"].shape == (256, 50, 32)
PY
}

collect_calibration_buffers() {
    phase collect_calibration "four result-blind LIBERO suites"
    local suites=(goal spatial object long) suite
    local all_valid=1
    for suite in "${suites[@]}"; do
        if ! calibration_valid "$ARTIFACTS/calibration/$suite.npz"; then
            all_valid=0
            break
        fi
    done
    if (( all_valid == 1 )); then
        echo "[table6 unattended] reuse all four calibration buffers"
        return
    fi
    wait_for_free_gpus 30000
    local pids=() index gpu out
    for index in "${!suites[@]}"; do
        suite="${suites[$index]}"
        gpu="${FREE_GPUS[$((index % ${#FREE_GPUS[@]}))]}"
        out="$ARTIFACTS/calibration/$suite.npz"
        if calibration_valid "$out"; then
            echo "[table6 unattended] reuse calibration $suite"
            continue
        fi
        if [[ -e "$out" || -e "$out.json" ]]; then
            echo "invalid existing calibration artifact retained: $out" >&2
            return 1
        fi
        (
            export PYTHONPATH="$LIBERO_ROOT:$OFFICIAL${PYTHONPATH:+:$PYTHONPATH}"
            export LIBERO_CONFIG_PATH=/home1/gyy/.libero
            export NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
            "$LIBERO_PY" "$ROOT/scripts/tools/collect_table6_libero_calibration.py" \
                --suite "$suite" --egl-device "$gpu" --out "$out" \
                >"$LOGS/calibration_${suite}.log" 2>&1
        ) &
        pids+=("$!")
    done
    if (( ${#pids[@]} > 0 )); then wait_for_pids "${pids[@]}"; fi
    for suite in "${suites[@]}"; do calibration_valid "$ARTIFACTS/calibration/$suite.npz"; done
}

selector_valid() {
    "$OPENPI_PY" - "$1" "$2" "$PI_CHECKPOINT_SHA" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text())
meta = value["meta"]
assert meta["suite"] == sys.argv[2]
assert meta["checkpoint_sha256"] == sys.argv[3]
assert meta["candidate_layers"] == len(value["layers"]) == 180
assert meta["uses_task_success"] is False
assert meta["uses_test_rollout_feedback"] is False
assert meta["uses_cka"] is False and meta["uses_cs"] is False
assert meta["held_out_initial_state_indices"] == list(range(10, 20))
PY
}

select_pi05_masks() {
    phase select_pi05_masks "result-blind compression-anchor selection"
    local suites=(goal spatial object long) suite
    local all_valid=1
    for suite in "${suites[@]}"; do
        if ! selector_valid "$ARTIFACTS/pi05/$suite/gdsq_vla_selector.plan.json" "$suite"; then
            all_valid=0
            break
        fi
    done
    if (( all_valid == 1 )); then
        echo "[table6 unattended] reuse all four pi0.5 selector plans"
        return
    fi
    wait_for_free_gpus 30000
    local pids=() index gpu out
    for index in "${!suites[@]}"; do
        suite="${suites[$index]}"
        gpu="${FREE_GPUS[$((index % ${#FREE_GPUS[@]}))]}"
        out="$ARTIFACTS/pi05/$suite/gdsq_vla_selector.plan.json"
        if selector_valid "$out" "$suite"; then
            echo "[table6 unattended] reuse pi0.5 selector $suite"
            continue
        fi
        if [[ -e "$out" ]]; then
            echo "invalid existing selector retained: $out" >&2
            return 1
        fi
        (
            export CUDA_VISIBLE_DEVICES="$gpu" OPENPI_MODEL_DTYPE=float16
            export TORCHDYNAMO_DISABLE=1 PYTHONNOUSERSITE=1
            "$OPENPI_PY" "$ROOT/scripts/tools/select_table6_pi05_libero_plan.py" \
                --buffer "$ARTIFACTS/calibration/$suite.npz" \
                --suite "$suite" --out "$out" \
                >"$LOGS/pi_selector_${suite}.log" 2>&1
        ) &
        pids+=("$!")
    done
    if (( ${#pids[@]} > 0 )); then wait_for_pids "${pids[@]}"; fi
    for suite in "${suites[@]}"; do selector_valid "$ARTIFACTS/pi05/$suite/gdsq_vla_selector.plan.json" "$suite"; done
}

gr00t_checkpoint() {
    case "$1" in
        goal) echo "$ROOT/checkpoints/gr00t/libero-goal" ;;
        spatial) echo "$ROOT/checkpoints/gr00t/libero-spatial" ;;
        object) echo "$ROOT/checkpoints/gr00t/libero-object" ;;
        long) echo "$ROOT/checkpoints/gr00t/libero-long" ;;
    esac
}

gr00t_pack() {
    local suffix="$1"; [[ "$suffix" == long ]] && suffix=10
    echo "$ROOT/checkpoints/packs/gr00t/duquant_packed_libero_${suffix}_w4a8_b64c32ls015"
}

gr00t_plan() {
    local suite="$1" config="$2"
    if [[ "$config" == quantvla_w4a8 ]]; then
        echo "$ARTIFACTS/gr00t/$suite/quantvla_w4a8.plan.json"
    elif [[ "$suite" == long ]]; then
        echo "$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_long_transfer_w6.json"
    else
        echo "$ROOT/checkpoints/packs/gr00t/baselines_${suite}/uniform_w6.json"
    fi
}

a8_valid() {
    local model="$1" suite="$2" config="$3" out sidecar plan buffer
    out="$ARTIFACTS/$model/$suite/${config}.a8.npz"
    buffer="$ARTIFACTS/calibration/$suite.npz"
    if [[ "$model" == gr00t ]]; then
        sidecar="${out}.meta.json"
        plan="$(gr00t_plan "$suite" "$config")"
    else
        sidecar="${out}.json"
        plan="$ARTIFACTS/pi05/$suite/${config}.plan.json"
    fi
    "$AUDIT_PY" - "$model" "$suite" "$out" "$sidecar" "$plan" "$buffer" "$PI_CHECKPOINT_SHA" <<'PY' >/dev/null 2>&1
import hashlib
import json
import pathlib
import sys

import numpy as np

model, suite = sys.argv[1:3]
out, sidecar, plan, buffer = map(pathlib.Path, sys.argv[3:7])
checkpoint_sha = sys.argv[7]
assert out.is_file() and sidecar.is_file() and plan.is_file() and buffer.is_file()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata = json.loads(sidecar.read_text(encoding="utf-8"))
with np.load(out, allow_pickle=False) as archive:
    names = list(archive.files)
    assert names
    if model == "gr00t":
        assert metadata["suite"] == suite
        assert metadata["plan_sha256"] == sha(plan)
        assert metadata["calibration_buffer_sha256"] == sha(buffer)
        assert metadata["test_rollout_feedback_used"] is False
        assert int(metadata["wrapped_layers"]) == len(names)
    else:
        assert metadata["schema_version"] == 1
        assert metadata["npz_sha256"] == sha(out)
        meta = metadata["metadata"]
        assert meta["plan_sha256"] == sha(plan)
        assert meta["checkpoint_sha256"] == checkpoint_sha
        assert meta["calibration_buffer_sha256"] == sha(buffer)
        layer_names = [str(value) for value in archive["layer_names"].tolist()]
        assert layer_names == metadata["layer_names"]
        assert int(meta["wrapped_layers"]) == len(layer_names)
        assert len([name for name in names if name.startswith("scale_")]) == len(layer_names)
PY
}

all_a8_valid() {
    local model suite config
    for model in gr00t pi05; do
        for config in uniform_w6 quantvla_w4a8; do
            for suite in goal spatial object long; do
                a8_valid "$model" "$suite" "$config" || return 1
            done
        done
    done
}

archive_invalid_a8() {
    local model="$1" suite="$2" config="$3" out sidecar archive stamp candidate
    out="$ARTIFACTS/$model/$suite/${config}.a8.npz"
    if [[ "$model" == gr00t ]]; then
        sidecar="${out}.meta.json"
    else
        sidecar="${out}.json"
    fi
    if [[ ! -e "$out" && ! -e "$sidecar" && ! -e "${out}.tmp" ]]; then return; fi
    stamp="$(date -u +%Y%m%dT%H%M%S.%N)"
    archive="$CONTROL/invalid_a8/${model}_${suite}_${config}_${stamp}"
    mkdir -p "$archive"
    for candidate in "$out" "$sidecar" "${out}.tmp" "${sidecar}.tmp"; do
        [[ -e "$candidate" ]] && mv "$candidate" "$archive/"
    done
    echo "[table6 unattended] archived incomplete A8 artifact -> $archive"
}

run_gr00t_a8() {
    local gpu="$1" suite="$2" config="$3"
    local out="$ARTIFACTS/gr00t/$suite/${config}.a8.npz" lock lock_fd
    mkdir -p "$CONTROL/a8_locks"
    lock="$CONTROL/a8_locks/gr00t_${suite}_${config}.lock"
    exec {lock_fd}>"$lock"
    flock "$lock_fd"
    if a8_valid gr00t "$suite" "$config"; then
        echo "[table6 unattended] reuse GR00T A8 $suite/$config"
        return
    fi
    archive_invalid_a8 gr00t "$suite" "$config"
    mkdir -p "$(dirname "$out")"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        # Prefer the repository's patched GR00T package.  The upstream
        # Omega-QVLA copy lacks the static-A8 persistence helpers used by the
        # Table 6 calibrator.
        export PYTHONPATH="$ROOT/code:$ROOT/scripts/tools:$OFFICIAL${PYTHONPATH:+:$PYTHONPATH}"
        export LIBERO_CONFIG_PATH=/home1/gyy/.libero
        export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
        export NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
        "$GR00T_PY" "$ROOT/scripts/tools/calibrate_table6_gr00t_a8.py" \
            --model-path "$(gr00t_checkpoint "$suite")" \
            --plan "$(gr00t_plan "$suite" "$config")" \
            --pack-dir "$(gr00t_pack "$suite")" \
            --buffer "$ARTIFACTS/calibration/$suite.npz" \
            --suite "$suite" --out "$out"
    ) >"$LOGS/a8_gr00t_${suite}_${config}.log" 2>&1
    a8_valid gr00t "$suite" "$config"
}

run_pi05_a8() {
    local gpu="$1" suite="$2" config="$3"
    local out="$ARTIFACTS/pi05/$suite/${config}.a8.npz" lock lock_fd
    mkdir -p "$CONTROL/a8_locks"
    lock="$CONTROL/a8_locks/pi05_${suite}_${config}.lock"
    exec {lock_fd}>"$lock"
    flock "$lock_fd"
    if a8_valid pi05 "$suite" "$config"; then
        echo "[table6 unattended] reuse pi0.5 A8 $suite/$config"
        return
    fi
    archive_invalid_a8 pi05 "$suite" "$config"
    mkdir -p "$(dirname "$out")"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
        export PYTHONNOUSERSITE=1
        "$OPENPI_PY" "$ROOT/scripts/tools/pi05_calibrate_a8.py" \
            --checkpoint-dir "$PI_CHECKPOINT" --config pi05_libero \
            --checkpoint-sha256 "$PI_CHECKPOINT_SHA" \
            --plan "$ARTIFACTS/pi05/$suite/${config}.plan.json" \
            --pack-dir "$PI_PACK" --buffer "$ARTIFACTS/calibration/$suite.npz" \
            --out "$out" --expected-wrapped 180 --n-frames 256 \
            --batch-size 8 --flow-steps 8 --no-permute
    ) >"$LOGS/a8_pi05_${suite}_${config}.log" 2>&1
    a8_valid pi05 "$suite" "$config"
}

run_a8_worker() {
    local worker="$1" workers="$2" gpu="$3" index model suite config
    local jobs=(
        gr00t:goal:quantvla_w4a8 gr00t:spatial:quantvla_w4a8
        gr00t:object:quantvla_w4a8 gr00t:long:quantvla_w4a8
        gr00t:goal:uniform_w6 gr00t:spatial:uniform_w6
        gr00t:object:uniform_w6 gr00t:long:uniform_w6
        pi05:goal:quantvla_w4a8 pi05:spatial:quantvla_w4a8
        pi05:object:quantvla_w4a8 pi05:long:quantvla_w4a8
        pi05:goal:uniform_w6 pi05:spatial:uniform_w6
        pi05:object:uniform_w6 pi05:long:uniform_w6
    )
    for ((index=worker; index<${#jobs[@]}; index+=workers)); do
        IFS=: read -r model suite config <<<"${jobs[$index]}"
        if [[ "$model" == gr00t ]]; then
            run_gr00t_a8 "$gpu" "$suite" "$config"
        else
            run_pi05_a8 "$gpu" "$suite" "$config"
        fi
    done
}

calibrate_static_a8() {
    phase calibrate_static_a8 "16 plan-specific artifacts"
    if all_a8_valid; then
        echo "[table6 unattended] reuse all 16 static-A8 artifacts"
        return
    fi
    # pi0.5 calibration peaks near 30.5 GiB while building its 180 wrapped
    # layers.  Require a measured safety margin so a later job in a worker's
    # sequence does not OOM after passing a GR00T-sized admission threshold.
    wait_for_free_gpus "${TABLE6_A8_MIN_FREE_MIB:-32000}"
    local workers="${#FREE_GPUS[@]}" pids=() worker
    (( workers > 6 )) && workers=6
    for ((worker=0; worker<workers; worker++)); do
        run_a8_worker "$worker" "$workers" "${FREE_GPUS[$worker]}" &
        pids+=("$!")
    done
    wait_for_pids "${pids[@]}"
    all_a8_valid
}

repeat_csv() {
    local value="$1" count="$2" result="" index
    for ((index=0; index<count; index++)); do
        [[ -n "$result" ]] && result+=,
        result+="$value"
    done
    echo "$result"
}

task_csv() {
    local count="$1" result="" index
    for ((index=0; index<count; index++)); do
        [[ -n "$result" ]] && result+=,
        result+="$index"
    done
    echo "$result"
}

policy_variable() {
    case "$1:$2" in
        gr00t:quantvla_w4a8) echo TABLE6_GR00T_W4_PROCS_PER_GPU ;;
        gr00t:uniform_w6) echo TABLE6_GR00T_W6_PROCS_PER_GPU ;;
        gr00t:gdsq_vla_selector) echo TABLE6_GR00T_SELECTOR_PROCS_PER_GPU ;;
        pi05:quantvla_w4a8) echo TABLE6_PI05_W4_PROCS_PER_GPU ;;
        pi05:uniform_w6) echo TABLE6_PI05_W6_PROCS_PER_GPU ;;
        pi05:gdsq_vla_selector) echo TABLE6_PI05_SELECTOR_PROCS_PER_GPU ;;
    esac
}

probe_one() {
    local ordinal="$1" model="$2" config="$3" gpu="$4"
    local replicas variable tag port log
    variable="$(policy_variable "$model" "$config")"
    mkdir -p "$CONTROL/concurrency"
    for replicas in 4 2 1; do
        tag="concurrency_${replicas}x_2trials_v1"
        port=$((21000 + ordinal * 20))
        log="$LOGS/probe_${model}_${config}_${replicas}x.log"
        echo "[table6 probe] $model/$config gpu=$gpu replicas=$replicas" >"$log"
        if env \
            TABLE6_GPU_LIST="$(repeat_csv "$gpu" "$replicas")" \
            TABLE6_PORT_BASE="$port" TABLE6_NUM_TRIALS=2 \
            TABLE6_TASK_IDS_OVERRIDE="$(task_csv "$replicas")" \
            TABLE6_SMOKE_TASKS="$replicas" TABLE6_SMOKE_TAG="$tag" \
            bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" \
                smoke-cell "$model" "$config" goal >>"$log" 2>&1; then
            printf 'export %s=%s\n' "$variable" "$replicas" \
                >"$CONTROL/concurrency/${model}_${config}.env"
            return
        fi
        sleep 10
    done
    echo "all concurrency probes failed for $model/$config on GPU $gpu" >&2
    return 1
}

run_probe_worker() {
    local worker="$1" workers="$2" gpu="$3" index model config
    local jobs=(
        gr00t:quantvla_w4a8 gr00t:uniform_w6 gr00t:gdsq_vla_selector
        pi05:quantvla_w4a8 pi05:uniform_w6 pi05:gdsq_vla_selector
    )
    for ((index=worker; index<${#jobs[@]}; index+=workers)); do
        IFS=: read -r model config <<<"${jobs[$index]}"
        probe_one "$index" "$model" "$config" "$gpu"
    done
}

probe_quant_concurrency() {
    phase probe_quant_concurrency "fast evaluator, four then two then one process per card"
    wait_for_free_gpus 40000
    local workers="${#FREE_GPUS[@]}" pids=() worker
    (( workers > 6 )) && workers=6
    for ((worker=0; worker<workers; worker++)); do
        run_probe_worker "$worker" "$workers" "${FREE_GPUS[$worker]}" &
        pids+=("$!")
    done
    wait_for_pids "${pids[@]}"
    local jobs=(
        gr00t_quantvla_w4a8 gr00t_uniform_w6 gr00t_gdsq_vla_selector
        pi05_quantvla_w4a8 pi05_uniform_w6 pi05_gdsq_vla_selector
    ) item
    : >"$CONTROL/concurrency.env"
    for item in "${jobs[@]}"; do
        test -s "$CONTROL/concurrency/${item}.env"
        sed -n '1p' "$CONTROL/concurrency/${item}.env" >>"$CONTROL/concurrency.env"
    done
    # shellcheck disable=SC1090
    source "$CONTROL/concurrency.env"
    env | sort | rg '^TABLE6_(GR00T|PI05)_.*_PROCS_PER_GPU=' \
        >"$CONTROL/concurrency_selected.txt"
}

run_quant_subset() {
    phase run_quant_subset "24 cells and 2400 episodes"
    # shellcheck disable=SC1090
    source "$CONTROL/concurrency.env"
    export TABLE6_GR00T_W4_PROCS_PER_GPU TABLE6_GR00T_W6_PROCS_PER_GPU
    export TABLE6_GR00T_SELECTOR_PROCS_PER_GPU TABLE6_PI05_W4_PROCS_PER_GPU
    export TABLE6_PI05_W6_PROCS_PER_GPU TABLE6_PI05_SELECTOR_PROCS_PER_GPU
    unset TABLE6_GPU_LIST TABLE6_PORT_BASE TABLE6_NUM_TRIALS
    unset TABLE6_TASK_IDS_OVERRIDE TABLE6_SMOKE_TASKS TABLE6_SMOKE_TAG
    bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" run \
        >>"$CONTROL/quant_subset.log" 2>&1
    bash "$ROOT/scripts/run_table6_libero_quant_subset.sh" status \
        >>"$CONTROL/quant_subset.log" 2>&1
}

aggregate_build_sync() {
    phase aggregate "exact 4000-episode coverage"
    "$AUDIT_PY" "$ROOT/scripts/tools/aggregate_table6_libero.py" \
        >"$LOGS/aggregate.log" 2>&1
    "$AUDIT_PY" "$ROOT/scripts/tools/render_omega_qvla_libero_table.py" \
        >"$LOGS/render_table6.log" 2>&1
    phase build_paper "fail-closed paper checks"
    make -C "$ROOT/docs/gdsq_vla_iclr2027" check \
        >"$LOGS/paper_check.log" 2>&1
    phase sync_overleaf "audited normal fast-forward"
    "$AUDIT_PY" "$ROOT/scripts/tools/sync_gdsq_overleaf.py" head-audit \
        >"$LOGS/overleaf_head_audit.log" 2>&1
    "$AUDIT_PY" "$ROOT/scripts/tools/sync_gdsq_overleaf.py" head-apply \
        >"$LOGS/overleaf_head_apply.log" 2>&1
    "$AUDIT_PY" "$ROOT/scripts/tools/sync_gdsq_overleaf.py" verify \
        >"$LOGS/overleaf_verify.log" 2>&1
}

main() {
    local mode="${1:-run}"
    [[ "$mode" == run || "$mode" == prepare-quant ]] || {
        echo "usage: $0 [run|prepare-quant]" >&2
        return 2
    }
    exec 9>"$CONTROL/unattended.lock"
    flock -n 9 || { echo "Table 6 unattended pipeline is already running" >&2; return 1; }
    phase prepare_uniform_plans "checkpoint-pinned artifacts"
    "$OPENPI_PY" "$ROOT/scripts/tools/prepare_table6_libero_plans.py" \
        >"$LOGS/prepare_plans.log" 2>&1
    start_pi_pack
    wait_for_omega_subset
    finish_pi_pack
    collect_calibration_buffers
    select_pi05_masks
    calibrate_static_a8
    if [[ "$mode" == prepare-quant ]]; then
        phase quant_artifacts_ready "16 validated static-A8 artifacts"
        return
    fi
    probe_quant_concurrency
    run_quant_subset
    aggregate_build_sync
    phase complete "Table 6 rendered, paper checked, Overleaf verified"
}

main "$@"
