#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
PAPER_ROOT="${PI05_PAPER_ROOT:-$REPO_ROOT/runs/pi05_quantvla_paper}"
BUFFER="$PAPER_ROOT/calibration/robocasa_real_observations_128.npz"
PACK="$PAPER_ROOT/packs/pi05_robocasa_block64_permute_w4a8_ls015"
PLAN="$PAPER_ROOT/plans/pi05_quantvla_paper_w4a8.plan.json"
A8="$PAPER_ROOT/a8/pi05_quantvla_paper_real32_p999_b32.npz"
ATM="$PAPER_ROOT/atm_ohb/pi05_quantvla_paper_real128_scalar.json"
GDSQ_PLAN="$PAPER_ROOT/plans/pi05_gdsq_vla_final_paper.plan.json"
GDSQ_A8="$PAPER_ROOT/a8/pi05_gdsq_vla_final_real32_p999_b32.npz"
GDSQ_ATM="$PAPER_ROOT/atm_ohb/pi05_gdsq_vla_final_real128_scalar.json"

usage() {
    echo "usage: $0 collect-buffer FP16_PORT EGL_GPU | build-pack [SHARDS] | calibrate GPU | audit | status" >&2
}

collect_buffer() {
    [[ $# -eq 2 ]] || { usage; exit 2; }
    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$REPO_ROOT/scripts/tools/pi05_collect_real_calibration_buffer.py" \
        --port "$1" --egl-device "$2" --out "$BUFFER"
}

build_pack() {
    local shards="${1:-8}"
    [[ "$shards" =~ ^[1-9][0-9]*$ ]] || { echo "invalid shard count: $shards" >&2; exit 2; }
    mkdir -p "$PACK"
    local index pids=()
    for index in $(seq 0 $((shards - 1))); do
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
            nice -n 10 "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_build_block64_pack.py" \
            --out "$PACK" --shard-index "$index" --num-shards "$shards" --block 64 \
            --lambda-smooth 0.15 --permute >"$PACK/build_shard_${index}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { echo "one or more paper pack shards failed" >&2; exit 1; }
    "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_build_block64_pack.py" \
        --out "$PACK" --block 64 --lambda-smooth 0.15 --permute --finalize
    "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_make_paper_quantvla_plan.py" \
        --pack-manifest "$PACK/manifest.json" --out "$PLAN"
}

calibrate() {
    [[ $# -eq 1 && "$1" =~ ^[0-7]$ ]] || { usage; exit 2; }
    local gpu="$1"
    [[ -f "$BUFFER" && -f "$PACK/manifest.json" && -f "$PLAN" ]] || {
        echo "buffer, pack, or plan is not ready" >&2; exit 1;
    }
    mkdir -p "$(dirname "$A8")" "$(dirname "$ATM")"
    "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_make_paper_gdsq_plan.py" \
        --pack-manifest "$PACK/manifest.json" --buffer "$BUFFER" --out "$GDSQ_PLAN"
    CUDA_VISIBLE_DEVICES="$gpu" "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_a8.py" \
        --plan "$PLAN" --pack-dir "$PACK" --buffer "$BUFFER" --out "$A8" \
        --expected-wrapped 180 --n-frames 32 --permute
    CUDA_VISIBLE_DEVICES="$gpu" "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_atm_ohb.py" \
        --plan "$PLAN" --pack-dir "$PACK" --buffer "$BUFFER" --a8-scale "$A8" \
        --out "$ATM" --n-frames 128 --scope expert --ohb-mode per_layer_post_projection \
        --atm-application fold_q_weight --ohb-application fold_o_weight \
        --permute --log-clamp 0.30 \
        --alpha-neutral 0.03 --beta-neutral 0.03
    CUDA_VISIBLE_DEVICES="$gpu" "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_a8.py" \
        --plan "$GDSQ_PLAN" --pack-dir "$PACK" --buffer "$BUFFER" --out "$GDSQ_A8" \
        --expected-wrapped 69 --n-frames 32 --permute
    CUDA_VISIBLE_DEVICES="$gpu" "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_atm_ohb.py" \
        --plan "$GDSQ_PLAN" --pack-dir "$PACK" --buffer "$BUFFER" --a8-scale "$GDSQ_A8" \
        --out "$GDSQ_ATM" --n-frames 128 --scope expert --ohb-mode per_layer_post_projection \
        --atm-application fold_q_weight --ohb-application fold_o_weight \
        --permute --log-clamp 0.30 \
        --alpha-neutral 0.03 --beta-neutral 0.03
    "$ROBOCASA_PY" "$REPO_ROOT/scripts/tools/pi05_audit_paper_quantvla.py" --root "$PAPER_ROOT"
}

audit() {
    "$ROBOCASA_PY" "$REPO_ROOT/scripts/tools/pi05_audit_paper_quantvla.py" --root "$PAPER_ROOT"
}

status() {
    local path
    for path in "$BUFFER" "$PACK/manifest.json" "$PLAN" "$A8" "$ATM" \
        "$GDSQ_PLAN" "$GDSQ_A8" "$GDSQ_ATM"; do
        if [[ -f "$path" ]]; then
            echo "ready $(sha256sum "$path" | cut -d' ' -f1) $path"
        else
            echo "pending $path"
        fi
    done
}

case "${1:-}" in
    collect-buffer) shift; collect_buffer "$@" ;;
    build-pack) shift; build_pack "$@" ;;
    calibrate) shift; calibrate "$@" ;;
    audit) audit ;;
    status) status ;;
    *) usage; exit 2 ;;
esac
