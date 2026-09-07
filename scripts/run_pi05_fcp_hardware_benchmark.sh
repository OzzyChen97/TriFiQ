#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
BENCH="$REPO_ROOT/scripts/tools/benchmark_pi05_hardware.py"
AGGREGATE="$REPO_ROOT/scripts/tools/aggregate_pi05_hardware.py"
PROTOCOL="$REPO_ROOT/scripts/quantvla_pi05_fcp_diagnostic_protocol.json"
ROOT="${PI05_FCP_ROOT:-$REPO_ROOT/runs/full_context_v2/pi05_fcp_diagnostic}"
TRIALS="$ROOT/hardware/trials"
GPU="${PI05_FCP_HARDWARE_GPU:-7}"

CHECKPOINT="$REPO_ROOT/checkpoints/robocasa/pi05_pretrain_human300_pytorch"
INPUT_BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/diagnostics/fp16_onpolicy_probe/fp16_onpolicy_target_4tasks_s0-1_r4_n32.npz"
CALIBRATION_BUFFER="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
PACK_DIR="$REPO_ROOT/runs/archive_previous_versions/errorfold_v4_iter/calibration/pi05_full_w4/identity_pack"

usage() {
    echo "usage: $0 run | status | aggregate" >&2
}

plan_for() {
    case "$1" in
        transferred_initializer) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/interventions_round_main/context_base.json" ;;
        projected_anchor) echo "$REPO_ROOT/runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json" ;;
        *) return 2 ;;
    esac
}

hessian_for() {
    case "$1" in
        transferred_initializer) echo "$ROOT/artifacts/transferred_initializer/hessian_w4.npz" ;;
        projected_anchor) echo "$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz" ;;
        *) return 2 ;;
    esac
}

run_one() {
    local sequence="$1" trial="$2" config_id="$3" output plan hessian
    output="$TRIALS/trial${trial}_${config_id}.json"
    if [[ -f "$output" ]]; then
        echo "[pi05-fcp/hardware] reuse sequence=$sequence trial=$trial config=$config_id"
        return
    fi
    plan="$(plan_for "$config_id")"
    hessian="$(hessian_for "$config_id")"
    echo "[pi05-fcp/hardware] sequence=$sequence trial=$trial config=$config_id gpu=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$BENCH" \
        --config-id "$config_id" \
        --trial "$trial" \
        --sequence-index "$sequence" \
        --physical-gpu "$GPU" \
        --checkpoint "$CHECKPOINT" \
        --input-buffer "$INPUT_BUFFER" \
        --calibration-buffer "$CALIBRATION_BUFFER" \
        --plan "$plan" \
        --hessian-w4 "$hessian" \
        --pack-dir "$PACK_DIR" \
        --warmup 10 \
        --requests 60 \
        --idle-seconds 5 \
        --out "$output"
}

run_all() {
    mkdir -p "$TRIALS"
    local required=(
        "$ROOT/preregistration.json"
        "$ROOT/artifacts/transferred_initializer/hessian_w4.npz"
        "$ROOT/artifacts/transferred_initializer/hessian_w4.npz.json"
        "$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz"
        "$REPO_ROOT/runs/full_context_v2/pi05_quick/artifacts/hessian_w4.npz.json"
    )
    local path
    for path in "${required[@]}"; do
        [[ -f "$path" ]] || { echo "missing hardware prerequisite: $path" >&2; exit 1; }
    done
    # Frozen interleaving: A B / B A / A B.
    run_one 0 0 transferred_initializer
    run_one 1 0 projected_anchor
    run_one 2 1 projected_anchor
    run_one 3 1 transferred_initializer
    run_one 4 2 transferred_initializer
    run_one 5 2 projected_anchor
    "$PYTHON" "$AGGREGATE" --trials "$TRIALS" --protocol "$PROTOCOL" --out "$ROOT/hardware/summary.json"
}

status() {
    "$PYTHON" - "$TRIALS" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for path in sorted(root.glob("trial*_*.json")):
    value=json.loads(path.read_text())
    rows.append((value.get("sequence_index"), value.get("trial"), value.get("config_id"), len(value.get("latency",{}).get("per_request_ms",[]))))
print(json.dumps({"completed_trials":len(rows),"measured_requests":sum(row[3] for row in rows),"rows":rows},indent=2))
PY
}

case "${1:-}" in
    run) run_all ;;
    status) status ;;
    aggregate) "$PYTHON" "$AGGREGATE" --trials "$TRIALS" --protocol "$PROTOCOL" --out "$ROOT/hardware/summary.json" ;;
    *) usage; exit 2 ;;
esac
