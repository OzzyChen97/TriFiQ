#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
FINAL_ROOT="$REPO_ROOT/runs/pi05_gdsq_gr00t_aligned"
OPENPI_PY="/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
CLIENT_PATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src"
SERVER_MANAGER="$REPO_ROOT/scripts/run_pi05_formal_server.sh"
BUFFER="$FINAL_ROOT/calibration/pi05_robocasa365_seed0_n256.npz"
PACK="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
FULL_PLAN="$FINAL_ROOT/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
GDSQ_REPORT="$FINAL_ROOT/adjudication/pi05_cscka_16to1_d4.report.json"
GDSQ_PLAN="$FINAL_ROOT/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
FULL_A8="$FINAL_ROOT/a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz"
GDSQ_A8="$FINAL_ROOT/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz"
FULL_ATM="$FINAL_ROOT/atm_ohb/pi05_quantvla_uniform_w4a8_d4_static_perhead.json"
GDSQ_ATM="$FINAL_ROOT/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"
ALIGNMENT_AUDIT="$REPO_ROOT/scripts/tools/audit_pi05_against_gr00t_final.py"

usage() {
    echo "usage: $0 finalize | smoke | all" >&2
}

sha256_canonical_json() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

wrapped_layers() {
    "$OPENPI_PY" - "$1" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(
    not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    for row in plan["layers"].values()
))
PY
}

finalize_plan() {
    mkdir -p "$FINAL_ROOT/selection" "$FINAL_ROOT/plans" \
        "$FINAL_ROOT/a8" "$FINAL_ROOT/atm_ohb"
    "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_transfer_gr00t_selection.py" \
        --pi05-report "$GDSQ_REPORT" \
        --selection-out "$FINAL_ROOT/selection/final_ratio_selection.json" \
        --dev-summary-out "$FINAL_ROOT/selection/dev_summary.json" \
        --equivalence-out "$FINAL_ROOT/selection/executable_equivalence.json"

    "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_freeze_final_plan.py" \
        --selection "$FINAL_ROOT/selection/final_ratio_selection.json" \
        --out "$GDSQ_PLAN"

    local final_wrapped
    final_wrapped="$(wrapped_layers "$GDSQ_PLAN")"

    local pids=()
    env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_a8.py" \
        --device cuda --plan "$FULL_PLAN" --pack-dir "$PACK" --buffer "$BUFFER" \
        --expected-wrapped 180 --out "$FULL_A8" &
    pids+=("$!")
    env CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_a8.py" \
        --device cuda --plan "$GDSQ_PLAN" --pack-dir "$PACK" --buffer "$BUFFER" \
        --expected-wrapped "$final_wrapped" --out "$GDSQ_A8" &
    pids+=("$!")
    local pid failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || return 1

    pids=()
    env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_atm_ohb.py" \
        --device cuda --plan "$FULL_PLAN" --a8-scale "$FULL_A8" \
        --pack-dir "$PACK" --buffer "$BUFFER" --n-frames 16 --batch-size 8 \
        --scope expert --ohb-mode per_head_pre_projection --out "$FULL_ATM" &
    pids+=("$!")
    env CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_calibrate_atm_ohb.py" \
        --device cuda --plan "$GDSQ_PLAN" --a8-scale "$GDSQ_A8" \
        --pack-dir "$PACK" --buffer "$BUFFER" --n-frames 16 --batch-size 8 \
        --scope expert --ohb-mode per_head_pre_projection --out "$GDSQ_ATM" &
    pids+=("$!")
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || return 1

    "$OPENPI_PY" "$ALIGNMENT_AUDIT" --phase artifacts \
        --out "$FINAL_ROOT/audit/gr00t_final_alignment.json"
}

smoke_env() {
    export PI05_CONTROL_DIR="$FINAL_ROOT/smoke/control"
    export PI05_PACK_DIR="$PACK"
    export PI05_CALIBRATION_BUFFER="$BUFFER"
    export PI05_FULL_PLAN="$FULL_PLAN"
    export PI05_FULL_A8="$FULL_A8"
    export PI05_FULL_ATM="$FULL_ATM"
    export PI05_GDSQ_PLAN="$GDSQ_PLAN"
    export PI05_GDSQ_A8="$GDSQ_A8"
    export PI05_GDSQ_ATM="$GDSQ_ATM"
}

stop_smoke_servers() {
    smoke_env
    "$SERVER_MANAGER" stop final_smoke_fp16 || true
    "$SERVER_MANAGER" stop final_smoke_w4 || true
    "$SERVER_MANAGER" stop final_smoke_gdsqatm || true
    "$SERVER_MANAGER" stop final_smoke_gdsq || true
}

smoke() {
    smoke_env
    mkdir -p "$FINAL_ROOT/smoke/control" "$FINAL_ROOT/smoke/results"
    local pids=()
    "$SERVER_MANAGER" start fp16 1 18501 final_smoke_fp16 & pids+=("$!")
    "$SERVER_MANAGER" start quantvla_w4a8_atmohb 2 18502 final_smoke_w4 & pids+=("$!")
    "$SERVER_MANAGER" start gdsq_vla_atmohb 3 18503 final_smoke_gdsqatm & pids+=("$!")
    "$SERVER_MANAGER" start gdsq_vla 4 18504 final_smoke_gdsq & pids+=("$!")
    local pid failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]] || { stop_smoke_servers; return 1; }

    PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
        "$OPENPI_PY" "$REPO_ROOT/scripts/tools/pi05_smoke_servers.py" \
        --servers fp16=18501,quantvla_w4a8_atmohb=18502,gdsq_vla_atmohb=18503,gdsq_vla=18504 \
        --gdsq-plan "$GDSQ_PLAN" \
        --buffer "$BUFFER" \
        --out "$FINAL_ROOT/smoke/websocket_smoke.json"

    local specs=(
        "fp16,18501,1,final_smoke_fp16"
        "quantvla_w4a8_atmohb,18502,2,final_smoke_w4"
        "gdsq_vla_atmohb,18503,3,final_smoke_gdsqatm"
        "gdsq_vla,18504,4,final_smoke_gdsq"
    )
    pids=()
    local spec config port gpu instance runtime hash outdir
    for spec in "${specs[@]}"; do
        IFS=',' read -r config port gpu instance <<<"$spec"
        runtime="$PI05_CONTROL_DIR/$instance.runtime.json"
        hash="$(sha256_canonical_json "$runtime")"
        outdir="$FINAL_ROOT/smoke/results/$config"
        mkdir -p "$outdir"
        env PYTHONPATH="$CLIENT_PATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$ROBOCASA_PY" "$REPO_ROOT/scripts/run_robocasa365_pi05_eval.py" \
            --port "$port" --config-id "$config" \
            --task-set atomic_seen --tasks OpenCabinet --trial-seeds 0 \
            --split target --replan-steps 16 --egl-device "$gpu" \
            --expected-server-metadata-sha256 "$hash" \
            --resume-dir "$outdir" --out "$outdir/OpenCabinet_seed0.jsonl" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    stop_smoke_servers
    [[ "$failed" == 0 ]] || return 1

    "$ROBOCASA_PY" - "$FINAL_ROOT/smoke/results" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
configs = ("fp16", "quantvla_w4a8_atmohb", "gdsq_vla_atmohb", "gdsq_vla")
for config in configs:
    rows = []
    for path in glob.glob(str(root / config / "*.jsonl")):
        rows.extend(json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
    if len(rows) != 1 or rows[0].get("status") != "complete":
        raise SystemExit(f"invalid closed-loop smoke matrix for {config}: {len(rows)} rows")
print("[pi05 final smoke] four configs x one task x one seed complete")
PY
}

case "${1:-}" in
    finalize) finalize_plan ;;
    smoke) smoke ;;
    all) finalize_plan; smoke ;;
    *) usage; exit 2 ;;
esac
