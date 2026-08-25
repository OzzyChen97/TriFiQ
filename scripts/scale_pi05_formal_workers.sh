#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUN_DIR="$REPO_ROOT/runs/pi05_gdsq_port/official_pretrain_paired50"
CONTROL_DIR="$RUN_DIR/control"
WORKER_CONTROL="$CONTROL_DIR/workers"
WORKER="$REPO_ROOT/scripts/run_pi05_formal_worker_seeded.sh"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"

metadata_hash() {
    "$ROBOCASA_PY" - "$1" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

start_one() {
    local config="$1" port="$2" gpu="$3" shard="$4" count="$5"
    local seeds="$6" worker_id="$7" runtime="$8"
    local pid_file="$WORKER_CONTROL/$worker_id.pid"
    local log_file="$WORKER_CONTROL/$worker_id.seed${seeds}.log"
    local pid=""
    if [[ -f "$pid_file" ]]; then pid="$(<"$pid_file")"; fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "reuse seeded worker=$worker_id pid=$pid"
        return
    fi
    local hash
    hash="$(metadata_hash "$runtime")"
    nohup setsid "$WORKER" "$config" "$port" "$gpu" "$shard" "$count" "$seeds" \
        "$worker_id" "$hash" "$RUN_DIR" >"$log_file" 2>&1 </dev/null &
    pid=$!
    echo "$pid" >"$pid_file"
    echo "started seeded worker=$worker_id pid=$pid seeds=$seeds config=$config shard=$shard/$count gpu=$gpu"
}

start_all() {
    mkdir -p "$WORKER_CONTROL"
    local i suffix seeds
    for i in 0 1 2 3; do
        for suffix in lo hi; do
            if [[ "$suffix" == lo ]]; then seeds=0-24; else seeds=25-49; fi
            local tag=""
            if [[ "$suffix" == hi ]]; then tag=_hi; fi
            start_one fp16 18101 1 "$i" 4 "$seeds" "fp16_g1_w${i}${tag}" \
                "$CONTROL_DIR/smoke_fp16.runtime.json"
            start_one quantvla_w4a8_atmohb 18102 2 "$i" 8 "$seeds" "w4_g2_w${i}${tag}" \
                "$CONTROL_DIR/smoke_quantvla_w4a8_atmohb.runtime.json"
            start_one gdsq_vla_atmohb 18103 3 "$i" 4 "$seeds" "gdsqatm_g3_w${i}${tag}" \
                "$CONTROL_DIR/smoke_gdsq_vla_atmohb.runtime.json"
            start_one gdsq_vla 18104 4 "$i" 8 "$seeds" "gdsq_g4_w${i}${tag}" \
                "$CONTROL_DIR/smoke_gdsq_vla.runtime.json"
            start_one quantvla_w4a8_atmohb 18105 5 "$((i + 4))" 8 "$seeds" "w4_g5_w${i}${tag}" \
                "$CONTROL_DIR/official_quantvla_w4a8_atmohb_g5.runtime.json"
            start_one gdsq_vla 18106 6 "$((i + 4))" 8 "$seeds" "gdsq_g6_w${i}${tag}" \
                "$CONTROL_DIR/official_gdsq_vla_g6.runtime.json"
        done
    done
}

case "${1:-}" in
    start) start_all ;;
    *) echo "usage: $0 start" >&2; exit 2 ;;
esac
