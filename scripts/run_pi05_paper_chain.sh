#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROBOCASA_PY="/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"
PAPER_ROOT="${PI05_PAPER_ROOT:-$REPO_ROOT/runs/pi05_quantvla_paper}"
FORMAL_ROOT="$REPO_ROOT/runs/pi05_gdsq_final/official_pretrain_paired50"
PREPARE="$REPO_ROOT/scripts/prepare_pi05_paper_quantvla.sh"
SERVER="$REPO_ROOT/scripts/run_pi05_paper_server.sh"
SANITY="$REPO_ROOT/scripts/run_pi05_paper_sanity30.sh"
LOG="$PAPER_ROOT/control/paper_chain.log"
TRANSITION="$PAPER_ROOT/control/diagnostic_transition.json"

mkdir -p "$PAPER_ROOT/control"
echo "[paper chain] waiting for the frozen Table-1 supervisor" 
while pgrep -f '/scripts/supervise_pi05_faithful_parallel.sh' >/dev/null 2>&1; do
    sleep 60
done

if [[ ! -f "$TRANSITION" ]]; then
    if [[ ! -f "$FORMAL_ROOT/aggregate/summary.json" ]]; then
        echo "[paper chain] diagnostic predecessor exited without transition evidence" >&2
        exit 1
    fi
    "$ROBOCASA_PY" - "$FORMAL_ROOT/aggregate/summary.json" "$FORMAL_ROOT/manifest.json" "$TRANSITION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("complete") is not True:
    raise SystemExit("diagnostic predecessor summary is not complete")
manifest = Path(sys.argv[2])
out = Path(sys.argv[3])
payload = {
    "status": "diagnostic_predecessor_complete_not_final",
    "episodes": 10000,
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "reason_not_final": (
        "QuantVLA row used permute=false, synthetic calibration and custom per-head OHB; "
        "source code evolved before the predecessor completion audit"
    ),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
    "$ROBOCASA_PY" - "$TRANSITION" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") not in {
    "diagnostic_predecessor_complete_not_final",
    "diagnostic_predecessor_aborted_for_corrected",
}:
    raise SystemExit(f"invalid diagnostic transition: {payload.get('status')!r}")
print(f"[paper chain] using transition={payload['status']} episodes={payload['episodes']}")
PY
fi
"$REPO_ROOT/scripts/run_pi05_faithful_parallel.sh" stop || true

echo "[paper chain] diagnostic predecessor complete; preparing corrected paper artifacts"
if [[ ! -f "$PAPER_ROOT/packs/pi05_robocasa_block64_permute_w4a8_ls015/manifest.json" ]]; then
    "$PREPARE" build-pack 4
fi
if [[ ! -f "$PAPER_ROOT/calibration/robocasa_real_observations_128.npz" ]]; then
    "$SERVER" start fp16 1 18601 paper_calibration_fp16
    trap '"$SERVER" stop paper_calibration_fp16 || true' EXIT INT TERM
    PYTHONPATH="$REPO_ROOT/code/pi05/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA_PY" "$REPO_ROOT/scripts/tools/pi05_collect_real_calibration_buffer.py" \
        --port 18601 --egl-device 1 \
        --out "$PAPER_ROOT/calibration/robocasa_real_observations_128.npz"
    "$SERVER" stop paper_calibration_fp16
    trap - EXIT INT TERM
fi
if [[ ! -f "$PAPER_ROOT/atm_ohb/pi05_gdsq_vla_final_real128_scalar.json" ]]; then
    "$PREPARE" calibrate 1
else
    "$PREPARE" audit
fi

echo "[paper chain] artifacts audited; starting paired and native sanity matrices"
"$SANITY" run-all
echo "[paper chain] sanity complete; starting corrected full Table-1 matrix"
"$REPO_ROOT/scripts/run_pi05_corrected_table1.sh" run-all
echo "[paper chain] complete"
