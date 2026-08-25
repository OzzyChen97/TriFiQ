#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
FINAL_ROOT="$REPO_ROOT/runs/pi05_gdsq_final"

export PI05_REQUIRE_FAITHFUL_FINAL=1
export PI05_AVOID_GPU2="${PI05_AVOID_GPU2:-1}"
export PI05_FORMAL_RUN_DIR="$FINAL_ROOT/official_pretrain_paired50"
export PI05_PACK_DIR="$REPO_ROOT/runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
export PI05_PACK_MANIFEST="$PI05_PACK_DIR/manifest.json"
export PI05_CALIBRATION_BUFFER="$REPO_ROOT/runs/pi05_gdsq_port/probes/pi05_robocasa_official_buffer32_state16.npz"
export PI05_SENSITIVITY="$FINAL_ROOT/probes/pi05_sensitivity_action_n16_w4_merged.json"
export PI05_FULL_PLAN="$REPO_ROOT/runs/pi05_gdsq_port/plans/pi05_quantvla_full_w4a8.plan.json"
export PI05_FULL_A8="$REPO_ROOT/runs/pi05_gdsq_port/a8_scales/pi05_full_w4a8_state16_p999_b32.npz"
export PI05_FULL_ATM="$REPO_ROOT/runs/pi05_gdsq_port/atm_ohb/pi05_full_w4a8_state16_static_expert.json"
export PI05_GDSQ_PLAN="$FINAL_ROOT/final/pi05_gdsq_vla_final.plan.json"
export PI05_GDSQ_A8="$FINAL_ROOT/final/pi05_gdsq_vla_final.a8_p999_b32.npz"
export PI05_GDSQ_ATM="$FINAL_ROOT/final/pi05_gdsq_vla_final.static_atm_ohb.json"
export PI05_FINAL_SELECTION="$FINAL_ROOT/final/final_ratio_selection.json"

WAVE_RUNNER="$REPO_ROOT/scripts/run_pi05_faithful_waves.sh"
case "${1:-}" in
    start) exec "$WAVE_RUNNER" run-all ;;
    prepare) exec "$WAVE_RUNNER" prepare ;;
    status) exec "$WAVE_RUNNER" status ;;
    stop-workers|stop-all|stop) exec "$WAVE_RUNNER" stop ;;
    run-wave)
        [[ $# == 2 ]] || { echo "usage: $0 run-wave CONFIG" >&2; exit 2; }
        exec "$WAVE_RUNNER" run-wave "$2"
        ;;
    *)
        echo "usage: $0 start | prepare | status | stop-all | run-wave CONFIG" >&2
        exit 2
        ;;
esac
