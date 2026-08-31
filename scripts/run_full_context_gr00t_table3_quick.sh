#!/usr/bin/env bash
set -euo pipefail

# Reduced closed-loop Table-3 ablation: all 50 tasks, seeds 0--9.  The frozen
# headline M0 rows are reused, so only the three component-removal arms run.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
export PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
MATERIALIZER="$REPO_ROOT/scripts/tools/materialize_gr00t_table3_quick.py"
CALIBRATOR="$REPO_ROOT/scripts/tools/calibrate_a8_plan_gr00t.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_gr00t_table3_quick.py"
HESSIAN_SUBSETTER="$REPO_ROOT/scripts/tools/materialize_full_context_hessian_subset.py"
A8_SUBSETTER="$REPO_ROOT/scripts/tools/materialize_gr00t_a8_subset.py"

ROOT="${FULL_CONTEXT_GR00T_TABLE3_QUICK_ROOT:-$REPO_ROOT/runs/full_context_v2/table3_quick}"
M0_PLAN="$REPO_ROOT/runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
BASELINE="$REPO_ROOT/runs/full_context_v2/table1/results/full_context_v2"
MANIFEST="$ROOT/preregistration.json"
RESULTS="$ROOT/results"
SEEDS="0,1,2,3,4,5,6,7,8,9"
ARMS=(static_a8 local_mse_selection no_fullnet_check)
SPLITS=(atomic_seen composite_seen composite_unseen)

usage() {
    echo "usage: $0 plan | calibrate-split SPLIT GPU | calibrate-all [GPU,GPU,GPU] | prepare | preflight | run-arm ARM GPU PORT [EGL_POOL] | run-parallel | wait-run | status | aggregate" >&2
}

top_free_gpus() {
    local count="$1"
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' \
        | sort -nr \
        | head -n "$count" \
        | awk '{print $2}'
}

usable_egl_pool() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) >= 13000) print $1}' \
        | paste -sd, -
}

plan() {
    "$PYTHON" "$MATERIALIZER" plan
}

calibrate_split() {
    local split="$1" gpu="$2"
    local checkpoint="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/$split/checkpoint-60000"
    local packdir="$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t/$split/identity_pack"
    local hessian="$REPO_ROOT/runs/full_context_v2/gr00t_main_hessian/$split/hessian_w4.npz"
    local out="$ROOT/static_a8/$split/m0_static_a8.npz"
    mkdir -p "$(dirname "$out")"
    echo "[table3/calibrate] split=$split gpu=$gpu out=$out" >&2
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$CALIBRATOR" \
        --model-path "$checkpoint" \
        --plan "$M0_PLAN" \
        --packdir "$packdir" \
        --hessian-w4 "$hessian" \
        --out "$out" \
        --device cuda \
        --denoising-steps 4 \
        --batch-size 8 \
        --calib-steps 32 \
        --calibration-seed 0
}

calibrate_all() {
    plan >/dev/null
    local requested="${1:-}"
    local gpus=()
    if [[ -n "$requested" ]]; then
        IFS=, read -r -a gpus <<<"$requested"
    else
        mapfile -t gpus < <(top_free_gpus 3)
    fi
    if [[ "${#gpus[@]}" -ne 3 ]]; then
        echo "calibrate-all requires exactly three GPUs" >&2
        exit 2
    fi
    local pids=()
    for index in 0 1 2; do
        calibrate_split "${SPLITS[$index]}" "${gpus[$index]}" \
            >"$ROOT/static_a8/${SPLITS[$index]}.calibrate.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "one or more Table-3 A8 calibrations failed" >&2
        exit 1
    fi
    echo "[table3/calibrate] all three plan-specific scale tables are complete" >&2
}

prepare() {
    plan >/dev/null
    for split in "${SPLITS[@]}"; do
        local parent="$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t/$split"
        "$PYTHON" "$HESSIAN_SUBSETTER" \
            --parent "$parent/hessian_w4.npz" \
            --plan "$ROOT/plans/local_mse_selection.json" \
            --model gr00t \
            --out "$ROOT/hessian/local_mse_selection/$split/hessian_w4.npz" \
            >/dev/null
        "$PYTHON" "$HESSIAN_SUBSETTER" \
            --parent "$parent/hessian_w4.npz" \
            --plan "$REPO_ROOT/runs/full_context_v2/p2/proposals_dynamic/single_best.json" \
            --model gr00t \
            --out "$ROOT/hessian/no_fullnet_check/$split/hessian_w4.npz" \
            >/dev/null
        "$PYTHON" "$A8_SUBSETTER" \
            --parent "$parent/a8_scales.npz" \
            --plan "$M0_PLAN" \
            --out "$ROOT/static_a8/$split/m0_static_a8.npz" \
            >/dev/null
    done
    "$PYTHON" "$MATERIALIZER" prepare
}

preflight() {
    prepare >/dev/null
    "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluation = manifest["evaluation"]
if evaluation["episodes_per_config"] != 500:
    raise SystemExit("Table-3 episode budget drift")
if evaluation["trial_seeds"] != list(range(10)):
    raise SystemExit("Table-3 seed set drift")
if evaluation["paired_baseline"]["episodes"] != 500:
    raise SystemExit("paired M0 baseline coverage drift")
if manifest["method"]["quantized_w4_layers"] != 100:
    raise SystemExit("M0 mask drift")
for split, audit in manifest["runtime_audits"].items():
    if not audit["identity_pack"]["same_for_all_arms"]:
        raise SystemExit(f"{split}: identity-pack mismatch")
print("[table3/preflight] 50 tasks x 10 seeds; 500 paired episodes per arm; frozen artifacts verified")
PY
}

rewrite_placement() {
    local source="$1" target="$2" gpu="$3" port="$4"
    "$PYTHON" - "$source" "$target" "$gpu" "$port" <<'PY'
import json
import os
import sys
import tempfile

source, target, gpu, port = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
payload["configs"][0]["gpu"] = int(gpu)
payload["configs"][0]["port"] = int(port)
directory = os.path.dirname(target)
os.makedirs(directory, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".execution-", suffix=".json", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

run_split() {
    local arm="$1" split="$2" gpu="$3" port="$4" egl_pool="$5"
    local source_spec="$ROOT/specs/$arm/$split.json"
    local execution_spec="$ROOT/specs/$arm/.execution-$split.json"
    local run_dir="$ROOT/matrix/$arm/$split"
    local result_dir="$RESULTS/$arm/$split"
    local checkpoint="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/$split/checkpoint-60000"
    local tasks
    tasks="$($PYTHON - "$MANIFEST" "$split" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(manifest["evaluation"]["task_sets"][sys.argv[2]]))
PY
)"
    rewrite_placement "$source_spec" "$execution_spec" "$gpu" "$port"
    "$PYTHON" "$RUNNER" \
        --spec "$execution_spec" \
        --run-dir "$run_dir" \
        --phase dev \
        --seeds "$SEEDS" \
        --checkpoint "$checkpoint" \
        --task-set "$split" \
        --tasks "$tasks" \
        --dev-tasks "$tasks" \
        --n-shards "$(awk -F, '{print NF}' <<<"$tasks")" \
        --seed-shards-per-task 1 \
        --egl-device-pool "$egl_pool" \
        --trial-batch-size 5 \
        --max-concurrent-clients "${TABLE3_MAX_CLIENTS_PER_ARM:-6}" \
        --action-noise paired \
        --diagnostic-only \
        --allow-shared-gpus
    mkdir -p "$result_dir"
    cp -n "$run_dir"/"${arm}"_s*.jsonl "$result_dir/" || true
}

run_arm() {
    local arm="$1" gpu="$2" port="$3" egl_pool="${4:-$(usable_egl_pool)}"
    local start_split="${TABLE3_START_SPLIT:-}"
    local started=0
    case "$arm" in
        static_a8|local_mse_selection|no_fullnet_check) ;;
        *) echo "unknown Table-3 arm: $arm" >&2; exit 2 ;;
    esac
    preflight >/dev/null
    echo "[table3/run] arm=$arm server_gpu=$gpu port=$port egl_pool=$egl_pool" >&2
    for split in "${SPLITS[@]}"; do
        if [[ -n "$start_split" && "$started" -eq 0 ]]; then
            if [[ "$split" != "$start_split" ]]; then
                continue
            fi
            started=1
        fi
        run_split "$arm" "$split" "$gpu" "$port" "$egl_pool"
    done
    if [[ -n "$start_split" && "$started" -eq 0 ]]; then
        echo "unknown TABLE3_START_SPLIT: $start_split" >&2
        exit 2
    fi
}

run_parallel() {
    preflight >/dev/null
    local gpus=()
    local egl_pool
    local placement="$ROOT/execution_placement.json"
    if [[ -f "$placement" ]]; then
        mapfile -t gpus < <("$PYTHON" - "$placement" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for gpu in payload["model_gpus"]:
    print(gpu)
PY
)
        egl_pool="$($PYTHON - "$placement" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["egl_pool"])
PY
)"
    else
        mapfile -t gpus < <(top_free_gpus 3)
        if [[ "${#gpus[@]}" -ne 3 ]]; then
            echo "unable to select three model GPUs" >&2
            exit 1
        fi
        egl_pool="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F, -v a="${gpus[0]}" -v b="${gpus[1]}" -v c="${gpus[2]}" \
                '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) >= 13000 && $1 != a && $1 != b && $1 != c) print $1}' \
            | paste -sd, -)"
        if [[ -z "$egl_pool" ]]; then
            echo "no non-model GPU has enough free memory for EGL clients" >&2
            exit 1
        fi
        "$PYTHON" - "$placement" "${gpus[0]}" "${gpus[1]}" "${gpus[2]}" "$egl_pool" <<'PY'
import json
import sys
from quantvla_outputimpact import atomic_json

path, *values = sys.argv[1:]
atomic_json(path, {
    "schema_version": 1,
    "kind": "gr00t_table3_quick_execution_placement",
    "immutable": True,
    "arm_order": ["static_a8", "local_mse_selection", "no_fullnet_check"],
    "model_gpus": [int(value) for value in values[:3]],
    "egl_pool": values[3],
    "selection_rule": "three largest free-memory GPUs for model servers; non-model GPUs with at least 13000 MiB for EGL",
})
PY
    fi
    local pids=()
    for index in 0 1 2; do
        run_arm "${ARMS[$index]}" "${gpus[$index]}" "$((19621 + index))" "$egl_pool" \
            >"$ROOT/${ARMS[$index]}.run.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "one or more Table-3 rollout arms failed; rerun is hash-aware and resumable" >&2
        exit 1
    fi
    aggregate
}

wait_run() {
    plan >/dev/null
    while pgrep -f '[r]un_robocasa365_pi05_eval.py|[r]un_pi05_formal_worker_seeded.sh' >/dev/null; do
        echo "[table3/wait] current pi0.5 rollout workers are active; preserving them" >&2
        sleep 30
    done
    echo "[table3/wait] rollout GPUs released; materializing inventory-exact deployment subsets" >&2
    prepare >/dev/null
    run_parallel
}

status() {
    "$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
arms = ("static_a8", "local_mse_selection", "no_fullnet_check")
result = {}
for arm in arms:
    seen = set()
    successes = 0
    for path in (root / "matrix" / arm).glob("**/*.jsonl"):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != arm or row.get("status") != "complete":
                continue
            key = (str(row["task"]), int(row["seed"]))
            if key not in seen:
                seen.add(key)
                successes += int(bool(row["success"]))
    result[arm] = {
        "completed": len(seen),
        "expected": 500,
        "remaining": 500 - len(seen),
        "successes_withheld_until_complete": None if len(seen) < 500 else successes,
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

aggregate() {
    "$PYTHON" "$AGGREGATOR" \
        --manifest "$MANIFEST" \
        --results "$RESULTS" \
        --baseline "$BASELINE" \
        --out "$ROOT/aggregate.json"
}

case "${1:-}" in
    plan) plan ;;
    calibrate-split) [[ $# -eq 3 ]] || { usage; exit 2; }; calibrate_split "$2" "$3" ;;
    calibrate-all) calibrate_all "${2:-}" ;;
    prepare) prepare ;;
    preflight) preflight ;;
    run-arm) [[ $# -ge 4 ]] || { usage; exit 2; }; run_arm "$2" "$3" "$4" "${5:-}" ;;
    run-parallel) run_parallel ;;
    wait-run) wait_run ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
