#!/usr/bin/env bash
set -euo pipefail

# Reduced post-selection diagnostic for every corrected FCP candidate:
# 50 RoboCasa365 tasks x paired seeds 0--9.  The frozen context-base control
# is reused; candidate outcomes cannot alter the already frozen FCP decision.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
export PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
MATERIALIZER="$REPO_ROOT/scripts/tools/materialize_fcp_candidate_rollouts.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_fcp_candidate_rollouts.py"
HESSIAN_SUBSETTER="$REPO_ROOT/scripts/tools/materialize_full_context_hessian_subset.py"

ROOT="${FCP_CLOSED_LOOP_ROOT:-$REPO_ROOT/runs/full_context_v2/fcp_completion/closed_loop}"
PROTOCOL="$REPO_ROOT/scripts/quantvla_fcp_hardware_protocol.json"
PROPOSALS="$REPO_ROOT/runs/full_context_v2/statistics_correction/proposals"
BASELINE="$REPO_ROOT/runs/full_context_v2/table1/results/full_context_v2"
MANIFEST="$ROOT/preregistration.json"
RESULTS="$ROOT/results"
SEEDS="0,1,2,3,4,5,6,7,8,9"
CANDIDATES=(single_best two_best attention_6 mlp_2 ff_pair_0 dp_full_lambda_1p0)
SPLITS=(atomic_seen composite_seen composite_unseen)

usage() {
    echo "usage: $0 prepare | preflight | run-candidate ID GPU PORT [EGL_POOL] | run-sequential GPU [PORT] [EGL_POOL] | status | aggregate" >&2
}

assert_idle_gpu() {
    local gpu="$1"
    local processes
    processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null || true)"
    local uuid
    uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
    if awk -F, -v uuid="$uuid" '{gsub(/ /,"",$1); if ($1 == uuid) found=1} END {exit !found}' <<<"$processes"; then
        echo "GPU $gpu already has a compute process; refusing to interfere" >&2
        exit 1
    fi
}

prepare() {
    local candidate split
    mkdir -p "$ROOT/hessian"
    for candidate in "${CANDIDATES[@]}"; do
        for split in "${SPLITS[@]}"; do
            local parent="$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t/$split/hessian_w4.npz"
            local plan="$PROPOSALS/$candidate.json"
            local out="$ROOT/hessian/$candidate/$split/hessian_w4.npz"
            "$PYTHON" "$HESSIAN_SUBSETTER" \
                --parent "$parent" \
                --plan "$plan" \
                --model gr00t \
                --out "$out" \
                >/dev/null
        done
    done
    "$PYTHON" "$MATERIALIZER"
}

preflight() {
    prepare >/dev/null
    "$PYTHON" - "$MANIFEST" "$PROTOCOL" <<'PY'
import hashlib
import json
import sys

manifest_path, protocol_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
protocol = json.load(open(protocol_path, encoding="utf-8"))
evaluation = manifest["evaluation"]
if evaluation["trial_seeds"] != list(range(10)):
    raise SystemExit("FCP seed-set drift")
if evaluation["episodes_per_candidate"] != 500:
    raise SystemExit("FCP episode-budget drift")
if evaluation["total_new_episodes"] != 3000:
    raise SystemExit("FCP total-episode drift")
if evaluation["candidate_ids"] != protocol["closed_loop_evaluation"]["candidate_ids"]:
    raise SystemExit("FCP candidate-order drift")
if evaluation["paired_control"]["episodes"] != 500:
    raise SystemExit("paired-control coverage drift")
print("[fcp/preflight] six candidates; 50 tasks x 10 paired seeds; 3000 new episodes")
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
    local candidate="$1" split="$2" gpu="$3" port="$4" egl_pool="$5"
    local source_spec="$ROOT/specs/$candidate/$split.json"
    local execution_spec="$ROOT/specs/$candidate/.execution-$split.json"
    local run_dir="$ROOT/matrix/$candidate/$split"
    local result_dir="$RESULTS/$candidate/$split"
    local checkpoint="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/$split/checkpoint-60000"
    local tasks n_shards effective_egl_pool
    tasks="$($PYTHON - "$MANIFEST" "$split" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(manifest["evaluation"]["task_sets"][sys.argv[2]]))
PY
)"
    n_shards="${FCP_N_SHARDS:-10}"
    effective_egl_pool="$egl_pool"
    # Resource-only resumes must reproduce an existing split manifest even if
    # later splits use fewer shards because shared-GPU headroom changed.
    if [[ -f "$run_dir/manifest.json" ]]; then
        n_shards="$($PYTHON - "$run_dir/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(manifest["shards"]))
PY
)"
        effective_egl_pool="$($PYTHON - "$run_dir/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(str(value) for value in manifest["protocol"]["egl_device_pool"]))
PY
)"
    fi
    # A completed split needs no GPU server. This is important on shared GPUs:
    # resource headroom may legitimately differ when a later split resumes.
    if [[ -f "$run_dir/manifest.json" ]] && "$PYTHON" - "$run_dir" "$candidate" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidate = sys.argv[2]
manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
expected = {(str(task), int(seed)) for task in manifest["tasks"] for seed in manifest["seeds"]}
seen = set()
for path in sorted(run_dir.glob(f"{candidate}_s*.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config") != candidate or row.get("status") != "complete":
            continue
        key = (str(row["task"]), int(row["seed"]))
        if key in seen:
            raise SystemExit(1)
        seen.add(key)
raise SystemExit(0 if seen == expected else 1)
PY
    then
        echo "[fcp/run] candidate=$candidate split=$split already complete; skipping server" >&2
        mkdir -p "$result_dir"
        cp -n "$run_dir"/"${candidate}"_s*.jsonl "$result_dir/" || true
        return 0
    fi
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
        --n-shards "$n_shards" \
        --seed-shards-per-task 1 \
        --egl-device-pool "$effective_egl_pool" \
        --trial-batch-size 5 \
        --max-concurrent-clients "${FCP_MAX_CLIENTS:-4}" \
        --action-noise paired \
        --diagnostic-only \
        --allow-shared-gpus
    mkdir -p "$result_dir"
    cp -n "$run_dir"/"${candidate}"_s*.jsonl "$result_dir/" || true
}

run_candidate() {
    local candidate="$1" gpu="$2" port="$3" egl_pool="${4:-$2}"
    case "$candidate" in
        single_best|two_best|attention_6|mlp_2|ff_pair_0|dp_full_lambda_1p0) ;;
        *) echo "unknown FCP candidate: $candidate" >&2; exit 2 ;;
    esac
    if [[ "${FCP_PREFLIGHT_DONE:-0}" != "1" ]]; then
        preflight >/dev/null
    fi
    echo "[fcp/run] candidate=$candidate server_gpu=$gpu port=$port egl_pool=$egl_pool" >&2
    for split in "${SPLITS[@]}"; do
        run_split "$candidate" "$split" "$gpu" "$port" "$egl_pool"
    done
}

run_sequential() {
    local gpu="$1" port="${2:-19700}" egl_pool="${3:-$1}"
    assert_idle_gpu "$gpu"
    preflight >/dev/null
    for candidate in "${CANDIDATES[@]}"; do
        FCP_PREFLIGHT_DONE=1 run_candidate "$candidate" "$gpu" "$port" "$egl_pool"
    done
    aggregate
}

status() {
    "$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = ("single_best", "two_best", "attention_6", "mlp_2", "ff_pair_0", "dp_full_lambda_1p0")
result = {}
for candidate in candidates:
    seen = set()
    successes = 0
    for path in (root / "matrix" / candidate).glob("**/*.jsonl"):
        if path.name.startswith(("gpu_efficiency", "gpu_server_efficiency")):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != candidate or row.get("status") != "complete":
                continue
            key = (str(row["task"]), int(row["seed"]))
            if key not in seen:
                seen.add(key)
                successes += int(bool(row["success"]))
    result[candidate] = {
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
    prepare) prepare ;;
    preflight) preflight ;;
    run-candidate) [[ $# -ge 4 ]] || { usage; exit 2; }; run_candidate "$2" "$3" "$4" "${5:-}" ;;
    run-sequential) [[ $# -ge 2 ]] || { usage; exit 2; }; run_sequential "$2" "${3:-}" "${4:-}" ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
