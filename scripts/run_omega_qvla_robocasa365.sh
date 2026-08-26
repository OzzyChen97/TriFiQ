#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
GROOT_PY="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
ROOT="$REPO_ROOT/runs/gdsq_extension_preregistered_v1/omega_qvla_robocasa365_v1"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining"
BUILDER="$REPO_ROOT/scripts/tools/omega_qvla_robocasa365_build.py"
PROTOCOL="$REPO_ROOT/scripts/tools/omega_qvla_robocasa365_protocol.py"
MATRIX="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
PARSER="$REPO_ROOT/scripts/tools/parse_robocasa_atomic_matrix.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_robocasa365_official.py"
SEEDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"

usage() {
    echo "usage: $0 prepare | build-packs | preflight | run-formal | run-unseen-and-aggregate | aggregate | run-all | status" >&2
}

formal_run_dir() {
    case "$1" in
        atomic_seen) printf '%s/results/atomic_seen_sixgpu_shared_v4\n' "$ROOT" ;;
        composite_seen) printf '%s/results/composite_seen_sixgpu_shared_v4\n' "$ROOT" ;;
        composite_unseen) printf '%s/results/composite_unseen_sixgpu_shared_v5\n' "$ROOT" ;;
        *) echo "unknown task set: $1" >&2; return 2 ;;
    esac
}

spec_for() {
    case "$1" in
        composite_unseen) printf '%s/specs/omega_qvla_composite_unseen_v4.json\n' "$ROOT" ;;
        *) printf '%s/specs/omega_qvla_%s_v3.json\n' "$ROOT" "$1" ;;
    esac
}

egl_pool_for() {
    case "$1" in
        composite_unseen) printf '%s\n' "1,2,3,5,6,7" ;;
        *) printf '%s\n' "1,2,4,5,6,7" ;;
    esac
}

checkpoint_for() {
    printf '%s/%s/checkpoint-60000\n' "$CHECKPOINT_ROOT" "$1"
}

prepare() {
    local task_set checkpoint
    for task_set in atomic_seen composite_seen composite_unseen; do
        checkpoint="$(checkpoint_for "$task_set")"
        "$GROOT_PY" "$BUILDER" prepare --task-set "$task_set" --checkpoint "$checkpoint"
    done
}

build_stage() {
    local task_set="$1" stage="$2" gpu="$3" checkpoint log
    checkpoint="$(checkpoint_for "$task_set")"
    log="$ROOT/logs/build_${task_set}_${stage}.log"
    mkdir -p "$ROOT/logs"
    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] build $task_set/$stage on GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$GROOT_PY" "$BUILDER" build \
            --task-set "$task_set" --checkpoint "$checkpoint" --stage "$stage"
    } >>"$log" 2>&1
}

build_packs() {
    prepare
    local pids=()
    # GPU 0 is reserved by another long-lived process.  Spread the six
    # independent stages over the currently empty devices and leave GPU 3,
    # which is shared by another user, out of the memory-heavy build phase.
    build_stage atomic_seen llm 1 & pids+=("$!")
    build_stage atomic_seen dit 2 & pids+=("$!")
    build_stage composite_seen llm 4 & pids+=("$!")
    build_stage composite_seen dit 5 & pids+=("$!")
    build_stage composite_unseen llm 6 & pids+=("$!")
    build_stage composite_unseen dit 7 & pids+=("$!")
    local pid
    for pid in "${pids[@]}"; do wait "$pid"; done
    local task_set checkpoint
    for task_set in atomic_seen composite_seen composite_unseen; do
        checkpoint="$(checkpoint_for "$task_set")"
        "$GROOT_PY" "$BUILDER" merge --task-set "$task_set" --checkpoint "$checkpoint"
        "$GROOT_PY" "$BUILDER" verify --task-set "$task_set" --checkpoint "$checkpoint"
        "$GROOT_PY" "$PROTOCOL" materialize --task-set "$task_set"
    done
}

preflight_valid() {
    local run_dir="$1"
    "$GROOT_PY" - "$run_dir" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); mp=root/'manifest.json'; sp=root/'summary.json'
if not mp.is_file() or not sp.is_file(): raise SystemExit(1)
m=json.load(open(mp)); s=json.load(open(sp))
checks=[
    m.get('diagnostic_only') is True,
    len(m.get('tasks', [])) == 2,
    m.get('seeds') == [0, 1],
    s.get('manifest_sha256') == hashlib.sha256(mp.read_bytes()).hexdigest(),
    not s.get('validation_errors'),
    s.get('configs', {}).get('omega_qvla_w4a4', {}).get('episodes') == 4,
]
raise SystemExit(0 if all(checks) else 1)
PY
}

run_matrix_until_complete() {
    local task_set="$1" phase="$2" tasks="$3" seeds="$4" run_dir="$5" diagnostic="$6"
    local checkpoint spec egl_pool
    checkpoint="$(checkpoint_for "$task_set")"
    spec="$(spec_for "$task_set")"
    egl_pool="$(egl_pool_for "$task_set")"
    local args=(
        --spec "$spec" --run-dir "$run_dir" --phase "$phase"
        --task-set "$task_set" --seeds "$seeds" --checkpoint "$checkpoint"
        --n-shards 12 --trial-batch-size 10 --trial-timeout 3600
        --action-noise paired --formal-provenance-v2
        --gpu-sample-interval 10 --egl-device-pool "$egl_pool"
        --allow-shared-gpus
    )
    [[ -n "$tasks" ]] && args+=(--tasks "$tasks")
    [[ "$diagnostic" == 1 ]] && args+=(--diagnostic-only)
    until "$GROOT_PY" "$MATRIX" "${args[@]}"; do
        echo "[omega-robocasa] matrix retry in 30s: $run_dir" >&2
        sleep 30
    done
    "$GROOT_PY" "$PARSER" --run-dir "$run_dir" --bootstrap 10000 \
        >"$run_dir/strict_parse.log"
}

formal_valid() {
    local task_set="$1" run_dir="$2" expected="$3"
    "$GROOT_PY" - "$task_set" "$run_dir" "$expected" <<'PY'
import hashlib, json, pathlib, sys
task_set, run_dir, expected = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
manifest_path, summary_path = run_dir / "manifest.json", run_dir / "summary.json"
if not manifest_path.is_file() or not summary_path.is_file():
    raise SystemExit(1)
manifest = json.load(open(manifest_path))
summary = json.load(open(summary_path))
row = summary.get("configs", {}).get("omega_qvla_w4a4", {})
checks = [
    manifest.get("task_set") == task_set,
    manifest.get("diagnostic_only") is False,
    len(manifest.get("seeds", [])) == 50,
    summary.get("manifest_sha256") == hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    not summary.get("validation_errors"),
    row.get("episodes") == expected,
    row.get("formal_failures") == 0,
]
raise SystemExit(0 if all(checks) else 1)
PY
}

run_formal_task() {
    local task_set="$1" expected run_dir
    case "$task_set" in
        atomic_seen) expected=900 ;;
        composite_seen|composite_unseen) expected=800 ;;
        *) echo "unknown task set: $task_set" >&2; return 2 ;;
    esac
    run_dir="$(formal_run_dir "$task_set")"
    if formal_valid "$task_set" "$run_dir" "$expected"; then
        echo "[omega-robocasa] reuse strict formal result: $task_set ($expected/$expected)"
        return
    fi
    run_matrix_until_complete "$task_set" formal "" "$SEEDS" "$run_dir" 0
    formal_valid "$task_set" "$run_dir" "$expected"
}

preflight() {
    local task_set tasks run_dir checkpoint
    for task_set in atomic_seen composite_seen composite_unseen; do
        "$GROOT_PY" "$PROTOCOL" materialize --task-set "$task_set"
        case "$task_set" in
            atomic_seen) tasks="OpenDrawer,TurnOnMicrowave" ;;
            composite_seen) tasks="DeliverStraw,PrepareCoffee" ;;
            composite_unseen) tasks="ArrangeTea,PanTransfer" ;;
        esac
        run_dir="$ROOT/preflight/${task_set}_sixgpu_shared_v4"
        if preflight_valid "$run_dir"; then
            echo "[omega-robocasa] reuse strict preflight: $task_set"
        else
            run_matrix_until_complete "$task_set" formal "$tasks" "0,1" "$run_dir" 1
            preflight_valid "$run_dir"
        fi
    done
}

run_formal() {
    local task_set
    for task_set in atomic_seen composite_seen composite_unseen; do
        run_formal_task "$task_set"
    done
}

aggregate() {
    mkdir -p "$ROOT/aggregate"
    "$GROOT_PY" "$AGGREGATOR" \
        --run-dir "$ROOT/results/atomic_seen_sixgpu_shared_v4" \
        --run-dir "$ROOT/results/composite_seen_sixgpu_shared_v4" \
        --run-dir "$ROOT/results/composite_unseen_sixgpu_shared_v5" \
        --out-dir "$ROOT/aggregate" --bootstrap 10000
}

run_unseen_and_aggregate() {
    run_formal_task composite_unseen
    aggregate
}

status() {
    "$GROOT_PY" - "$ROOT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
versions={'atomic_seen':'v4','composite_seen':'v4','composite_unseen':'v5'}
for task_set, expected in [('atomic_seen',900),('composite_seen',800),('composite_unseen',800)]:
    pack=root/'packs'/task_set/'omega_qvla_w4a4.pt'
    summary=root/'results'/f'{task_set}_sixgpu_shared_{versions[task_set]}'/'summary.json'
    state='missing'
    if pack.is_file(): state=f'attested={pack.with_suffix(".attestation.json").is_file()} bytes={pack.stat().st_size}'
    episodes=0; errors='pending'
    if summary.is_file():
        value=json.load(open(summary)); row=value.get('configs',{}).get('omega_qvla_w4a4',{})
        episodes=row.get('episodes',0); errors=len(value.get('validation_errors',[]))
    print(f'{task_set}: pack={state} episodes={episodes}/{expected} validation_errors={errors}')
PY
}

cd "$REPO_ROOT"
case "${1:-}" in
    prepare) prepare ;;
    build-packs) build_packs ;;
    preflight) preflight ;;
    run-formal) run_formal ;;
    run-unseen-and-aggregate) run_unseen_and_aggregate ;;
    aggregate) aggregate ;;
    run-all) build_packs; preflight; run_formal; aggregate ;;
    status) status ;;
    *) usage; exit 2 ;;
esac
