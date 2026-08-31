#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
EVAL_PY=/home1/gyy/probe/miniforge3/envs/libero_test/bin/python
ART=$RUN_ROOT/artifacts/gr00t
CONTROL=$RUN_ROOT/control/gr00t_pipeline
BUFFER=$RUN_ROOT/calibration/gr00t/selection_144.npz
INVENTORY=$ART/candidate_inventory.json
SUITES=(goal spatial object long)
IFS=, read -r -a SHARDS_BY_SUITE <<<"${GR00T_DYPAC_SHARDS_BY_SUITE:-1,1,1,1}"
IFS=, read -r -a SCORE_GPUS <<<"${GR00T_DYPAC_SCORE_GPUS:-1,2,3,5}"
(( ${#SHARDS_BY_SUITE[@]} == 4 )) || { echo "GR00T_DYPAC_SHARDS_BY_SUITE needs four entries" >&2; exit 2; }
TOTAL_SCORE_WORKERS=0
for count in "${SHARDS_BY_SUITE[@]}"; do
    (( count >= 1 )) || { echo "invalid GR00T suite shard count" >&2; exit 2; }
    TOTAL_SCORE_WORKERS=$((TOTAL_SCORE_WORKERS + count))
done
(( ${#SCORE_GPUS[@]} == TOTAL_SCORE_WORKERS )) || {
    echo "GR00T_DYPAC_SCORE_GPUS needs $TOTAL_SCORE_WORKERS entries" >&2; exit 2;
}
mkdir -p "$CONTROL"
exec 9>"$RUN_ROOT/control/gr00t_pipeline.lock"
flock -n 9 || { echo "GR00T LIBERO DyPAC pipeline is already active" >&2; exit 1; }
export PYTHONPATH="$ROOT/code:$ROOT/scripts/tools"
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

phase() {
    printf '%s phase=%s detail=%s\n' "$(date --iso-8601=seconds)" "$1" "${2:-}" | tee -a "$CONTROL/progress.log"
}

checkpoint() { echo "$ROOT/checkpoints/gr00t/libero-$1"; }

suite_worker_offset() {
    local suite_index=$1 index offset=0
    for ((index=0; index<suite_index; index++)); do
        offset=$((offset + SHARDS_BY_SUITE[index]))
    done
    printf '%s\n' "$offset"
}

require_base_artifacts() {
    [[ -f "$BUFFER" && -f "$INVENTORY" ]] || { echo "missing GR00T calibration/inventory" >&2; exit 1; }
    for suite in "${SUITES[@]}"; do
        for path in "$ART/$suite/all_w4.plan.json" "$ART/$suite/hessian_w4.npz" "$ART/$suite/hessian_w4.npz.json" "$ART/$suite/identity_pack"; do
            [[ -e "$path" ]] || { echo "missing GR00T DyPAC base artifact: $path" >&2; exit 1; }
        done
    done
}

outputimpact_ready() {
    "$PY" - "$CONTROL/outputimpact/global/shard_0.json" "$INVENTORY" >/dev/null 2>&1 <<'PY'
import json, pathlib, sys
score_path, inventory_path = map(pathlib.Path, sys.argv[1:])
score = json.loads(score_path.read_text())
inventory = json.loads(inventory_path.read_text())
expected = {row["name"] for row in inventory["layers"]}
assert score.get("kind") == "dypac_libero_gr00t_single_layer_outputimpact"
assert score.get("model") == "gr00t" and score.get("complete") is True
assert set(score.get("layers", {})) == expected
PY
}

initial_mask_ready() {
    "$PY" - "$ART/initial_m0.global.plan.json" "$ART/coordinate_flips.global.manifest.json" >/dev/null 2>&1 <<'PY'
import hashlib, json, pathlib, sys
plan_path, manifest_path = map(pathlib.Path, sys.argv[1:])
plan = json.loads(plan_path.read_text())
manifest = json.loads(manifest_path.read_text())
digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert plan.get("meta", {}).get("initial_mask") is True
assert len(plan.get("layers", {})) == 116
assert manifest.get("initial_plan_sha256") == digest
assert len(manifest.get("candidates", {})) == 117
PY
}

run_outputimpact() {
    phase outputimpact "four suite checkpoints, 116 coordinate primitives"
    local pids=() suite_index suite worker gpu failed=0
    for suite_index in "${!SUITES[@]}"; do
        suite=${SUITES[$suite_index]}
        mkdir -p "$CONTROL/outputimpact/$suite"
        worker=$(suite_worker_offset "$suite_index"); gpu=${SCORE_GPUS[$worker]}
        (
            export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
            "$PY" -u "$ROOT/scripts/tools/probe_libero_dypac_gr00t_outputimpact.py" \
                --suite "$suite" --checkpoint "$(checkpoint "$suite")" \
                --inventory "$INVENTORY" --plan "$ART/$suite/all_w4.plan.json" \
                --pack-dir "$ART/$suite/identity_pack" --hessian "$ART/$suite/hessian_w4.npz" \
                --buffer "$BUFFER" --shard-index 0 --shard-count 1 \
                --batch-size 4 --out "$CONTROL/outputimpact/$suite/shard_0.json"
        ) >"$CONTROL/outputimpact/$suite/shard_0.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    (( failed == 0 ))
    mkdir -p "$CONTROL/outputimpact/global"
    local inputs=()
    for suite in "${SUITES[@]}"; do inputs+=("$CONTROL/outputimpact/$suite/shard_0.json"); done
    "$PY" "$ROOT/scripts/tools/merge_libero_dypac_gr00t_scores.py" \
        --kind outputimpact --inputs "${inputs[@]}" \
        --out "$CONTROL/outputimpact/global/shard_0.json"
}

select_initial() {
    phase initial_mask "global reliability-shrunk exact-byte DP"
    local scores=("$CONTROL/outputimpact/global"/shard_*.json)
    "$PY" "$ROOT/scripts/tools/select_libero_dypac_initial_mask.py" \
        --inventory "$INVENTORY" --base-plan "$ART/spatial/all_w4.plan.json" \
        --scores "${scores[@]}" --out "$ART/initial_m0.global.plan.json" \
        >"$CONTROL/initial_mask.log" 2>&1
    "$PY" "$ROOT/scripts/tools/make_libero_dypac_coordinate_manifest.py" \
        --initial-plan "$ART/initial_m0.global.plan.json" --inventory "$INVENTORY" \
        --out "$ART/coordinate_flips.global.manifest.json" >>"$CONTROL/initial_mask.log" 2>&1
}

score_manifest() {
    local manifest=$1 noise=$2 destination=$3
    local count shards configured suite_index suite shard worker offset gpu failed=0 pids=() inputs=()
    count=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidates"]))' "$manifest")
    for suite_index in "${!SUITES[@]}"; do
        suite=${SUITES[$suite_index]}; mkdir -p "$destination/$suite"
        configured=${SHARDS_BY_SUITE[$suite_index]}
        shards=$configured; (( shards > count )) && shards=$count
        offset=$(suite_worker_offset "$suite_index")
        for ((shard=0; shard<shards; shard++)); do
            worker=$((offset + shard)); gpu=${SCORE_GPUS[$worker]}
            (
                export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
                "$PY" -u "$ROOT/scripts/tools/score_libero_dypac_gr00t_masks.py" \
                    --suite "$suite" --checkpoint "$(checkpoint "$suite")" \
                    --inventory "$INVENTORY" --plan "$ART/$suite/all_w4.plan.json" \
                    --pack-dir "$ART/$suite/identity_pack" --hessian "$ART/$suite/hessian_w4.npz" \
                    --buffer "$BUFFER" --manifest "$manifest" --noise "$noise" \
                    --shard-index "$shard" --shard-count "$shards" --batch-size 4 \
                    --out "$destination/$suite/shard_${shard}.json"
            ) >"$destination/$suite/shard_${shard}.log" 2>&1 &
            pids+=("$!")
            inputs+=("$destination/$suite/shard_${shard}.json")
        done
    done
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    (( failed == 0 ))
    mkdir -p "$destination/global"
    "$PY" "$ROOT/scripts/tools/merge_libero_dypac_gr00t_scores.py" \
        --kind mask --inputs "${inputs[@]}" --out "$destination/global/merged.json"
}

run_coordinate_scores() {
    phase fcp_coordinate "complete-network adjudication of every GR00T flip"
    score_manifest "$ART/coordinate_flips.global.manifest.json" A "$CONTROL/coordinate"
}

make_proposals() {
    phase fcp_proposals "exact-byte structured proposals"
    local scores=("$CONTROL/coordinate/global/merged.json")
    "$PY" "$ROOT/scripts/tools/propose_libero_dypac_fcp.py" \
        --coordinate-manifest "$ART/coordinate_flips.global.manifest.json" \
        --coordinate-scores "${scores[@]}" --initial-plan "$ART/initial_m0.global.plan.json" \
        --inventory "$INVENTORY" --out-dir "$ART/fcp_proposals" \
        >"$CONTROL/proposals.log" 2>&1
}

rank_top3() {
    phase fcp_fullnet "score proposals and freeze Top-3 before noise B"
    score_manifest "$ART/fcp_proposals/manifest.json" A "$CONTROL/fullnet_a"
    local scores=("$CONTROL/fullnet_a/global/merged.json")
    "$PY" "$ROOT/scripts/tools/select_libero_dypac_fcp.py" rank \
        --proposals "$ART/fcp_proposals/manifest.json" --scores-a "${scores[@]}" \
        --out "$ART/fcp_top3.global.manifest.json" >"$CONTROL/top3.log" 2>&1
}

audit_noise_b() {
    phase fcp_noise_b "held-out noise B on frozen Top-3"
    score_manifest "$ART/fcp_top3.global.manifest.json" B "$CONTROL/top3_noise_b"
}

audit_candidate_states() {
    phase fcp_candidate_state "Top-3 candidate-policy state audit across four checkpoints"
    IFS=, read -r -a audit_gpus <<<"${GR00T_DYPAC_AUDIT_GPUS:-1,2,3,5}"
    (( ${#audit_gpus[@]} == 4 )) || { echo "GR00T_DYPAC_AUDIT_GPUS must contain four GPUs" >&2; return 2; }
    local jobs=$CONTROL/candidate_state_jobs.tsv
    "$PY" - "$ART/fcp_top3.global.manifest.json" >"$jobs" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for i,c in enumerate(d['top3']): print(i,c,d['candidates'][c]['path'],sep='\t')
PY
    local candidate_index identifier candidate_plan suite_index suite gpu port pid failed
    local global_scores=()
    while IFS=$'\t' read -r candidate_index identifier candidate_plan; do
        local server_pids=() collector_pids=() suite_scores=()
        for suite_index in "${!SUITES[@]}"; do
            suite=${SUITES[$suite_index]}; gpu=${audit_gpus[$suite_index]}; port=$((22200 + candidate_index * 10 + suite_index))
            (
                export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
                "$PY" -u "$ROOT/scripts/tools/serve_libero_dypac_gr00t_candidate_audit.py" \
                    --suite "$suite" --checkpoint "$(checkpoint "$suite")" \
                    --inventory "$INVENTORY" --plan "$ART/$suite/all_w4.plan.json" \
                    --candidate-plan "$candidate_plan" --candidate-id "$identifier" \
                    --pack-dir "$ART/$suite/identity_pack" --hessian "$ART/$suite/hessian_w4.npz" \
                    --port "$port"
            ) >"$CONTROL/candidate_server_${identifier}_${suite}.log" 2>&1 &
            server_pids+=("$!")
        done
        for suite_index in "${!SUITES[@]}"; do
            suite=${SUITES[$suite_index]}; port=$((22200 + candidate_index * 10 + suite_index)); ready=0
            for _ in $(seq 1 180); do
                if timeout 10 "$PY" - "$port" "$identifier" >/dev/null 2>&1 <<'PY'
import sys
from gr00t.eval.service import ExternalRobotInferenceClient
x=ExternalRobotInferenceClient(host='127.0.0.1',port=int(sys.argv[1])).call_endpoint('get_runtime_info')
assert x['candidate_id']==sys.argv[2]
PY
                then ready=1; break; fi
                sleep 2
            done
            (( ready == 1 )) || { for pid in "${server_pids[@]}"; do kill "$pid" 2>/dev/null || true; done; return 1; }
        done
        for suite_index in "${!SUITES[@]}"; do
            suite=${SUITES[$suite_index]}; gpu=${audit_gpus[$suite_index]}; port=$((22200 + candidate_index * 10 + suite_index))
            (
                export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$gpu"
                export LIBERO_CONFIG_PATH=/home1/gyy/.libero NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
                export PYTHONPATH="$ROOT/code:$ROOT/code/LIBERO:$ROOT/scripts/tools"
                "$EVAL_PY" -u "$ROOT/scripts/tools/collect_libero_dypac_gr00t_candidate_state.py" \
                    --candidate-id "$identifier" --suite "$suite" --port "$port" \
                    --egl-device "$gpu" --out "$CONTROL/candidate_state_${identifier}_${suite}.json"
            ) >"$CONTROL/candidate_state_${identifier}_${suite}.log" 2>&1 &
            collector_pids+=("$!")
            suite_scores+=("$CONTROL/candidate_state_${identifier}_${suite}.json")
        done
        failed=0
        for pid in "${collector_pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
        for pid in "${server_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
        (( failed == 0 )) || return 1
        "$PY" "$ROOT/scripts/tools/merge_libero_dypac_gr00t_candidate_state_suites.py" \
            --inputs "${suite_scores[@]}" --out "$CONTROL/candidate_state_${identifier}.global.json"
        global_scores+=("$CONTROL/candidate_state_${identifier}.global.json")
    done <"$jobs"
    "$PY" "$ROOT/scripts/tools/merge_libero_dypac_candidate_state.py" \
        --top3 "$ART/fcp_top3.global.manifest.json" --scores "${global_scores[@]}" \
        --out "$ART/fcp_candidate_state_audit.json" >"$CONTROL/candidate_state_merge.log" 2>&1
}

freeze_and_deploy() {
    phase freeze "FCP fail-closed global mask and suite-specific deployment binding"
    local scores_b=("$CONTROL/top3_noise_b/global/merged.json")
    "$PY" "$ROOT/scripts/tools/select_libero_dypac_fcp.py" freeze \
        --top3 "$ART/fcp_top3.global.manifest.json" --scores-b "${scores_b[@]}" \
        --candidate-state-audit "$ART/fcp_candidate_state_audit.json" \
        --out "$ART/dypac_vla_libero.global.frozen.plan.json" >"$CONTROL/freeze.log" 2>&1
    "$PY" "$ROOT/scripts/tools/materialize_libero_dypac_gr00t_suite_plans.py" \
        --frozen-mask "$ART/dypac_vla_libero.global.frozen.plan.json" \
        --inventory "$INVENTORY" --out-dir "$ART" >>"$CONTROL/freeze.log" 2>&1
    for suite in "${SUITES[@]}"; do
        "$PY" "$ROOT/scripts/tools/materialize_full_context_hessian_subset.py" \
            --parent "$ART/$suite/hessian_w4.npz" \
            --plan "$ART/$suite/dypac_vla_libero.frozen.plan.json" --model gr00t \
            --out "$ART/$suite/hessian_w4.frozen_subset.npz" >>"$CONTROL/freeze.log" 2>&1
    done
}

parity() {
    phase parity "suite-specific frozen packed-W4 artifacts"
    local suite_index suite gpu pids=() failed=0
    for suite_index in "${!SUITES[@]}"; do
        suite=${SUITES[$suite_index]}; gpu=${SCORE_GPUS[$(suite_worker_offset "$suite_index")]}
        (
            export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
            "$PY" "$ROOT/scripts/tools/check_libero_dypac_gr00t_fused_parity.py" \
                --hessian "$ART/$suite/hessian_w4.frozen_subset.npz" \
                --out "$ART/$suite/fused_parity.json"
        ) >"$CONTROL/parity_${suite}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    (( failed == 0 ))
}

require_base_artifacts
if outputimpact_ready; then
    phase outputimpact_reuse "validated 4-suite 116-layer global artifact"
else
    run_outputimpact
fi
if initial_mask_ready; then
    phase initial_mask_reuse "validated M0 and 117-candidate coordinate manifest"
else
    select_initial
fi
run_coordinate_scores
make_proposals
rank_top3
audit_noise_b
audit_candidate_states
freeze_and_deploy
parity
if [[ "${GR00T_DYPAC_START_EVAL:-0}" == 1 ]]; then
    phase canary "four suites, task0/state10"
    bash "$ROOT/scripts/run_libero_dypac_gr00t_eval.sh" canary all
    phase formal "start four 100-episode GR00T LIBERO cells"
    bash "$ROOT/scripts/run_libero_dypac_gr00t_eval.sh" formal all
fi
phase complete "GR00T full LIBERO DyPAC implementation complete"
