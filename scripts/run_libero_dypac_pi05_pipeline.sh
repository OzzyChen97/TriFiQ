#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/gyy/vla/QuantVLA
RUN_ROOT=${LIBERO_DYPAC_RUN_ROOT:-$ROOT/runs/libero_dypac_v1}
OPENPI_PY=/home1/gyy/probe/miniforge3/envs/openpi/bin/python
LIBERO_PY=/home1/gyy/probe/miniforge3/envs/libero_test/bin/python
CHECKPOINT=$ROOT/code/pi05/checkpoints/pi05_libero_pytorch
CHECKPOINT_SHA=0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74
ART=$RUN_ROOT/artifacts/pi05
CONTROL=$RUN_ROOT/control/pi05_pipeline
BUFFER=$RUN_ROOT/calibration/pi05/calibration_256.npz
SELECTION=$RUN_ROOT/calibration/pi05/selection_144.npz
INVENTORY=$ART/candidate_inventory.json
BASE_PLAN=$ART/all_w4.plan.json
PACK=$ART/identity_pack
HESSIAN=$ART/hessian_w4.npz
if [[ -n "${LIBERO_DYPAC_SELECTOR_GPUS:-}" ]]; then
    IFS=, read -r -a GPUS <<<"$LIBERO_DYPAC_SELECTOR_GPUS"
else
    mapfile -t GPUS < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 >= 12000 && $1 != 0) print $1}'
    )
fi
(( ${#GPUS[@]} > 0 )) || { echo "no GPU has the 12 GiB safety margin required by the π0.5 selector" >&2; exit 1; }
SHARDS=${#GPUS[@]}
if [[ -n "${LIBERO_DYPAC_FALLBACK_GPUS:-}" ]]; then
    IFS=, read -r -a FALLBACK_GPUS <<<"$LIBERO_DYPAC_FALLBACK_GPUS"
else
    FALLBACK_GPUS=(1 2 3 4 5 6 7)
fi

mkdir -p "$CONTROL"
exec 9>"$RUN_ROOT/control/pi05_pipeline.lock"
flock -n 9 || { echo "π0.5 LIBERO DyPAC pipeline is already active" >&2; exit 1; }

phase() {
    printf '%s phase=%s detail=%s\n' "$(date --iso-8601=seconds)" "$1" "${2:-}" | tee -a "$CONTROL/progress.log"
}

wait_for_hessian() {
    phase wait_hessian "$HESSIAN"
    while [[ ! -f "$HESSIAN" || ! -f "$HESSIAN.json" ]]; do sleep 20; done
    "$OPENPI_PY" - "$HESSIAN" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); m=json.loads(pathlib.Path(str(p)+'.json').read_text())
assert m['schema_version']==3 and m['group_size']==64 and len(m['layer_names'])==180
assert m['npz_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
PY
}

score_shard_complete() {
    local output=$1 kind=$2 shard=$3 shards=$4 manifest=${5:--} noise=${6:--}
    [[ -f "$output" ]] || return 1
    "$OPENPI_PY" - "$output" "$kind" "$shard" "$shards" "$manifest" "$noise" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output, kind, shard, shards, manifest, noise = sys.argv[1:]
try:
    value = json.loads(Path(output).read_text(encoding="utf-8"))
    valid = (
        value.get("complete") is True
        and value.get("kind") == kind
        and int(value.get("shard_index", -1)) == int(shard)
        and int(value.get("shard_count", -1)) == int(shards)
    )
    if manifest != "-":
        digest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        valid = valid and value.get("manifest_sha256") == digest
    if noise != "-":
        valid = valid and value.get("noise") == noise
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

available_selector_gpus() {
    local gpu free seen=,
    for gpu in "${GPUS[@]}" "${FALLBACK_GPUS[@]}"; do
        [[ "$seen" == *",$gpu,"* ]] && continue
        seen+="$gpu,"
        free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
        [[ "$free" =~ ^[0-9]+$ ]] || continue
        # The complete-mask scorer occupies about 15 GiB.  Keep a launch
        # margin above that measured peak;
        # if an external job races us, the incomplete shard is retried below.
        (( free >= 18000 )) && printf '%s %s\n' "$free" "$gpu"
    done | sort -nr | awk '{print $2}'
}

run_outputimpact() {
    phase outputimpact "$SHARDS GPU shards, 180 single-layer interventions"
    mkdir -p "$CONTROL/outputimpact"
    local -A attempts=()
    local pending=() safe_gpus=() pids=() launched=() shard gpu pid attempt
    while true; do
        pending=()
        for ((shard=0; shard<SHARDS; shard++)); do
            score_shard_complete "$CONTROL/outputimpact/shard_${shard}.json" \
                dypac_libero_pi05_single_layer_outputimpact "$shard" "$SHARDS" \
                || pending+=("$shard")
        done
        (( ${#pending[@]} > 0 )) || return 0
        mapfile -t safe_gpus < <(available_selector_gpus)
        if (( ${#safe_gpus[@]} == 0 )); then
            phase outputimpact_wait "no allowed GPU currently has 18 GiB free; retaining partial shards"
            sleep 20
            continue
        fi
        pids=(); launched=()
        for shard in "${pending[@]}"; do
            (( ${#launched[@]} < ${#safe_gpus[@]} )) || break
            attempt=$((${attempts[$shard]:-0} + 1))
            attempts[$shard]=$attempt
            (( attempt <= 8 )) || { echo "OutputImpact shard $shard failed eight resumable attempts" >&2; return 1; }
            gpu=${safe_gpus[${#launched[@]}]}
            phase outputimpact_shard "shard=$shard gpu=$gpu attempt=$attempt"
            (
                export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
                export PYTHONPATH="$ROOT/scripts/tools:$ROOT/code/pi05/openpi/src:$ROOT/code/pi05/openpi/packages/openpi-client/src"
                "$OPENPI_PY" -u "$ROOT/scripts/tools/probe_libero_dypac_pi05_outputimpact.py" \
                    --checkpoint-dir "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
                    --inventory "$INVENTORY" --plan "$BASE_PLAN" --pack-dir "$PACK" \
                    --hessian "$HESSIAN" --buffer "$SELECTION" \
                    --shard-index "$shard" --shard-count "$SHARDS" --batch-size 8 \
                    --out "$CONTROL/outputimpact/shard_${shard}.json"
            ) >"$CONTROL/outputimpact/shard_${shard}.attempt_${attempt}.log" 2>&1 &
            pids+=("$!"); launched+=("$shard")
        done
        for pid in "${pids[@]}"; do wait "$pid" || true; done
    done
}

select_initial() {
    phase initial_mask "reliability-shrunk D_PAC exact-byte DP"
    local scores=()
    for ((shard=0; shard<SHARDS; shard++)); do scores+=("$CONTROL/outputimpact/shard_${shard}.json"); done
    export PYTHONPATH="$ROOT/scripts/tools"
    "$OPENPI_PY" "$ROOT/scripts/tools/select_libero_dypac_initial_mask.py" \
        --inventory "$INVENTORY" --base-plan "$BASE_PLAN" --scores "${scores[@]}" \
        --out "$ART/initial_m0.plan.json" >"$CONTROL/initial_mask.log" 2>&1
    "$OPENPI_PY" "$ROOT/scripts/tools/make_libero_dypac_coordinate_manifest.py" \
        --initial-plan "$ART/initial_m0.plan.json" --inventory "$INVENTORY" \
        --out "$ART/coordinate_flips.manifest.json" >>"$CONTROL/initial_mask.log" 2>&1
}

run_mask_score_shards() {
    local manifest=$1 noise=$2 output_dir=$3 shards=$4 label=$5
    local -A attempts=()
    local pending=() safe_gpus=() pids=() launched=() shard gpu pid attempt
    mkdir -p "$output_dir"
    while true; do
        pending=()
        for ((shard=0; shard<shards; shard++)); do
            score_shard_complete "$output_dir/shard_${shard}.json" \
                dypac_libero_pi05_complete_mask_scores "$shard" "$shards" "$manifest" "$noise" \
                || pending+=("$shard")
        done
        (( ${#pending[@]} > 0 )) || return 0
        mapfile -t safe_gpus < <(available_selector_gpus)
        if (( ${#safe_gpus[@]} == 0 )); then
            phase "${label}_wait" "no allowed GPU currently has 18 GiB free; retaining partial shards"
            sleep 20
            continue
        fi
        pids=(); launched=()
        for shard in "${pending[@]}"; do
            (( ${#launched[@]} < ${#safe_gpus[@]} )) || break
            attempt=$((${attempts[$shard]:-0} + 1))
            attempts[$shard]=$attempt
            (( attempt <= 8 )) || { echo "$label shard $shard failed eight resumable attempts" >&2; return 1; }
            gpu=${safe_gpus[${#launched[@]}]}
            phase "${label}_shard" "shard=$shard gpu=$gpu attempt=$attempt"
            (
                export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
                export PYTHONPATH="$ROOT/scripts/tools:$ROOT/code/pi05/openpi/src:$ROOT/code/pi05/openpi/packages/openpi-client/src"
                "$OPENPI_PY" -u "$ROOT/scripts/tools/score_libero_dypac_pi05_masks.py" \
                    --checkpoint-dir "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
                    --inventory "$INVENTORY" --plan "$BASE_PLAN" --pack-dir "$PACK" \
                    --hessian "$HESSIAN" --buffer "$SELECTION" --manifest "$manifest" --noise "$noise" \
                    --shard-index "$shard" --shard-count "$shards" --batch-size 8 \
                    --out "$output_dir/shard_${shard}.json"
            ) >"$output_dir/shard_${shard}.attempt_${attempt}.log" 2>&1 &
            pids+=("$!"); launched+=("$shard")
        done
        for pid in "${pids[@]}"; do wait "$pid" || true; done
    done
}

run_coordinate_scores() {
    phase fcp_coordinate "$SHARDS GPU shards, complete-network 180 flips"
    run_mask_score_shards "$ART/coordinate_flips.manifest.json" A "$CONTROL/coordinate" "$SHARDS" fcp_coordinate
}

make_proposals() {
    phase fcp_proposals "exact-byte structured proposals"
    local scores=()
    for ((shard=0; shard<SHARDS; shard++)); do scores+=("$CONTROL/coordinate/shard_${shard}.json"); done
    export PYTHONPATH="$ROOT/scripts/tools"
    "$OPENPI_PY" "$ROOT/scripts/tools/propose_libero_dypac_fcp.py" \
        --coordinate-manifest "$ART/coordinate_flips.manifest.json" \
        --coordinate-scores "${scores[@]}" --initial-plan "$ART/initial_m0.plan.json" \
        --inventory "$INVENTORY" --out-dir "$ART/fcp_proposals" \
        >"$CONTROL/proposals.log" 2>&1
}

score_manifest() {
    local manifest=$1 noise=$2 output_dir=$3
    local count shards
    count=$("$OPENPI_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidates"]))' "$manifest")
    shards=$count; (( shards > SHARDS )) && shards=$SHARDS
    run_mask_score_shards "$manifest" "$noise" "$output_dir" "$shards" "fcp_${noise}"
}

rank_top3() {
    phase fcp_fullnet "score structured proposals and freeze Top-3 before noise B"
    score_manifest "$ART/fcp_proposals/manifest.json" A "$CONTROL/fullnet_a"
    local scores=("$CONTROL/fullnet_a"/shard_*.json)
    export PYTHONPATH="$ROOT/scripts/tools"
    "$OPENPI_PY" "$ROOT/scripts/tools/select_libero_dypac_fcp.py" rank \
        --proposals "$ART/fcp_proposals/manifest.json" --scores-a "${scores[@]}" \
        --out "$ART/fcp_top3.manifest.json" >"$CONTROL/top3.log" 2>&1
}

audit_noise_b() {
    phase fcp_noise_b "held-out noise B on frozen Top-3"
    score_manifest "$ART/fcp_top3.manifest.json" B "$CONTROL/top3_noise_b"
}

audit_candidate_states() {
    phase fcp_candidate_state "Top-3 on-policy state audit, no success labels"
    local jobs=$CONTROL/candidate_state_jobs.tsv
    "$OPENPI_PY" - "$ART/fcp_top3.manifest.json" >"$jobs" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for i,c in enumerate(d['top3']): print(i,c,d['candidates'][c]['path'],sep='\t')
PY
    local nonbaseline
    nonbaseline=$("$OPENPI_PY" - "$ART/fcp_top3.manifest.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(sum(identifier != 'context_base' for identifier in d['top3']))
PY
)
    if (( nonbaseline == 0 )); then
        phase fcp_candidate_state_short_circuit "baseline only; no alternative candidate exists to audit"
        export PYTHONPATH="$ROOT/scripts/tools"
        "$OPENPI_PY" - "$ART/fcp_top3.manifest.json" "$ART/fcp_candidate_state_audit.json" <<'PY'
import json,sys
from pathlib import Path
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file

top3_path=Path(sys.argv[1]).resolve(); top3=json.loads(top3_path.read_text())
assert top3['top3'] == ['context_base'] and set(top3['candidates']) == {'context_base'}
atomic_json(sys.argv[2], {
    'schema_version': 1,
    'kind': 'dypac_libero_candidate_state_audit',
    'protocol_id': PROTOCOL['protocol_id'],
    'protocol_sha256': sha256_file(PROTOCOL_PATH),
    'top3_manifest': str(top3_path),
    'top3_manifest_sha256': sha256_file(top3_path),
    'candidates': {
        'context_base': {
            'eligible': True,
            'baseline': True,
            'audit_required': False,
            'reason': 'no_nonbaseline_candidate_survived_noise_a_gate',
        }
    },
    'sources': [],
    'baseline_only_short_circuit': True,
    'uses_success_labels': False,
    'uses_test_rollout_feedback': False,
})
PY
        return 0
    fi
    local server_pids=() collector_pids=() audit_gpus=() index identifier plan gpu port job_count
    job_count=$(wc -l <"$jobs")
    while true; do
        mapfile -t audit_gpus < <(available_selector_gpus)
        (( ${#audit_gpus[@]} >= job_count )) && break
        phase fcp_candidate_state_wait "need=$job_count safe_gpus=${#audit_gpus[@]}"
        sleep 20
    done
    while IFS=$'\t' read -r index identifier plan; do
        gpu=${audit_gpus[$index]}; port=$((21400+index))
        (
            export CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1
            export PYTHONPATH="$ROOT/scripts/tools:$ROOT/code/pi05/openpi/src:$ROOT/code/pi05/openpi/packages/openpi-client/src"
            exec setsid "$OPENPI_PY" -u "$ROOT/scripts/tools/serve_libero_dypac_candidate_audit.py" \
                --checkpoint-dir "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
                --inventory "$INVENTORY" --plan "$BASE_PLAN" --candidate-plan "$plan" \
                --candidate-id "$identifier" --pack-dir "$PACK" --hessian "$HESSIAN" \
                --buffer "$SELECTION" --port "$port"
        ) >"$CONTROL/candidate_state_server_${identifier}.log" 2>&1 &
        server_pids+=("$!")
    done <"$jobs"
    for ((index=0; index<job_count; index++)); do
        port=$((21400+index))
        for _ in $(seq 1 240); do
            if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then break; fi
            sleep 2
        done
        curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null
    done
    while IFS=$'\t' read -r index identifier plan; do
        gpu=${audit_gpus[$index]}; port=$((21400+index))
        (
            export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$gpu"
            export LIBERO_CONFIG_PATH=/home1/gyy/.libero NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
            export PYTHONPATH="$ROOT/code/LIBERO:$ROOT/external/Omega-QVLA:$ROOT/code/pi05/openpi/packages/openpi-client/src:$ROOT/scripts/tools"
            "$LIBERO_PY" -u "$ROOT/scripts/tools/collect_libero_dypac_candidate_state.py" \
                --candidate-id "$identifier" --port "$port" --egl-device "$gpu" \
                --out "$CONTROL/candidate_state_${identifier}.json"
        ) >"$CONTROL/candidate_state_${identifier}.log" 2>&1 &
        collector_pids+=("$!")
    done <"$jobs"
    local failed=0 pid
    for pid in "${collector_pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    for pid in "${server_pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
    (( failed == 0 ))
    local score_paths=("$CONTROL"/candidate_state_*.json)
    export PYTHONPATH="$ROOT/scripts/tools"
    "$OPENPI_PY" "$ROOT/scripts/tools/merge_libero_dypac_candidate_state.py" \
        --top3 "$ART/fcp_top3.manifest.json" --scores "${score_paths[@]}" \
        --out "$ART/fcp_candidate_state_audit.json" >"$CONTROL/candidate_state_merge.log" 2>&1
}

freeze_and_deploy() {
    phase freeze "FCP fail-closed final mask and inventory-exact runtime"
    local scores_b=("$CONTROL/top3_noise_b"/shard_*.json)
    export PYTHONPATH="$ROOT/scripts/tools"
    "$OPENPI_PY" "$ROOT/scripts/tools/select_libero_dypac_fcp.py" freeze \
        --top3 "$ART/fcp_top3.manifest.json" --scores-b "${scores_b[@]}" \
        --candidate-state-audit "$ART/fcp_candidate_state_audit.json" \
        --out "$ART/dypac_vla_libero.frozen.plan.json" >"$CONTROL/freeze.log" 2>&1
    "$OPENPI_PY" "$ROOT/scripts/tools/materialize_full_context_hessian_subset.py" \
        --parent "$HESSIAN" --plan "$ART/dypac_vla_libero.frozen.plan.json" \
        --model pi05 --out "$ART/hessian_w4.frozen_subset.npz" >>"$CONTROL/freeze.log" 2>&1
}

parity_and_test() {
    phase parity "actual frozen Hessian subset, dynamic A8 fast kernel"
    local parity_gpus=()
    while true; do
        mapfile -t parity_gpus < <(available_selector_gpus)
        (( ${#parity_gpus[@]} > 0 )) && break
        phase parity_wait "no allowed GPU currently has 18 GiB free"
        sleep 20
    done
    export CUDA_VISIBLE_DEVICES=${parity_gpus[0]} PYTHONNOUSERSITE=1
    export PYTHONPATH="$ROOT/scripts/tools:$ROOT/code/pi05/openpi/src"
    "$OPENPI_PY" "$ROOT/scripts/tools/check_libero_dypac_pi05_triton_parity.py" \
        --hessian "$ART/hessian_w4.frozen_subset.npz" --out "$ART/triton_parity.json" \
        >"$CONTROL/parity.log" 2>&1
    phase canary "four suites in parallel, task0/state10"
    local canary_gpus=() canary_pids=() canary_failed=0 suite index
    while true; do
        mapfile -t canary_gpus < <(available_selector_gpus)
        (( ${#canary_gpus[@]} >= 4 )) && break
        phase canary_wait "need=4 safe_gpus=${#canary_gpus[@]}"
        sleep 20
    done
    local canary_suites=(goal spatial object long)
    for index in "${!canary_suites[@]}"; do
        suite=${canary_suites[$index]}
        (
            export LIBERO_DYPAC_CANARY_GPUS=${canary_gpus[$index]}
            export LIBERO_DYPAC_PORT_BASE=$((21600 + index * 40))
            bash "$ROOT/scripts/run_libero_dypac_pi05_eval.sh" canary "$suite"
        ) >"$CONTROL/canary_${suite}.log" 2>&1 &
        canary_pids+=("$!")
    done
    for index in "${!canary_pids[@]}"; do
        wait "${canary_pids[$index]}" || canary_failed=1
    done
    (( canary_failed == 0 ))
    for suite in goal spatial object long; do
        "$OPENPI_PY" "$ROOT/scripts/tools/validate_table6_libero_cell.py" \
            "$RUN_ROOT/canary/pi05/dypac_vla/$suite/merged_summary.json" --trials 1 --tasks 1 --offset 10
    done
    phase formal "start four 100-episode π0.5 LIBERO cells"
    bash "$ROOT/scripts/run_libero_dypac_pi05_eval.sh" formal all
    for suite in goal spatial object long; do
        "$OPENPI_PY" "$ROOT/scripts/tools/validate_table6_libero_cell.py" \
            "$RUN_ROOT/results/pi05/dypac_vla/$suite/merged_summary.json"
    done
}

wait_for_hessian
run_outputimpact
select_initial
run_coordinate_scores
make_proposals
rank_top3
audit_noise_b
audit_candidate_states
freeze_and_deploy
parity_and_test
phase complete "π0.5 full LIBERO DyPAC implementation and evaluation complete"
