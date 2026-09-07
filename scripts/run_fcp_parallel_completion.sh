#!/usr/bin/env bash
set -uo pipefail

# Persistent, resource-only coordinator for the six frozen FCP diagnostics.
# Candidate plans, tasks, seeds, checkpoints, and statistics remain governed
# by run_fcp_candidate_completion.sh and the immutable preregistration.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
RUNNER="$REPO_ROOT/scripts/run_fcp_candidate_completion.sh"
LOG_PREFIX="[fcp/parallel]"
MAX_ATTEMPTS="${FCP_PARALLEL_MAX_ATTEMPTS:-5}"
RETRY_SECONDS="${FCP_PARALLEL_RETRY_SECONDS:-60}"

children=()

cleanup() {
    local pid
    trap - INT TERM
    for pid in "${children[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

run_with_retries() {
    local candidate="$1" gpu="$2" port="$3" shards="$4" clients="$5" egl_pool="${6:-$2}"
    local attempt rc candidate_pid=""
    stop_candidate() {
        trap - INT TERM
        if [[ -n "$candidate_pid" ]]; then
            kill -INT "$candidate_pid" 2>/dev/null || true
            wait "$candidate_pid" 2>/dev/null || true
        fi
        exit 130
    }
    trap stop_candidate INT TERM
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
        printf '%s candidate=%s attempt=%d/%d gpu=%s shards=%s clients=%s egl_pool=%s\n' \
            "$LOG_PREFIX" "$candidate" "$attempt" "$MAX_ATTEMPTS" \
            "$gpu" "$shards" "$clients" "$egl_pool"
        FCP_PREFLIGHT_DONE=1 FCP_N_SHARDS="$shards" FCP_MAX_CLIENTS="$clients" \
            "$RUNNER" run-candidate "$candidate" "$gpu" "$port" "$egl_pool" &
        candidate_pid="$!"
        if wait "$candidate_pid"; then
            candidate_pid=""
            printf '%s candidate=%s complete\n' "$LOG_PREFIX" "$candidate"
            return 0
        else
            rc=$?
            candidate_pid=""
        fi
        printf '%s candidate=%s exit=%d; retrying in %ss\n' \
            "$LOG_PREFIX" "$candidate" "$rc" "$RETRY_SECONDS" >&2
        sleep "$RETRY_SECONDS"
    done
    printf '%s candidate=%s exhausted retries\n' "$LOG_PREFIX" "$candidate" >&2
    return 1
}

cd "$REPO_ROOT" || exit 1

run_with_retries single_best 4 19700 10 8 4 & children+=("$!")
run_with_retries two_best 7 19707 8 8 7 & children+=("$!")
# Future Composite-Unseen manifests share their EGL clients with GPU 4 after
# single_best finishes. Existing split manifests retain their original pools.
run_with_retries attention_6 6 19706 10 10 6,4 & children+=("$!")
run_with_retries mlp_2 1 19701 7 7 1 & children+=("$!")
# GPU 3 gained a larger foreign process after Atomic-Seen. Five shards retain
# the runner's conservative 2-GiB guard for the two remaining splits.
run_with_retries ff_pair_0 3 19703 8 8 3,4 & children+=("$!")
run_with_retries dp_full_lambda_1p0 5 19705 8 8 5,4 & children+=("$!")

status=0
for pid in "${children[@]}"; do
    wait "$pid" || status=1
done
children=()

if [[ "$status" -eq 0 ]]; then
    "$RUNNER" aggregate
    printf '%s all candidates and aggregate complete\n' "$LOG_PREFIX"
fi
exit "$status"
