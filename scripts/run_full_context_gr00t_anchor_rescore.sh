#!/usr/bin/env bash
set -euo pipefail

# Offline re-scoring of the four v2 anchors on the expanded cross-context
# buffer. Runs the GR00T offline scorer once per split (one checkpoint per
# split), then ranks the anchors and emits the stop decision: if full-W4
# ranks no worse than the historical main mask on any split while the
# closed-loop quick gate showed main >> full-W4, the 296 layer flips must
# not be rerun.

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="${V2_ANCHOR_PY:-/home1/gyy/probe/miniforge3/envs/groot_test/bin/python}"
SCORER="$REPO_ROOT/scripts/tools/gr00t_score_outputimpact_plans.py"
SPLITTER="$REPO_ROOT/scripts/tools/split_v2_buffer_by_split.py"
RANKER="$REPO_ROOT/scripts/tools/rank_v2_anchors.py"
V2_ROOT="${FULL_CONTEXT_V2_ROOT:-$REPO_ROOT/runs/full_context_v2}"
SPEC_PATH="$V2_ROOT/selection_context_spec.json"
BUFFER_PATH="$V2_ROOT/selection_buffer.npz"
SPLIT_DIR="$V2_ROOT/selection_buffer_splits"
SCORE_DIR="$V2_ROOT/anchor_scores"

FULL_W4_PLAN="$REPO_ROOT/runs/full_context_v1/gr00t/round1/gr00t_full_context_round1_frozen.json"
BASE_FULL_W4_PLAN="${V2_ANCHOR_BASE_PLAN:-$REPO_ROOT/runs/full_context_v1/gr00t/round1/gr00t_full_context_round1_frozen.json}"
MAIN_MASK_PLAN="$REPO_ROOT/runs/full_context_v2/gr00t_main_pruned_to_table1_budget.json"
HESSIAN_ROOT="${V2_ANCHOR_HESSIAN_ROOT:-$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t}"
PACK_ROOT="${V2_ANCHOR_PACK_ROOT:-$REPO_ROOT/runs/errorfold_v3_15x20/calibration/gr00t}"

usage() {
    echo "usage: $0 split | score | rank | all" >&2
}

checkpoint_for() {
    case "$1" in
        atomic_seen)
            echo "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
            ;;
        composite_seen)
            echo "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
            ;;
        composite_unseen)
            echo "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000"
            ;;
    esac
}

split() {
    [[ -f "$SPEC_PATH" && -f "$BUFFER_PATH" ]] || {
        echo "missing spec or pooled buffer; run the v2 selection capture first" >&2
        return 1
    }
    "$PYTHON" "$SPLITTER" --spec "$SPEC_PATH" --buffer "$BUFFER_PATH" --out-dir "$SPLIT_DIR"
}

score() {
    [[ -f "$MAIN_MASK_PLAN" ]] || {
        echo "missing main-mask anchor plan; run prune_gr00t_main_to_budget.py first" >&2
        return 1
    }
    mkdir -p "$SCORE_DIR"
    for split in atomic_seen composite_seen composite_unseen; do
        echo "[v2-anchor-score] $split" >&2
        "$PYTHON" "$SCORER" \
            --checkpoint "$(checkpoint_for "$split")" \
            --base-full-w4-plan "$BASE_FULL_W4_PLAN" \
            --pack-dir "$PACK_ROOT/$split/identity_pack" \
            --hessian-w4 "$HESSIAN_ROOT/$split/hessian_w4.npz" \
            --activation-mode dynamic_a8 \
            --buffer "$SPLIT_DIR/selection_buffer_$split.npz" \
            --n-obs 48 \
            --candidate-plan "full_w4=$BASE_FULL_W4_PLAN" \
            --candidate-plan "main_mask=$MAIN_MASK_PLAN" \
            --out "$SCORE_DIR/scores_$split.json"
    done
}

rank() {
    "$PYTHON" "$RANKER" \
        --payloads "atomic_seen=$SCORE_DIR/scores_atomic_seen.json,composite_seen=$SCORE_DIR/scores_composite_seen.json,composite_unseen=$SCORE_DIR/scores_composite_unseen.json" \
        --out "$V2_ROOT/anchor_ranking.json"
}

case "${1:-}" in
    split) split ;;
    score) score ;;
    rank) rank ;;
    all) split && score && rank ;;
    *) usage ;;
esac