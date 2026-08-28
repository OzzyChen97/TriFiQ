#!/usr/bin/env bash
set -euo pipefail

# GR00T v2 full-context Table-1 matrix (strict held-out 50-task x 50-seed).
# Candidate = frozen v2 winner on the common runtime; baselines reuse the
# official paired-50 runs (fp16 / quantvla_w4a8 / cscka gdsq main).

REPO_ROOT="/home1/gyy/vla/QuantVLA"
PYTHON="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
export PYTHONPATH="$REPO_ROOT/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
RUNNER="$REPO_ROOT/scripts/tools/run_robocasa_atomic_matrix.py"
MATERIALIZER="$REPO_ROOT/scripts/tools/materialize_full_context_table1.py"
AGGREGATOR="$REPO_ROOT/scripts/tools/aggregate_full_context_table1.py"

ROOT="${FULL_CONTEXT_GR00T_TABLE1_ROOT:-$REPO_ROOT/runs/full_context_v2/table1}"
MANIFEST="$ROOT/manifest.json"
BASELINES="$ROOT/baselines"
RESULTS="$ROOT/results"
SPEC_ROOT="$ROOT/specs"
FROZEN_WINNER="$REPO_ROOT/runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
ACTIVATION="$REPO_ROOT/runs/full_context_v2/p2/activation_attribution.json"
QUICK_REPORT="$REPO_ROOT/runs/full_context_v2/quick/aggregate.json"
CANDIDATE_GPU="${FULL_CONTEXT_GR00T_TABLE1_GPU:-5}"
EGL_POOL="${FULL_CONTEXT_GR00T_TABLE1_EGL_POOL:-1,4,7}"
SEED_SHARDS="${FULL_CONTEXT_GR00T_TABLE1_SEED_SHARDS:-1}"
SEEDS="$(seq -s, 0 49)"
SPLIT_DIRS=(atomic composite_seen composite_unseen)
SPLIT_SETS=(atomic_seen composite_seen composite_unseen)

usage() {
    echo "usage: $0 prepare | assemble-baselines | run | status | aggregate" >&2
}

rewrite_placement() {
    local source="$1" target="$2" subset="$3"
    "$PYTHON" - "$source" "$target" "$subset" <<'PY'
import json
import os
import sys
import tempfile

source, target, subset = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
placements = {
    "fp16": (5, 19570),
    "quantvla_w4a8": (5, 19571),
    "gdsq_vla_main": (6, 19572),
    "full_context_v2": (6, 19573),
}
wanted = {item.strip() for item in subset.split(",") if item.strip()}
payload["configs"] = [row for row in payload["configs"] if row["id"] in wanted]
for row in payload["configs"]:
    row["gpu"], row["port"] = placements[row["id"]]
payload["config_subset"] = sorted(wanted)
directory = os.path.dirname(target)
os.makedirs(directory, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".execution.", suffix=".json", dir=directory)
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

build_specs() {
    "$PYTHON" - "$SPEC_ROOT" "$FROZEN_WINNER" <<'PY'
import json
import sys
from pathlib import Path

spec_root, winner_path = Path(sys.argv[1]), Path(sys.argv[2])
from quantvla_cross_model_protocol import sha256_file

spec_root.mkdir(parents=True, exist_ok=True)
calib = Path("/home1/gyy/vla/QuantVLA/runs/errorfold_v3_15x20/calibration/gr00t")
subset = Path("/home1/gyy/vla/QuantVLA/runs/full_context_v2/gr00t_main_hessian")
repo = Path("/home1/gyy/vla/QuantVLA")
winner = json.loads(winner_path.read_text(encoding="utf-8"))
expected_wrapped = int(winner["quantized_w4_layers"])
quick_specs = {
    split: json.loads((repo / "runs/full_context_v2/quick/specs" / f"{split}.json").read_text())
    for split in ("atomic_seen", "composite_seen", "composite_unseen")
}
official_names = {"atomic_seen": "atomic", "composite_seen": "composite_seen", "composite_unseen": "composite_unseen"}

for split in ("atomic_seen", "composite_seen", "composite_unseen"):
    official = json.loads(
        (repo / "runs" / f"robocasa365_official_full_{official_names[split]}_spec.json").read_text()
    )
    official_rows = {row["id"]: row for row in official["configs"]}
    quantvla = dict(official_rows["w4a8_atmohb"])
    for key in ("gpu", "port"):
        quantvla.pop(key, None)
    quantvla.update(
        {
            "id": "quantvla_w4a8",
            "activation_mode": "static_a8",
            "act_scale": str(
                repo / "runs/full_context_v2/table1/a8" / f"quantvla_{split}.npz"
            ),
            "meta": {"role": "table1_quantvla_baseline"},
        }
    )
    gdsq = None
    for row in quick_specs[split]["configs"]:
        if row["id"] == "gdsq_main":
            gdsq = {k: v for k, v in row.items() if k not in ("gpu", "port")}
    if gdsq is None:
        raise SystemExit(f"{split}: quick spec lacks gdsq_main")
    gdsq.update({"id": "gdsq_vla_main", "meta": {"role": "table1_gdsq_vla_main_baseline"}})
    hessian = subset / split / "hessian_w4.npz"
    candidate = {
        "id": "full_context_v2",
        "expected_wrapped": expected_wrapped,
        "plan": str(winner_path),
        "packdir": str(calib / split / "identity_pack"),
        "hessian_w4": str(hessian),
        "act_scale": None,
        "errorfold": None,
        "omega_pack": None,
        "omega_pack_attestation": None,
        "omega_calibration_manifest": None,
        "omega_include": None,
        "atm": None,
        "ohb": False,
        "ohb_only": False,
        "activation_mode": "dynamic_a8",
        "meta": {
            "role": "frozen_full_context_v2_table1_candidate",
            "formal_failure_on_crash": True,
        },
    }
    fp16 = {
        "id": "fp16",
        "expected_wrapped": 0,
        "plan": None,
        "packdir": None,
        "hessian_w4": None,
        "act_scale": None,
        "errorfold": None,
        "omega_pack": None,
        "omega_pack_attestation": None,
        "omega_calibration_manifest": None,
        "omega_include": None,
        "atm": None,
        "ohb": False,
        "ohb_only": False,
        "activation_mode": "fp16",
        "meta": {"role": "table1_fp16_baseline", "formal_failure_on_crash": True},
    }
    spec = {
        "schema_version": 1,
        "kind": "full_context_v2_table1_execution_spec",
        "purpose": "table1_four_config_matrix",
        "task_set": split,
        "seeds": "0-49",
        "configs": [fp16, quantvla, gdsq, candidate],
        "comparisons": {
            "pairs": [["full_context_v2", "fp16"], ["full_context_v2", "quantvla_w4a8"],
                      ["full_context_v2", "gdsq_vla_main"]],
            "unit": "paired_task_seed",
        },
        "decision": {"role": "table1_formal", "result_feedback_allowed": False},
    }
    (spec_root / f"{split}.json").write_text(json.dumps(spec, indent=2) + "\n")
print("[table1] specs built")
PY
}

prepare() {
    mkdir -p "$ROOT"
    if [[ ! -f "$MANIFEST" ]]; then
        "$PYTHON" "$MATERIALIZER" \
            --model gr00t \
            --frozen-plan "$FROZEN_WINNER" \
            --activation-attribution "$ACTIVATION" \
            --quick-report "$QUICK_REPORT" \
            --checkpoint "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000" \
            --hessian-w4 "$REPO_ROOT/runs/full_context_v2/gr00t_main_hessian/atomic_seen/hessian_w4.npz" \
            --out "$MANIFEST"
    fi
    build_specs
}

assemble_baselines() {
    echo "[table1] baselines run fresh inside the four-config matrix (assemble-baselines is a no-op)" >&2
    return 0
}

run_split() {
    local split="$1" tasks="$2" subset="$3"
    local tag
    tag="$(echo "$subset" | tr ',' '_')"
    local frozen_spec="$SPEC_ROOT/$split.json"
    local execution_spec="$SPEC_ROOT/.execution-${split}_${tag}.json"
    local run_dir="$RESULTS/matrix/${split}_${tag}"
    rewrite_placement "$frozen_spec" "$execution_spec" "$subset"
    "$PYTHON" "$RUNNER" \
        --spec "$execution_spec" \
        --run-dir "$run_dir" \
        --phase formal \
        --seeds "$SEEDS" \
        --checkpoint "$REPO_ROOT/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/$split/checkpoint-60000" \
        --task-set "$split" \
        --tasks "$tasks" \
        --n-shards "$(awk -F, '{print NF}' <<<"$tasks")" \
        --seed-shards-per-task 1 \
        --egl-device-pool "$EGL_POOL" \
        --trial-batch-size 5 \
        --action-noise paired \
        --formal-provenance-v2 \
        --allow-shared-gpus
}

collect_split() {
    local split="$1" subset="$2"
    local tag
    tag="$(echo "$subset" | tr ',' '_')"
    local run_dir="$RESULTS/matrix/${split}_${tag}"
    for config in ${subset//,/ }; do
        mkdir -p "$RESULTS/$config/$split"
        cp -n "$run_dir"/"${config}"_s*.jsonl "$RESULTS/$config/$split/" || true
    done
}

run_all() {
    prepare
    mkdir -p "$RESULTS"
    local tasks_file="$ROOT/.tasks.txt"
    "$PYTHON" - > "$tasks_file" <<'PY'
import json
from quantvla_full_context import PROTOCOL

for split in ("atomic_seen", "composite_seen", "composite_unseen"):
    print(f"{split} {','.join(PROTOCOL['table1']['tasks'][split])}")
PY
    while read -r split tasks; do
        [[ -n "$split" ]] || continue
        # Two waves per split: the paper-critical pair first, then the
        # reference baselines.  Each wave runs two model servers (GPU 5/6)
        # with its EGL clients spread over the server-free pool devices.
        run_split "$split" "$tasks" "full_context_v2,gdsq_vla_main"
        collect_split "$split" "full_context_v2,gdsq_vla_main"
        run_split "$split" "$tasks" "quantvla_w4a8,fp16"
        collect_split "$split" "quantvla_w4a8,fp16"
    done < "$tasks_file"
    aggregate
}

status() {
    "$PYTHON" - "$RESULTS" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys

counts = Counter()
for path in Path(sys.argv[1]).glob("**/*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "complete":
            counts[str(row.get("config"))] += 1
for config in ("fp16", "quantvla_w4a8", "gdsq_vla_main", "full_context_v2"):
    print(f"{config}={counts[config]}/2500", end=" ")
print()
PY
}

aggregate() {
    "$PYTHON" "$AGGREGATOR" \
        --manifest "$MANIFEST" \
        --candidate-dir "$RESULTS/full_context_v2" \
        --baseline "fp16=$RESULTS/fp16" \
        --baseline "quantvla_w4a8=$RESULTS/quantvla_w4a8" \
        --baseline "gdsq_vla_main=$RESULTS/gdsq_vla_main" \
        --out "$ROOT/aggregate.json"
}

case "${1:-}" in
    prepare) prepare ;;
    assemble-baselines) assemble_baselines ;;
    run) run_all ;;
    status) status ;;
    aggregate) aggregate ;;
    *) usage; exit 2 ;;
esac
