#!/usr/bin/env python3
"""Read-only, CPU-only replay of GR00T M0 selection and artifact consistency.

Writes a separate diagnostic report, never a plan, selection, or rollout result.
Threshold variants describe existing measurements; they must not reopen selection.
No model forward pass, GPU job, or success-based candidate selection is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from quantvla_full_context import paired_candidate_summary, select_frozen_candidate, sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/full_context_v2"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compact(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "components"} | {
        "components": {
            key: {k: v for k, v in row.items() if k != "delta"}
            for key, row in summary["components"].items()
        }
    }


def passes(summary: dict, *, se_factor: float = 1.0, all_only: bool = False,
           components: bool = True) -> bool:
    clusters = [row for metric in summary["metrics"].values()
                for row in metric["clusters"] if not all_only or row["cluster"] == "all"]
    return all(row["delta_mean"] + se_factor * row["delta_se"] < 0 for row in clusters) and (
        not components or all(row["mean"] + se_factor * row["se"] <= 0
                              for row in summary["components"].values()))


def replay() -> dict:
    manifest_path = RUN / "p2/interventions_dynamic/manifest.json"
    scores_path = RUN / "p2/flip_scores/flip_scores_cross_split.json"
    report_path = RUN / "fcp_completion/gr00t_corrected_fcp_frozen.json.selection.json"
    manifest, document, frozen = map(load, (manifest_path, scores_path, report_path))
    sources = {str(path.relative_to(ROOT)): sha256_file(path)
               for path in (manifest_path, scores_path, report_path, Path(__file__),
                            ROOT / "scripts/tools/quantvla_full_context.py")}
    require(document["complete"] is True, "Incomplete flip scores")
    by_id = {row["candidate_id"]: row for row in manifest["candidates"]}
    require(set(by_id) == set(document["scores"]), "Flip inventory mismatch")
    m0 = set(by_id["context_base"]["protected_layers"])
    rows = []
    for identifier, row in by_id.items():
        if not row.get("flip"):
            continue
        path = Path(row["path"])
        require(sha256_file(path) == row["sha256"], f"Flip plan drift: {identifier}")
        plan = load(path)
        protected = set(row["protected_layers"])
        require(protected ^ m0 == {row["flip"]["layer"]}, f"Not a single flip: {identifier}")
        active = {name for name, value in plan["layers"].items()
                  if not value.get("skip", False) and int(value.get("bits", 0) or 0) == 4}
        require(len(active) == document["scores"][identifier]["quantized_w4_layers"],
                f"Scored mask inventory drift: {identifier}")
        variable = manifest["all_w4_total_bytes"] + sum(
            manifest["byte_rows"][name]["extra_fp16_bytes"] for name in protected)
        require(variable == row["total_bytes"], f"Variable byte mismatch: {identifier}")
        summary = paired_candidate_summary(document["scores"][identifier], document["scores"]["context_base"])
        rows.append({"candidate_id": identifier, "flip": row["flip"],
                     "within_budget": variable <= manifest["budget_bytes"],
                     "static_bytes": plan["table1_total_static_bytes"],
                     "summary": compact(summary)})
    require(len(rows) == 116, "Expected all 116 flips")
    proposals = load(Path(frozen["proposal_manifest"]))
    full_scores = load(Path(frozen["full_network_scores"]))
    plan_rows = {row["candidate_id"]: row for row in proposals["candidates"]}
    for key in ("full_network_scores", "proposal_manifest", "frozen_plan", "selected_source", "finalizer"):
        require(sha256_file(Path(frozen[key])) == frozen[key + "_sha256"], f"Frozen source drift: {key}")
    for row in proposals["candidates"]:
        require(sha256_file(Path(row["path"])) == row["sha256"], "Proposal artifact drift")
    selected = select_frozen_candidate(scores=full_scores["scores"], baseline_id="context_base", plan_rows=plan_rows)
    require(selected == frozen["selection"], "Frozen selection replay differs")
    emitted = [{"candidate_id": identifier,
                "changed_layers_from_m0": len(set(row["protected_layers"]) ^ m0),
                "retained_fp16": len(row["protected_layers"]),
                "summary": compact(selected["summaries"][identifier])}
               for identifier, row in plan_rows.items() if identifier != "context_base"]
    variants = {
        "registered": {},
        "omit_component_guards": {"components": False},
        "zero_se_keep_components": {"se_factor": 0.0},
        "zero_se_omit_components": {"se_factor": 0.0, "components": False},
        "all_tasks_only_keep_components": {"all_only": True},
        "all_tasks_only_omit_components": {"all_only": True, "components": False},
    }
    counts = {
        name: {"single_flips": sum(row["within_budget"] and passes(row["summary"], **kwargs) for row in rows),
               "emitted_proposals": sum(passes(row["summary"], **kwargs) for row in emitted)}
        for name, kwargs in variants.items()
    }
    return {"sources": sources, "selection_replay_exact": True,
            "selected_id": selected["selected_id"], "single_flip_count": len(rows),
            "budget_feasible_single_flips": sum(row["within_budget"] for row in rows),
            "sensitivity_counts_posthoc_not_a_selection_rule": counts,
            "single_flips": sorted(rows, key=lambda row: row["summary"]["objective"]),
            "emitted_proposals": emitted}


def stream_hash(reader) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def runtime_audit(deep: bool) -> dict:
    full_scores = load(RUN / "fcp_completion/teacher_state/scores_cross_split.json")
    result = []
    parent_hashes, entry_hashes = {}, {}
    directory = RUN / "fcp_completion/closed_loop/matrix"
    for runtime_path in sorted(directory.glob("*/*/runtime_info.json")):
        identifier, split = runtime_path.parent.parent.name, runtime_path.parent.name
        runtime = load(runtime_path)[identifier]
        plan_path = Path(runtime["plan"])
        plan = load(plan_path)
        require(sha256_file(plan_path) == runtime["plan_sha256"], "Runtime plan hash drift")
        require(runtime["plan_sha256"] == full_scores["candidate_plans"][identifier]["sha256"],
                "Scoring/deployment mask mismatch")
        score_split = load(Path(full_scores["combined_splits"][split]["path"]))
        active = {name for name, row in plan["layers"].items()
                  if not row.get("skip", False) and int(row.get("bits", 0) or 0) == 4}
        require(len(active) == runtime["wrapped_layers"], "Runtime wrapped count mismatch")
        contract = runtime["quantization_contract"]
        require(contract["calibration_policy"] == "online_dynamic_per_forward_per_channel_amax", "A8 mode drift")
        require(runtime["denoising_steps"] == score_split["flow_steps"] == 4, "Flow-step drift")
        require(not runtime["runtime_selector"]["enabled"], "Unexpected runtime selector")
        require(not runtime["atm_enabled"] and not runtime["ohb_enabled"], "Unexpected output corrections")
        subset_path = Path(runtime["hessian_w4_path"])
        subset = load(Path(str(subset_path) + ".json"))
        parent_path = Path(subset["parent_hessian_path"])
        parent = load(Path(str(parent_path) + ".json"))
        require(subset["parent_hessian_sha256"] == score_split["hessian_w4_sha256"], "Offline weight lineage mismatch")
        require(set(subset["layer_names"]) == active, "Subset W4 inventory mismatch")
        require(subset["requantized"] is False, "Unexpected requantization")
        require(subset["deployment_plan_sha256"] == runtime["plan_sha256"], "Subset plan mismatch")
        if parent_path not in parent_hashes:
            parent_hashes[parent_path] = sha256_file(parent_path)
        require(parent_hashes[parent_path] == subset["parent_hessian_sha256"], "Parent hash mismatch")
        require(sha256_file(subset_path) == subset["npz_sha256"] == runtime["hessian_w4_sha256"], "Subset hash mismatch")
        checked = 0
        if deep:
            with zipfile.ZipFile(parent_path) as source, zipfile.ZipFile(subset_path) as target:
                actual_names = np.load(target.open("layer_names.npy"), allow_pickle=False).tolist()
                require(actual_names == subset["layer_names"], "NPZ name order mismatch")
                for new_index, name in enumerate(actual_names):
                    old_index = parent["layer_names"].index(name)
                    for prefix in ("packed", "scales", "clipping", "error"):
                        source_name, target_name = f"{prefix}_{old_index:04d}.npy", f"{prefix}_{new_index:04d}.npy"
                        key = (parent_path, source_name)
                        if key not in entry_hashes:
                            with source.open(source_name) as reader:
                                entry_hashes[key] = stream_hash(reader)
                        with target.open(target_name) as reader:
                            require(stream_hash(reader) == entry_hashes[key], f"Weight content differs: {identifier}/{split}/{name}/{prefix}")
                        checked += 1
        result.append({"candidate_id": identifier, "split": split, "w4_layers": len(active),
                       "runtime_info_sha256": sha256_file(runtime_path),
                       "parent_sha256": parent_hashes[parent_path], "subset_sha256": subset["npz_sha256"],
                       "verified_array_entries": checked, "status": "passed"})
        print(f"runtime checked: {identifier}/{split}", flush=True)
    require(len(result) == 18, "Expected six candidates across three splits")
    return {"configurations": result, "deep_array_comparison": deep,
            "action_output_parity": "not_tested_requires_model_forward",
            "scope": "mask identities, flow steps, activation rule, corrections, exact packed-array contents; not full action-output equivalence"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-weights", action="store_true")
    parser.add_argument("--out", type=Path, default=RUN / "m0_diagnostic_20260906/replay.json")
    args = parser.parse_args()
    output = args.out.resolve()
    require(output.parent.name.startswith("m0_diagnostic"), "Use a separate m0_diagnostic directory")
    data = {"kind": "posthoc_gr00t_m0_readonly_diagnostic", "selection_feedback_allowed": False,
            "gpu_jobs_started": False, "frozen_artifacts_modified": False, **replay()}
    print(json.dumps(data["sensitivity_counts_posthoc_not_a_selection_rule"], indent=2), flush=True)
    data["runtime_consistency"] = runtime_audit(args.deep_weights)
    data["valid"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"DIAGNOSTIC_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()