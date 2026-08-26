#!/usr/bin/env python3
"""Fail-closed final audit for the v3 four-config ErrorFold experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fit_softfold_compensation import require_grid, score_rows, summarize_candidates
from quantvla_cross_model_protocol import (
    PROTOCOL,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    canonical_hash,
    require_protocol_attestation,
    sha256_file,
    validate_closed_loop_row,
)


REPO = Path(__file__).resolve().parents[2]
CONFIGS = tuple(row["id"] for row in PROTOCOL["evaluation_matrix"]["configs"])


def _record(failures: list[dict[str, str]], name: str, fn) -> None:
    try:
        fn()
    except Exception as error:
        failures.append({"check": name, "error": str(error)})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit_frozen_manifest(root: Path) -> None:
    frozen = root / "preregistered_manifest.json"
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    _require(canonical_hash(payload) == PROTOCOL_SHA256, "frozen manifest drift")
    _require(payload["evaluation_matrix"]["closed_loop_success_used_for_selection"] is False, "success leaked into selection")


def _audit_hessian(path: Path) -> None:
    sidecar = Path(str(path) + ".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    _require(meta.get("schema_version") == 3, "Hessian artifact schema drift")
    _require(meta.get("group_size") == 64, "Hessian artifact is not group-64")
    _require(meta.get("npz_sha256") == sha256_file(path), "Hessian artifact hash drift")
    _require(meta.get("packed_weight_bytes", 0) > 0, "Hessian artifact has zero packed bytes")
    _require(meta.get("gradient_updates") is False, "gradient update declared")
    _require(meta.get("fp16_weight_updates") is False, "FP16 weight update declared")


def _audit_grid(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_protocol_attestation(payload, source=str(path))
    _require(payload.get("schema_version") == 3, "grid schema drift")
    summaries = summarize_candidates(score_rows(payload), "d_pac_v2")
    require_grid(summaries)
    _require(len(summaries) == 81, "grid is not exactly 9x9")


def _audit_selected(path: Path, metric: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_protocol_attestation(payload, source=str(path))
    _require(payload.get("schema_version") == 3, "selected ErrorFold schema drift")
    _require(payload.get("metric") == metric, f"selected metric is not {metric}")
    _require(payload.get("selection", {}).get("paired_one_standard_error") is True, "paired one-SE missing")
    _require(payload.get("selection", {}).get("uses_rollout_success") is False, "success leaked into selection")


def _audit_calibration_lineage(path: Path) -> None:
    directory = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require_protocol_attestation(manifest, source=str(path))
    _require(manifest.get("schema_version") == 3, "calibration schema drift")
    _require(manifest.get("success_labels_used") is False, "success label leaked into calibration")
    _require(manifest.get("gradient_updates") is False, "gradient update declared")
    _require(manifest.get("fp16_weight_updates") is False, "FP16 weight update declared")
    source_paths = {
        "calibrator": REPO / "scripts/tools/calibrate_errorfold_v3.py",
        "capture": REPO / "scripts/tools/quantvla_v3_capture.py",
        "model_adapter": REPO / "scripts/tools/quantvla_model_adapters.py",
        "a8_builder": REPO / "scripts/tools/build_v3_a8_artifact.py",
        "hessian_builder": REPO / "scripts/tools/build_hessian_w4_artifact.py",
        "errorfold_builder": REPO / "scripts/tools/build_errorfold_artifact.py",
        "hessian_method": REPO / "scripts/tools/quantvla_hessian_w4.py",
        "errorfold_method": REPO / "scripts/tools/quantvla_errorfold.py",
    }
    expected_sources = {name: sha256_file(value) for name, value in source_paths.items()}
    _require(manifest.get("source_sha256") == expected_sources, "calibration source lineage drift")
    for name, row in (manifest.get("artifacts") or {}).items():
        artifact = Path(row["path"])
        _require(artifact.is_file(), f"calibration artifact missing: {name}/{artifact}")
        _require(sha256_file(artifact) == row["sha256"], f"calibration artifact hash drift: {name}")
    identity = json.loads((directory / "identity_pack/manifest.json").read_text(encoding="utf-8"))
    _require(identity.get("protocol_sha256") == PROTOCOL_SHA256, "identity pack protocol drift")
    _require(identity.get("plan_sha256") == manifest.get("plan_sha256"), "identity pack plan drift")

    fp16 = directory / "fp16_capture.npz"
    paired = directory / "paired_errorfold_capture.npz"
    hessian = json.loads((directory / "hessian_w4.npz.json").read_text(encoding="utf-8"))
    raw = json.loads((directory / "raw_errorfold.json").read_text(encoding="utf-8"))
    raw_meta = raw.get("meta") or {}
    _require(hessian.get("capture_sha256") == sha256_file(fp16), "Hessian/FP16 capture lineage drift")
    _require(raw_meta.get("capture_sha256") == sha256_file(paired), "ErrorFold/paired capture lineage drift")
    _require(raw_meta.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"), "raw correction checkpoint drift")
    _require(raw_meta.get("plan_sha256") == manifest.get("plan_sha256"), "raw correction plan drift")
    _require(
        raw_meta.get("calibration_buffer_sha256")
        == PROTOCOL["data"]["calibration_buffer"]["sha256"],
        "raw correction buffer drift",
    )

    merged_path = directory / "scores_merged.json"
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    _require(merged.get("raw_correction_sha256") == sha256_file(directory / "raw_errorfold.json"), "grid/raw lineage drift")
    _require(merged.get("hessian_w4_sha256") == sha256_file(directory / "hessian_w4.npz"), "grid/Hessian lineage drift")
    _require(merged.get("plan_sha256") == manifest.get("plan_sha256"), "grid plan lineage drift")
    _require(merged.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"), "grid checkpoint lineage drift")
    for filename in ("errorfold_dfunc.json", "errorfold_dpac_v2.json"):
        selected = json.loads((directory / filename).read_text(encoding="utf-8"))
        meta = selected.get("meta") or {}
        _require(meta.get("raw_correction_sha256") == sha256_file(directory / "raw_errorfold.json"), f"{filename}: raw lineage drift")
        _require(meta.get("validation_scores_sha256") == sha256_file(merged_path), f"{filename}: score lineage drift")
        _require(selected.get("teacher_checkpoint_sha256") == manifest.get("checkpoint_sha256"), f"{filename}: teacher drift")
        _require(selected.get("quant_plan_sha256") == manifest.get("plan_sha256"), f"{filename}: plan drift")
        _require(selected.get("buffer_sha256") == PROTOCOL["data"]["calibration_buffer"]["sha256"], f"{filename}: buffer drift")


def _closed_loop_rows(root: Path, model: str) -> dict[str, list[tuple[str, int]]]:
    patterns = (
        list((root / "closed_loop/gr00t").glob("**/*.jsonl"))
        if model == "gr00t"
        else list((root / "closed_loop/pi05/results").glob("**/*.jsonl"))
    )
    rows: dict[str, list[tuple[str, int]]] = {config: [] for config in CONFIGS}
    for path in patterns:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            config = str(row.get("config") or row.get("config_id"))
            if config not in rows:
                continue
            validate_closed_loop_row(row, source=f"{path}:{line_number}")
            rows[config].append((str(row["task"]), int(row["seed"])))
    return rows


def _audit_matrix(root: Path, model: str) -> None:
    expected = {
        (task, int(seed))
        for tasks in PROTOCOL["closed_loop"]["tasks"].values()
        for task in tasks
        for seed in PROTOCOL["closed_loop"]["seeds"]
    }
    rows = _closed_loop_rows(root, model)
    for config in CONFIGS:
        actual = rows[config]
        _require(
            len(actual) == 300 and set(actual) == expected,
            f"{model}/{config} matrix rows={len(actual)} unique={len(set(actual))}",
        )


def _audit_noise_b(root: Path, model: str) -> None:
    paths = sorted((root / "calibration" / model).glob("**/noise_b_audit.json"))
    expected_count = 3 if model == "gr00t" else 1
    _require(len(paths) == expected_count, f"{model}: noise-B audit count={len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require(payload.get("selection_frozen_before_noise_B") is True, f"{path}: selection not frozen")
        _require("heldout_noise_B" in payload, f"{path}: heldout noise-B result missing")
        _require(payload.get("noise_B_used_for_selection") is False, f"{path}: noise-B leaked into selection")
        _require(len(payload.get("candidates") or {}) == 2, f"{path}: frozen candidate count drift")


def _audit_deployment(root: Path, model: str) -> None:
    reports = sorted((root / "deployment" / model).glob("*.json"))
    expected = (
        {(task_set, config) for task_set in PROTOCOL["closed_loop"]["tasks"] for config in CONFIGS}
        if model == "gr00t"
        else {(None, config) for config in CONFIGS}
    )
    actual = set()
    for path in reports:
        row = json.loads(path.read_text(encoding="utf-8"))
        actual.add((row.get("task_set"), row.get("config_id")))
        _require(
            int(row.get("theoretical_static_bytes", -1))
            == int(row.get("packed_weight_bytes", 0))
            + int(row.get("auxiliary_static_bytes", 0)),
            f"{path}: static byte accounting drift",
        )
        if row.get("config_id") == "fp16":
            continue
        _require(row.get("packed_low_bit_residency") is True, f"{path}: W4 not resident")
        _require(row.get("packed_weight_bytes", 0) > 0, f"{path}: packed bytes zero")
        _require(row.get("fp_weight_sized_buffers") == 0, f"{path}: FP weight copy retained")
        _require(row.get("weight_bits") == 4 and row.get("activation_bits") == 8, f"{path}: not W4A8")
        _require(row.get("block_in") == 64 and row.get("block_out") == 64, f"{path}: group/block drift")
        _require(row.get("gpu_memory_bytes", {}).get("peak_allocated", 0) > 0, f"{path}: peak memory missing")
        _require(row.get("latency_ms", {}).get("mean", 0) > 0, f"{path}: latency missing")
        if str(row.get("config_id", "")).startswith("errorfold_"):
            _require(row.get("hessian_group_size") == 64, f"{path}: Hessian group drift")
            _require(row.get("hessian_w4_loaded") == row.get("wrapped_layers") > 0, f"{path}: Hessian inventory drift")
            _require(row.get("row_rotation") == "0", f"{path}: non-identity row rotation")
            _require(row.get("runtime_selector_enabled") is False, f"{path}: selector enabled")
        if row.get("compression_ratio_to_quantvla", 1.0) > 1.05:
            _require(row.get("compression_rate_claim_allowed") is False, f"{path}: invalid compression claim")
    _require(actual == expected, f"{model}: deployment report matrix drift")


def _audit_statistics(root: Path) -> None:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    require_protocol_attestation(summary, source=str(root / "summary.json"))
    _require(summary.get("episodes_per_config") == 300, "summary episode count drift")
    _require(set(summary.get("models") or {}) == set(PROTOCOL["models"]), "summary model set drift")
    for model, model_row in summary["models"].items():
        configs = model_row.get("configs") or {}
        _require(set(configs) == set(CONFIGS), f"{model}: summary config drift")
        _require(all(row.get("episodes") == 300 for row in configs.values()), f"{model}: incomplete summary")
        comparisons = model_row.get("preregistered_comparisons") or {}
        _require(len(comparisons) == 4, f"{model}: comparison family drift")
        for name, row in comparisons.items():
            _require(row.get("paired_episodes") == 300, f"{model}/{name}: not paired 300")
            _require("mcnemar_exact_two_sided_p" in row, f"{model}/{name}: McNemar missing")
            _require(row.get("holm_family_size") == 4 and "holm_adjusted_p" in row, f"{model}/{name}: Holm missing")
    provenance = json.loads((root / "source_provenance.json").read_text(encoding="utf-8"))
    _require(len(provenance.get("sources") or {}) == 2 * len(CONFIGS), "statistics source lineage drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(REPO / "runs/errorfold_v3_15x20"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    failures: list[dict[str, str]] = []
    _record(failures, "frozen_manifest", lambda: _audit_frozen_manifest(root))
    hessian_paths = sorted((root / "calibration").glob("**/hessian_w4.npz"))
    _record(failures, "hessian_artifact_count", lambda: _require(len(hessian_paths) == 4, f"found {len(hessian_paths)} Hessian artifacts"))
    for path in hessian_paths:
        _record(failures, f"hessian:{path.relative_to(root)}", lambda value=path: _audit_hessian(value))
    calibration_manifests = sorted((root / "calibration").glob("**/calibration_manifest.json"))
    _record(failures, "calibration_lineage_count", lambda: _require(len(calibration_manifests) == 4, f"found {len(calibration_manifests)} calibration manifests"))
    for path in calibration_manifests:
        _record(failures, f"lineage:{path.relative_to(root)}", lambda value=path: _audit_calibration_lineage(value))
    score_paths = sorted((root / "calibration").glob("**/scores_merged.json"))
    _record(failures, "grid_artifact_count", lambda: _require(len(score_paths) == 4, f"found {len(score_paths)} grids"))
    for path in score_paths:
        _record(failures, f"grid:{path.relative_to(root)}", lambda value=path: _audit_grid(value))
    for metric, filename in (("d_func_v1", "errorfold_dfunc.json"), ("d_pac_v2", "errorfold_dpac_v2.json")):
        paths = sorted((root / "calibration").glob(f"**/{filename}"))
        _record(failures, f"selected_count:{filename}", lambda values=paths: _require(len(values) == 4, f"found {len(values)} {filename}"))
        for path in paths:
            _record(failures, f"selected:{path.relative_to(root)}", lambda value=path, name=metric: _audit_selected(value, name))
    for model in PROTOCOL["models"]:
        _record(failures, f"noise_b:{model}", lambda value=model: _audit_noise_b(root, value))
        _record(failures, f"matrix:{model}", lambda value=model: _audit_matrix(root, value))
        _record(failures, f"deployment:{model}", lambda value=model: _audit_deployment(root, value))
    _record(failures, "statistics", lambda: _audit_statistics(root))
    payload = {
        "schema_version": 3,
        "kind": "errorfold_v3_final_audit",
        "run_root": str(root),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": PROTOCOL_SHA256,
        "complete": not failures,
        "failures": failures,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
