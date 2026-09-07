#!/usr/bin/env python3
"""Gate and render ActQuant Table-1 evidence plus the QVLA failure disclosure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "gdsq_vla_iclr2027"
AGGREGATE = ROOT / "runs" / "qvla_actquant_table1" / "formal" / "aggregate.json"
PARTIAL = (
    ROOT
    / "runs"
    / "qvla_actquant_table1"
    / "formal"
    / "partial"
    / "table1_partial_snapshot.json"
)
BEGIN = "<!-- BEGIN QVLA_ACTQUANT_TABLE1 -->"
END = "<!-- END QVLA_ACTQUANT_TABLE1 -->"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def replace_block(path: Path, body: str) -> None:
    original = path.read_text(encoding="utf-8")
    block = f"{BEGIN}\n{body.rstrip()}\n{END}"
    if BEGIN in original or END in original:
        require(original.count(BEGIN) == 1 and original.count(END) == 1, f"broken marker block: {path}")
        start = original.index(BEGIN)
        stop = original.index(END, start) + len(END)
        updated = original[:start] + block + original[stop:]
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    atomic_text(path, updated)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def validate(value: dict[str, Any]) -> None:
    require(value.get("complete") is True, "aggregate incomplete")
    require(value.get("ready_for_table_update") is True, "table update gate disabled")
    require(value.get("formal_episode_count_new") == 5_000, "formal episode count drift")
    require(value.get("bootstrap_draws") == 10_000, "bootstrap draw drift")
    require(len(value.get("holm_family") or []) == 2, "Holm family drift")
    arms = value.get("arms") or {}
    require(
        set(arms)
        == {"actquant_gr00t", "actquant_pi05", "dypac_gr00t", "dypac_pi05"},
        "ActQuant arm inventory drift",
    )
    for model in ("gr00t", "pi05"):
        key = f"actquant_{model}"
        row = arms[key]
        require(row.get("episodes") == 2500, f"{key} coverage drift")
        require(len(row.get("per_task_success_rate") or {}) == 50, f"{key} task drift")
        require(row.get("storage") is not None, f"{key} storage missing")


def validate_partial(value: dict[str, Any]) -> None:
    require(
        value.get("kind")
        == "qvla_actquant_robocasa365_table1_partial_snapshot",
        "partial snapshot kind drift",
    )
    require(value.get("complete") is False, "partial snapshot claims completion")
    require(
        value.get("ready_for_final_table_update") is False
        and value.get("ready_for_partial_table_update") is True,
        "partial/final table gates drift",
    )
    require(
        value.get("paired_inference_reported") is False,
        "partial snapshot reports paired inference",
    )
    require(
        value.get("test_results_used_for_pack_or_method_selection") is False,
        "partial snapshot permits test-result tuning",
    )
    observed = 0
    for method in ("qvla", "actquant"):
        for model in ("gr00t", "pi05"):
            key = f"{method}_{model}"
            row = (value.get("arms") or {}).get(key) or {}
            episodes = int(row.get("episodes", 0))
            require(0 < episodes < 2500, f"{key} partial coverage drift")
            require(row.get("episodes_expected") == 2500, f"{key} expected coverage drift")
            require(0 < int(row.get("tasks_observed", 0)) <= 50, f"{key} task drift")
            require(row.get("storage") is not None, f"{key} storage missing")
            canonical = Path(row["canonical_jsonl"])
            require(
                canonical.is_file()
                and sha256_file(canonical) == row["canonical_jsonl_sha256"],
                f"{key} canonical snapshot drift",
            )
            observed += episodes
    require(
        observed == value.get("formal_episode_count_observed")
        and value.get("formal_episode_count_expected") == 10_000,
        "partial total coverage drift",
    )


def final_versions_block(value: dict[str, Any]) -> str:
    lines = [
        "## ActQuant RoboCasa365 Table-1 extension",
        "",
        f"- Aggregate: `runs/qvla_actquant_table1/formal/aggregate.json` (sha256 `{sha256_file(AGGREGATE)}`).",
        "- Calibration: `fp16_teacher_proxy`; `source_protocol_equivalent=false`; formal seeds 0--49 were excluded from calibration.",
        "- ActQuant follows the pinned official GitHub implementation for its allocation and quantization semantics; local code supplies the GR00T/pi0.5 and RoboCasa365 adapters.",
        "- The frozen ActQuant service source is archived under `runs/qvla_actquant_table1/frozen_sources/` and verified against the implementation hash recorded before rollout.",
        "- Coverage: two new rows, each 50 tasks × 50 paired seeds = 2,500 episodes; 5,000/5,000 new formal episodes total.",
        "- QVLA-code is omitted from Table 1 after the GitHub-based RoboCasa365 cross-architecture reproduction produced zero successes before completion; the public release lacks the target architecture/benchmark paths, so the appendix records that incomplete open sourcing cannot be distinguished from an error in the released code path.",
        "- Reported size is measured static pack/GGUF storage only. No PyTorch runtime-memory, C++ runtime, or latency claim is made.",
        "",
        "| Method/model | Task-macro SR | Micro SR | Atomic | C-Seen | C-Unseen | Static GiB | Compression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("gr00t", "pi05"):
        row = value["arms"][f"actquant_{model}"]
        splits = row["split_task_macro_success_rate"]
        storage = row["storage"]
        gib = storage["static_gib_total_deployed_checkpoint_set"] / storage["checkpoint_count"]
        lines.append(
            f"| actquant / {model} | {pct(row['task_macro_success_rate'])} | "
            f"{pct(row['micro_success_rate'])} | {pct(splits['atomic_seen'])} | "
            f"{pct(splits['composite_seen'])} | {pct(splits['composite_unseen'])} | "
            f"{gib:.3f} | {storage['compression_ratio_total_deployed_checkpoint_set']:.3f}× |"
        )
    lines += [
        "",
        "The registered paired family contains ActQuant versus the protocol-compatible DyPAC row for each model (four flow steps for both GR00T and pi0.5; two exact McNemar tests and Holm correction).",
    ]
    return "\n".join(lines)


def partial_versions_block(value: dict[str, Any]) -> str:
    observed = sum(value["arms"][f"actquant_{model}"]["episodes"] for model in ("gr00t", "pi05"))
    lines = [
        "## ActQuant RoboCasa365 Table-1 extension (interim)",
        "",
        f"- Partial snapshot: `runs/qvla_actquant_table1/formal/partial/table1_partial_snapshot.json` (sha256 `{sha256_file(PARTIAL)}`).",
        f"- Snapshot time: `{value['snapshot_completed_at']}`; ActQuant coverage {observed}/5,000 completed episode rows.",
        "- Status: descriptive interim evidence only; uneven task/seed coverage; not a frozen final result.",
        "- Calibration: `fp16_teacher_proxy`; `source_protocol_equivalent=false`; formal seeds 0--49 were excluded from calibration.",
        "- ActQuant follows the pinned official GitHub implementation for allocation and quantization semantics; local code supplies the cross-architecture evaluation adapters.",
        "- The frozen ActQuant method and packs continue unchanged. This snapshot is not used for tuning or selection.",
        "- QVLA-code is omitted from Table 1; the appendix discloses its failed GitHub-based RoboCasa365 reproduction.",
        "- McNemar and Holm results remain withheld until both ActQuant arms reach 2,500/2,500 episodes.",
        "",
        "| Method/model | Episodes | Tasks seen | Partial task-macro SR | Micro SR | Atomic | C-Seen | C-Unseen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("gr00t", "pi05"):
        row = value["arms"][f"actquant_{model}"]
        splits = row["split_task_macro_success_rate"]
        split_values = [
            "--" if splits[name] is None else pct(splits[name])
            for name in ("atomic_seen", "composite_seen", "composite_unseen")
        ]
        lines.append(
            f"| actquant / {model} | {row['episodes']}/2,500 | "
            f"{row['tasks_observed']}/50 | {pct(row['task_macro_success_rate'])} | "
            f"{pct(row['micro_success_rate'])} | {split_values[0]} | "
            f"{split_values[1]} | {split_values[2]} |"
        )
    lines += [
        "",
        "Partial task-macro SR first averages the observed seeds within each observed task and then weights observed tasks equally. Incomplete tasks therefore use only their currently observed seeds; values must not be compared as if they were complete paired rows.",
    ]
    return "\n".join(lines)


def reference_audit_block(value: dict[str, Any]) -> str:
    comparison_lines = []
    for name in value["holm_family"]:
        row = value["comparisons"][name]
        comparison_lines.append(
            f"- `{name}`: wins/losses {row['paired_wins']}/{row['paired_losses']}; "
            f"exact McNemar p={row['exact_two_sided_mcnemar_p']:.6g}, "
            f"Holm p={row['holm_adjusted_mcnemar_p']:.6g}."
        )
    return "\n".join(
        [
            "## ActQuant local reproduction and QVLA failure audit",
            "",
            "- QVLA semantic source: arXiv:2602.03782v1 and official commit `26cc4821a3be4c003d09d3c7997b38db2a347982`.",
            "- ActQuant semantic source: arXiv:2605.24011v3 and official commit `b64791125070652fe6b554e244fe809c79ef5246`.",
            "- ActQuant follows the pinned official GitHub implementation for allocation and quantization semantics. Local code is limited to model-specific tensor routing, artifact packaging, calibration integration, and RoboCasa365 evaluation.",
            "- The exact frozen `scripts/inference_service.py` used by the rollout is retained in the content-addressed `runs/qvla_actquant_table1/frozen_sources/` archive; finalization verifies it against the pre-rollout implementation SHA rather than the later working-tree revision.",
            "- QVLA-code followed the pinned official GitHub quantizer and channel-allocation implementation, but the public release contains no GR00T, pi0.5, RoboCasa365, or corresponding mixed-row export path. The local adapters produced 0/1,728 GR00T and 0/1,517 pi0.5 successes before the run was stopped. The public artifacts cannot distinguish incomplete open sourcing from an error in the released code path; QVLA-code is omitted from Table 1 and this is not treated as a failure of the native OpenVLA/LIBERO setup.",
            "- Calibration is labeled `fp16_teacher_proxy` and `source_protocol_equivalent=false`; results are reported as measured without post-hoc tuning.",
            "- Every reported candidate has exactly 2,500 unique target-split episodes and a frozen pack/runtime metadata chain. Static pack size and success rate are measured; runtime speed or memory savings are not claimed.",
            "",
            *comparison_lines,
        ]
    )


def partial_reference_audit_block(value: dict[str, Any]) -> str:
    coverage = ", ".join(
        f"`{name}` {row['episodes']}/2500 over {row['tasks_observed']}/50 observed tasks"
        for name, row in value["arms"].items()
        if name.startswith("actquant_")
    )
    return "\n".join(
        [
            "## ActQuant local reproduction and QVLA failure audit (interim snapshot)",
            "",
            "- QVLA semantic source: arXiv:2602.03782v1 and official commit `26cc4821a3be4c003d09d3c7997b38db2a347982`.",
            "- ActQuant semantic source: arXiv:2605.24011v3 and official commit `b64791125070652fe6b554e244fe809c79ef5246`.",
            f"- Content-bound partial snapshot: `runs/qvla_actquant_table1/formal/partial/table1_partial_snapshot.json`, SHA256 `{sha256_file(PARTIAL)}`, completed `{value['snapshot_completed_at']}`.",
            f"- Coverage at this non-atomic cross-file cut: {coverage}.",
            "- Every included JSONL row passes the frozen protocol and server-metadata checks; canonical rows and source-file hashes are retained.",
            "- Partial task-macro values are descriptive under uneven coverage. No paired significance statistic or final comparative claim is emitted.",
            "- ActQuant follows the pinned official GitHub implementation for allocation and quantization semantics. Local code supplies model-specific routing, packaging, calibration integration, and RoboCasa365 evaluation.",
            "- QVLA-code followed the pinned official GitHub quantizer and channel-allocation implementation, but the public release contains no GR00T, pi0.5, RoboCasa365, or corresponding mixed-row export path. The local adapters produced 0/1,728 GR00T and 0/1,517 pi0.5 successes before the run was stopped. The public artifacts cannot distinguish incomplete open sourcing from an error in the released code path; QVLA-code is omitted from Table 1 and this is not treated as evidence against the native OpenVLA/LIBERO setup.",
            "- Calibration is labeled `fp16_teacher_proxy` and `source_protocol_equivalent=false`; interim results are reported as measured without post-hoc tuning.",
            "- Snapshot inspection does not change allocations, packs, services, or rollout scheduling and is forbidden as tuning feedback.",
        ]
    )


def main() -> None:
    final = AGGREGATE.is_file()
    evidence = AGGREGATE if final else PARTIAL
    require(evidence.is_file(), f"missing final aggregate and partial snapshot: {AGGREGATE}, {PARTIAL}")
    value = json.loads(evidence.read_text(encoding="utf-8"))
    if final:
        validate(value)
    else:
        validate_partial(value)
    subprocess.run(
        [
            "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python",
            str(ROOT / "scripts" / "tools" / "render_dypac_paper.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    replace_block(
        ROOT / "FINAL_VERSIONS.md",
        final_versions_block(value) if final else partial_versions_block(value),
    )
    replace_block(
        PAPER / "reference_audit.md",
        reference_audit_block(value) if final else partial_reference_audit_block(value),
    )
    audit = {
        "schema_version": 1,
        "kind": (
            "qvla_actquant_paper_extension_audit"
            if final
            else "qvla_actquant_partial_paper_extension_audit"
        ),
        "valid": True,
        "complete": final,
        "evidence": str(evidence.relative_to(ROOT)),
        "evidence_sha256": sha256_file(evidence),
        "table": str((PAPER / "tables" / "main_results.tex").relative_to(ROOT)),
        "table_sha256": sha256_file(PAPER / "tables" / "main_results.tex"),
        "final_versions_sha256": sha256_file(ROOT / "FINAL_VERSIONS.md"),
        "reference_audit_sha256": sha256_file(PAPER / "reference_audit.md"),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "new_formal_episodes": (
            value["formal_episode_count_new"]
            if final
            else sum(
                value["arms"][f"actquant_{model}"]["episodes"]
                for model in ("gr00t", "pi05")
            )
        ),
        "paired_inference_reported": final,
        "qvla_code_table1_included": False,
        "qvla_code_reproduction_status": "failed_local_cross_architecture_reproduction",
    }
    output = PAPER / "tables" / "qvla_actquant_extension.audit.json"
    atomic_text(output, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": True, "audit": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
