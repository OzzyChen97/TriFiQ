#!/usr/bin/env python3
"""Gate and render the completed QVLA/ActQuant Table-1 paper extension."""

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
    require(value.get("formal_episode_count_new") == 10_000, "formal episode count drift")
    require(value.get("bootstrap_draws") == 10_000, "bootstrap draw drift")
    require(len(value.get("holm_family") or []) == 4, "Holm family drift")
    for method in ("qvla", "actquant"):
        for model in ("gr00t", "pi05"):
            key = f"{method}_{model}"
            row = (value.get("arms") or {}).get(key) or {}
            require(row.get("episodes") == 2500, f"{key} coverage drift")
            require(len(row.get("per_task_success_rate") or {}) == 50, f"{key} task drift")
            require(row.get("storage") is not None, f"{key} storage missing")


def final_versions_block(value: dict[str, Any]) -> str:
    lines = [
        "## QVLA / ActQuant RoboCasa365 Table-1 extension",
        "",
        f"- Aggregate: `runs/qvla_actquant_table1/formal/aggregate.json` (sha256 `{sha256_file(AGGREGATE)}`).",
        "- Calibration: `fp16_teacher_proxy`; `source_protocol_equivalent=false`; formal seeds 0--49 were excluded from calibration.",
        "- Coverage: four new rows, each 50 tasks × 50 paired seeds = 2,500 episodes; 10,000/10,000 new formal episodes total.",
        "- Reported size is measured static pack/GGUF storage only. No PyTorch runtime-memory, C++ runtime, or latency claim is made.",
        "",
        "| Method/model | Task-macro SR | Micro SR | Atomic | C-Seen | C-Unseen | Static GiB | Compression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("qvla", "actquant"):
        for model in ("gr00t", "pi05"):
            row = value["arms"][f"{method}_{model}"]
            splits = row["split_task_macro_success_rate"]
            storage = row["storage"]
            gib = storage["static_gib_total_deployed_checkpoint_set"] / storage["checkpoint_count"]
            lines.append(
                f"| {method} / {model} | {pct(row['task_macro_success_rate'])} | "
                f"{pct(row['micro_success_rate'])} | {pct(splits['atomic_seen'])} | "
                f"{pct(splits['composite_seen'])} | {pct(splits['composite_unseen'])} | "
                f"{gib:.3f} | {storage['compression_ratio_total_deployed_checkpoint_set']:.3f}× |"
            )
    lines += [
        "",
        "The registered paired family contains QVLA and ActQuant versus the protocol-compatible DyPAC row for each model (four flow steps for both GR00T and pi0.5; four exact McNemar tests, task-then-seed bootstrap, and Holm correction).",
    ]
    return "\n".join(lines)


def reference_audit_block(value: dict[str, Any]) -> str:
    comparison_lines = []
    for name in value["holm_family"]:
        row = value["comparisons"][name]
        boot = row["task_then_seed_hierarchical_bootstrap"]
        comparison_lines.append(
            f"- `{name}`: wins/losses {row['paired_wins']}/{row['paired_losses']}; "
            f"exact McNemar p={row['exact_two_sided_mcnemar_p']:.6g}, "
            f"Holm p={row['holm_adjusted_mcnemar_p']:.6g}; task-then-seed 95% CI "
            f"[{boot['ci95_low']:.4f}, {boot['ci95_high']:.4f}]."
        )
    return "\n".join(
        [
            "## QVLA / ActQuant local reproduction audit",
            "",
            "- QVLA semantic source: arXiv:2602.03782v1 and official commit `26cc4821a3be4c003d09d3c7997b38db2a347982`.",
            "- ActQuant semantic source: arXiv:2605.24011v3 and official commit `b64791125070652fe6b554e244fe809c79ef5246`.",
            "- Source/license/patch/code-versus-paper details are frozen in `docs/gdsq_vla_iclr2027/third_party/qvla_actquant/AUDIT.md` and `runs/qvla_actquant_table1/provenance.json`.",
            "- The official target-human demonstration URL returned HTTP 404, so all calibration is explicitly labeled `fp16_teacher_proxy` and `source_protocol_equivalent=false`; unsuccessful teacher trajectories were retained.",
            "- QVLA follows released-code channel gates and fake-quant semantics. ActQuant uses released IQ/QK formats and allocation, continuous-action HSIC/Fisher adaptation, the official π0.5 LLM GGUF path, and a graph-free GR00T component exporter patch.",
            "- Every reported candidate has exactly 2,500 unique target-split episodes and a frozen pack/runtime metadata chain. Static pack size and success rate are measured; runtime speed or memory savings are not claimed.",
            "",
            *comparison_lines,
        ]
    )


def main() -> None:
    require(AGGREGATE.is_file(), f"missing aggregate: {AGGREGATE}")
    value = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    validate(value)
    subprocess.run(
        [
            "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python",
            str(ROOT / "scripts" / "tools" / "render_dypac_paper.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    replace_block(ROOT / "FINAL_VERSIONS.md", final_versions_block(value))
    replace_block(PAPER / "reference_audit.md", reference_audit_block(value))
    audit = {
        "schema_version": 1,
        "kind": "qvla_actquant_paper_extension_audit",
        "valid": True,
        "aggregate": str(AGGREGATE.relative_to(ROOT)),
        "aggregate_sha256": sha256_file(AGGREGATE),
        "table": str((PAPER / "tables" / "main_results.tex").relative_to(ROOT)),
        "table_sha256": sha256_file(PAPER / "tables" / "main_results.tex"),
        "final_versions_sha256": sha256_file(ROOT / "FINAL_VERSIONS.md"),
        "reference_audit_sha256": sha256_file(PAPER / "reference_audit.md"),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "new_formal_episodes": 10_000,
    }
    output = PAPER / "tables" / "qvla_actquant_extension.audit.json"
    atomic_text(output, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": True, "audit": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
