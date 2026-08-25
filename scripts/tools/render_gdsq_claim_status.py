#!/usr/bin/env python3
"""Render dynamic claim-safe paper prose from audited week-1 evidence."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "docs/gdsq_vla_cvpr2026/experiment_registry.json"
TEX = REPO_ROOT / "docs/gdsq_vla_cvpr2026/tables/claim_status.tex"
AUDIT = REPO_ROOT / "docs/gdsq_vla_cvpr2026/tables/claim_status.audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def verified(record: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = resolve(record["path"])
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256_file(path) == record["sha256"], f"{label} SHA drift")
    return path, json.loads(path.read_text(encoding="utf-8"))


def number(value: float, digits: str = "0.1") -> str:
    return str(Decimal(str(value)).quantize(Decimal(digits), rounding=ROUND_HALF_UP))


def macro(name: str, value: str) -> str:
    return f"\\providecommand{{\\{name}}}{{{value}}}"


def selector_state(registry: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    record = registry["experiments"]["pi05_runtime_selector_official50"]
    sources: dict[str, Any] = {"selector_summary": None}
    if record.get("status") != "complete" or not record.get("summary"):
        sentence = (
            "The formal $\\pi_{0.5}$ selector evaluation remains claim-gated until "
            "all 2,500 manifested episodes pass the audit; no partial success rate is reported."
        )
        return {
            "PiSelectorRunStatus": (
                "At the time of this draft the run is active; no partial success rate is "
                "reported or used to tune the three v8 thresholds."
            ),
            "PiSelectorResultStatus": sentence,
            "PiSelectorAbstractStatus": (
                "The completed $\\pi_{0.5}$ static-mask transfer is statistically "
                "indistinguishable from its fixed low-bit baseline, while its formal "
                "selector evaluation remains claim-gated until full coverage."
            ),
        }, sources
    path, summary = verified(record["summary"], "pi0.5 selector summary")
    require(summary.get("complete") is True, "complete selector record has incomplete summary")
    row = summary["configs"]["gdsq_vla_runtime_selector"]
    static_row = summary["configs"]["gdsq_vla"]
    groups = row["task_set_macro_sr"]
    mean = 100 * float(row["task_macro_sr"])
    static_mean = 100 * float(static_row["task_macro_sr"])
    static_contrast = summary["comparisons"]["gdsq_vla_runtime_selector_vs_gdsq_vla"]
    fp16_contrast = summary["comparisons"]["gdsq_vla_runtime_selector_vs_fp16"]
    static_delta = 100 * float(static_contrast["all50_task_macro_delta"])
    static_ci = [100 * float(value) for value in static_contrast["all50_task_cluster_ci95"]]
    static_p = float(static_contrast["holm_adjusted_p"])
    fp16_delta = 100 * float(fp16_contrast["all50_task_macro_delta"])
    fp16_ci = [100 * float(value) for value in fp16_contrast["all50_task_cluster_ci95"]]
    fp16_p = float(fp16_contrast["holm_adjusted_p"])
    values = ", ".join(
        number(100 * float(groups[name])) + "\\%"
        for name in ("atomic_seen", "composite_seen", "composite_unseen")
    )
    sources["selector_summary"] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
    return {
        "PiSelectorRunStatus": (
            f"The audited run completes all 2,500 manifested episodes; its three task-group "
            f"macro SRs are {values}, for {number(mean)}\\% over all 50 tasks."
        ),
        "PiSelectorResultStatus": (
            f"The final $\\pi_{{0.5}}$ selector obtains {number(mean)}\\% all-task macro SR. "
            f"It improves over the uncorrected static mask ({number(static_mean)}\\%) by "
            f"{number(static_delta)} points (95\\% task CI "
            f"[{number(static_ci[0])}, {number(static_ci[1])}], Holm-adjusted "
            f"$p={number(static_p, '0.001')}$). Its {number(fp16_delta)}-point difference "
            f"from FP16 has CI [{number(fp16_ci[0])}, {number(fp16_ci[1])}] and Holm-adjusted "
            f"$p={number(fp16_p, '0.001')}$, so this comparison is competitive rather than superior."
        ),
        "PiSelectorAbstractStatus": (
            f"The audited $\\pi_{{0.5}}$ selector obtains {number(mean)}\\% task-macro SR "
            f"over all 50 tasks and improves over its uncorrected static mask by "
            f"{number(static_delta)} points (95\\% task CI "
            f"[{number(static_ci[0])}, {number(static_ci[1])}])."
        ),
    }, sources


def same_budget_state(registry: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    claim = registry["paper_claims"]["same_budget_superiority"]
    sources: dict[str, Any] = {"joint_statistics": None}
    if not claim.get("joint_statistics"):
        pending = (
            "Both uniform-W6 controls have complete audited coverage and are reported "
            "descriptively. The search-matched random and action-only controls remain excluded "
            "from superiority claims until their exact manifests reach full coverage."
        )
        return {
            "SameBudgetControlStatus": pending,
            "SameBudgetAbstractStatus": (
                "The current evidence supports the mechanism and a favorable GR00T "
                "storage--accuracy point, but not yet same-budget superiority across architectures."
            ),
            "SameBudgetConclusionStatus": (
                "Same-budget controls must complete before supporting a causal superiority claim."
            ),
        }, sources
    path, summary = verified(claim["joint_statistics"], "joint week-1 statistics")
    require(summary.get("complete") is True, "joint statistics incomplete")
    sources["joint_statistics"] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
    fragments = []
    for model, display in (("gr00t", "GR00T"), ("pi05", "$\\pi_{0.5}$")):
        gate = summary["strongest_baseline_gates"][model]
        contrast = summary["contrasts"][gate["heldout_strongest_baseline"]]
        delta = 100 * float(contrast["task_macro_delta_ours_minus_baseline"])
        ci = [100 * float(value) for value in contrast["task_cluster_ci95"]]
        p_value = float(contrast["holm_adjusted_p"])
        if gate["superiority_claim_enabled"]:
            conclusion = "supports superiority on the preregistered held-out scope"
        else:
            conclusion = "does not support superiority and is reported as competitive/mechanistic evidence"
        fragments.append(
            f"For {display}, ours minus the strongest held-out same-budget baseline is "
            f"{number(delta)} points (95\\% task CI [{number(ci[0])}, {number(ci[1])}], "
            f"Holm $p={number(p_value, '0.001')}$), which {conclusion}."
        )
    enabled = bool(summary["same_budget_superiority_claim_enabled_across_architectures"])
    overall = (
        "The cross-architecture same-budget superiority claim passes its preregistered gate."
        if enabled
        else "The cross-architecture same-budget superiority claim remains disabled."
    )
    return {
        "SameBudgetControlStatus": "All preregistered same-budget controls have complete audited coverage. " + " ".join(fragments) + " " + overall,
        "SameBudgetAbstractStatus": overall,
        "SameBudgetConclusionStatus": " ".join(fragments) + " " + overall,
    }, sources


def render() -> tuple[str, dict[str, Any]]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    selector, selector_sources = selector_state(registry)
    controls, control_sources = same_budget_state(registry)
    values = {**selector, **controls}
    values["AbstractEvidenceStatus"] = (
        values["PiSelectorAbstractStatus"] + " " + values["SameBudgetAbstractStatus"]
    )
    values["IntroEvidenceStatus"] = (
        values["PiSelectorResultStatus"] + " " + values["SameBudgetControlStatus"]
    )
    values["ConclusionEvidenceStatus"] = (
        values["PiSelectorResultStatus"] + " " + values["SameBudgetConclusionStatus"]
    )
    lines = [
        "% AUTO-GENERATED by scripts/tools/render_gdsq_claim_status.py; DO NOT EDIT.",
        *(macro(name, value) for name, value in values.items()),
        "",
    ]
    audit = {
        "schema_version": 1,
        "kind": "gdsq_vla_dynamic_claim_status",
        "registry": {"path": str(REGISTRY.relative_to(REPO_ROOT)), "sha256": sha256_file(REGISTRY)},
        "sources": {**selector_sources, **control_sources},
        "macros": values,
        "superiority_wording_generated_only_from_joint_gate": True,
    }
    return "\n".join(lines), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tex, audit = render()
    audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.check:
        require(TEX.read_text(encoding="utf-8") == tex, "claim-status TeX is stale")
        require(AUDIT.read_text(encoding="utf-8") == audit_text, "claim-status audit is stale")
        print(f"claim-status sources verified: {TEX}")
        return
    TEX.write_text(tex, encoding="utf-8")
    AUDIT.write_text(audit_text, encoding="utf-8")
    print(f"claim status rendered: {TEX}")


if __name__ == "__main__":
    main()
