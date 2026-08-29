#!/usr/bin/env python3
"""Render and audit the DyPAC-VLA paper cells from frozen final-v2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/gdsq_vla_iclr2027"  # Legacy directory name; paper identity is DyPAC-VLA.

SOURCES = {
    "gr00t_plan": (
        ROOT / "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json",
        "969c5ae8bc84719229c81a88e8b6ace745cbb87cea551528ec8cf1fd3820e3a2",
    ),
    "gr00t_aggregate": (
        ROOT / "runs/full_context_v2/table1/aggregate.json",
        "cbb59547a6149456f9ae8fc0bac0fedea1267e5ff5f4d144448112f15b0f459e",
    ),
    "pi05_plan": (
        ROOT / "runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json",
        "75d03ac67f17e44570055b54b1a25e44b88ee22bc2282d91bda5acc1f1c61b98",
    ),
    "pi05_main_plan": (
        ROOT / "runs/full_context_v2/pi05_p2/interventions_round_main/context_base.json",
        "fed603b2d91ca8ef419b22e665325e73f97bc400018f6514aa86319dd821e4d4",
    ),
    "pi05_quick": (
        ROOT / "runs/full_context_v2/pi05_quick/combined_aggregate.json",
        "9633554be93cdfd88448b8e2ebec8e0999e26a7351520752c20ee2eb6d3fbade",
    ),
    "pi05_guard": (
        ROOT / "runs/full_context_v2/pi05_quick/non_inferiority_anchor.json",
        "b7922b77071480e5d7d5b37e0492ad22bb42fac23c25f83f22ff872fb190f7b8",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, tol: float = 1e-10) -> bool:
    return abs(actual - expected) <= tol


def load_sources() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, (path, expected_hash) in SOURCES.items():
        require(path.is_file(), f"missing frozen source: {path}")
        actual_hash = sha256(path)
        require(actual_hash == expected_hash, f"{name} SHA drift: {actual_hash}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def audit(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    plan = data["gr00t_plan"]
    agg = data["gr00t_aggregate"]
    candidate = agg["candidate"]
    comparisons = agg["comparisons"]
    layers = list(plan["layers"].values())

    require(sum(x["bits"] == 4 and not x["skip"] for x in layers) == 100, "GR00T W4 count drift")
    require(sum(bool(x["skip"]) for x in layers) == 16, "GR00T FP16 count drift")
    require(plan["meta"]["activation_mode"] == "dynamic_a8", "GR00T activation mode drift")
    require(plan["meta"]["runtime_correction"] is False, "runtime correction must remain disabled")
    require(plan["meta"]["runtime_selector"] is False, "runtime selector must remain disabled")
    require(plan["table1_total_static_bytes"] == 962_068_480, "GR00T byte total drift")
    require(close(plan["table1_total_static_compression"], 2.223893051771117), "GR00T compression drift")

    require(candidate["episodes"] == 2500 and candidate["successes"] == 1350, "GR00T coverage drift")
    require(close(candidate["task_macro_success_rate"], 0.54), "GR00T headline drift")
    require(agg["formal_superiority_over_gdsq_main"] is True, "GR00T superiority guard failed")
    gdsq = comparisons["gdsq_vla_main"]
    fp16 = comparisons["fp16"]
    quantvla = comparisons["quantvla_w4a8"]
    require(close(gdsq["holm_adjusted_mcnemar_p"], 0.003619369860097224), "GDSQ p-value drift")
    require((gdsq["paired_wins"], gdsq["paired_losses"]) == (370, 289), "GDSQ discordance drift")
    require(close(fp16["holm_adjusted_mcnemar_p"], 0.29286269346886834), "FP16 p-value drift")
    require(close(quantvla["baseline"]["task_macro_success_rate"], 0.3044), "QuantVLA result drift")

    pi_plan = data["pi05_plan"]
    pi_layers = list(pi_plan["layers"].values())
    require(sum(x["bits"] == 4 and not x["skip"] for x in pi_layers) == 121, "pi0.5 W4 count drift")
    require(sum(bool(x["skip"]) for x in pi_layers) == 59, "pi0.5 FP16 count drift")
    require(pi_plan["table1_total_static_bytes"] == 1_634_828_288, "pi0.5 byte total drift")
    require(close(pi_plan["table1_total_static_compression"], 2.701569421338518), "pi0.5 compression drift")

    pi_main = data["pi05_main_plan"]
    require(pi_main["meta"]["quantized_layers"] == 80 and pi_main["retained_fp16_layers"] == 100, "pi0.5 main mask drift")
    require(pi_main["table1_total_static_bytes"] == 1_845_100_544, "pi0.5 main byte drift")
    quick = data["pi05_quick"]
    guard = data["pi05_guard"]
    require((quick["candidate_successes"], quick["main_successes"]) == (29, 33), "pi0.5 quick result drift")
    require((quick["paired_wins"], quick["paired_losses"]) == (7, 11), "pi0.5 discordance drift")
    require(guard["status"] == "not_superior_to_gdsq_main", "pi0.5 claim guard drift")

    return {
        "schema_version": 1,
        "kind": "dypac_vla_final_paper_evidence",
        "valid": True,
        "paper_identity": {
            "short_name": "DyPAC-VLA",
            "expanded_name": "Dynamic-Range and Prefix-Accumulated Control-aware Quantization for Vision-Language-Action Models",
            "title": "DyPAC-VLA: Full-Context Mixed-Precision Quantization for Vision-Language-Action Models",
        },
        "gr00t": {
            "ours_task_macro_success_rate": candidate["task_macro_success_rate"],
            "ours_successes": candidate["successes"],
            "episodes": candidate["episodes"],
            "split_task_macro_success_rate": candidate["split_task_macro_success_rate"],
            "w4_layers": 100,
            "fp16_layers": 16,
            "static_component_bytes": plan["table1_total_static_bytes"],
            "compression": plan["table1_total_static_compression"],
            "versus_gdsq": {
                "delta": candidate["task_macro_success_rate"] - gdsq["baseline"]["task_macro_success_rate"],
                "wins": gdsq["paired_wins"],
                "losses": gdsq["paired_losses"],
                "mcnemar_p": gdsq["exact_two_sided_mcnemar_p"],
                "holm_p": gdsq["holm_adjusted_mcnemar_p"],
                "bootstrap": gdsq["task_then_seed_hierarchical_bootstrap"],
                "status": "formal_superiority",
            },
            "versus_fp16": {
                "delta": candidate["task_macro_success_rate"] - fp16["baseline"]["task_macro_success_rate"],
                "wins": fp16["paired_wins"],
                "losses": fp16["paired_losses"],
                "holm_p": fp16["holm_adjusted_mcnemar_p"],
                "bootstrap": fp16["task_then_seed_hierarchical_bootstrap"],
                "status": "statistically_tied",
            },
        },
        "pi05": {
            "role": "compression_anchor_only",
            "w4_layers": 121,
            "fp16_layers": 59,
            "static_component_bytes": pi_plan["table1_total_static_bytes"],
            "compression": pi_plan["table1_total_static_compression"],
            "candidate_successes": quick["candidate_successes"],
            "main_successes": quick["main_successes"],
            "episodes_per_configuration": 100,
            "claim_status": guard["status"],
        },
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": expected_hash}
            for name, (path, expected_hash) in SOURCES.items()
        },
    }


def pct(value: float) -> str:
    # Use conventional half-up display rather than Python's ties-to-even formatting.
    rounded = math.floor(1000 * value + 0.5) / 10
    return f"{rounded:.1f}"


def main_table(data: dict[str, dict[str, Any]]) -> str:
    agg = data["gr00t_aggregate"]
    c = agg["candidate"]
    fp = agg["comparisons"]["fp16"]["baseline"]
    q = agg["comparisons"]["quantvla_w4a8"]["baseline"]
    g = agg["comparisons"]["gdsq_vla_main"]["baseline"]
    cs = c["split_task_macro_success_rate"]
    fs = fp["split_task_macro_success_rate"]
    qs = q["split_task_macro_success_rate"]
    gs = g["split_task_macro_success_rate"]
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{RoboCasa365 task-macro success rate (SR, \\%) over 50 paired scenarios per task.}}
\\label{{tab:main_results}}
\\small
\\setlength{{\\tabcolsep}}{{4.0pt}}
\\renewcommand{{\\arraystretch}}{{1.06}}
\\begin{{tabular}}{{@{{}}lcrrrrrr@{{}}}}
\\toprule
Configuration & \\shortstack{{Low-bit\\\\layers}} & \\shortstack{{Atomic\\\\SR $\\uparrow$}} & \\shortstack{{C-Seen\\\\SR $\\uparrow$}} & \\shortstack{{C-Unseen\\\\SR $\\uparrow$}} & \\shortstack{{All\\\\SR $\\uparrow$}} & \\shortstack{{Size\\\\(GiB) $\\downarrow$}} & \\shortstack{{Comp.\\\\$\\uparrow$}} \\\\
\\midrule
\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{GR00T N1.5}}\\enspace\\textit{{(target split)}}}} \\\\
\\quad FP16 & -- & {pct(fs['atomic_seen'])} & {pct(fs['composite_seen'])} & {pct(fs['composite_unseen'])} & {pct(fp['task_macro_success_rate'])} & 1.993 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 116 W4 & {pct(qs['atomic_seen'])} & {pct(qs['composite_seen'])} & {pct(qs['composite_unseen'])} & {pct(q['task_macro_success_rate'])} & 0.898 & 2.22$\\times$ \\\\
\\quad Uniform W6 & 116 W6 & 68.4 & 42.9 & 41.8 & 51.7 & 1.109 & 1.80$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 180 W4 & 60.1 & 25.9 & 27.5 & 38.7 & 0.599 & 3.33$\\times$ \\\\
\\quad GDSQ-VLA (previous) & 100 W4 & {pct(gs['atomic_seen'])} & {pct(gs['composite_seen'])} & {pct(gs['composite_unseen'])} & {pct(g['task_macro_success_rate'])} & 1.001 & 1.99$\\times$ \\\\
\\quad \\textbf{{\\method (Ours)}} & 100 W4 & {pct(cs['atomic_seen'])} & {pct(cs['composite_seen'])} & {pct(cs['composite_unseen'])} & \\best{{{pct(c['task_macro_success_rate'])}}} & \\best{{0.896}} & \\best{{2.22$\\times$}} \\\\
\\midrule
\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{$\\pi_{{0.5}}$}}~\\cite{{intelligence2025pi05}}\\enspace\\textit{{(legacy target-split comparison)}}}} \\\\
\\quad FP16 & -- & 58.3 & 14.0 & 2.1 & 26.2 & 4.113 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 180 W4 & 56.1 & 12.8 & 1.8 & 24.8 & 1.388 & 2.96$\\times$ \\\\
\\quad Uniform W6 & 180 W6 & 56.6 & 10.9 & 2.1 & 24.5 & 1.902 & 2.16$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 252 W4 & 49.6 & 10.4 & 1.0 & 21.5 & 1.307 & 3.27$\\times$ \\\\
\\quad GDSQ-VLA (legacy) & 80 W4 & 57.8 & 17.0 & 3.9 & 27.5 & 1.868 & 2.20$\\times$ \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\textwidth}}{{\\footnotesize C-Seen/C-Unseen denote Composite-Seen/Composite-Unseen. DyPAC-VLA is ours only in the GR00T block; the $\\pi_{{0.5}}$ rows restore the previous audited comparison and are not DyPAC-VLA results. Ours uses exact total static-component accounting (962,068,480 bytes); legacy rows retain their archived packed-component scopes. $^{{\\ddagger}}\\Omega$-QVLA uses model- and task-set-specific RoboCasa365 calibration.}}
\\end{{table}}
"""


def claim_status() -> str:
    return r"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\providecommand{\GRHeadlineStatus}{DyPAC-VLA obtains 54.0\% over 2,500 GR00T episodes at 2.22$\times$ static component compression.}
\providecommand{\GRComparisonStatus}{The gain over GDSQ-VLA is 3.2 points with Holm-adjusted $p=0.0036$; the comparison with FP16 is statistically tied ($p=0.2929$).}
\providecommand{\PiAnchorStatus}{The $\pi_{0.5}$ result is a 2.702$\times$ compression anchor and does not support a success-rate improvement claim.}
"""


def generated_files(data: dict[str, dict[str, Any]], registry: dict[str, Any]) -> dict[Path, str]:
    audit_payload = {
        "schema_version": 1,
        "kind": "dypac_vla_paper_render_audit",
        "valid": True,
        "source_hashes": {name: digest for name, (_, digest) in SOURCES.items()},
        "headline": registry["gr00t"],
        "pi05_claim_guard": registry["pi05"],
    }
    return {
        PAPER / "tables/main_results.tex": main_table(data),
        PAPER / "tables/claim_status.tex": claim_status(),
        PAPER / "dypac_evidence_registry.json": json.dumps(registry, indent=2, sort_keys=True) + "\n",
        PAPER / "tables/dypac_paper.audit.json": json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    data = load_sources()
    registry = audit(data)
    outputs = generated_files(data, registry)
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    require(not stale, f"stale generated DyPAC paper files: {stale}")
    print(json.dumps({"valid": True, "checked": sorted(str(p.relative_to(ROOT)) for p in outputs)}, indent=2))


if __name__ == "__main__":
    main()
