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
    "activation_attribution": (
        ROOT / "runs/full_context_v2/p2/activation_attribution.json",
        "3c59f27bf648b575bf87864ba9ed4a44590d077d8a57c298d841e6ff4bd879b9",
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
    fp16 = comparisons["fp16"]
    quantvla = comparisons["quantvla_w4a8"]
    require(close(fp16["holm_adjusted_mcnemar_p"], 0.29286269346886834), "FP16 p-value drift")
    require(close(quantvla["baseline"]["task_macro_success_rate"], 0.3044), "QuantVLA result drift")
    require((quantvla["paired_wins"], quantvla["paired_losses"]) == (721, 132), "QuantVLA discordance drift")
    require(
        close(quantvla["holm_adjusted_mcnemar_p"], 1.8894854853781264e-98),
        "QuantVLA p-value drift",
    )

    activation = data["activation_attribution"]
    require(activation["selected_activation_mode"] == "dynamic_a8", "activation decision drift")
    require(close(activation["static_vs_a16"]["objective"], 467.5204744598872), "static-A8 objective drift")
    require(close(activation["dynamic_vs_a16"]["objective"], 0.5178887111755559), "DyRange objective drift")
    summaries = plan["meta"]["selection_result"]["summaries"]
    require(len(summaries) == 5, "full-network proposal count drift")
    require(not any(item["eligible"] for item in summaries.values()), "full-network eligibility drift")

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
    require(guard["candidate_successes"] == 29 and guard["main_successes"] == 33, "pi0.5 guard drift")

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
            "versus_quantvla": {
                "delta": candidate["task_macro_success_rate"] - quantvla["baseline"]["task_macro_success_rate"],
                "wins": quantvla["paired_wins"],
                "losses": quantvla["paired_losses"],
                "mcnemar_p": quantvla["exact_two_sided_mcnemar_p"],
                "holm_p": quantvla["holm_adjusted_mcnemar_p"],
                "bootstrap": quantvla["task_then_seed_hierarchical_bootstrap"],
                "status": "formal_superiority",
            },
            "versus_fp16": {
                "delta": candidate["task_macro_success_rate"] - fp16["baseline"]["task_macro_success_rate"],
                "wins": fp16["paired_wins"],
                "losses": fp16["paired_losses"],
                "holm_p": fp16["holm_adjusted_mcnemar_p"],
                "bootstrap": fp16["task_then_seed_hierarchical_bootstrap"],
                "status": "difference_not_significant_no_equivalence_claim",
            },
            "core_mechanism_evidence": {
                "static_a8_objective": activation["static_vs_a16"]["objective"],
                "dynamic_a8_objective": activation["dynamic_vs_a16"]["objective"],
                "one_layer_flips_tested": len(layers),
                "beneficial_one_layer_flips": 0,
                "structured_alternatives_tested": len(summaries),
                "eligible_structured_alternatives": sum(item["eligible"] for item in summaries.values()),
                "structured_objective_min": min(item["objective"] for item in summaries.values()),
                "structured_objective_max": max(item["objective"] for item in summaries.values()),
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
            "claim_status": "quick_screen_does_not_support_success_superiority",
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
    cs = c["split_task_macro_success_rate"]
    fs = fp["split_task_macro_success_rate"]
    qs = q["split_task_macro_success_rate"]
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{GR00T N1.5 task-macro success rate (SR, \\%) on the RoboCasa365 target split, with 50 paired scenarios per task.}}
\\label{{tab:main_results}}
\\small
\\setlength{{\\tabcolsep}}{{4.0pt}}
\\renewcommand{{\\arraystretch}}{{1.06}}
\\begin{{tabular}}{{@{{}}lcrrrrrr@{{}}}}
\\toprule
Configuration & \\shortstack{{Low-bit\\\\layers}} & \\shortstack{{Atomic\\\\SR $\\uparrow$}} & \\shortstack{{C-Seen\\\\SR $\\uparrow$}} & \\shortstack{{C-Unseen\\\\SR $\\uparrow$}} & \\shortstack{{All\\\\SR $\\uparrow$}} & \\shortstack{{Size\\\\(GiB) $\\downarrow$}} & \\shortstack{{Comp.\\\\$\\uparrow$}} \\\\
\\midrule
\\quad FP16 & -- & {pct(fs['atomic_seen'])} & {pct(fs['composite_seen'])} & {pct(fs['composite_unseen'])} & {pct(fp['task_macro_success_rate'])} & 1.993 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 116 W4 & {pct(qs['atomic_seen'])} & {pct(qs['composite_seen'])} & {pct(qs['composite_unseen'])} & {pct(q['task_macro_success_rate'])} & 0.898 & 2.22$\\times$ \\\\
\\quad Uniform W6 & 116 W6 & 68.4 & 42.9 & 41.8 & 51.7 & 1.109 & 1.80$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 180 W4 & 60.1 & 25.9 & 27.5 & 38.7 & 0.599 & 3.33$\\times$ \\\\
\\quad \\textbf{{\\method (Ours)}} & \\textbf{{100 W4}} & \\textbf{{{pct(cs['atomic_seen'])}}} & \\textbf{{{pct(cs['composite_seen'])}}} & \\textbf{{{pct(cs['composite_unseen'])}}} & \\textbf{{{pct(c['task_macro_success_rate'])}}} & \\textbf{{0.896}} & \\textbf{{2.22$\\times$}} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\textwidth}}{{\\footnotesize C-Seen/C-Unseen denote Composite-Seen/Composite-Unseen. Ours uses exact total static-component accounting (962,068,480 bytes). $^{{\\ddagger}}\\Omega$-QVLA uses model- and task-set-specific RoboCasa365 calibration. Compression ratios denote static component storage, not runtime memory or latency.}}
\\end{{table}}
"""


def core_ablation_table(data: dict[str, dict[str, Any]]) -> str:
    activation = data["activation_attribution"]
    summaries = data["gr00t_plan"]["meta"]["selection_result"]["summaries"]
    objectives = [item["objective"] for item in summaries.values()]
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{Core mechanism ablations on the frozen 12-task, 36-sequence context buffer. These are paired offline diagnostics, not closed-loop success rates; lower objective is better.}}
\\label{{tab:core_ablation}}
\\small
\\setlength{{\\tabcolsep}}{{4.0pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}
\\begin{{tabular}}{{@{{}}p{{0.16\\linewidth}}p{{0.33\\linewidth}}p{{0.24\\linewidth}}p{{0.18\\linewidth}}@{{}}}}
\\toprule
Factor & Controlled counterfactual & Audited evidence & Decision \\\\
\\midrule
Range & Static A8 vs. FP16-activation control & $J={activation['static_vs_a16']['objective']:.2f}$ & Reject static A8 \\\\
Range & Dynamic A8 vs. FP16-activation control & $J={activation['dynamic_vs_a16']['objective']:.3f}$ & Use \\dyrange \\\\
Protection & Every valid one-layer state flip & 0/{len(data['gr00t_plan']['layers'])} positive benefits & Keep base mask \\\\
Adjudication & {len(summaries)} structured mask alternatives & 0/{len(summaries)} eligible; $J\\in[{min(objectives):.2f},{max(objectives):.2f}]$ & Reject alternatives \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\textwidth}}{{\\footnotesize $J$ is the worst normalized one-standard-error upper bound over $\\dpac$, $\\dfunc$, and task clusters. The FP16-activation control is diagnostic only. A mask is eligible only when $J<0$ and pose, stitch, and gripper upper bounds are non-positive.}}
\\end{{table}}
"""


def claim_status() -> str:
    return r"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\providecommand{\GRHeadlineStatus}{DyPAC-VLA obtains 54.0\% over 2,500 GR00T episodes at 2.22$\times$ static component compression.}
\providecommand{\GRComparisonStatus}{The gain over QuantVLA is 23.6 points with Holm-adjusted $p<10^{-4}$; the difference from FP16 is not significant ($p=0.2929$).}
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
        PAPER / "tables/core_ablation.tex": core_ablation_table(data),
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
