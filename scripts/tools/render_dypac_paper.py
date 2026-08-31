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
QVLA_ACTQUANT_AGGREGATE = ROOT / "runs/qvla_actquant_table1/formal/aggregate.json"

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
        "e502f7cd7d126517c000f8b5b8e7c6e5537b31226910a2dd6834caaebf736c83",
    ),
    "pi05_main_plan": (
        ROOT / "runs/full_context_v2/pi05_p2/interventions_round_main/context_base.json",
        "461531ea2c48cdd5e0f063bc944c882ede6a36ed73482beeb19134ec7e55e42b",
    ),
    "pi05_dypac_formal": (
        ROOT / "runs/full_context_v2/pi05_table1/aggregate.json",
        "512480a2e0836423254217b5215489bcb218e17a4c7d0fff1cdf5aba9acf4f73",
    ),
    "pi05_protocol_correction": (
        ROOT / "runs/full_context_v2/pi05_table1/protocol_correction.json",
        "cd5e07baaaf7b40e53c1ace9a881eb49aabb9407d8d20189d46e3a11b2186770",
    ),
    "pi05_static_official": (
        ROOT / "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/aggregate/summary.json",
        "c7b6441c6ca0a49aa5cfe8e673d1ee465a5d101076b6ad866b149fb545d10cbb",
    ),
    "pi05_uniform_w6": (
        ROOT / "runs/gdsq_week1_preregistered_v1/execution/runs/pi05_uniform_w6_official50/aggregate/summary.json",
        "849e1e159daf6b515bcb1724a8862fea17e9c0c95053ed5fefc99b9c145e052a",
    ),
    "pi05_omega": (
        ROOT / "runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1/aggregate/summary.json",
        "d898552b26fb68db253decfe9375977f1dbf38daf89bb94d73c6d62564114ae9",
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


def task_direction_counts(
    candidate: dict[str, Any], baseline: dict[str, Any], tol: float = 1e-12
) -> dict[str, int]:
    """Count task-level directions from frozen per-task success rates."""
    candidate_rates = candidate["per_task_success_rate"]
    baseline_rates = baseline["per_task_success_rate"]
    require(candidate_rates.keys() == baseline_rates.keys(), "per-task key drift")
    counts = {"better": 0, "equal": 0, "worse": 0}
    for task in candidate_rates:
        delta = candidate_rates[task] - baseline_rates[task]
        direction = "equal" if abs(delta) <= tol else ("better" if delta > 0 else "worse")
        counts[direction] += 1
    return counts


def load_sources() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, (path, expected_hash) in SOURCES.items():
        require(path.is_file(), f"missing frozen source: {path}")
        actual_hash = sha256(path)
        require(actual_hash == expected_hash, f"{name} SHA drift: {actual_hash}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if QVLA_ACTQUANT_AGGREGATE.is_file():
        loaded["qvla_actquant"] = json.loads(
            QVLA_ACTQUANT_AGGREGATE.read_text(encoding="utf-8")
        )
    return loaded


def audit_qvla_actquant(value: dict[str, Any]) -> dict[str, Any]:
    require(value.get("complete") is True, "QVLA/ActQuant aggregate incomplete")
    require(value.get("ready_for_table_update") is True, "QVLA/ActQuant table gate disabled")
    require(value.get("formal_episode_count_new") == 10_000, "QVLA/ActQuant coverage drift")
    require(value.get("bootstrap_draws") == 10_000, "QVLA/ActQuant bootstrap drift")
    require(value.get("bootstrap_seed") == 0, "QVLA/ActQuant bootstrap seed drift")
    require(len(value.get("holm_family") or []) == 4, "QVLA/ActQuant Holm family drift")
    expected = {
        "qvla_gr00t", "actquant_gr00t", "qvla_pi05", "actquant_pi05",
        "dypac_gr00t", "dypac_pi05",
    }
    arms = value.get("arms") or {}
    require(set(arms) == expected, "QVLA/ActQuant arm inventory drift")
    candidate_records = {}
    for arm in sorted(expected - {"dypac_gr00t", "dypac_pi05"}):
        row = arms[arm]
        require(row.get("episodes") == 2500, f"{arm} coverage drift")
        require(len(row.get("per_task_success_rate") or {}) == 50, f"{arm} task drift")
        require(set((row.get("split_task_macro_success_rate") or {})) == {
            "atomic_seen", "composite_seen", "composite_unseen"
        }, f"{arm} split drift")
        artifacts = row.get("artifacts") or []
        expected_artifacts = 3 if arm.endswith("gr00t") else 1
        require(len(artifacts) == expected_artifacts, f"{arm} artifact coverage drift")
        for artifact in artifacts:
            manifest_path = Path(artifact["pack_manifest"])
            require(manifest_path.is_file(), f"{arm} pack manifest missing")
            require(sha256(manifest_path) == artifact["pack_manifest_sha256"], f"{arm} pack manifest SHA drift")
            pack = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(pack.get("test_results_used") is False, f"{arm} used test feedback")
            require(pack.get("source_protocol_equivalent") is False, f"{arm} source label drift")
            precision = pack.get("model_precision") or {}
            if arm.startswith("qvla"):
                require(float(pack["actual_average_channel_bits"]) <= 4.0, f"{arm} Wavg drift")
                require(
                    precision.get("resolved") == "bfloat16"
                    and precision.get("strict_all_linear_conv_bf16") is True
                    and pack.get("activation_compute_dtype") == "bfloat16",
                    f"{arm} QVLA precision drift",
                )
            else:
                require(float(pack["achieved_bpw"]) <= 4.0 + 1e-6, f"{arm} BPW drift")
                require(
                    precision.get("resolved") == "float16"
                    and precision.get("strict_all_linear_conv_fp16") is True
                    and pack.get("activation_compute_dtype") == "float16",
                    f"{arm} ActQuant precision drift",
                )
        candidate_records[arm] = {
            "episodes": row["episodes"],
            "successes": row["successes"],
            "task_macro_success_rate": row["task_macro_success_rate"],
            "micro_success_rate": row["micro_success_rate"],
            "split_task_macro_success_rate": row["split_task_macro_success_rate"],
            "storage": row["storage"],
        }
    for comparison in value.get("holm_family") or []:
        record = (value.get("comparisons") or {}).get(comparison) or {}
        require("exact_two_sided_mcnemar_p" in record, f"{comparison} McNemar missing")
        require("holm_adjusted_mcnemar_p" in record, f"{comparison} Holm p missing")
        bootstrap = record.get("task_then_seed_hierarchical_bootstrap") or {}
        require(bootstrap.get("draws") == 10_000 and bootstrap.get("seed") == 0,
                f"{comparison} bootstrap drift")
    return {
        "aggregate_path": str(QVLA_ACTQUANT_AGGREGATE.relative_to(ROOT)),
        "aggregate_sha256": sha256(QVLA_ACTQUANT_AGGREGATE),
        "calibration_source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "formal_episode_count_new": 10_000,
        "arms": candidate_records,
        "comparison_scope": value["comparison_scope"],
    }


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
    quant_task_direction = task_direction_counts(candidate, quantvla["baseline"])
    fp16_task_direction = task_direction_counts(candidate, fp16["baseline"])
    require(close(fp16["holm_adjusted_mcnemar_p"], 0.29286269346886834), "FP16 p-value drift")
    require(close(quantvla["baseline"]["task_macro_success_rate"], 0.3044), "QuantVLA result drift")
    require((quantvla["paired_wins"], quantvla["paired_losses"]) == (721, 132), "QuantVLA discordance drift")
    require(
        close(quantvla["holm_adjusted_mcnemar_p"], 1.8894854853781264e-98),
        "QuantVLA p-value drift",
    )
    require(quant_task_direction == {"better": 48, "equal": 1, "worse": 1}, "QuantVLA task-direction drift")
    require(fp16_task_direction == {"better": 18, "equal": 6, "worse": 26}, "FP16 task-direction drift")

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
    pi_dypac = data["pi05_dypac_formal"]
    pi_dypac_result = pi_dypac["result"]
    require(pi_dypac["complete"] is True, "pi0.5 DyPAC aggregate incomplete")
    require(
        pi_dypac["model"] == "pi0.5" and pi_dypac["method"] == "DyPAC-VLA",
        "pi0.5 DyPAC identity drift",
    )
    require(pi_dypac_result["episodes"] == 2500, "pi0.5 DyPAC coverage drift")
    require(pi_dypac_result["successes"] == 693, "pi0.5 DyPAC success-count drift")
    require(close(pi_dypac_result["task_macro_success_rate"], 0.2772), "pi0.5 DyPAC headline drift")
    require(
        pi_dypac_result["split_task_macro_success_rate"]
        == {
            "atomic_seen": 0.5955555555555555,
            "composite_seen": 0.15375,
            "composite_unseen": 0.042499999999999996,
        },
        "pi0.5 DyPAC split drift",
    )

    pi_static = data["pi05_static_official"]
    pi_fp16 = pi_static["configs"]["fp16"]
    pi_quant = pi_static["configs"]["quantvla_w4a8_atmohb"]
    pi_w6 = data["pi05_uniform_w6"]["configs"]["uniform_w6"]
    pi_omega = data["pi05_omega"]
    for name, row in (("FP16", pi_fp16), ("QuantVLA", pi_quant), ("Uniform W6", pi_w6)):
        require(row["completed_episodes"] == 2500, f"pi0.5 {name} coverage drift")
    require(pi_omega["complete"] is True and pi_omega["episodes"] == 2500, "pi0.5 Omega coverage drift")
    require(close(pi_fp16["task_macro_sr"], 0.2616), "pi0.5 FP16 result drift")
    require(close(pi_quant["task_macro_sr"], 0.2484), "pi0.5 QuantVLA result drift")
    require(close(pi_w6["task_macro_sr"], 0.2452), "pi0.5 W6 result drift")
    require(close(pi_omega["task_macro_sr"], 0.2148), "pi0.5 Omega result drift")

    result = {
        "schema_version": 1,
        "kind": "dypac_vla_final_paper_evidence",
        "valid": True,
        "paper_identity": {
            "short_name": "DyPAC-VLA",
            "expanded_name": "Full-context PTQ under policy-induced deployment distributions",
            "title": "Quantization Changes the Data: Full-Context Post-Training Quantization for Vision-Language-Action Policies",
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
                "status": "formal_superiority",
            },
            "versus_fp16": {
                "delta": candidate["task_macro_success_rate"] - fp16["baseline"]["task_macro_success_rate"],
                "wins": fp16["paired_wins"],
                "losses": fp16["paired_losses"],
                "holm_p": fp16["holm_adjusted_mcnemar_p"],
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
            "task_level_direction": {
                "versus_quantvla": quant_task_direction,
                "versus_fp16": fp16_task_direction,
            },
        },
        "pi05": {
            "role": "formal_target_split_result",
            "flow_steps": 4,
            "w4_layers": 121,
            "fp16_layers": 59,
            "static_component_bytes": pi_plan["table1_total_static_bytes"],
            "compression": pi_plan["table1_total_static_compression"],
            "successes": pi_dypac_result["successes"],
            "episodes_per_configuration": pi_dypac_result["episodes"],
            "task_macro_success_rate": pi_dypac_result["task_macro_success_rate"],
            "split_task_macro_success_rate": pi_dypac_result["split_task_macro_success_rate"],
            "claim_status": "formal_target_split_result_descriptive_cross_row_comparison",
            "official_target_split_baselines": {
                "fp16": pi_fp16["task_macro_sr"],
                "quantvla_w4a8": pi_quant["task_macro_sr"],
                "uniform_w6": pi_w6["task_macro_sr"],
                "omega_qvla_w4a4": pi_omega["task_macro_sr"],
                "episodes_per_configuration": 2500,
            },
        },
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": expected_hash}
            for name, (path, expected_hash) in SOURCES.items()
        },
    }
    if "qvla_actquant" in data:
        result["qvla_actquant_extension"] = audit_qvla_actquant(
            data["qvla_actquant"]
        )
    return result


def pct(value: float) -> str:
    # Use conventional half-up display rather than Python's ties-to-even formatting.
    # The frozen JSON may encode decimal ties a few ulps below their exact value.
    rounded = math.floor(1000 * value + 0.5 + 1e-9) / 10
    return f"{rounded:.1f}"


def reproduction_table_row(data: dict[str, dict[str, Any]], method: str, model: str) -> str:
    extension = data.get("qvla_actquant")
    label = (
        r"QVLA-code Wavg4/A16$^{\S}$"
        if method == "qvla"
        else r"ActQuant 4.0 BPW/A16$^{\S}$"
    )
    allocation = (
        r"Ch. 0/2/4/8/16"
        if method == "qvla"
        else r"Tensor IQ2--Q4"
    )
    if extension is None:
        return (
            f"\\quad {label} & {allocation} & "
            "\\multicolumn{6}{c}{\\textit{Pending}} \\\\"
        )
    row = extension["arms"][f"{method}_{model}"]
    splits = row["split_task_macro_success_rate"]
    storage = row["storage"]
    static_gib = (
        float(storage["static_gib_total_deployed_checkpoint_set"])
        / int(storage["checkpoint_count"])
    )
    return (
        f"\\quad {label} & {allocation} & {pct(splits['atomic_seen'])} & "
        f"{pct(splits['composite_seen'])} & {pct(splits['composite_unseen'])} & "
        f"{pct(row['task_macro_success_rate'])} & {static_gib:.3f} & "
        f"{float(storage['compression_ratio_total_deployed_checkpoint_set']):.2f}$\\times$ \\\\"
    )


def main_table(data: dict[str, dict[str, Any]]) -> str:
    agg = data["gr00t_aggregate"]
    c = agg["candidate"]
    fp = agg["comparisons"]["fp16"]["baseline"]
    q = agg["comparisons"]["quantvla_w4a8"]["baseline"]
    cs = c["split_task_macro_success_rate"]
    fs = fp["split_task_macro_success_rate"]
    qs = q["split_task_macro_success_rate"]
    pi_static = data["pi05_static_official"]
    pi_fp = pi_static["configs"]["fp16"]
    pi_q = pi_static["configs"]["quantvla_w4a8_atmohb"]
    pi_w6 = data["pi05_uniform_w6"]["configs"]["uniform_w6"]
    pi_omega = data["pi05_omega"]
    pi_dypac = data["pi05_dypac_formal"]["result"]
    pifs = pi_fp["task_set_macro_sr"]
    piqs = pi_q["task_set_macro_sr"]
    piws = pi_w6["task_set_macro_sr"]
    pios = {name: row["task_macro_sr"] for name, row in pi_omega["task_sets"].items()}
    pids = pi_dypac["split_task_macro_success_rate"]
    gr_qvla = reproduction_table_row(data, "qvla", "gr00t")
    gr_actquant = reproduction_table_row(data, "actquant", "gr00t")
    pi_qvla = reproduction_table_row(data, "qvla", "pi05")
    pi_actquant = reproduction_table_row(data, "actquant", "pi05")
    allocation_header = "Allocation"
    table_column_separation = "1.5pt"
    extension_note = (
        "$^{\\S}$QVLA/ActQuant~\\cite{xu2026qvla,akbari2026actquant} use local FP16-teacher proxy calibration "
        "(source-protocol-equivalent=false). Their sizes are measured static packs; "
        "GR00T reports the mean across its three split checkpoints. "
        if "qvla_actquant" in data
        else "$^{\\S}$QVLA/ActQuant~\\cite{xu2026qvla,akbari2026actquant} formal extensions are pending. "
    )
    return f"""% AUTO-GENERATED by scripts/tools/render_dypac_paper.py; DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\caption{{RoboCasa365 results and exact static storage for GR00T N1.5 and $\\pi_{{0.5}}$. Completed target-split rows use 50 paired scenarios per task.}}
\\label{{tab:main_results}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{{table_column_separation}}}
\\renewcommand{{\\arraystretch}}{{1.06}}
\\begin{{tabular}}{{@{{}}lcrrrrrr@{{}}}}
\\toprule
Configuration & {allocation_header} & \\shortstack{{Atomic\\\\SR $\\uparrow$}} & \\shortstack{{C-Seen\\\\SR $\\uparrow$}} & \\shortstack{{C-Unseen\\\\SR $\\uparrow$}} & \\shortstack{{All\\\\SR $\\uparrow$}} & \\shortstack{{Size\\\\(GiB) $\\downarrow$}} & \\shortstack{{Comp.\\\\$\\uparrow$}} \\\\
\\midrule
\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{GR00T N1.5}}\\enspace\\textit{{(formal target split)}}}} \\\\
\\quad FP16 & -- & {pct(fs['atomic_seen'])} & {pct(fs['composite_seen'])} & {pct(fs['composite_unseen'])} & {pct(fp['task_macro_success_rate'])} & 1.993 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 116 W4 & {pct(qs['atomic_seen'])} & {pct(qs['composite_seen'])} & {pct(qs['composite_unseen'])} & {pct(q['task_macro_success_rate'])} & 0.898 & 2.22$\\times$ \\\\
\\quad Uniform W6 & 116 W6 & 68.4 & 42.9 & 41.8 & 51.7 & 1.109 & 1.80$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 180 W4 & 60.1 & 25.9 & 27.5 & 38.7 & 0.599 & 3.33$\\times$ \\\\
{gr_qvla}
{gr_actquant}
\\quad \\textbf{{\\method (Ours)}}$^{{\\dagger}}$ & \\textbf{{100 W4}} & \\textbf{{{pct(cs['atomic_seen'])}}} & \\textbf{{{pct(cs['composite_seen'])}}} & \\textbf{{{pct(cs['composite_unseen'])}}} & \\textbf{{{pct(c['task_macro_success_rate'])}}} & \\textbf{{0.896}} & \\textbf{{2.22$\\times$}} \\\\
\\midrule
\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{$\\pi_{{0.5}}$}}~\\cite{{intelligence2025pi05}}\\enspace\\textit{{(formal target split)}}}} \\\\
\\quad FP16 & -- & {pct(pifs['atomic_seen'])} & {pct(pifs['composite_seen'])} & {pct(pifs['composite_unseen'])} & {pct(pi_fp['task_macro_sr'])} & 4.113 & 1.00$\\times$ \\\\
\\quad \\quantvla W4A8 & 180 W4 & {pct(piqs['atomic_seen'])} & {pct(piqs['composite_seen'])} & {pct(piqs['composite_unseen'])} & {pct(pi_q['task_macro_sr'])} & 1.388 & 2.96$\\times$ \\\\
\\quad Uniform W6 & 180 W6 & {pct(piws['atomic_seen'])} & {pct(piws['composite_seen'])} & {pct(piws['composite_unseen'])} & {pct(pi_w6['task_macro_sr'])} & 1.902 & 2.16$\\times$ \\\\
\\quad $\\Omega$-QVLA W4A4$^{{\\ddagger}}$ & 252 W4 & {pct(pios['atomic_seen'])} & {pct(pios['composite_seen'])} & {pct(pios['composite_unseen'])} & {pct(pi_omega['task_macro_sr'])} & 1.307 & 3.27$\\times$ \\\\
{pi_qvla}
{pi_actquant}
\\quad \\textbf{{\\method (Ours)}}$^{{\\dagger}}$ & \\textbf{{121 W4}} & \\textbf{{{pct(pids['atomic_seen'])}}} & \\textbf{{{pct(pids['composite_seen'])}}} & \\textbf{{{pct(pids['composite_unseen'])}}} & \\textbf{{{pct(pi_dypac['task_macro_success_rate'])}}} & \\textbf{{1.523}} & \\textbf{{2.70$\\times$}} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{0.99\\textwidth}}{{\\footnotesize C-Seen/C-Unseen denote Composite-Seen/Composite-Unseen. All completed $\\pi_{{0.5}}$ RoboCasa365 rows use four flow steps. $^{{\\dagger}}$For each Ours row, the four tasks used to select the initializer ratio are included in the 50-task aggregate. GR00T ours uses 962,068,480 bytes; $\\pi_{{0.5}}$ ours uses 1,634,828,288 bytes. $^{{\\ddagger}}\\Omega$-QVLA uses model- and task-set-specific calibration. {extension_note}Compression denotes static storage, not PyTorch/C++ runtime memory or latency.}}
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
Proposal audit & Every valid one-layer state flip & 0/{len(data['gr00t_plan']['layers'])} positive benefits & No eligible flip \\\\
Policy audit & {len(summaries)} structured mask alternatives & 0/{len(summaries)} eligible; $J\\in[{min(objectives):.2f},{max(objectives):.2f}]$ & Abstain; retain $M_0$ \\\\
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
\providecommand{\PiAnchorStatus}{DyPAC-VLA obtains 27.7\% over 2,500 $\pi_{0.5}$ episodes at 2.702$\times$ static component compression.}
"""


def generated_files(data: dict[str, dict[str, Any]], registry: dict[str, Any]) -> dict[Path, str]:
    source_hashes = {name: digest for name, (_, digest) in SOURCES.items()}
    if "qvla_actquant" in data:
        source_hashes["qvla_actquant"] = sha256(QVLA_ACTQUANT_AGGREGATE)
    audit_payload = {
        "schema_version": 1,
        "kind": "dypac_vla_paper_render_audit",
        "valid": True,
        "source_hashes": source_hashes,
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
