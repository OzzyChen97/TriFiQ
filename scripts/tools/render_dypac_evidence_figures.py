#!/usr/bin/env python3
"""Render the audited evidence figures used by the DyPAC paper.

The script reads only frozen experiment artifacts.  It deliberately separates
closed-loop behavior, descriptive activation diagnostics, and the fail-closed
selection audit so that the visual evidence cannot imply a stronger claim than
the underlying experiment supports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from quantvla_full_context import conservative_fp16_benefit, task_group_map


GIB = float(1 << 30)
ACTION_RE = re.compile(
    r"^action_head\.model\.transformer_blocks\.(\d+)\.ff\.net\.(0\.proj|2)$"
)

SOURCES = {
    "gr_result": "runs/full_context_v2/table1/aggregate.json",
    "gr_manifest": "runs/full_context_v2/table1/manifest.json",
    "gr_plan": "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json",
    "gr_uniform": (
        "runs/gdsq_week1_preregistered_v1/execution/aggregate/"
        "gr00t_uniform_w6/summary.json"
    ),
    "gr_omega": (
        "runs/gdsq_extension_preregistered_v1/omega_qvla_robocasa365_v1/"
        "aggregate/summary.json"
    ),
    "gr_omega_memory": (
        "runs/gdsq_extension_preregistered_v1/omega_qvla_robocasa365_v1/"
        "aggregate/atomic_seen_paper_memory.json"
    ),
    "pi_result": "runs/full_context_v2/pi05_table1/aggregate.json",
    "pi_plan": "runs/full_context_v2/pi05_p2/pi05_full_context_v2_frozen.json",
    "pi_baselines": (
        "runs/pi05_gdsq_gr00t_aligned/official_target_paired50/aggregate/"
        "summary.json"
    ),
    "pi_uniform": (
        "runs/gdsq_week1_preregistered_v1/execution/runs/"
        "pi05_uniform_w6_official50/aggregate/summary.json"
    ),
    "pi_omega": (
        "runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1/"
        "aggregate/summary.json"
    ),
    "pi_omega_memory": (
        "runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1/"
        "aggregate/pi05_paper_memory.json"
    ),
    "activation": "runs/full_context_v2/p2/activation_attribution.json",
    "rollout": "runs/full_context_v2/table3_quick/aggregate.json",
    "static_atomic": (
        "runs/full_context_v2/table3_quick/static_a8/atomic_seen/m0_static_a8.npz"
    ),
    "static_seen": (
        "runs/full_context_v2/table3_quick/static_a8/composite_seen/"
        "m0_static_a8.npz"
    ),
    "static_unseen": (
        "runs/full_context_v2/table3_quick/static_a8/composite_unseen/"
        "m0_static_a8.npz"
    ),
    "flip_scores": "runs/full_context_v2/p2/flip_scores/flip_scores_cross_split.json",
    "flip_manifest": "runs/full_context_v2/p2/interventions_dynamic/manifest.json",
    "selection": "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json.selection.json",
}

OUT_DIR = "docs/gdsq_vla_iclr2027/figures"

COLORS = {
    "ours": "#0072B2",
    "quantvla": "#D55E00",
    "fp16": "#4D5963",
    "uniform": "#009E73",
    "omega": "#CC79A7",
    "atomic_seen": "#0072B2",
    "composite_seen": "#E69F00",
    "composite_unseen": "#009E73",
    "static": "#D55E00",
    "dynamic": "#0072B2",
    "ink": "#243746",
    "muted": "#687780",
    "grid": "#D9E0E5",
    "soft": "#F3F6F8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(observed: float, expected: float, name: str, tol: float = 1e-9) -> None:
    if not np.isclose(observed, expected, atol=tol, rtol=0.0):
        raise ValueError(f"{name}: expected {expected}, observed {observed}")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def clean_axis(axis: mpl.axes.Axes, grid_axis: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis:
        axis.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)


def save_figure(fig: mpl.figure.Figure, stem: Path) -> tuple[Path, Path]:
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(
        png_path,
        dpi=320,
        facecolor="white",
        metadata={"Software": Path(__file__).name},
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={"Creator": Path(__file__).name, "CreationDate": None},
    )
    plt.close(fig)
    return png_path, pdf_path


def precision(layer: dict[str, Any]) -> str:
    return "FP16" if layer.get("skip") else f"W{int(layer['bits'])}"


def action_layer_order(layers: dict[str, dict[str, Any]]) -> list[str]:
    parsed: list[tuple[int, int, str]] = []
    for name in layers:
        match = ACTION_RE.match(name)
        if match:
            parsed.append(
                (int(match.group(1)), 0 if match.group(2) == "0.proj" else 1, name)
            )
    parsed.sort()
    names = [row[2] for row in parsed]
    if len(names) != 32:
        raise ValueError(f"Expected 32 action-head FFN targets, found {len(names)}")
    return names


def load_operating_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    gr_result = load_json(paths["gr_result"])
    gr_manifest = load_json(paths["gr_manifest"])
    gr_plan = load_json(paths["gr_plan"])
    gr_uniform = load_json(paths["gr_uniform"])
    gr_omega = load_json(paths["gr_omega"])
    gr_omega_memory = load_json(paths["gr_omega_memory"])
    pi_result = load_json(paths["pi_result"])
    pi_plan = load_json(paths["pi_plan"])
    pi_baselines = load_json(paths["pi_baselines"])
    pi_uniform = load_json(paths["pi_uniform"])
    pi_omega = load_json(paths["pi_omega"])
    pi_omega_memory = load_json(paths["pi_omega_memory"])

    gr_quant_bytes = int(
        gr_manifest["compression_claim"]["quantvla_storage_cell_bytes"]
    )
    gr_uniform_scope = gr_uniform["paper_style_memory"]["by_task_set"][
        "atomic_seen"
    ]
    gr_uniform_memory = gr_uniform_scope["configs"]["uniform_w6"]
    gr_points = [
        {
            "id": "omega",
            "label": r"$\Omega$-QVLA",
            "size": float(gr_omega_memory["packed"]["component_bytes"]) / GIB,
            "success": float(gr_omega["configs"]["omega_qvla_w4a4"]["task_macro_sr"]),
        },
        {
            "id": "quantvla",
            "label": "QuantVLA",
            "size": gr_quant_bytes / GIB,
            "success": float(
                gr_result["comparisons"]["quantvla_w4a8"]["baseline"][
                    "task_macro_success_rate"
                ]
            ),
        },
        {
            "id": "ours",
            "label": "DyPAC",
            "size": float(gr_plan["table1_total_static_bytes"]) / GIB,
            "success": float(gr_result["candidate"]["task_macro_success_rate"]),
        },
        {
            "id": "uniform",
            "label": "Uniform W6",
            "size": float(gr_uniform_memory["component_bytes"]) / GIB,
            "success": float(gr_uniform["configs"]["uniform_w6"]["task_macro_sr"]),
        },
        {
            "id": "fp16",
            "label": "FP16",
            "size": float(gr_uniform_scope["fp16_component_bytes"]) / GIB,
            "success": float(
                gr_result["comparisons"]["fp16"]["baseline"][
                    "task_macro_success_rate"
                ]
            ),
        },
    ]

    pi_memory = pi_baselines["paper_style_memory"]["by_config"]
    pi_points = [
        {
            "id": "omega",
            "label": r"$\Omega$-QVLA",
            "size": float(pi_omega_memory["packed"]["component_bytes"]) / GIB,
            "success": float(pi_omega["task_macro_sr"]),
        },
        {
            "id": "quantvla",
            "label": "QuantVLA",
            "size": float(pi_memory["quantvla_w4a8_atmohb"]["bytes"]) / GIB,
            "success": float(
                pi_baselines["configs"]["quantvla_w4a8_atmohb"]["task_macro_sr"]
            ),
        },
        {
            "id": "ours",
            "label": "DyPAC",
            "size": float(pi_plan["table1_total_static_bytes"]) / GIB,
            "success": float(pi_result["result"]["task_macro_success_rate"]),
        },
        {
            "id": "uniform",
            "label": "Uniform W6",
            "size": float(pi_uniform["quantization"]["theoretical_static_bytes"])
            / GIB,
            "success": float(pi_uniform["configs"]["uniform_w6"]["task_macro_sr"]),
        },
        {
            "id": "fp16",
            "label": "FP16",
            "size": float(pi_memory["fp16"]["bytes"]) / GIB,
            "success": float(pi_baselines["configs"]["fp16"]["task_macro_sr"]),
        },
    ]

    assert_close(gr_points[2]["success"], 0.540, "GR00T DyPAC SR")
    assert_close(gr_points[1]["success"], 0.3044, "GR00T QuantVLA SR")
    assert_close(gr_points[3]["success"], 0.5172, "GR00T Uniform W6 SR")
    assert_close(pi_points[2]["success"], 0.2772, "pi0.5 DyPAC SR")
    assert_close(pi_points[1]["success"], 0.2484, "pi0.5 QuantVLA SR")

    ours_tasks = gr_result["candidate"]["per_task_success_rate"]
    baselines = {
        "QuantVLA": gr_result["comparisons"]["quantvla_w4a8"]["baseline"][
            "per_task_success_rate"
        ],
        "FP16": gr_result["comparisons"]["fp16"]["baseline"][
            "per_task_success_rate"
        ],
    }
    groups = task_group_map()
    task_deltas: dict[str, list[dict[str, Any]]] = {}
    direction_counts: dict[str, dict[str, int]] = {}
    for baseline_name, baseline_tasks in baselines.items():
        rows = []
        for task in sorted(ours_tasks):
            rows.append(
                {
                    "task": task,
                    "group": groups[task],
                    "delta": float(ours_tasks[task] - baseline_tasks[task]),
                }
            )
        values = np.asarray([row["delta"] for row in rows])
        direction_counts[baseline_name] = {
            "better": int(np.count_nonzero(values > 1e-12)),
            "tie": int(np.count_nonzero(np.abs(values) <= 1e-12)),
            "worse": int(np.count_nonzero(values < -1e-12)),
        }
        task_deltas[baseline_name] = rows
    if direction_counts["QuantVLA"] != {"better": 48, "tie": 1, "worse": 1}:
        raise ValueError("Unexpected GR00T task-direction counts against QuantVLA")
    if direction_counts["FP16"] != {"better": 18, "tie": 6, "worse": 26}:
        raise ValueError("Unexpected GR00T task-direction counts against FP16")

    return {
        "gr_points": gr_points,
        "pi_points": pi_points,
        "task_deltas": task_deltas,
        "direction_counts": direction_counts,
        "gr_paired": {
            key: {
                item: gr_result["comparisons"][key][item]
                for item in (
                    "paired_wins",
                    "paired_losses",
                    "paired_ties",
                    "holm_adjusted_mcnemar_p",
                )
            }
            for key in ("quantvla_w4a8", "fp16")
        },
    }


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    for point in sorted(points, key=lambda row: row["size"]):
        dominated = any(
            other["size"] <= point["size"]
            and other["success"] >= point["success"]
            and (
                other["size"] < point["size"]
                or other["success"] > point["success"]
            )
            for other in points
        )
        if not dominated:
            front.append(point)
    return front


def plot_operating_panel(
    axis: mpl.axes.Axes,
    points: list[dict[str, Any]],
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    annotations: dict[str, tuple[int, int]],
) -> list[str]:
    front = pareto_front(points)
    axis.plot(
        [row["size"] for row in front],
        [row["success"] for row in front],
        color=COLORS["ink"],
        linewidth=1.1,
        linestyle="--",
        alpha=0.75,
        zorder=1,
        label="non-dominated envelope",
    )
    for point in points:
        is_ours = point["id"] == "ours"
        axis.scatter(
            point["size"],
            point["success"],
            s=58 if is_ours else 35,
            marker="*" if is_ours else "o",
            facecolor=COLORS[point["id"]],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        dx, dy = annotations[point["id"]]
        axis.annotate(
            f"{point['label']}\n{point['success']:.1%}, {point['size']:.3f}",
            (point["size"], point["success"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=6.2,
            color=COLORS["ink"],
            linespacing=1.05,
        )
    axis.set_title(title, loc="left", fontweight="bold", pad=6)
    axis.set_xlabel("Static size (GiB; lower is better)")
    axis.set_ylabel("Task-macro success")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    clean_axis(axis, "both")
    return [row["id"] for row in front]


def render_operating_figure(output_dir: Path, evidence: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    fig = plt.figure(figsize=(7.25, 2.92), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.12),
        left=0.075,
        right=0.985,
        bottom=0.255,
        top=0.875,
        wspace=0.30,
    )
    ax_a, ax_b = [fig.add_subplot(grid[0, index]) for index in range(2)]

    gr_front = plot_operating_panel(
        ax_a,
        evidence["gr_points"],
        "a   GR00T success–storage",
        (0.48, 2.12),
        (0.26, 0.59),
        {
            "omega": (4, 4),
            "quantvla": (4, 5),
            "ours": (-4, 5),
            "uniform": (4, -5),
            "fp16": (-4, -5),
        },
    )

    rng = np.random.default_rng(0)
    group_names = ("atomic_seen", "composite_seen", "composite_unseen")
    group_labels = ("Atomic", "C-Seen", "C-Unseen")
    x_positions = {"QuantVLA": 0.0, "FP16": 1.0}
    for baseline_name, base_x in x_positions.items():
        rows = evidence["task_deltas"][baseline_name]
        values = np.asarray([row["delta"] for row in rows])
        box = ax_b.boxplot(
            [values],
            positions=[base_x],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": COLORS["ink"], "linewidth": 1.3},
            whiskerprops={"color": COLORS["muted"], "linewidth": 0.8},
            capprops={"color": COLORS["muted"], "linewidth": 0.8},
        )
        box["boxes"][0].set_facecolor(COLORS["soft"])
        box["boxes"][0].set_edgecolor(COLORS["muted"])
        for group_name in group_names:
            group_rows = [row for row in rows if row["group"] == group_name]
            jitter = rng.uniform(-0.14, 0.14, len(group_rows))
            ax_b.scatter(
                base_x + jitter,
                [row["delta"] for row in group_rows],
                s=14,
                color=COLORS[group_name],
                edgecolor="white",
                linewidth=0.3,
                alpha=0.83,
                zorder=3,
            )
        counts = evidence["direction_counts"][baseline_name]
        ax_b.text(
            base_x,
            0.585,
            f"{counts['better']} / {counts['tie']} / {counts['worse']}",
            ha="center",
            va="top",
            fontsize=6.4,
            fontweight="bold",
            color=COLORS["ink"],
        )
    ax_b.axhline(0.0, color="#87949C", linewidth=0.8, zorder=1)
    ax_b.set_title("b   GR00T task-level consistency", loc="left", fontweight="bold", pad=6)
    ax_b.set_xticks([0, 1], labels=["vs. QuantVLA", "vs. FP16"])
    ax_b.set_ylabel("DyPAC $-$ baseline success (pp)")
    ax_b.set_ylim(-0.17, 0.61)
    ax_b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    clean_axis(ax_b, "y")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[name],
            markeredgecolor="white",
            markersize=5,
            label=label,
        )
        for name, label in zip(group_names, group_labels)
    ]
    ax_b.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=3,
        frameon=False,
        columnspacing=0.7,
        handletextpad=0.2,
    )

    pi_front = [row["id"] for row in pareto_front(evidence["pi_points"])]
    return (*save_figure(fig, output_dir / "dypac_operating_points"), {
        "gr_pareto_ids": gr_front,
        "pi05_pareto_ids": pi_front,
    })


def load_dynamic_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    plan = load_json(paths["gr_plan"])
    activation = load_json(paths["activation"])
    rollout = load_json(paths["rollout"])
    action_names = action_layer_order(plan["layers"])
    archives = [
        np.load(paths["static_atomic"]),
        np.load(paths["static_seen"]),
        np.load(paths["static_unseen"]),
    ]
    drift_by_layer: dict[str, np.ndarray] = {}
    for name in action_names:
        if precision(plan["layers"][name]) == "FP16":
            continue
        tables = [archive[name] for archive in archives]
        if any(table.ndim != 2 or table.shape[0] != 4 for table in tables):
            raise ValueError(f"Expected four flow-step rows for {name}")
        scale = np.median(np.stack(tables, axis=0), axis=0).astype(np.float64)
        denominator = max(float(np.linalg.norm(scale[0])), 1e-12)
        drift_by_layer[name] = np.linalg.norm(scale - scale[0], axis=1) / denominator
    heatmap = np.full((len(action_names), 4), np.nan, dtype=np.float64)
    for index, name in enumerate(action_names):
        if name in drift_by_layer:
            heatmap[index] = drift_by_layer[name]
    roles = {
        "FFN expand": np.stack(
            [
                drift_by_layer[name]
                for name in action_names
                if name in drift_by_layer and name.endswith("net.0.proj")
            ]
        ),
        "FFN contract": np.stack(
            [
                drift_by_layer[name]
                for name in action_names
                if name in drift_by_layer and name.endswith("net.2")
            ]
        ),
    }
    baseline = rollout["baseline"]
    static = rollout["arms"]["static_a8"]
    if baseline["episodes"] != 500 or static["episodes"] != 500:
        raise ValueError("Dynamic/static rollout is not 500 episodes per arm")
    static_j = float(activation["static_vs_a16"]["objective"])
    dynamic_j = float(activation["dynamic_vs_a16"]["objective"])
    assert_close(static_j, 467.5204744598872, "static-A8 attribution J")
    assert_close(dynamic_j, 0.5178887111755559, "dynamic-A8 attribution J")
    if static["successes"] != 0 or baseline["successes"] != 269:
        raise ValueError("Unexpected dynamic/static rollout result")
    return {
        "action_names": action_names,
        "layers": plan["layers"],
        "heatmap": heatmap,
        "roles": roles,
        "static_j": static_j,
        "dynamic_j": dynamic_j,
        "dynamic": baseline,
        "static": static,
    }


def render_dynamic_figure_legacy(output_dir: Path, evidence: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    fig = plt.figure(figsize=(7.25, 5.02), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.12),
        height_ratios=(1.0, 0.88),
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.925,
        wspace=0.34,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    heat_grid = outer[0, 1].subgridspec(1, 2, width_ratios=(1.0, 0.055), wspace=0.05)
    ax_b = fig.add_subplot(heat_grid[0, 0])
    ax_bits = fig.add_subplot(heat_grid[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    x_step = np.arange(1, 5)
    for label, color in (("FFN expand", COLORS["ours"]), ("FFN contract", COLORS["static"])):
        values = evidence["roles"][label]
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        ax_a.fill_between(x_step, q25, q75, color=color, alpha=0.17, linewidth=0)
        ax_a.plot(x_step, median, color=color, marker="o", markersize=3.7, label=label)
        ax_a.annotate(
            f"{median[-1]:.1%}",
            (4, median[-1]),
            xytext=(-4, 5 if label == "FFN contract" else -11),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
            fontweight="bold",
            color=color,
        )
    ax_a.set_title("a   Frozen-range scale drift", loc="left", fontweight="bold", pad=6)
    ax_a.set_xlabel("Flow step")
    ax_a.set_ylabel(r"$\|s_t-s_1\|_2/\|s_1\|_2$")
    ax_a.set_xticks(x_step)
    ax_a.set_ylim(-0.01, 0.50)
    ax_a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_a.legend(frameon=False, loc="upper left")
    ax_a.text(
        0.02,
        0.035,
        "median ± IQR across applicable W4 layers",
        transform=ax_a.transAxes,
        fontsize=6.0,
        color=COLORS["muted"],
    )
    clean_axis(ax_a, "y")

    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#D9DDE0")
    image = ax_b.imshow(
        np.ma.masked_invalid(evidence["heatmap"]),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=0.50,
    )
    ax_b.set_title("b   Action-head layer × step", loc="left", fontweight="bold", pad=6)
    ax_b.set_xlabel("Flow step")
    ax_b.set_xticks(np.arange(4), labels=("1", "2", "3", "4"))
    ax_b.set_yticks(np.arange(0.5, 32, 2), labels=[f"B{i}" for i in range(16)])
    ax_b.set_ylabel("Transformer block")
    ax_b.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, 32, 2), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.35, alpha=0.75)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    bits = np.asarray(
        [
            [0 if precision(evidence["layers"][name]) == "W4" else 1]
            for name in evidence["action_names"]
        ]
    )
    ax_bits.imshow(
        bits,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap((COLORS["ours"], COLORS["static"])),
        vmin=0,
        vmax=1,
    )
    ax_bits.set_title("bits", fontsize=6.2, pad=8)
    ax_bits.set_xticks([])
    ax_bits.set_yticks([])
    for spine in ax_bits.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=(ax_b, ax_bits), fraction=0.055, pad=0.16, aspect=27)
    colorbar.set_ticks((0.0, 0.25, 0.50), labels=("0%", "25%", "50%"))
    colorbar.ax.tick_params(labelsize=5.9, length=2)
    colorbar.set_label("relative L2 scale drift", fontsize=6.1, labelpad=1)
    colorbar.outline.set_linewidth(0.45)

    labels = ("Frozen A8", "Dynamic A8")
    objectives = (evidence["static_j"], evidence["dynamic_j"])
    bars = ax_c.barh(
        [0, 1],
        objectives,
        color=(COLORS["static"], COLORS["dynamic"]),
        height=0.52,
        zorder=3,
    )
    ax_c.set_xscale("log")
    ax_c.set_yticks([0, 1], labels=labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel(r"Offline minimax objective $J$  $\leftarrow$ smaller")
    ax_c.set_title("c   Matched A8 attribution", loc="left", fontweight="bold", pad=6)
    ax_c.set_xlim(0.25, 900)
    for bar, value in zip(bars, objectives):
        ax_c.text(
            value * 1.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=COLORS["ink"],
        )
    ax_c.text(
        0.98,
        0.92,
        f"{objectives[0] / objectives[1]:.0f}× lower",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color=COLORS["dynamic"],
    )
    clean_axis(ax_c, "x")

    split_keys = ("atomic_seen", "composite_seen", "composite_unseen")
    split_labels = ("Atomic", "C-Seen", "C-Unseen", "All")
    dynamic_values = [
        evidence["dynamic"]["split_task_macro_success_rate"][key] for key in split_keys
    ] + [evidence["dynamic"]["task_macro_success_rate"]]
    static_values = [
        evidence["static"]["split_task_macro_success_rate"][key] for key in split_keys
    ] + [evidence["static"]["task_macro_success_rate"]]
    x = np.arange(4)
    width = 0.34
    ax_d.bar(
        x - width / 2,
        static_values,
        width,
        color=COLORS["static"],
        label="Frozen A8",
        zorder=3,
    )
    dynamic_bars = ax_d.bar(
        x + width / 2,
        dynamic_values,
        width,
        color=COLORS["dynamic"],
        label="Dynamic A8",
        zorder=3,
    )
    for bar, value in zip(dynamic_bars, dynamic_values):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=5.9,
            fontweight="bold",
            color=COLORS["ink"],
        )
    ax_d.set_title("d   Paired closed-loop intervention", loc="left", fontweight="bold", pad=6)
    ax_d.set_xticks(x, labels=split_labels)
    ax_d.set_ylabel("Task-macro success")
    ax_d.set_ylim(0, 0.88)
    ax_d.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_d.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.8)
    ax_d.text(
        0.50,
        0.035,
        "paired outcome: 0/500  →  269/500 successes",
        transform=ax_d.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.1,
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "none", "alpha": 0.90},
    )
    clean_axis(ax_d, "y")

    valid = evidence["heatmap"][np.isfinite(evidence["heatmap"])]
    max_row, max_step = np.unravel_index(
        np.nanargmax(evidence["heatmap"]), evidence["heatmap"].shape
    )
    derived = {
        "all_action_head_median_drift_by_step": [
            float(value) for value in np.nanmedian(evidence["heatmap"], axis=0)
        ],
        "maximum_drift": float(valid.max()),
        "maximum_drift_layer": evidence["action_names"][max_row],
        "maximum_drift_flow_step": int(max_step + 1),
        "static_objective": evidence["static_j"],
        "dynamic_objective": evidence["dynamic_j"],
        "objective_ratio": evidence["static_j"] / evidence["dynamic_j"],
        "static_successes": evidence["static"]["successes"],
        "dynamic_successes": evidence["dynamic"]["successes"],
    }
    return (*save_figure(fig, output_dir / "dypac_dynamic_range_evidence"), derived)


def render_dynamic_figure(output_dir: Path, evidence: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    """Render a compact main-paper view of the registered range evidence."""
    fig = plt.figure(figsize=(7.25, 2.42), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.08, 0.82, 1.30),
        left=0.070,
        right=0.988,
        bottom=0.235,
        top=0.865,
        wspace=0.42,
    )
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, index]) for index in range(3)]

    x_step = np.arange(1, 5)
    for label, color in (("FFN expand", COLORS["ours"]), ("FFN contract", COLORS["static"])):
        values = evidence["roles"][label]
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        ax_a.fill_between(x_step, q25, q75, color=color, alpha=0.16, linewidth=0)
        ax_a.plot(x_step, median, color=color, marker="o", markersize=3.4, label=label)
        ax_a.annotate(
            f"{median[-1]:.1%}",
            (4, median[-1]),
            xytext=(-3, 5 if label == "FFN contract" else -10),
            textcoords="offset points",
            ha="right",
            fontsize=6.2,
            fontweight="bold",
            color=color,
        )
    ax_a.set_title("a   Frozen-range drift", loc="left", fontweight="bold", pad=5)
    ax_a.set_xlabel("Flow step")
    ax_a.set_ylabel(r"$\|s_t-s_1\|_2/\|s_1\|_2$")
    ax_a.set_xticks(x_step)
    ax_a.set_ylim(-0.01, 0.50)
    ax_a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_a.legend(frameon=False, loc="upper left", handlelength=1.5)
    clean_axis(ax_a, "y")

    labels = ("Frozen A8", "Dynamic A8")
    objectives = (evidence["static_j"], evidence["dynamic_j"])
    bars = ax_b.barh(
        [0, 1],
        objectives,
        color=(COLORS["static"], COLORS["dynamic"]),
        height=0.48,
        zorder=3,
    )
    ax_b.set_xscale("log")
    ax_b.set_yticks([0, 1], labels=labels)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"Objective $J$  $\leftarrow$")
    ax_b.set_title("b   Matched attribution", loc="left", fontweight="bold", pad=5)
    ax_b.set_xlim(0.25, 900)
    for bar, value in zip(bars, objectives):
        ax_b.text(
            value * 1.10,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=COLORS["ink"],
        )
    ax_b.text(
        0.98,
        0.94,
        f"{objectives[0] / objectives[1]:.0f}× lower",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["dynamic"],
    )
    clean_axis(ax_b, "x")

    split_keys = ("atomic_seen", "composite_seen", "composite_unseen")
    split_labels = ("Atomic", "C-Seen", "C-Unseen", "All")
    dynamic_values = [
        evidence["dynamic"]["split_task_macro_success_rate"][key] for key in split_keys
    ] + [evidence["dynamic"]["task_macro_success_rate"]]
    static_values = [
        evidence["static"]["split_task_macro_success_rate"][key] for key in split_keys
    ] + [evidence["static"]["task_macro_success_rate"]]
    x = np.arange(4)
    width = 0.34
    ax_c.bar(
        x - width / 2,
        static_values,
        width,
        color=COLORS["static"],
        label="Frozen A8",
        zorder=3,
    )
    dynamic_bars = ax_c.bar(
        x + width / 2,
        dynamic_values,
        width,
        color=COLORS["dynamic"],
        label="Dynamic A8",
        zorder=3,
    )
    for bar, value in zip(dynamic_bars, dynamic_values):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
            color=COLORS["ink"],
        )
    ax_c.set_title("c   Paired closed-loop intervention", loc="left", fontweight="bold", pad=5)
    ax_c.set_xticks(x, labels=split_labels)
    ax_c.set_ylabel("Task-macro success")
    ax_c.set_ylim(0, 0.88)
    ax_c.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_c.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.7)
    ax_c.text(
        0.50,
        0.035,
        "0/500  →  269/500 successes",
        transform=ax_c.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.90},
    )
    clean_axis(ax_c, "y")

    valid = evidence["heatmap"][np.isfinite(evidence["heatmap"])]
    max_row, max_step = np.unravel_index(
        np.nanargmax(evidence["heatmap"]), evidence["heatmap"].shape
    )
    derived = {
        "all_action_head_median_drift_by_step": [
            float(value) for value in np.nanmedian(evidence["heatmap"], axis=0)
        ],
        "maximum_drift": float(valid.max()),
        "maximum_drift_layer": evidence["action_names"][max_row],
        "maximum_drift_flow_step": int(max_step + 1),
        "static_objective": evidence["static_j"],
        "dynamic_objective": evidence["dynamic_j"],
        "objective_ratio": evidence["static_j"] / evidence["dynamic_j"],
        "static_successes": evidence["static"]["successes"],
        "dynamic_successes": evidence["dynamic"]["successes"],
    }
    return (*save_figure(fig, output_dir / "dypac_dynamic_range_evidence"), derived)


def load_fcp_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    score_payload = load_json(paths["flip_scores"])
    score_rows = score_payload["scores"]
    manifest = load_json(paths["flip_manifest"])
    plan = load_json(paths["gr_plan"])
    selection = load_json(paths["selection"])
    baseline = score_rows["context_base"]
    changes = []
    verified_candidates = 0
    for record in manifest["candidates"]:
        candidate_id = record["candidate_id"]
        recorded_path = Path(record["path"])
        if not recorded_path.exists():
            marker = "/runs/"
            recorded_value = str(record["path"])
            if marker not in recorded_value:
                raise FileNotFoundError(recorded_value)
            repo_root = paths["gr_plan"].parents[3]
            recorded_path = repo_root / ("runs/" + recorded_value.split(marker, 1)[1])
        if sha256(recorded_path) != record["sha256"]:
            raise ValueError(f"Candidate-plan hash mismatch: {candidate_id}")
        verified_candidates += 1
        if candidate_id == "context_base":
            continue
        flip = record["flip"]
        current_is_fp16 = flip["from"].lower() == "fp16"
        benefit = conservative_fp16_benefit(
            current_is_fp16=current_is_fp16,
            flip=score_rows[candidate_id],
            baseline=baseline,
        )
        changes.append(
            {
                "candidate_id": candidate_id,
                "layer": flip["layer"],
                "direction": "FP16→W4" if current_is_fp16 else "W4→FP16",
                "benefit_d_func": float(benefit["d_func"]),
                "benefit_d_pac": float(benefit["d_pac"]),
            }
        )
    if verified_candidates != 117 or len(changes) != 116:
        raise ValueError("Expected 117 verified plans and 116 one-layer flips")
    benefit_func = np.asarray([row["benefit_d_func"] for row in changes])
    benefit_pac = np.asarray([row["benefit_d_pac"] for row in changes])
    if np.any(benefit_func > 0.0) or np.any(benefit_pac > 0.0):
        raise ValueError("Unexpected positive conservative one-layer benefit")

    summaries = selection["selection"]["summaries"]
    structured = [
        {
            "id": key,
            "label": {
                "attention_6": "Attention-6",
                "ff_pair_15": "FF pair-15",
                "mlp_6": "MLP-6",
                "single_best": "Best single",
                "two_best": "Best pair",
            }[key],
            "objective": float(summaries[key]["objective"]),
            "component_pass": bool(summaries[key]["component_constraints_pass"]),
            "eligible": bool(summaries[key]["eligible"]),
        }
        for key in ("attention_6", "ff_pair_15", "mlp_6", "single_best", "two_best")
    ]
    if any(row["eligible"] or row["component_pass"] for row in structured):
        raise ValueError("A structured proposal unexpectedly passes selection")

    action_names = set(action_layer_order(plan["layers"]))
    counts = {
        "Action head": {"W4": 0, "FP16": 0},
        "Backbone": {"W4": 0, "FP16": 0},
    }
    for name, layer in plan["layers"].items():
        subsystem = "Action head" if name in action_names else "Backbone"
        counts[subsystem][precision(layer)] += 1
    if counts != {
        "Action head": {"W4": 31, "FP16": 1},
        "Backbone": {"W4": 69, "FP16": 15},
    }:
        raise ValueError(f"Unexpected mask allocation: {counts}")
    return {
        "changes": changes,
        "structured": structured,
        "counts": counts,
        "verified_candidates": verified_candidates,
    }


def render_fcp_figure(output_dir: Path, evidence: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    fig = plt.figure(figsize=(7.25, 2.82), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.22, 1.02, 0.80),
        left=0.075,
        right=0.985,
        bottom=0.245,
        top=0.875,
        wspace=0.42,
    )
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, index]) for index in range(3)]

    direction_style = {
        "W4→FP16": (COLORS["ours"], "o"),
        "FP16→W4": (COLORS["static"], "^"),
    }
    for direction, (color, marker) in direction_style.items():
        selected = [row for row in evidence["changes"] if row["direction"] == direction]
        ax_a.scatter(
            [row["benefit_d_func"] for row in selected],
            [row["benefit_d_pac"] for row in selected],
            s=17,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.83,
            label=f"{direction} (n={len(selected)})",
            zorder=3,
        )
    ax_a.axhline(0, color="#87949C", linewidth=0.75)
    ax_a.axvline(0, color="#87949C", linewidth=0.75)
    ax_a.set_xlim(-4.35, 0.18)
    ax_a.set_ylim(-2.18, 0.10)
    ax_a.set_title("a   All one-layer flips rejected", loc="left", fontweight="bold", pad=6)
    ax_a.set_xlabel(r"Conservative FP16 benefit: $D_{\rm func}$")
    ax_a.set_ylabel(r"Conservative FP16 benefit: $D_{\rm PAC}$")
    ax_a.legend(frameon=False, loc="lower left")
    ax_a.text(
        0.98,
        0.95,
        "benefit > 0 required\n0 / 116 positive on either axis",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color=COLORS["muted"],
    )
    clean_axis(ax_a, "both")

    objectives = [row["objective"] for row in evidence["structured"]]
    y = np.arange(len(objectives))
    bars = ax_b.barh(
        y,
        objectives,
        color=["#9AA7AE", "#71858F", "#8A9AA2", "#536B78", "#657C87"],
        height=0.58,
        zorder=3,
    )
    ax_b.axvspan(-0.45, 0.0, color=COLORS["dynamic"], alpha=0.08, zorder=0)
    ax_b.axvline(0.0, color=COLORS["dynamic"], linewidth=1.2)
    ax_b.set_yticks(y, labels=[row["label"] for row in evidence["structured"]])
    ax_b.invert_yaxis()
    ax_b.set_xlim(-0.45, 7.65)
    ax_b.set_xlabel(r"Paired minimax objective $J$")
    ax_b.set_title("b   Structured proposals rejected", loc="left", fontweight="bold", pad=6)
    for bar, value in zip(bars, objectives):
        ax_b.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color=COLORS["ink"],
        )
    ax_b.text(
        0.03,
        0.97,
        "$J<0$ and all component guards required",
        transform=ax_b.transAxes,
        va="top",
        fontsize=5.9,
        color=COLORS["muted"],
    )
    clean_axis(ax_b, "x")

    subsystems = ("Action head", "Backbone")
    w4 = np.asarray([evidence["counts"][name]["W4"] for name in subsystems])
    fp16 = np.asarray([evidence["counts"][name]["FP16"] for name in subsystems])
    y = np.arange(2)
    ax_c.barh(y, w4, color=COLORS["ours"], height=0.54, label="W4", zorder=3)
    ax_c.barh(
        y,
        fp16,
        left=w4,
        color=COLORS["static"],
        height=0.54,
        label="FP16",
        zorder=3,
    )
    for index, (w4_count, fp16_count) in enumerate(zip(w4, fp16)):
        ax_c.text(
            w4_count / 2,
            index,
            str(w4_count),
            ha="center",
            va="center",
            color="white",
            fontsize=6.6,
            fontweight="bold",
        )
        ax_c.text(
            w4_count + fp16_count / 2,
            index,
            str(fp16_count),
            ha="center",
            va="center",
            color="white",
            fontsize=6.4,
            fontweight="bold",
        )
    ax_c.set_yticks(y, labels=subsystems)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Target Linear layers")
    ax_c.set_title("c   Audit abstains; retain $M_0$", loc="left", fontweight="bold", pad=6)
    ax_c.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.7)
    clean_axis(ax_c, "x")

    func = np.asarray([row["benefit_d_func"] for row in evidence["changes"]])
    pac = np.asarray([row["benefit_d_pac"] for row in evidence["changes"]])
    derived = {
        "verified_candidate_plans": evidence["verified_candidates"],
        "one_layer_flips": len(evidence["changes"]),
        "positive_d_func_benefits": int(np.count_nonzero(func > 0.0)),
        "positive_d_pac_benefits": int(np.count_nonzero(pac > 0.0)),
        "benefit_d_func_range": [float(func.min()), float(func.max())],
        "benefit_d_pac_range": [float(pac.min()), float(pac.max())],
        "structured_objectives": {
            row["id"]: row["objective"] for row in evidence["structured"]
        },
        "mask_allocation": evidence["counts"],
    }
    return (*save_figure(fig, output_dir / "dypac_fcp_audit"), derived)


def point_audit(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "size_gib": row["size"],
            "task_macro_success_rate": row["success"],
        }
        for row in points
    ]


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / OUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: repo_root / relative for name, relative in SOURCES.items()}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source {name}: {path}")

    configure_style()
    operating = load_operating_evidence(paths)
    operating_png, operating_pdf, operating_derived = render_operating_figure(
        output_dir, operating
    )
    dynamic = load_dynamic_evidence(paths)
    dynamic_png, dynamic_pdf, dynamic_derived = render_dynamic_figure(
        output_dir, dynamic
    )
    fcp = load_fcp_evidence(paths)
    fcp_png, fcp_pdf, fcp_derived = render_fcp_figure(output_dir, fcp)

    outputs = [
        operating_png,
        operating_pdf,
        dynamic_png,
        dynamic_pdf,
        fcp_png,
        fcp_pdf,
    ]
    audit = {
        "schema_version": 1,
        "kind": "dypac_evidence_figures",
        "claim_evidence_boundaries": {
            "operating_points": (
                "formal closed-loop success and exact static component storage; "
                "pi0.5 cross-row differences remain descriptive"
            ),
            "dynamic_range": (
                "scale drift is descriptive; the 500-episode matched rollout is "
                "the closed-loop intervention"
            ),
            "fcp": (
                "offline acceptance/rejection evidence within the declared "
                "counterfactual neighborhood; not a success gain or global optimum"
            ),
        },
        "formulas": {
            "flow_drift": "||s_layer,t - s_layer,1||_2 / ||s_layer,1||_2",
            "scale_aggregation": (
                "channelwise median across atomic-seen, composite-seen, and "
                "composite-unseen frozen-A8 tables"
            ),
            "task_delta": "task-level SR(DyPAC) - task-level SR(baseline)",
            "pareto": "non-dominated under smaller static size and larger success",
            "fcp_benefit": (
                "production conservative_fp16_benefit with task-cluster "
                "mean±jackknife-SE bounds"
            ),
        },
        "derived": {
            "operating": {
                **operating_derived,
                "gr_points": point_audit(operating["gr_points"]),
                "pi05_points": point_audit(operating["pi_points"]),
                "task_direction_counts": operating["direction_counts"],
                "gr_paired_comparisons": operating["gr_paired"],
            },
            "dynamic_range": dynamic_derived,
            "fcp": fcp_derived,
        },
        "sources": [
            {
                "name": name,
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256(path),
            }
            for name, path in sorted(paths.items())
        ],
        "outputs": [
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    audit_path = output_dir / "dypac_evidence_figures.audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    for path in outputs:
        print(f"wrote {path}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
