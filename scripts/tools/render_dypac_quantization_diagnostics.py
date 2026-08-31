#!/usr/bin/env python3
"""Render an audited DyPAC quantization-diagnostics figure.

Only frozen experiment artifacts are read: split-specific frozen-A8 tables,
full-context one-layer counterfactuals, the deployment plan, and the paired
component rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr, spearmanr


STATIC_TABLES = (
    "runs/full_context_v2/table3_quick/static_a8/atomic_seen/m0_static_a8.npz",
    "runs/full_context_v2/table3_quick/static_a8/composite_seen/m0_static_a8.npz",
    "runs/full_context_v2/table3_quick/static_a8/composite_unseen/m0_static_a8.npz",
)
FLIP_SCORES = "runs/full_context_v2/p2/flip_scores/flip_scores_cross_split.json"
FROZEN_PLAN = "runs/full_context_v2/p2/gr00t_full_context_v2_frozen.json"
ROLLOUT = "runs/full_context_v2/table3_quick/aggregate.json"
OUT_STEM = "docs/gdsq_vla_iclr2027/figures/dypac_quantization_diagnostics"
ACTION_RE = re.compile(
    r"^action_head\.model\.transformer_blocks\.(\d+)\.ff\.net\.(0\.proj|2)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-stem", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_recorded_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    marker = "/runs/"
    if marker in value:
        return repo_root / ("runs/" + value.split(marker, 1)[1])
    raise FileNotFoundError(value)


def precision(layer: dict) -> str:
    return "FP16" if layer.get("skip") else f"W{int(layer['bits'])}"


def action_layer_order(layers: dict[str, dict]) -> list[str]:
    parsed: list[tuple[int, int, str]] = []
    for name in layers:
        match = ACTION_RE.match(name)
        if match:
            role = 0 if match.group(2) == "0.proj" else 1
            parsed.append((int(match.group(1)), role, name))
    parsed.sort()
    names = [item[2] for item in parsed]
    if len(names) != 32:
        raise ValueError(f"Expected 32 action-head FFN targets, found {len(names)}")
    return names


def load_flow_drift(
    repo_root: Path, action_names: list[str], layers: dict[str, dict]
) -> tuple[np.ndarray, dict[str, np.ndarray], list[Path]]:
    paths = [repo_root / item for item in STATIC_TABLES]
    archives = [np.load(path) for path in paths]
    drift_by_layer: dict[str, np.ndarray] = {}
    for name in action_names:
        if precision(layers[name]) == "FP16":
            continue
        tables = [archive[name] for archive in archives]
        if any(table.ndim != 2 or table.shape[0] != 4 for table in tables):
            raise ValueError(f"Expected four flow-step rows for {name}")
        if len({table.shape for table in tables}) != 1:
            raise ValueError(f"Mismatched scale shapes for {name}")
        scale = np.median(np.stack(tables, axis=0), axis=0).astype(np.float64)
        denominator = max(float(np.linalg.norm(scale[0])), 1.0e-12)
        drift_by_layer[name] = np.linalg.norm(scale - scale[0], axis=1) / denominator

    heatmap = np.full((len(action_names), 4), np.nan, dtype=np.float64)
    for row, name in enumerate(action_names):
        if name in drift_by_layer:
            heatmap[row] = drift_by_layer[name]
    return heatmap, drift_by_layer, paths


def load_flip_changes(repo_root: Path, score_path: Path) -> tuple[list[dict], dict]:
    payload = load_json(score_path)
    scores = payload["scores"]
    baseline = scores["context_base"]
    plan_records = payload["candidate_plans"]
    checked_plans: list[tuple[str, str]] = []

    def checked_plan(candidate_id: str) -> dict:
        record = plan_records[candidate_id]
        path = resolve_recorded_path(repo_root, record["path"])
        observed_sha = sha256(path)
        if observed_sha != record["sha256"]:
            raise ValueError(f"Candidate-plan hash mismatch: {candidate_id}")
        checked_plans.append((candidate_id, observed_sha))
        return load_json(path)

    base_path = resolve_recorded_path(repo_root, plan_records["context_base"]["path"])
    base_layers = checked_plan("context_base")["layers"]
    changes: list[dict] = []
    for candidate_id, score in scores.items():
        if candidate_id == "context_base":
            continue
        candidate_layers = checked_plan(candidate_id)["layers"]
        changed = [
            name for name in base_layers if base_layers[name] != candidate_layers[name]
        ]
        if len(changed) != 1:
            raise ValueError(f"{candidate_id} changes {len(changed)} layers")
        name = changed[0]
        changes.append(
            {
                "candidate_id": candidate_id,
                "layer": name,
                "direction": (
                    f"{precision(base_layers[name])}→{precision(candidate_layers[name])}"
                ),
                "delta_d_func": float(baseline["d_func"] - score["d_func"]),
                "delta_d_pac": float(baseline["d_pac"] - score["d_pac"]),
            }
        )
    if len(changes) != 116:
        raise ValueError(f"Expected 116 one-layer flips, found {len(changes)}")
    if len(checked_plans) != 117:
        raise ValueError(f"Expected 117 verified candidate plans, found {len(checked_plans)}")
    plan_digest = hashlib.sha256()
    for candidate_id, digest in sorted(checked_plans):
        plan_digest.update(candidate_id.encode("utf-8") + b"\0")
        plan_digest.update(digest.encode("ascii") + b"\n")
    return changes, {
        "base_path": str(base_path.resolve().relative_to(repo_root)),
        "count": len(checked_plans),
        "all_recorded_sha256_verified": True,
        "candidate_id_sha256_digest": plan_digest.hexdigest(),
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.3,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem is not None
        else repo_root / OUT_STEM
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    plan_path = repo_root / FROZEN_PLAN
    score_path = repo_root / FLIP_SCORES
    rollout_path = repo_root / ROLLOUT
    plan = load_json(plan_path)
    layers = plan["layers"]
    action_names = action_layer_order(layers)
    heatmap, drift_by_layer, scale_paths = load_flow_drift(
        repo_root, action_names, layers
    )
    changes, candidate_plan_audit = load_flip_changes(repo_root, score_path)
    rollout = load_json(rollout_path)

    if plan["quantized_w4_layers"] != 100 or plan["retained_fp16_layers"] != 16:
        raise ValueError("Frozen plan is not the expected 100-W4/16-FP16 mask")
    static_arm = rollout["arms"]["static_a8"]
    baseline = rollout["baseline"]
    if static_arm["episodes"] != 500 or baseline["episodes"] != 500:
        raise ValueError("Component-rollout coverage is not 500 episodes per arm")

    roles = {
        "FFN expand": [
            drift_by_layer[name]
            for name in action_names
            if name in drift_by_layer and name.endswith("net.0.proj")
        ],
        "FFN contract": [
            drift_by_layer[name]
            for name in action_names
            if name in drift_by_layer and name.endswith("net.2")
        ],
    }
    role_arrays = {name: np.stack(values) for name, values in roles.items()}
    delta_func = np.asarray([item["delta_d_func"] for item in changes])
    delta_pac = np.asarray([item["delta_d_pac"] for item in changes])
    pearson = pearsonr(delta_func, delta_pac)
    spearman = spearmanr(delta_func, delta_pac)
    sign_mismatch = int(
        np.count_nonzero(np.signbit(delta_func) != np.signbit(delta_pac))
    )

    configure_style()
    blue, orange, teal, dark, grid = (
        "#2F6B9A",
        "#D36B3D",
        "#267A78",
        "#243746",
        "#D9E0E5",
    )
    fig = plt.figure(figsize=(7.25, 4.35), facecolor="white")
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.04, 1.20, 0.92),
        left=0.070,
        right=0.985,
        bottom=0.170,
        top=0.815,
        wspace=0.44,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    right_grid = outer[0, 2].subgridspec(1, 2, width_ratios=(1.0, 0.13), wspace=0.10)
    ax_c = fig.add_subplot(right_grid[0, 0])
    ax_precision = fig.add_subplot(right_grid[0, 1])

    fig.text(
        0.070,
        0.940,
        "FROZEN, AUDITED DIAGNOSTICS",
        color=teal,
        fontsize=7.0,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.070,
        0.895,
        "GR00T  ·  116 target Linears  ·  4 flow steps  ·  group-64 W4  ·  final mask 100 W4 / 16 FP16",
        color=dark,
        fontsize=7.4,
        va="center",
    )
    fig.text(
        0.985,
        0.940,
        f"paired rollout: frozen A8  {static_arm['successes']}/500   →   dynamic A8  {baseline['successes']}/500",
        color=dark,
        fontsize=7.0,
        fontweight="bold",
        va="center",
        ha="right",
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": "#F2F7F6",
            "ec": "#B9D2CF",
            "lw": 0.6,
        },
    )

    x_step = np.arange(1, 5)
    for label, color in (("FFN expand", blue), ("FFN contract", orange)):
        values = role_arrays[label]
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        ax_a.fill_between(x_step, q25, q75, color=color, alpha=0.16, linewidth=0)
        ax_a.plot(x_step, median, color=color, marker="o", ms=3.8, label=label)
        ax_a.annotate(
            f"{median[-1]:.1%}",
            (4, median[-1]),
            xytext=(-3, 5 if label == "FFN contract" else -11),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=6.6,
            fontweight="bold",
        )
    ax_a.set_title("a   Flow-step scale drift", loc="left", fontweight="bold", pad=7)
    ax_a.set_xlabel("Flow step")
    ax_a.set_ylabel(r"$\|s_t-s_1\|_2/\|s_1\|_2$")
    ax_a.set_xticks(x_step)
    ax_a.set_ylim(-0.01, 0.50)
    ax_a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_a.grid(axis="y", color=grid, linewidth=0.55)
    ax_a.legend(frameon=False, loc="upper left", handlelength=1.8)
    ax_a.text(
        0.02,
        0.03,
        "line: median  ·  ribbon: IQR",
        transform=ax_a.transAxes,
        color="#65727C",
        fontsize=6.2,
    )

    direction_style = {"W4→FP16": (blue, "o"), "FP16→W4": (orange, "^")}
    for direction, (color, marker) in direction_style.items():
        selected = [item for item in changes if item["direction"] == direction]
        ax_b.scatter(
            [item["delta_d_func"] for item in selected],
            [item["delta_d_pac"] for item in selected],
            s=18,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            label=f"{direction}  (n={len(selected)})",
            zorder=3,
        )
    fit = np.polyfit(delta_func, delta_pac, 1)
    fit_x = np.linspace(float(delta_func.min()), float(delta_func.max()), 200)
    ax_b.plot(fit_x, np.polyval(fit, fit_x), color=dark, lw=1.1, ls="--", zorder=2)
    ax_b.axhline(0, color="#8A969E", lw=0.7, zorder=1)
    ax_b.axvline(0, color="#8A969E", lw=0.7, zorder=1)
    ax_b.set_title("b   FCP metric agreement", loc="left", fontweight="bold", pad=7)
    ax_b.set_xlabel(r"$\Delta D_{\mathrm{func}}$  (base $-$ flip)")
    ax_b.set_ylabel(r"$\Delta D_{\mathrm{PAC}}$  (base $-$ flip)")
    ax_b.grid(color=grid, linewidth=0.45, alpha=0.7)
    ax_b.legend(frameon=False, loc="lower right", handletextpad=0.4)
    ax_b.text(
        0.03,
        0.965,
        f"Pearson $r$={pearson.statistic:.3f}\n"
        f"Spearman $\\rho$={spearman.statistic:.3f}\n"
        f"sign mismatch: {sign_mismatch}/116",
        transform=ax_b.transAxes,
        va="top",
        color=dark,
        fontsize=6.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": grid,
            "alpha": 0.92,
        },
    )

    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#D9DDE0")
    image = ax_c.imshow(
        np.ma.masked_invalid(heatmap),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=0.50,
    )
    ax_c.set_title("c   Layer × step", loc="left", fontweight="bold", pad=7)
    ax_c.set_xlabel("Flow step")
    ax_c.set_xticks(np.arange(4), labels=("1", "2", "3", "4"))
    ax_c.set_yticks(np.arange(0.5, 32, 2), labels=[f"B{i}" for i in range(16)])
    ax_c.set_ylabel("Transformer block")
    ax_c.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, 32, 2), minor=True)
    ax_c.grid(which="minor", color="white", linewidth=0.38, alpha=0.75)
    ax_c.tick_params(which="minor", bottom=False, left=False)
    precision_values = np.asarray(
        [[0 if precision(layers[name]) == "W4" else 1] for name in action_names]
    )
    ax_precision.imshow(
        precision_values,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap((blue, orange)),
        vmin=0,
        vmax=1,
    )
    ax_precision.set_title("bits", fontsize=6.4, pad=9)
    ax_precision.set_xticks([])
    ax_precision.set_yticks([])
    for spine in ax_precision.spines.values():
        spine.set_visible(False)
    ax_precision.text(
        0.5,
        -0.060,
        "W4",
        transform=ax_precision.transAxes,
        ha="center",
        fontsize=5.8,
        color=blue,
    )
    ax_precision.text(
        0.5,
        -0.115,
        "FP16",
        transform=ax_precision.transAxes,
        ha="center",
        fontsize=5.8,
        color=orange,
    )
    colorbar = fig.colorbar(
        image,
        ax=(ax_c, ax_precision),
        orientation="horizontal",
        fraction=0.055,
        pad=0.18,
        aspect=24,
    )
    colorbar.set_ticks((0.0, 0.25, 0.50), labels=("0%", "25%", "50%"))
    colorbar.ax.tick_params(labelsize=5.9, length=2)
    colorbar.set_label("relative L2 scale drift", fontsize=6.2, labelpad=1)
    colorbar.outline.set_linewidth(0.45)
    for axis in (ax_a, ax_b, ax_c):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(
        png_path,
        dpi=320,
        facecolor="white",
        metadata={"Software": "render_dypac_quantization_diagnostics.py"},
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Creator": "render_dypac_quantization_diagnostics.py",
            "CreationDate": None,
        },
    )
    plt.close(fig)

    valid_drift = heatmap[np.isfinite(heatmap)]
    max_index = np.nanargmax(heatmap)
    max_row, max_step = np.unravel_index(max_index, heatmap.shape)
    source_paths = [*scale_paths, score_path, plan_path, rollout_path]
    audit = {
        "schema_version": 1,
        "kind": "dypac_quantization_diagnostics",
        "formula": {
            "flow_drift": "||s_layer,t - s_layer,1||_2 / ||s_layer,1||_2",
            "scale_aggregation": "channelwise median across three split-specific frozen-A8 tables",
            "flip_delta": "D(context_base) - D(one_layer_flip); positive favors the flip",
        },
        "protocol": {
            "target_linears": len(layers),
            "flow_steps": 4,
            "weight_group_size": 64,
            "w4_layers": plan["quantized_w4_layers"],
            "fp16_layers": plan["retained_fp16_layers"],
            "action_head_targets": len(action_names),
            "action_head_w4_with_scale_tables": len(drift_by_layer),
            "component_rollout_episodes_per_arm": baseline["episodes"],
            "verified_candidate_plans": candidate_plan_audit,
        },
        "derived": {
            "all_w4_action_head_median_drift_by_step": [
                float(value) for value in np.nanmedian(heatmap, axis=0)
            ],
            "all_w4_action_head_max_drift": float(valid_drift.max()),
            "max_drift_layer": action_names[max_row],
            "max_drift_flow_step": int(max_step + 1),
            "ffn_expand_median_drift_by_step": [
                float(value)
                for value in np.median(role_arrays["FFN expand"], axis=0)
            ],
            "ffn_contract_median_drift_by_step": [
                float(value)
                for value in np.median(role_arrays["FFN contract"], axis=0)
            ],
            "one_layer_flips": len(changes),
            "w4_to_fp16_flips": sum(
                item["direction"] == "W4→FP16" for item in changes
            ),
            "fp16_to_w4_flips": sum(
                item["direction"] == "FP16→W4" for item in changes
            ),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "metric_sign_mismatches": sign_mismatch,
            "frozen_a8_successes": static_arm["successes"],
            "dynamic_a8_successes": baseline["successes"],
        },
        "sources": [
            {
                "path": str(path.resolve().relative_to(repo_root)),
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
        "outputs": {
            "png": {
                "path": str(png_path.relative_to(repo_root)),
                "sha256": sha256(png_path),
            },
            "pdf": {
                "path": str(pdf_path.relative_to(repo_root)),
                "sha256": sha256(pdf_path),
            },
        },
    }
    audit_path = output_stem.with_suffix(".audit.json")
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
