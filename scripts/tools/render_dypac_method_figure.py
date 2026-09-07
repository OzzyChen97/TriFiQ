#!/usr/bin/env python3
"""Render the paper's method overview as an editable Matplotlib figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


NAVY = "#17365D"
BLUE = "#2C7FB8"
TEAL = "#087F74"
ORANGE = "#D95F02"
INK = "#243746"
MUTED = "#667780"
LINE = "#B7C5CF"
SOFT_BLUE = "#F2F7FB"
SOFT_TEAL = "#F0F8F6"
SOFT_ORANGE = "#FFF5EC"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/gdsq_vla_iclr2027/figures"),
    )
    return parser.parse_args()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def rounded(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = WHITE,
    ec: str = NAVY,
    lw: float = 1.0,
    fontsize: float = 7.2,
    weight: str = "normal",
    color: str = INK,
    radius: float = 0.012,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.12,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    lw: float = 1.2,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def panel(
    ax: mpl.axes.Axes,
    x: float,
    width: float,
    number: str,
    title: str,
    subtitle: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.055),
            width,
            0.89,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            facecolor=WHITE,
            edgecolor=NAVY,
            linewidth=1.15,
            zorder=0,
        )
    )
    ax.text(
        x + 0.018,
        0.900,
        number,
        ha="left",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color=WHITE,
        bbox={"boxstyle": "round,pad=0.24", "fc": NAVY, "ec": NAVY},
    )
    ax.text(
        x + 0.058,
        0.902,
        title,
        ha="left",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + width - 0.018,
        0.902,
        subtitle,
        ha="right",
        va="center",
        fontsize=6.6,
        color=MUTED,
    )
    ax.plot(
        [x + 0.018, x + width - 0.018],
        [0.858, 0.858],
        color=LINE,
        linewidth=0.75,
        zorder=1,
    )


def draw_evidence(ax: mpl.axes.Axes, x: float, w: float) -> None:
    box_x = x + 0.065
    box_w = w - 0.130
    stages = (
        (0.705, "Fixed teacher\ncalibration data", SOFT_BLUE, BLUE),
        (0.535, "Local tensor\nreconstruction score", WHITE, MUTED),
        (0.365, "Forced\nprecision mask", SOFT_ORANGE, ORANGE),
        (0.195, "Frozen activation\nlookup table", "#F7F8F9", MUTED),
    )
    for index, (y, label, face, edge) in enumerate(stages):
        rounded(
            ax,
            box_x,
            y,
            box_w,
            0.090,
            label,
            fc=face,
            ec=edge,
            fontsize=7.0,
            weight="bold",
            color=edge if edge != MUTED else INK,
        )
        if index:
            previous_y = stages[index - 1][0]
            arrow(
                ax,
                (x + w / 2, previous_y),
                (x + w / 2, y + 0.090),
                color=NAVY,
            )
    ax.text(
        x + w / 2,
        0.115,
        "Assumption: deployment inputs follow calibration data",
        ha="center",
        va="center",
        fontsize=6.3,
        color=MUTED,
    )


def draw_audit(ax: mpl.axes.Axes, x: float, w: float) -> None:
    center_x = x + w / 2
    nodes = {
        "quant": (center_x - 0.068, 0.695, "Quantization\nerror", SOFT_ORANGE, ORANGE),
        "action": (x + w - 0.150, 0.505, "Action\ndrift", WHITE, NAVY),
        "state": (center_x - 0.068, 0.300, "Future-state\nshift", SOFT_BLUE, BLUE),
        "range": (x + 0.018, 0.505, "Activation-\nrange shift", SOFT_TEAL, TEAL),
    }
    for node_x, node_y, label, face, edge in nodes.values():
        rounded(
            ax,
            node_x,
            node_y,
            0.136,
            0.090,
            label,
            fc=face,
            ec=edge,
            fontsize=7.0,
            weight="bold",
            color=edge,
        )
    arrow(
        ax,
        (center_x + 0.068, 0.740),
        (x + w - 0.150, 0.578),
        color=ORANGE,
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        ax,
        (x + w - 0.082, 0.505),
        (center_x + 0.050, 0.390),
        color=NAVY,
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        ax,
        (center_x - 0.068, 0.345),
        (x + 0.154, 0.535),
        color=BLUE,
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        ax,
        (x + 0.086, 0.595),
        (center_x - 0.050, 0.695),
        color=TEAL,
        connectionstyle="arc3,rad=-0.12",
    )
    ax.text(
        center_x,
        0.635,
        "policy-induced data",
        ha="center",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=MUTED,
    )
    ax.text(
        center_x,
        0.155,
        "The policy changes the data used to evaluate it",
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=NAVY,
    )


def draw_deployment(ax: mpl.axes.Axes, x: float, w: float) -> None:
    rows = (
        (
            0.680,
            "Precision allocation",
            "Action-weighted CS–CKA\nhard guards • exact bytes",
            SOFT_BLUE,
            BLUE,
        ),
        (
            0.445,
            "Full-context verification",
            "Physical sequence + complete policy\npropose locally • may abstain",
            SOFT_ORANGE,
            ORANGE,
        ),
        (
            0.210,
            "Deployment adaptation",
            "Input-conditioned channelwise A8\nno frozen range lookup",
            SOFT_TEAL,
            TEAL,
        ),
    )
    for y, heading, body, face, edge in rows:
        ax.text(
            x + 0.025,
            y + 0.120,
            heading,
            ha="left",
            va="center",
            fontsize=6.6,
            fontweight="bold",
            color=edge,
        )
        rounded(
            ax,
            x + 0.025,
            y,
            w - 0.050,
            0.095,
            body,
            fc=face,
            ec=edge,
            fontsize=6.8,
            weight="bold",
            color=INK,
        )
    ax.text(
        x + w / 2,
        0.105,
        "allocate under budget × audit in context × adapt at deployment",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=NAVY,
    )


def render(output_dir: Path) -> tuple[Path, Path]:
    configure()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.2, 4.7), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    specs = (
        (0.012, 0.305, "A", "Conventional PTQ", "exogenous data"),
        (0.342, 0.316, "B", "Quantization–control loop", "endogenous data"),
        (0.683, 0.305, "C", "Full-context response", "policy-level"),
    )
    for x, width, number, title, subtitle in specs:
        panel(ax, x, width, number, title, subtitle)
    draw_evidence(ax, specs[0][0], specs[0][1])
    draw_audit(ax, specs[1][0], specs[1][1])
    draw_deployment(ax, specs[2][0], specs[2][1])

    arrow(ax, (0.319, 0.50), (0.338, 0.50), color=NAVY, lw=1.8)
    arrow(ax, (0.660, 0.50), (0.679, 0.50), color=NAVY, lw=1.8)

    png_path = output_dir / "dypac_main_architecture.png"
    pdf_path = output_dir / "dypac_main_architecture.pdf"
    fig.savefig(png_path, dpi=260, facecolor=WHITE)
    fig.savefig(
        pdf_path,
        facecolor=WHITE,
        metadata={"Creator": Path(__file__).name, "CreationDate": None},
    )
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    png_path, pdf_path = render(args.output_dir)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
