#!/usr/bin/env python3
"""Render an alternative single-column tolerance-aware Pareto figure.

This candidate follows the legacy figure's strongest idea while treating
speedups within 1% as practically equivalent. Among such near-equal-cost
points, the higher-quality operating point dominates. This prevents tiny
timing differences from putting a substantially worse method on the frontier.
It uses current InTraScale terminology and a speedup axis that remains readable
at AAAI single-column width.

The script deliberately writes a separate candidate and does not replace the
figure referenced by rewrite/main.tex.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
REWRITE_DIR = SCRIPT_DIR.parents[1]
AAAI27_DIR = SCRIPT_DIR.parents[2]
DATA_PATH = AAAI27_DIR / "results" / "warm_quality_efficiency_20260722.csv"
PDF_PATH = REWRITE_DIR / "figures" / "fig_quality_efficiency_pareto_v2.pdf"
PNG_PATH = REWRITE_DIR / "figures" / "fig_quality_efficiency_pareto_v2.png"
SPEEDUP_TOLERANCE = 0.01


@dataclass(frozen=True)
class Point:
    case: str
    label: str
    family: str
    latency: float
    latency_std: float
    quality: float
    speedup: float
    speedup_std: float


CASE_SPEC = {
    "full_hr50": ("Native-HR", "native"),
    "lightx2v_cr40": ("Trilinear@40", "fixed"),
    "lightx2v_cr45": ("Trilinear@45", "fixed"),
    "lightx2v_cr48": ("Trilinear@48", "fixed"),
    "ralu_nt45": ("RALU@45", "ralu"),
    "talh40": ("InTraScale@40", "intrascale"),
    "talh45": ("InTraScale@45", "intrascale"),
    "full_lr50_stage2_5hr": ("ITU-5HR (LUVE-style)", "endpoint"),
}

COLORS = {
    "native": "#4B5563",
    "fixed": "#56B4E9",
    "ralu": "#CC79A7",
    "endpoint": "#009E73",
    "intrascale": "#D55E00",
    "pareto": "#222222",
    "grid": "#D9D9D9",
    "text": "#242424",
}

MARKERS = {
    "native": "s",
    "fixed": "o",
    "ralu": "X",
    "endpoint": "D",
    "intrascale": "o",
}


def load_points() -> dict[str, Point]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing canonical data source: {DATA_PATH}")
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["case"]: row for row in csv.DictReader(handle)}

    missing = sorted(set(CASE_SPEC) - set(rows))
    if missing:
        raise RuntimeError(f"Missing required cases in {DATA_PATH.name}: {missing}")

    native_latency = float(rows["full_hr50"]["pipeline_mean_s"])
    native_latency_std = float(rows["full_hr50"]["pipeline_std_s"])
    points: dict[str, Point] = {}
    for case, (label, family) in CASE_SPEC.items():
        row = rows[case]
        latency = float(row["pipeline_mean_s"])
        latency_std = float(row["pipeline_std_s"])
        quality = float(row["quality_value"])
        values = (latency, latency_std, quality)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{case}: non-finite value in {values}")
        if latency <= 0 or latency_std < 0:
            raise RuntimeError(f"{case}: invalid latency summary {values[:2]}")
        points[case] = Point(
            case=case,
            label=label,
            family=family,
            latency=latency,
            latency_std=latency_std,
            quality=quality,
            speedup=native_latency / latency,
            speedup_std=(
                0.0
                if case == "full_hr50"
                else (native_latency / latency)
                * math.sqrt(
                    (native_latency_std / native_latency) ** 2
                    + (latency_std / latency) ** 2
                )
            ),
        )
    return points


def epsilon_pareto(points: dict[str, Point]) -> list[Point]:
    """Return non-dominated points with a relative speedup tolerance.

    A higher-quality point may dominate a lower-quality point when its speedup
    is no more than SPEEDUP_TOLERANCE slower. This treats sub-percent timing
    differences as practically tied rather than as meaningful improvements.
    """
    frontier: list[Point] = []
    values = list(points.values())
    for point in values:
        dominated = any(
            other.case != point.case
            and other.speedup >= point.speedup * (1.0 - SPEEDUP_TOLERANCE)
            and other.quality >= point.quality
            and (
                other.quality > point.quality
                or other.speedup > point.speedup
            )
            for other in values
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: point.speedup)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.72,
            "axes.edgecolor": "#333333",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )


def draw(points: dict[str, Point]) -> list[Point]:
    configure_style()
    frontier = epsilon_pareto(points)
    frontier_cases = {point.case for point in frontier}

    fig, ax = plt.subplots(figsize=(3.30, 2.05))
    fig.subplots_adjust(left=0.145, right=0.955, bottom=0.245, top=0.855)

    family_paths = {
        "fixed": ["lightx2v_cr40", "lightx2v_cr45", "lightx2v_cr48"],
        "intrascale": ["talh40", "talh45"],
    }
    for family, cases in family_paths.items():
        family_points = [points[case] for case in cases]
        ax.plot(
            [point.speedup for point in family_points],
            [point.quality for point in family_points],
            color=COLORS[family],
            linewidth=0.9 if family != "intrascale" else 1.15,
            linestyle=(0, (2.2, 2.0)) if family == "fixed" else "-",
            alpha=0.58 if family != "intrascale" else 0.75,
            zorder=1,
        )

    ax.plot(
        [point.speedup for point in frontier],
        [point.quality for point in frontier],
        color=COLORS["pareto"],
        linewidth=1.25,
        linestyle=(0, (3.0, 2.0)),
        zorder=2,
    )

    for point in points.values():
        on_frontier = point.case in frontier_cases
        ax.errorbar(
            point.speedup,
            point.quality,
            xerr=point.speedup_std if point.speedup_std > 0 else None,
            fmt=MARKERS[point.family],
            markersize=6.5 if point.family == "intrascale" else 5.7,
            color=COLORS[point.family],
            markerfacecolor=COLORS[point.family],
            markeredgecolor=COLORS["pareto"] if on_frontier else "white",
            markeredgewidth=0.85 if on_frontier else 0.65,
            ecolor=COLORS[point.family],
            elinewidth=0.55,
            capsize=1.35,
            capthick=0.55,
            zorder=4,
        )

    annotations = {
        "full_hr50": ("Native-HR", (-2, -10), "left"),
        "talh40": ("InTraScale@40", (0, 12), "center"),
        "talh45": ("InTraScale@45", (4, 10), "left"),
        "lightx2v_cr40": ("Trilinear@40", (0, 6), "center"),
        "lightx2v_cr45": ("Trilinear@45", (0, 6), "center"),
        "lightx2v_cr48": ("Trilinear@48", (8, 7), "right"),
    }
    for case, (text, offset, alignment) in annotations.items():
        point = points[case]
        is_ours = point.family == "intrascale"
        ax.annotate(
            text,
            xy=(point.speedup, point.quality),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=6.25 if not is_ours else 6.55,
            fontweight="bold" if is_ours else "normal",
            color=COLORS[point.family] if point.family != "native" else COLORS["text"],
            zorder=5,
        )

    endpoint = points["full_lr50_stage2_5hr"]
    ax.annotate(
        "ITU-5HR (LUVE-style)",
        xy=(endpoint.speedup, endpoint.quality),
        xytext=(3.15, 0.793),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=6.15,
        color=COLORS["endpoint"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["endpoint"],
            "linewidth": 0.55,
            "shrinkA": 2.0,
            "shrinkB": 4.0,
        },
        zorder=5,
    )

    ralu = points["ralu_nt45"]
    ax.annotate(
        "RALU@45",
        xy=(ralu.speedup, ralu.quality),
        xytext=(ralu.speedup, 0.7855),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=6.25,
        color=COLORS["ralu"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["ralu"],
            "linewidth": 0.55,
            "shrinkA": 2.0,
            "shrinkB": 4.0,
        },
        zorder=5,
    )

    ax.set_xlim(0.72, 6.12)
    ax.set_ylim(0.759, 0.8335)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks([0.76, 0.78, 0.80, 0.82])
    ax.set_xlabel(r"Speedup over Native-HR ($\times$)")
    ax.set_ylabel("VBench-5", labelpad=1.0)
    ax.tick_params(axis="y", pad=2.0)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.72)
    ax.set_axisbelow(True)

    ax.annotate(
        "better",
        xy=(5.92, 0.8315),
        xytext=(5.28, 0.8225),
        ha="center",
        va="center",
        fontsize=6.4,
        color="#555555",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#777777",
            "linewidth": 0.8,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["pareto"],
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.2,
            label=r"$\epsilon$-Pareto (1% speed)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.145, 0.985),
        ncol=1,
        handlelength=1.35,
        handletextpad=0.32,
        borderaxespad=0.0,
    )

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PDF_PATH,
        metadata={
            "Title": "Wan50 tolerance-aware Pareto candidate",
            "Author": "Anonymous",
            "Subject": f"Generated from {DATA_PATH.name}",
        },
    )
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    return frontier


def main() -> None:
    points = load_points()
    frontier = draw(points)
    print(f"Epsilon-Pareto frontier ({SPEEDUP_TOLERANCE:.0%} speed tolerance):")
    for point in frontier:
        print(
            f"  {point.label}: {point.speedup:.3f}x, "
            f"VBench-5={point.quality:.5f}"
        )
    print(f"Data: {DATA_PATH}")
    print(f"PDF:  {PDF_PATH}")
    print(f"PNG:  {PNG_PATH}")


if __name__ == "__main__":
    main()
