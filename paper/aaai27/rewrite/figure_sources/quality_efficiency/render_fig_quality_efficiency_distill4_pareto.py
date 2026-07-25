#!/usr/bin/env python3
"""Render a single-column Distill4 quality--efficiency Pareto candidate.

The figure uses the five operating points reported in the paper's Distill4
main table. Latency summaries and VBench-5 scores come from the newest
validated P0/P1/P3 export. All five points are strictly non-dominated under
higher-speedup/higher-quality Pareto dominance.

This script writes a separate candidate and does not modify rewrite/main.tex.
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
DATA_PATH = (
    AAAI27_DIR
    / "results"
    / "distill4_final_exports_20260723"
    / "distill4_p0_p1_p3_final_20260723T064558Z"
    / "main_suite"
    / "warm_quality_efficiency"
    / "quality_efficiency_warm.csv"
)
PDF_PATH = REWRITE_DIR / "figures" / "fig_quality_efficiency_distill4_pareto.pdf"
PNG_PATH = REWRITE_DIR / "figures" / "fig_quality_efficiency_distill4_pareto.png"


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
    "native_hr4": ("Native-HR4", "native"),
    "endpoint_stage2_2hr": ("LUVE-style", "luve"),
    "endpoint_rgb_1hr": ("MrFlow-style", "mrflow"),
    "talh3": ("InTraScale-D4@3", "intrascale"),
    "interp3": ("Trilinear@3", "trilinear"),
}

# Rounded values printed in the Distill4 table in rewrite/main.tex.
TABLE_VALUES = {
    "native_hr4": (42.49, 0.86041),
    "endpoint_stage2_2hr": (28.76, 0.85981),
    "endpoint_rgb_1hr": (27.12, 0.85903),
    "talh3": (19.92, 0.85680),
    "interp3": (18.33, 0.81796),
}

COLORS = {
    "native": "#4B5563",
    "trilinear": "#56B4E9",
    "luve": "#009E73",
    "mrflow": "#CC79A7",
    "intrascale": "#D55E00",
    "pareto": "#222222",
    "grid": "#D9D9D9",
    "text": "#242424",
}

MARKERS = {
    "native": "s",
    "trilinear": "o",
    "luve": "D",
    "mrflow": "X",
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

    native_latency = float(rows["native_hr4"]["pipeline_mean_s"])
    native_latency_std = float(rows["native_hr4"]["pipeline_std_s"])
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

        expected_latency, expected_quality = TABLE_VALUES[case]
        if round(latency, 2) != expected_latency:
            raise RuntimeError(
                f"{case}: latency {latency:.5f} disagrees with table value "
                f"{expected_latency:.2f}"
            )
        if round(quality, 5) != expected_quality:
            raise RuntimeError(
                f"{case}: VBench-5 {quality:.8f} disagrees with table value "
                f"{expected_quality:.5f}"
            )

        speedup = native_latency / latency
        points[case] = Point(
            case=case,
            label=label,
            family=family,
            latency=latency,
            latency_std=latency_std,
            quality=quality,
            speedup=speedup,
            speedup_std=(
                0.0
                if case == "native_hr4"
                else speedup
                * math.sqrt(
                    (native_latency_std / native_latency) ** 2
                    + (latency_std / latency) ** 2
                )
            ),
        )
    return points


def empirical_pareto(points: dict[str, Point]) -> list[Point]:
    """Return globally non-dominated points for higher speedup and quality."""
    frontier: list[Point] = []
    values = list(points.values())
    for point in values:
        dominated = any(
            other.case != point.case
            and other.speedup >= point.speedup
            and other.quality >= point.quality
            and (
                other.speedup > point.speedup
                or other.quality > point.quality
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
    frontier = empirical_pareto(points)
    frontier_cases = {point.case for point in frontier}

    fig, ax = plt.subplots(figsize=(3.30, 2.05))
    fig.subplots_adjust(left=0.145, right=0.955, bottom=0.245, top=0.855)

    ax.plot(
        [point.speedup for point in frontier],
        [point.quality for point in frontier],
        color=COLORS["pareto"],
        linewidth=1.25,
        linestyle=(0, (3.0, 2.0)),
        zorder=2,
    )

    for point in points.values():
        ax.errorbar(
            point.speedup,
            point.quality,
            xerr=point.speedup_std if point.speedup_std > 0 else None,
            fmt=MARKERS[point.family],
            markersize=6.5 if point.family == "intrascale" else 5.8,
            color=COLORS[point.family],
            markerfacecolor=COLORS[point.family],
            markeredgecolor=COLORS["pareto"] if point.case in frontier_cases else "white",
            markeredgewidth=0.85,
            ecolor=COLORS[point.family],
            elinewidth=0.55,
            capsize=1.35,
            capthick=0.55,
            zorder=4,
        )

    native = points["native_hr4"]
    ax.annotate(
        "Native-HR4",
        xy=(native.speedup, native.quality),
        xytext=(-2, -11),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.25,
        color=COLORS["text"],
        zorder=5,
    )

    ours = points["talh3"]
    ax.annotate(
        "InTraScale-D4@3",
        xy=(ours.speedup, ours.quality),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.55,
        fontweight="bold",
        color=COLORS["intrascale"],
        zorder=5,
    )

    trilinear = points["interp3"]
    ax.annotate(
        "Trilinear@3",
        xy=(trilinear.speedup, trilinear.quality),
        xytext=(-3, 8),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=6.25,
        color=COLORS["trilinear"],
        zorder=5,
    )

    leader_labels = {
        "endpoint_stage2_2hr": ("LUVE-style", (1.31, 0.8508)),
        "endpoint_rgb_1hr": ("MrFlow-style", (1.73, 0.8448)),
    }
    for case, (text, text_position) in leader_labels.items():
        point = points[case]
        ax.annotate(
            text,
            xy=(point.speedup, point.quality),
            xytext=text_position,
            textcoords="data",
            ha="center",
            va="center",
            fontsize=6.2,
            color=COLORS[point.family],
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[point.family],
                "linewidth": 0.55,
                "shrinkA": 2.0,
                "shrinkB": 4.0,
            },
            zorder=5,
        )

    ax.set_xlim(0.90, 2.40)
    ax.set_ylim(0.815, 0.8645)
    ax.set_xticks([1.0, 1.4, 1.8, 2.2])
    ax.set_yticks([0.82, 0.83, 0.84, 0.85, 0.86])
    ax.set_xlabel(r"Speedup over Native-HR4 ($\times$)")
    ax.set_ylabel("VBench-5", labelpad=1.0)
    ax.tick_params(axis="y", pad=2.0)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.72)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["pareto"],
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.2,
            label="Pareto (means)",
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
            "Title": "Distill4 quality-efficiency Pareto candidate",
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
    print("Empirical Pareto frontier:")
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
