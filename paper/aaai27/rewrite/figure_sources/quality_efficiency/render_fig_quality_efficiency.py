#!/usr/bin/env python3
"""Render the single-column Wan50 quality--efficiency figure.

The chart reads the canonical warm-model summary used by Table 1 and converts
latency to speedup relative to Native-HR. It intentionally excludes Distill4,
whose model size and denoising budget are not directly comparable to Wan50.
"""

from __future__ import annotations

import csv
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
PDF_PATH = REWRITE_DIR / "figures" / "fig_quality_efficiency.pdf"
PNG_PATH = REWRITE_DIR / "figures" / "fig_quality_efficiency.png"


@dataclass(frozen=True)
class Point:
    case: str
    label: str
    family: str
    latency: float
    latency_std: float
    quality: float
    speedup: float


CASE_SPEC = {
    "full_hr50": ("Native-HR", "native"),
    "lightx2v_cr40": ("Trilinear@40", "fixed"),
    "talh40": ("InTraScale@40", "intrascale"),
    "lightx2v_cr45": ("Trilinear@45", "fixed"),
    "ralu_nt45": ("RALU-style@45", "ralu"),
    "talh45": ("InTraScale@45", "intrascale"),
    "lightx2v_cr48": ("Trilinear@48", "fixed"),
    "full_lr50_stage2_5hr": ("Endpoint-ITU, 5 HR", "endpoint"),
}

# Rounded values printed in Table 1. These guards make source drift explicit.
TABLE_VALUES = {
    "full_hr50": (187.54, 0.82836),
    "lightx2v_cr40": (58.12, 0.77812),
    "talh40": (58.49, 0.80983),
    "lightx2v_cr45": (41.98, 0.76776),
    "ralu_nt45": (42.01, 0.80440),
    "talh45": (42.26, 0.80792),
    "lightx2v_cr48": (32.27, 0.76409),
    "full_lr50_stage2_5hr": (44.00, 0.80073),
}

COLORS = {
    "native": "#222222",
    "fixed": "#8A8A8A",
    "ralu": "#0072B2",
    "endpoint": "#009E73",
    "intrascale": "#D55E00",
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
    points: dict[str, Point] = {}
    for case, (label, family) in CASE_SPEC.items():
        row = rows[case]
        latency = float(row["pipeline_mean_s"])
        latency_std = float(row["pipeline_std_s"])
        quality = float(row["quality_value"])
        expected_latency, expected_quality = TABLE_VALUES[case]
        if round(latency, 2) != expected_latency:
            raise RuntimeError(
                f"{case}: latency {latency:.5f} disagrees with Table 1 "
                f"value {expected_latency:.2f}"
            )
        if round(quality, 5) != expected_quality:
            raise RuntimeError(
                f"{case}: VBench-5 {quality:.8f} disagrees with Table 1 "
                f"value {expected_quality:.5f}"
            )
        points[case] = Point(
            case=case,
            label=label,
            family=family,
            latency=latency,
            latency_std=latency_std,
            quality=quality,
            speedup=native_latency / latency,
        )
    return points


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
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


def draw(points: dict[str, Point]) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.30, 2.18))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.76)

    # Discrete operating-point paths; no fitted or interpolated curve is implied.
    fixed_cases = ["lightx2v_cr40", "lightx2v_cr45", "lightx2v_cr48"]
    fixed = [points[case] for case in fixed_cases]
    ax.plot(
        [point.speedup for point in fixed],
        [point.quality for point in fixed],
        color=COLORS["fixed"],
        linewidth=1.0,
        linestyle=(0, (2.2, 2.0)),
        alpha=0.8,
        zorder=1,
    )

    frontier_cases = ["full_hr50", "talh40", "talh45"]
    frontier = [points[case] for case in frontier_cases]
    ax.plot(
        [point.speedup for point in frontier],
        [point.quality for point in frontier],
        color=COLORS["intrascale"],
        linewidth=1.25,
        linestyle="-",
        alpha=0.82,
        zorder=2,
    )

    marker_spec = {
        "native": ("s", 34, 0.9),
        "fixed": ("X", 31, 0.7),
        "ralu": ("^", 38, 0.8),
        "endpoint": ("D", 34, 0.8),
        "intrascale": ("o", 42, 0.9),
    }
    for point in points.values():
        marker, size, linewidth = marker_spec[point.family]
        edgecolor = "white" if point.family in {"ralu", "endpoint", "intrascale"} else COLORS[point.family]
        ax.scatter(
            point.speedup,
            point.quality,
            s=size,
            marker=marker,
            facecolor=COLORS[point.family],
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=4,
        )

    # Direct schedule labels keep the compact legend semantic rather than case-heavy.
    annotations = {
        "full_hr50": ("Native-HR", (5, -10), "left"),
        "talh40": ("InTraScale@40", (-4, 8), "right"),
        "talh45": ("InTraScale@45", (4, 8), "left"),
        "ralu_nt45": ("RALU@45", (5, -12), "left"),
        "full_lr50_stage2_5hr": ("Endpoint-ITU", (-6, -11), "right"),
        "lightx2v_cr40": ("Trilinear@40", (0, 6), "center"),
        "lightx2v_cr45": ("Trilinear@45", (0, 7), "center"),
        "lightx2v_cr48": ("Trilinear@48", (-3, 6), "right"),
    }
    for case, (text, offset, horizontal_alignment) in annotations.items():
        point = points[case]
        is_ours = point.family == "intrascale"
        ax.annotate(
            text,
            xy=(point.speedup, point.quality),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=6.4 if not is_ours else 6.6,
            fontweight="bold" if is_ours else "normal",
            color=COLORS[point.family] if point.family != "fixed" else "#666666",
            zorder=5,
        )

    ax.set_xlim(0.72, 6.12)
    ax.set_ylim(0.759, 0.8335)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks([0.76, 0.78, 0.80, 0.82])
    ax.set_xlabel(r"Speedup over Native-HR ($\times$)")
    ax.set_ylabel("VBench-5")
    ax.grid(True, axis="both", color="#D9D9D9", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)

    ax.annotate(
        "better",
        xy=(5.92, 0.8315),
        xytext=(5.28, 0.8225),
        ha="center",
        va="center",
        fontsize=6.5,
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
            marker="s",
            linestyle="none",
            color=COLORS["native"],
            markerfacecolor=COLORS["native"],
            markersize=4.8,
            label="Native-HR",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle=(0, (2.2, 2.0)),
            color=COLORS["fixed"],
            markerfacecolor=COLORS["fixed"],
            markersize=4.6,
            linewidth=0.9,
            label="Trilinear",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            color=COLORS["ralu"],
            markerfacecolor=COLORS["ralu"],
            markersize=5.0,
            label="RALU-style",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color=COLORS["endpoint"],
            markerfacecolor=COLORS["endpoint"],
            markersize=4.6,
            label="Endpoint-ITU",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=COLORS["intrascale"],
            markerfacecolor=COLORS["intrascale"],
            markersize=5.0,
            linewidth=1.1,
            label="InTraScale",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.45,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PDF_PATH,
        metadata={
            "Title": "Wan50 quality-efficiency trade-off",
            "Author": "Anonymous",
            "Subject": f"Generated from {DATA_PATH.name}",
        },
    )
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    points = load_points()
    draw(points)
    print(f"Data: {DATA_PATH}")
    print(f"PDF:  {PDF_PATH}")
    print(f"PNG:  {PNG_PATH}")


if __name__ == "__main__":
    main()
