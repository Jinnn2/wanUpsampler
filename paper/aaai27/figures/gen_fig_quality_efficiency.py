#!/usr/bin/env python3
"""Generate the quality--latency Pareto and per-dimension VBench figure.

Outputs:
    fig_quality_efficiency.pdf
    fig_quality_efficiency.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _figure_style import (
    COLORS,
    REPO_ROOT,
    TABLE_DIR,
    apply_publication_style,
    panel_title,
    save_figure,
)


WARM_TABLE = (
    REPO_ROOT
    / "paper"
    / "aaai27"
    / "results"
    / "warm_quality_efficiency_20260722.csv"
)

POINT_STYLE = {
    "full_hr50": (COLORS["native"], "s"),
    "lightx2v_cr40": ("#56B4E9", "o"),
    "lightx2v_cr45": ("#56B4E9", "o"),
    "lightx2v_cr48": ("#56B4E9", "o"),
    "ralu_quality": ("#CC79A7", "X"),
    "talh40": (COLORS["talh_q"], "o"),
    "talh45": (COLORS["talh_e"], "^"),
    "full_lr50_stage2_0hr": ("#009E73", "D"),
    "full_lr50_stage2_1hr": ("#009E73", "D"),
    "full_lr50_stage2_2hr": ("#009E73", "D"),
    "full_lr50_stage2_5hr": ("#009E73", "D"),
}


def pareto_frontier(rows: pd.DataFrame) -> pd.DataFrame:
    """Return points that are non-dominated for lower latency/higher quality."""
    ordered = rows.sort_values("pipeline_mean_s")
    keep: list[str] = []
    best_quality = -np.inf
    for case, row in ordered.iterrows():
        quality = float(row["quality_value"])
        if quality > best_quality:
            keep.append(case)
            best_quality = quality
    return ordered.loc[keep]


def main() -> None:
    apply_publication_style()
    summary = pd.read_csv(WARM_TABLE).set_index("case")
    dimensions = pd.read_csv(TABLE_DIR / "vbench_case_summary.csv")
    dimensions = dimensions.loc[
        dimensions["family"].eq("wan50_quality_efficiency")
    ].set_index("case")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.62),
        gridspec_kw={"width_ratios": [1.23, 1.0], "wspace": 0.29},
    )

    # (a) Unified quality--latency comparison and empirical Pareto frontier.
    ax = axes[0]
    family_paths = {
        "lightx2v": ["lightx2v_cr48", "lightx2v_cr45", "lightx2v_cr40"],
        "endpoint": [
            "full_lr50_stage2_0hr",
            "full_lr50_stage2_1hr",
            "full_lr50_stage2_2hr",
            "full_lr50_stage2_5hr",
        ],
        "talh": ["talh45", "talh40"],
    }
    for cases in family_paths.values():
        ax.plot(
            summary.loc[cases, "pipeline_mean_s"],
            summary.loc[cases, "quality_value"],
            color=POINT_STYLE[cases[0]][0],
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    frontier = pareto_frontier(summary)
    ax.plot(
        frontier["pipeline_mean_s"],
        frontier["quality_value"],
        color="#333333",
        linestyle=(0, (3, 2)),
        linewidth=1.25,
        label="Empirical Pareto frontier",
        zorder=2,
    )

    for case, row in summary.iterrows():
        color, marker = POINT_STYLE[case]
        is_trajscale = case.startswith("talh")
        ax.errorbar(
            row["pipeline_mean_s"],
            row["quality_value"],
            xerr=row["pipeline_std_s"],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=7.3 if is_trajscale else (6.7 if case == "ralu_quality" else 6.0),
            ecolor=color,
            elinewidth=0.7,
            capsize=1.7,
            zorder=4 if is_trajscale else 3,
        )

    annotations = {
        "full_hr50": ("Native-HR", (-5, -13), "right"),
        "lightx2v_cr40": ("L40", (-4, 7), "right"),
        "lightx2v_cr45": ("L45", (5, -1), "left"),
        "lightx2v_cr48": ("L48", (4, -8), "left"),
        "ralu_quality": ("RALU-Q", (8, -12), "left"),
        "full_lr50_stage2_0hr": ("0 HR", (4, -10), "left"),
        "full_lr50_stage2_1hr": ("1 HR", (4, -10), "left"),
        "full_lr50_stage2_2hr": ("2 HR", (-3, 10), "right"),
        "full_lr50_stage2_5hr": ("5 HR", (7, -20), "left"),
    }
    for case, (label, offset, alignment) in annotations.items():
        row = summary.loc[case]
        ax.annotate(
            label,
            (row["pipeline_mean_s"], row["quality_value"]),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="baseline",
            fontsize=6.5,
            fontweight="normal",
            color=COLORS["text"],
            zorder=5,
        )

    for case, label, label_xy in [
        ("talh45", "TrajScale-45", (49.0, 0.8170)),
        ("talh40", "TrajScale-40", (76.0, 0.8130)),
    ]:
        row = summary.loc[case]
        ax.annotate(
            label,
            (row["pipeline_mean_s"], row["quality_value"]),
            xytext=label_xy,
            textcoords="data",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=POINT_STYLE[case][0],
            arrowprops={
                "arrowstyle": "-",
                "color": POINT_STYLE[case][0],
                "linewidth": 0.65,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            zorder=5,
        )

    handles = [
        plt.Line2D([], [], color="#333333", linestyle=(0, (3, 2)), label="Pareto"),
        plt.Line2D([], [], color="#56B4E9", marker="o", linestyle="", label="LightX2V"),
        plt.Line2D([], [], color="#CC79A7", marker="X", linestyle="", label="RALU"),
        plt.Line2D([], [], color="#009E73", marker="D", linestyle="", label="Post-Gen."),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    panel_title(ax, "a", "Quality--latency Pareto frontier")
    ax.set_xlabel(r"Warm-model latency per video (s)  $\leftarrow$ faster")
    ax.set_ylabel(r"VBench-5  $\uparrow$")
    ax.set_xlim(20, 196)
    ax.set_ylim(0.758, 0.835)
    ax.set_xticks([25, 50, 100, 150, 190])
    ax.grid(axis="both")

    # (b) All six original VBench dimensions relative to Native-HR.
    ax = axes[1]
    component_keys = [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
    ]
    labels = ["Subject", "Backgr.", "Motion", "Dynamic", "Aesthetic", "Imaging"]
    native = dimensions.loc["full_hr50", component_keys].to_numpy(dtype=float)
    q = dimensions.loc["talh40", component_keys].to_numpy(dtype=float)
    e = dimensions.loc["talh45", component_keys].to_numpy(dtype=float)
    delta_q = q - native
    delta_e = e - native

    positions = np.arange(len(labels))
    width = 0.35
    ax.bar(
        positions - width / 2,
        delta_q,
        width,
        label="TrajScale-40",
        color=COLORS["talh_q"],
        hatch="///",
        edgecolor="white",
        linewidth=0.55,
    )
    ax.bar(
        positions + width / 2,
        delta_e,
        width,
        label="TrajScale-45",
        color=COLORS["talh_e"],
        hatch="\\\\",
        edgecolor="white",
        linewidth=0.55,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    panel_title(ax, "b", "VBench dimensions vs. Native-HR")
    ax.set_ylabel("Absolute score change")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.055, 0.115)
    ax.legend(loc="upper left", ncol=2, handlelength=1.4)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.86)
    pdf_path, png_path = save_figure(fig, "fig_quality_efficiency")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
