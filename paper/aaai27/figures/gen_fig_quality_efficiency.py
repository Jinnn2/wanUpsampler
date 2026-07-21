#!/usr/bin/env python3
"""Generate the system-efficiency and per-dimension VBench figure.

Outputs:
    fig_quality_efficiency.pdf
    fig_quality_efficiency.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _figure_style import COLORS, TABLE_DIR, apply_publication_style, panel_title, save_figure


DISPLAY = {
    "full_hr50": "Native-HR",
    "talh40": "TrajScale-Q",
    "talh45": "TrajScale-E",
    "full_lr50_stage2_1hr": "Post-Gen. Cascade",
}

POINT_STYLE = {
    "full_hr50": (COLORS["native"], "s"),
    "talh40": (COLORS["talh_q"], "o"),
    "talh45": (COLORS["talh_e"], "^"),
    "full_lr50_stage2_1hr": (COLORS["endpoint"], "D"),
}


def main() -> None:
    apply_publication_style()
    summary = pd.read_csv(TABLE_DIR / "quality_efficiency_summary.csv").set_index("case")
    dimensions = pd.read_csv(TABLE_DIR / "vbench_case_summary.csv")
    dimensions = dimensions.loc[
        dimensions["family"].eq("wan50_quality_efficiency")
    ].set_index("case")

    order = ["full_lr50_stage2_1hr", "talh45", "talh40", "full_hr50"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.52),
        gridspec_kw={"width_ratios": [0.90, 1.35], "wspace": 0.30},
    )

    # (a) Latency and speedup without a custom quality aggregate.
    ax = axes[0]
    y = np.arange(len(order))
    latency = summary.loc[order, "elapsed_mean_s"].to_numpy()
    latency_std = summary.loc[order, "elapsed_std_s"].to_numpy()
    colors = [POINT_STYLE[case][0] for case in order]
    ax.barh(y, latency, xerr=latency_std, color=colors, height=0.58, capsize=2)
    for pos, case, value in zip(y, order, latency):
        speedup = summary.loc[case, "speedup_vs_full_hr"]
        ax.text(value + 5, pos, f"{speedup:.2f}x", va="center", fontsize=7.3)
    panel_title(ax, "a", "End-to-end efficiency")
    ax.set_xlabel(r"Cold-start latency per video (s)  $\leftarrow$ faster")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[case] for case in order])
    ax.set_xlim(0, 285)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

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
        label="TrajScale-Q",
        color=COLORS["talh_q"],
        hatch="///",
        edgecolor="white",
        linewidth=0.55,
    )
    ax.bar(
        positions + width / 2,
        delta_e,
        width,
        label="TrajScale-E",
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

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.23, top=0.86)
    pdf_path, png_path = save_figure(fig, "fig_quality_efficiency")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
