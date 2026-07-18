#!/usr/bin/env python3
"""Generate Figure 2: TALH quality--efficiency trade-off.

Outputs:
    fig_quality_efficiency.pdf
    fig_quality_efficiency.png
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _figure_style import COLORS, TABLE_DIR, apply_publication_style, panel_title, save_figure


DISPLAY = {
    "full_hr50": "Native-HR",
    "talh40": "TALH-Q",
    "talh45": "TALH-E",
    "full_lr50_stage2_1hr": "Endpoint\nRe-entry",
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
    details = pd.read_csv(TABLE_DIR / "wan50_final_quality_efficiency.csv").set_index("case")

    order = ["full_lr50_stage2_1hr", "talh45", "talh40", "full_hr50"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.52),
        gridspec_kw={"width_ratios": [1.02, 1.15], "wspace": 0.30},
    )

    # (a) Quality--latency operating points.
    ax = axes[0]
    x = summary.loc[order, "elapsed_mean_s"].to_numpy()
    y = summary.loc[order, "quality5_mean"].to_numpy()
    ax.plot(x, y, color="#B8B8B8", linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)

    offsets = {
        "full_lr50_stage2_1hr": (5, 8),
        "talh45": (8, -20),
        "talh40": (6, 8),
        "full_hr50": (-5, 9),
    }
    for case in order:
        row = summary.loc[case]
        color, marker = POINT_STYLE[case]
        ax.errorbar(
            row["elapsed_mean_s"],
            row["quality5_mean"],
            xerr=row["elapsed_std_s"],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=7.1,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )
        suffix = ""
        if case == "talh40":
            suffix = "  1.83x"
        elif case == "talh45":
            suffix = "  2.22x"
        ax.annotate(
            DISPLAY[case] + suffix,
            (row["elapsed_mean_s"], row["quality5_mean"]),
            xytext=offsets[case],
            textcoords="offset points",
            ha="left",
            va="baseline",
            fontsize=7.3,
            fontweight="bold" if case.startswith("talh") else "normal",
            color=color if case.startswith("talh") else COLORS["text"],
        )

    panel_title(ax, "a", "Quality--latency operating points")
    ax.set_xlabel(r"End-to-end latency per video (s)  $\leftarrow$ faster")
    ax.set_ylabel("VBench-5  (higher is better)")
    ax.set_xlim(75, 270)
    ax.set_ylim(0.797, 0.8335)
    ax.set_xticks([80, 120, 160, 200, 240])
    ax.grid(axis="both")

    # (b) Per-dimension change from actual 720p Native-HR Sampling.
    ax = axes[1]
    component_keys = [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "aesthetic_quality",
        "imaging_quality",
    ]
    labels = ["Subject", "Background", "Motion", "Aesthetic", "Imaging"]
    native = json.loads(details.loc["full_hr50", "quality_components"])
    q = json.loads(details.loc["talh40", "quality_components"])
    e = json.loads(details.loc["talh45", "quality_components"])
    delta_q = np.array([q[key] - native[key] for key in component_keys])
    delta_e = np.array([e[key] - native[key] for key in component_keys])

    positions = np.arange(len(labels))
    width = 0.35
    bars_q = ax.bar(
        positions - width / 2,
        delta_q,
        width,
        label="TALH-Q",
        color=COLORS["talh_q"],
        hatch="///",
        edgecolor="white",
        linewidth=0.55,
    )
    bars_e = ax.bar(
        positions + width / 2,
        delta_e,
        width,
        label="TALH-E",
        color=COLORS["talh_e"],
        hatch="\\\\",
        edgecolor="white",
        linewidth=0.55,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    for bars, values in [(bars_q, delta_q), (bars_e, delta_e)]:
        for bar, value in zip(bars, values):
            label = f"{value:.4f}" if abs(value) < 0.001 else f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.0014,
                label,
                ha="center",
                va="top",
                rotation=90,
                fontsize=6.5,
                color="white" if value < -0.009 else COLORS["text"],
            )

    panel_title(ax, "b", "VBench change from Native-HR")
    ax.set_ylabel("Absolute score change")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.0505, 0.006)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, handlelength=1.4)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.86)
    # The manuscript uses the selected ImageGen rendering. Keep this
    # deterministic implementation as an archived, non-canonical alternative.
    pdf_path, png_path = save_figure(
        fig,
        "_archive/unused_alternatives/fig_quality_efficiency_vector",
    )
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
