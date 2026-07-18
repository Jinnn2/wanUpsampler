#!/usr/bin/env python3
"""Generate Figure 3: TALH component evidence and factorial interaction.

Outputs:
    fig_component_evidence.pdf
    fig_component_evidence.png
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _figure_style import COLORS, TABLE_DIR, apply_publication_style, panel_title, save_figure


def operator_reductions(path: str) -> np.ndarray:
    table = pd.read_csv(TABLE_DIR / path).set_index("metric")
    metrics = ["latent_l1", "lpips", "temporal_l1"]
    return np.array(
        [100.0 * (table.loc[m, "interp_mean"] - table.loc[m, "trained_mean"]) / table.loc[m, "interp_mean"] for m in metrics]
    )


def endpoint_pairs() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    step40 = pd.read_csv(TABLE_DIR / "wan50_step40_endpoint_paired_statistics.csv")
    row40 = step40[(step40["metric"] == "l1") & (step40["strength_tag"] == "0p75")].iloc[0]
    step45 = pd.read_csv(TABLE_DIR / "wan50_step45_final_endpoint_paired_statistics.csv")
    row45 = step45[step45["metric"] == "l1"].iloc[0]
    distill = pd.read_csv(TABLE_DIR / "distill_transfer_paired_statistics.csv")
    row_d4 = distill[distill["metric"] == "l1"].iloc[0]
    before = np.array([row40["original_mean"], row45["original_mean"], row_d4["original_mean"]])
    after = np.array([row40["lora_mean"], row45["lora_mean"], row_d4["lora_mean"]])
    relative = 100.0 * (before - after) / before
    return ["Wan50 @ 40", "Wan50 @ 45", "Distill4 @ 3/4"], before, after, relative


def factorial_matrix() -> np.ndarray:
    table = pd.read_csv(TABLE_DIR / "vbench_case_summary.csv")
    rows = [
        ("wan50_step40_strength", ("step40_base_interp", "step40_lora_s0p75_interp", "step40_base_stage2", "step40_lora_s0p75_stage2")),
        ("wan50", ("step45_base_interp", "step45_lora_interp", "step45_base_stage2", "step45_lora_stage2")),
        ("distill4", ("step3_base_interp", "step3_lora_interp", "step3_base_stage2", "step3_lora_stage2")),
    ]
    matrix = []
    for family, cases in rows:
        family_table = table[table["family"] == family].set_index("case")
        scores = family_table.loc[list(cases), "quality5_mean"].to_numpy(dtype=float)
        matrix.append(scores - scores[0])
    return np.asarray(matrix)


def main() -> None:
    apply_publication_style()
    fig = plt.figure(figsize=(7.0, 2.72))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.08, 1.68], wspace=0.43)

    # (a) CLL error reduction.
    ax = fig.add_subplot(grid[0, 0])
    reduction_480 = operator_reductions("operator_480p.csv")
    reduction_368 = operator_reductions("operator_368p.csv")
    labels = ["Latent L1", "LPIPS", "Temporal L1"]
    x = np.arange(len(labels))
    width = 0.36
    bars_480 = ax.bar(
        x - width / 2,
        reduction_480,
        width,
        color=COLORS["cll"],
        hatch="///",
        label="480x832 -> 720p",
    )
    bars_368 = ax.bar(
        x + width / 2,
        reduction_368,
        width,
        color="#66BFA3",
        hatch="\\\\",
        label="368x640 -> 720p",
    )
    for bars, values in [(bars_480, reduction_480), (bars_368, reduction_368)]:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.7,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )
    panel_title(ax, "a", "CLL lifting")
    ax.set_ylabel("Error reduction vs. trilinear (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Latent\nL1", "LPIPS", "Temporal\nL1"])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", handlelength=1.3, borderaxespad=0.2)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    # (b) TAA endpoint alignment.
    ax = fig.add_subplot(grid[0, 1])
    config_labels, before, after, relative = endpoint_pairs()
    y = np.arange(len(config_labels))[::-1]
    for yi, x0, x1, reduction in zip(y, before, after, relative):
        ax.plot([x1, x0], [yi, yi], color="#A9A9A9", linewidth=1.4, zorder=1)
        ax.scatter(x0, yi, s=34, color=COLORS["baseline"], marker="o", edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(x1, yi, s=40, color=COLORS["taa"], marker="^", edgecolor="white", linewidth=0.6, zorder=3)
        ax.text((x0 + x1) / 2, yi + 0.19, f"-{reduction:.1f}%", ha="center", va="bottom", fontsize=7.1, color=COLORS["taa"], fontweight="bold")
    panel_title(ax, "b", "TAA alignment")
    ax.set_xlabel("Endpoint L1  (lower is better)")
    ax.set_yticks(y)
    ax.set_yticklabels(config_labels)
    ax.set_xlim(0.015, 0.046)
    ax.set_ylim(-0.55, 2.55)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], color=COLORS["baseline"], marker="o", label="Unaligned")
    ax.scatter([], [], color=COLORS["taa"], marker="^", label="TAA-aligned")
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)

    # (c) Factorial interaction as within-row VBench-5 deltas.
    ax = fig.add_subplot(grid[0, 2])
    matrix = factorial_matrix()
    cmap = LinearSegmentedColormap.from_list("talh_blue", ["#FFFFFF", "#D6E9F4", COLORS["talh_q"]])
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=0.045, aspect="auto", interpolation="nearest")
    row_labels = ["Wan50 @ 40", "Wan50 @ 45", "Distill4 @ 3/4"]
    col_labels = ["Unaligned\n+ Trilinear", "TAA\n+ Trilinear", "Unaligned\n+ CLL", "TAA + CLL\n(TALH)"]
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if value < 0:
                ax.add_patch(Rectangle((col - 0.47, row - 0.47), 0.94, 0.94, fill=False, edgecolor=COLORS["talh_e"], linewidth=1.3))
            text_color = "white" if value > 0.028 else COLORS["text"]
            label = "0.000" if abs(value) < 0.00005 else (f"{value:+.4f}" if abs(value) < 0.001 else f"{value:+.3f}")
            ax.text(col, row, label, ha="center", va="center", fontsize=7.1, color=text_color, fontweight="bold" if col >= 2 else "normal")
    panel_title(ax, "c", "Factorial interaction")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(col_labels, fontsize=6.8)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label(r"$\Delta$ VBench-5", fontsize=7.3)
    cbar.ax.tick_params(labelsize=6.8, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.86)
    pdf_path, png_path = save_figure(fig, "fig_component_evidence")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
