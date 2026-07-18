#!/usr/bin/env python3
"""Generate Figure 1: deterministic vector overview of TALH.

The original figure plan permits a deterministic vector fallback when exact
diagram labels cannot be guaranteed by a generative image model.

Outputs:
    fig_talh_overview.pdf
    fig_talh_overview.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _figure_style import COLORS, apply_publication_style, save_figure


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str = COLORS["border"],
    face: str = "white",
    fontsize: float = 7.0,
    weight: str = "normal",
    text_color: str = COLORS["text"],
    linewidth: float = 0.8,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        linespacing=1.05,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#777777",
    style: str = "-",
    width: float = 0.9,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=width,
            linestyle=style,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def band(ax: plt.Axes, y: float, height: float, label: str, accent: str) -> None:
    ax.add_patch(
        Rectangle(
            (0.012, y),
            0.976,
            height,
            transform=ax.transAxes,
            facecolor=COLORS["band"],
            edgecolor=COLORS["border"],
            linewidth=0.75,
        )
    )
    ax.add_patch(Rectangle((0.012, y), 0.009, height, transform=ax.transAxes, facecolor=accent, edgecolor="none"))
    ax.text(0.029, y + height - 0.025, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=accent)


def schedule_bar(ax: plt.Axes, y: float, name: str, lr_fraction: float) -> None:
    x0, width, height = 0.105, 0.255, 0.045
    ax.text(0.035, y + height / 2, name, transform=ax.transAxes, ha="left", va="center", fontsize=7.1, fontweight="bold" if name.startswith("TALH") else "normal")
    if lr_fraction > 0:
        ax.add_patch(Rectangle((x0, y), width * lr_fraction, height, transform=ax.transAxes, facecolor=COLORS["lr"], edgecolor="white", linewidth=0.4))
    if lr_fraction < 1:
        ax.add_patch(Rectangle((x0 + width * lr_fraction, y), width * (1 - lr_fraction), height, transform=ax.transAxes, facecolor=COLORS["hr"], edgecolor="white", linewidth=0.4))
    ax.add_patch(Rectangle((x0, y), width, height, transform=ax.transAxes, fill=False, edgecolor="#777777", linewidth=0.55))


def main() -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.22))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Upper band: hybrid-resolution inference.
    band(ax, 0.53, 0.455, "INFERENCE: HYBRID-RESOLUTION TRAJECTORY", COLORS["lr"])
    ax.text(0.035, 0.902, "Resolution schedules", transform=ax.transAxes, fontsize=7.6, fontweight="bold")
    ax.annotate(
        "Structure & Motion",
        xy=(0.105, 0.865),
        xytext=(0.105, 0.865),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.7,
        color=COLORS["lr"],
        fontweight="bold",
    )
    arrow(ax, (0.205, 0.865), (0.355, 0.865), color="#888888", width=0.7)
    ax.text(0.355, 0.865, "Texture & Detail", transform=ax.transAxes, ha="right", va="center", fontsize=6.7, color=COLORS["hr"], fontweight="bold")

    schedule_bar(ax, 0.795, "Native-HR", 0.0)
    schedule_bar(ax, 0.715, "TALH-Q", 0.8)
    schedule_bar(ax, 0.635, "TALH-E", 0.9)
    for step, xpos in [("0", 0.105), ("40", 0.309), ("45", 0.3345), ("50", 0.36)]:
        ax.text(xpos, 0.598, step, transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color=COLORS["muted"])
    ax.text(0.232, 0.565, "Denoising evaluation", transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color=COLORS["muted"])

    ax.plot([0.385, 0.385], [0.565, 0.93], transform=ax.transAxes, color=COLORS["border"], linewidth=0.7)
    ax.text(0.405, 0.902, "Learnable resolution handoff", transform=ax.transAxes, fontsize=7.6, fontweight="bold")

    box_y, box_h = 0.715, 0.095
    boxes = [
        (0.405, 0.073, "LR Prefix", COLORS["lr"], "#EAF2FA"),
        (0.495, 0.072, "TAA\nLoRA", COLORS["taa"], "#F7EAF2"),
        (0.584, 0.078, "Aligned\nclean LR", COLORS["lr"], "white"),
        (0.679, 0.058, "CLL", COLORS["cll"], "#E8F5EF"),
        (0.754, 0.078, "Lifted\nclean HR", COLORS["cll"], "white"),
        (0.849, 0.058, "HTR", COLORS["hr"], "#FBEDE7"),
        (0.924, 0.061, "HR\nSuffix", COLORS["hr"], "#FBEDE7"),
    ]
    for x, w, label, edge, face in boxes:
        rounded_box(ax, x, box_y, w, box_h, label, edge=edge, face=face, fontsize=6.6, weight="bold" if label in {"TAA\nLoRA", "CLL", "HTR"} else "normal")
    for (x, w, *_), (next_x, *_rest) in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (x + w, box_y + box_h / 2), (next_x - 0.004, box_y + box_h / 2), color="#737373", width=0.75)
    ax.text(0.531, 0.684, "base frozen", transform=ax.transAxes, ha="center", fontsize=6.1, color=COLORS["muted"])
    ax.text(0.708, 0.684, "LR -> HR", transform=ax.transAxes, ha="center", fontsize=6.1, color=COLORS["muted"])
    ax.text(0.878, 0.684, "re-noise", transform=ax.transAxes, ha="center", fontsize=6.1, color=COLORS["muted"])
    ax.text(0.531, 0.617, r"$E_{traj}(s)$", transform=ax.transAxes, ha="center", fontsize=7.0, color=COLORS["taa"])
    ax.text(0.708, 0.617, r"$E_{lift}(s)$", transform=ax.transAxes, ha="center", fontsize=7.0, color=COLORS["cll"])
    ax.text(0.878, 0.617, r"$E_{refine}(s)$", transform=ax.transAxes, ha="center", fontsize=7.0, color=COLORS["hr"])
    for xpos, color in [(0.531, COLORS["taa"]), (0.708, COLORS["cll"]), (0.878, COLORS["hr"])]:
        ax.plot([xpos, xpos], [0.642, 0.673], transform=ax.transAxes, color=color, linestyle=(0, (2, 2)), linewidth=0.8)

    # Lower band: model-internal supervision.
    band(ax, 0.055, 0.445, "MODEL-INTERNAL SUPERVISION", COLORS["cll"])
    ax.add_patch(Rectangle((0.032, 0.105), 0.445, 0.325, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["border"], linewidth=0.65))
    ax.add_patch(Rectangle((0.495, 0.105), 0.472, 0.325, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["border"], linewidth=0.65))
    ax.add_patch(Rectangle((0.032, 0.105), 0.006, 0.325, transform=ax.transAxes, facecolor=COLORS["taa"], edgecolor="none"))
    ax.add_patch(Rectangle((0.495, 0.105), 0.006, 0.325, transform=ax.transAxes, facecolor=COLORS["cll"], edgecolor="none"))
    ax.text(0.048, 0.405, "TAA: trajectory-alignment pair", transform=ax.transAxes, ha="left", va="top", fontsize=7.3, fontweight="bold", color=COLORS["taa"])
    ax.text(0.511, 0.405, "CLL: cross-resolution lifting pair", transform=ax.transAxes, ha="left", va="top", fontsize=7.3, fontweight="bold", color=COLORS["cll"])

    rounded_box(ax, 0.053, 0.235, 0.105, 0.085, "Frozen Wan\nLR rollout", edge=COLORS["native"], face="#F0F2F3", fontsize=6.7)
    rounded_box(ax, 0.200, 0.293, 0.105, 0.075, "Cached state\n" + r"$x_s^L$" + "\n(inference state)", edge=COLORS["lr"], fontsize=5.7)
    rounded_box(ax, 0.200, 0.170, 0.105, 0.075, "Full endpoint\n" + r"$z_T^L$", edge=COLORS["native"], fontsize=6.6)
    rounded_box(ax, 0.355, 0.230, 0.090, 0.090, "TAA\ntraining pair", edge=COLORS["taa"], face="#F7EAF2", fontsize=6.7, weight="bold")
    arrow(ax, (0.158, 0.277), (0.196, 0.330), color=COLORS["lr"], width=0.8)
    arrow(ax, (0.158, 0.277), (0.196, 0.207), color=COLORS["native"], width=0.8)
    arrow(ax, (0.305, 0.330), (0.351, 0.285), color=COLORS["lr"], width=0.8)
    arrow(ax, (0.305, 0.207), (0.351, 0.260), color=COLORS["native"], width=0.8)
    ax.text(0.251, 0.128, "same prompt / seed / scheduler / CFG", transform=ax.transAxes, ha="center", va="center", fontsize=5.9, color=COLORS["muted"])

    rounded_box(ax, 0.516, 0.235, 0.078, 0.085, "Frozen\nWan", edge=COLORS["native"], face="#F0F2F3", fontsize=6.6)
    rounded_box(ax, 0.620, 0.235, 0.075, 0.085, "HR\nvideo", edge=COLORS["hr"], fontsize=6.6)
    rounded_box(ax, 0.735, 0.302, 0.071, 0.065, "Wan VAE", edge=COLORS["native"], fontsize=6.2)
    rounded_box(ax, 0.735, 0.158, 0.071, 0.065, "RGB\ndownsample", edge=COLORS["lr"], fontsize=5.9)
    rounded_box(ax, 0.833, 0.158, 0.071, 0.065, "same\nWan VAE", edge=COLORS["native"], fontsize=5.9)
    rounded_box(ax, 0.925, 0.302, 0.037, 0.065, r"$z_0^H$", edge=COLORS["hr"], fontsize=7.0)
    rounded_box(ax, 0.925, 0.158, 0.037, 0.065, r"$z_0^L$", edge=COLORS["lr"], fontsize=7.0)
    arrow(ax, (0.594, 0.277), (0.616, 0.277), color=COLORS["native"], width=0.8)
    arrow(ax, (0.695, 0.277), (0.731, 0.334), color=COLORS["hr"], width=0.8)
    arrow(ax, (0.695, 0.277), (0.731, 0.190), color=COLORS["lr"], width=0.8)
    arrow(ax, (0.806, 0.334), (0.921, 0.334), color=COLORS["hr"], width=0.8)
    arrow(ax, (0.806, 0.190), (0.829, 0.190), color=COLORS["lr"], width=0.8)
    arrow(ax, (0.904, 0.190), (0.921, 0.190), color=COLORS["lr"], width=0.8)
    ax.plot([0.948, 0.948], [0.232, 0.293], transform=ax.transAxes, color=COLORS["cll"], linestyle=(0, (2, 2)), linewidth=0.9)
    ax.text(0.869, 0.265, "paired clean latents", transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color=COLORS["cll"], fontweight="bold")

    ax.text(
        0.5,
        0.075,
        "No external paired videos, SR weights, or extra teacher",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.995)
    # The manuscript uses the selected ImageGen rendering. Keep this
    # deterministic implementation as an archived, non-canonical alternative.
    pdf_path, png_path = save_figure(
        fig,
        "_archive/unused_alternatives/fig_talh_overview_vector",
        png_dpi=350,
    )
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
