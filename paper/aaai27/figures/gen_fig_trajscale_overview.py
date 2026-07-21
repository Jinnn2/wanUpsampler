#!/usr/bin/env python3
"""Generate the no-TRR TrajScale method overview.

The figure makes the paper's revised causal order explicit:
CRLU defines the cross-resolution capability, EAA calibrates the deployment
input, and a parameter-free scheduler initialization connects the result to
the frozen high-resolution suffix.

Outputs:
    fig_trajscale_overview.pdf
    fig_trajscale_overview.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _figure_style import COLORS, apply_publication_style, save_figure


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    edge: str = COLORS["border"],
    face: str = "white",
    fontsize: float = 7.0,
    weight: str = "normal",
    linewidth: float = 0.9,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.08,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#6B6B6B",
    width: float = 0.9,
    style: str = "-",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=width,
            linestyle=style,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
            clip_on=False,
        )
    )


def section(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, accent: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            facecolor=COLORS["band"],
            edgecolor=COLORS["border"],
            linewidth=0.75,
        )
    )
    ax.add_patch(Rectangle((x, y), 0.007, h, transform=ax.transAxes, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.015, y + h - 0.022, title, transform=ax.transAxes, ha="left", va="top", fontsize=7.7, fontweight="bold", color=accent)


def schedule_bar(ax: plt.Axes, y: float, name: str, lr_fraction: float) -> None:
    x0, width, height = 0.108, 0.258, 0.048
    ax.text(0.025, y + height / 2, name, transform=ax.transAxes, ha="left", va="center", fontsize=7.0, fontweight="bold" if name.startswith("TrajScale") else "normal")
    if lr_fraction:
        ax.add_patch(Rectangle((x0, y), width * lr_fraction, height, transform=ax.transAxes, facecolor=COLORS["lr"], edgecolor="white", linewidth=0.35))
    if lr_fraction < 1:
        ax.add_patch(Rectangle((x0 + width * lr_fraction, y), width * (1 - lr_fraction), height, transform=ax.transAxes, facecolor=COLORS["hr"], edgecolor="white", linewidth=0.35))
    ax.add_patch(Rectangle((x0, y), width, height, transform=ax.transAxes, fill=False, edgecolor="#777777", linewidth=0.55))


def draw_inference_panel(ax: plt.Axes, *, include_border: bool = True) -> None:
    """Draw the exact no-TRR inference panel used by the teaser."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if include_border:
        section(ax, 0.008, 0.035, 0.984, 0.93, "INFERENCE: MIXED-RESOLUTION SAMPLING", COLORS["trajscale_q"])

    ax.text(0.025, 0.805, "Resolution schedules", transform=ax.transAxes, fontsize=7.5, fontweight="bold")
    ax.text(0.110, 0.720, "Structure & motion", transform=ax.transAxes, fontsize=6.5, color=COLORS["lr"], fontweight="bold")
    arrow(ax, (0.215, 0.727), (0.352, 0.727), color="#888888", width=0.7)
    ax.text(0.358, 0.720, "Texture & detail", transform=ax.transAxes, ha="right", fontsize=6.5, color=COLORS["hr"], fontweight="bold")
    schedule_bar(ax, 0.625, "Native-HR", 0.0)
    schedule_bar(ax, 0.490, "TrajScale-Q", 0.80)
    schedule_bar(ax, 0.355, "TrajScale-E", 0.90)
    for value, xpos in [("0", 0.108), ("40", 0.314), ("45", 0.340), ("50", 0.366)]:
        ax.text(xpos, 0.300, value, transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=COLORS["muted"])
    ax.text(0.237, 0.225, "Denoising evaluation", transform=ax.transAxes, ha="center", fontsize=6.3, color=COLORS["muted"])

    ax.plot([0.392, 0.392], [0.18, 0.88], transform=ax.transAxes, color=COLORS["border"], linewidth=0.75)
    ax.text(0.414, 0.805, "Resolution transition: two learned roles", transform=ax.transAxes, fontsize=7.5, fontweight="bold")

    y, h = 0.505, 0.145
    specs = [
        (0.414, 0.075, "LR prefix", COLORS["lr"], "#EAF2FA"),
        (0.513, 0.067, "EAA", COLORS["eaa"], "#F8EBF3"),
        (0.604, 0.092, "Endpoint-like\nclean LR", COLORS["lr"], "white"),
        (0.720, 0.067, "CRLU", COLORS["crlu"], "#E8F5EF"),
        (0.811, 0.088, "Target-grid\nclean HR", COLORS["crlu"], "white"),
        (0.925, 0.062, "Frozen\nHR suffix", COLORS["hr"], "#FBEDE7"),
    ]
    for x, w, label, edge, face in specs:
        box(ax, x, y, w, h, label, edge=edge, face=face, fontsize=6.3, weight="bold" if label in {"EAA", "CRLU"} else "normal")
    for current, nxt in zip(specs[:-1], specs[1:]):
        x, w = current[0], current[1]
        nx = nxt[0]
        arrow(ax, (x + w, y + h / 2), (nx - 0.004, y + h / 2), width=0.75)

    ax.text(0.546, 0.400, "deployment\ncalibration", transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=COLORS["eaa"], fontweight="bold")
    ax.text(0.753, 0.400, "learned scale\ntransfer", transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=COLORS["crlu"], fontweight="bold")
    ax.text(0.906, 0.400, "scheduler-consistent initialization\n(parameter-free)", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.956, 0.230, "native detail\nsynthesis", transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=COLORS["hr"], fontweight="bold")


def draw_supervision_panel(ax: plt.Axes) -> None:
    """Draw the two independent model-endogenous supervision streams."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    section(ax, 0.012, 0.535, 0.468, 0.425, "EAA: TRANSITION-LOCAL ENDPOINT SUPERVISION", COLORS["eaa"])
    section(ax, 0.500, 0.535, 0.488, 0.425, "CRLU: CROSS-RESOLUTION PAIR SUPERVISION", COLORS["crlu"])

    # EAA stream.
    box(ax, 0.035, 0.665, 0.095, 0.105, "Frozen Wan\nLR rollout", edge=COLORS["native"], face="#F0F2F3", fontsize=6.3)
    box(ax, 0.172, 0.770, 0.108, 0.080, r"Cached $x_s^L$", edge=COLORS["lr"], fontsize=6.5)
    box(ax, 0.172, 0.590, 0.108, 0.080, r"Endpoint $z_{end}^L$", edge=COLORS["native"], fontsize=6.2)
    box(ax, 0.335, 0.660, 0.105, 0.115, "EAA\ntraining pair", edge=COLORS["eaa"], face="#F8EBF3", fontsize=6.4, weight="bold")
    arrow(ax, (0.130, 0.718), (0.168, 0.810), color=COLORS["lr"])
    arrow(ax, (0.130, 0.718), (0.168, 0.630), color=COLORS["native"])
    arrow(ax, (0.280, 0.810), (0.331, 0.735), color=COLORS["lr"])
    arrow(ax, (0.280, 0.630), (0.331, 0.695), color=COLORS["native"])
    ax.text(0.242, 0.555, "same prompt, seed, scheduler, and CFG", transform=ax.transAxes, ha="center", fontsize=5.8, color=COLORS["muted"])

    # CRLU stream.
    box(ax, 0.520, 0.665, 0.078, 0.105, "Frozen\nWan", edge=COLORS["native"], face="#F0F2F3", fontsize=6.3)
    box(ax, 0.625, 0.665, 0.072, 0.105, "HR\nvideo", edge=COLORS["hr"], fontsize=6.4)
    box(ax, 0.733, 0.770, 0.075, 0.080, "Wan VAE", edge=COLORS["native"], fontsize=6.1)
    box(ax, 0.720, 0.585, 0.100, 0.085, "RGB downsample\n+ same Wan VAE", edge=COLORS["lr"], fontsize=5.7)
    box(ax, 0.854, 0.770, 0.050, 0.080, r"$z_{pair}^H$", edge=COLORS["hr"], fontsize=6.7)
    box(ax, 0.854, 0.585, 0.050, 0.085, r"$z_{pair}^L$", edge=COLORS["lr"], fontsize=6.7)
    box(ax, 0.930, 0.660, 0.045, 0.115, "CRLU\npair", edge=COLORS["crlu"], face="#E8F5EF", fontsize=6.0, weight="bold")
    arrow(ax, (0.598, 0.718), (0.621, 0.718), color=COLORS["native"])
    arrow(ax, (0.697, 0.718), (0.729, 0.810), color=COLORS["hr"])
    arrow(ax, (0.697, 0.718), (0.716, 0.628), color=COLORS["lr"])
    arrow(ax, (0.808, 0.810), (0.850, 0.810), color=COLORS["hr"])
    arrow(ax, (0.820, 0.628), (0.850, 0.628), color=COLORS["lr"])
    arrow(ax, (0.904, 0.810), (0.926, 0.735), color=COLORS["hr"])
    arrow(ax, (0.904, 0.628), (0.926, 0.695), color=COLORS["lr"])

    # Distribution bridge: precise about what EAA does and does not solve.
    section(ax, 0.012, 0.075, 0.976, 0.395, "DEPLOYMENT BRIDGE", COLORS["trajscale_q"])
    box(ax, 0.050, 0.205, 0.155, 0.115, "Intermediate clean prediction\n" + r"$p_{handoff}^{(s)}(\widehat z_s^L)$", edge=COLORS["lr"], fontsize=6.2)
    box(ax, 0.268, 0.205, 0.090, 0.115, "EAA", edge=COLORS["eaa"], face="#F8EBF3", fontsize=7.0, weight="bold")
    box(ax, 0.420, 0.205, 0.155, 0.115, "Endpoint-like LR domain\n" + r"$p_{end}(z_{end}^L)$", edge=COLORS["lr"], fontsize=6.2)
    box(ax, 0.638, 0.205, 0.090, 0.115, "CRLU", edge=COLORS["crlu"], face="#E8F5EF", fontsize=7.0, weight="bold")
    box(ax, 0.790, 0.205, 0.160, 0.115, "Target-grid clean latent\n" + r"$\widehat z_s^H$", edge=COLORS["crlu"], fontsize=6.4)
    for start, end in [((0.205, 0.262), (0.264, 0.262)), ((0.358, 0.262), (0.416, 0.262)), ((0.575, 0.262), (0.634, 0.262)), ((0.728, 0.262), (0.786, 0.262))]:
        arrow(ax, start, end)
    ax.text(0.313, 0.145, "reduces transition residual", transform=ax.transAxes, ha="center", fontsize=5.9, color=COLORS["eaa"], fontweight="bold")
    ax.text(0.683, 0.145, "learns the VAE-induced scale mapping", transform=ax.transAxes, ha="center", fontsize=5.9, color=COLORS["crlu"], fontweight="bold")
    ax.text(0.500, 0.095, "EAA does not perform upsampling; CRLU does not predict the unfinished LR trajectory.", transform=ax.transAxes, ha="center", fontsize=6.2, color=COLORS["muted"], fontweight="bold")


def main() -> None:
    apply_publication_style()
    fig = plt.figure(figsize=(7.0, 3.55), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.82, 1.18], hspace=0.035, left=0.015, right=0.995, bottom=0.015, top=0.995)
    draw_inference_panel(fig.add_subplot(grid[0]), include_border=True)
    draw_supervision_panel(fig.add_subplot(grid[1]))
    pdf_path, png_path = save_figure(fig, "fig_trajscale_overview", png_dpi=400)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
