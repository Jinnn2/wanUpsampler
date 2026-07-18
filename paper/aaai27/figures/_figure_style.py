#!/usr/bin/env python3
"""Shared AAAI figure styling for the TALH paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[2]
TABLE_DIR = REPO_ROOT / "paper" / "aaai27" / "results" / "integrated_20260718" / "compiled_tables"

COLORS = {
    "native": "#7B8794",
    "lr": "#4A90D9",
    "taa": "#CC79A7",
    "cll": "#009E73",
    "hr": "#D55E00",
    "talh_q": "#0072B2",
    "talh_e": "#E69F00",
    "baseline": "#B0B0B0",
    "endpoint": "#5F6B73",
    "text": "#242424",
    "muted": "#666666",
    "grid": "#D9D9D9",
    "band": "#F7F7F5",
    "border": "#D7D7D7",
}


def apply_publication_style() -> None:
    """Configure compact, Times-compatible styling for AAAI figures."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.grid": True,
            "grid.alpha": 0.45,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "lines.linewidth": 1.5,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter})  {title}", loc="left", pad=5.0, fontweight="bold")


def save_figure(fig: plt.Figure, stem: str, *, png_dpi: int = 300) -> tuple[Path, Path]:
    """Save a figure as vector PDF and high-resolution PNG."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=png_dpi, facecolor="white")
    return pdf_path, png_path
