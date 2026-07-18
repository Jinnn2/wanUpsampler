#!/usr/bin/env python3
"""Generate the TALH teaser from audited experiment videos and the overview.

The visual panel uses prompt 05 / seed 9705 / frame 41 and Crop A from the
existing qualitative audit. Video pixels are decoded directly from the MP4
artifacts and are never generatively edited. The mechanism panel reuses the
inference half of the selected TALH overview without redrawing its content.

Outputs:
    fig_teaser.pdf
    fig_teaser.png
    fig_teaser_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from _figure_style import COLORS, apply_publication_style, save_figure
from gen_fig_qualitative import (
    SPATIAL_CROPS,
    SPATIAL_FRAME,
    VIDEO_ROOT,
    add_crop_box,
    crop,
    decode_frames,
    find_video,
    sha256,
    show_image,
)


OUTPUT_STEM = "fig_teaser"
FIGURE_DIR = Path(__file__).resolve().parent
OVERVIEW_PATH = FIGURE_DIR / "fig_talh_overview.png"
# Exact upper inference panel in the locked 1774 x 887 overview rendering.
OVERVIEW_CROP_BOX = (0, 0, 1774, 409)
SPATIAL_DIR = VIDEO_ROOT / "TALH-Q_spatial_detail" / "prompt_05_seed9705"
CROP_NAME = "A"
VIDEO_SPECS = (
    ("Native-HR (estimated)", "Native-HR", COLORS["native"], True),
    ("Trilinear @ 40", "Trilinear-at-40", COLORS["baseline"], False),
    ("CLL-only @ 40", "CLL-only-at-40", COLORS["cll"], False),
    ("TALH-Q @ 40", "TALH-Q-at-40", COLORS["talh_q"], False),
)


def frame_sha256(image: Image.Image) -> str:
    """Hash decoded RGB pixels, independent of the source container."""

    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def add_method_accent(ax: plt.Axes, color: str) -> None:
    """Add a thin method-color rule without covering video content."""

    ax.add_patch(
        Rectangle(
            (0.0, 0.985),
            1.0,
            0.015,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
    )


def main() -> None:
    apply_publication_style()

    visual_data: list[tuple[str, str, bool, Path, Image.Image]] = []
    manifest_videos: list[dict[str, object]] = []
    for label, token, color, estimated in VIDEO_SPECS:
        path = find_video(SPATIAL_DIR, token)
        image = decode_frames(path, {SPATIAL_FRAME})[SPATIAL_FRAME]
        visual_data.append((label, color, estimated, path, image))
        crop_image = crop(image, SPATIAL_CROPS[CROP_NAME])
        manifest_videos.append(
            {
                "method": label.replace("\n", " "),
                "path": str(path),
                "video_sha256": sha256(path),
                "decoded_frame_sha256": frame_sha256(image),
                "displayed_crop_sha256": frame_sha256(crop_image),
            }
        )

    overview = Image.open(OVERVIEW_PATH).convert("RGB")
    if overview.size != (OVERVIEW_CROP_BOX[2], 887):
        raise ValueError(f"Unexpected locked overview size: {overview.size}")
    overview_inference = overview.crop(OVERVIEW_CROP_BOX)

    fig = plt.figure(figsize=(7.0, 3.12), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.030,
        right=0.995,
        bottom=0.018,
        top=0.885,
        height_ratios=[1.0, 1.54],
        hspace=0.170,
    )
    visual_grid = outer[0].subgridspec(1, 4, wspace=0.028)

    fig.text(0.008, 0.978, "(a)  Matched real video frames — prompt 05, seed 9705, frame 41", ha="left", va="top", fontsize=8.5, fontweight="bold")

    for column, (label, color, estimated, _path, image) in enumerate(visual_data):
        full_ax = fig.add_subplot(visual_grid[0, column])
        show_image(full_ax, image)
        full_ax.set_anchor("N")
        full_ax.set_title(label, fontsize=7.0, fontweight="bold", color=color, pad=3.0, linespacing=0.90)
        add_method_accent(full_ax, color)
        add_crop_box(full_ax, image, SPATIAL_CROPS[CROP_NAME], COLORS["cll"])
        crop_ax = full_ax.inset_axes([0.535, 0.025, 0.440, 0.760])
        show_image(crop_ax, crop(image, SPATIAL_CROPS[CROP_NAME]))
        for spine in crop_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(COLORS["cll"])
            spine.set_linewidth(0.9)
        crop_ax.text(0.03, 0.96, "Crop A", transform=crop_ax.transAxes, ha="left", va="top", fontsize=7.0, fontweight="bold", color=COLORS["cll"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0})

    mechanism_ax = fig.add_subplot(outer[1])
    show_image(mechanism_ax, overview_inference)
    mechanism_ax.set_anchor("N")
    mechanism_ax.text(
        -0.024,
        0.995,
        "(b)",
        transform=mechanism_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )

    pdf_path, png_path = save_figure(fig, OUTPUT_STEM, png_dpi=400)
    plt.close(fig)

    manifest = {
        "source": "Audited TALH qualitative videos plus the locked inference panel from fig_talh_overview.png.",
        "layout": "Panel (a) above panel (b).",
        "prompt_index": 5,
        "seed": 9705,
        "frame_zero_based": SPATIAL_FRAME,
        "frame_displayed_one_based": SPATIAL_FRAME + 1,
        "crop_name": CROP_NAME,
        "crop_box_in_1248x720": SPATIAL_CROPS[CROP_NAME],
        "native_hr_estimated_note": (
            "Content-aligned 640x368 full LR trajectory used for context only; "
            "its actual frame and normalized Crop A are displayed; it is not used as a 720p sharpness baseline."
        ),
        "overview_inference_panel": {
            "path": str(OVERVIEW_PATH),
            "source_sha256": sha256(OVERVIEW_PATH),
            "crop_box": OVERVIEW_CROP_BOX,
            "displayed_crop_sha256": frame_sha256(overview_inference),
        },
        "operating_points": {
            "TALH-Q": {"handoff_step": 40, "lr_hr_evaluations": [40, 10], "speedup": 1.83, "vbench5_retained_percent": 97.76},
            "TALH-E": {"handoff_step": 45, "lr_hr_evaluations": [45, 5], "speedup": 2.22, "vbench5_retained_percent": 97.53},
        },
        "videos": manifest_videos,
    }
    manifest_path = Path(__file__).resolve().parent / f"{OUTPUT_STEM}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
