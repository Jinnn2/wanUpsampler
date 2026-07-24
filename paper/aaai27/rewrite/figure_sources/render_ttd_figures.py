#!/usr/bin/env python3
"""Render the canonical figures whose labels contain TTD.

The templates deliberately contain no terminology label. This keeps the
working sources reusable without requiring the external videos used by the
original alignment-figure generator.
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path(__file__).resolve().parent
REWRITE_DIR = SOURCE_DIR.parent
FIGURE_DIR = REWRITE_DIR / "figures"
TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
TTD_LABEL = "TTD"


def render_overall_framework() -> Path:
    template = (
        SOURCE_DIR
        / "overall_framework"
        / "fig_overall_framework_template.pdf"
    )
    output = FIGURE_DIR / "fig_overall_framework.pdf"
    temporary = output.with_suffix(".tmp.pdf")

    document = fitz.open(template)
    page = document[0]
    font = fitz.Font(fontfile=str(TIMES_BOLD))
    labels = (
        {
            "box": fitz.Rect(268.5, 47.9, 287.0, 55.6),
            "baseline": 53.922016,
            "size": 6.4125,
            "color": (17 / 255, 17 / 255, 17 / 255),
        },
        {
            "box": fitz.Rect(158.0, 93.9, 178.8, 102.5),
            "baseline": 100.656395,
            "size": 7.215,
            "color": (193 / 255, 59 / 255, 131 / 255),
        },
    )
    for label in labels:
        box = label["box"]
        size = label["size"]
        width = font.text_length(TTD_LABEL, fontsize=size)
        origin = ((box.x0 + box.x1 - width) / 2, label["baseline"])
        page.insert_text(
            origin,
            TTD_LABEL,
            fontsize=size,
            fontname="TimesNewRomanBold",
            fontfile=str(TIMES_BOLD),
            color=label["color"],
            overlay=True,
        )

    if temporary.exists():
        temporary.unlink()
    document.save(temporary, garbage=4, deflate=True)
    document.close()
    os.replace(temporary, output)
    return output


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill="#202124")


def render_challenge_alignment() -> Path:
    template = (
        SOURCE_DIR
        / "challenge_alignment"
        / "fig_challenge_alignment_template.png"
    )
    output = FIGURE_DIR / "fig_challenge_alignment.pdf"
    temporary = output.with_suffix(".tmp.pdf")

    with Image.open(template) as loaded:
        image = loaded.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(TIMES_BOLD), size=29)
    centered_text(draw, (660, 18, 971, 56), "TTD-aligned", font)

    if temporary.exists():
        temporary.unlink()
    image.save(temporary, "PDF", resolution=300.0)
    os.replace(temporary, output)
    return output


def main() -> None:
    if not TIMES_BOLD.exists():
        raise FileNotFoundError(f"Required font not found: {TIMES_BOLD}")
    print(f"Rendered {render_overall_framework()}")
    print(f"Rendered {render_challenge_alignment()}")


if __name__ == "__main__":
    main()
