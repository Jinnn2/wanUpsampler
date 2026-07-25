#!/usr/bin/env python3
"""Render the canonical figures whose labels contain TTD.

The templates deliberately contain no terminology label. This keeps the
working sources reusable without requiring the external videos used by the
original alignment-figure generator.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path(__file__).resolve().parent
REWRITE_DIR = SOURCE_DIR.parent
FIGURE_DIR = REWRITE_DIR / "figures"
TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
TIMES_REGULAR = Path(r"C:\Windows\Fonts\times.ttf")
TTD_LABEL = "TTD"
TEXT_OBJECT = re.compile(rb"BT(?P<body>.*?)ET", re.DOTALL)


def remove_replaced_labels(document: fitz.Document, page: fitz.Page) -> None:
    """Remove template labels that are replaced with canonical terminology."""

    replaced_objects = {34, 38, 41, 56, 57}
    text_object_index = 0
    for xref in page.get_contents():
        stream = document.xref_stream(xref)

        def keep_or_remove(match: re.Match[bytes]) -> bytes:
            nonlocal text_object_index
            current = text_object_index
            text_object_index += 1
            return b"" if current in replaced_objects else match.group(0)

        updated = TEXT_OBJECT.sub(keep_or_remove, stream)
        if updated != stream:
            document.update_stream(xref, updated)

    if text_object_index != 73:
        raise RuntimeError(
            "Unexpected overall-framework template text-object count: "
            f"{text_object_index}"
        )


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
    remove_replaced_labels(document, page)
    bold_font = fitz.Font(fontfile=str(TIMES_BOLD))
    regular_font = fitz.Font(fontfile=str(TIMES_REGULAR))
    labels = (
        {
            "text": "steps 1,\u2026,s\u22121",
            "center_x": 183.2603,
            "baseline": 40.4103,
            "size": 6.4125,
            "color": (39 / 255, 112 / 255, 186 / 255),
            "font": regular_font,
            "fontname": "TimesNewRoman",
            "fontfile": TIMES_REGULAR,
        },
        {
            "text": "step s",
            "center_x": 306.5201,
            "baseline": 33.0744,
            "size": 6.4125,
            "color": (193 / 255, 59 / 255, 131 / 255),
            "font": regular_font,
            "fontname": "TimesNewRoman",
            "fontfile": TIMES_REGULAR,
        },
        {
            "text": "steps s+1,\u2026,T",
            "center_x": 380.2802,
            "baseline": 40.4103,
            "size": 6.4125,
            "color": (220 / 255, 91 / 255, 13 / 255),
            "font": regular_font,
            "fontname": "TimesNewRoman",
            "fontfile": TIMES_REGULAR,
        },
        {
            "text": "active only at step s",
            "center_x": 188.9894,
            "baseline": 158.3830,
            "size": 5.61,
            "color": (193 / 255, 59 / 255, 131 / 255),
            "font": regular_font,
            "fontname": "TimesNewRoman",
            "fontfile": TIMES_REGULAR,
        },
        {
            "text": "TTD",
            "center_x": 277.75,
            "baseline": 53.922016,
            "size": 6.4125,
            "color": (17 / 255, 17 / 255, 17 / 255),
            "font": bold_font,
            "fontname": "TimesNewRomanBold",
            "fontfile": TIMES_BOLD,
        },
        {
            "text": "Trajectory-Tail Distillation (TTD)",
            "center_x": 168.4,
            "baseline": 100.656395,
            "size": 7.215,
            "color": (193 / 255, 59 / 255, 131 / 255),
            "font": bold_font,
            "fontname": "TimesNewRomanBold",
            "fontfile": TIMES_BOLD,
        },
        {
            "text": "In-Trajectory Upsampler (ITU)",
            "center_x": 295.5,
            "baseline": 100.656395,
            "size": 7.215,
            "color": (22 / 255, 128 / 255, 79 / 255),
            "font": bold_font,
            "fontname": "TimesNewRomanBold",
            "fontfile": TIMES_BOLD,
        },
    )
    for label in labels:
        text = label["text"]
        size = label["size"]
        width = label["font"].text_length(text, fontsize=size)
        origin = (label["center_x"] - width / 2, label["baseline"])
        page.insert_text(
            origin,
            text,
            fontsize=size,
            fontname=label["fontname"],
            fontfile=str(label["fontfile"]),
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
    centered_text(draw, (660, 18, 971, 56), "TTD-calibrated", font)

    if temporary.exists():
        temporary.unlink()
    image.save(temporary, "PDF", resolution=300.0)
    os.replace(temporary, output)
    return output


def main() -> None:
    for font_path in (TIMES_BOLD, TIMES_REGULAR):
        if not font_path.exists():
            raise FileNotFoundError(f"Required font not found: {font_path}")
    print(f"Rendered {render_overall_framework()}")
    print(f"Rendered {render_challenge_alignment()}")


if __name__ == "__main__":
    main()
