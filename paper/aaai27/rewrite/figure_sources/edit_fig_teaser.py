#!/usr/bin/env python3
"""Apply the LR-endpoint semantic-label revision to the raster teaser figure.

The original 2930 x 970 raster is preserved in ``fig_teaser_source.pdf``.
This script changes only the left annotation region:

1. Rename the completed 50-step low-resolution row to ``LR endpoint``.
2. Remove the vertical speedup arrow.
3. Add ``4.44× speedup over Native-HR`` below the InTraScale label.

All decoded-video pixels, temporal frames, and crop regions remain unchanged.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE_PDF = HERE / "fig_teaser_source.pdf"
DEFAULT_OUTPUT = HERE.parent / "figures" / "fig_teaser.pdf"

TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
WHITE = (255, 255, 255)
REFERENCE = (101, 121, 130)
INTRASCALE = (0, 151, 111)


def centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    """Draw one text line centered inside a pixel-space bounding box."""

    left, top, right, bottom = bounds
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - text_box[1]
    draw.text((x, y), text, font=font, fill=fill)


def extract_single_raster(
    source_pdf: Path,
) -> tuple[Image.Image, bytes, fitz.Rect]:
    """Extract the teaser's single embedded raster and retain its PDF page size."""

    with fitz.open(source_pdf) as document:
        page = document[0]
        images = page.get_images(full=True)
        if len(images) != 1:
            raise ValueError(f"Expected one embedded raster, found {len(images)}")
        extracted = document.extract_image(images[0][0])
        source_bytes = extracted["image"]
        image = Image.open(BytesIO(source_bytes)).convert("RGB")
        page_rect = fitz.Rect(page.rect)

    if image.size != (2930, 970):
        raise ValueError(f"Unexpected teaser raster size: {image.size}")
    return image, source_bytes, page_rect


def revise_labels(image: Image.Image) -> Image.Image:
    """Redraw only the approved label and speedup-annotation regions."""

    revised = image.copy()
    draw = ImageDraw.Draw(revised)

    # Remove the original vertical arrow and its rotated "4x+ speedup" badge.
    draw.rectangle((0, 165, 126, 842), fill=WHITE)

    # The first row is the completed 50-step low-resolution trajectory.
    # Replace the legacy proxy/reference wording with its semantic role.
    draw.rectangle((122, 172, 466, 235), fill=WHITE)
    reference_font = ImageFont.truetype(str(TIMES_BOLD), 31)
    centered_text(
        draw,
        (122, 174, 466, 231),
        "LR endpoint",
        reference_font,
        REFERENCE,
    )

    # Place the measured speedup directly below the InTraScale evaluation
    # count, without a surrounding badge or frame.
    speed_font = ImageFont.truetype(str(TIMES_BOLD), 24)
    centered_text(
        draw,
        (122, 851, 466, 901),
        "4.44× speedup over Native-HR",
        speed_font,
        INTRASCALE,
    )
    return revised


def save_pdf(
    source_jpeg: bytes,
    revised: Image.Image,
    page_rect: fitz.Rect,
    output: Path,
) -> None:
    """Preserve the source JPEG and overlay only the left annotation region."""

    # The first video frame starts at x=469. Restrict the lossless overlay to
    # x<=466 so every decoded-video and crop pixel remains sourced directly
    # from the original embedded JPEG bytes.
    overlay_right = 467
    overlay = revised.crop((0, 0, overlay_right, revised.height))
    overlay_png = BytesIO()
    overlay.save(overlay_png, format="PNG", optimize=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=page_rect.width, height=page_rect.height)
    page.insert_image(page.rect, stream=source_jpeg)
    overlay_rect = fitz.Rect(
        0,
        0,
        page_rect.width * overlay_right / revised.width,
        page_rect.height,
    )
    page.insert_image(overlay_rect, stream=overlay_png.getvalue())
    document.save(output, garbage=4, deflate=True)
    document.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, source_jpeg, page_rect = extract_single_raster(args.source)
    revised = revise_labels(image)
    save_pdf(source_jpeg, revised, page_rect, args.output)
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        revised.save(args.preview)
    print(f"Saved: {args.output}")
    if args.preview is not None:
        print(f"Saved: {args.preview}")


if __name__ == "__main__":
    main()
