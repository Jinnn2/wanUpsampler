#!/usr/bin/env python3
"""Relabel the first Figure 3 column as the completed LR endpoint.

``fig_challenge_interpolation_source.pdf`` preserves the original 990 x 462
raster. The legacy ``Native-HR`` label described a 640 x 360 video produced by
the completed low-resolution trajectory. This script changes only that label;
all decoded frames, crops, borders, and the other two columns remain byte-for-
byte sourced from the original embedded JPEG.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE_PDF = HERE / "fig_challenge_interpolation_source.pdf"
DEFAULT_OUTPUT = HERE.parent / "figures" / "fig_challenge_interpolation.pdf"

TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
WHITE = (255, 255, 255)
TEXT = (32, 33, 36)


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
    """Extract the single embedded raster and retain the PDF page size."""

    with fitz.open(source_pdf) as document:
        page = document[0]
        images = page.get_images(full=True)
        if len(images) != 1:
            raise ValueError(f"Expected one embedded raster, found {len(images)}")
        extracted = document.extract_image(images[0][0])
        source_bytes = extracted["image"]
        image = Image.open(BytesIO(source_bytes)).convert("RGB")
        page_rect = fitz.Rect(page.rect)

    if image.size != (990, 462):
        raise ValueError(f"Unexpected Figure 3 raster size: {image.size}")
    return image, source_bytes, page_rect


def revise_label(image: Image.Image) -> Image.Image:
    """Replace only the legacy first-column label."""

    revised = image.copy()
    draw = ImageDraw.Draw(revised)
    label_bounds = (18, 18, 328, 56)
    draw.rectangle(label_bounds, fill=WHITE)
    font = ImageFont.truetype(str(TIMES_BOLD), 29)
    centered_text(draw, label_bounds, "LR endpoint", font, TEXT)
    return revised


def save_pdf(
    source_jpeg: bytes,
    revised: Image.Image,
    page_rect: fitz.Rect,
    output: Path,
) -> None:
    """Preserve the source JPEG and overlay only the first label region."""

    overlay_bounds = (18, 18, 328, 56)
    overlay = revised.crop(overlay_bounds)
    overlay_png = BytesIO()
    overlay.save(overlay_png, format="PNG", optimize=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=page_rect.width, height=page_rect.height)
    page.insert_image(page.rect, stream=source_jpeg)
    x_scale = page_rect.width / revised.width
    y_scale = page_rect.height / revised.height
    overlay_rect = fitz.Rect(
        overlay_bounds[0] * x_scale,
        overlay_bounds[1] * y_scale,
        overlay_bounds[2] * x_scale,
        overlay_bounds[3] * y_scale,
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
    revised = revise_label(image)
    save_pdf(source_jpeg, revised, page_rect, args.output)
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        revised.save(args.preview)
    print(f"Saved: {args.output}")
    if args.preview is not None:
        print(f"Saved: {args.preview}")


if __name__ == "__main__":
    main()
