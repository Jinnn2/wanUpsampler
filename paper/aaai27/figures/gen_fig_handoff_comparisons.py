#!/usr/bin/env python3
"""Generate the two single-column handoff challenge comparisons.

The script uses the Windows WPF extractor next to this file, so it does not
require OpenCV or ffmpeg. Pillow is the only Python dependency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTOR = SCRIPT_DIR / "extract_video_frame_wpf.ps1"
EDGE_EXTRACTOR_PAGE = SCRIPT_DIR / "extract_video_last_frame_edge.html"
WORK_DIR = SCRIPT_DIR / "_challenge_work"
FRAME_DIR = WORK_DIR / "frames"
EDGE_PROFILE_DIR = WORK_DIR / "edge_profiles"

EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
)

CANVAS_WIDTH = 990
OUTER_MARGIN = 18
COLUMN_GAP = 10
CELL_WIDTH = (CANVAS_WIDTH - 2 * OUTER_MARGIN - 2 * COLUMN_GAP) // 3
CELL_HEIGHT = round(CELL_WIDTH * 9 / 16)
LABEL_HEIGHT = 38
ROW_LABEL_HEIGHT = 28
ROW_GAP = 10
CANVAS_HEIGHT = (
    OUTER_MARGIN
    + LABEL_HEIGHT
    + CELL_HEIGHT
    + ROW_GAP
    + ROW_LABEL_HEIGHT
    + CELL_HEIGHT
    + OUTER_MARGIN
)

BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#202124"
CELL_BORDER = "#D5D7DA"
CROP_COLOR = "#E69F00"

CONTACT_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class VideoEntry:
    key: str
    label: str
    path: Path


@dataclass(frozen=True)
class FigureGroup:
    key: str
    crop_position: str
    magnification: int
    output_stem: str
    videos: tuple[VideoEntry, VideoEntry, VideoEntry]


def load_font(bold: bool, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [Path(r"C:\Windows\Fonts\timesbd.ttf"), Path(r"C:\Windows\Fonts\cambria.ttc")]
        if bold
        else [Path(r"C:\Windows\Fonts\times.ttf"), Path(r"C:\Windows\Fonts\cambria.ttc")]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


LABEL_FONT = load_font(bold=True, size=29)
ROW_FONT = load_font(bold=False, size=23)
CONTACT_FONT = load_font(bold=True, size=24)
CONTACT_SMALL_FONT = load_font(bold=False, size=20)


def groups(source_root: Path) -> tuple[FigureGroup, FigureGroup]:
    downloaded = source_root / "drive-download-20260723T153420Z-1-001"
    interpolation = FigureGroup(
        key="interpolation",
        crop_position="center",
        magnification=3,
        output_stem="fig_challenge_interpolation",
        videos=(
            VideoEntry("native_hr", "Native-HR", downloaded / "01_Native-HR.mp4"),
            VideoEntry(
                "trilinear",
                "Trilinear",
                downloaded / "02_Trilinear-at-45.mp4",
            ),
            VideoEntry(
                "intrascale",
                "InTraScale",
                downloaded / "04_TALH-E-at-45.mp4",
            ),
        ),
    )
    alignment = FigureGroup(
        key="alignment",
        crop_position="bottom_left",
        magnification=2,
        output_stem="fig_challenge_alignment",
        videos=(
            VideoEntry(
                "lr_endpoint",
                "LR endpoint",
                source_root / "ori_50_02_seed9702.mp4",
            ),
            VideoEntry(
                "unaligned",
                "Unaligned",
                source_root / "ori_45_02_seed9702.mp4",
            ),
            VideoEntry(
                "ttda_aligned",
                "TTDA-aligned",
                source_root / "lora_45_02_seed9702.mp4",
            ),
        ),
    )
    return interpolation, alignment


def validate_sources(figure_groups: tuple[FigureGroup, ...]) -> None:
    missing = [
        str(video.path)
        for group in figure_groups
        for video in group.videos
        if not video.path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing source videos:\n" + "\n".join(missing))


def frame_path(group: FigureGroup, video: VideoEntry, fraction: float) -> Path:
    fraction_tag = (
        "last_edge"
        if fraction >= 0.999
        else f"{fraction:.4f}".replace(".", "p")
    )
    return FRAME_DIR / group.key / f"{video.key}_{fraction_tag}.png"


def find_edge() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Microsoft Edge or Google Chrome is required")


def extract_last_frame_edge(
    group: FigureGroup,
    video: VideoEntry,
    output: Path,
) -> Path:
    edge = find_edge()
    profile = EDGE_PROFILE_DIR / group.key / video.key
    profile.mkdir(parents=True, exist_ok=True)
    source_uri = quote(video.path.resolve().as_uri(), safe="")
    page_url = f"{EDGE_EXTRACTOR_PAGE.resolve().as_uri()}?src={source_uri}"
    command = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--allow-file-access-from-files",
        "--autoplay-policy=no-user-gesture-required",
        "--force-device-scale-factor=1",
        "--window-size=640,368",
        "--virtual-time-budget=8000",
        "--run-all-compositor-stages-before-draw",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        f"--screenshot={output}",
        page_url,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    deadline = time.monotonic() + 10.0
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not output.exists():
        raise RuntimeError(
            "Browser did not produce screenshot: "
            f"{output}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    print(f"Captured final frame with Chromium: {output}")
    return output


def extract_frame(group: FigureGroup, video: VideoEntry, fraction: float) -> Path:
    output = frame_path(group, video, fraction)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if fraction >= 0.999:
        return extract_last_frame_edge(group, video, output)

    command = [
        "powershell.exe",
        "-NoProfile",
        "-STA",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(EXTRACTOR),
        "-InputVideo",
        str(video.path),
        "-OutputPng",
        str(output),
    ]
    command.extend(["-Fraction", f"{fraction:.8f}"])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    return output


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = xy
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2 - box[1]
    draw.text((x, y), text, font=font, fill=fill)


def normalized_frame(image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"),
        (image.width, round(image.width * 9 / 16)),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def crop_box(
    image: Image.Image,
    position: str,
    magnification: int,
) -> tuple[int, int, int, int]:
    crop_width = image.width // magnification
    crop_height = image.height // magnification
    if position == "center":
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
    elif position == "bottom_left":
        left = 0
        top = image.height - crop_height
    else:
        raise ValueError(f"Unknown crop position: {position}")
    return left, top, left + crop_width, top + crop_height


def add_cell_border(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], width: int = 2
) -> None:
    draw.rectangle(box, outline=CELL_BORDER, width=width)


def render_final(group: FigureGroup, fraction: float) -> dict[str, object]:
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    full_top = OUTER_MARGIN + LABEL_HEIGHT
    crop_label_top = full_top + CELL_HEIGHT + ROW_GAP
    crop_top = crop_label_top + ROW_LABEL_HEIGHT

    text_center(
        draw,
        (0, crop_label_top, CANVAS_WIDTH, crop_top),
        f"{group.magnification}× crop",
        ROW_FONT,
        TEXT_COLOR,
    )

    manifest_videos: list[dict[str, object]] = []
    for column, video in enumerate(group.videos):
        source_frame = extract_frame(group, video, fraction)
        with Image.open(source_frame) as loaded:
            frame = normalized_frame(loaded)

        left = OUTER_MARGIN + column * (CELL_WIDTH + COLUMN_GAP)
        right = left + CELL_WIDTH
        label_box = (left, OUTER_MARGIN, right, full_top)
        text_center(draw, label_box, video.label, LABEL_FONT, TEXT_COLOR)

        roi = crop_box(frame, group.crop_position, group.magnification)
        full = frame.resize(
            (CELL_WIDTH, CELL_HEIGHT),
            resample=Image.Resampling.LANCZOS,
        )
        full_draw = ImageDraw.Draw(full)
        scale_x = CELL_WIDTH / frame.width
        scale_y = CELL_HEIGHT / frame.height
        display_roi = (
            round(roi[0] * scale_x),
            round(roi[1] * scale_y),
            round(roi[2] * scale_x),
            round(roi[3] * scale_y),
        )
        full_draw.rectangle(display_roi, outline=CROP_COLOR, width=5)
        canvas.paste(full, (left, full_top))
        add_cell_border(
            draw,
            (left, full_top, right - 1, full_top + CELL_HEIGHT - 1),
        )

        crop = frame.crop(roi).resize(
            (CELL_WIDTH, CELL_HEIGHT),
            resample=Image.Resampling.LANCZOS,
        )
        canvas.paste(crop, (left, crop_top))
        add_cell_border(
            draw,
            (left, crop_top, right - 1, crop_top + CELL_HEIGHT - 1),
        )

        manifest_videos.append(
            {
                "key": video.key,
                "label": video.label,
                "video": str(video.path),
                "frame": str(source_frame),
                "source_size": [frame.width, frame.height],
                "crop_box": list(roi),
            }
        )

    png_path = SCRIPT_DIR / f"{group.output_stem}.png"
    pdf_path = SCRIPT_DIR / f"{group.output_stem}.pdf"
    canvas.save(png_path, dpi=(300, 300), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=300.0)

    manifest = {
        "group": group.key,
        "fraction": fraction,
        "frame_selection": "last_edge" if fraction >= 0.999 else "fraction",
        "crop_position": group.crop_position,
        "linear_magnification": group.magnification,
        "canvas_pixels": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "png": str(png_path),
        "pdf": str(pdf_path),
        "videos": manifest_videos,
    }
    manifest_path = SCRIPT_DIR / f"{group.output_stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {manifest_path}")
    return manifest


def render_contact_sheet(group: FigureGroup) -> Path:
    thumb_width = 300
    thumb_height = round(thumb_width * 9 / 16)
    label_width = 190
    header_height = 42
    row_gap = 8
    column_gap = 8
    width = (
        label_width
        + len(CONTACT_FRACTIONS) * thumb_width
        + (len(CONTACT_FRACTIONS) - 1) * column_gap
    )
    height = (
        header_height
        + len(group.videos) * thumb_height
        + (len(group.videos) - 1) * row_gap
    )
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for index, fraction in enumerate(CONTACT_FRACTIONS):
        left = label_width + index * (thumb_width + column_gap)
        text_center(
            draw,
            (left, 0, left + thumb_width, header_height),
            f"{fraction:.1f}",
            CONTACT_SMALL_FONT,
            TEXT_COLOR,
        )

    for row, video in enumerate(group.videos):
        top = header_height + row * (thumb_height + row_gap)
        text_center(
            draw,
            (0, top, label_width - 8, top + thumb_height),
            video.label,
            CONTACT_FONT,
            TEXT_COLOR,
        )
        for column, fraction in enumerate(CONTACT_FRACTIONS):
            source_frame = extract_frame(group, video, fraction)
            with Image.open(source_frame) as loaded:
                frame = normalized_frame(loaded)
            thumb = frame.resize(
                (thumb_width, thumb_height),
                resample=Image.Resampling.LANCZOS,
            )
            left = label_width + column * (thumb_width + column_gap)
            sheet.paste(thumb, (left, top))
            add_cell_border(
                draw,
                (left, top, left + thumb_width - 1, top + thumb_height - 1),
            )

    output = WORK_DIR / f"contact_{group.key}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    print(f"Saved {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:\Downloads\AAAI27"),
    )
    parser.add_argument(
        "--mode",
        choices=("contact", "final", "all"),
        default="all",
    )
    parser.add_argument(
        "--interpolation-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--alignment-fraction",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_groups = groups(args.source_root)
    validate_sources(figure_groups)
    interpolation, alignment = figure_groups

    if args.mode in {"contact", "all"}:
        render_contact_sheet(interpolation)
        render_contact_sheet(alignment)
    if args.mode in {"final", "all"}:
        render_final(interpolation, args.interpolation_fraction)
        render_final(alignment, args.alignment_fraction)


if __name__ == "__main__":
    main()
