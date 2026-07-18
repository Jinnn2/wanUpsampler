#!/usr/bin/env python3
"""Audit TALH qualitative videos and build deterministic candidate contact sheets.

This script prepares review artifacts only. It does not select the paper example or
produce the final publication figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import textwrap
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


FRAME_INDICES = (0, 20, 40, 60, 80)
EXPECTED_SIZE = (1248, 720)  # width, height
ESTIMATED_REFERENCE_SIZE = (640, 368)  # content-aligned full LR trajectory
EXPECTED_FRAMES = 81
EXPECTED_FPS = 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def parse_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_selected_frames(path: Path) -> tuple[dict[str, object], dict[int, Image.Image], int]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    selected: dict[int, Image.Image] = {}
    frame_count = 0
    try:
        for frame_index, frame_bytes in enumerate(reader):
            frame_count = frame_index + 1
            if frame_index in FRAME_INDICES:
                selected[frame_index] = Image.frombytes("RGB", (width, height), frame_bytes)
    finally:
        reader.close()
    return metadata, selected, frame_count


def method_label(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    if "native-hr" in stem.lower() or "native_hr" in stem.lower():
        return "Native-HR (estimated)"
    return stem.replace("-at-", " @ ").replace("-", " ")


def is_estimated_reference(path: Path) -> bool:
    normalized = path.stem.lower().replace("_", "-")
    return "native-hr" in normalized


def build_contact_sheet(
    group_name: str,
    metadata: dict[str, str],
    rows: list[tuple[str, dict[int, Image.Image]]],
    output_path: Path,
) -> None:
    thumb_width = 288
    thumb_height = round(thumb_width * EXPECTED_SIZE[1] / EXPECTED_SIZE[0])
    label_width = 270
    header_height = 138
    sheet_width = label_width + len(FRAME_INDICES) * thumb_width
    sheet_height = header_height + len(rows) * thumb_height

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(28, bold=True)
    body_font = load_font(20)
    label_font = load_font(22, bold=True)
    small_font = load_font(18)

    title = f"{group_name}: prompt {metadata.get('prompt_index', '?')}, seed {metadata.get('seed', '?')}"
    draw.text((12, 8), title, fill="#222222", font=title_font)
    prompt = metadata.get("prompt", "")
    prompt_lines = textwrap.wrap(prompt, width=125)[:2]
    draw.multiline_text((12, 42), "\n".join(prompt_lines), fill="#444444", font=body_font, spacing=2)
    draw.text(
        (12, 94),
        "Native-HR (estimated): content-aligned 368p proxy; all other rows: 720p outputs.",
        fill="#6B6B6B",
        font=small_font,
    )

    for column, frame_index in enumerate(FRAME_INDICES):
        x = label_width + column * thumb_width + thumb_width // 2
        draw.text(
            (x, header_height - 24),
            f"Frame {frame_index + 1}",
            fill="#444444",
            font=small_font,
            anchor="mm",
        )

    for row_index, (label, frames) in enumerate(rows):
        y = header_height + row_index * thumb_height
        if row_index % 2:
            draw.rectangle((0, y, sheet_width, y + thumb_height), fill="#F7F7F5")
        draw.text((12, y + thumb_height // 2), label, fill="#222222", font=label_font, anchor="lm")
        for column, frame_index in enumerate(FRAME_INDICES):
            frame = frames[frame_index].resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            x = label_width + column * thumb_width
            sheet.paste(frame, (x, y))
        draw.line((0, y, sheet_width, y), fill="#DDDDDD", width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, dpi=(300, 300), optimize=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, object]] = []
    issues: list[str] = []
    sheets: list[str] = []

    prompt_dirs = sorted(path.parent for path in input_root.rglob("metadata.txt"))
    for prompt_dir in prompt_dirs:
        group_name = prompt_dir.parent.name
        metadata = parse_metadata(prompt_dir / "metadata.txt")
        video_paths = sorted(prompt_dir.glob("*.mp4"))
        if len(video_paths) != 4:
            issues.append(f"{prompt_dir}: expected 4 videos, found {len(video_paths)}")
            continue

        sheet_rows: list[tuple[str, dict[int, Image.Image]]] = []
        for video_path in video_paths:
            media, selected, frame_count = decode_selected_frames(video_path)
            width, height = media["size"]
            fps = float(media.get("fps", 0.0))
            duration = float(media.get("duration", 0.0))
            missing_frames = sorted(set(FRAME_INDICES) - set(selected))
            label = method_label(video_path)
            estimated_reference = is_estimated_reference(video_path)
            expected_size = ESTIMATED_REFERENCE_SIZE if estimated_reference else EXPECTED_SIZE
            role = "aligned_estimated_reference" if estimated_reference else "720p_method_output"

            if (width, height) != expected_size:
                issues.append(f"{video_path}: size {(width, height)} != expected {expected_size} for {role}")
            if frame_count != EXPECTED_FRAMES:
                issues.append(f"{video_path}: frames {frame_count} != {EXPECTED_FRAMES}")
            if not math.isclose(fps, EXPECTED_FPS, abs_tol=0.01):
                issues.append(f"{video_path}: fps {fps} != {EXPECTED_FPS}")
            if missing_frames:
                issues.append(f"{video_path}: missing selected frames {missing_frames}")

            audit_rows.append(
                {
                    "group": group_name,
                    "prompt_index": metadata.get("prompt_index", ""),
                    "seed": metadata.get("seed", ""),
                    "method": label,
                    "role": role,
                    "eligible_for_720p_detail_comparison": not estimated_reference,
                    "video": str(video_path),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "duration_s": duration,
                    "frames": frame_count,
                    "sha256": sha256(video_path),
                }
            )
            sheet_rows.append((label, selected))

        sheet_name = f"contact_{group_name}_{prompt_dir.name}.png"
        build_contact_sheet(group_name, metadata, sheet_rows, output_root / sheet_name)
        sheets.append(sheet_name)

    audit_csv = output_root / "video_audit.csv"
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "input_root": str(input_root),
        "prompt_groups": len(prompt_dirs),
        "videos": len(audit_rows),
        "frame_indices_zero_based": list(FRAME_INDICES),
        "estimated_reference_note": (
            "Native-HR (estimated) is the content-aligned 640x368 full-trajectory proxy used "
            "for qualitative context only; it is not the actual 720p Native-HR quantitative baseline."
        ),
        "issues": issues,
        "contact_sheets": sheets,
    }
    (output_root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
