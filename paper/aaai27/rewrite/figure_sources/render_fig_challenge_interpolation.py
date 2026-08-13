#!/usr/bin/env python3
"""Render rewrite Figure 3 from the matched prompt-08 video group.

The renderer reuses the canonical handoff-comparison layout while keeping the
experimental frames deterministic and local.  The selected group uses
prompt index 08, seed 9708, frame fraction 0.5, and a centered 3x crop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz
from PIL import Image


HERE = Path(__file__).resolve().parent
REWRITE_DIR = HERE.parent
AAAI27_DIR = REWRITE_DIR.parent
REPO_DIR = AAAI27_DIR.parents[1]
CANONICAL_RENDERER = (
    AAAI27_DIR / "figures" / "gen_fig_handoff_comparisons.py"
)
CANDIDATE_DIR = (
    REPO_DIR
    / "outputs"
    / "aaai27_figure_work"
    / "talh_figure_video_candidates"
    / "TALH-Q_spatial_detail"
    / "prompt_08_seed9708"
)

SOURCE_STEM = "fig_challenge_interpolation_source"
FINAL_STEM = "fig_challenge_interpolation"
FRAME_INDEX = 40
FPS = 16
PROMPT = (
    "A glass greenhouse filled with tropical flowers, sunlight beams through "
    "mist, slow dolly movement, rich color detail."
)


def find_ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return Path(executable)
    bundled = (
        REPO_DIR
        / "outputs"
        / "aaai27_figure_work"
        / "python_deps"
        / "imageio_ffmpeg"
        / "binaries"
        / "ffmpeg-win-x86_64-v7.1.exe"
    )
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        "ffmpeg is required to decode the same exact frame from each video"
    )


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "canonical_handoff_renderer",
        CANONICAL_RENDERER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load renderer: {CANONICAL_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_exact_frames(renderer, group) -> None:
    ffmpeg = find_ffmpeg()
    for video in group.videos:
        output = renderer.frame_path(group, video, fraction=0.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video.path),
                "-vf",
                f"select=eq(n\\,{FRAME_INDEX})",
                "-frames:v",
                "1",
                str(output),
            ],
            check=True,
        )
        print(f"Decoded frame {FRAME_INDEX}: {output}")


def write_deterministic_pdf(png_path: Path, pdf_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
    document = fitz.open()
    page = document.new_page(
        width=width * 72 / 300,
        height=height * 72 / 300,
    )
    page.insert_image(page.rect, filename=str(png_path))
    document.set_metadata(
        {
            "producer": "InTraScale deterministic figure renderer",
            "creationDate": "D:20260729000000+08'00'",
            "modDate": "D:20260729000000+08'00'",
        }
    )
    temporary = pdf_path.with_suffix(".deterministic.tmp.pdf")
    if temporary.exists():
        temporary.unlink()
    document.save(
        temporary,
        garbage=4,
        deflate=True,
        no_new_id=True,
    )
    document.close()
    os.replace(temporary, pdf_path)


def main() -> None:
    renderer = load_renderer()
    renderer.ROW_FONT = renderer.load_font(bold=False, size=29)

    # Keep decoded-frame caches outside the paper tree; only final sources and
    # their manifest are published into the rewrite.
    renderer.SCRIPT_DIR = HERE
    renderer.WORK_DIR = (
        Path(tempfile.gettempdir())
        / "wanupsampler_fig3_render"
        / "prompt08_seed9708"
    )
    renderer.FRAME_DIR = renderer.WORK_DIR / "frames"
    renderer.EDGE_PROFILE_DIR = renderer.WORK_DIR / "edge_profiles"

    group = renderer.FigureGroup(
        key="interpolation_prompt08_seed9708",
        crop_position="center",
        magnification=3,
        output_stem=SOURCE_STEM,
        videos=(
            renderer.VideoEntry(
                "lr_endpoint",
                "LR endpoint",
                CANDIDATE_DIR / "01_Native-HR.mp4",
            ),
            renderer.VideoEntry(
                "trilinear",
                "Trilinear",
                CANDIDATE_DIR / "02_Trilinear-at-40.mp4",
            ),
            renderer.VideoEntry(
                "intrascale",
                "InTraScale",
                CANDIDATE_DIR / "04_TALH-Q-at-40.mp4",
            ),
        ),
    )
    renderer.validate_sources((group,))
    extract_exact_frames(renderer, group)
    manifest = renderer.render_final(group, fraction=0.5)
    write_deterministic_pdf(
        HERE / f"{SOURCE_STEM}.png",
        HERE / f"{SOURCE_STEM}.pdf",
    )

    manifest.update(
        {
            "prompt_index": "08",
            "seed": 9708,
            "prompt": PROMPT,
            "transition_step": 40,
            "frame_index_zero_based": FRAME_INDEX,
            "frame_timestamp_seconds": FRAME_INDEX / FPS,
            "protocol": {
                "inference_steps": 50,
                "sample_shift": 8,
                "cfg": 6,
                "frames": 81,
                "fps": FPS,
                "method_output_resolution": [1248, 720],
                "lr_endpoint_resolution": [640, 368],
            },
        }
    )
    manifest_path = HERE / f"{SOURCE_STEM}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    final_dir = REWRITE_DIR / "figures"
    final_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        source = HERE / f"{SOURCE_STEM}{suffix}"
        destination = final_dir / f"{FINAL_STEM}{suffix}"
        shutil.copy2(source, destination)
        print(f"Published: {destination}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
