from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LastStepSkipLoRALMDBDataset  # noqa: E402
from wan_sr.data.video_io import write_video  # noqa: E402
from wan_sr.vae import WanVAEWrapper  # noqa: E402


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    dataset = LastStepSkipLoRALMDBDataset(args.data_dir, dtype=torch.float32)
    indices = select_indices(len(dataset), args.indices, args.num_samples)
    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    panel_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    vae = WanVAEWrapper(
        args.model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=args.device,
        dtype=dtype,
    )

    manifest: list[dict[str, Any]] = []
    for sample_index in indices:
        row = dataset[sample_index]
        sample_name = f"sample_{sample_index:05d}"
        print(f"decode {sample_name}: {row['sample_id']} prompt={row['prompt'][:80]!r}")

        x3_video = vae.decode(row["x3_lr"])[0]
        z4_lr_video = vae.decode(row["z4_lr_teacher"])[0]
        z4_hr_video = vae.decode(row["z0_hr"])[0]

        x3_path = video_dir / f"{sample_name}_x3_lr.mp4"
        z4_lr_path = video_dir / f"{sample_name}_z4_lr_teacher.mp4"
        z4_hr_path = video_dir / f"{sample_name}_z4_hr.mp4"
        write_video(x3_path, x3_video, fps=args.fps)
        write_video(z4_lr_path, z4_lr_video, fps=args.fps)
        write_video(z4_hr_path, z4_hr_video, fps=args.fps)

        panel_path = panel_dir / f"{sample_name}_x3_z4lr_z4hr_compare.mp4"
        panel_status = "ok"
        try:
            make_three_panel(
                x3_path=x3_path,
                z4_lr_path=z4_lr_path,
                z4_hr_path=z4_hr_path,
                output_path=panel_path,
                panel_height=args.panel_height,
                panel_width=args.panel_width,
                fps=args.fps,
            )
        except Exception as exc:
            panel_status = f"failed: {exc}"
            print(f"[warn] failed to make panel for {sample_name}: {exc}", file=sys.stderr)

        meta = json.loads(row["meta_json"])
        record = {
            "sample_index": sample_index,
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "seed": row["seed"],
            "x3_lr_shape": list(row["x3_lr"].shape),
            "z4_lr_teacher_shape": list(row["z4_lr_teacher"].shape),
            "z4_hr_shape": list(row["z0_hr"].shape),
            "x3_minus_z4_lr_l1": float((row["x3_lr"] - row["z4_lr_teacher"]).abs().mean()),
            "paths": {
                "x3_lr": str(x3_path),
                "z4_lr_teacher": str(z4_lr_path),
                "z4_hr": str(z4_hr_path),
                "compare_panel": str(panel_path) if panel_status == "ok" else None,
            },
            "panel_status": panel_status,
            "last_step_skip_recipe": meta.get("last_step_skip_recipe", {}),
        }
        manifest.append(record)

    manifest_path = out_dir / "preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Preview complete: {out_dir}")
    print(f"Manifest: {manifest_path}")


def select_indices(length: int, indices_arg: str | None, num_samples: int) -> list[int]:
    if length <= 0:
        raise ValueError("dataset is empty")
    if indices_arg:
        indices = [int(part.strip()) for part in indices_arg.split(",") if part.strip()]
    elif num_samples >= length:
        indices = list(range(length))
    else:
        if num_samples <= 1:
            indices = [0]
        else:
            indices = torch.linspace(0, length - 1, steps=num_samples).round().to(torch.long).tolist()
    unique: list[int] = []
    for index in indices:
        if index < 0 or index >= length:
            raise ValueError(f"sample index out of range [0, {length - 1}]: {index}")
        if index not in unique:
            unique.append(index)
    return unique


def make_three_panel(
    *,
    x3_path: Path,
    z4_lr_path: Path,
    z4_hr_path: Path,
    output_path: Path,
    panel_height: int,
    panel_width: int,
    fps: int,
) -> None:
    labeled = [
        make_labeled_panel(x3_path, output_path.with_name(output_path.stem + "_x3_lr.tmp.mp4"), "x3_lr", panel_height, panel_width, fps),
        make_labeled_panel(
            z4_lr_path,
            output_path.with_name(output_path.stem + "_z4_lr_teacher.tmp.mp4"),
            "z4_lr_teacher",
            panel_height,
            panel_width,
            fps,
        ),
        make_labeled_panel(z4_hr_path, output_path.with_name(output_path.stem + "_z4_hr.tmp.mp4"), "z4_hr", panel_height, panel_width, fps),
    ]
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(labeled[0]),
                "-i",
                str(labeled[1]),
                "-i",
                str(labeled[2]),
                "-filter_complex",
                "[0:v][1:v][2:v]hstack=inputs=3[v]",
                "-map",
                "[v]",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(output_path),
            ],
            check=True,
        )
    finally:
        for path in labeled:
            path.unlink(missing_ok=True)


def make_labeled_panel(input_path: Path, output_path: Path, label: str, height: int, width: int, fps: int) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            (
                f"scale={width}:{height}:flags=bicubic,fps={fps},"
                "drawbox=x=0:y=0:w=iw:h=46:color=black@0.55:t=fill,"
                f"drawtext=text='{label}':x=20:y=12:fontsize=24:fontcolor=white"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    default_model_root = os.environ.get(
        "CR_DISTILL_MODEL_ROOT",
        os.environ.get("MODEL_ROOT", "/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill"),
    )
    default_vae_path = os.environ.get("VAE_PATH")
    if default_vae_path is None:
        default_vae_path = str(Path(default_model_root) / "Wan2.1_VAE.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.environ.get(
            "CR_DISTILL_LORA_LMDB_DIR",
            "data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3",
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/changing_resolution_distill_last_step_skip_lora_preview",
    )
    parser.add_argument("--model_root", default=default_model_root)
    parser.add_argument("--vae_path", default=default_vae_path)
    parser.add_argument("--wan_repo", default=os.environ.get("LIGHTX2V_REPO") or os.environ.get("WAN_REPO"))
    parser.add_argument("--vae_backend", choices=["auto", "official", "lightx2v", "diffusers"], default="lightx2v")
    parser.add_argument("--indices", help="Comma-separated sample indices. Overrides --num_samples.")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--panel_height", type=int, default=360)
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


if __name__ == "__main__":
    main()
