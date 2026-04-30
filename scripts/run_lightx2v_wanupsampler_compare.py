from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data.video_io import read_video_frames, write_video


def main() -> None:
    args = parse_args()
    with Path(args.config_json).open("r", encoding="utf-8") as f:
        config = json.load(f)

    save_path = Path(args.save_result_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stem = save_path.stem
    lowres_path = save_path.with_name(f"{stem}_lowres.mp4")
    bicubic_path = save_path.with_name(f"{stem}_bicubic.mp4")
    wanupsampler_path = save_path.with_name(f"{stem}_wanupsampler.mp4")
    native_hr_path = save_path.with_name(f"{stem}_native_highres.mp4")
    meta_path = save_path.with_suffix(".json")

    lightx2v_path = Path(config["paths"]["lightx2v_path"])
    model_path = config["paths"]["model_path"]
    wanupsampler_cfg = config["wan_upsampler"]
    fps = int(wanupsampler_cfg.get("output_fps", 16))

    with tempfile.TemporaryDirectory(prefix="wanupsampler_cfg_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        lowres_cfg_path = tmp_dir_path / "generator_lowres.json"
        lowres_cfg_path.write_text(json.dumps(config["generator_lowres"], ensure_ascii=False, indent=2), encoding="utf-8")

        run_lightx2v_infer(
            lightx2v_path=lightx2v_path,
            model_cls=args.model_cls,
            task=args.task,
            model_path=model_path,
            config_json=lowres_cfg_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            save_result_path=lowres_path,
            seed=args.seed,
        )

        if config.get("comparison", {}).get("make_bicubic", True):
            make_bicubic_video(lowres_path, bicubic_path, fps=fps)

        run_wanupsampler(
            video_path=lowres_path,
            save_result_path=wanupsampler_path,
            upsampler_config=wanupsampler_cfg,
        )

        native_enabled = bool(config.get("generator_native_highres", {}).get("enabled", False)) and bool(config.get("comparison", {}).get("make_native_highres", True))
        if native_enabled:
            native_cfg_path = tmp_dir_path / "generator_native_highres.json"
            native_cfg_path.write_text(json.dumps({k: v for k, v in config["generator_native_highres"].items() if k != "enabled"}, ensure_ascii=False, indent=2), encoding="utf-8")
            run_lightx2v_infer(
                lightx2v_path=lightx2v_path,
                model_cls=args.model_cls,
                task=args.task,
                model_path=model_path,
                config_json=native_cfg_path,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                save_result_path=native_hr_path,
                seed=args.seed,
            )

    make_comparison_video(
        layout=config.get("comparison", {}).get("layout", ["bicubic", "wan_upsampler"]),
        fps=fps,
        output_path=save_path,
        bicubic_path=bicubic_path if bicubic_path.exists() else None,
        wanupsampler_path=wanupsampler_path,
        native_highres_path=native_hr_path if native_hr_path.exists() else None,
    )

    meta = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "lowres_video": str(lowres_path),
        "bicubic_video": str(bicubic_path) if bicubic_path.exists() else None,
        "wanupsampler_video": str(wanupsampler_path),
        "native_highres_video": str(native_hr_path) if native_hr_path.exists() else None,
        "comparison_video": str(save_path),
        "layout": config.get("comparison", {}).get("layout", ["bicubic", "wan_upsampler"]),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {save_path}")


def run_lightx2v_infer(
    lightx2v_path: Path,
    model_cls: str,
    task: str,
    model_path: str,
    config_json: Path,
    prompt: str,
    negative_prompt: str,
    save_result_path: Path,
    seed: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "lightx2v.infer",
        "--seed",
        str(seed),
        "--model_cls",
        model_cls,
        "--task",
        task,
        "--model_path",
        model_path,
        "--config_json",
        str(config_json),
        "--prompt",
        prompt,
        "--negative_prompt",
        negative_prompt,
        "--save_result_path",
        str(save_result_path),
    ]
    subprocess.run(command, cwd=lightx2v_path, check=True)


def run_wanupsampler(video_path: Path, save_result_path: Path, upsampler_config: dict) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "apply_wan_upsampler_to_video.py"),
        "--video_path",
        str(video_path),
        "--save_result_path",
        str(save_result_path),
        "--checkpoint",
        upsampler_config["checkpoint"],
        "--train_config",
        upsampler_config["train_config"],
        "--model_root",
        upsampler_config["model_root"],
        "--vae_path",
        upsampler_config["vae_path"],
        "--wan_repo",
        upsampler_config["wan_repo"],
        "--vae_backend",
        upsampler_config.get("vae_backend", "lightx2v"),
        "--precision",
        upsampler_config.get("precision", "bf16"),
        "--sigma",
        str(upsampler_config.get("sigma", 0.0)),
        "--output_fps",
        str(upsampler_config.get("output_fps", 16)),
    ]
    if upsampler_config.get("use_ema", True):
        command.append("--use_ema")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def make_bicubic_video(video_path: Path, save_result_path: Path, fps: int) -> None:
    video = read_video_frames(video_path)
    t, h, w, c = video.shape
    frames = video.permute(0, 3, 1, 2)
    up = F.interpolate(frames, size=(h * 2, w * 2), mode="bicubic", align_corners=False).clamp(0, 1)
    write_video(save_result_path, up.permute(0, 2, 3, 1), fps=fps)


def make_comparison_video(
    layout: list[str],
    fps: int,
    output_path: Path,
    bicubic_path: Path | None,
    wanupsampler_path: Path,
    native_highres_path: Path | None,
) -> None:
    mapping = {
        "bicubic": bicubic_path,
        "wan_upsampler": wanupsampler_path,
        "native_highres": native_highres_path,
    }
    videos = []
    for key in layout:
        path = mapping.get(key)
        if path is None or not path.exists():
            continue
        videos.append(read_video_frames(path))
    if not videos:
        raise ValueError("No videos available for comparison")

    min_frames = min(video.shape[0] for video in videos)
    target_h = min(video.shape[1] for video in videos)
    target_w = min(video.shape[2] for video in videos)
    resized = []
    for video in videos:
        clip = video[:min_frames]
        if clip.shape[1] != target_h or clip.shape[2] != target_w:
            frames = clip.permute(0, 3, 1, 2)
            frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False).clamp(0, 1)
            clip = frames.permute(0, 2, 3, 1)
        resized.append(clip)
    comparison = torch.cat(resized, dim=2)
    write_video(output_path, comparison, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cls", type=str, default="wan2.1")
    parser.add_argument("--task", type=str, default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--save_result_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
