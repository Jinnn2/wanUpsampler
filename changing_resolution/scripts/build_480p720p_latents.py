from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data.degradation import center_crop_resize_video, resize_video
from wan_sr.data.video_io import iter_fixed_length_clips, list_videos, read_video_frames
from wan_sr.vae import WanVAEWrapper


DEFAULT_MODEL_ROOT = "/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B"


def main() -> None:
    args = parse_args()
    model_root = args.model_root or DEFAULT_MODEL_ROOT
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.video_dir}")

    vae = WanVAEWrapper(
        model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=device,
        dtype=dtype,
    )

    next_id = find_next_sample_id(out_dir)
    saved = 0
    for video_path in tqdm(videos, desc="videos", dynamic_ncols=True):
        try:
            frames = read_video_frames(video_path, max_frames=args.max_video_frames)
        except Exception as exc:
            if args.skip_bad_videos:
                print(f"[warn] skip bad video {video_path}: {exc}", file=sys.stderr)
                continue
            raise

        clips = iter_fixed_length_clips(
            frames,
            num_frames=args.num_frames,
            stride=args.stride,
            max_clips=args.max_clips_per_video,
        )
        for clip_index, clip in enumerate(clips):
            hr_clip = center_crop_resize_video(clip, tuple(args.hr_size))
            lr_clip = resize_video(hr_clip, tuple(args.lr_size), mode=args.lr_resize_mode).clamp(0, 1)

            z0_lr = vae.encode(lr_clip).squeeze(0)
            z0_hr = vae.encode(hr_clip).squeeze(0)

            sample_dir = out_dir / f"{next_id:06d}"
            sample_dir.mkdir(parents=True, exist_ok=False)
            save_file({"latent": z0_lr.contiguous()}, str(sample_dir / "z0_lr.safetensors"))
            save_file({"latent": z0_hr.contiguous()}, str(sample_dir / "z0_hr.safetensors"))

            meta = {
                "task": "changing_resolution_clean_480p_to_720p",
                "vae": "Wan2.1",
                "model_root": str(model_root),
                "source_video": str(video_path),
                "clip_index": clip_index,
                "frames": args.num_frames,
                "fps": args.fps,
                "hr_size": args.hr_size,
                "lr_size": args.lr_size,
                "latent_scale_h": z0_hr.shape[-2] / z0_lr.shape[-2],
                "latent_scale_w": z0_hr.shape[-1] / z0_lr.shape[-1],
                "z0_lr_shape": list(z0_lr.shape),
                "z0_hr_shape": list(z0_hr.shape),
                "degradation": {
                    "type": "resize_only",
                    "resize_kernel": args.lr_resize_mode,
                },
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            next_id += 1
            saved += 1
            if args.max_samples is not None and saved >= args.max_samples:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", default="data/changing_resolution/latent_pairs_480p720p")
    parser.add_argument("--model_root")
    parser.add_argument("--vae_path", help="Path to Wan2.1_VAE.pth")
    parser.add_argument("--wan_repo", help="Path to LightX2V repo or official Wan repo")
    parser.add_argument("--vae_backend", choices=["auto", "official", "lightx2v", "diffusers"], default="lightx2v")
    parser.add_argument("--hr_size", type=int, nargs=2, default=[720, 1248], metavar=("H", "W"))
    parser.add_argument("--lr_size", type=int, nargs=2, default=[480, 832], metavar=("H", "W"))
    parser.add_argument("--lr_resize_mode", choices=["bicubic", "bilinear", "area"], default="bicubic")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max_video_frames", type=int)
    parser.add_argument("--max_clips_per_video", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--skip_bad_videos", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def find_next_sample_id(out_dir: Path) -> int:
    ids = [int(path.name) for path in out_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return max(ids, default=-1) + 1


if __name__ == "__main__":
    main()
