from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data.degradation import center_crop_resize_video, resize_video
from wan_sr.data.video_io import iter_fixed_length_clips, list_videos, read_video_frames
from wan_sr.vae import WanVAEWrapper


def scale_key(scale: float) -> str:
    return f"{float(scale):g}"


def storage_scale_key(scale: str) -> str:
    return scale.replace(".", "p")


def parse_hw(value: str) -> tuple[int, int]:
    """Parse an explicit spatial size written as HxW (or H,W)."""
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"size must be HxW, got {value!r}")
    height, width = (int(part) for part in parts)
    if height < 1 or width < 1:
        raise argparse.ArgumentTypeError(f"size must be positive, got {value!r}")
    return height, width


def key(name: str, row: int) -> bytes:
    return f"{name}_{row:08d}".encode()


class PairWriter:
    def __init__(self, root: Path, hr_shape: tuple[int, ...], lr_shapes: dict[str, tuple[int, ...]], map_size_gb: int):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.env = lmdb.open(str(root), map_size=map_size_gb * 1024**3, subdir=True, meminit=False)
        self.row = 0
        self.meta = {
            "schema": "wan_uni_clean_v1",
            "num_samples": 0,
            "dtype": "float16",
            "hr_shape": list(hr_shape),
            "lr_shapes": {name: list(shape) for name, shape in lr_shapes.items()},
            "scales": sorted(lr_shapes, key=float),
        }
        self._write_metadata()

    def write(self, z0_hr: torch.Tensor, lr_by_scale: dict[str, torch.Tensor], *, prompt: str, meta: dict) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(key("z0_hr", self.row), z0_hr.detach().cpu().numpy().astype(np.float16).tobytes())
            for scale, latent in lr_by_scale.items():
                txn.put(key(f"z0_lr_{storage_scale_key(scale)}", self.row), latent.detach().cpu().numpy().astype(np.float16).tobytes())
            txn.put(key("prompt", self.row), prompt.encode("utf-8"))
            txn.put(key("meta", self.row), json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self.row += 1
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.meta["num_samples"] = self.row
        with self.env.begin(write=True) as txn:
            txn.put(b"metadata", json.dumps(self.meta, ensure_ascii=False, indent=2).encode())

    def close(self) -> None:
        self._write_metadata()
        self.env.sync()
        self.env.close()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    scales = [float(value) for value in args.scales]
    if args.lr_sizes is not None and len(args.lr_sizes) != len(scales):
        raise ValueError("--lr_sizes must provide exactly one HxW size per --scales value")
    explicit_lr_sizes = args.lr_sizes
    videos = list_videos(args.video_dir)
    videos = videos[max(0, int(args.video_offset)):]
    if args.max_samples is not None:
        videos = videos[: args.max_samples]
    if not videos:
        raise FileNotFoundError(f"No videos under {args.video_dir}")

    vae = WanVAEWrapper(
        args.model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=device,
        dtype=dtype,
    )
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    processed = 0
    writer: PairWriter | None = None
    for video_path in tqdm(videos, desc="build universal clean pairs"):
        frames = read_video_frames(video_path, max_frames=args.max_video_frames)
        for clip_index, clip in enumerate(iter_fixed_length_clips(frames, args.num_frames, args.stride, args.max_clips_per_video)):
            hr = center_crop_resize_video(clip, tuple(args.hr_size))
            z0_hr = vae.encode(hr).squeeze(0).contiguous()
            lr_by_scale: dict[str, torch.Tensor] = {}
            degradation: dict[str, object] = {}
            for scale_index, scale in enumerate(scales):
                lr_size = explicit_lr_sizes[scale_index] if explicit_lr_sizes is not None else (
                    max(8, int(round(args.hr_size[0] / scale))),
                    max(8, int(round(args.hr_size[1] / scale))),
                )
                lr = resize_video(hr, lr_size, mode=args.resize_mode).clamp(0, 1)
                z0_lr = vae.encode(lr).squeeze(0).contiguous()
                name = scale_key(scale)
                lr_by_scale[name] = z0_lr
                degradation[name] = {"rgb_size": list(lr_size), "resize_mode": args.resize_mode}
            if writer is None:
                writer = PairWriter(out / "shard_00000", tuple(z0_hr.shape), {k: tuple(v.shape) for k, v in lr_by_scale.items()}, args.map_size_gb)
            if tuple(z0_hr.shape) != tuple(writer.meta["hr_shape"]):
                raise ValueError("All samples must share HR latent shape; use fixed frames and HR size")
            for scale, latent in lr_by_scale.items():
                if tuple(latent.shape) != tuple(writer.meta["lr_shapes"][scale]):
                    raise ValueError(f"Scale {scale} produced inconsistent latent shape {tuple(latent.shape)}")
            writer.write(
                z0_hr,
                lr_by_scale,
                prompt=video_path.stem,
                meta={
                    "schema": "wan_uni_clean_v1",
                    "source_video": str(video_path),
                    "clip_index": clip_index,
                    "frames": args.num_frames,
                    "hr_size": list(args.hr_size),
                    "scales": scales,
                    "degradation": degradation,
                    "z0_hr_shape": list(z0_hr.shape),
                    "z0_lr_shapes": {k: list(v.shape) for k, v in lr_by_scale.items()},
                },
            )
            processed += 1
    if writer is None:
        raise RuntimeError("No fixed-length clips were found")
    writer.close()
    print(f"Universal clean LMDB ready: {out} samples={processed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_root", default=".")
    parser.add_argument("--vae_path")
    parser.add_argument("--wan_repo")
    parser.add_argument("--vae_backend", choices=["auto", "official", "lightx2v", "diffusers"], default="auto")
    parser.add_argument("--hr_size", type=int, nargs=2, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    parser.add_argument(
        "--lr_sizes",
        type=parse_hw,
        nargs="+",
        help="Optional explicit LR RGB sizes, one HxW value per --scales entry",
    )
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max_clips_per_video", type=int)
    parser.add_argument("--max_video_frames", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--video_offset", type=int, default=0)
    parser.add_argument("--resize_mode", choices=["bilinear", "bicubic", "area"], default="bicubic")
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


if __name__ == "__main__":
    main()
