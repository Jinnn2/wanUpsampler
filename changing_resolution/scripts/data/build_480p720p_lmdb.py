from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data.degradation import center_crop_resize_video, resize_video
from wan_sr.data.video_io import iter_fixed_length_clips, list_videos, read_video_frames
from wan_sr.vae import WanVAEWrapper


DEFAULT_MODEL_ROOT = "/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"


def main() -> None:
    args = parse_args()
    model_root = args.model_root or DEFAULT_MODEL_ROOT
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32

    out_dir = Path(args.out_dir)
    prepare_output_dir(out_dir, overwrite=args.overwrite)

    videos = list_videos(args.video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.video_dir}")

    prompts = load_prompts(args.prompts_file) if args.prompts_file else []
    vae = WanVAEWrapper(
        model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=device,
        dtype=dtype,
    )

    writer = ShardedCleanLatentLMDBWriter(
        out_dir=out_dir,
        shard_size=args.shard_size,
        map_size_gb=args.map_size_gb,
        compression_dtype=np.float16,
    )

    saved = 0
    try:
        for video_path in tqdm(videos, desc="videos", dynamic_ncols=True):
            if args.max_samples is not None and saved >= args.max_samples:
                break
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
            prompt = infer_prompt(video_path, prompts)
            for clip_index, clip in enumerate(clips):
                if args.max_samples is not None and saved >= args.max_samples:
                    break

                hr_clip = center_crop_resize_video(clip, tuple(args.hr_size))
                lr_clip = resize_video(hr_clip, tuple(args.lr_size), mode=args.lr_resize_mode).clamp(0, 1)

                z0_lr = vae.encode(lr_clip).squeeze(0).contiguous()
                z0_hr = vae.encode(hr_clip).squeeze(0).contiguous()

                meta = {
                    "task": "changing_resolution_clean_480p_to_720p",
                    "format": "lmdb",
                    "vae": "Wan2.1",
                    "model_root": str(model_root),
                    "source_video": str(video_path),
                    "clip_index": clip_index,
                    "prompt": prompt,
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
                writer.write(z0_lr, z0_hr, prompt=prompt, meta=meta)
                saved += 1
    finally:
        writer.close()

    if args.require_samples is not None and saved < args.require_samples:
        raise RuntimeError(f"LMDB build only saved {saved} samples, required {args.require_samples}")
    print(f"Clean latent LMDB ready: {out_dir} ({saved} samples)")


@dataclass
class ShardedCleanLatentLMDBWriter:
    out_dir: Path
    shard_size: int
    map_size_gb: int
    compression_dtype: np.dtype

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env: lmdb.Environment | None = None
        self.shard_index = -1
        self.row_index = 0
        self.total = 0
        self.current_meta: dict[str, Any] | None = None

    def write(self, z0_lr: torch.Tensor, z0_hr: torch.Tensor, prompt: str, meta: dict[str, Any]) -> None:
        if self.env is None or self.row_index >= self.shard_size:
            self._open_next_shard(z0_lr, z0_hr)

        assert self.env is not None
        row = self.row_index
        z0_lr_np = _to_numpy_fp(z0_lr, self.compression_dtype)
        z0_hr_np = _to_numpy_fp(z0_hr, self.compression_dtype)
        meta_json = json.dumps(meta, ensure_ascii=False)

        with self.env.begin(write=True) as txn:
            txn.put(_key("z0_lr", row), z0_lr_np.tobytes())
            txn.put(_key("z0_hr", row), z0_hr_np.tobytes())
            txn.put(_key("prompt", row), prompt.encode("utf-8"))
            txn.put(_key("meta", row), meta_json.encode("utf-8"))

        self.row_index += 1
        self.total += 1
        self._write_metadata()

    def close(self) -> None:
        if self.env is not None:
            self._write_metadata()
            self.env.sync()
            self.env.close()
            self.env = None

    def _open_next_shard(self, z0_lr: torch.Tensor, z0_hr: torch.Tensor) -> None:
        self.close()
        self.shard_index += 1
        self.row_index = 0
        shard_dir = self.out_dir / f"shard_{self.shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        self.env = lmdb.open(str(shard_dir), map_size=self.map_size_gb * 1024**3, subdir=True, meminit=False)
        self.current_meta = {
            "num_samples": 0,
            "dtype": "float16",
            "z0_lr_shape": list(z0_lr.shape),
            "z0_hr_shape": list(z0_hr.shape),
            "schema": "wan_clean_latent_pair_lmdb_v1",
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self.env is None or self.current_meta is None:
            return
        meta = dict(self.current_meta)
        meta["num_samples"] = self.row_index
        with self.env.begin(write=True) as txn:
            txn.put(b"metadata", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
            txn.put(b"num_samples", str(self.row_index).encode("utf-8"))
            txn.put(b"z0_lr_shape", " ".join(map(str, meta["z0_lr_shape"])).encode("utf-8"))
            txn.put(b"z0_hr_shape", " ".join(map(str, meta["z0_hr_shape"])).encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", default="data/changing_resolution/lmdb_480p720p")
    parser.add_argument("--prompts_file")
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
    parser.add_argument("--require_samples", type=int)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--skip_bad_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output LMDB dir is not empty: {out_dir}. Pass --overwrite to rebuild.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def load_prompts(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def infer_prompt(video_path: Path, prompts: list[str]) -> str:
    match = re.search(r"_(\d{3,8})_seed\d+", video_path.stem)
    if match is not None:
        index = int(match.group(1))
        if index < len(prompts):
            return prompts[index]
    return ""


def _to_numpy_fp(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float16).numpy().astype(dtype, copy=False)


def _key(name: str, row_id: int) -> bytes:
    return f"{name}_{row_id:08d}_data".encode("utf-8")


if __name__ == "__main__":
    main()
