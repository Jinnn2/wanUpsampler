from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import CleanLatentLMDBDataset, CleanLatentPairDataset
from wan_sr.data.video_io import write_video
from wan_sr.models import WanCleanLatentResizer
from wan_sr.training.checkpoint import load_checkpoint
from wan_sr.training.config import load_yaml
from wan_sr.training.ema import EMA
from wan_sr.vae import WanVAEWrapper


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.train_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint.get("config", {}).get("model", train_config.get("model", {}))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32

    dataset = load_dataset(args.data_dir, args.data_format)
    indices = select_indices(len(dataset), train_config, args.split, args.offset, args.limit)
    if not indices:
        raise RuntimeError("No samples selected for operator compare")

    model = WanCleanLatentResizer(**model_config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and "ema" in checkpoint:
        ema = EMA(model)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()

    vae = WanVAEWrapper(
        args.model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=device,
        dtype=dtype,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / f"metrics_{args.split}_offset{args.offset}_limit{args.limit}.jsonl"
    similarity = VideoSimilarityMetrics(args.metrics, device=device)

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}

    with metric_path.open("w", encoding="utf-8") as metric_file:
        for local_index, sample_index in enumerate(indices):
            sample = dataset[sample_index]
            z0_lr = sample["z0_lr"].unsqueeze(0)
            z0_hr = sample["z0_hr"].unsqueeze(0)
            z0_lr_device = z0_lr.to(device)
            target_size = (z0_hr.shape[2], z0_hr.shape[3], z0_hr.shape[4])

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                interp_z0_hr = F.interpolate(z0_lr_device, size=target_size, mode="trilinear", align_corners=False)
                trained_z0_hr = model(z0_lr_device, output_size=(z0_hr.shape[-2], z0_hr.shape[-1]))

            interp_cpu = interp_z0_hr.float().cpu()
            trained_cpu = trained_z0_hr.float().cpu()
            target_cpu = z0_hr.float().cpu()

            lr_video = vae.decode(z0_lr)[0]
            target_video = vae.decode(target_cpu)[0]
            interp_video = vae.decode(interp_cpu)[0]
            trained_video = vae.decode(trained_cpu)[0]

            sample_name = f"{local_index:03d}_idx{sample_index:06d}"
            paths = write_operator_videos(
                out_dir=out_dir,
                sample_name=sample_name,
                lr_video=lr_video,
                target_video=target_video,
                interp_video=interp_video,
                trained_video=trained_video,
                fps=args.fps,
                panel_height=args.panel_height,
                panel_width=args.panel_width,
            )

            metrics = compute_metrics(
                sample_index=sample_index,
                sample_id=str(sample.get("sample_id", sample_index)),
                interp_z0_hr=interp_cpu,
                trained_z0_hr=trained_cpu,
                target_z0_hr=target_cpu,
                interp_video=interp_video,
                trained_video=trained_video,
                target_video=target_video,
                similarity=similarity,
                metric_batch_size=args.metric_batch_size,
                paths=paths,
            )
            metric_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            metric_file.flush()
            print(
                f"[{local_index + 1}/{len(indices)}] idx={sample_index} "
                f"latent_l1 interp={metrics['interp_latent_l1']:.6f} "
                f"trained={metrics['trained_latent_l1']:.6f} "
                f"psnr interp={metrics['interp_psnr']:.3f} "
                f"trained={metrics['trained_psnr']:.3f} "
                f"ssim interp={metrics['interp_ssim']:.4f} "
                f"trained={metrics['trained_ssim']:.4f} "
                f"lpips interp={metrics['interp_lpips']:.4f} "
                f"trained={metrics['trained_lpips']:.4f}",
                flush=True,
            )

    print(f"operator compare ready: {out_dir}")
    print(f"metrics: {metric_path}")


def load_dataset(data_dir: str, data_format: str):
    if data_format == "lmdb":
        return CleanLatentLMDBDataset(data_dir, strict_channels=True)
    if data_format == "files":
        return CleanLatentPairDataset(data_dir, strict_channels=True)
    raise ValueError(f"data_format must be lmdb or files, got {data_format}")


def select_indices(total: int, config: dict, split: str, offset: int, limit: int) -> list[int]:
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 1234))
    val_ratio = float(train_cfg.get("val_ratio", 0.05))
    val_max_samples = int(train_cfg.get("val_max_samples", 100))

    import random

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(total * val_ratio)))
    if val_max_samples > 0:
        val_count = min(val_count, val_max_samples)
    val_count = min(val_count, total - 1)

    if split == "val":
        selected = sorted(indices[:val_count])
    elif split == "train":
        selected = sorted(indices[val_count:])
    elif split == "all":
        selected = list(range(total))
    else:
        raise ValueError(f"split must be val, train, or all, got {split}")
    return selected[offset : offset + limit]


def write_operator_videos(
    *,
    out_dir: Path,
    sample_name: str,
    lr_video: torch.Tensor,
    target_video: torch.Tensor,
    interp_video: torch.Tensor,
    trained_video: torch.Tensor,
    fps: int,
    panel_height: int,
    panel_width: int,
) -> dict[str, str]:
    video_dir = out_dir / "videos"
    panel_dir = out_dir / "panels"
    compare_dir = out_dir / "compare"
    for path in (video_dir, panel_dir, compare_dir):
        path.mkdir(parents=True, exist_ok=True)

    lr_path = video_dir / f"{sample_name}_lr480_decode.mp4"
    target_path = video_dir / f"{sample_name}_ori720_decode.mp4"
    interp_path = video_dir / f"{sample_name}_interp720_decode.mp4"
    trained_path = video_dir / f"{sample_name}_trained720_decode.mp4"
    compare_path = compare_dir / f"{sample_name}_operator_compare.mp4"

    write_video(lr_path, lr_video, fps=fps)
    write_video(target_path, target_video, fps=fps)
    write_video(interp_path, interp_video, fps=fps)
    write_video(trained_path, trained_video, fps=fps)

    panel_paths = [
        make_labeled_panel(lr_path, panel_dir / f"{sample_name}_panel_lr480.mp4", "lr 480 decode", panel_height, panel_width, fps),
        make_labeled_panel(target_path, panel_dir / f"{sample_name}_panel_ori720.mp4", "ori 720 decode", panel_height, panel_width, fps),
        make_labeled_panel(interp_path, panel_dir / f"{sample_name}_panel_interp720.mp4", "interp 720 decode", panel_height, panel_width, fps),
        make_labeled_panel(trained_path, panel_dir / f"{sample_name}_panel_trained720.mp4", "trained 720 decode", panel_height, panel_width, fps),
    ]
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *sum((["-i", str(path)] for path in panel_paths), []),
            "-filter_complex",
            "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(compare_path),
        ],
        check=True,
    )
    return {
        "lr480_decode": str(lr_path),
        "ori720_decode": str(target_path),
        "interp720_decode": str(interp_path),
        "trained720_decode": str(trained_path),
        "compare": str(compare_path),
    }


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


def compute_metrics(
    *,
    sample_index: int,
    sample_id: str,
    interp_z0_hr: torch.Tensor,
    trained_z0_hr: torch.Tensor,
    target_z0_hr: torch.Tensor,
    interp_video: torch.Tensor,
    trained_video: torch.Tensor,
    target_video: torch.Tensor,
    similarity: VideoSimilarityMetrics,
    metric_batch_size: int,
    paths: dict[str, str],
) -> dict[str, object]:
    interp_latent_l1 = float((interp_z0_hr - target_z0_hr).abs().mean())
    trained_latent_l1 = float((trained_z0_hr - target_z0_hr).abs().mean())
    interp_pixel_mse = float(F.mse_loss(interp_video, target_video))
    trained_pixel_mse = float(F.mse_loss(trained_video, target_video))
    interp_temporal = temporal_l1(interp_video, target_video)
    trained_temporal = temporal_l1(trained_video, target_video)
    interp_similarity = similarity.compute(interp_video, target_video, batch_size=metric_batch_size)
    trained_similarity = similarity.compute(trained_video, target_video, batch_size=metric_batch_size)
    return {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "interp_latent_l1": interp_latent_l1,
        "trained_latent_l1": trained_latent_l1,
        "latent_l1_delta_trained_minus_interp": trained_latent_l1 - interp_latent_l1,
        "interp_pixel_mse": interp_pixel_mse,
        "trained_pixel_mse": trained_pixel_mse,
        "interp_pixel_psnr_manual": psnr(interp_pixel_mse),
        "trained_pixel_psnr_manual": psnr(trained_pixel_mse),
        **{f"interp_{name}": value for name, value in interp_similarity.items()},
        **{f"trained_{name}": value for name, value in trained_similarity.items()},
        "psnr_delta_trained_minus_interp": trained_similarity.get("psnr", float("nan"))
        - interp_similarity.get("psnr", float("nan")),
        "ssim_delta_trained_minus_interp": trained_similarity.get("ssim", float("nan"))
        - interp_similarity.get("ssim", float("nan")),
        "lpips_delta_trained_minus_interp": trained_similarity.get("lpips", float("nan"))
        - interp_similarity.get("lpips", float("nan")),
        "interp_temporal_l1": interp_temporal,
        "trained_temporal_l1": trained_temporal,
        "paths": paths,
    }


def temporal_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape[0] < 2:
        return 0.0
    return float(((pred[1:] - pred[:-1]) - (target[1:] - target[:-1])).abs().mean())


def psnr(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


class VideoSimilarityMetrics:
    """TorchMetrics implementation matching x-attention eval/HunyuanVideo/similarity.py."""

    def __init__(self, metric_names: list[str], device: torch.device) -> None:
        self.metric_names = metric_names
        self.device = device
        self.metrics = self._build_metrics(metric_names, device)

    def compute(self, gen_video: torch.Tensor, ref_video: torch.Tensor, batch_size: int) -> dict[str, float]:
        gen_frames = _video_to_nchw(gen_video).to(self.device)
        ref_frames = _video_to_nchw(ref_video).to(self.device)
        if gen_frames.shape != ref_frames.shape:
            raise ValueError(f"gen/ref video shape mismatch: {gen_frames.shape} vs {ref_frames.shape}")

        batch_size = max(1, int(batch_size))
        with torch.no_grad():
            for metric in self.metrics.values():
                metric.reset()
            for start in range(0, gen_frames.shape[0], batch_size):
                gen_batch = gen_frames[start : start + batch_size]
                ref_batch = ref_frames[start : start + batch_size]
                for metric in self.metrics.values():
                    metric.update(gen_batch, ref_batch)
            return {name: float(metric.compute().item()) for name, metric in self.metrics.items()}

    def _build_metrics(self, metric_names: list[str], device: torch.device):
        try:
            from torchmetrics.image import (
                LearnedPerceptualImagePatchSimilarity,
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )
        except Exception as exc:
            raise RuntimeError(
                "PSNR/SSIM/LPIPS metrics require torchmetrics and lpips. "
                "Install them with: pip install torchmetrics lpips"
            ) from exc

        metrics = {}
        for metric_name in metric_names:
            if metric_name == "psnr":
                metric = PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3))
            elif metric_name == "ssim":
                metric = StructuralSimilarityIndexMeasure(data_range=(0, 1))
            elif metric_name == "lpips":
                metric = LearnedPerceptualImagePatchSimilarity(normalize=True)
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
            metrics[metric_name] = metric.to(device)
        return metrics


def _video_to_nchw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must be [T,H,W,C], got {tuple(video.shape)}")
    return video.float().clamp(0, 1).permute(0, 3, 1, 2).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_format", choices=["files", "lmdb"], default="lmdb")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--wan_repo", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--vae_backend", choices=["auto", "official", "lightx2v", "diffusers"], default="lightx2v")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"])
    parser.add_argument("--metric_batch_size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--panel_height", type=int, default=720)
    parser.add_argument("--panel_width", type=int, default=1248)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
