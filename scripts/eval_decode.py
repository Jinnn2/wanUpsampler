from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LatentPairDataset
from wan_sr.data.video_io import write_video
from wan_sr.models import WanNoisyLatentUpsampler
from wan_sr.schedulers.noise_utils import spatial_upsample_latent
from wan_sr.training.checkpoint import load_checkpoint
from wan_sr.training.config import load_yaml
from wan_sr.vae import WanVAEWrapper


DEFAULT_MODEL_ROOT = "/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint.get("config", {}).get("model", config.get("model", {}))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LatentPairDataset(args.data_dir or config.get("data_dir"), sigma_mode=args.sigma_mode, strict_meta=False)
    indices = list(range(args.start_index, min(len(dataset), args.start_index + args.num_samples)))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)

    model = WanNoisyLatentUpsampler(**model_config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and "ema" in checkpoint:
        from wan_sr.training.ema import EMA

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
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}

    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True):
            sample_id = batch["sample_id"][0]
            sample_dir = out_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            x_t_lr = batch["x_t_lr"].to(device)
            sigma = batch["sigma"].to(device)
            z0_lr = batch["z0_lr"].to(device)
            z0_hr = batch["z0_hr"].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred = model(x_t_lr, sigma)

            baseline = spatial_upsample_latent(z0_lr, scale=2, mode="trilinear")
            save_file({"latent": pred.squeeze(0).cpu()}, str(sample_dir / "pred_z0_hr.safetensors"))
            save_file({"latent": baseline.squeeze(0).cpu()}, str(sample_dir / "interp_z0_hr.safetensors"))

            pred_video = vae.decode(pred.squeeze(0).cpu())[0]
            gt_video = vae.decode(z0_hr.squeeze(0).cpu())[0]
            interp_video = vae.decode(baseline.squeeze(0).cpu())[0]

            write_video(sample_dir / "pred.mp4", pred_video, fps=args.fps)
            write_video(sample_dir / "gt.mp4", gt_video, fps=args.fps)
            write_video(sample_dir / "latent_interp.mp4", interp_video, fps=args.fps)
            write_video(sample_dir / "comparison.mp4", torch.cat([interp_video, pred_video, gt_video], dim=2), fps=args.fps)

            meta = {
                "sample_id": sample_id,
                "sigma": float(sigma.item()),
                "comparison_order": "latent_interp | pred | gt",
            }
            with (sample_dir / "eval_decode_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_wan21_x2_512.yaml")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir", default="outputs/eval_decode")
    parser.add_argument("--model_root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--vae_path", help="Path to official Wan2.1_VAE.pth")
    parser.add_argument("--wan_repo", help="Path to the official Wan2.1 source repo containing wan/modules/vae.py")
    parser.add_argument("--vae_backend", choices=["auto", "official", "diffusers"], default="auto")
    parser.add_argument("--sigma_mode", choices=["mid", "uniform", "clean"], default="mid")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
