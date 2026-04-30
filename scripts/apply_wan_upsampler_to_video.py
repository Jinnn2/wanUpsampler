from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data.video_io import read_video_frames, write_video
from wan_sr.models import WanNoisyLatentUpsampler
from wan_sr.schedulers.noise_utils import add_flow_noise
from wan_sr.training.checkpoint import load_checkpoint
from wan_sr.training.config import load_yaml
from wan_sr.vae import WanVAEWrapper


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_config = load_yaml(args.train_config)
    model_config = checkpoint.get("config", {}).get("model", train_config.get("model", {}))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32

    video = read_video_frames(args.video_path)
    vae = WanVAEWrapper(
        args.model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=device,
        dtype=dtype,
    )

    latents = vae.encode(video).to(device)
    sigma = torch.full((latents.shape[0],), float(args.sigma), device=device, dtype=torch.float32)
    x_t_lr = latents
    if args.sigma > 0:
        x_t_lr, _ = add_flow_noise(latents, sigma)

    model = WanNoisyLatentUpsampler(**model_config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and "ema" in checkpoint:
        from wan_sr.training.ema import EMA

        ema = EMA(model)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
        pred_z0_hr = model(x_t_lr, sigma)

    pred_video = vae.decode(pred_z0_hr.squeeze(0).cpu())[0]
    out_path = Path(args.save_result_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(out_path, pred_video, fps=args.output_fps)
    print(f"saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--save_result_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--wan_repo", required=True)
    parser.add_argument("--vae_backend", choices=["auto", "official", "lightx2v", "diffusers"], default="lightx2v")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--output_fps", type=int, default=16)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
