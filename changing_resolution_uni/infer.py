from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_universal_upsampler


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, payload = load_universal_upsampler(args.checkpoint, device=device, use_ema=not args.no_ema)
    array = np.load(args.input)
    latent = torch.from_numpy(array)
    # Accept the repository's conventional CTHW and THWC .npy layouts.
    if latent.ndim == 4 and latent.shape[0] == 16:
        cthw = latent
    elif latent.ndim == 4 and latent.shape[-1] == 16:
        cthw = latent.permute(3, 0, 1, 2)
    else:
        raise ValueError(f"input must be CTHW or THWC with 16 channels, got {tuple(latent.shape)}")
    target_size = (int(args.target_h), int(args.target_w))
    prediction = model(cthw.unsqueeze(0).to(device=device), output_size=target_size).squeeze(0).cpu()
    if args.output:
        np.save(args.output, prediction.numpy())
    print({
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload.get("step"),
        "input_shape": list(cthw.shape),
        "output_shape": list(prediction.shape),
        "target_size": list(target_size),
        "output": str(args.output) if args.output else None,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one universal clean-latent upsampling checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target_h", type=int, required=True)
    parser.add_argument("--target_w", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_ema", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
