from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.models import WanNoisyLatentUpsampler
from wan_sr.pipelines import transition_lr_to_hr
from wan_sr.training.checkpoint import load_checkpoint
from wan_sr.training.config import load_yaml


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint.get("config", {}).get("model", config.get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tensors = load_file(args.x_t_lr, device="cpu")
    x_t_lr = tensors["latent"] if "latent" in tensors else next(iter(tensors.values()))
    if x_t_lr.ndim == 4:
        x_t_lr = x_t_lr.unsqueeze(0)
    sigma = torch.tensor([args.sigma], dtype=torch.float32)

    model = WanNoisyLatentUpsampler(**model_config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and "ema" in checkpoint:
        from wan_sr.training.ema import EMA

        ema = EMA(model)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)

    x_t_hr, pred_z0_hr = transition_lr_to_hr(x_t_lr.to(device), sigma.to(device), model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "x_t_hr": x_t_hr.squeeze(0).cpu(),
            "pred_z0_hr": pred_z0_hr.squeeze(0).cpu(),
        },
        str(out_path),
    )
    print(f"saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--x_t_lr", required=True, help="safetensors containing key 'latent' or one tensor")
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--out", default="outputs/transition.safetensors")
    parser.add_argument("--config", default="configs/infer_transition.yaml")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
