from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LatentPairDataset
from wan_sr.losses import LatentUpsamplerLoss
from wan_sr.models import WanNoisyLatentUpsampler
from wan_sr.training.checkpoint import load_checkpoint
from wan_sr.training.config import load_yaml


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint.get("config", {}).get("model", config.get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = LatentPairDataset(args.data_dir or config.get("data_dir"), sigma_mode=args.sigma_mode, strict_meta=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = WanNoisyLatentUpsampler(**model_config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and "ema" in checkpoint:
        from wan_sr.training.ema import EMA

        ema = EMA(model)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()

    criterion = LatentUpsamplerLoss(**config.get("loss", {}))
    totals: dict[str, float] = {}
    count = 0
    precision = args.precision
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x_t_lr = batch["x_t_lr"].to(device)
            sigma = batch["sigma"].to(device)
            z0_lr = batch["z0_lr"].to(device)
            z0_hr = batch["z0_hr"].to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred = model(x_t_lr, sigma)
            _, losses = criterion(pred.float(), z0_hr.float(), z0_lr.float())
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1

    if count == 0:
        raise RuntimeError("No batches evaluated")
    for key, value in totals.items():
        print(f"{key}: {value / count:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_wan21_x2_512.yaml")
    parser.add_argument("--data_dir")
    parser.add_argument("--sigma_mode", choices=["mid", "uniform", "clean"], default="mid")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
