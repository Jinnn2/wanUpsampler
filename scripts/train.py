from __future__ import annotations

import argparse
import itertools
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LatentPairDataset
from wan_sr.losses import LatentUpsamplerLoss
from wan_sr.models import WanNoisyLatentUpsampler
from wan_sr.training.checkpoint import load_checkpoint, save_checkpoint
from wan_sr.training.config import deep_update, load_yaml
from wan_sr.training.ema import EMA


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config = apply_cli_overrides(config, args)

    set_seed(int(config["train"].get("seed", 1234)))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        shutil.copy2(args.config, out_dir / "train_config.yaml")

    dataset = LatentPairDataset(
        config["data_dir"],
        sigma_mode=config.get("sigma", {}).get("mode", "mid"),
        strict_meta=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    batches = itertools.cycle(loader)

    model = WanNoisyLatentUpsampler(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    ema = EMA(model, decay=float(config["train"].get("ema_decay", 0.9999)))
    criterion = LatentUpsamplerLoss(**config.get("loss", {}))

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer=optimizer, ema=ema, map_location=device)

    precision = config["train"].get("precision", "bf16")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    max_steps = int(config["train"]["max_steps"])
    warmup_clean_steps = int(config["train"].get("warmup_clean_steps", 0))
    grad_accum = int(config["train"].get("grad_accum", 1))
    log_every = int(config["train"].get("log_every", 20))
    save_every = int(config["train"].get("save_every", 1000))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))

    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}

    for step in progress:
        model.train()
        for accum_index in range(grad_accum):
            batch = next(batches)
            x_t_lr = batch["x_t_lr"].to(device, non_blocking=True)
            sigma = batch["sigma"].to(device, non_blocking=True)
            z0_lr = batch["z0_lr"].to(device, non_blocking=True)
            z0_hr = batch["z0_hr"].to(device, non_blocking=True)

            if step < warmup_clean_steps:
                x_t_lr = z0_lr
                sigma = torch.zeros_like(sigma)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred = model(x_t_lr, sigma)
                loss, loss_items = criterion(pred.float(), z0_hr.float(), z0_lr.float())
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            for name, value in loss_items.items():
                running[name] = running.get(name, 0.0) + float(value)
            running["loss"] = running.get("loss", 0.0) + float(loss.detach()) * grad_accum

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        actual_step = step + 1
        if actual_step % log_every == 0:
            denom = log_every * grad_accum
            postfix = {key: value / denom for key, value in running.items()}
            postfix["stage"] = 0.0 if actual_step <= warmup_clean_steps else 1.0
            progress.set_postfix({k: f"{v:.4f}" for k, v in postfix.items()})
            running.clear()

        if actual_step % save_every == 0 or actual_step == max_steps:
            save_checkpoint(out_dir / f"step_{actual_step:07d}.pt", model, optimizer, ema, actual_step, config)
            save_checkpoint(out_dir / "latest.pt", model, optimizer, ema, actual_step, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_wan21_x2_512.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--scale", type=int)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--hidden_channels", type=int)
    parser.add_argument("--num_res_blocks", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--sigma_mode", choices=["mid", "uniform", "clean"])
    parser.add_argument("--warmup_clean_steps", type=int)
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update(
        {
            "data_dir": "data/latent_pairs_wan21_512",
            "out_dir": "outputs/wan_traj_upsampler_x2",
            "model": {
                "in_channels": 16,
                "out_channels": 16,
                "hidden_channels": 256,
                "num_res_blocks": 8,
                "sigma_embed_dim": 256,
                "scale": 2,
            },
            "train": {
                "max_steps": 100000,
                "warmup_clean_steps": 5000,
                "batch_size": 1,
                "num_workers": 8,
                "grad_accum": 8,
                "lr": 1e-4,
                "weight_decay": 0.01,
                "precision": "bf16",
                "ema_decay": 0.9999,
                "log_every": 20,
                "save_every": 1000,
                "seed": 1234,
            },
            "sigma": {"mode": "mid"},
            "loss": {},
        },
        config,
    )
    for key in ("data_dir", "out_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("scale", "in_channels", "hidden_channels", "num_res_blocks"):
        value = getattr(args, key)
        if value is not None:
            config["model"][key] = value
    for key in ("batch_size", "grad_accum", "lr", "max_steps", "precision", "warmup_clean_steps"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    if args.sigma_mode is not None:
        config["sigma"]["mode"] = args.sigma_mode
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
