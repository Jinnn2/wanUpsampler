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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import CleanLatentPairDataset
from wan_sr.losses import CleanLatentResizeLoss
from wan_sr.models import WanCleanLatentResizer
from wan_sr.training.checkpoint import load_checkpoint, save_checkpoint
from wan_sr.training.config import deep_update, load_yaml
from wan_sr.training.ema import EMA


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_yaml(args.config), args)

    set_seed(int(config["train"].get("seed", 1234)))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        shutil.copy2(args.config, out_dir / "train_config.yaml")

    dataset = CleanLatentPairDataset(config["data_dir"], strict_channels=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    batches = itertools.cycle(loader)

    model = WanCleanLatentResizer(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    ema = EMA(model, decay=float(config["train"].get("ema_decay", 0.9999)))
    criterion = CleanLatentResizeLoss(**config.get("loss", {}))

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer=optimizer, ema=ema, map_location=device)

    precision = config["train"].get("precision", "bf16")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    max_steps = int(config["train"]["max_steps"])
    grad_accum = int(config["train"].get("grad_accum", 1))
    log_every = int(config["train"].get("log_every", 20))
    save_every = int(config["train"].get("save_every", 1000))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))

    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}

    for step in progress:
        model.train()
        for _ in range(grad_accum):
            batch = next(batches)
            z0_lr = batch["z0_lr"].to(device, non_blocking=True)
            z0_hr = batch["z0_hr"].to(device, non_blocking=True)
            target_spatial = (z0_hr.shape[-2], z0_hr.shape[-1])

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred = model(z0_lr, output_size=target_spatial)
                loss, loss_items = criterion(pred.float(), z0_hr.float(), z0_lr.float())
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            for name, value in loss_items.items():
                running[name] = running.get(name, 0.0) + float(value)

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
            progress.set_postfix({key: f"{value:.4f}" for key, value in postfix.items()})
            running.clear()

        if actual_step % save_every == 0 or actual_step == max_steps:
            save_checkpoint(out_dir / f"step_{actual_step:07d}.pt", model, optimizer, ema, actual_step, config)
            save_checkpoint(out_dir / "latest.pt", model, optimizer, ema, actual_step, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="changing_resolution/configs/train_clean_480p_to_720p.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--hidden_channels", type=int)
    parser.add_argument("--num_res_blocks", type=int)
    parser.add_argument("--scale_factor", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update(
        {
            "data_dir": "data/changing_resolution/latent_pairs_480p720p",
            "out_dir": "outputs/changing_resolution_clean_480p720p",
            "model": {
                "in_channels": 16,
                "out_channels": 16,
                "hidden_channels": 256,
                "num_res_blocks": 8,
                "scale_factor": 1.5,
                "residual_skip": True,
            },
            "train": {
                "max_steps": 100000,
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
                "grad_clip_norm": 1.0,
            },
            "loss": {},
        },
        config,
    )
    for key in ("data_dir", "out_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("hidden_channels", "num_res_blocks", "scale_factor"):
        value = getattr(args, key)
        if value is not None:
            config["model"][key] = value
    for key in ("batch_size", "grad_accum", "lr", "max_steps", "precision"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
