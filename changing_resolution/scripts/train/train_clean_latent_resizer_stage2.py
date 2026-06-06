from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import CleanLatentLMDBDataset, CleanLatentPairDataset
from wan_sr.losses import CleanLatentResizeLoss
from wan_sr.models import WanCleanLatentResizerStage2
from wan_sr.training.checkpoint import load_checkpoint, save_checkpoint
from wan_sr.training.config import deep_update, load_yaml
from wan_sr.training.ema import EMA


def main() -> None:
    args = parse_args()
    # 先读 YAML，再用命令行参数覆盖；这样 bash 脚本里的 MAX_STEPS/LR 等会优先生效。
    config = apply_cli_overrides(load_yaml(args.config), args)

    # 固定随机种子，保证 train/val split 和 DataLoader shuffle 的主随机源可复现。
    set_seed(int(config["train"].get("seed", 1234)))
    torch.set_float32_matmul_precision("high")
    device = get_training_device()

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        # 把本次训练用到的原始配置一并归档，方便之后追溯 checkpoint 对应的超参。
        shutil.copy2(args.config, out_dir / "train_config.yaml")

    # Stage 2 训练的是 clean latent pair：z0_lr -> z0_hr，不在这里解码 RGB。
    dataset = build_dataset(config)
    train_dataset, val_dataset = split_dataset(dataset, config)
    print(
        f"dataset={len(dataset)} train={len(train_dataset)} val={len(val_dataset)} "
        f"format={config.get('data_format', 'files')}",
        flush=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        # 丢掉最后不足 batch 的样本，保证每个 micro-batch 的 loss 尺度一致。
        drop_last=True,
    )
    # 训练按 max_steps 控制，不按 epoch 控制；cycle 会在一个 epoch 结束后继续重洗 DataLoader。
    batches = itertools.cycle(loader)

    # WanCleanLatentResizerStage2 是 LTX2 风格的 1.5x 空间 clean-latent 放大器。
    model = WanCleanLatentResizerStage2(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    # EMA 保存一份平滑后的模型权重；验证默认用 EMA，通常比即时权重更稳。
    ema = EMA(model, decay=float(config["train"].get("ema_decay", 0.9999)))
    # Loss = HR latent fidelity + LR consistency + temporal difference + optional residual regularization。
    criterion = CleanLatentResizeLoss(**config.get("loss", {}))

    start_step = 0
    if args.resume:
        # 续训会同时恢复 model / optimizer / EMA，并从 checkpoint 记录的 step 继续。
        start_step = load_checkpoint(args.resume, model, optimizer=optimizer, ema=ema, map_location=device)

    precision = config["train"].get("precision", "bf16")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    # bf16 的动态范围较大，通常不需要 GradScaler；只有 fp16 时才启用 scaler。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    max_steps = int(config["train"]["max_steps"])
    grad_accum = int(config["train"].get("grad_accum", 1))
    log_every = int(config["train"].get("log_every", 20))
    save_every = int(config["train"].get("save_every", 1000))
    eval_every = int(config["train"].get("eval_every", 1000))
    val_batches = int(config["train"].get("val_batches", 0))
    eval_use_ema = bool(config["train"].get("eval_use_ema", True))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    metrics_path = out_dir / "metrics.jsonl"
    # 如果目录里已经有 best_val.json，继续沿用历史最优值，避免续训时覆盖更好的 best_val.pt。
    best_val = load_best_val(out_dir / "best_val.json")

    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}

    for step in progress:
        model.train()
        # 一个 optimizer step 内累积多个 micro-batch，等效 batch = batch_size * grad_accum。
        for _ in range(grad_accum):
            batch = next(batches)
            z0_lr = batch["z0_lr"].to(device, non_blocking=True)
            z0_hr = batch["z0_hr"].to(device, non_blocking=True)
            # 以真实 HR latent 的 H/W 作为目标尺寸；Stage 2 内部会检查它是否等于 1.5x。
            target_spatial = (z0_hr.shape[-2], z0_hr.shape[-1])

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred = model(z0_lr, output_size=target_spatial)
                # loss 内部会把 pred 再下采样到 z0_lr，约束低频/结构一致性。
                loss, loss_items = criterion(pred.float(), z0_hr.float(), z0_lr.float())
                # 梯度累积时必须除以 grad_accum，避免等效学习率被放大 grad_accum 倍。
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            # loss_items 是未除以 grad_accum 的可读指标，后面按 micro-batch 数再平均。
            for name, value in loss_items.items():
                running[name] = running.get(name, 0.0) + float(value)

        if grad_clip_norm > 0:
            # clip 前先 unscale，确保裁剪的是实际梯度范数而不是 fp16 scaler 放大后的梯度。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # 只在真实 optimizer step 后更新 EMA，而不是每个 micro-batch 更新。
        ema.update(model)

        actual_step = step + 1
        if actual_step % log_every == 0:
            denom = log_every * grad_accum
            postfix = {key: value / denom for key, value in running.items()}
            progress.set_postfix({key: f"{value:.4f}" for key, value in postfix.items()})
            # metrics.jsonl 每行一条 JSON，方便后续用脚本画 train/val 曲线。
            append_metrics(metrics_path, {"step": actual_step, "split": "train", **postfix})
            running.clear()

        if len(val_dataset) > 0 and eval_every > 0 and (actual_step % eval_every == 0 or actual_step == max_steps):
            # 验证默认使用 EMA 权重；maybe_ema_weights 会在验证后恢复训练中的即时权重。
            val_items = evaluate(
                model,
                ema,
                val_dataset,
                criterion,
                device=device,
                precision=precision,
                batch_size=int(config["train"]["batch_size"]),
                num_workers=int(config["train"].get("num_workers", 4)),
                max_batches=val_batches,
                use_ema=eval_use_ema,
            )
            progress.write(
                "val "
                + " ".join(f"{key}={value:.6f}" for key, value in val_items.items())
                + f" step={actual_step}"
            )
            append_metrics(metrics_path, {"step": actual_step, "split": "val", **val_items})
            if val_items["loss"] < best_val:
                best_val = val_items["loss"]
                # best_val.pt 保存当前即时权重、optimizer 和 EMA；推理时可再选择是否加载 EMA。
                save_checkpoint(out_dir / "best_val.pt", model, optimizer, ema, actual_step, config)
                with (out_dir / "best_val.json").open("w", encoding="utf-8") as f:
                    json.dump({"step": actual_step, **val_items}, f, ensure_ascii=False, indent=2)

        if actual_step % save_every == 0 or actual_step == max_steps:
            # step checkpoint 用于回滚和横向比较，latest.pt 用于续训入口。
            save_checkpoint(out_dir / f"step_{actual_step:07d}.pt", model, optimizer, ema, actual_step, config)
            save_checkpoint(out_dir / "latest.pt", model, optimizer, ema, actual_step, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data_dir")
    parser.add_argument("--data_format", choices=["files", "lmdb"])
    parser.add_argument("--out_dir")
    parser.add_argument("--hidden_channels", type=int)
    parser.add_argument("--num_res_blocks", type=int)
    parser.add_argument("--scale_factor", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--ema_decay", type=float)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--no_residual_skip", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    # 这里的默认值是代码兜底；实际训练优先使用 YAML，其次被命令行覆盖。
    config = deep_update(
        {
            "data_dir": "data/changing_resolution/lmdb_480p720p_1k",
            "data_format": "lmdb",
            "out_dir": "outputs/changing_resolution_clean_480p720p_stage2_lmdb",
            "model": {
                "in_channels": 16,
                "out_channels": 16,
                "hidden_channels": 256,
                "num_res_blocks": 8,
                "scale_factor": 1.5,
                "dropout": 0.0,
                "residual_skip": False,
                "resblock_type": "ltx2",
                "resize_op": "rational_conv3d_pixel_shuffle",
            },
            "train": {
                "max_steps": 50000,
                "batch_size": 1,
                "num_workers": 8,
                "grad_accum": 8,
                "lr": 1e-4,
                "weight_decay": 0.01,
                "precision": "bf16",
                "ema_decay": 0.9999,
                "log_every": 20,
                "save_every": 1000,
                "eval_every": 1000,
                "val_ratio": 0.05,
                "val_max_samples": 100,
                "val_batches": 0,
                "eval_use_ema": True,
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
    if args.data_format is not None:
        config["data_format"] = args.data_format
    for key in ("hidden_channels", "num_res_blocks", "scale_factor"):
        value = getattr(args, key)
        if value is not None:
            config["model"][key] = value
    if args.no_residual_skip:
        config["model"]["residual_skip"] = False
    for key in ("batch_size", "grad_accum", "lr", "max_steps", "ema_decay", "precision"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    return config


def get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("ALLOW_CPU_TRAINING") == "1":
        print("WARNING: CUDA is unavailable; ALLOW_CPU_TRAINING=1 so Stage 2 training will run on CPU.", flush=True)
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is unavailable, refusing to run Stage 2 training on CPU. "
        "Check NVIDIA driver / CUDA-compatible PyTorch / CUDA_VISIBLE_DEVICES. "
        "Set ALLOW_CPU_TRAINING=1 only for tiny smoke tests."
    )


def build_dataset(config: dict) -> Dataset:
    dataset_format = str(config.get("data_format", "files")).lower()
    if dataset_format == "files":
        # 兼容早期 safetensors 文件目录格式。
        return CleanLatentPairDataset(config["data_dir"], strict_channels=True)
    if dataset_format == "lmdb":
        # 当前 Stage 2 默认使用分片 LMDB，减少大量小文件读取开销。
        return CleanLatentLMDBDataset(config["data_dir"], strict_channels=True)
    raise ValueError(f"data_format must be 'files' or 'lmdb', got {dataset_format!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: Dataset, config: dict) -> tuple[Dataset, Dataset]:
    total = len(dataset)
    train_cfg = config.get("train", {})
    val_ratio = float(train_cfg.get("val_ratio", 0.0))
    val_max_samples = int(train_cfg.get("val_max_samples", 0))
    seed = int(train_cfg.get("seed", 1234))

    if val_ratio <= 0 or total < 2:
        return dataset, Subset(dataset, [])

    val_count = max(1, int(round(total * val_ratio)))
    if val_max_samples > 0:
        val_count = min(val_count, val_max_samples)
    val_count = min(val_count, total - 1)

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    # 排序不是为了随机性，而是让 Subset 访问顺序稳定；训练阶段 DataLoader 仍会 shuffle。
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    ema: EMA,
    dataset: Dataset,
    criterion: CleanLatentResizeLoss,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    use_ema: bool,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    totals: dict[str, float] = {}
    count = 0

    model.eval()
    with maybe_ema_weights(model, ema, enabled=use_ema):
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            z0_lr = batch["z0_lr"].to(device, non_blocking=True)
            z0_hr = batch["z0_hr"].to(device, non_blocking=True)
            target_spatial = (z0_hr.shape[-2], z0_hr.shape[-1])
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred = model(z0_lr, output_size=target_spatial)
                _, loss_items = criterion(pred.float(), z0_hr.float(), z0_lr.float())
            batch_size_actual = int(z0_lr.shape[0])
            count += batch_size_actual
            # 验证集可能最后一个 batch 不满，所以按真实 batch size 加权平均。
            for name, value in loss_items.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size_actual

    model.train()
    return {name: value / max(count, 1) for name, value in totals.items()}


@contextlib.contextmanager
def maybe_ema_weights(model: torch.nn.Module, ema: EMA, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    params = {name: param for name, param in model.named_parameters() if name in ema.shadow}
    # 临时把模型参数替换成 EMA 参数做验证，finally 中恢复即时训练权重。
    backup = {name: param.detach().clone() for name, param in params.items()}
    try:
        ema.copy_to(model)
        yield
    finally:
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(backup[name])


def append_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def load_best_val(path: Path) -> float:
    if not path.exists():
        return float("inf")
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return float(payload.get("loss", float("inf")))
    except Exception:
        return float("inf")


if __name__ == "__main__":
    main()
