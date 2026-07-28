"""Optimize ITU with the objective and split protocol in paper Secs. 3.2 and 4."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
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


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def main() -> None:
    args = parse_args()
    # 先读 YAML，再用命令行参数覆盖；这样 bash 脚本里的 MAX_STEPS/LR 等会优先生效。
    config = apply_cli_overrides(load_yaml(args.config), args)
    dist_ctx = init_distributed()
    try:
        # 所有 rank 使用相同初始化；DistributedSampler 负责让每张卡读取不同数据切片。
        set_seed(int(config["train"].get("seed", 1234)))
        torch.set_float32_matmul_precision("high")
        device = get_training_device(dist_ctx)
        log_rank_device(dist_ctx, device)

        out_dir = Path(config["out_dir"])
        if dist_ctx.is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.config:
                shutil.copy2(args.config, out_dir / "train_config.yaml")
        barrier_if_distributed(dist_ctx)

        # Stage 2 训练的是 clean latent pair：z0_lr -> z0_hr，不在这里解码 RGB。
        dataset = build_dataset(config)
        train_dataset, val_dataset = split_dataset(dataset, config)
        log_main(
            dist_ctx,
            f"dataset={len(dataset)} train={len(train_dataset)} val={len(val_dataset)} "
            f"format={config.get('data_format', 'files')} world_size={dist_ctx.world_size}",
        )

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=dist_ctx.world_size,
                rank=dist_ctx.rank,
                shuffle=True,
                seed=int(config["train"].get("seed", 1234)),
                drop_last=True,
            )
            if dist_ctx.enabled
            else None
        )
        loader = DataLoader(
            train_dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=int(config["train"].get("num_workers", 4)),
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        if len(loader) == 0:
            raise RuntimeError("Training DataLoader is empty. Reduce batch_size or provide more samples.")
        loader_iter = iter(loader)
        data_epoch = 0

        # 这里既支持原 1.5x rational 模型，也支持 2x pixel-shuffle + crop 模型。
        raw_model = WanCleanLatentResizerStage2(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"].get("weight_decay", 0.01)),
        )
        ema = EMA(raw_model, decay=float(config["train"].get("ema_decay", 0.9999)))
        criterion = CleanLatentResizeLoss(**config.get("loss", {}))

        start_step = 0
        if args.resume:
            start_step = load_checkpoint(
                args.resume,
                raw_model,
                optimizer=optimizer,
                ema=ema,
                map_location=device,
            )

        model: torch.nn.Module
        if dist_ctx.enabled:
            model = DistributedDataParallel(
                raw_model,
                device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
                output_device=dist_ctx.local_rank if device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model = raw_model

        precision = config["train"].get("precision", "bf16")
        autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
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
        best_val = load_best_val(out_dir / "best_val.json") if dist_ctx.is_main else float("inf")

        progress = tqdm(
            range(start_step, max_steps),
            initial=start_step,
            total=max_steps,
            dynamic_ncols=True,
            disable=not dist_ctx.is_main,
        )
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}

        for step in progress:
            model.train()
            for micro_step in range(grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    data_epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(data_epoch)
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                z0_lr = batch["z0_lr"].to(device, non_blocking=True)
                z0_hr = batch["z0_hr"].to(device, non_blocking=True)
                target_spatial = (z0_hr.shape[-2], z0_hr.shape[-1])
                should_sync = not dist_ctx.enabled or micro_step == grad_accum - 1
                sync_context = contextlib.nullcontext() if should_sync else model.no_sync()
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        pred = model(z0_lr, output_size=target_spatial)
                        loss, loss_items = criterion(pred.float(), z0_hr.float(), z0_lr.float())
                        loss = loss / grad_accum
                    scaler.scale(loss).backward()

                for name, value in loss_items.items():
                    running[name] = running.get(name, 0.0) + float(value)

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_model)

            actual_step = step + 1
            if actual_step % log_every == 0:
                postfix = reduce_running_metrics(
                    running,
                    denom=log_every * grad_accum,
                    dist_ctx=dist_ctx,
                    device=device,
                )
                if dist_ctx.is_main:
                    progress.set_postfix({key: f"{value:.4f}" for key, value in postfix.items()})
                    append_metrics(metrics_path, {"step": actual_step, "split": "train", **postfix})
                running.clear()

            should_eval = (
                len(val_dataset) > 0
                and eval_every > 0
                and (actual_step % eval_every == 0 or actual_step == max_steps)
            )
            if should_eval and dist_ctx.is_main:
                val_items = evaluate(
                    raw_model,
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
                    save_checkpoint(out_dir / "best_val.pt", raw_model, optimizer, ema, actual_step, config)
                    with (out_dir / "best_val.json").open("w", encoding="utf-8") as f:
                        json.dump({"step": actual_step, **val_items}, f, ensure_ascii=False, indent=2)
            if should_eval:
                barrier_if_distributed(dist_ctx)

            should_save = actual_step % save_every == 0 or actual_step == max_steps
            if should_save and dist_ctx.is_main:
                save_checkpoint(
                    out_dir / f"step_{actual_step:07d}.pt",
                    raw_model,
                    optimizer,
                    ema,
                    actual_step,
                    config,
                )
                save_checkpoint(out_dir / "latest.pt", raw_model, optimizer, ema, actual_step, config)
            if should_save:
                barrier_if_distributed(dist_ctx)
    finally:
        cleanup_distributed(dist_ctx)


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
    parser.add_argument("--num_workers", type=int)
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
    for key in ("batch_size", "grad_accum", "num_workers", "lr", "max_steps", "ema_decay", "precision"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    return config


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if not enabled:
        return DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = os.environ.get("DIST_BACKEND")
    if not backend:
        backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    if torch.cuda.is_available() and backend != "nccl":
        raise RuntimeError("CUDA distributed Stage2 training requires the NCCL backend.")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(dist_ctx: DistributedContext) -> None:
    if dist_ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier_if_distributed(dist_ctx: DistributedContext) -> None:
    if dist_ctx.enabled:
        dist.barrier()


def log_main(dist_ctx: DistributedContext, message: str) -> None:
    if dist_ctx.is_main:
        print(message, flush=True)


def log_rank_device(dist_ctx: DistributedContext, device: torch.device) -> None:
    current_cuda = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(
        "rank_device "
        f"rank={dist_ctx.rank} local_rank={dist_ctx.local_rank} world_size={dist_ctx.world_size} "
        f"device={device} current_cuda={current_cuda} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )


def reduce_running_metrics(
    running: dict[str, float],
    *,
    denom: int,
    dist_ctx: DistributedContext,
    device: torch.device,
) -> dict[str, float]:
    if not running:
        return {}
    keys = sorted(running)
    values = torch.tensor([running[key] for key in keys], dtype=torch.float64, device=device)
    if dist_ctx.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    normalizer = float(denom * dist_ctx.world_size)
    return {key: float(value) / normalizer for key, value in zip(keys, values.detach().cpu().tolist())}


def get_training_device(dist_ctx: DistributedContext) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", dist_ctx.local_rank) if dist_ctx.enabled else torch.device("cuda")
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
