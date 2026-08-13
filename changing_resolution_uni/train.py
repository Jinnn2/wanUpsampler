from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.training.checkpoint import load_checkpoint, save_checkpoint
from wan_sr.training.ema import EMA
from wan_sr.training.config import load_yaml
from changing_resolution_uni.data import ScaleBucketBatchSampler, UniversalCleanLatentDataset
from changing_resolution_uni.losses import UniversalCleanUpsampleLoss
from changing_resolution_uni.model import UniversalCleanLatentUpsampler


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    apply_overrides(config, args)
    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["train"].get("seed", 1234)) + rank)
    out_dir = Path(config["out_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, out_dir / "train_config.yaml")
    barrier(world_size)

    dataset = UniversalCleanLatentDataset(config["data_dir"], seed=int(config["train"].get("seed", 1234)))
    train_set, val_set = split_dataset(dataset, config)
    sampler = ScaleBucketBatchSampler(
        train_set.dataset if isinstance(train_set, Subset) else train_set,
        int(config["train"]["batch_size"]),
        seed=int(config["train"].get("seed", 1234)),
        rank=rank,
        world_size=world_size,
    )
    if isinstance(train_set, Subset):
        # Keep validation split deterministic while mapping Subset indices into the bucket sampler.
        sampler = ScaleBucketBatchSampler(IndexedDataset(train_set), int(config["train"]["batch_size"]), seed=int(config["train"].get("seed", 1234)), rank=rank, world_size=world_size)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=int(config["train"].get("num_workers", 4)), pin_memory=device.type == "cuda")
    if len(loader) == 0:
        raise RuntimeError("Empty training loader")

    raw_model = UniversalCleanLatentUpsampler(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=float(config["train"]["lr"]), weight_decay=float(config["train"].get("weight_decay", 0.01)))
    ema = EMA(raw_model, decay=float(config["train"].get("ema_decay", 0.9999)))
    criterion = UniversalCleanUpsampleLoss(**config.get("loss", {}))
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, raw_model, optimizer=optimizer, ema=ema, map_location=device)
    model = DistributedDataParallel(raw_model, device_ids=[local_rank]) if world_size > 1 and device.type == "cuda" else raw_model
    grad_accum = int(config["train"].get("grad_accum", 1))
    max_steps = int(config["train"]["max_steps"])
    data_epoch = 0
    sampler.set_epoch(data_epoch)
    iterator = iter(loader)
    step = start_step
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        metric_sum: dict[str, float] = {}
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                data_epoch += 1
                sampler.set_epoch(data_epoch)
                iterator = iter(loader)
                batch = next(iterator)
            lr = batch["z0_lr"].to(device, non_blocking=True)
            hr = batch["z0_hr"].to(device, non_blocking=True)
            target_size = (int(hr.shape[-2]), int(hr.shape[-1]))
            autocast = device.type == "cuda" and config["train"].get("precision") in {"bf16", "fp16"}
            dtype = torch.bfloat16 if config["train"].get("precision") == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                prediction = model(lr, output_size=target_size)
                loss, items = criterion(prediction.float(), hr.float(), lr.float())
            (loss / grad_accum).backward()
            for name, value in items.items():
                metric_sum[name] = metric_sum.get(name, 0.0) + float(value) / grad_accum
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(config["train"].get("grad_clip_norm", 1.0)))
        optimizer.step()
        ema.update(raw_model)
        step += 1
        if rank == 0 and (step % int(config["train"].get("log_every", 20)) == 0 or step == 1):
            payload = {"step": step, "split": "train", **metric_sum}
            print(f"step={step} " + " ".join(f"{k}={v:.6f}" for k, v in metric_sum.items()), flush=True)
            append_metrics(out_dir / "metrics.jsonl", payload)
        eval_every = int(config["train"].get("eval_every", 0))
        if len(val_set) > 0 and eval_every > 0 and step % eval_every == 0:
            barrier(world_size)
            if rank == 0:
                values = evaluate(raw_model, ema, val_set, criterion, device, config)
                print(f"val step={step} " + " ".join(f"{k}={v:.6f}" for k, v in values.items()), flush=True)
                append_metrics(out_dir / "metrics.jsonl", {"step": step, "split": "val", **values})
            barrier(world_size)
        if rank == 0 and step % int(config["train"].get("save_every", 1000)) == 0:
            save_checkpoint(out_dir / f"step_{step:07d}.pt", raw_model, optimizer, ema, step, config)
    if rank == 0:
        save_checkpoint(out_dir / "last.pt", raw_model, optimizer, ema, step, config)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


class IndexedDataset:
    def __init__(self, subset: Subset):
        self.subset = subset
        self.dataset = subset.dataset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, index):
        return self.subset[index]
    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
    def selected_scale(self, index):
        return self.dataset.selected_scale(self.subset.indices[index])
    def bucket_key(self, index):
        return self.dataset.bucket_key(self.subset.indices[index])


def split_dataset(dataset, config):
    ratio = float(config["train"].get("val_ratio", 0.0))
    if ratio <= 0 or len(dataset) < 2:
        return dataset, Subset(dataset, [])
    source_count = int(getattr(dataset, "num_source_samples", len(dataset)))
    count = min(source_count - 1, max(1, int(round(source_count * ratio))))
    if int(config["train"].get("val_max_samples", 0)) > 0:
        count = min(count, int(config["train"]["val_max_samples"]))
    rng = random.Random(int(config["train"].get("seed", 1234)))
    source_ids = list(range(source_count)); rng.shuffle(source_ids)
    val_sources = sorted(source_ids[:count])
    train_sources = sorted(source_ids[count:])
    return Subset(dataset, dataset.virtual_indices_for_sources(train_sources)), Subset(dataset, dataset.virtual_indices_for_sources(val_sources))


@torch.no_grad()
def evaluate(model, ema, dataset, criterion, device, config):
    proxy = IndexedDataset(dataset) if isinstance(dataset, Subset) else dataset
    sampler = ScaleBucketBatchSampler(proxy, int(config["train"]["batch_size"]), shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=int(config["train"].get("num_workers", 4)), pin_memory=device.type == "cuda")
    backup = {name: param.detach().clone() for name, param in model.named_parameters() if name in ema.shadow}
    ema.copy_to(model)
    model.eval()
    totals = {}; count = 0
    try:
        for batch_index, batch in enumerate(loader):
            if int(config["train"].get("val_batches", 0)) > 0 and batch_index >= int(config["train"]["val_batches"]):
                break
            lr = batch["z0_lr"].to(device, non_blocking=True); hr = batch["z0_hr"].to(device, non_blocking=True)
            prediction = model(lr, output_size=hr.shape[-2:])
            _, items = criterion(prediction.float(), hr.float(), lr.float())
            batch_n = int(lr.shape[0]); count += batch_n
            for name, value in items.items(): totals[name] = totals.get(name, 0.0) + float(value) * batch_n
    finally:
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name, value in backup.items(): params[name].copy_(value)
        model.train()
    return {name: value / max(count, 1) for name, value in totals.items()}


def append_metrics(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    return rank, world, local


def barrier(world_size):
    if world_size > 1: dist.barrier()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def apply_overrides(config, args):
    if args.data_dir: config["data_dir"] = args.data_dir
    if args.out_dir: config["out_dir"] = args.out_dir
    if args.max_steps is not None: config["train"]["max_steps"] = args.max_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--data_dir"); p.add_argument("--out_dir"); p.add_argument("--max_steps", type=int); p.add_argument("--resume")
    return p.parse_args()


if __name__ == "__main__":
    main()
