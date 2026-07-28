from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_distill.scripts.train.train_last_step_skip_lora import (  # noqa: E402
    append_metrics,
    average_trainable_gradients,
    barrier_if_distributed,
    build_wan_training_module,
    cleanup_distributed,
    dedupe_keep_order,
    expand_model_path_entry,
    get_training_device,
    init_distributed,
    load_resume,
    log_main,
    log_rank_device,
    normalize_model_paths,
    reduce_running_metrics,
    set_seed,
    split_dataset,
    strip_prefix,
)
from wan_sr.data import TailSkipLoRALMDBDataset  # noqa: E402
from wan_sr.losses.latent_losses import temporal_difference_loss  # noqa: E402
from wan_sr.training.config import deep_update, load_yaml  # noqa: E402


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_yaml(args.config), args)
    if str(config["train"].get("training_mode", "cached_x_pre_step")) != "cached_x_pre_step":
        raise ValueError("This trainer only supports training_mode=cached_x_pre_step.")

    dist_ctx = init_distributed()
    try:
        set_seed(int(config["train"].get("seed", 1234)) + dist_ctx.rank)
        torch.set_float32_matmul_precision("high")

        device = get_training_device(dist_ctx)
        log_rank_device(dist_ctx, device)
        precision = str(config["train"].get("precision", "bf16"))
        autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

        out_dir = Path(config["output"]["out_dir"])
        if dist_ctx.is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.config, out_dir / "train_config.yaml")
        barrier_if_distributed(dist_ctx)

        dataset = build_training_dataset(config)
        validate_tail_skip_metadata(dataset, config)
        train_dataset, val_dataset = split_dataset(dataset, config)
        log_main(
            dist_ctx,
            f"dataset={len(dataset)} train={len(train_dataset)} val={len(val_dataset)} "
            f"world_size={dist_ctx.world_size}",
        )

        module = build_wan_training_module(config, device=device)
        module.train()
        params = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found. Check lora_base_model and lora_target_modules.")
        log_main(dist_ctx, f"trainable_params={sum(param.numel() for _, param in params):,}")

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

        optimizer = torch.optim.AdamW(
            [param for _, param in params],
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"].get("weight_decay", 0.01)),
        )

        start_step = 0
        if args.resume:
            start_step = load_resume(Path(args.resume), module, optimizer, device)

        max_steps = int(config["train"]["max_steps"])
        grad_accum = int(config["train"].get("grad_accum", 1))
        log_every = int(config["train"].get("log_every", 10))
        save_every = int(config["train"].get("save_every", 500))
        eval_every = int(config["train"].get("eval_every", 500))
        grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))
        metrics_path = out_dir / "metrics.jsonl"
        running: dict[str, float] = {}

        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            range(start_step, max_steps),
            initial=start_step,
            total=max_steps,
            dynamic_ncols=True,
            disable=not dist_ctx.is_main,
        )
        for step in progress:
            module.train()
            for _ in range(grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    data_epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(data_epoch)
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    loss, loss_items = compute_batch_loss(module, batch, config, device)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                for key, value in loss_items.items():
                    running[key] = running.get(key, 0.0) + float(value)

            average_trainable_gradients(params, dist_ctx)
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for _, param in params], grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            actual_step = step + 1
            if actual_step % log_every == 0:
                denom = log_every * grad_accum
                payload = reduce_running_metrics(running, denom, dist_ctx, device)
                if dist_ctx.is_main:
                    progress.set_postfix({key: f"{value:.5f}" for key, value in payload.items()})
                    append_metrics(metrics_path, {"step": actual_step, "split": "train", **payload})
                running.clear()

            should_eval = len(val_dataset) > 0 and eval_every > 0 and (actual_step % eval_every == 0 or actual_step == max_steps)
            if should_eval and dist_ctx.is_main:
                val_items = evaluate(module, val_dataset, config, device=device, precision=precision)
                progress.write(
                    "val " + " ".join(f"{key}={value:.6f}" for key, value in val_items.items()) + f" step={actual_step}"
                )
                append_metrics(metrics_path, {"step": actual_step, "split": "val", **val_items})
            if should_eval:
                barrier_if_distributed(dist_ctx)

            should_save = actual_step % save_every == 0 or actual_step == max_steps
            if should_save and dist_ctx.is_main:
                save_training_state(out_dir, module, optimizer, actual_step, config)
            if should_save:
                barrier_if_distributed(dist_ctx)
    finally:
        cleanup_distributed(dist_ctx)


def build_training_dataset(config: dict) -> Dataset:
    dataset: Dataset = TailSkipLoRALMDBDataset(config["data"]["lmdb_dir"], strict_channels=True)
    max_samples = config["data"].get("max_samples")
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(int(max_samples), len(dataset)))))
    return dataset


def compute_batch_loss(
    module: torch.nn.Module,
    batch: dict[str, Any],
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    pipe = module.pipe
    x_pre_step = batch["x_pre_step_lr"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    target = batch["z_final_lr_teacher"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    prompts = list(batch["prompt"])
    sigmas = extract_train_sigma(batch).to(device=device, dtype=torch.float32)
    timesteps = extract_train_timestep(batch).to(device=device, dtype=pipe.torch_dtype)

    flow_pred = predict_flow(module, x_pre_step, timesteps, prompts, config)
    pred = x_pre_step.float() - sigmas.view(-1, 1, 1, 1, 1) * flow_pred.float()
    target_f = target.float()
    l1 = F.l1_loss(pred, target_f)
    mse = F.mse_loss(pred, target_f)

    loss_cfg = config.get("loss", {})
    total = float(loss_cfg.get("l1_weight", 1.0)) * l1 + float(loss_cfg.get("mse_weight", 0.1)) * mse
    items = {"loss": float(total.detach()), "clean_l1": float(l1.detach()), "clean_mse": float(mse.detach())}

    temporal_weight = float(loss_cfg.get("temporal_weight", 0.0) or 0.0)
    if temporal_weight > 0:
        temporal_l1 = temporal_difference_loss(pred, target_f)
        total = total + temporal_weight * temporal_l1
        items.update({"loss": float(total.detach()), "clean_temporal_l1": float(temporal_l1.detach())})

    velocity_mse_weight = float(loss_cfg.get("velocity_mse_weight", 0.0) or 0.0)
    velocity_l1_weight = float(loss_cfg.get("velocity_l1_weight", 0.0) or 0.0)
    if velocity_mse_weight > 0 or velocity_l1_weight > 0:
        target_flow = (x_pre_step.float() - target_f) / sigmas.view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        velocity_mse = F.mse_loss(flow_pred.float(), target_flow.detach())
        velocity_l1 = F.l1_loss(flow_pred.float(), target_flow.detach())
        total = total + velocity_mse_weight * velocity_mse + velocity_l1_weight * velocity_l1
        items.update(
            {
                "loss": float(total.detach()),
                "velocity_mse": float(velocity_mse.detach()),
                "velocity_l1": float(velocity_l1.detach()),
            }
        )
    return total, items


def predict_flow(
    module: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompts: list[str],
    config: dict,
) -> torch.Tensor:
    pipe = module.pipe
    model_cfg = config.get("model", {})
    use_cfg = bool(model_cfg.get("enable_cfg", False))
    cond_context = encode_prompts(pipe, prompts)
    common = {
        "dit": pipe.dit,
        "latents": latents,
        "timestep": timesteps,
        "use_gradient_checkpointing": bool(config["train"].get("gradient_checkpointing", True)),
        "use_gradient_checkpointing_offload": bool(config["train"].get("gradient_checkpointing_offload", False)),
    }
    flow_cond = pipe.model_fn(context=cond_context, **common)
    if not use_cfg:
        return flow_cond
    negative_prompt = str(model_cfg.get("negative_prompt", ""))
    uncond_context = encode_prompts(pipe, [negative_prompt] * len(prompts))
    flow_uncond = pipe.model_fn(context=uncond_context, **common)
    guide_scale = float(model_cfg.get("sample_guide_scale", config.get("wan", {}).get("sample_guide_scale", 6.0)))
    return flow_uncond + guide_scale * (flow_cond - flow_uncond)


@torch.no_grad()
def encode_prompts(pipe: Any, prompts: list[str]) -> torch.Tensor:
    ids, mask = pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, valid_len in enumerate(seq_lens):
        prompt_emb[i, valid_len:] = 0
    return prompt_emb.to(dtype=pipe.torch_dtype, device=pipe.device)


def extract_train_sigma(batch: dict[str, Any]) -> torch.Tensor:
    sigmas = []
    for meta in parse_batch_meta(batch):
        recipe = meta.get("tail_skip_recipe", {})
        if "train_sigma" not in recipe:
            raise KeyError("Missing tail_skip_recipe.train_sigma in LMDB metadata.")
        sigmas.append(float(recipe["train_sigma"]))
    return torch.tensor(sigmas, dtype=torch.float32)


def extract_train_timestep(batch: dict[str, Any]) -> torch.Tensor:
    timesteps = []
    for meta in parse_batch_meta(batch):
        recipe = meta.get("tail_skip_recipe", {})
        if "train_timestep" in recipe:
            timesteps.append(float(recipe["train_timestep"]))
        elif "train_sigma" in recipe:
            timesteps.append(float(recipe["train_sigma"]) * 1000.0)
        else:
            raise KeyError("Missing tail_skip_recipe.train_timestep/train_sigma in LMDB metadata.")
    return torch.tensor(timesteps, dtype=torch.float32)


def parse_batch_meta(batch: dict[str, Any]) -> list[dict[str, Any]]:
    meta_json = batch["meta_json"]
    if isinstance(meta_json, str):
        meta_items = [meta_json]
    else:
        meta_items = list(meta_json)
    return [json.loads(text) for text in meta_items]


@torch.no_grad()
def evaluate(module: torch.nn.Module, dataset: Dataset, config: dict, *, device: torch.device, precision: str) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    max_batches = int(config["train"].get("val_batches", 0))
    totals: dict[str, float] = {}
    count = 0
    module.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            _, items = compute_batch_loss(module, batch, config, device)
        batch_size = int(batch["x_pre_step_lr"].shape[0])
        count += batch_size
        for key, value in items.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    module.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def validate_tail_skip_metadata(dataset: Dataset, config: dict, max_checks: int = 16) -> None:
    expected_train_step = config.get("wan", {}).get("train_step") or config.get("task", {}).get("train_step")
    expected_infer_steps = config.get("wan", {}).get("infer_steps") or config.get("task", {}).get("target_step")
    if expected_train_step is None and expected_infer_steps is None:
        return
    checked = 0
    for idx in range(min(len(dataset), max_checks)):
        row = dataset[idx]
        meta = json.loads(row["meta_json"])
        recipe = meta.get("tail_skip_recipe", {})
        if expected_train_step is not None and int(recipe.get("train_step", -1)) != int(expected_train_step):
            raise ValueError(
                f"Tail-skip LMDB train_step mismatch: requested {expected_train_step}, "
                f"sample {row['sample_id']} has {recipe.get('train_step')}"
            )
        if expected_infer_steps is not None and int(recipe.get("infer_steps", -1)) != int(expected_infer_steps):
            raise ValueError(
                f"Tail-skip LMDB infer_steps mismatch: requested {expected_infer_steps}, "
                f"sample {row['sample_id']} has {recipe.get('infer_steps')}"
            )
        checked += 1
    print(f"validated tail-skip LMDB metadata on {checked} sample(s)", flush=True)


def save_training_state(out_dir: Path, module: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, config: dict) -> None:
    trainable_state = {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    remove_prefix = config["output"].get("remove_prefix_in_ckpt") or config["model"].get("remove_prefix_in_ckpt") or ""
    lora_state = {
        strip_prefix(name, remove_prefix): tensor
        for name, tensor in trainable_state.items()
    }
    metadata = {
        "step": str(step),
        "recipe": str(config["task"]["recipe"]),
        "input_key": str(config["task"].get("input_key", "x_pre_step_lr")),
        "target_key": str(config["task"].get("target_key", "z_final_lr_teacher")),
        "base_model_id": str(config["model"].get("base_model_id", "")),
        "train_step": str(config.get("wan", {}).get("train_step", config["task"].get("train_step", ""))),
        "target_step": str(config["task"].get("target_step", config.get("wan", {}).get("infer_steps", ""))),
        "lora_rank": str(config["model"].get("lora_rank", "")),
        "lora_target_modules": str(config["model"].get("lora_target_modules", "")),
        "temporal_weight": str(config.get("loss", {}).get("temporal_weight", 0.0)),
    }
    safetensors_path = out_dir / f"step_{step:07d}.safetensors"
    save_file(lora_state, safetensors_path, metadata=metadata)
    save_file(lora_state, out_dir / "latest.safetensors", metadata=metadata)
    torch.save(
        {
            "step": step,
            "trainable_state": trainable_state,
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        out_dir / "latest.pt",
    )
    print(f"saved {safetensors_path}", flush=True)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update(
        {
            "task": {
                "name": "tail_skip_lora",
                "recipe": "wan21_50step_tail_skip_lora_step45_to_step50",
                "input_key": "x_pre_step_lr",
                "target_key": "z_final_lr_teacher",
                "train_step": 45,
                "target_step": 50,
            },
            "data": {"lmdb_dir": "data/changing_resolution/lmdb_tail_skip_lora_step45_to_step50"},
            "model": {
                "lora_base_model": "dit",
                "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
                "lora_rank": 32,
                "enable_cfg": True,
                "sample_guide_scale": 6.0,
                "negative_prompt": "",
            },
            "wan": {"infer_steps": 50, "train_step": 45, "sample_shift": 8, "sample_guide_scale": 6},
            "train": {
                "batch_size": 1,
                "grad_accum": 8,
                "lr": 5e-5,
                "max_steps": 10000,
                "precision": "bf16",
                "training_mode": "cached_x_pre_step",
            },
            "loss": {
                "l1_weight": 1.0,
                "mse_weight": 0.1,
                "temporal_weight": 0.0,
                "velocity_mse_weight": 0.0,
                "velocity_l1_weight": 0.0,
            },
            "output": {"out_dir": "outputs/changing_resolution_tail_skip_lora_step45_to_step50"},
        },
        config,
    )
    if args.data_dir is not None:
        config["data"]["lmdb_dir"] = args.data_dir
    if args.out_dir is not None:
        config["output"]["out_dir"] = args.out_dir
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    for key in ("batch_size", "grad_accum", "lr", "max_steps", "precision", "training_mode"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    if args.train_step is not None:
        config.setdefault("wan", {})["train_step"] = int(args.train_step)
        config.setdefault("task", {})["train_step"] = int(args.train_step)
    if args.model_paths is not None:
        config["model"]["model_paths"] = args.model_paths
    if args.model_id_with_origin_paths is not None:
        config["model"]["model_id_with_origin_paths"] = args.model_id_with_origin_paths
    if args.tokenizer_path is not None:
        config["model"]["tokenizer_path"] = args.tokenizer_path
    if args.lora_rank is not None:
        config["model"]["lora_rank"] = args.lora_rank
    if args.lora_target_modules is not None:
        config["model"]["lora_target_modules"] = args.lora_target_modules
    if args.lora_checkpoint is not None:
        config["model"]["lora_checkpoint"] = args.lora_checkpoint
    if args.enable_cfg is not None:
        config["model"]["enable_cfg"] = bool(args.enable_cfg)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="changing_resolution/configs/train_tail_skip_lora_step45.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--training_mode", choices=["cached_x_pre_step"])
    parser.add_argument("--train_step", type=int)
    parser.add_argument("--model_paths")
    parser.add_argument("--model_id_with_origin_paths")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_target_modules")
    parser.add_argument("--lora_checkpoint")
    parser.add_argument("--enable_cfg", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    main()
