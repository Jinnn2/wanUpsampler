from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_distill.scripts.train.train_last_step_skip_lora import (  # noqa: E402
    append_metrics,
    build_wan_training_module,
    encode_prompts,
    flow_euler_step,
    get_training_device,
    load_resume,
    make_deterministic_noise,
    predict_flow,
    save_training_state,
    set_seed,
)
from wan_sr.data import TeacherTrajectoryLoRALMDBDataset  # noqa: E402
from wan_sr.training.config import deep_update, load_yaml  # noqa: E402


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_yaml(args.config), args)
    set_seed(int(config["train"].get("seed", 1234)))
    torch.set_float32_matmul_precision("high")

    device = get_training_device()
    precision = str(config["train"].get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "train_config.yaml")

    dataset = build_training_dataset(config)
    max_samples = config["data"].get("max_samples")
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(int(max_samples), len(dataset)))))
    train_dataset, val_dataset = split_dataset(dataset, config)
    print(f"dataset={len(dataset)} train={len(train_dataset)} val={len(val_dataset)}", flush=True)

    module = build_wan_training_module(config, device=device)
    module.train()
    params = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check lora_base_model and lora_target_modules.")
    print(f"trainable_params={sum(param.numel() for _, param in params):,}", flush=True)

    loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("Training DataLoader is empty. Reduce batch_size or provide more samples.")
    batches = itertools.cycle(loader)

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
    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, dynamic_ncols=True)
    for step in progress:
        module.train()
        for _ in range(grad_accum):
            batch = next(batches)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                loss, loss_items = compute_batch_loss(module, batch, config, device)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            for key, value in loss_items.items():
                running[key] = running.get(key, 0.0) + float(value)

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([param for _, param in params], grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        actual_step = step + 1
        if actual_step % log_every == 0:
            denom = log_every * grad_accum
            payload = {key: value / denom for key, value in running.items()}
            progress.set_postfix({key: f"{value:.5f}" for key, value in payload.items()})
            append_metrics(metrics_path, {"step": actual_step, "split": "train", **payload})
            running.clear()

        if len(val_dataset) > 0 and eval_every > 0 and (actual_step % eval_every == 0 or actual_step == max_steps):
            val_items = evaluate(module, val_dataset, config, device=device, precision=precision)
            progress.write("val " + " ".join(f"{key}={value:.6f}" for key, value in val_items.items()) + f" step={actual_step}")
            append_metrics(metrics_path, {"step": actual_step, "split": "val", **val_items})

        if actual_step % save_every == 0 or actual_step == max_steps:
            save_training_state(out_dir, module, optimizer, actual_step, config)


def build_training_dataset(config: dict) -> Dataset:
    return TeacherTrajectoryLoRALMDBDataset(config["data"]["lmdb_dir"], strict_channels=True)


def compute_batch_loss(
    module: torch.nn.Module,
    batch: dict[str, Any],
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if str(config["train"].get("training_mode", "cached_teacher")) == "on_policy":
        return compute_on_policy_batch_loss(module, batch, config, device)
    return compute_cached_teacher_batch_loss(module, batch, config, device)


def compute_cached_teacher_batch_loss(
    module: torch.nn.Module,
    batch: dict[str, Any],
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    pipe = module.pipe
    x_pre = batch["x_pre_train_step"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    target = batch["z_teacher_final"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    prompts = list(batch["prompt"])
    sigmas = extract_train_sigma(batch).to(device=device, dtype=torch.float32)
    timesteps = (sigmas * 1000.0).to(device=device, dtype=pipe.torch_dtype)
    context = encode_prompts(pipe, prompts)

    flow_pred = pipe.model_fn(
        dit=pipe.dit,
        latents=x_pre,
        timestep=timesteps,
        context=context,
        use_gradient_checkpointing=bool(config["train"].get("gradient_checkpointing", True)),
        use_gradient_checkpointing_offload=bool(config["train"].get("gradient_checkpointing_offload", False)),
    )
    pred = x_pre.float() - sigmas.view(-1, 1, 1, 1, 1) * flow_pred.float()
    target_f = target.float()
    l1 = F.l1_loss(pred, target_f)
    mse = F.mse_loss(pred, target_f)
    l1_weight = float(config.get("loss", {}).get("l1_weight", 1.0))
    mse_weight = float(config.get("loss", {}).get("mse_weight", 0.1))
    total = l1_weight * l1 + mse_weight * mse
    return total, {"loss": float(total.detach()), "l1": float(l1.detach()), "mse": float(mse.detach())}


def compute_on_policy_batch_loss(
    module: torch.nn.Module,
    batch: dict[str, Any],
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    pipe = module.pipe
    target = batch["z_teacher_final"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    prompts = list(batch["prompt"])
    context = encode_prompts(pipe, prompts)
    meta = parse_batch_meta(batch)
    sigmas = extract_sigmas_from_meta(meta, device=device)
    train_step_index = extract_single_train_step_index(meta)

    noise = make_deterministic_noise(target, batch, config, device=device)
    x = noise

    active_steps = config["train"].get("on_policy_active_steps", "all_before_train")
    with torch.no_grad():
        for step_index in range(train_step_index):
            step_latents = x.to(dtype=pipe.torch_dtype)
            flow_pred = predict_flow(
                pipe,
                step_latents,
                sigmas[:, step_index],
                context,
                config,
                use_lora=should_use_lora_on_policy_step(active_steps, step_index),
            )
            x = flow_euler_step(x, flow_pred, sigmas[:, step_index], sigmas[:, step_index + 1]).detach()

    x_current = x.to(dtype=pipe.torch_dtype)
    flow_pred = predict_flow(
        pipe,
        x_current,
        sigmas[:, train_step_index],
        context,
        config,
        use_lora=True,
    )
    sigma_v = sigmas[:, train_step_index].view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    target_f = target.float()
    pred_clean = x_current.float() - sigma_v * flow_pred.float()
    clean_l1 = F.l1_loss(pred_clean, target_f)
    clean_mse = F.mse_loss(pred_clean, target_f)

    loss_type = str(config["train"].get("on_policy_loss_type", "velocity_target"))
    if loss_type == "clean_l1_mse":
        l1_weight = float(config.get("loss", {}).get("l1_weight", 1.0))
        mse_weight = float(config.get("loss", {}).get("mse_weight", 0.1))
        total = l1_weight * clean_l1 + mse_weight * clean_mse
        return total, {
            "loss": float(total.detach()),
            "on_policy_clean_l1": float(clean_l1.detach()),
            "on_policy_clean_mse": float(clean_mse.detach()),
        }
    if loss_type != "velocity_target":
        raise ValueError(f"Unknown on_policy_loss_type: {loss_type}")

    target_flow = (x_current.float() - target_f) / sigma_v
    velocity_mse = F.mse_loss(flow_pred.float(), target_flow.detach())
    velocity_l1 = F.l1_loss(flow_pred.float(), target_flow.detach())
    velocity_mse_weight = float(config.get("loss", {}).get("velocity_mse_weight", 1.0))
    velocity_l1_weight = float(config.get("loss", {}).get("velocity_l1_weight", 0.0))
    total = velocity_mse_weight * velocity_mse + velocity_l1_weight * velocity_l1
    return total, {
        "loss": float(total.detach()),
        "velocity_mse": float(velocity_mse.detach()),
        "velocity_l1": float(velocity_l1.detach()),
        "on_policy_clean_l1": float(clean_l1.detach()),
        "on_policy_clean_mse": float(clean_mse.detach()),
    }


def extract_train_sigma(batch: dict[str, Any]) -> torch.Tensor:
    meta_json = batch["meta_json"]
    if isinstance(meta_json, str):
        meta_items = [meta_json]
    else:
        meta_items = list(meta_json)
    sigmas = []
    for text in meta_items:
        meta = json.loads(text)
        recipe = meta.get("teacher_trajectory_recipe", {})
        if "train_sigma" not in recipe:
            raise KeyError("Missing teacher_trajectory_recipe.train_sigma in LMDB metadata.")
        sigmas.append(float(recipe["train_sigma"]))
    return torch.tensor(sigmas, dtype=torch.float32)


def parse_batch_meta(batch: dict[str, Any]) -> list[dict[str, Any]]:
    meta_json = batch["meta_json"]
    if isinstance(meta_json, str):
        meta_items = [meta_json]
    else:
        meta_items = list(meta_json)
    return [json.loads(text) for text in meta_items]


def extract_sigmas_from_meta(meta_items: list[dict[str, Any]], *, device: torch.device) -> torch.Tensor:
    rows = []
    for meta in meta_items:
        recipe = meta.get("teacher_trajectory_recipe", {})
        infer_steps = int(recipe.get("infer_steps", 0))
        if infer_steps <= 0:
            raise KeyError("Missing teacher_trajectory_recipe.infer_steps in LMDB metadata.")
        sigmas = [None] * infer_steps
        for step in recipe.get("executed_teacher_steps", []) + recipe.get("target_teacher_steps", []):
            index = int(step["step_index"])
            sigmas[index] = float(step["sigma"])
        if any(value is None for value in sigmas):
            raise KeyError("Incomplete teacher trajectory sigmas in LMDB metadata.")
        rows.append([float(value) for value in sigmas])
    return torch.tensor(rows, device=device, dtype=torch.float32)


def extract_single_train_step_index(meta_items: list[dict[str, Any]]) -> int:
    indices = {
        int(meta.get("teacher_trajectory_recipe", {}).get("train_step_index", -1))
        for meta in meta_items
    }
    if len(indices) != 1 or -1 in indices:
        raise ValueError(f"Expected one valid train_step_index per batch, got {sorted(indices)}")
    return next(iter(indices))


def should_use_lora_on_policy_step(active_steps: Any, step_index: int) -> bool:
    if active_steps in (None, True):
        return True
    if active_steps is False:
        return False
    if isinstance(active_steps, str):
        text = active_steps.strip().lower()
        if text in {"", "all_before_train", "all", "true", "1"}:
            return True
        if text in {"none", "false", "0"}:
            return False
        active_steps = [item.strip() for item in text.replace(",", " ").split() if item.strip()]
    return (step_index + 1) in {int(step) for step in active_steps}


@torch.no_grad()
def evaluate(
    module: torch.nn.Module,
    dataset: Dataset,
    config: dict,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
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
        batch_size = int(batch["x_pre_train_step"].shape[0])
        count += batch_size
        for key, value in items.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    module.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def split_dataset(dataset: Dataset, config: dict) -> tuple[Dataset, Dataset]:
    total = len(dataset)
    data_cfg = config.get("data", {})
    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    val_max_samples = int(data_cfg.get("val_max_samples", 0) or 0)
    seed = int(config.get("train", {}).get("seed", 1234))
    if val_ratio <= 0 or total < 2:
        return dataset, Subset(dataset, [])
    val_count = max(1, int(round(total * val_ratio)))
    if val_max_samples > 0:
        val_count = min(val_count, val_max_samples)
    val_count = min(val_count, total - 1)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return Subset(dataset, sorted(indices[val_count:])), Subset(dataset, sorted(indices[:val_count]))


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update(
        {
            "data": {"lmdb_dir": "data/changing_resolution_distill/lmdb_teacher_trajectory_lora_14b_cfgdistill_5k_step3"},
            "model": {"lora_base_model": "dit", "lora_target_modules": "q,k,v,o,ffn.0,ffn.2", "lora_rank": 32},
            "train": {"batch_size": 1, "grad_accum": 8, "lr": 1e-4, "max_steps": 10000, "precision": "bf16"},
            "loss": {"l1_weight": 1.0, "mse_weight": 0.1},
            "output": {"out_dir": "outputs/changing_resolution_distill_teacher_trajectory_lora_14b_cfgdistill_5k_step3"},
        },
        config,
    )
    if args.data_dir is not None:
        config["data"]["lmdb_dir"] = args.data_dir
    if args.out_dir is not None:
        config["output"]["out_dir"] = args.out_dir
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    for key in ("batch_size", "grad_accum", "lr", "max_steps", "precision", "training_mode", "on_policy_loss_type", "on_policy_active_steps"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
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
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="changing_resolution_distill/configs/train_teacher_trajectory_lora_distill.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--training_mode", choices=["cached_teacher", "on_policy"])
    parser.add_argument("--on_policy_loss_type", choices=["velocity_target", "clean_l1_mse"])
    parser.add_argument("--on_policy_active_steps")
    parser.add_argument("--model_paths")
    parser.add_argument("--model_id_with_origin_paths")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_target_modules")
    return parser.parse_args()


if __name__ == "__main__":
    main()
