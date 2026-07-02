from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LastStepSkipLoRALMDBDataset  # noqa: E402
from wan_sr.training.config import deep_update, load_yaml  # noqa: E402


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_yaml(args.config), args)
    if str(config["train"].get("training_mode", "cached_x_pre_step3")) != "cached_x_pre_step3":
        raise ValueError("This mainline trainer only supports training_mode=cached_x_pre_step3.")

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
    dataset: Dataset = LastStepSkipLoRALMDBDataset(config["data"]["lmdb_dir"], strict_channels=True)
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
    x_pre_step3 = batch["x_pre_step3_lr"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    target = batch["z4_lr_teacher"].to(device, dtype=pipe.torch_dtype, non_blocking=True)
    prompts = list(batch["prompt"])
    sigmas = extract_train_sigma(batch).to(device=device, dtype=torch.float32)
    timesteps = (sigmas * 1000.0).to(device=device, dtype=pipe.torch_dtype)
    context = encode_prompts(pipe, prompts)

    flow_pred = pipe.model_fn(
        dit=pipe.dit,
        latents=x_pre_step3,
        timestep=timesteps,
        context=context,
        use_gradient_checkpointing=bool(config["train"].get("gradient_checkpointing", True)),
        use_gradient_checkpointing_offload=bool(config["train"].get("gradient_checkpointing_offload", False)),
    )
    pred = x_pre_step3.float() - sigmas.view(-1, 1, 1, 1, 1) * flow_pred.float()
    target_f = target.float()
    l1 = F.l1_loss(pred, target_f)
    mse = F.mse_loss(pred, target_f)

    loss_cfg = config.get("loss", {})
    l1_weight = float(loss_cfg.get("l1_weight", 1.0))
    mse_weight = float(loss_cfg.get("mse_weight", 0.1))
    total = l1_weight * l1 + mse_weight * mse
    items = {"loss": float(total.detach()), "clean_l1": float(l1.detach()), "clean_mse": float(mse.detach())}

    velocity_mse_weight = float(loss_cfg.get("velocity_mse_weight", 0.0) or 0.0)
    velocity_l1_weight = float(loss_cfg.get("velocity_l1_weight", 0.0) or 0.0)
    if velocity_mse_weight > 0 or velocity_l1_weight > 0:
        target_flow = (x_pre_step3.float() - target_f) / sigmas.view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        velocity_mse = F.mse_loss(flow_pred.float(), target_flow.detach())
        velocity_l1 = F.l1_loss(flow_pred.float(), target_flow.detach())
        velocity_loss = velocity_mse_weight * velocity_mse + velocity_l1_weight * velocity_l1
        total = total + velocity_loss
        items.update(
            {
                "loss": float(total.detach()),
                "velocity_mse": float(velocity_mse.detach()),
                "velocity_l1": float(velocity_l1.detach()),
            }
        )

    return total, items


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
        recipe = meta.get("last_step_skip_recipe", {})
        if "train_sigma" not in recipe:
            raise KeyError("Missing last_step_skip_recipe.train_sigma in LMDB metadata.")
        sigmas.append(float(recipe["train_sigma"]))
    return torch.tensor(sigmas, dtype=torch.float32)


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
        batch_size = int(batch["x_pre_step3_lr"].shape[0])
        count += batch_size
        for key, value in items.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    module.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def build_wan_training_module(config: dict, *, device: torch.device) -> torch.nn.Module:
    diffsynth_repo = config.get("diffsynth_repo") or os.environ.get("DIFFSYNTH_REPO")
    if diffsynth_repo:
        if str(diffsynth_repo) not in sys.path:
            sys.path.insert(0, str(diffsynth_repo))
        examples_path = Path(diffsynth_repo) / "examples" / "wanvideo" / "model_training" / "train.py"
    else:
        examples_path = Path("examples/wanvideo/model_training/train.py")

    if not examples_path.exists():
        raise FileNotFoundError(
            f"DiffSynth Wan training entry not found: {examples_path}. "
            "Set DIFFSYNTH_REPO or run setup_last_step_skip_lora_env.sh install."
        )

    spec = importlib.util.spec_from_file_location("diffsynth_wan_train_entry", examples_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load DiffSynth train.py from {examples_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    WanTrainingModule = module.WanTrainingModule

    model_cfg = config["model"]
    train_cfg = config["train"]
    return WanTrainingModule(
        model_paths=normalize_model_paths(model_cfg.get("model_paths")),
        model_id_with_origin_paths=model_cfg.get("model_id_with_origin_paths"),
        tokenizer_path=model_cfg.get("tokenizer_path"),
        trainable_models=None,
        lora_base_model=model_cfg.get("lora_base_model", "dit"),
        lora_target_modules=model_cfg.get("lora_target_modules", "q,k,v,o,ffn.0,ffn.2"),
        lora_rank=int(model_cfg.get("lora_rank", 32)),
        lora_checkpoint=model_cfg.get("lora_checkpoint"),
        preset_lora_path=model_cfg.get("preset_lora_path"),
        preset_lora_model=model_cfg.get("preset_lora_model"),
        use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        use_gradient_checkpointing_offload=bool(train_cfg.get("gradient_checkpointing_offload", False)),
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=model_cfg.get("remove_prefix_in_ckpt"),
        task="sft",
        device=device,
    )


def normalize_model_paths(model_paths: Any) -> str | None:
    if model_paths is None:
        return None
    if isinstance(model_paths, (list, tuple)):
        paths = list(itertools.chain.from_iterable(expand_model_path_entry(path) for path in model_paths))
        return json.dumps(dedupe_keep_order(paths))
    if isinstance(model_paths, str):
        text = model_paths.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps(expand_model_path_entry(text))
        if isinstance(parsed, list):
            paths = list(itertools.chain.from_iterable(expand_model_path_entry(path) for path in parsed))
            return json.dumps(dedupe_keep_order(paths))
        return json.dumps(expand_model_path_entry(parsed))
    return json.dumps(expand_model_path_entry(model_paths))


def expand_model_path_entry(path_like: Any) -> list[str]:
    path = Path(str(path_like))
    if not path.is_dir():
        return [str(path)]
    candidates = [
        path / "distill_model.pt",
        path / "models_t5_umt5-xxl-enc-bf16.pth",
        path / "models_t5_umt5-xxl-enc-bf16.safetensors",
    ]
    existing = [str(candidate) for candidate in candidates if candidate.is_file()]
    return existing or [str(path)]


def dedupe_keep_order(paths: list[str]) -> list[str]:
    seen = set()
    result = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


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
        "input_key": str(config["task"].get("input_key", "x_pre_step3_lr")),
        "target_key": str(config["task"].get("target_key", "z4_lr_teacher")),
        "base_model_id": str(config["model"].get("base_model_id", "")),
        "lora_rank": str(config["model"].get("lora_rank", "")),
        "lora_target_modules": str(config["model"].get("lora_target_modules", "")),
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


def load_resume(path: Path, module: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    payload = torch.load(path, map_location=device)
    state = payload.get("trainable_state", {})
    named = dict(module.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            if name in named:
                named[name].copy_(tensor.to(device=device, dtype=named[name].dtype))
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


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
            "task": {
                "name": "last_step_skip_lora",
                "recipe": "distill_4step_cached_x_pre_step3_to_teacher4_clean",
                "input_key": "x_pre_step3_lr",
                "target_key": "z4_lr_teacher",
            },
            "data": {"lmdb_dir": "data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3"},
            "model": {"lora_base_model": "dit", "lora_target_modules": "q,k,v,o,ffn.0,ffn.2", "lora_rank": 32},
            "train": {
                "batch_size": 1,
                "grad_accum": 8,
                "lr": 5e-5,
                "max_steps": 10000,
                "precision": "bf16",
                "training_mode": "cached_x_pre_step3",
            },
            "loss": {"l1_weight": 1.0, "mse_weight": 0.1, "velocity_mse_weight": 0.0, "velocity_l1_weight": 0.0},
            "output": {"out_dir": "outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3"},
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


def get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("ALLOW_CPU_TRAINING") == "1":
        return torch.device("cpu")
    raise RuntimeError("CUDA is unavailable. Set ALLOW_CPU_TRAINING=1 only for tiny smoke tests.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def strip_prefix(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        return name[len(prefix) :]
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--training_mode", choices=["cached_x_pre_step3"])
    parser.add_argument("--model_paths")
    parser.add_argument("--model_id_with_origin_paths")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_target_modules")
    return parser.parse_args()


if __name__ == "__main__":
    main()
