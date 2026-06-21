from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LastStepSkipLoRALMDBDataset  # noqa: E402
from wan_sr.training.config import load_yaml  # noqa: E402


def main() -> None:
    args = parse_args()
    indices = parse_indices(args.indices, args.num_samples)
    if args.backend == "diffsynth":
        records = eval_diffsynth(args, indices)
    elif args.backend == "lightx2v":
        records = eval_lightx2v(args, indices)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"cached_x3_{args.backend}_loss.jsonl"
    csv_path = out_dir / f"cached_x3_{args.backend}_loss.csv"
    jsonl_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    wins = sum(1 for record in records if float(record["lora_l1"]) < float(record["orig_l1"]))
    print(f"[cached-x3] backend={args.backend} samples={len(records)} lora_wins={wins}/{len(records)}")
    print(f"[cached-x3] jsonl={jsonl_path}")
    print(f"[cached-x3] csv={csv_path}")


def eval_diffsynth(args: argparse.Namespace, indices: list[int]) -> list[dict[str, Any]]:
    train_mod = load_train_module()
    config = train_mod.load_yaml(args.train_config)
    cli_args = argparse.Namespace(
        data_dir=args.data_dir,
        out_dir=None,
        max_samples=None,
        batch_size=1,
        grad_accum=None,
        lr=None,
        max_steps=None,
        precision=args.precision,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
    )
    config = train_mod.apply_cli_overrides(config, cli_args)
    config["train"]["gradient_checkpointing"] = False
    config["train"]["gradient_checkpointing_offload"] = False

    device = train_mod.get_training_device()
    dataset = LastStepSkipLoRALMDBDataset(args.data_dir, dtype=torch.float32)

    base_config = json.loads(json.dumps(config))
    base_config["model"]["lora_checkpoint"] = None
    lora_config = json.loads(json.dumps(config))
    lora_config["model"]["lora_checkpoint"] = args.lora_ckpt

    print("[cached-x3][diffsynth] loading base model")
    base_module = train_mod.build_wan_training_module(base_config, device=device).eval()
    print("[cached-x3][diffsynth] loading LoRA model")
    lora_module = train_mod.build_wan_training_module(lora_config, device=device).eval()

    records = []
    for sample_index in indices:
        row = dataset[sample_index]
        batch = {
            "x3_lr": row["x3_lr"].unsqueeze(0),
            "z4_lr_teacher": row["z4_lr_teacher"].unsqueeze(0),
            "prompt": [row["prompt"]],
            "meta_json": [row["meta_json"]],
        }
        orig = predict_diffsynth(base_module, batch, train_mod, device)
        lora = predict_diffsynth(lora_module, batch, train_mod, device)
        target = batch["z4_lr_teacher"].float().to(orig.device)
        records.append(make_record(args.backend, sample_index, row, orig, lora, target))
    return records


@torch.no_grad()
def predict_diffsynth(module: torch.nn.Module, batch: dict[str, Any], train_mod: Any, device: torch.device) -> torch.Tensor:
    pipe = module.pipe
    x3_lr = batch["x3_lr"].to(device, dtype=pipe.torch_dtype)
    sigmas = train_mod.extract_train_sigma(batch).to(device=device, dtype=torch.float32)
    timesteps = (sigmas * 1000.0).to(device=device, dtype=pipe.torch_dtype)
    context = train_mod.encode_prompts(pipe, list(batch["prompt"]))
    flow_pred = pipe.model_fn(
        dit=pipe.dit,
        latents=x3_lr,
        timestep=timesteps,
        context=context,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )
    return x3_lr.float() - sigmas.view(-1, 1, 1, 1, 1) * flow_pred.float()


def eval_lightx2v(args: argparse.Namespace, indices: list[int]) -> list[dict[str, Any]]:
    lightx2v_repo = os.environ.get("LIGHTX2V_REPO")
    if lightx2v_repo and lightx2v_repo not in sys.path:
        sys.path.insert(0, lightx2v_repo)

    import importlib

    importlib.import_module("lightx2v.common.ops")
    import changing_resolution_distill.lightx2v_distill_bridge  # noqa: F401
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.registry_factory import RUNNER_REGISTER
    from lightx2v.utils.set_config import set_config
    from lightx2v.utils.utils import seed_all, validate_config_paths
    from lightx2v_platform.base.global_var import AI_DEVICE

    dataset = LastStepSkipLoRALMDBDataset(args.data_dir, dtype=torch.float32)

    print("[cached-x3][lightx2v] loading base runner")
    base_runner = build_lightx2v_runner(
        args,
        model_cls="wan2.1_distill",
        config_json=args.lightx2v_base_config,
        set_config=set_config,
        validate_config_paths=validate_config_paths,
        RUNNER_REGISTER=RUNNER_REGISTER,
    )
    print("[cached-x3][lightx2v] loading LoRA runner")
    lora_runner = build_lightx2v_runner(
        args,
        model_cls="wan2.1_distill_last_step_lora",
        config_json=args.lightx2v_lora_config,
        set_config=set_config,
        validate_config_paths=validate_config_paths,
        RUNNER_REGISTER=RUNNER_REGISTER,
    )

    records = []
    for sample_index in indices:
        row = dataset[sample_index]
        seed = int(row["seed"] if row["seed"] is not None else args.seed)
        seed_all(seed)
        orig = predict_lightx2v(base_runner, row, args, init_empty_input_info, update_input_info_from_dict, AI_DEVICE, lora_strength=None)
        seed_all(seed)
        lora = predict_lightx2v(lora_runner, row, args, init_empty_input_info, update_input_info_from_dict, AI_DEVICE, lora_strength=args.lora_strength)
        target = row["z4_lr_teacher"].unsqueeze(0).float().to(orig.device)
        records.append(make_record(args.backend, sample_index, row, orig, lora, target))
    return records


def build_lightx2v_runner(args, *, model_cls, config_json, set_config, validate_config_paths, RUNNER_REGISTER):
    ns = argparse.Namespace(
        seed=args.seed,
        model_cls=model_cls,
        task="t2v",
        support_tasks=[],
        model_path=args.model_path,
        sf_model_path=None,
        config_json=config_json,
        use_prompt_enhancer=False,
        prompt="",
        negative_prompt="",
        image_path="",
        last_frame_path="",
        audio_path="",
        image_strength="1.0",
        image_frame_idx="",
        src_ref_images=None,
        src_video=None,
        src_mask=None,
        src_pose_path=None,
        src_face_path=None,
        src_bg_path=None,
        src_mask_path=None,
        pose=None,
        action_path=None,
        action_ckpt=None,
        save_result_path=None,
        return_result_tensor=False,
        target_shape=[],
        target_video_length=args.target_video_length,
        aspect_ratio="",
        video_path=None,
        sr_ratio=2.0,
    )
    config = set_config(ns)
    validate_config_paths(config)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    return runner


@torch.no_grad()
def predict_lightx2v(runner, row, args, init_empty_input_info, update_input_info_from_dict, ai_device: str, *, lora_strength: float | None):
    input_info = init_empty_input_info("t2v", [])
    update_input_info_from_dict(
        input_info,
        {
            "seed": int(row["seed"] if row["seed"] is not None else args.seed),
            "prompt": row["prompt"],
            "negative_prompt": "",
            "save_result_path": None,
            "return_result_tensor": True,
            "target_video_length": args.target_video_length,
        },
    )
    runner.input_info = input_info
    runner.inputs = runner.run_input_encoder()
    runner.init_run()
    scheduler = runner.model.scheduler
    train_step_index = args.train_step_index
    x3 = row["x3_lr"].to(device=ai_device, dtype=scheduler.latents.dtype)
    scheduler.latents = x3
    if lora_strength is not None:
        runner.model._update_lora(args.lora_ckpt, float(lora_strength))
    scheduler.step_pre(step_index=train_step_index)
    runner.model.infer(runner.inputs)
    sigma = scheduler.sigmas[train_step_index].to(device=ai_device, dtype=torch.float32)
    pred = scheduler.latents.to(torch.float32) - sigma * scheduler.noise_pred.to(torch.float32)
    runner.end_run()
    return pred.unsqueeze(0) if pred.ndim == 4 else pred


def make_record(backend: str, sample_index: int, row: dict[str, Any], orig: torch.Tensor, lora: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    orig_f = orig.float()
    lora_f = lora.float()
    target_f = target.float()
    orig_l1 = torch.nn.functional.l1_loss(orig_f, target_f).item()
    lora_l1 = torch.nn.functional.l1_loss(lora_f, target_f).item()
    return {
        "backend": backend,
        "sample_index": sample_index,
        "sample_id": row["sample_id"],
        "seed": row["seed"],
        "prompt": row["prompt"],
        "orig_l1": orig_l1,
        "lora_l1": lora_l1,
        "delta_l1": lora_l1 - orig_l1,
        "lora_wins": lora_l1 < orig_l1,
        "orig_lora_l1": torch.nn.functional.l1_loss(orig_f, lora_f).item(),
        "x3_target_l1": torch.nn.functional.l1_loss(row["x3_lr"].unsqueeze(0).float().to(target_f.device), target_f).item(),
    }


def load_train_module():
    path = REPO_ROOT / "changing_resolution_distill" / "scripts" / "train" / "train_last_step_skip_lora.py"
    spec = importlib.util.spec_from_file_location("train_last_step_skip_lora_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_indices(indices_arg: str | None, num_samples: int) -> list[int]:
    if indices_arg:
        return [int(part.strip()) for part in indices_arg.split(",") if part.strip()]
    return list(range(num_samples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["diffsynth", "lightx2v"], required=True)
    parser.add_argument("--data_dir", default=os.environ.get("CR_DISTILL_LORA_LMDB_DIR", "data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3"))
    parser.add_argument("--out_dir", default="outputs/changing_resolution_distill_last_step_skip_lora_cached_x3_eval")
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--indices")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_step_index", type=int, default=2)
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--train_config", default="changing_resolution_distill/configs/train_last_step_skip_lora_distill.yaml")
    parser.add_argument("--model_paths")
    parser.add_argument("--model_id_with_origin_paths")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")

    parser.add_argument("--model_path", default=os.environ.get("CR_DISTILL_MODEL_ROOT", "/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill"))
    parser.add_argument("--lightx2v_base_config", default="changing_resolution_distill/configs/cached_x3_base_tmp.json")
    parser.add_argument("--lightx2v_lora_config", default="changing_resolution_distill/configs/cached_x3_lora_tmp.json")
    parser.add_argument("--lora_strength", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
