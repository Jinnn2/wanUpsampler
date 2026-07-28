from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution.scripts.data.build_x0pred_480p720p_stage3_lmdb import (  # noqa: E402
    merge_meta,
    prepare_output_dir,
)
from wan_sr.data import CleanLatentLMDBDataset  # noqa: E402


def main() -> None:
    args = parse_args()
    prepare_output_dir(Path(args.out_dir), overwrite=args.overwrite)

    # 源 LMDB 是 clean 480p/720p latent pair 数据集。这里会保留匹配的 HR
    # latent，方便后续 clean upsampler 阶段复用；但本阶段 LoRA 目标本身只
    # 需要下面构造出来的 LR teacher trajectory。
    source = CleanLatentLMDBDataset(args.source_lmdb, strict_channels=True)
    start_index = int(args.offset)
    if start_index < 0 or start_index >= len(source):
        raise ValueError(f"offset must be in [0, {len(source) - 1}], got {start_index}")
    end_index = min(len(source), start_index + int(args.max_samples)) if args.max_samples is not None else len(source)
    if end_index <= start_index:
        raise RuntimeError("No source samples selected.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    generator = None
    if args.mode == "lightx2v_distill":
        generator = LightX2VDistillLastStepSkipGenerator(args, device=device, dtype=dtype)
    elif args.mode != "clean_copy":
        raise ValueError(f"Unsupported mode={args.mode!r}")

    writer = ShardedLastStepSkipLoRALMDBWriter(
        out_dir=Path(args.out_dir),
        shard_size=args.shard_size,
        map_size_gb=args.map_size_gb,
        compression_dtype=np.float16,
    )

    saved = 0
    try:
        for source_index in tqdm(range(start_index, end_index), desc="last-step-skip lora x_pre_step3", dynamic_ncols=True):
            row = source[source_index]
            prompt = row["prompt"]
            z0_lr_ref = row["z0_lr"].float()
            z0_hr = row["z0_hr"].float()
            source_meta = merge_meta(row.get("meta_json", "{}"))
            # 尽量沿用原始 clean pair 的生成 seed，保证 prompt/source sample
            # 和重新 rollout 的 teacher trajectory 对齐。如果源 metadata 里没
            # 有 seed，则用 base_seed + index 做确定性兜底，保证可复现。
            seed, seed_source = resolve_source_seed(source_meta, fallback=int(args.base_seed) + source_index)

            if generator is None:
                # 仅用于检查数据管线是否通畅的 debug 路径。它不会产生真实
                # LoRA trainer 所需的 train_sigma metadata。
                x_pre_step3_lr = z0_lr_ref
                z4_lr_teacher = z0_lr_ref
                recipe = {
                    "mode": "clean_copy",
                    "recipe": "last_step_skip_lora_vA",
                    "semantic_input_name": "x_pre_step3_lr",
                    "alias": "x2_lr",
                    "actual_input_step": "debug_clean_copy",
                    "note": "debug path only; x_pre_step3_lr and z4_lr_teacher are copied from clean LR reference",
                }
            else:
                x_pre_step3_lr, z4_lr_teacher, recipe = generator.make_pair(z0_lr_ref, prompt=prompt, seed=seed)

            # 同时保存新的显式字段名和旧的 x3_lr alias。保留 alias 是为了兼
            # 容一些旧 eval 工具；语义上这里其实是“训练 step 之前的 latent”，
            # 不一定应该被理解成严格执行三次更新后的 latent。
            meta = dict(source_meta)
            meta.update(
                {
                    "task": "wan_distill_last_step_skip_lora",
                    "schema": "wan_last_step_skip_lora_lmdb_v2",
                    "source_lmdb": str(args.source_lmdb),
                    "source_index": source_index,
                    "source_sample_id": row["sample_id"],
                    "prompt": prompt,
                    "seed": seed,
                    "seed_source": seed_source,
                    "semantic_input_name": "x_pre_step3_lr",
                    "semantic_input_alias": "x2_lr",
                    "legacy_input_key": "x3_lr",
                    "actual_input_step": "after_teacher_step2_before_teacher_step3",
                    "x_pre_step3_lr_shape": list(x_pre_step3_lr.shape),
                    "x3_lr_shape": list(x_pre_step3_lr.shape),
                    "z4_lr_teacher_shape": list(z4_lr_teacher.shape),
                    "z0_lr_ref_shape": list(z0_lr_ref.shape),
                    "z0_hr_shape": list(z0_hr.shape),
                    "last_step_skip_recipe": recipe,
                }
            )
            writer.write(x_pre_step3_lr, z4_lr_teacher, z0_hr, prompt=prompt, seed=seed, meta=meta)
            saved += 1
    finally:
        writer.close()
        if generator is not None:
            generator.close()

    if args.require_samples is not None and saved < args.require_samples:
        raise RuntimeError(f"Last-step-skip LoRA LMDB build only saved {saved} samples, required {args.require_samples}")
    print(f"Last-step-skip LoRA LMDB ready: {args.out_dir} ({saved} samples)")


class LightX2VDistillLastStepSkipGenerator:
    """Cache the pre-train-step LR state and its matched 4-step teacher target."""

    def __init__(self, args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> None:
        # LightX2V 这里只负责离线 teacher rollout。后面的训练脚本会用
        # DiffSynth 做反向传播，从而更新 LoRA 权重。
        lightx2v_repo = args.lightx2v_repo or os.environ.get("LIGHTX2V_REPO")
        if lightx2v_repo and lightx2v_repo not in sys.path:
            sys.path.insert(0, lightx2v_repo)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        import importlib

        importlib.import_module("lightx2v.common.ops")
        importlib.import_module("lightx2v.models.runners.wan.wan_distill_runner")
        from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
        from lightx2v.utils.registry_factory import RUNNER_REGISTER
        from lightx2v.utils.set_config import set_config, set_parallel_config
        from lightx2v.utils.utils import seed_all, validate_config_paths
        from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

        self.args = args
        self.device = device
        self.dtype = dtype
        self.init_empty_input_info = init_empty_input_info
        self.update_input_info_from_dict = update_input_info_from_dict
        seed_all(args.base_seed)

        config_args = self._make_lightx2v_args(args)
        config = set_config(config_args)
        with config.temporarily_unlocked():
            config.update(
                {
                    "infer_steps": len(args.denoising_step_list),
                    "denoising_step_list": list(args.denoising_step_list),
                    "target_video_length": int(args.num_frames),
                    "sample_shift": float(args.sample_shift),
                    "sample_guide_scale": float(args.sample_guide_scale),
                    "enable_cfg": bool(args.enable_cfg),
                }
            )
        if config["parallel"]:
            platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
            platform_device.init_parallel_env()
            set_parallel_config(config)
        validate_config_paths(config)

        torch.set_grad_enabled(False)
        self.config = config
        self.runner = RUNNER_REGISTER[config["model_cls"]](config)
        # 14B 模型初始化很重，所以保持 runner 常驻，并在样本循环里复用它，
        # 避免每条样本都重新加载模型。
        self.runner.init_modules()

    @torch.no_grad()
    def make_pair(self, z0_lr_ref: torch.Tensor, *, prompt: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        input_info = self.init_empty_input_info(self.args.task, [])
        self.update_input_info_from_dict(
            input_info,
            {
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": self.args.negative_prompt,
                "save_result_path": None,
                "return_result_tensor": True,
            },
        )

        self.runner.input_info = input_info
        self.runner.inputs = self.runner.run_input_encoder()
        self.runner.init_run()

        scheduler = self.runner.model.scheduler
        train_step_index = int(self.args.train_step_index)
        infer_steps = int(scheduler.infer_steps)
        if train_step_index < 1 or train_step_index >= infer_steps:
            raise ValueError(f"train_step_index must be in [1, {infer_steps - 1}], got {train_step_index}")

        # 从 clean LR latent 重新构造 teacher 的初始 noisy latent。如果能拿到
        # 原始 seed，就沿用同一个 seed，使缓存轨迹仍然和 prompt/source sample
        # 保持配对。
        z0_device = z0_lr_ref.to(device=self.device, dtype=torch.float32)
        noise = torch.randn(
            z0_device.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=torch.float32,
        )
        sigma0 = scheduler.sigmas[0].to(device=self.device, dtype=torch.float32)
        scheduler.latents = scheduler.add_noise(z0_device, noise, sigma0)

        # 先运行 base teacher 到 LoRA 训练 step 之前，但不执行该训练 step。
        # 默认 train_step_index=2，也就是执行 teacher step1/step2 后，把
        # step3 之前的 latent 缓存为 x_pre_step3_lr。
        executed_steps: list[dict[str, Any]] = []
        for step_index in range(train_step_index):
            scheduler.step_pre(step_index=step_index)
            self.runner.model.infer(self.runner.inputs)
            flow_pred = scheduler.noise_pred.to(torch.float32)
            sigma = scheduler.sigmas[step_index].to(device=self.device, dtype=torch.float32)
            clean_pred = scheduler.latents.to(torch.float32) - sigma * flow_pred
            scheduler.step_post()
            sigma_next = None
            if step_index + 1 < infer_steps:
                sigma_next = float(scheduler.sigmas[step_index + 1].detach().cpu())
            executed_steps.append(
                {
                    "step_index": step_index,
                    "step_name": f"teacher_step{step_index + 1}",
                    "sigma": float(sigma.detach().cpu()),
                    "sigma_next": sigma_next,
                    "clean_pred_shape": list(clean_pred.shape),
                }
            )

        x_pre_step3_lr = scheduler.latents.detach().to(torch.float32).cpu()
        train_sigma = scheduler.sigmas[train_step_index].to(device=self.device, dtype=torch.float32)

        # 继续运行原始 teacher 到结束。最终 latent 会作为监督目标，要求 LoRA
        # 在缓存的训练状态上用一步 clean prediction 对齐它。
        target_steps: list[dict[str, Any]] = []
        for step_index in range(train_step_index, infer_steps):
            scheduler.step_pre(step_index=step_index)
            self.runner.model.infer(self.runner.inputs)
            flow_pred = scheduler.noise_pred.to(torch.float32)
            sigma = scheduler.sigmas[step_index].to(device=self.device, dtype=torch.float32)
            clean_pred = scheduler.latents.to(torch.float32) - sigma * flow_pred
            scheduler.step_post()
            sigma_next = None
            if step_index + 1 < infer_steps:
                sigma_next = float(scheduler.sigmas[step_index + 1].detach().cpu())
            target_steps.append(
                {
                    "step_index": step_index,
                    "step_name": f"teacher_step{step_index + 1}",
                    "sigma": float(sigma.detach().cpu()),
                    "sigma_next": sigma_next,
                    "clean_pred_shape": list(clean_pred.shape),
                }
            )

        z4_lr_teacher = scheduler.latents.detach().to(torch.float32).cpu()

        self.runner.end_run()
        # trainer 会从这个 recipe 里读取 train_sigma。这里保留足够 metadata，
        # 方便检查 checkpoint 和 LMDB 是否使用了同一套 schedule、模型和 step
        # 语义。
        recipe = {
            "mode": "lightx2v_distill",
            "recipe": "last_step_skip_lora_vA",
            "semantic_input_name": "x_pre_step3_lr",
            "alias": "x2_lr",
            "actual_input_step": "after_teacher_step2_before_teacher_step3",
            "distill_model_id": str(self.args.distill_model_id),
            "model_path": str(self.args.model_path),
            "config_json": str(self.args.config_json),
            "dit_original_ckpt": str(self.config.get("dit_original_ckpt", "")),
            "model_cls": str(self.config["model_cls"]),
            "infer_steps": infer_steps,
            "denoising_step_list": list(self.args.denoising_step_list),
            "executed_teacher_steps": executed_steps,
            "target_teacher_steps": target_steps,
            "train_step_index": train_step_index,
            "train_step_name": f"step{train_step_index + 1}",
            "target_step_name": "teacher_step4_clean",
            "train_sigma": float(train_sigma.detach().cpu()),
            "sample_shift": float(self.args.sample_shift),
            "sample_guide_scale": float(self.args.sample_guide_scale),
            "enable_cfg": bool(self.args.enable_cfg),
            "seed": seed,
        }
        return x_pre_step3_lr, z4_lr_teacher, recipe

    def close(self) -> None:
        if hasattr(self.runner, "end_run"):
            try:
                self.runner.end_run()
            except Exception:
                pass

    def _make_lightx2v_args(self, args: argparse.Namespace) -> argparse.Namespace:
        return argparse.Namespace(
            seed=args.base_seed,
            model_cls=args.model_cls,
            task=args.task,
            support_tasks=[],
            model_path=args.model_path,
            sf_model_path=None,
            config_json=args.config_json,
            use_prompt_enhancer=False,
            prompt="",
            negative_prompt=args.negative_prompt,
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
            save_result_path=None,
            return_result_tensor=True,
            target_shape=[],
            target_video_length=args.num_frames,
            aspect_ratio="",
            video_path=None,
            sr_ratio=2.0,
        )


@dataclass
class ShardedLastStepSkipLoRALMDBWriter:
    out_dir: Path
    shard_size: int
    map_size_gb: int
    compression_dtype: np.dtype

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env: lmdb.Environment | None = None
        self.shard_index = -1
        self.row_index = 0
        self.total = 0
        self.current_meta: dict[str, Any] | None = None

    def write(
        self,
        x_pre_step3_lr: torch.Tensor,
        z4_lr_teacher: torch.Tensor,
        z0_hr: torch.Tensor,
        *,
        prompt: str,
        seed: int,
        meta: dict[str, Any],
    ) -> None:
        if self.env is None or self.row_index >= self.shard_size:
            self._open_next_shard(x_pre_step3_lr, z4_lr_teacher, z0_hr)

        assert self.env is not None
        row = self.row_index
        # LMDB 存的是原始 bytes，所以 shape 和 dtype 要记录在 shard-level
        # metadata 里。float16 压缩可以让 5k 14B trajectory cache 足够小，
        # reader 读取时再恢复成 float32 tensor。
        x_pre_step3_lr_np = _to_numpy_fp(x_pre_step3_lr, self.compression_dtype)
        z4_lr_teacher_np = _to_numpy_fp(z4_lr_teacher, self.compression_dtype)
        z0_hr_np = _to_numpy_fp(z0_hr, self.compression_dtype)
        meta_json = json.dumps(meta, ensure_ascii=False)

        with self.env.begin(write=True) as txn:
            txn.put(_key("x_pre_step3_lr", row), x_pre_step3_lr_np.tobytes())
            txn.put(_key("x3_lr", row), x_pre_step3_lr_np.tobytes())
            txn.put(_key("z4_lr_teacher", row), z4_lr_teacher_np.tobytes())
            txn.put(_key("z0_hr", row), z0_hr_np.tobytes())
            txn.put(_key("prompt", row), prompt.encode("utf-8"))
            txn.put(_key("seed", row), str(seed).encode("utf-8"))
            txn.put(_key("meta", row), meta_json.encode("utf-8"))

        self.row_index += 1
        self.total += 1
        self._write_metadata()

    def close(self) -> None:
        if self.env is not None:
            self._write_metadata()
            self.env.sync()
            self.env.close()
            self.env = None

    def _open_next_shard(self, x_pre_step3_lr: torch.Tensor, z4_lr_teacher: torch.Tensor, z0_hr: torch.Tensor) -> None:
        self.close()
        self.shard_index += 1
        self.row_index = 0
        shard_dir = self.out_dir / f"shard_{self.shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        self.env = lmdb.open(str(shard_dir), map_size=self.map_size_gb * 1024**3, subdir=True, meminit=False)
        self.current_meta = {
            "num_samples": 0,
            "dtype": "float16",
            "x_pre_step3_lr_shape": list(x_pre_step3_lr.shape),
            "x3_lr_shape": list(x_pre_step3_lr.shape),
            "z4_lr_teacher_shape": list(z4_lr_teacher.shape),
            "z0_hr_shape": list(z0_hr.shape),
            "schema": "wan_last_step_skip_lora_lmdb_v2",
            "semantic_input_name": "x_pre_step3_lr",
            "legacy_input_key": "x3_lr",
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self.env is None or self.current_meta is None:
            return
        meta = dict(self.current_meta)
        meta["num_samples"] = self.row_index
        with self.env.begin(write=True) as txn:
            txn.put(b"metadata", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_lmdb",
        default=os.environ.get(
            "CR_DISTILL_CLEAN_LMDB_DIR",
            "data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_5k",
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=os.environ.get(
            "CR_DISTILL_LORA_LMDB_DIR",
            "data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3",
        ),
    )
    parser.add_argument("--mode", choices=["lightx2v_distill", "clean_copy"], default="lightx2v_distill")
    parser.add_argument(
        "--distill_model_id",
        default=os.environ.get("CR_DISTILL_MODEL_ID", "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill"),
    )
    parser.add_argument("--lightx2v_repo", default=os.environ.get("LIGHTX2V_REPO"))
    parser.add_argument(
        "--model_path",
        default=os.environ.get(
            "CR_DISTILL_MODEL_ROOT",
            "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill",
        ),
    )
    parser.add_argument(
        "--config_json",
        default="changing_resolution_distill/configs/wan_t2v_distill_stage3_x0pred_480p.json",
    )
    parser.add_argument("--model_cls", default="wan2.1_distill")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--denoising_step_list", type=int, nargs="+", default=[1000, 750, 500, 250])
    parser.add_argument("--train_step_index", type=int, default=2, help="0-based step index for LoRA training input; 2 means step3.")
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--enable_cfg", action="store_true")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--require_samples", type=int)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--base_seed", type=int, default=9500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_source_seed(source_meta: dict[str, Any], *, fallback: int) -> tuple[int, str]:
    for key in ("seed", "generation_seed", "base_seed", "video_seed"):
        value = source_meta.get(key)
        if value is not None and str(value).strip() != "":
            return int(value), f"source_meta.{key}"
    nested = source_meta.get("source_meta")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except json.JSONDecodeError:
            nested = None
    if isinstance(nested, dict):
        nested_seed, nested_source = resolve_source_seed(nested, fallback=fallback)
        if nested_source != "base_seed_plus_index":
            return nested_seed, f"source_meta.{nested_source}"
    return int(fallback), "base_seed_plus_index"


def _to_numpy_fp(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float16).numpy().astype(dtype, copy=False)


def _key(name: str, row_id: int) -> bytes:
    return f"{name}_{row_id:08d}_data".encode("utf-8")


if __name__ == "__main__":
    main()
