from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution.scripts.data.build_x0pred_480p720p_stage3_lmdb import (  # noqa: E402
    ShardedX0PredLatentLMDBWriter,
    merge_meta,
    prepare_output_dir,
)
from wan_sr.data import CleanLatentLMDBDataset  # noqa: E402


def main() -> None:
    args = parse_args()
    prepare_output_dir(Path(args.out_dir), overwrite=args.overwrite)

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
        generator = LightX2VDistillX0PredGenerator(args, device=device, dtype=dtype)
    elif args.mode != "clean_copy":
        raise ValueError(f"Unsupported mode={args.mode!r}")

    writer = ShardedX0PredLatentLMDBWriter(
        out_dir=Path(args.out_dir),
        shard_size=args.shard_size,
        map_size_gb=args.map_size_gb,
        compression_dtype=np.float16,
    )

    saved = 0
    try:
        for source_index in tqdm(range(start_index, end_index), desc="stage3 distill x0_pred", dynamic_ncols=True):
            row = source[source_index]
            prompt = row["prompt"]
            z0_lr = row["z0_lr"].float()
            z0_hr = row["z0_hr"].float()

            if generator is None:
                x0_pred_lr = z0_lr
                recipe = {
                    "mode": "clean_copy",
                    "recipe": "distill_4step",
                    "note": "debug path only; not a real distill Stage 3 training domain",
                }
            else:
                seed = int(args.base_seed) + source_index
                x0_pred_lr, recipe = generator.make_x0_pred(z0_lr, prompt=prompt, seed=seed)

            meta = merge_meta(row.get("meta_json", "{}"))
            meta.update(
                {
                    "task": "changing_resolution_x0pred_stage3_distill_480p_to_720p",
                    "schema": "wan_x0pred_latent_pair_lmdb_v1",
                    "source_lmdb": str(args.source_lmdb),
                    "source_index": source_index,
                    "source_sample_id": row["sample_id"],
                    "prompt": prompt,
                    "x0_pred_lr_shape": list(x0_pred_lr.shape),
                    "z0_lr_shape": list(z0_lr.shape),
                    "z0_hr_shape": list(z0_hr.shape),
                    "stage3_recipe": recipe,
                }
            )
            writer.write(x0_pred_lr, z0_lr, z0_hr, prompt=prompt, meta=meta)
            saved += 1
    finally:
        writer.close()
        if generator is not None:
            generator.close()

    if args.require_samples is not None and saved < args.require_samples:
        raise RuntimeError(f"Stage 3 distill LMDB build only saved {saved} samples, required {args.require_samples}")
    print(f"Stage 3 distill x0-pred LMDB ready: {args.out_dir} ({saved} samples)")


class LightX2VDistillX0PredGenerator:
    """Generate x0_pred from the 4-step Wan distill scheduler."""

    def __init__(self, args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> None:
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
        self.runner.init_modules()

    @torch.no_grad()
    def make_x0_pred(self, z0_lr: torch.Tensor, *, prompt: str, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
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
        handoff_step = int(self.args.handoff_step)
        step_index = handoff_step - 1
        infer_steps = int(scheduler.infer_steps)
        if handoff_step < 1 or handoff_step > infer_steps:
            raise ValueError(f"handoff_step must be in [1, {infer_steps}], got {handoff_step}")

        z0_device = z0_lr.to(device=self.device, dtype=torch.float32)
        noise = torch.randn(
            z0_device.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=torch.float32,
        )
        sigma = scheduler.sigmas[step_index].to(device=self.device, dtype=torch.float32)
        x_t = scheduler.add_noise(z0_device, noise, sigma)

        scheduler.latents = x_t
        scheduler.step_pre(step_index=step_index)
        self.runner.model.infer(self.runner.inputs)
        flow_pred = scheduler.noise_pred.to(torch.float32)
        x0_pred = scheduler.latents.to(torch.float32) - sigma * flow_pred

        sigma_next = None
        if step_index + 1 < infer_steps:
            sigma_next = float(scheduler.sigmas[step_index + 1].detach().cpu())

        self.runner.end_run()
        recipe = {
            "mode": "lightx2v_distill",
            "recipe": "distill_4step",
            "distill_model_id": str(self.args.distill_model_id),
            "model_path": str(self.args.model_path),
            "config_json": str(self.args.config_json),
            "dit_original_ckpt": str(self.config.get("dit_original_ckpt", "")),
            "model_cls": str(self.config["model_cls"]),
            "infer_steps": infer_steps,
            "denoising_step_list": list(self.args.denoising_step_list),
            "handoff_step": handoff_step,
            "denoise_step": handoff_step,
            "step_index": step_index,
            "sigma": float(sigma.detach().cpu()),
            "sigma_next": sigma_next,
            "sample_shift": float(self.args.sample_shift),
            "sample_guide_scale": float(self.args.sample_guide_scale),
            "enable_cfg": bool(self.args.enable_cfg),
            "seed": seed,
        }
        return x0_pred.detach().cpu(), recipe

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


def parse_args() -> argparse.Namespace:
    default_stage3_tag = os.environ.get("CR_DISTILL_STAGE3_TAG", "14b_cfgdistill")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_lmdb",
        default=os.environ.get(
            "CR_DISTILL_CLEAN_LMDB_DIR",
            "data/changing_resolution_distill/lmdb_clean_480p720p_14b_cfgdistill_1k",
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=f"data/changing_resolution_distill/lmdb_x0pred_480p720p_stage3_{default_stage3_tag}_step2",
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
            "/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill",
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
    parser.add_argument("--handoff_step", type=int, default=2)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--enable_cfg", action="store_true")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--require_samples", type=int)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--base_seed", type=int, default=9400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
