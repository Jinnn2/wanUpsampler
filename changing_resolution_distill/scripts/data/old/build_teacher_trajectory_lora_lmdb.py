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

from changing_resolution.scripts.data.build_x0pred_480p720p_stage3_lmdb import prepare_output_dir  # noqa: E402


def main() -> None:
    args = parse_args()
    prepare_output_dir(Path(args.out_dir), overwrite=args.overwrite)
    prompts = load_prompts(Path(args.prompts_file))
    selected = prompts[int(args.prompt_offset) :]
    if args.max_samples is not None:
        selected = selected[: int(args.max_samples)]
    if not selected:
        raise RuntimeError("No prompts selected.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    generator = LightX2VTeacherTrajectoryGenerator(args, device=device, dtype=dtype)
    writer = ShardedTeacherTrajectoryLoRALMDBWriter(
        out_dir=Path(args.out_dir),
        shard_size=args.shard_size,
        map_size_gb=args.map_size_gb,
        compression_dtype=np.float16,
    )

    saved = 0
    try:
        for local_index, prompt in enumerate(tqdm(selected, desc="teacher trajectory lora", dynamic_ncols=True)):
            prompt_index = int(args.prompt_offset) + local_index
            seed = int(args.base_seed) + prompt_index
            x_pre, z_teacher, recipe = generator.make_pair(prompt=prompt, seed=seed)
            meta = {
                "task": "wan_distill_teacher_trajectory_lora",
                "schema": "wan_teacher_trajectory_lora_lmdb_v1",
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seed": seed,
                "x_pre_train_step_shape": list(x_pre.shape),
                "z_teacher_final_shape": list(z_teacher.shape),
                "teacher_trajectory_recipe": recipe,
            }
            writer.write(x_pre, z_teacher, prompt=prompt, seed=seed, meta=meta)
            saved += 1
    finally:
        writer.close()
        generator.close()

    if args.require_samples is not None and saved < args.require_samples:
        raise RuntimeError(f"Teacher trajectory LMDB build only saved {saved} samples, required {args.require_samples}")
    print(f"Teacher trajectory LoRA LMDB ready: {args.out_dir} ({saved} samples)")


class LightX2VTeacherTrajectoryGenerator:
    """Caches a teacher trajectory prefix state and final teacher latent."""

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
    def make_pair(self, *, prompt: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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

        executed_steps: list[dict[str, Any]] = []
        target_steps: list[dict[str, Any]] = []
        x_pre_train_step: torch.Tensor | None = None
        train_sigma = scheduler.sigmas[train_step_index].to(device=self.device, dtype=torch.float32)

        for step_index in range(infer_steps):
            if step_index == train_step_index:
                x_pre_train_step = normalize_latent(scheduler.latents).detach().to(torch.float32).cpu()

            scheduler.step_pre(step_index=step_index)
            self.runner.model.infer(self.runner.inputs)
            flow_pred = scheduler.noise_pred.to(torch.float32)
            sigma = scheduler.sigmas[step_index].to(device=self.device, dtype=torch.float32)
            clean_pred = scheduler.latents.to(torch.float32) - sigma * flow_pred
            scheduler.step_post()

            sigma_next = None
            if step_index + 1 < infer_steps:
                sigma_next = float(scheduler.sigmas[step_index + 1].detach().cpu())
            item = {
                "step_index": step_index,
                "step_name": f"teacher_step{step_index + 1}",
                "sigma": float(sigma.detach().cpu()),
                "sigma_next": sigma_next,
                "clean_pred_shape": list(normalize_latent(clean_pred).shape),
            }
            if step_index < train_step_index:
                executed_steps.append(item)
            else:
                target_steps.append(item)

        if x_pre_train_step is None:
            raise RuntimeError("Failed to capture x_pre_train_step.")
        z_teacher_final = normalize_latent(scheduler.latents).detach().to(torch.float32).cpu()
        self.runner.end_run()

        recipe = {
            "mode": "lightx2v_distill_teacher_only",
            "recipe": "teacher_trajectory_lora_v1",
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
            "target_step_name": f"teacher_step{infer_steps}_final",
            "train_sigma": float(train_sigma.detach().cpu()),
            "sample_shift": float(self.args.sample_shift),
            "sample_guide_scale": float(self.args.sample_guide_scale),
            "enable_cfg": bool(self.args.enable_cfg),
            "seed": seed,
        }
        return x_pre_train_step, z_teacher_final, recipe

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
class ShardedTeacherTrajectoryLoRALMDBWriter:
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
        x_pre_train_step: torch.Tensor,
        z_teacher_final: torch.Tensor,
        *,
        prompt: str,
        seed: int,
        meta: dict[str, Any],
    ) -> None:
        if self.env is None or self.row_index >= self.shard_size:
            self._open_next_shard(x_pre_train_step, z_teacher_final)

        assert self.env is not None
        row = self.row_index
        x_np = _to_numpy_fp(x_pre_train_step, self.compression_dtype)
        z_np = _to_numpy_fp(z_teacher_final, self.compression_dtype)
        meta_json = json.dumps(meta, ensure_ascii=False)

        with self.env.begin(write=True) as txn:
            txn.put(_key("x_pre_train_step", row), x_np.tobytes())
            txn.put(_key("z_teacher_final", row), z_np.tobytes())
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

    def _open_next_shard(self, x_pre_train_step: torch.Tensor, z_teacher_final: torch.Tensor) -> None:
        self.close()
        self.shard_index += 1
        self.row_index = 0
        shard_dir = self.out_dir / f"shard_{self.shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        self.env = lmdb.open(str(shard_dir), map_size=self.map_size_gb * 1024**3, subdir=True, meminit=False)
        self.current_meta = {
            "num_samples": 0,
            "dtype": "float16",
            "x_pre_train_step_shape": list(x_pre_train_step.shape),
            "z_teacher_final_shape": list(z_teacher_final.shape),
            "schema": "wan_teacher_trajectory_lora_lmdb_v1",
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self.env is None or self.current_meta is None:
            return
        meta = dict(self.current_meta)
        meta["num_samples"] = self.row_index
        with self.env.begin(write=True) as txn:
            txn.put(b"metadata", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))


def normalize_latent(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected latent shape [C,T,H,W] or [1,C,T,H,W], got {tuple(tensor.shape)}")
    return tensor


def load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file",
        default=os.environ.get(
            "CR_DISTILL_PROMPTS_FILE",
            os.environ.get("CR_HF_PROMPTS_FILE", "prompts/vidprom_filtered_extended.txt"),
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=os.environ.get(
            "CR_DISTILL_TEACHER_TRAJ_LMDB_DIR",
            "data/changing_resolution_distill/lmdb_teacher_trajectory_lora_14b_cfgdistill_5k_step3",
        ),
    )
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
    parser.add_argument("--train_step_index", type=int, default=2, help="0-based step index; 2 captures x before step3.")
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--enable_cfg", action="store_true")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--require_samples", type=int)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--base_seed", type=int, default=9500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _to_numpy_fp(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float16).numpy().astype(dtype, copy=False)


def _key(name: str, row_id: int) -> bytes:
    return f"{name}_{row_id:08d}_data".encode("utf-8")


if __name__ == "__main__":
    main()
