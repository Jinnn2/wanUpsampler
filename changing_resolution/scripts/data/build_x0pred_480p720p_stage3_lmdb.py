from __future__ import annotations

import argparse
import json
import os
import shutil
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

from wan_sr.data import CleanLatentLMDBDataset


def main() -> None:
    args = parse_args()
    prepare_output_dir(Path(args.out_dir), overwrite=args.overwrite)

    source = CleanLatentLMDBDataset(args.source_lmdb, strict_channels=True)
    start_index = int(args.offset)
    if start_index < 0 or start_index >= len(source):
        raise ValueError(f"offset must be in [0, {len(source) - 1}], got {start_index}")
    if args.max_samples is not None:
        end_index = min(len(source), start_index + int(args.max_samples))
    else:
        end_index = len(source)
    sample_count = end_index - start_index
    if sample_count <= 0:
        raise RuntimeError("No source samples selected.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    generator = None
    if args.mode == "lightx2v":
        generator = LightX2VX0PredGenerator(args, device=device, dtype=dtype)
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
        for source_index in tqdm(range(start_index, end_index), desc="stage3 x0_pred", dynamic_ncols=True):
            row = source[source_index]
            prompt = row["prompt"]
            z0_lr = row["z0_lr"].float()
            z0_hr = row["z0_hr"].float()

            if generator is None:
                x0_pred_lr = z0_lr
                recipe = {"mode": "clean_copy", "note": "debug path only; not a real Stage 3 training domain"}
            else:
                seed = int(args.base_seed) + source_index
                x0_pred_lr, recipe = generator.make_x0_pred(z0_lr, prompt=prompt, seed=seed)

            meta = merge_meta(row.get("meta_json", "{}"))
            meta.update(
                {
                    "task": "changing_resolution_x0pred_stage3_480p_to_720p",
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
        raise RuntimeError(f"Stage 3 LMDB build only saved {saved} samples, required {args.require_samples}")
    print(f"Stage 3 x0-pred LMDB ready: {args.out_dir} ({saved} samples)")


class LightX2VX0PredGenerator:
    """Generate x0_pred by noising clean LR latent and running one Wan denoiser step."""

    def __init__(self, args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> None:
        lightx2v_repo = args.lightx2v_repo or os.environ.get("LIGHTX2V_REPO")
        if lightx2v_repo and lightx2v_repo not in sys.path:
            sys.path.insert(0, lightx2v_repo)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        import importlib

        importlib.import_module("lightx2v.common.ops")
        from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
        from lightx2v.utils.registry_factory import RUNNER_REGISTER
        from lightx2v.utils.set_config import set_config, set_parallel_config
        from lightx2v.utils.utils import seed_all, validate_config_paths
        from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

        import changing_resolution.lightx2v_clean_bridge  # noqa: F401

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
                    "infer_steps": int(args.infer_steps),
                    "target_video_length": int(args.num_frames),
                    "sample_shift": float(args.sample_shift),
                    "sample_guide_scale": float(args.sample_guide_scale),
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
        denoise_step = int(self.args.denoise_step)
        step_index = denoise_step - 1
        if denoise_step < 1 or denoise_step > int(self.args.infer_steps):
            raise ValueError(f"denoise_step must be in [1, {self.args.infer_steps}], got {denoise_step}")

        z0_device = z0_lr.to(device=self.device, dtype=torch.float32)
        noise = torch.randn(
            z0_device.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=torch.float32,
        )
        sigma = scheduler.sigmas[step_index].to(device=self.device, dtype=torch.float32)
        x_t = (1.0 - sigma) * z0_device + sigma * noise

        scheduler.latents = x_t
        scheduler.step_pre(step_index=step_index)
        self.runner.model.infer(self.runner.inputs)
        noise_pred = scheduler.noise_pred.to(torch.float32)
        x0_pred = scheduler.latents.to(torch.float32) - sigma * noise_pred

        self.runner.end_run()
        recipe = {+
            "mode": "lightx2v",
            "infer_steps": int(self.args.infer_steps),
            "denoise_step": denoise_step,
            "step_index": step_index,
            "sigma": float(sigma.detach().cpu()),
            "sample_shift": float(self.args.sample_shift),
            "sample_guide_scale": float(self.args.sample_guide_scale),
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


@dataclass
class ShardedX0PredLatentLMDBWriter:
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

    def write(self, x0_pred_lr: torch.Tensor, z0_lr: torch.Tensor, z0_hr: torch.Tensor, prompt: str, meta: dict[str, Any]) -> None:
        if self.env is None or self.row_index >= self.shard_size:
            self._open_next_shard(x0_pred_lr, z0_lr, z0_hr)

        assert self.env is not None
        row = self.row_index
        x0_pred_lr_np = _to_numpy_fp(x0_pred_lr, self.compression_dtype)
        z0_lr_np = _to_numpy_fp(z0_lr, self.compression_dtype)
        z0_hr_np = _to_numpy_fp(z0_hr, self.compression_dtype)
        meta_json = json.dumps(meta, ensure_ascii=False)

        with self.env.begin(write=True) as txn:
            txn.put(_key("x0_pred_lr", row), x0_pred_lr_np.tobytes())
            txn.put(_key("z0_lr", row), z0_lr_np.tobytes())
            txn.put(_key("z0_hr", row), z0_hr_np.tobytes())
            txn.put(_key("prompt", row), prompt.encode("utf-8"))
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

    def _open_next_shard(self, x0_pred_lr: torch.Tensor, z0_lr: torch.Tensor, z0_hr: torch.Tensor) -> None:
        self.close()
        self.shard_index += 1
        self.row_index = 0
        shard_dir = self.out_dir / f"shard_{self.shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        self.env = lmdb.open(str(shard_dir), map_size=self.map_size_gb * 1024**3, subdir=True, meminit=False)
        self.current_meta = {
            "num_samples": 0,
            "dtype": "float16",
            "x0_pred_lr_shape": list(x0_pred_lr.shape),
            "z0_lr_shape": list(z0_lr.shape),
            "z0_hr_shape": list(z0_hr.shape),
            "schema": "wan_x0pred_latent_pair_lmdb_v1",
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
    parser.add_argument("--source_lmdb", default="data/changing_resolution/lmdb_480p720p_1k")
    parser.add_argument("--out_dir", default="data/changing_resolution/lmdb_x0pred_480p720p_stage3_step45")
    parser.add_argument("--mode", choices=["lightx2v", "clean_copy"], default="lightx2v")
    parser.add_argument("--lightx2v_repo", default=os.environ.get("LIGHTX2V_REPO"))
    parser.add_argument("--model_path", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--config_json", default="changing_resolution/configs/wan_t2v_stage3_x0pred_480p.json")
    parser.add_argument("--model_cls", default="wan2.1")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--denoise_step", type=int, default=45)
    parser.add_argument("--sample_shift", type=float, default=8.0)
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--require_samples", type=int)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--base_seed", type=int, default=9300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output LMDB dir is not empty: {out_dir}. Pass --overwrite to rebuild.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def merge_meta(meta_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(meta_json)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"source_meta": meta_json}


def _to_numpy_fp(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float16).numpy().astype(dtype, copy=False)


def _key(name: str, row_id: int) -> bytes:
    return f"{name}_{row_id:08d}_data".encode("utf-8")


if __name__ == "__main__":
    main()
