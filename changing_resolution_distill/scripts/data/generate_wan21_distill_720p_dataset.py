from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

lightx2v_repo = os.environ.get("LIGHTX2V_REPO")
if lightx2v_repo and lightx2v_repo not in sys.path:
    sys.path.insert(0, lightx2v_repo)

import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm

from lightx2v.common.ops import *  # noqa: F403
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@dataclass(frozen=True)
class GenerationJob:
    prompt: str
    prompt_index: int
    local_index: int
    seed: int
    out_path: Path


def main() -> None:
    args = parse_args()
    jobs = collect_jobs(args)
    pending = [job for job in jobs if not is_valid_video(job.out_path)]
    skipped = len(jobs) - len(pending)
    print(
        f"Selected prompts={len(jobs)} pending={len(pending)} skipped_existing={skipped} out_dir={args.out_dir}",
        flush=True,
    )
    if not pending:
        return

    seed_all(args.seed)
    torch.set_grad_enabled(False)
    config_args = make_lightx2v_args(args)
    config = set_config(config_args)
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)
    validate_config_paths(config)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    logger.info(f"Loaded {config['model_cls']} once; generating {len(pending)} video(s).")

    try:
        for job in tqdm(pending, desc="14B CfgDistill 720p", dynamic_ncols=True):
            generate_one(runner, args, job)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if hasattr(runner, "end_run"):
            try:
                runner.end_run()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group cleaned up")


def generate_one(runner, args: argparse.Namespace, job: GenerationJob) -> None:
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    input_info = init_empty_input_info(args.task, args.support_tasks)
    update_input_info_from_dict(
        input_info,
        {
            "seed": job.seed,
            "prompt": job.prompt,
            "negative_prompt": args.negative_prompt,
            "save_result_path": str(job.out_path),
            "return_result_tensor": False,
        },
    )
    logger.info(f"Generating {job.out_path} seed={job.seed} prompt_index={job.prompt_index}")
    runner.run_pipeline(input_info)


def collect_jobs(args: argparse.Namespace) -> list[GenerationJob]:
    prompts = load_prompts(args.prompts_file)
    selected = prompts[args.prompt_offset : args.prompt_offset + args.max_prompts]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[GenerationJob] = []
    for local_index, prompt in enumerate(selected):
        prompt_index = args.prompt_offset + local_index
        seed = args.start_seed + local_index
        sample_id = f"{prompt_index:06d}"
        out_path = args.out_dir / f"{args.filename_prefix}_{sample_id}_seed{seed}.mp4"
        jobs.append(
            GenerationJob(
                prompt=prompt,
                prompt_index=prompt_index,
                local_index=local_index,
                seed=seed,
                out_path=out_path,
            )
        )
    return jobs


def load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def is_valid_video(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return True
    result = subprocess.run(
        [ffprobe, "-v", "error", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def make_lightx2v_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        model_cls=args.model_cls,
        task=args.task,
        support_tasks=args.support_tasks,
        model_path=str(args.model_path),
        sf_model_path=None,
        config_json=str(args.config_json),
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
        action_ckpt=None,
        save_result_path=None,
        return_result_tensor=False,
        target_shape=[],
        target_video_length=args.target_video_length,
        aspect_ratio="",
        video_path=None,
        sr_ratio=2.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cls", default="wan2.1_distill")
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--support_tasks", nargs="+", default=[])
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--config_json", type=Path, required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--start_seed", type=int, default=620000)
    parser.add_argument("--max_prompts", type=int, default=5000)
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument("--filename_prefix", default="wan21_14b_cfgdistill_720p")
    return parser.parse_args()


if __name__ == "__main__":
    main()
