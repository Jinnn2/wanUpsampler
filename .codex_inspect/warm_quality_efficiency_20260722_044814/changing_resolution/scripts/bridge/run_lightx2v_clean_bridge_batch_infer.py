from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

lightx2v_repo = os.environ.get("LIGHTX2V_REPO")
if lightx2v_repo and lightx2v_repo not in sys.path:
    sys.path.insert(0, lightx2v_repo)

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *  # noqa: F403
from lightx2v.models.runners.wan.wan_runner import WanRunner  # noqa: F401
import changing_resolution.lightx2v_clean_bridge  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts_file), offset=args.prompt_offset, limit=args.limit)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts_file}")

    seed_all(args.seed)
    config = set_config(args)
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)
    validate_config_paths(config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for index, prompt in enumerate(prompts, start=args.prompt_offset):
        seed = args.seed + index if args.increment_seed else args.seed
        output = out_dir / f"{args.name_prefix}_{index:02d}_seed{seed}.mp4"
        if args.skip_existing and output.is_file() and output.stat().st_size > 0:
            logger.info(f"[batch:skip] {output}")
            continue
        jobs.append((index, prompt, seed, output))

    if not jobs:
        logger.info(f"[batch] all {len(prompts)} output(s) already exist; skip model initialization")
        cleanup_distributed()
        return

    timing_path = Path(args.timing_jsonl).resolve() if args.timing_jsonl else None
    if timing_path is not None:
        if args.skip_existing:
            raise SystemExit("--timing-jsonl cannot be combined with --skip-existing")
        if args.timing_warmup < 0 or args.timing_warmup >= len(jobs):
            raise SystemExit(
                f"--timing-warmup must be in [0, {len(jobs) - 1}] for {len(jobs)} job(s)"
            )
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("", encoding="utf-8")

    with ProfilingContext4DebugL1("Batch Total Cost"):
        synchronize_device()
        init_started = time.perf_counter()
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        synchronize_device()
        init_elapsed = time.perf_counter() - init_started
        if timing_path is not None:
            append_timing(
                timing_path,
                {
                    "kind": "initialization",
                    "model_cls": config["model_cls"],
                    "elapsed_s": init_elapsed,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                },
            )

        timing_state: dict[str, list[float]] = {"segments": []}
        if timing_path is not None:
            original_run_segment = runner.run_segment

            def timed_run_segment(*segment_args, **segment_kwargs):
                synchronize_device()
                segment_started = time.perf_counter()
                result = original_run_segment(*segment_args, **segment_kwargs)
                synchronize_device()
                timing_state["segments"].append(time.perf_counter() - segment_started)
                return result

            runner.run_segment = timed_run_segment

        for job_index, (index, prompt, seed, output) in enumerate(jobs):
            input_info = init_empty_input_info(args.task, args.support_tasks)
            payload = vars(args).copy()
            payload.update(
                {
                    "seed": seed,
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "save_result_path": str(output),
                    "target_video_length": args.target_video_length,
                }
            )
            logger.info(f"[batch] {args.name_prefix} index={index:02d} seed={seed} output={output}")
            logger.info(f"[batch] prompt={prompt}")
            seed_all(seed)
            update_input_info_from_dict(input_info, payload)
            timing_state["segments"] = []
            synchronize_device()
            pipeline_started = time.perf_counter()
            runner.run_pipeline(input_info)
            synchronize_device()
            pipeline_elapsed = time.perf_counter() - pipeline_started
            if timing_path is not None:
                append_timing(
                    timing_path,
                    {
                        "kind": "video",
                        "phase": "warmup" if job_index < args.timing_warmup else "measured",
                        "repeat": (
                            job_index
                            if job_index < args.timing_warmup
                            else job_index - args.timing_warmup
                        ),
                        "prompt_index": index,
                        "seed": seed,
                        "model_cls": config["model_cls"],
                        "pipeline_elapsed_s": pipeline_elapsed,
                        "denoise_elapsed_s": sum(timing_state["segments"]),
                        "segment_count": len(timing_state["segments"]),
                        "output": str(output),
                    },
                )

    cleanup_distributed()


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def append_timing(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


def load_prompts(path: Path, *, offset: int, limit: int) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return prompts[offset : offset + limit] if limit > 0 else prompts[offset:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one WAN configuration over multiple prompts with one model load.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--increment_seed", action="store_true")
    parser.add_argument("--model_cls", required=True)
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--support_tasks", nargs="+", default=[])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sf_model_path")
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--image_path", default="")
    parser.add_argument("--last_frame_path", default="")
    parser.add_argument("--audio_path", default="")
    parser.add_argument("--image_strength", default="1.0")
    parser.add_argument("--image_frame_idx", default="")
    parser.add_argument("--src_ref_images")
    parser.add_argument("--src_video")
    parser.add_argument("--src_mask")
    parser.add_argument("--src_pose_path")
    parser.add_argument("--src_face_path")
    parser.add_argument("--src_bg_path")
    parser.add_argument("--src_mask_path")
    parser.add_argument("--pose")
    parser.add_argument("--action_ckpt")
    parser.add_argument("--return_result_tensor", action="store_true")
    parser.add_argument("--target_shape", type=int, nargs="+", default=[])
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument("--aspect_ratio", default="")
    parser.add_argument("--video_path")
    parser.add_argument("--sr_ratio", type=float, default=2.0)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--timing-jsonl",
        default="",
        help="Optional raw timing output. Measures model initialization, each full pipeline, and run_segment.",
    )
    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=0,
        help="Number of leading jobs labeled warmup in --timing-jsonl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
