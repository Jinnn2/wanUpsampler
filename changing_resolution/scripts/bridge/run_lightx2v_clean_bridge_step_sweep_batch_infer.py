from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

lightx2v_repo = os.environ.get("LIGHTX2V_REPO")
if lightx2v_repo and lightx2v_repo not in sys.path:
    sys.path.insert(0, lightx2v_repo)

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *  # noqa: F403
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

from changing_resolution.lightx2v_clean_bridge import (
    WanScheduler4CleanResizerBridgeInterface,
)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(
        Path(args.prompts_file), offset=args.prompt_offset, limit=args.limit
    )
    steps = parse_steps(args.change_steps, infer_steps=args.infer_steps)
    if not prompts:
        raise SystemExit(f"No prompts selected from {args.prompts_file}")

    seed_all(args.seed)
    config = set_config(args)
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(
            os.getenv("PLATFORM", "cuda"), None
        )
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)
    validate_config_paths(config)

    out_root = Path(args.out_root)
    for name in ("stage2_handoff", "interp_handoff", "baseline50_lowres"):
        (out_root / "videos" / name).mkdir(parents=True, exist_ok=True)

    # The expensive WAN, text encoder, VAE, and Stage2 weights are initialized
    # exactly once. Every job below only replaces the lightweight scheduler.
    with ProfilingContext4DebugL1("360p Step Sweep Batch Total Cost"):
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        loaded_clean_resizer = runner.clean_latent_resizer

        for local_index, prompt in enumerate(prompts):
            global_index = args.prompt_offset + local_index
            seed = args.seed + global_index
            sample_label = f"{global_index:03d}"

            run_job(
                runner,
                loaded_clean_resizer,
                args,
                mode="baseline",
                step=args.infer_steps,
                prompt=prompt,
                seed=seed,
                output=out_root
                / "videos"
                / "baseline50_lowres"
                / f"{sample_label}_seed{seed}_baseline50_lowres.mp4",
            )

            for step in steps:
                stem = f"{sample_label}_seed{seed}_step{step:02d}"
                run_job(
                    runner,
                    loaded_clean_resizer,
                    args,
                    mode="stage2",
                    step=step,
                    prompt=prompt,
                    seed=seed,
                    output=out_root
                    / "videos"
                    / "stage2_handoff"
                    / f"{stem}_stage2.mp4",
                )
                run_job(
                    runner,
                    loaded_clean_resizer,
                    args,
                    mode="interp",
                    step=step,
                    prompt=prompt,
                    seed=seed,
                    output=out_root
                    / "videos"
                    / "interp_handoff"
                    / f"{stem}_interp.mp4",
                )

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


def run_job(runner, loaded_clean_resizer, args, *, mode, step, prompt, seed, output):
    if args.skip_existing and output.is_file() and output.stat().st_size > 0:
        logger.info(f"[skip] {output}")
        return

    configure_scheduler(
        runner,
        loaded_clean_resizer,
        mode=mode,
        step=step,
        infer_steps=args.infer_steps,
        lr_height=args.lr_height,
        lr_width=args.lr_width,
        hr_height=args.hr_height,
        hr_width=args.hr_width,
        lr_latent_height=args.lr_latent_height,
        lr_latent_width=args.lr_latent_width,
    )

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
    logger.info(
        f"[360p-sweep] mode={mode} step={step}/{args.infer_steps} "
        f"seed={seed} output={output}"
    )
    logger.info(f"[360p-sweep] prompt={prompt}")
    seed_all(seed)
    update_input_info_from_dict(input_info, payload)
    runner.run_pipeline(input_info)


def configure_scheduler(
    runner,
    loaded_clean_resizer,
    *,
    mode,
    step,
    infer_steps,
    lr_height,
    lr_width,
    hr_height,
    hr_width,
    lr_latent_height,
    lr_latent_width,
):
    if mode == "baseline":
        runner.set_config(
            {
                "target_height": lr_height,
                "target_width": lr_width,
                "changing_resolution": False,
            }
        )
        scheduler = WanScheduler(runner.config)
        runner.clean_latent_resizer = loaded_clean_resizer
    elif mode in {"stage2", "interp"}:
        runner.set_config(
            {
                "target_height": hr_height,
                "target_width": hr_width,
                "changing_resolution": True,
                "resolution_rate": [lr_height / hr_height],
                "wan_lowres_latent_size": [lr_latent_height, lr_latent_width],
                "changing_resolution_steps": [step],
                "infer_steps": infer_steps,
            }
        )
        scheduler = WanScheduler4CleanResizerBridgeInterface(
            WanScheduler, runner.config
        )
        if mode == "stage2":
            runner.clean_latent_resizer = loaded_clean_resizer
            scheduler.set_clean_latent_resizer(loaded_clean_resizer)
        else:
            # WanCleanResizerBridgeRunner.init_run() attaches this attribute to
            # a new scheduler. Set it to None only for the interpolation job.
            runner.clean_latent_resizer = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    runner.scheduler = scheduler
    runner.model.set_scheduler(scheduler)


def load_prompts(path: Path, *, offset: int, limit: int) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return prompts[offset : offset + limit]


def parse_steps(value: str, *, infer_steps: int) -> list[int]:
    steps = [int(item) for item in value.split() if item]
    if not steps:
        raise SystemExit("No change steps provided")
    invalid = [step for step in steps if step < 1 or step > infer_steps]
    if invalid:
        raise SystemExit(f"Invalid change steps {invalid}; expected [1, {infer_steps}]")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--model_cls", type=str, default="wan2.1_clean_resizer_bridge")
    parser.add_argument("--task", type=str, default="t2v")
    parser.add_argument("--support_tasks", type=str, nargs="+", default=[])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sf_model_path", type=str, required=False)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--last_frame_path", type=str, default="")
    parser.add_argument("--audio_path", type=str, default="")
    parser.add_argument("--image_strength", type=str, default="1.0")
    parser.add_argument("--image_frame_idx", type=str, default="")
    parser.add_argument("--src_ref_images", type=str, default=None)
    parser.add_argument("--src_video", type=str, default=None)
    parser.add_argument("--src_mask", type=str, default=None)
    parser.add_argument("--src_pose_path", type=str, default=None)
    parser.add_argument("--src_face_path", type=str, default=None)
    parser.add_argument("--src_bg_path", type=str, default=None)
    parser.add_argument("--src_mask_path", type=str, default=None)
    parser.add_argument("--pose", type=str, default=None)
    parser.add_argument("--action_path", type=str, default=None)
    parser.add_argument("--action_ckpt", type=str, default=None)
    parser.add_argument("--return_result_tensor", action="store_true")
    parser.add_argument("--target_shape", type=int, nargs="+", default=[])
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument("--aspect_ratio", type=str, default="")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--sr_ratio", type=float, default=2.0)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--change_steps", type=str, required=True)
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--lr_height", type=int, default=368)
    parser.add_argument("--lr_width", type=int, default=640)
    parser.add_argument("--hr_height", type=int, default=720)
    parser.add_argument("--hr_width", type=int, default=1248)
    parser.add_argument("--lr_latent_height", type=int, default=46)
    parser.add_argument("--lr_latent_width", type=int, default=80)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
