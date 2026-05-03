from __future__ import annotations

import argparse
import os
import sys

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
    seed_all(args.seed)

    config = set_config(args)
    input_info = init_empty_input_info(args.task, args.support_tasks)

    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)
    validate_config_paths(config)

    with ProfilingContext4DebugL1("Total Cost"):
        runner = init_runner(config)
        update_input_info_from_dict(input_info, args.__dict__)
        runner.run_pipeline(input_info)

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


def init_runner(config):
    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    return runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cls", type=str, required=True)
    parser.add_argument("--task", type=str, default="t2v")
    parser.add_argument("--support_tasks", type=str, nargs="+", default=[])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sf_model_path", type=str, required=False)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
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
    parser.add_argument("--save_result_path", type=str, default=None)
    parser.add_argument("--return_result_tensor", action="store_true")
    parser.add_argument("--target_shape", type=int, nargs="+", default=[])
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument("--aspect_ratio", type=str, default="")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--sr_ratio", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
