from __future__ import annotations

# ruff: noqa: E402

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
import UNIV_adaptor.wan_runner  # noqa: F401,E402
from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all, validate_config_paths


def main() -> None:
    args = parse_args()
    job = load_job(args.job_json)
    rows = job["rows"]
    args.seed = int(rows[0]["seed"])
    validate_wan21_t2v_model_root(args.model_path)
    seed_all(args.seed)
    config = set_config(args)
    if config.get("parallel", False):
        raise SystemExit("UNIV sparse batch runner is single-device only")
    print_config(config)
    validate_config_paths(config)

    timing_path = Path(args.timing_jsonl).resolve()
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text("", encoding="utf-8")
    for row in rows:
        output = Path(row["output"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    synchronize_device()
    init_started = time.perf_counter()
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    synchronize_device()
    append_jsonl(
        timing_path,
        {
            "kind": "initialization",
            "job_id": job["job_id"],
            "action_id": job["action_id"],
            "model_cls": config["model_cls"],
            "elapsed_s": time.perf_counter() - init_started,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            **cuda_memory_snapshot("post_initialization"),
        },
    )

    segment_times: list[float] = []
    original_run_segment = runner.run_segment

    def timed_run_segment(*segment_args, **segment_kwargs):
        synchronize_device()
        started = time.perf_counter()
        result = original_run_segment(*segment_args, **segment_kwargs)
        synchronize_device()
        segment_times.append(time.perf_counter() - started)
        return result

    runner.run_segment = timed_run_segment
    for position, row in enumerate(rows):
        seed = int(row["seed"])
        output = Path(row["output"]).resolve()
        input_info = init_empty_input_info(args.task, args.support_tasks)
        payload = vars(args).copy()
        payload.update(
            {
                "seed": seed,
                "prompt": row["prompt"],
                "negative_prompt": args.negative_prompt,
                "save_result_path": str(output),
                "target_video_length": args.target_video_length,
            }
        )
        seed_all(seed)
        update_input_info_from_dict(input_info, payload)
        segment_times.clear()
        reset_peak_memory()
        synchronize_device()
        started = time.perf_counter()
        runner.run_pipeline(input_info)
        synchronize_device()
        pipeline_s = time.perf_counter() - started
        if not output.is_file() or output.stat().st_size < 1024:
            raise RuntimeError(f"missing or undersized generated video: {output}")
        timing = {
            "kind": "video",
            "job_id": job["job_id"],
            "action_id": job["action_id"],
            "observation_id": row["observation_id"],
            "prompt_key": row["prompt_key"],
            "prompt_sha256": row["prompt_sha256"],
            "position": position,
            "base_seed": int(row["base_seed"]),
            "seed": seed,
            "model_cls": config["model_cls"],
            "pipeline_elapsed_s": pipeline_s,
            "segment_elapsed_s": sum(segment_times),
            "segment_count": len(segment_times),
            "output": str(output),
            "output_bytes": output.stat().st_size,
            **cuda_memory_snapshot("video_peak"),
        }
        runtime = getattr(runner, "univ_runtime_record", None)
        if isinstance(runtime, dict) and runtime:
            timing["univ_stage_timing_s"] = runtime.get("timing_seconds", {})
            timing["univ_schedule"] = runtime.get("schedule", {})
            timing["univ_transition"] = runtime.get("transition", {})
        append_jsonl(timing_path, timing)
        logger.info(
            f"[sparse-action] {row['observation_id']} seed={seed} "
            f"pipeline={pipeline_s:.3f}s output={output}"
        )
    cleanup_distributed()


def load_job(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "univ_sparse_action_job_input_v1":
        raise ValueError("unsupported sparse action job input")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sparse action job must contain rows")
    observation_ids = []
    for row in rows:
        for field in (
            "observation_id",
            "prompt_key",
            "prompt",
            "prompt_sha256",
            "base_seed",
            "seed",
            "action_id",
            "output",
        ):
            if field not in row:
                raise ValueError(f"sparse action job row requires {field}")
        if row["action_id"] != value.get("action_id"):
            raise ValueError("job row action_id mismatch")
        observation_ids.append(row["observation_id"])
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("duplicate observation id in sparse action job")
    return value


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_memory_snapshot(label: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"memory_label": label, "cuda_memory_available": False}
    gib = 1024.0**3
    return {
        "memory_label": label,
        "cuda_memory_available": True,
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one explicit sparse prompt-action job with one model load"
    )
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--model_cls", required=True)
    parser.add_argument("--task", default="t2v")
    parser.add_argument("--support_tasks", nargs="+", default=[])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--return_result_tensor", action="store_true")
    parser.add_argument("--target_shape", type=int, nargs="+", default=[])
    parser.add_argument("--target_video_length", type=int, default=81)
    parser.add_argument("--timing-jsonl", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
