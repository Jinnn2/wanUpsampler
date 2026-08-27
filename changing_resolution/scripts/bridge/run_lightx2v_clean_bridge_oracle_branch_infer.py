# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


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
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.base.global_var import AI_DEVICE

import changing_resolution.lightx2v_clean_bridge  # noqa: F401,E402
from changing_resolution.lightx2v_clean_bridge import (  # noqa: E402
    WanScheduler4CleanResizerBridgeInterface,
    count_lora_branches,
)


FORMAL_STEPS = [30, 35, *range(40, 51)]


def main() -> None:
    args = parse_args()
    steps = parse_steps(args.change_steps, infer_steps=args.infer_steps)
    prompts = load_prompts(
        Path(args.prompts_file), offset=args.prompt_offset, limit=args.limit
    )
    if not prompts:
        raise SystemExit(f"No prompts selected from {args.prompts_file}")

    protocol_prompt_offset = (
        args.protocol_prompt_offset
        if args.protocol_prompt_offset is not None
        else args.prompt_offset
    )
    protocol_prompt_limit = (
        args.protocol_prompt_limit
        if args.protocol_prompt_limit is not None
        else args.limit
    )
    if protocol_prompt_offset < 0 or protocol_prompt_limit < 1:
        raise SystemExit(
            "--protocol-prompt-offset must be non-negative and "
            "--protocol-prompt-limit must be positive"
        )
    execution_end = args.prompt_offset + len(prompts)
    protocol_end = protocol_prompt_offset + protocol_prompt_limit
    if args.prompt_offset < protocol_prompt_offset or execution_end > protocol_end:
        raise SystemExit(
            "Execution prompt slice must be contained in the canonical protocol slice: "
            f"execution=[{args.prompt_offset}, {execution_end}), "
            f"protocol=[{protocol_prompt_offset}, {protocol_end})"
        )
    protocol_prompts = load_prompts(
        Path(args.prompts_file),
        offset=protocol_prompt_offset,
        limit=protocol_prompt_limit,
    )
    if len(protocol_prompts) != protocol_prompt_limit:
        raise SystemExit(
            "Canonical protocol prompt slice is incomplete: "
            f"expected={protocol_prompt_limit}, found={len(protocol_prompts)}, "
            f"offset={protocol_prompt_offset}"
        )

    seed_all(args.seed)
    config = set_config(args)
    validate_protocol(args, config, steps)
    print_config(config)
    validate_config_paths(config)

    out_root = Path(args.out_root).resolve()
    prepare_output_tree(out_root, steps, args.execution_mode)
    write_protocol(
        out_root,
        args,
        config,
        steps,
        protocol_prompts,
        protocol_prompt_offset=protocol_prompt_offset,
    )
    write_prompt_map(
        out_root,
        args,
        steps,
        protocol_prompts,
        prompt_offset=protocol_prompt_offset,
    )

    try:
        with ProfilingContext4DebugL1("TAA-free oracle sweep model residency"):
            synchronize_device()
            init_started = time.perf_counter()
            runner = RUNNER_REGISTER[config["model_cls"]](config)
            runner.init_modules()
            synchronize_device()
            model_init_seconds = time.perf_counter() - init_started
            clean_resizer = runner.clean_latent_resizer

            runtime_lora_branches = count_lora_branches(runner.model)
            if runtime_lora_branches:
                raise RuntimeError(
                    "TAA-free oracle loaded runtime LoRA branches unexpectedly: "
                    f"count={runtime_lora_branches}"
                )

            if args.execution_mode == "independent" and args.independent_warmup:
                warmup_seed = args.seed + args.prompt_offset
                warmup_output = (
                    out_root
                    / "independent"
                    / "warmup"
                    / f"step{steps[0]:02d}_seed{warmup_seed}.mp4"
                )
                configure_bridge_scheduler(
                    runner,
                    clean_resizer,
                    args,
                    change_step=steps[0],
                )
                warmup_seconds = run_standard_pipeline(
                    runner,
                    args,
                    prompt=prompts[0],
                    seed=warmup_seed,
                    output=warmup_output,
                )
                write_json(
                    out_root / "independent" / "warmup.json",
                    {
                        "candidate_step": steps[0],
                        "seed": warmup_seed,
                        "warm_pipeline_seconds": warmup_seconds,
                        "output": str(warmup_output),
                        "excluded_from_measurements": True,
                    },
                )

            for local_index, prompt in enumerate(prompts):
                prompt_index = args.prompt_offset + local_index
                seed = args.seed + prompt_index
                sample_id = f"{prompt_index:04d}_seed{seed}"
                if args.execution_mode == "branch":
                    run_branch_sample(
                        runner,
                        clean_resizer,
                        args,
                        steps=steps,
                        prompt=prompt,
                        prompt_index=prompt_index,
                        seed=seed,
                        sample_id=sample_id,
                        out_root=out_root,
                        model_init_seconds=model_init_seconds,
                    )
                else:
                    run_independent_sample(
                        runner,
                        clean_resizer,
                        args,
                        steps=steps,
                        prompt=prompt,
                        prompt_index=prompt_index,
                        seed=seed,
                        sample_id=sample_id,
                        out_root=out_root,
                        model_init_seconds=model_init_seconds,
                    )
    finally:
        cleanup_distributed()


def run_branch_sample(
    runner,
    clean_resizer,
    args,
    *,
    steps: list[int],
    prompt: str,
    prompt_index: int,
    seed: int,
    sample_id: str,
    out_root: Path,
    model_init_seconds: float,
) -> None:
    manifest_path = out_root / "manifests" / f"{sample_id}.json"
    previous = load_json_object(manifest_path)
    previous_branches = {
        int(row["candidate_step"]): row
        for row in previous.get("branches", [])
        if isinstance(row, dict) and "candidate_step" in row
    }

    pending_steps = []
    for step in steps:
        output = branch_video_path(out_root, step, sample_id)
        latent = branch_latent_path(out_root, step, sample_id)
        complete = output.is_file() and output.stat().st_size > 0
        if args.save_latents:
            complete = complete and latent.is_file() and latent.stat().st_size > 0
        if not (args.skip_existing and complete and step in previous_branches):
            pending_steps.append(step)

    native_output = out_root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4"
    native_complete = native_output.is_file() and native_output.stat().st_size > 0
    need_native = args.include_native_hr and not (
        args.skip_existing and native_complete and previous.get("native_hr")
    )

    if not pending_steps and not need_native:
        logger.info(
            f"[oracle:skip] sample={sample_id}; all branch/native outputs complete"
        )
        return

    logger.info(
        f"[oracle:branch] sample={sample_id} pending_steps={pending_steps} "
        f"native_hr={need_native}"
    )
    logger.info(f"[oracle:branch] prompt={prompt}")

    new_rows: list[dict[str, Any]] = []
    if pending_steps:
        new_rows = collect_and_run_branches(
            runner,
            clean_resizer,
            args,
            steps=pending_steps,
            prompt=prompt,
            seed=seed,
            sample_id=sample_id,
            out_root=out_root,
        )

    native_row = previous.get("native_hr")
    if need_native:
        native_row = run_native_hr_job(
            runner,
            args,
            prompt=prompt,
            seed=seed,
            output=native_output,
        )

    merged = dict(previous_branches)
    merged.update({int(row["candidate_step"]): row for row in new_rows})
    manifest = {
        "schema": "wan_taa_free_oracle_v1",
        "execution_mode": "branch",
        "prompt_index": prompt_index,
        "prompt": prompt,
        "seed": seed,
        "candidate_steps": steps,
        "taa_enabled": False,
        "model_init_seconds_process_shared": model_init_seconds,
        "timing_note": (
            "estimated_warm_pipeline_seconds reconstructs one online branch from "
            "input encoding, scheduler preparation, the measured LR prefix, lifting, "
            "the HR suffix, VAE decode, and video save. Use execution_mode=independent "
            "for confirmatory single-branch timing."
        ),
        "branches": [merged[step] for step in steps if step in merged],
        "native_hr": native_row,
    }
    write_json(manifest_path, manifest)


@torch.no_grad()
def collect_and_run_branches(
    runner,
    clean_resizer,
    args,
    *,
    steps: list[int],
    prompt: str,
    seed: int,
    sample_id: str,
    out_root: Path,
) -> list[dict[str, Any]]:
    with managed_runner_run(runner):
        return _collect_and_run_branches_impl(
            runner,
            clean_resizer,
            args,
            steps=steps,
            prompt=prompt,
            seed=seed,
            sample_id=sample_id,
            out_root=out_root,
        )


@torch.no_grad()
def _collect_and_run_branches_impl(
    runner,
    clean_resizer,
    args,
    *,
    steps: list[int],
    prompt: str,
    seed: int,
    sample_id: str,
    out_root: Path,
) -> list[dict[str, Any]]:
    configure_bridge_scheduler(runner, clean_resizer, args, change_step=min(steps))
    input_info = make_input_info(args, prompt=prompt, seed=seed, output=None)
    seed_all(seed)
    runner.input_info = input_info

    synchronize_device()
    encode_started = time.perf_counter()
    runner.inputs = runner.run_input_encoder()
    synchronize_device()
    input_encode_seconds = time.perf_counter() - encode_started

    synchronize_device()
    prepare_started = time.perf_counter()
    runner.init_run()
    synchronize_device()
    scheduler_prepare_seconds = time.perf_counter() - prepare_started
    if runner.video_segment_num != 1:
        raise RuntimeError(
            "Oracle branching currently requires a single WAN video segment; "
            f"got {runner.video_segment_num}."
        )
    runner.init_run_segment(0)

    prefix_scheduler = runner.model.scheduler
    if (
        not hasattr(prefix_scheduler, "latents_list")
        or len(prefix_scheduler.latents_list) != 2
    ):
        raise RuntimeError(
            "Expected one LR and one HR latent noise tensor from the changing-resolution scheduler."
        )
    lr_shape = tuple(prefix_scheduler.latents_list[0].shape)
    hr_shape = tuple(prefix_scheduler.latents_list[1].shape)
    expected_lr_spatial = (args.lr_latent_height, args.lr_latent_width)
    expected_hr_spatial = (args.hr_height // 8, args.hr_width // 8)
    if lr_shape[-2:] != expected_lr_spatial or hr_shape[-2:] != expected_hr_spatial:
        raise RuntimeError(
            "Resolved latent geometry does not match the oracle protocol: "
            f"LR={lr_shape[-2:]} expected={expected_lr_spatial}, "
            f"HR={hr_shape[-2:]} expected={expected_hr_spatial}"
        )
    hr_noise = prefix_scheduler.latents_list[1]
    requested = set(steps)
    max_step = max(steps)
    cached: dict[int, dict[str, Any]] = {}
    elapsed_completed_lr_steps = 0.0

    for step_number in range(1, max_step + 1):
        runner.check_stop()
        step_index = step_number - 1

        synchronize_device()
        infer_started = time.perf_counter()
        prefix_scheduler.step_pre(step_index=step_index)
        runner.model.infer(runner.inputs)
        synchronize_device()
        infer_seconds = time.perf_counter() - infer_started

        if step_number in requested:
            sample = prefix_scheduler.latents.detach().to(torch.float32)
            model_output = prefix_scheduler.noise_pred.detach().to(torch.float32)
            sigma = float(prefix_scheduler.sigmas[step_index])
            clean_pred = sample - sigma * model_output
            # Keep the in-memory branch state at the scheduler's float32
            # precision. LATENT_SAVE_DTYPE only controls the research archive;
            # it must not perturb generated oracle videos.
            x0_branch_cpu = clean_pred.to(device="cpu", dtype=torch.float32)
            x_t_cpu = sample.to(device="cpu", dtype=save_dtype(args.latent_save_dtype))
            x0_saved_cpu = clean_pred.to(
                device="cpu", dtype=save_dtype(args.latent_save_dtype)
            )
            prefix_seconds = elapsed_completed_lr_steps + infer_seconds
            latent_path = branch_latent_path(out_root, step_number, sample_id)
            latent_record = {
                "schema": "wan_taa_free_oracle_latent_v1",
                "sample_id": sample_id,
                "prompt": prompt,
                "seed": seed,
                "candidate_step": step_number,
                "step_index_zero_based": step_index,
                "infer_steps": args.infer_steps,
                "sigma": sigma,
                "taa_enabled": False,
                "x_t_lr": x_t_cpu,
                "x0_pred_lr": x0_saved_cpu,
                "x_t_stats": tensor_stats(x_t_cpu),
                "x0_pred_stats": tensor_stats(x0_saved_cpu),
            }
            if args.save_latents:
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent_record, latent_path)
            cached[step_number] = {
                "x0_pred_lr": x0_branch_cpu,
                "sigma": sigma,
                "prefix_seconds": prefix_seconds,
                "latent_path": str(latent_path) if args.save_latents else None,
                "x_t_stats": latent_record["x_t_stats"],
                "x0_pred_stats": latent_record["x0_pred_stats"],
            }

        if step_number < max_step:
            synchronize_device()
            post_started = time.perf_counter()
            # Bypass the changing-resolution mixin. This is the exact frozen LR
            # prefix shared by every branch; TAA/LoRA is absent by construction.
            WanScheduler.step_post(prefix_scheduler)
            synchronize_device()
            post_seconds = time.perf_counter() - post_started
            elapsed_completed_lr_steps += infer_seconds + post_seconds

    rows: list[dict[str, Any]] = []
    for step in steps:
        rows.append(
            run_one_cached_branch(
                runner,
                clean_resizer,
                args,
                cache=cached[step],
                step=step,
                seed=seed,
                sample_id=sample_id,
                out_root=out_root,
                hr_shape=hr_shape,
                hr_noise=hr_noise,
                input_encode_seconds=input_encode_seconds,
                scheduler_prepare_seconds=scheduler_prepare_seconds,
            )
        )

    return rows


@torch.no_grad()
def run_one_cached_branch(
    runner,
    clean_resizer,
    args,
    *,
    cache: dict[str, Any],
    step: int,
    seed: int,
    sample_id: str,
    out_root: Path,
    hr_shape: tuple[int, ...],
    hr_noise: torch.Tensor,
    input_encode_seconds: float,
    scheduler_prepare_seconds: float,
) -> dict[str, Any]:
    output = branch_video_path(out_root, step, sample_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    x0_pred_lr = cache["x0_pred_lr"].to(device=hr_noise.device, dtype=hr_noise.dtype)
    synchronize_device()

    branch_scheduler = WanScheduler(runner.config)
    branch_scheduler.prepare(seed=seed, latent_shape=hr_shape)
    branch_scheduler.set_timesteps(
        args.infer_steps,
        device=AI_DEVICE,
        shift=float(runner.config["sample_shift"]) + 1.0,
    )

    synchronize_device()
    lift_started = time.perf_counter()
    clean_hr = clean_resizer.resize(
        latent=x0_pred_lr,
        target_latent_shape=hr_shape,
        step_index=step - 1,
        changing_resolution_index=0,
    )
    if step < args.infer_steps:
        sigma = torch.as_tensor(
            cache["sigma"], device=clean_hr.device, dtype=clean_hr.dtype
        )
        branch_scheduler.latents = (1.0 - sigma) * clean_hr + sigma * hr_noise
    else:
        branch_scheduler.latents = clean_hr
    synchronize_device()
    lift_seconds = time.perf_counter() - lift_started

    runner.scheduler = branch_scheduler
    runner.model.set_scheduler(branch_scheduler)

    synchronize_device()
    suffix_started = time.perf_counter()
    for step_index in range(step, args.infer_steps):
        runner.check_stop()
        branch_scheduler.step_pre(step_index=step_index)
        runner.model.infer(runner.inputs)
        branch_scheduler.step_post()
    synchronize_device()
    hr_suffix_seconds = time.perf_counter() - suffix_started

    final_latents = branch_scheduler.latents
    synchronize_device()
    decode_started = time.perf_counter()
    decoded = runner.run_vae_decoder(final_latents)
    synchronize_device()
    vae_decode_seconds = time.perf_counter() - decode_started

    runner.gen_video = decoded
    runner.end_run_segment(0)
    runner.input_info.save_result_path = str(output)
    runner.input_info.return_result_tensor = False
    synchronize_device()
    save_started = time.perf_counter()
    runner.process_images_after_vae_decoder()
    synchronize_device()
    video_save_seconds = time.perf_counter() - save_started

    branch_compute_seconds = (
        lift_seconds + hr_suffix_seconds + vae_decode_seconds + video_save_seconds
    )
    estimated_warm_pipeline_seconds = (
        input_encode_seconds
        + scheduler_prepare_seconds
        + float(cache["prefix_seconds"])
        + branch_compute_seconds
    )
    del decoded, final_latents, clean_hr, x0_pred_lr
    runner.gen_video = None
    runner.gen_video_final = None
    empty_device_cache()

    return {
        "candidate_step": step,
        "lr_evaluations": step,
        "hr_evaluations": args.infer_steps - step,
        "sigma": cache["sigma"],
        "output": str(output),
        "latent_path": cache["latent_path"],
        "x_t_stats": cache["x_t_stats"],
        "x0_pred_stats": cache["x0_pred_stats"],
        "input_encode_seconds_shared": input_encode_seconds,
        "scheduler_prepare_seconds_shared": scheduler_prepare_seconds,
        "lr_prefix_seconds": cache["prefix_seconds"],
        "lift_and_renoise_seconds": lift_seconds,
        "hr_suffix_seconds": hr_suffix_seconds,
        "vae_decode_seconds": vae_decode_seconds,
        "video_save_seconds": video_save_seconds,
        "cached_branch_compute_seconds": branch_compute_seconds,
        "estimated_warm_pipeline_seconds": estimated_warm_pipeline_seconds,
    }


def run_independent_sample(
    runner,
    clean_resizer,
    args,
    *,
    steps: list[int],
    prompt: str,
    prompt_index: int,
    seed: int,
    sample_id: str,
    out_root: Path,
    model_init_seconds: float,
) -> None:
    manifest_path = out_root / "independent" / "manifests" / f"{sample_id}.json"
    previous = load_json_object(manifest_path)
    previous_branches = {
        int(row["candidate_step"]): row
        for row in previous.get("branches", [])
        if isinstance(row, dict) and "candidate_step" in row
    }
    rows = dict(previous_branches)

    logger.info(f"[oracle:independent] sample={sample_id} prompt={prompt}")
    for step in steps:
        output = independent_video_path(out_root, step, sample_id)
        complete = output.is_file() and output.stat().st_size > 0
        if args.skip_existing and complete and step in previous_branches:
            continue
        configure_bridge_scheduler(runner, clean_resizer, args, change_step=step)
        elapsed = run_standard_pipeline(
            runner, args, prompt=prompt, seed=seed, output=output
        )
        rows[step] = {
            "candidate_step": step,
            "lr_evaluations": step,
            "hr_evaluations": args.infer_steps - step,
            "warm_pipeline_seconds": elapsed,
            "output": str(output),
        }

    native_row = previous.get("native_hr")
    native_output = (
        out_root / "independent" / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4"
    )
    native_complete = native_output.is_file() and native_output.stat().st_size > 0
    if args.include_native_hr and not (
        args.skip_existing and native_complete and native_row
    ):
        native_row = run_native_hr_job(
            runner, args, prompt=prompt, seed=seed, output=native_output
        )

    manifest = {
        "schema": "wan_taa_free_oracle_v1",
        "execution_mode": "independent",
        "prompt_index": prompt_index,
        "prompt": prompt,
        "seed": seed,
        "candidate_steps": steps,
        "taa_enabled": False,
        "model_init_seconds_process_shared": model_init_seconds,
        "branches": [rows[step] for step in steps if step in rows],
        "native_hr": native_row,
    }
    write_json(manifest_path, manifest)


def run_native_hr_job(
    runner, args, *, prompt: str, seed: int, output: Path
) -> dict[str, Any]:
    configure_native_scheduler(runner, args)
    elapsed = run_standard_pipeline(
        runner, args, prompt=prompt, seed=seed, output=output
    )
    return {
        "lr_evaluations": 0,
        "hr_evaluations": args.infer_steps,
        "warm_pipeline_seconds": elapsed,
        "output": str(output),
    }


def run_standard_pipeline(
    runner, args, *, prompt: str, seed: int, output: Path
) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    input_info = make_input_info(args, prompt=prompt, seed=seed, output=output)
    seed_all(seed)
    synchronize_device()
    started = time.perf_counter()
    try:
        runner.run_pipeline(input_info)
    except BaseException:
        cleanup_runner_state(runner, suppress_errors=True)
        raise
    synchronize_device()
    return time.perf_counter() - started


def configure_bridge_scheduler(
    runner, clean_resizer, args, *, change_step: int
) -> None:
    runner.set_config(
        {
            "target_height": args.hr_height,
            "target_width": args.hr_width,
            "changing_resolution": True,
            "resolution_rate": [args.lr_height / args.hr_height],
            "wan_lowres_latent_size": [
                args.lr_latent_height,
                args.lr_latent_width,
            ],
            "changing_resolution_steps": [change_step],
            "infer_steps": args.infer_steps,
        }
    )
    scheduler = WanScheduler4CleanResizerBridgeInterface(WanScheduler, runner.config)
    scheduler.set_clean_latent_resizer(clean_resizer)
    runner.clean_latent_resizer = clean_resizer
    runner.scheduler = scheduler
    runner.model.set_scheduler(scheduler)


def configure_native_scheduler(runner, args) -> None:
    runner.set_config(
        {
            "target_height": args.hr_height,
            "target_width": args.hr_width,
            "changing_resolution": False,
            "infer_steps": args.infer_steps,
        }
    )
    scheduler = WanScheduler(runner.config)
    runner.scheduler = scheduler
    runner.model.set_scheduler(scheduler)


def make_input_info(args, *, prompt: str, seed: int, output: Path | None):
    input_info = init_empty_input_info(args.task, args.support_tasks)
    payload = vars(args).copy()
    payload.update(
        {
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "save_result_path": None if output is None else str(output),
            "return_result_tensor": False,
            "target_video_length": args.target_video_length,
        }
    )
    update_input_info_from_dict(input_info, payload)
    return input_info


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int] | str]:
    value = tensor.to(dtype=torch.float32)
    spatial_h = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
    spatial_w = (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
    if value.shape[-3] > 1:
        temporal = (value[..., 1:, :, :] - value[..., :-1, :, :]).abs().mean()
    else:
        temporal = torch.zeros((), dtype=value.dtype)
    return {
        "shape": list(value.shape),
        "stored_dtype": str(tensor.dtype).replace("torch.", ""),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "abs_mean": float(value.abs().mean()),
        "rms": float(torch.sqrt(torch.mean(value.square()))),
        "spatial_gradient_abs_mean": float((spatial_h + spatial_w) * 0.5),
        "temporal_gradient_abs_mean": float(temporal),
    }


def write_protocol(
    out_root: Path,
    args,
    config,
    steps: list[int],
    prompts: list[str],
    *,
    protocol_prompt_offset: int,
) -> None:
    prompt_payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    config_json_path = Path(args.config_json).resolve()
    stage2_checkpoint_path = Path(config["wan_clean_resizer_ckpt"]).resolve()
    stage2_checkpoint_stat = stage2_checkpoint_path.stat()
    stage2_train_config_path = Path(config["wan_clean_resizer_train_config"]).resolve()
    protocol = {
        "schema": "wan_taa_free_oracle_protocol_v1",
        "execution_mode": args.execution_mode,
        "candidate_steps": steps,
        "formal_candidate_steps": FORMAL_STEPS,
        "strict_protocol": args.strict_protocol,
        "infer_steps": args.infer_steps,
        "prompt_count": len(prompts),
        "selected_prompts_sha256": hashlib.sha256(
            prompt_payload.encode("utf-8")
        ).hexdigest(),
        "prompt_offset": protocol_prompt_offset,
        "start_seed": args.seed,
        "taa_enabled": False,
        "runtime_lora_allowed": False,
        "model_cls": config["model_cls"],
        "model_path": str(Path(config["model_path"]).resolve()),
        "config_json": str(config_json_path),
        "config_json_sha256": sha256_file(config_json_path),
        "feature_caching": config["feature_caching"],
        "sample_shift_lr": float(config["sample_shift"]),
        "sample_shift_hr": float(config["sample_shift"]) + 1.0,
        "sample_guide_scale": config["sample_guide_scale"],
        "enable_cfg": bool(config["enable_cfg"]),
        "target_video_length": int(config["target_video_length"]),
        "lr_rgb_size": [args.lr_height, args.lr_width],
        "hr_rgb_size": [args.hr_height, args.hr_width],
        "lr_latent_size": [args.lr_latent_height, args.lr_latent_width],
        "stage2_checkpoint": str(stage2_checkpoint_path),
        "stage2_checkpoint_size_bytes": stage2_checkpoint_stat.st_size,
        "stage2_checkpoint_mtime_ns": stage2_checkpoint_stat.st_mtime_ns,
        "stage2_train_config": str(stage2_train_config_path),
        "stage2_train_config_sha256": sha256_file(stage2_train_config_path),
        "include_native_hr": args.include_native_hr,
        "independent_warmup": args.independent_warmup,
        "save_latents": args.save_latents,
        "latent_save_dtype": args.latent_save_dtype,
    }
    protocol_root = (
        out_root if args.execution_mode == "branch" else out_root / "independent"
    )
    protocol_path = protocol_root / "protocol.json"
    previous = load_json_object(protocol_path)
    if args.skip_existing and previous and previous != protocol:
        changed = sorted(
            key
            for key in set(previous) | set(protocol)
            if previous.get(key) != protocol.get(key)
        )
        raise SystemExit(
            "Refusing to resume into an output root with a different protocol. "
            f"Changed fields: {changed}. Use a new OUT_ROOT or set SKIP_EXISTING=0 "
            "to intentionally replace the run."
        )
    write_json(protocol_path, protocol)


def write_prompt_map(
    out_root: Path,
    args,
    steps: list[int],
    prompts: list[str],
    *,
    prompt_offset: int,
) -> None:
    mapping: dict[str, str] = {}
    for local_index, prompt in enumerate(prompts):
        prompt_index = prompt_offset + local_index
        seed = args.seed + prompt_index
        sample_id = f"{prompt_index:04d}_seed{seed}"
        for step in steps:
            if args.execution_mode == "branch":
                path = branch_video_path(out_root, step, sample_id)
            else:
                path = independent_video_path(out_root, step, sample_id)
            mapping[str(path)] = prompt
        if args.include_native_hr:
            if args.execution_mode == "branch":
                native = (
                    out_root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4"
                )
            else:
                native = (
                    out_root
                    / "independent"
                    / "videos"
                    / "native_hr"
                    / f"{sample_id}_native_hr.mp4"
                )
            mapping[str(native)] = prompt
    write_json(out_root / f"prompt_map_{args.execution_mode}.json", mapping)


def validate_protocol(args, config, steps: list[int]) -> None:
    if args.execution_mode not in {"branch", "independent"}:
        raise SystemExit("--execution-mode must be branch or independent")
    if args.strict_protocol and steps != FORMAL_STEPS:
        raise SystemExit(
            f"Strict protocol requires candidate steps {FORMAL_STEPS}; got {steps}. "
            "Use --no-strict-protocol only for a smoke run."
        )
    if args.infer_steps != 50:
        raise SystemExit("This oracle protocol requires exactly 50 inference steps")
    if int(config.get("infer_steps", -1)) != args.infer_steps:
        raise SystemExit(
            "Resolved LightX2V config changed infer_steps unexpectedly: "
            f"config={config.get('infer_steps')} CLI={args.infer_steps}"
        )
    if int(config.get("target_video_length", -1)) != args.target_video_length:
        raise SystemExit(
            "Resolved LightX2V config changed target_video_length unexpectedly: "
            f"config={config.get('target_video_length')} CLI={args.target_video_length}"
        )
    source_config = load_json_object(Path(args.config_json))
    authoritative_keys = (
        "infer_steps",
        "target_video_length",
        "sample_guide_scale",
        "sample_shift",
        "enable_cfg",
        "feature_caching",
        "parallel",
        "wan_clean_resizer_ckpt",
        "wan_clean_resizer_train_config",
        "wan_clean_resizer_model_class",
        "wan_clean_resizer_use_ema",
    )
    missing = [key for key in authoritative_keys if key not in source_config]
    if missing:
        raise SystemExit(f"Oracle config JSON is missing required keys: {missing}")
    drifted = [
        key for key in authoritative_keys if config.get(key) != source_config.get(key)
    ]
    if drifted:
        raise SystemExit(
            "Model metadata overrode authoritative oracle settings from config_json: "
            f"{drifted}"
        )
    if args.task != "t2v":
        raise SystemExit("This oracle branch runner currently supports t2v only")
    if config["model_cls"] != "wan2.1_clean_resizer_bridge":
        raise SystemExit(
            "TAA-free oracle collection requires model_cls=wan2.1_clean_resizer_bridge"
        )
    if config.get("feature_caching") != "NoCaching":
        raise SystemExit(
            "Oracle branches require feature_caching=NoCaching so no cache state leaks across branches"
        )
    if config.get("parallel"):
        raise SystemExit("Oracle branch collection is currently single-device only")
    forbidden = {
        "lora_configs": config.get("lora_configs"),
        "lora_active_steps": config.get("lora_active_steps"),
        "lora_dynamic_apply": config.get("lora_dynamic_apply"),
    }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise SystemExit(
            f"TAA/LoRA must be disabled; forbidden config keys enabled: {enabled}"
        )
    if args.lr_height != 368 or args.lr_width != 640:
        raise SystemExit("Formal oracle collection requires LR RGB size 368x640")
    if args.hr_height != 720 or args.hr_width != 1248:
        raise SystemExit("Formal oracle collection requires HR RGB size 720x1248")
    if args.lr_latent_height != 46 or args.lr_latent_width != 80:
        raise SystemExit("Formal oracle collection requires LR latent size 46x80")


def prepare_output_tree(out_root: Path, steps: list[int], execution_mode: str) -> None:
    if execution_mode == "branch":
        base = out_root
    else:
        base = out_root / "independent"
    (base / "manifests").mkdir(parents=True, exist_ok=True)
    (base / "videos" / "native_hr").mkdir(parents=True, exist_ok=True)
    for step in steps:
        (base / "videos" / f"step{step:02d}").mkdir(parents=True, exist_ok=True)
        if execution_mode == "branch":
            (base / "latents" / f"step{step:02d}").mkdir(parents=True, exist_ok=True)


def branch_video_path(out_root: Path, step: int, sample_id: str) -> Path:
    return out_root / "videos" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.mp4"


def independent_video_path(out_root: Path, step: int, sample_id: str) -> Path:
    return (
        out_root
        / "independent"
        / "videos"
        / f"step{step:02d}"
        / f"{sample_id}_step{step:02d}.mp4"
    )


def branch_latent_path(out_root: Path, step: int, sample_id: str) -> Path:
    return out_root / "latents" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.pt"


def save_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported latent save dtype: {name}")


def synchronize_device() -> None:
    device_module = getattr(torch, AI_DEVICE, None)
    if device_module is not None and hasattr(device_module, "synchronize"):
        device_module.synchronize()


def empty_device_cache() -> None:
    device_module = getattr(torch, AI_DEVICE, None)
    if device_module is not None and hasattr(device_module, "empty_cache"):
        device_module.empty_cache()


@contextmanager
def managed_runner_run(runner):
    """Close a manually driven runner on both success and failure."""

    try:
        yield
    except BaseException:
        cleanup_runner_state(runner, suppress_errors=True)
        raise
    else:
        cleanup_runner_state(runner, suppress_errors=False)


def cleanup_runner_state(runner, *, suppress_errors: bool) -> None:
    try:
        runner.end_run()
    except Exception:
        if not suppress_errors:
            raise
        logger.exception("Runner cleanup failed while preserving the primary exception")
    finally:
        runner.gen_video = None
        runner.gen_video_final = None
        empty_device_cache()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        preserve_primary_exception = sys.exc_info()[0] is not None
        try:
            dist.destroy_process_group()
            logger.info("Distributed process group cleaned up")
        except Exception:
            if not preserve_primary_exception:
                raise
            logger.exception(
                "Distributed cleanup failed while preserving the primary exception"
            )


def load_prompts(path: Path, *, offset: int, limit: int) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return prompts[offset : offset + limit] if limit > 0 else prompts[offset:]


def parse_steps(value: str, *, infer_steps: int) -> list[int]:
    normalized = value.replace(",", " ")
    steps = [int(item) for item in normalized.split() if item]
    if not steps:
        raise SystemExit("No candidate steps provided")
    if steps != sorted(set(steps)):
        raise SystemExit("Candidate steps must be unique and strictly increasing")
    invalid = [step for step in steps if step < 1 or step > infer_steps]
    if invalid:
        raise SystemExit(
            f"Invalid candidate steps {invalid}; expected integers in [1, {infer_steps}]"
        )
    return steps


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a TAA-free Wan handoff oracle by sharing one LR prefix across "
            "steps 30, 35, and every step from 40 through 50."
        )
    )
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--model_cls", default="wan2.1_clean_resizer_bridge")
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
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--protocol-prompt-offset",
        type=int,
        default=None,
        help=(
            "Canonical prompt-slice offset recorded in protocol.json. Use with "
            "--protocol-prompt-limit when parallel workers execute disjoint subsets "
            "of one existing output root."
        ),
    )
    parser.add_argument(
        "--protocol-prompt-limit",
        type=int,
        default=None,
        help="Canonical prompt-slice size recorded in protocol.json.",
    )
    parser.add_argument("--change-steps", required=True)
    parser.add_argument("--infer-steps", type=int, default=50)
    parser.add_argument("--lr-height", type=int, default=368)
    parser.add_argument("--lr-width", type=int, default=640)
    parser.add_argument("--hr-height", type=int, default=720)
    parser.add_argument("--hr-width", type=int, default=1248)
    parser.add_argument("--lr-latent-height", type=int, default=46)
    parser.add_argument("--lr-latent-width", type=int, default=80)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--execution-mode", choices=("branch", "independent"), default="branch"
    )
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument(
        "--latent-save-dtype", choices=("fp16", "bf16", "fp32"), default="fp16"
    )
    parser.add_argument("--include-native-hr", action="store_true")
    parser.add_argument("--independent-warmup", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--strict-protocol", dest="strict_protocol", action="store_true", default=True
    )
    parser.add_argument(
        "--no-strict-protocol", dest="strict_protocol", action="store_false"
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
