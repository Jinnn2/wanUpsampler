from __future__ import annotations

# ruff: noqa: E402 -- support both ``python -m`` and direct script execution.

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.checkpoint import build_universal_upsampler_from_payload
from changing_resolution_uni.data import UniversalCleanLatentDataset
from changing_resolution_uni.evaluation.aggregation import (
    deduplicate_sample_rows,
    read_jsonl,
    write_summary_files,
)
from changing_resolution_uni.evaluation.baselines import (
    load_specialist,
    make_interpolation_runner,
)
from changing_resolution_uni.evaluation.latent_metrics import compute_latent_metrics
from changing_resolution_uni.evaluation.protocol import (
    canonical_scale,
    checkpoint_split_config,
    environment_record,
    load_or_create_manifest,
    select_manifest_sources,
    sha256_file,
    virtual_index,
)
from changing_resolution_uni.evaluation.rgb_metrics import (
    VideoRGBMetrics,
    save_keyframes,
)
from changing_resolution_uni.evaluation.visualization import make_comparison_panel


MethodRunner = Callable[[torch.Tensor, tuple[int, int]], torch.Tensor]
VALID_METHODS = ("raw", "ema", "nearest", "trilinear", "bicubic", "specialist")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed(cpu=args.cpu)
    device = choose_device(args, local_rank)
    checkpoints = resolve_checkpoints(args)
    checkpoint_payloads = load_checkpoint_metadata_distributed(
        checkpoints,
        rank=rank,
        world_size=world_size,
    )
    assert_split_compatible(checkpoint_payloads)
    split_config = checkpoint_split_config(checkpoint_payloads[0])
    requested_split_override = any(
        value is not None for value in (args.val_ratio, args.val_max_samples, args.seed)
    )
    if requested_split_override and not args.allow_split_override:
        raise ValueError(
            "Split overrides are disabled by default because evaluation must reproduce training. "
            "Pass --allow_split_override only for an explicitly separate protocol."
        )
    if args.val_ratio is not None:
        split_config["val_ratio"] = args.val_ratio
    if args.val_max_samples is not None:
        split_config["val_max_samples"] = args.val_max_samples
    if args.seed is not None:
        split_config["seed"] = args.seed

    data_dir = Path(
        args.data_dir or checkpoint_payloads[0].get("config", {}).get("data_dir", "")
    )
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Evaluation LMDB directory not found: {data_dir}")
    dataset = UniversalCleanLatentDataset(data_dir, seed=int(split_config["seed"]))
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else out_dir / f"manifest_{args.split}.json"
    )
    if rank == 0:
        manifest = load_or_create_manifest(
            manifest_path,
            dataset,
            data_dir=data_dir,
            split=args.split,
            split_config=split_config,
            allow_dataset_mismatch=args.allow_dataset_mismatch,
        )
        if not manifest["sources"]:
            raise RuntimeError(f"The selected {args.split!r} split is empty")
    barrier(world_size)
    manifest = load_or_create_manifest(
        manifest_path,
        dataset,
        data_dir=data_dir,
        split=args.split,
        split_config=split_config,
        allow_dataset_mismatch=args.allow_dataset_mismatch,
    )

    checkpoint_hashes = compute_checkpoint_hashes(
        checkpoints, rank=rank, world_size=world_size
    )
    if rank == 0:
        environment = environment_record(
            checkpoints=checkpoints,
            data_dir=data_dir,
            manifest=manifest,
            checkpoint_hashes=checkpoint_hashes,
        )
        environment.update(
            {
                "rank_world_size": world_size,
                "precision": args.precision,
                "methods": args.methods,
                "split_config": split_config,
            }
        )
        environment_path = out_dir / f"environment_{args.mode}.json"
        environment_path.write_text(
            json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        common_environment_path = out_dir / "environment.json"
        if not common_environment_path.exists():
            common_environment_path.write_text(
                json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    barrier(world_size)

    if args.mode in {"latent", "sweep", "all"}:
        evaluate_latent(
            args,
            dataset=dataset,
            manifest=manifest,
            checkpoints=checkpoints,
            checkpoint_payloads=checkpoint_payloads,
            checkpoint_hashes=checkpoint_hashes,
            out_dir=out_dir,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    if args.mode in {"rgb", "all"}:
        evaluate_rgb(
            args,
            dataset=dataset,
            manifest=manifest,
            checkpoints=checkpoints,
            checkpoint_payloads=checkpoint_payloads,
            checkpoint_hashes=checkpoint_hashes,
            out_dir=out_dir,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    if args.mode in {"timing", "all"}:
        if world_size == 1:
            evaluate_timing(
                args,
                dataset=dataset,
                manifest=manifest,
                checkpoints=checkpoints,
                checkpoint_payloads=checkpoint_payloads,
                checkpoint_hashes=checkpoint_hashes,
                out_dir=out_dir,
                device=device,
            )
        elif rank == 0:
            print(
                "Skipping timing under multi-GPU execution; run MODE=timing on one isolated GPU.",
                flush=True,
            )

    barrier(world_size)
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def evaluate_latent(
    args: argparse.Namespace,
    *,
    dataset: UniversalCleanLatentDataset,
    manifest: dict[str, Any],
    checkpoints: list[Path],
    checkpoint_payloads: list[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
    out_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    global_records = select_manifest_sources(
        manifest,
        offset=args.source_offset,
        max_sources=args.max_sources,
        rank=0,
        world_size=1,
    )
    if not global_records:
        raise RuntimeError("No manifest sources selected for latent evaluation")
    records = [
        record
        for index, record in enumerate(global_records)
        if index % world_size == rank
    ]
    rank_path = out_dir / f"latent_samples.rank{rank:03d}.jsonl"
    completed = load_completed_keys(rank_path) if args.resume else set()
    mode = "a" if args.resume and rank_path.exists() else "w"
    with rank_path.open(mode, encoding="utf-8") as handle:
        for checkpoint, payload in zip(checkpoints, checkpoint_payloads):
            checkpoint_label = make_checkpoint_label(checkpoint, payload)
            checkpoint_step = int(payload.get("step", 0))
            universal_payload = (
                load_inference_payload(checkpoint)
                if {"raw", "ema"} & set(args.methods)
                else None
            )
            epsilon = (
                args.charbonnier_eps
                if args.charbonnier_eps is not None
                else float(
                    payload.get("config", {})
                    .get("loss", {})
                    .get("charbonnier_eps", 1e-3)
                )
            )
            for method in args.methods:
                runner, resource, weights = load_method_runner(
                    method,
                    checkpoint=checkpoint,
                    args=args,
                    device=device,
                    universal_payload=universal_payload,
                )
                try:
                    total = len(records) * len(dataset.scales)
                    done = 0
                    for source in records:
                        for scale in dataset.scales:
                            done += 1
                            key = (
                                checkpoint_label,
                                method,
                                args.precision,
                                source["source_uid"],
                                canonical_scale(scale),
                            )
                            if key in completed:
                                continue
                            sample = dataset[virtual_index(dataset, source, scale)]
                            row = base_sample_row(
                                kind="latent",
                                checkpoint=checkpoint,
                                checkpoint_label=checkpoint_label,
                                checkpoint_step=checkpoint_step,
                                checkpoint_sha256=checkpoint_hashes[str(checkpoint)],
                                method=method,
                                weights=weights,
                                variant=args.precision,
                                source=source,
                                scale=scale,
                                sample=sample,
                            )
                            try:
                                lr = (
                                    sample["z0_lr"]
                                    .unsqueeze(0)
                                    .to(device, non_blocking=True)
                                )
                                hr = (
                                    sample["z0_hr"]
                                    .unsqueeze(0)
                                    .to(device, non_blocking=True)
                                )
                                with inference_context(device, args.precision):
                                    prediction = runner(
                                        lr, tuple(int(value) for value in hr.shape[-2:])
                                    )
                                row.update(
                                    compute_latent_metrics(
                                        prediction.float(),
                                        hr.float(),
                                        lr.float(),
                                        charbonnier_eps=epsilon,
                                    )
                                )
                                row["status"] = "ok"
                            except Exception as exc:
                                if (
                                    method != "specialist"
                                    or not args.skip_incompatible_specialist
                                    or not isinstance(exc, ValueError)
                                ):
                                    raise
                                row.update({"status": "unsupported", "error": str(exc)})
                            write_jsonl_row(handle, row)
                            print(
                                f"[rank {rank}] latent {checkpoint_label} {method} "
                                f"{done}/{total} source={source['source_uid']} scale={scale} "
                                f"status={row['status']}",
                                flush=True,
                            )
                finally:
                    del runner, resource
                    release_device(device)
            del universal_payload
            release_device(device)
    barrier(world_size)
    if rank == 0:
        rank_paths = [
            out_dir / f"latent_samples.rank{item:03d}.jsonl"
            for item in range(world_size)
        ]
        rows = filter_expected_rows(
            read_jsonl(rank_paths),
            checkpoints=checkpoints,
            checkpoint_payloads=checkpoint_payloads,
            methods=args.methods,
            variant=args.precision,
            sources=global_records,
            scales=dataset.scales,
        )
        validate_coverage(
            rows,
            checkpoints=checkpoints,
            checkpoint_payloads=checkpoint_payloads,
            methods=args.methods,
            variant=args.precision,
            sources=global_records,
            scales=dataset.scales,
            path=out_dir / "latent_coverage.json",
        )
        merged = out_dir / "latent_samples.jsonl"
        write_merged_jsonl(merged, rows)
        paths = write_summary_files(
            rows,
            out_dir,
            stem="latent",
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_checkpoint_sweep(out_dir / "checkpoint_sweep.csv", paths[0])
        print(f"Latent evaluation ready: {paths[0]}", flush=True)
    barrier(world_size)


def evaluate_rgb(
    args: argparse.Namespace,
    *,
    dataset: UniversalCleanLatentDataset,
    manifest: dict[str, Any],
    checkpoints: list[Path],
    checkpoint_payloads: list[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
    out_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    require_rgb_paths(args)
    limits = [
        value for value in (args.max_sources, args.decode_max_sources) if value > 0
    ]
    decode_limit = min(limits) if limits else 0
    global_records = select_manifest_sources(
        manifest,
        offset=args.source_offset,
        max_sources=decode_limit,
        rank=0,
        world_size=1,
    )
    if not global_records:
        raise RuntimeError("No manifest sources selected for RGB evaluation")
    visual_uids = {
        record["source_uid"] for record in global_records[: args.visual_max_sources]
    }
    records = [
        record
        for index, record in enumerate(global_records)
        if index % world_size == rank
    ]
    dtype = precision_dtype(args.precision)
    from wan_sr.vae import WanVAEWrapper

    vae = WanVAEWrapper(
        args.model_root,
        vae_path=args.vae_path,
        wan_repo=args.wan_repo,
        backend=args.vae_backend,
        device=device,
        dtype=dtype,
    )
    rgb_variant = f"{args.precision}:{vae.backend_kind}"
    # Rank 0 initialises first so that any remote model-weight downloads
    # (e.g. the LPIPS VGG/AlexNet checkpoint ~233 MB) happen exactly once.
    # The other ranks wait at the barrier and then load from the local cache,
    # avoiding a multi-rank download race that causes NCCL watchdog timeouts.
    if rank == 0:
        similarity = VideoRGBMetrics(args.rgb_metrics, device=device)
    barrier(world_size)
    if rank != 0:
        similarity = VideoRGBMetrics(args.rgb_metrics, device=device)
    rank_path = out_dir / f"rgb_samples.rank{rank:03d}.jsonl"
    completed = load_completed_keys(rank_path) if args.resume else set()
    mode = "a" if args.resume and rank_path.exists() else "w"
    with rank_path.open(mode, encoding="utf-8") as handle:
        for checkpoint, payload in zip(checkpoints, checkpoint_payloads):
            checkpoint_label = make_checkpoint_label(checkpoint, payload)
            checkpoint_step = int(payload.get("step", 0))
            universal_payload = (
                load_inference_payload(checkpoint)
                if {"raw", "ema"} & set(args.methods)
                else None
            )
            method_runners = []
            try:
                for method in args.methods:
                    runner, resource, weights = load_method_runner(
                        method,
                        checkpoint=checkpoint,
                        args=args,
                        device=device,
                        universal_payload=universal_payload,
                    )
                    method_runners.append((method, runner, resource, weights))
                for source in records:
                    source_pending = any(
                        (
                            checkpoint_label,
                            method,
                            rgb_variant,
                            source["source_uid"],
                            canonical_scale(scale),
                        )
                        not in completed
                        for method in args.methods
                        for scale in dataset.scales
                    )
                    if not source_pending:
                        continue
                    reference_sample = dataset[
                        virtual_index(dataset, source, dataset.scales[0])
                    ]
                    target_video = vae.decode(reference_sample["z0_hr"].unsqueeze(0))[0]
                    for scale in dataset.scales:
                        sample = dataset[virtual_index(dataset, source, scale)]
                        lr_cpu = sample["z0_lr"].unsqueeze(0)
                        hr_cpu = sample["z0_hr"].unsqueeze(0)
                        lr = lr_cpu.to(device, non_blocking=True)
                        target_size = tuple(int(value) for value in hr_cpu.shape[-2:])
                        for method, runner, _, weights in method_runners:
                            key = (
                                checkpoint_label,
                                method,
                                rgb_variant,
                                source["source_uid"],
                                canonical_scale(scale),
                            )
                            if key in completed:
                                continue
                            sample = dataset[virtual_index(dataset, source, scale)]
                            row = base_sample_row(
                                kind="rgb",
                                checkpoint=checkpoint,
                                checkpoint_label=checkpoint_label,
                                checkpoint_step=checkpoint_step,
                                checkpoint_sha256=checkpoint_hashes[str(checkpoint)],
                                method=method,
                                weights=weights,
                                variant=rgb_variant,
                                source=source,
                                scale=scale,
                                sample=sample,
                            )
                            row["vae_backend"] = vae.backend_kind
                            try:
                                with inference_context(device, args.precision):
                                    prediction = runner(lr, target_size)
                                prediction_cpu = prediction.float().cpu()
                                prediction_video = vae.decode(prediction_cpu)[0]
                                row.update(
                                    similarity.compute(
                                        prediction_video,
                                        target_video,
                                        batch_size=args.metric_batch_size,
                                    )
                                )
                                row["status"] = "ok"
                                if (
                                    args.save_visuals
                                    and source["source_uid"] in visual_uids
                                ):
                                    row["paths"] = save_visual_outputs(
                                        out_dir=out_dir,
                                        checkpoint_label=checkpoint_label,
                                        source_uid=source["source_uid"],
                                        scale=canonical_scale(scale),
                                        method=method,
                                        lr_latent=lr_cpu,
                                        target_video=target_video,
                                        prediction_video=prediction_video,
                                        vae=vae,
                                        fps=args.fps,
                                    )
                            except Exception as exc:
                                if (
                                    method != "specialist"
                                    or not args.skip_incompatible_specialist
                                    or not isinstance(exc, ValueError)
                                ):
                                    raise
                                row.update({"status": "unsupported", "error": str(exc)})
                            write_jsonl_row(handle, row)
                            print(
                                f"[rank {rank}] rgb {checkpoint_label} {method} "
                                f"source={source['source_uid']} scale={scale} status={row['status']}",
                                flush=True,
                            )
            finally:
                for _, runner, resource, _ in method_runners:
                    del runner, resource
                del method_runners
                release_device(device)
            if args.save_visuals:
                build_visual_panels(
                    args,
                    out_dir=out_dir,
                    checkpoint_label=checkpoint_label,
                    records=records,
                    scales=dataset.scales,
                    visual_uids=visual_uids,
                )
            del universal_payload
            release_device(device)
    del similarity, vae
    release_device(device)
    barrier(world_size)
    if rank == 0:
        rank_paths = [
            out_dir / f"rgb_samples.rank{item:03d}.jsonl" for item in range(world_size)
        ]
        rows = filter_expected_rows(
            read_jsonl(rank_paths),
            checkpoints=checkpoints,
            checkpoint_payloads=checkpoint_payloads,
            methods=args.methods,
            variant=rgb_variant,
            sources=global_records,
            scales=dataset.scales,
        )
        validate_coverage(
            rows,
            checkpoints=checkpoints,
            checkpoint_payloads=checkpoint_payloads,
            methods=args.methods,
            variant=rgb_variant,
            sources=global_records,
            scales=dataset.scales,
            path=out_dir / "rgb_coverage.json",
        )
        write_merged_jsonl(out_dir / "rgb_samples.jsonl", rows)
        paths = write_summary_files(
            rows,
            out_dir,
            stem="rgb",
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(f"RGB evaluation ready: {paths[0]}", flush=True)
    barrier(world_size)


def evaluate_timing(
    args: argparse.Namespace,
    *,
    dataset: UniversalCleanLatentDataset,
    manifest: dict[str, Any],
    checkpoints: list[Path],
    checkpoint_payloads: list[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
    out_dir: Path,
    device: torch.device,
) -> None:
    records = select_manifest_sources(
        manifest,
        offset=args.source_offset,
        max_sources=1,
        rank=0,
        world_size=1,
    )
    if not records:
        raise RuntimeError("No source is available for timing")
    source = records[0]
    rows = []
    for checkpoint, payload in zip(checkpoints, checkpoint_payloads):
        checkpoint_label = make_checkpoint_label(checkpoint, payload)
        universal_payload = (
            load_inference_payload(checkpoint)
            if {"raw", "ema"} & set(args.methods)
            else None
        )
        for method in args.methods:
            runner, resource, weights = load_method_runner(
                method,
                checkpoint=checkpoint,
                args=args,
                device=device,
                universal_payload=universal_payload,
            )
            try:
                for scale in dataset.scales:
                    sample = dataset[virtual_index(dataset, source, scale)]
                    lr = sample["z0_lr"].unsqueeze(0).to(device)
                    target_size = tuple(
                        int(value) for value in sample["z0_hr"].shape[-2:]
                    )
                    row = {
                        "checkpoint": checkpoint_label,
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_step": int(payload.get("step", 0)),
                        "checkpoint_sha256": checkpoint_hashes[str(checkpoint)],
                        "method": method,
                        "weights": weights,
                        "variant": args.precision,
                        "precision": args.precision,
                        "scale": canonical_scale(scale),
                        "source_size": list(lr.shape[-2:]),
                        "target_size": list(target_size),
                        "batch_size": 1,
                        "warmup": args.timing_warmup,
                        "repeats": args.timing_repeats,
                    }
                    try:
                        for _ in range(args.timing_warmup):
                            with inference_context(device, args.precision):
                                output = runner(lr, target_size)
                            del output
                        synchronize(device)
                        base_memory = (
                            torch.cuda.memory_allocated(device)
                            if device.type == "cuda"
                            else 0
                        )
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        timings = []
                        for _ in range(args.timing_repeats):
                            synchronize(device)
                            start = time.perf_counter()
                            with inference_context(device, args.precision):
                                output = runner(lr, target_size)
                            synchronize(device)
                            timings.append((time.perf_counter() - start) * 1000.0)
                            del output
                        values = np.asarray(timings, dtype=np.float64)
                        row.update(
                            {
                                "status": "ok",
                                "latency_mean_ms": float(values.mean()),
                                "latency_std_ms": float(values.std(ddof=1))
                                if values.size > 1
                                else 0.0,
                                "latency_p50_ms": float(np.quantile(values, 0.5)),
                                "latency_p90_ms": float(np.quantile(values, 0.9)),
                            }
                        )
                        if device.type == "cuda":
                            peak = torch.cuda.max_memory_allocated(device)
                            row["peak_allocated_mb"] = peak / 1024**2
                            row["incremental_peak_mb"] = (
                                max(0, peak - base_memory) / 1024**2
                            )
                    except Exception as exc:
                        if (
                            method != "specialist"
                            or not args.skip_incompatible_specialist
                            or not isinstance(exc, ValueError)
                        ):
                            raise
                        row.update({"status": "unsupported", "error": str(exc)})
                    rows.append(row)
            finally:
                del runner, resource
                release_device(device)
        del universal_payload
        release_device(device)
    (out_dir / "timing.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "timing.csv", rows)
    print(f"Timing evaluation ready: {out_dir / 'timing.json'}", flush=True)


def load_method_runner(
    method: str,
    *,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    universal_payload: dict[str, Any] | None = None,
) -> tuple[MethodRunner, Any, str]:
    if method in {"nearest", "trilinear", "bicubic"}:
        return make_interpolation_runner(method), None, "none"
    if method in {"raw", "ema"}:
        if universal_payload is None:
            raise ValueError(
                "Raw/EMA evaluation requires a loaded universal checkpoint payload"
            )
        model = build_universal_upsampler_from_payload(
            universal_payload,
            device=device,
            use_ema=method == "ema",
        )
        if method == "ema" and "ema" not in universal_payload:
            raise ValueError(f"Checkpoint has no EMA state: {checkpoint}")
        return (
            lambda latent, output_size: model(latent, output_size=output_size),
            model,
            method,
        )
    if method == "specialist":
        if not args.specialist_checkpoint:
            raise ValueError("Method 'specialist' requires --specialist_checkpoint")
        model, _ = load_specialist(
            args.specialist_checkpoint,
            device=device,
            train_config_path=args.specialist_config,
            use_ema=args.specialist_use_ema,
        )
        weights = "ema" if args.specialist_use_ema else "raw"
        return (
            lambda latent, output_size: model(latent, output_size=output_size),
            model,
            weights,
        )
    raise ValueError(f"Unsupported method: {method}")


def base_sample_row(
    *,
    kind: str,
    checkpoint: Path,
    checkpoint_label: str,
    checkpoint_step: int,
    checkpoint_sha256: str,
    method: str,
    weights: str,
    variant: str,
    source: dict[str, Any],
    scale: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    source_size = [int(value) for value in sample["z0_lr"].shape[-2:]]
    target_size = [int(value) for value in sample["z0_hr"].shape[-2:]]
    return {
        "kind": kind,
        "checkpoint": checkpoint_label,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "method": method,
        "weights": weights,
        "variant": variant,
        "precision": variant.split(":", 1)[0],
        "source_uid": source["source_uid"],
        "source_index": int(source["source_index"]),
        "shard": source["shard"],
        "row_id": int(source["row_id"]),
        "scale": canonical_scale(scale),
        "source_size": source_size,
        "target_size": target_size,
        "actual_scale_hw": [
            target_size[0] / source_size[0],
            target_size[1] / source_size[1],
        ],
        "frames": int(sample["z0_hr"].shape[1]),
        "grid_unit": "latent",
    }


def save_visual_outputs(
    *,
    out_dir: Path,
    checkpoint_label: str,
    source_uid: str,
    scale: str,
    method: str,
    lr_latent: torch.Tensor,
    target_video: torch.Tensor,
    prediction_video: torch.Tensor,
    vae: Any,
    fps: int,
) -> dict[str, Any]:
    from wan_sr.data.video_io import write_video

    root = out_dir / "visuals" / safe_name(checkpoint_label)
    video_dir = root / "videos"
    frame_dir = root / "keyframes"
    video_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{source_uid}_scale{safe_name(scale)}"
    gt_path = video_dir / f"{prefix}_gt.mp4"
    lr_path = video_dir / f"{prefix}_lr.mp4"
    prediction_path = video_dir / f"{prefix}_{safe_name(method)}.mp4"
    if not gt_path.exists():
        write_video(gt_path, target_video, fps=fps)
    if not lr_path.exists():
        lr_video = vae.decode(lr_latent)[0]
        write_video(lr_path, lr_video, fps=fps)
        save_keyframes(lr_video, frame_dir, f"{prefix}_lr")
    write_video(prediction_path, prediction_video, fps=fps)
    return {
        "lr_video": str(lr_path),
        "gt_video": str(gt_path),
        "prediction_video": str(prediction_path),
        "prediction_keyframes": save_keyframes(
            prediction_video,
            frame_dir,
            f"{prefix}_{safe_name(method)}",
        ),
        "gt_keyframes": save_keyframes(target_video, frame_dir, f"{prefix}_gt"),
    }


def build_visual_panels(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    checkpoint_label: str,
    records: list[dict[str, Any]],
    scales: tuple[str, ...],
    visual_uids: set[str],
) -> None:
    root = out_dir / "visuals" / safe_name(checkpoint_label)
    video_dir = root / "videos"
    compare_dir = root / "compare"
    for source in records:
        if source["source_uid"] not in visual_uids:
            continue
        for scale in scales:
            prefix = f"{source['source_uid']}_scale{safe_name(scale)}"
            labeled = []
            for filename, label in (
                (f"{prefix}_lr.mp4", "LR decode"),
                (f"{prefix}_gt.mp4", "GT HR decode"),
            ):
                path = video_dir / filename
                if path.exists():
                    labeled.append((path, label))
            for method in args.methods:
                path = video_dir / f"{prefix}_{safe_name(method)}.mp4"
                if path.exists():
                    labeled.append((path, method))
            if len(labeled) >= 2:
                make_comparison_panel(
                    labeled,
                    compare_dir / f"{prefix}_compare.mp4",
                    width=args.panel_width,
                    height=args.panel_height,
                    fps=args.fps,
                )


def inference_context(device: torch.device, precision: str):
    autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = precision_dtype(precision)
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
        if autocast
        else nullcontext()
    )
    return _CombinedContext(torch.inference_mode(), autocast_context)


class _CombinedContext:
    def __init__(self, *contexts: Any) -> None:
        self.contexts = contexts

    def __enter__(self) -> "_CombinedContext":
        for context in self.contexts:
            context.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        suppressed = False
        for context in reversed(self.contexts):
            suppressed = context.__exit__(exc_type, exc, traceback) or suppressed
        return suppressed


def precision_dtype(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        precision
    ]


def choose_device(args: argparse.Namespace, local_rank: int) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        if args.precision != "fp32":
            print(
                "CUDA unavailable or --cpu selected; forcing fp32 evaluation.",
                flush=True,
            )
            args.precision = "fp32"
        return torch.device("cpu")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def init_distributed(*, cpu: bool = False) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() and not cpu
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo")
    return rank, world_size, local_rank


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in (args.checkpoint or [])]
    if args.checkpoint_dir:
        paths.extend(
            sorted(
                Path(args.checkpoint_dir).glob(args.checkpoint_glob),
                key=checkpoint_path_sort_key,
            )
        )
        if args.include_last and (Path(args.checkpoint_dir) / "last.pt").is_file():
            paths.append(Path(args.checkpoint_dir) / "last.pt")
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("Pass --checkpoint and/or --checkpoint_dir")
    return unique


def checkpoint_path_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"step_(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**18, path.name)


def assert_split_compatible(payloads: list[dict[str, Any]]) -> None:
    expected = checkpoint_split_config(payloads[0])
    for payload in payloads[1:]:
        current = checkpoint_split_config(payload)
        if current != expected:
            raise ValueError(
                f"Checkpoint sweep mixes validation splits: expected {expected}, found {current}"
            )


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    payload = _torch_load(path, map_location="cpu")
    metadata = {
        "step": int(payload.get("step", 0)),
        "config": payload.get("config", {}),
        "has_ema": isinstance(payload.get("ema"), dict),
    }
    del payload
    gc.collect()
    return metadata


def load_checkpoint_metadata_distributed(
    checkpoints: list[Path],
    *,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    metadata = (
        [load_checkpoint_metadata(path) for path in checkpoints] if rank == 0 else []
    )
    if world_size > 1:
        objects: list[Any] = [metadata]
        dist.broadcast_object_list(objects, src=0)
        metadata = objects[0]
    return metadata


def load_inference_payload(path: str | Path) -> dict[str, Any]:
    payload = _torch_load(path, map_location="cpu")
    if "model" not in payload:
        raise ValueError(
            f"Evaluation requires a full U-ITU training checkpoint: {path}"
        )
    # Optimizer moments are irrelevant to inference and can be several times
    # larger than model+EMA weights. Drop them before evaluating both weights.
    return {
        key: payload[key]
        for key in ("step", "config", "model", "ema", "model_config")
        if key in payload
    }


def compute_checkpoint_hashes(
    checkpoints: list[Path],
    *,
    rank: int,
    world_size: int,
) -> dict[str, str]:
    values = {str(path): sha256_file(path) for path in checkpoints} if rank == 0 else {}
    if world_size > 1:
        objects: list[Any] = [values]
        dist.broadcast_object_list(objects, src=0)
        values = objects[0]
    return values


def make_checkpoint_label(path: Path, payload: dict[str, Any]) -> str:
    return f"{path.stem}@{int(payload.get('step', 0))}"


def load_completed_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    keys = set()
    for row in read_jsonl([path]):
        keys.add(
            (
                str(row["checkpoint"]),
                str(row["method"]),
                str(row.get("variant", row.get("precision", "default"))),
                str(row["source_uid"]),
                canonical_scale(row["scale"]),
            )
        )
    return keys


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def filter_expected_rows(
    rows: list[dict[str, Any]],
    *,
    checkpoints: list[Path],
    checkpoint_payloads: list[dict[str, Any]],
    methods: list[str],
    variant: str,
    sources: list[dict[str, Any]],
    scales: tuple[str, ...],
) -> list[dict[str, Any]]:
    checkpoint_labels = {
        make_checkpoint_label(path, payload)
        for path, payload in zip(checkpoints, checkpoint_payloads)
    }
    source_uids = {str(source["source_uid"]) for source in sources}
    scale_names = {canonical_scale(scale) for scale in scales}
    return deduplicate_sample_rows(
        row
        for row in rows
        if str(row.get("checkpoint")) in checkpoint_labels
        and str(row.get("method")) in methods
        and str(row.get("variant", "default")) == variant
        and str(row.get("source_uid")) in source_uids
        and canonical_scale(row.get("scale", 0)) in scale_names
    )


def validate_coverage(
    rows: list[dict[str, Any]],
    *,
    checkpoints: list[Path],
    checkpoint_payloads: list[dict[str, Any]],
    methods: list[str],
    variant: str,
    sources: list[dict[str, Any]],
    scales: tuple[str, ...],
    path: Path,
) -> None:
    expected = len(sources) * len(scales)
    report = []
    failed = []
    for checkpoint, payload in zip(checkpoints, checkpoint_payloads):
        checkpoint_label = make_checkpoint_label(checkpoint, payload)
        for method in methods:
            selected = [
                row
                for row in rows
                if row.get("checkpoint") == checkpoint_label
                and row.get("method") == method
                and row.get("variant") == variant
            ]
            counts: dict[str, int] = {}
            for row in selected:
                status = str(row.get("status", "ok"))
                counts[status] = counts.get(status, 0) + 1
            item = {
                "checkpoint": checkpoint_label,
                "method": method,
                "variant": variant,
                "expected": expected,
                "observed": len(selected),
                "status_counts": counts,
                "complete": len(selected) == expected,
            }
            report.append(item)
            if len(selected) != expected:
                failed.append(item)
            elif method != "specialist" and counts.get("ok", 0) != expected:
                failed.append(item)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        raise RuntimeError(f"Evaluation coverage is incomplete; inspect {path}")


def write_merged_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(
        deduplicate_sample_rows(rows),
        key=lambda row: (
            str(row.get("checkpoint", "")),
            str(row.get("method", "")),
            str(row.get("variant", "default")),
            int(row.get("source_index", -1)),
            float(row.get("scale", 0)),
        ),
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            )


def write_checkpoint_sweep(path: Path, summary_json_path: Path) -> None:
    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["summary"]
        if row["method"] in {"raw", "ema"}
        and row["metric"] in {"latent_charbonnier", "temporal_delta_l1"}
    ]
    write_csv(path, rows)
    candidates = [
        row
        for row in rows
        if row["scale"] == "macro" and row["metric"] == "latent_charbonnier"
    ]
    best_by_weights = {}
    for method in ("raw", "ema"):
        method_rows = [row for row in candidates if row["method"] == method]
        if method_rows:
            best_by_weights[method] = min(
                method_rows, key=lambda row: float(row["mean"])
            )
    if candidates:
        best_by_weights["overall"] = min(candidates, key=lambda row: float(row["mean"]))
    path.with_name("best_checkpoint.json").write_text(
        json.dumps(best_by_weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def require_rgb_paths(args: argparse.Namespace) -> None:
    missing = [name for name in ("model_root",) if not getattr(args, name)]
    if missing:
        raise ValueError(
            f"RGB evaluation requires: {', '.join('--' + name for name in missing)}"
        )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _torch_load(
    path: str | Path, *, map_location: str | torch.device
) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full validation suite for universal clean-latent U-ITU"
    )
    parser.add_argument(
        "--mode", choices=["latent", "rgb", "timing", "sweep", "all"], default="latent"
    )
    parser.add_argument("--checkpoint", nargs="+", action="extend", default=[])
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--checkpoint_glob", default="step_*.pt")
    parser.add_argument("--include_last", action="store_true")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=VALID_METHODS,
        default=["raw", "ema", "nearest", "trilinear", "bicubic"],
    )
    parser.add_argument("--source_offset", type=int, default=0)
    parser.add_argument(
        "--max_sources", type=int, default=0, help="0 evaluates every manifest source"
    )
    parser.add_argument(
        "--decode_max_sources", type=int, default=20, help="0 uses --max_sources/all"
    )
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--val_max_samples", type=int)
    parser.add_argument("--allow_dataset_mismatch", action="store_true")
    parser.add_argument("--allow_split_override", action="store_true")
    parser.add_argument("--charbonnier_eps", type=float)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--specialist_checkpoint")
    parser.add_argument("--specialist_config")
    parser.add_argument("--specialist_use_ema", action="store_true")
    parser.add_argument(
        "--skip_incompatible_specialist",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--model_root")
    parser.add_argument("--vae_path")
    parser.add_argument("--wan_repo")
    parser.add_argument(
        "--vae_backend",
        choices=["auto", "official", "lightx2v", "diffusers"],
        default="lightx2v",
    )
    parser.add_argument("--rgb_metrics", nargs="+", default=["psnr", "ssim", "lpips"])
    parser.add_argument("--metric_batch_size", type=int, default=4)
    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--visual_max_sources", type=int, default=8)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--panel_width", type=int, default=416)
    parser.add_argument("--panel_height", type=int, default=240)

    parser.add_argument("--timing_warmup", type=int, default=5)
    parser.add_argument("--timing_repeats", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap_samples < 0 or args.timing_warmup < 0 or args.timing_repeats < 1:
        parser.error(
            "bootstrap/timing counts must be non-negative and timing repeats must be positive"
        )
    if args.charbonnier_eps is not None and args.charbonnier_eps <= 0:
        parser.error("--charbonnier_eps must be positive")
    return args


if __name__ == "__main__":
    main()
