#!/usr/bin/env python3
"""Build lambda-independent online state features for the 1500-prompt router."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (  # noqa: E402
    FORMAL_STEPS,
    QUALITY5_DIMENSIONS,
    OracleRecordError,
    validate_scored_record,
)

PLAN_SCHEMA = "oracle_1500_8gpu_generation_plan_v1"
LATENT_SCHEMA = "wan_taa_free_oracle_latent_v1"
OUTPUT_SCHEMA = "variable_lambda_online_state_dataset_v1"
LATENCY_PROFILE_SCHEMA = "train_calibrated_latency_profile_v1"
EXPECTED_LATENT_SHAPE = (1, 16, 21, 46, 80)
EXPECTED_LATENT_CORE_SHAPE = EXPECTED_LATENT_SHAPE[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--scored-train-dir", required=True)
    parser.add_argument("--scored-eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--latency-profile",
        required=True,
        help="Locked train-calibrated latency profile used by every selection run.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation"],
        help="Selection defaults to train+validation so test labels remain unread.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.progress_every < 1 or args.torch_threads < 1:
        parser.error("progress-every and torch-threads must be positive")
    if len(set(args.splits)) != len(args.splits):
        parser.error("splits must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_plan(generation_root: Path) -> tuple[Path, dict[str, Any]]:
    path = generation_root / "generation_plan.json"
    plan = load_json(path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unexpected generation plan schema: {plan.get('schema')!r}")
    if [int(step) for step in plan.get("candidate_steps", [])] != FORMAL_STEPS:
        raise ValueError("Generation plan candidate steps differ from FORMAL_STEPS")
    required = {"train", "validation", "test"}
    if set(plan.get("splits", {})) != required:
        raise ValueError("Generation plan must contain train/validation/test splits")
    if plan.get("artifacts", {}).get("latent_schema") != LATENT_SCHEMA:
        raise ValueError("Generation plan does not require the expected latent schema")
    return path, plan


def load_latency_profile(
    path: Path,
    *,
    candidate_steps: list[int],
    scored_train_manifest_sha256: str,
    expected_train_prompts: int,
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != LATENCY_PROFILE_SCHEMA:
        raise ValueError(f"Unexpected latency profile schema: {payload.get('schema')!r}")
    if [int(value) for value in payload.get("candidate_steps", [])] != candidate_steps:
        raise ValueError("Latency profile candidate steps differ from generation plan")
    if payload.get("source_split") != "train":
        raise ValueError("Latency profile must be calibrated from train only")
    if payload.get("monotonic_nonincreasing") is not True:
        raise ValueError(
            "Latency profile is not monotonic non-increasing across candidate steps"
        )
    if int(payload.get("source_prompt_count", -1)) != expected_train_prompts:
        raise ValueError("Latency profile train prompt count mismatch")
    if payload.get("source_scored_manifest_sha256") != scored_train_manifest_sha256:
        raise ValueError("Latency profile is not bound to the selected scored train manifest")
    costs = np.asarray(payload.get("selected_normalized_cost_profile"), dtype=np.float64)
    seconds = np.asarray(payload.get("calibrated_candidate_latency_seconds"), dtype=np.float64)
    if costs.shape != (len(candidate_steps),) or seconds.shape != costs.shape:
        raise ValueError("Latency profile vector shape mismatch")
    if not np.isfinite(costs).all() or not np.isfinite(seconds).all():
        raise ValueError("Latency profile contains non-finite values")
    if np.any(costs <= 0) or np.any(seconds <= 0):
        raise ValueError("Latency profile costs and seconds must be positive")
    native = float(payload.get("calibrated_native_latency_seconds", 0.0))
    if not math.isfinite(native) or native <= 0:
        raise ValueError("Latency profile calibrated native latency must be positive")
    return payload


def prompt_ids(split_spec: dict[str, Any]) -> list[int]:
    offset = int(split_spec["prompt_offset"])
    count = int(split_spec["prompt_count"])
    return list(range(offset, offset + count))


def expected_actual_seeds(prompt_id: int, split_spec: dict[str, Any]) -> set[int]:
    return {int(base_seed) + prompt_id for base_seed in split_spec["seeds"]}


def load_scored_records(
    scored_dir: Path,
    allowed_prompt_ids: set[int],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any], Path]:
    manifest_path = scored_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("quality_profile") != "strict_vbench5_v1":
        raise ValueError(f"Scored dataset is not strict_vbench5_v1: {manifest_path}")
    if manifest.get("quality_dimensions") != QUALITY5_DIMENSIONS:
        raise ValueError(
            f"Scored dataset has unexpected quality dimensions: {manifest_path}"
        )
    if manifest.get("is_complete") is not True:
        raise ValueError(f"Scored dataset is incomplete: {manifest_path}")
    names = manifest.get("record_files")
    hashes = manifest.get("record_sha256")
    if not isinstance(names, list) or not isinstance(hashes, dict):
        raise ValueError(f"Scored manifest lacks record file hashes: {manifest_path}")
    if set(hashes) != {str(name) for name in names}:
        raise ValueError(f"Scored manifest hash coverage mismatch: {manifest_path}")

    records_dir = (scored_dir / "records").resolve()
    records: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        try:
            prompt_token = name.split("_s", 1)[0].removeprefix("p")
            prompt_id = int(prompt_token)
        except (ValueError, IndexError):
            continue
        if prompt_id not in allowed_prompt_ids:
            continue
        path = (records_dir / name).resolve()
        try:
            if path.parent != records_dir:
                raise ValueError("record path escapes records directory")
            if sha256_file(path) != hashes[name]:
                raise ValueError("record SHA256 differs from manifest")
            raw = load_json(path)
            normalized = validate_scored_record(
                raw,
                candidate_steps=FORMAL_STEPS,
                require_dimensions=True,
                require_native_dimensions=True,
                require_provenance=True,
            )
            key = (int(normalized["prompt_id"]), int(normalized["seed"]))
            if key in records:
                raise ValueError(f"duplicate record key {key}")
            normalized["record_path"] = str(path)
            normalized["record_sha256"] = hashes[name]
            records[key] = normalized
        except (OSError, json.JSONDecodeError, OracleRecordError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        raise RuntimeError(f"Failed to load scored records:\n{preview}")
    return records, manifest, manifest_path


def discover_sample_manifests(
    physical_root: Path,
    allowed_prompt_ids: set[int],
) -> dict[tuple[int, int], Path]:
    seed_dirs = list(physical_root.glob("raw_samples/seed_*"))
    seed_dirs.extend(physical_root.glob("_parts/part_*/raw_samples/seed_*"))
    result: dict[tuple[int, int], Path] = {}
    errors: list[str] = []
    for seed_dir in sorted({path.resolve() for path in seed_dirs if path.is_dir()}):
        for path in sorted((seed_dir / "manifests").glob("*.json")):
            try:
                prompt_id_from_name = int(path.stem.split("_seed", 1)[0])
            except (ValueError, IndexError):
                continue
            if prompt_id_from_name not in allowed_prompt_ids:
                continue
            try:
                payload = load_json(path)
                key = (int(payload["prompt_index"]), int(payload["seed"]))
                if key[0] != prompt_id_from_name:
                    raise ValueError("manifest filename and prompt_index differ")
                if key in result:
                    raise ValueError(f"duplicate sample manifest also in {result[key]}")
                result[key] = path.resolve()
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        raise RuntimeError(f"Failed to index sample manifests:\n{preview}")
    return result


def resolve_latent_paths(
    sample_manifest_path: Path,
    sample_manifest: dict[str, Any],
) -> list[Path]:
    seed_dir = sample_manifest_path.parent.parent
    prompt_id = int(sample_manifest["prompt_index"])
    seed = int(sample_manifest["seed"])
    sample_id = f"{prompt_id:04d}_seed{seed}"
    branches = {
        int(branch["candidate_step"]): branch
        for branch in sample_manifest.get("branches", [])
        if isinstance(branch, dict) and "candidate_step" in branch
    }
    if set(branches) != set(FORMAL_STEPS):
        raise ValueError(
            f"{sample_manifest_path}: branch coverage differs from {FORMAL_STEPS}"
        )
    paths = []
    for step in FORMAL_STEPS:
        recorded = branches[step].get("latent_path")
        path = Path(recorded).resolve() if recorded else Path()
        if not recorded or not path.is_file():
            path = (
                seed_dir
                / "latents"
                / f"step{step:02d}"
                / f"{sample_id}_step{step:02d}.pt"
            ).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing latent archive: {path}")
        paths.append(path)
    return paths


def global_stats(value: torch.Tensor) -> dict[str, float]:
    spatial_h = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
    spatial_w = (value[..., 1:] - value[..., :-1]).abs().mean()
    temporal = (value[:, 1:] - value[:, :-1]).abs().mean()
    temporal_second = (
        (value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]).abs().mean()
        if value.shape[1] > 2
        else value.new_tensor(0.0)
    )
    laplacian = (
        (value[..., 2:, 1:-1] - 2.0 * value[..., 1:-1, 1:-1] + value[..., :-2, 1:-1])
        .abs()
        .mean()
        + (value[..., 1:-1, 2:] - 2.0 * value[..., 1:-1, 1:-1] + value[..., 1:-1, :-2])
        .abs()
        .mean()
    ) * 0.5
    return {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "abs_mean": float(value.abs().mean()),
        "rms": float(value.square().mean().sqrt()),
        "spatial_gradient_abs_mean": float((spatial_h + spatial_w) * 0.5),
        "temporal_gradient_abs_mean": float(temporal),
        "temporal_second_abs_mean": float(temporal_second),
        "spatial_laplacian_abs_mean": float(laplacian),
    }


def channel_stats(value: torch.Tensor) -> dict[str, torch.Tensor]:
    dims = (1, 2, 3)
    return {
        "mean": value.mean(dim=dims),
        "std": value.std(dim=dims, unbiased=False),
        "rms": value.square().mean(dim=dims).sqrt(),
    }


def local_energy_stats(value: torch.Tensor) -> dict[str, float]:
    energy = value.square().mean(dim=0, keepdim=True).unsqueeze(0)
    pooled = F.adaptive_avg_pool3d(energy, output_size=(3, 4, 4)).sqrt().flatten()
    return {
        "mean": float(pooled.mean()),
        "std": float(pooled.std(unbiased=False)),
        "max": float(pooled.max()),
        "p90": float(torch.quantile(pooled, 0.90)),
        "max_over_mean": float(pooled.max() / pooled.mean().clamp(min=1e-8)),
    }


def cosine_per_channel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp(min=1e-8)


def append_mapping(
    values: list[float],
    names: list[str],
    groups: dict[str, list[int]],
    group: str,
    prefix: str,
    mapping: dict[str, float],
) -> None:
    for key, value in mapping.items():
        groups[group].append(len(values))
        values.append(float(value))
        names.append(f"{prefix}.{key}")


def append_vector(
    values: list[float],
    names: list[str],
    groups: dict[str, list[int]],
    group: str,
    prefix: str,
    vector: torch.Tensor,
) -> None:
    for index, value in enumerate(vector.tolist()):
        groups[group].append(len(values))
        values.append(float(value))
        names.append(f"{prefix}.channel_{index:02d}")


def extract_step_features(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    sigma: float,
    previous_x0: torch.Tensor | None,
    previous_sigma: float | None,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    x_t = x_t.squeeze(0).to(dtype=torch.float32)
    x0 = x0.squeeze(0).to(dtype=torch.float32)
    if x_t.shape != x0.shape or x0.ndim != 4 or x0.shape[0] != 16:
        raise ValueError(
            "Expected matching [16,T,H,W] tensors after batch normalization, "
            f"got {x_t.shape}, {x0.shape}"
        )
    if not torch.isfinite(x_t).all() or not torch.isfinite(x0).all():
        raise ValueError("Latent archive contains non-finite values")
    residual = x_t - x0

    values: list[float] = []
    names: list[str] = []
    groups: dict[str, list[int]] = defaultdict(list)
    append_mapping(values, names, groups, "x0_global", "x0", global_stats(x0))
    append_mapping(
        values, names, groups, "residual_global", "residual", global_stats(residual)
    )

    x0_channel = channel_stats(x0)
    residual_channel = channel_stats(residual)
    for statistic in ("mean", "std", "rms"):
        append_vector(
            values,
            names,
            groups,
            "x0_channel",
            f"x0.{statistic}",
            x0_channel[statistic],
        )
    for statistic in ("std", "rms"):
        append_vector(
            values,
            names,
            groups,
            "residual_channel",
            f"residual.{statistic}",
            residual_channel[statistic],
        )
    append_vector(
        values,
        names,
        groups,
        "residual_channel",
        "residual_signal_ratio.rms",
        residual_channel["rms"] / x0_channel["rms"].clamp(min=1e-8),
    )
    append_mapping(
        values, names, groups, "local_energy", "x0.local", local_energy_stats(x0)
    )
    append_mapping(
        values,
        names,
        groups,
        "local_energy",
        "residual.local",
        local_energy_stats(residual),
    )

    has_previous = previous_x0 is not None and previous_sigma is not None
    delta_sigma = abs(float(sigma) - float(previous_sigma)) if has_previous else 0.0
    trajectory_scalars = {
        "has_previous": float(has_previous),
        "delta_sigma": delta_sigma,
        "delta_rms_per_sigma": 0.0,
        "global_cosine": 0.0,
    }
    if has_previous:
        previous = previous_x0.squeeze(0).to(dtype=torch.float32)
        delta = x0 - previous
        denominator = max(delta_sigma, 1e-6)
        trajectory_scalars["delta_rms_per_sigma"] = float(
            delta.square().mean().sqrt() / denominator
        )
        trajectory_scalars["global_cosine"] = float(
            F.cosine_similarity(x0.flatten(), previous.flatten(), dim=0)
        )
        delta_channel_rms = delta.square().mean(dim=(1, 2, 3)).sqrt() / denominator
        channel_cosine = cosine_per_channel(x0, previous)
    else:
        delta_channel_rms = torch.zeros(16)
        channel_cosine = torch.zeros(16)
    append_mapping(
        values,
        names,
        groups,
        "trajectory_delta",
        "trajectory",
        trajectory_scalars,
    )
    append_vector(
        values,
        names,
        groups,
        "trajectory_delta",
        "trajectory.delta_rms_per_sigma",
        delta_channel_rms,
    )
    append_vector(
        values,
        names,
        groups,
        "trajectory_delta",
        "trajectory.cosine",
        channel_cosine,
    )
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Extracted feature vector contains non-finite values")
    return array, names, dict(groups)


def load_latent_archive(
    path: Path,
    prompt_id: int,
    seed: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != LATENT_SCHEMA:
        raise ValueError(f"{path}: unexpected latent schema {payload.get('schema')!r}")
    expected_sample_id = f"{prompt_id:04d}_seed{seed}"
    if str(payload.get("sample_id")) != expected_sample_id:
        raise ValueError(f"{path}: sample_id does not match {expected_sample_id}")
    if int(payload.get("seed")) != seed or int(payload.get("candidate_step")) != step:
        raise ValueError(f"{path}: latent identity mismatch")
    x_t = payload.get("x_t_lr")
    x0 = payload.get("x0_pred_lr")
    if not isinstance(x_t, torch.Tensor) or not isinstance(x0, torch.Tensor):
        raise ValueError(f"{path}: missing latent tensors")

    # LightX2V has emitted both layouts over time: some scheduler versions keep
    # a singleton batch axis while others expose the per-sample latent directly.
    # They represent the same sample, so normalize both to the documented
    # [1,C,T,H,W] contract without rewriting the immutable source archives.
    accepted_shapes = {EXPECTED_LATENT_SHAPE, EXPECTED_LATENT_CORE_SHAPE}
    x_t_shape = tuple(x_t.shape)
    x0_shape = tuple(x0.shape)
    if (
        x_t_shape not in accepted_shapes
        or x0_shape not in accepted_shapes
        or x_t_shape != x0_shape
    ):
        raise ValueError(
            f"{path}: expected latent shape {EXPECTED_LATENT_SHAPE} or "
            f"{EXPECTED_LATENT_CORE_SHAPE}, "
            f"got x_t={x_t_shape} x0={x0_shape}"
        )
    if x_t.ndim == 4:
        x_t = x_t.unsqueeze(0)
    if x0.ndim == 4:
        x0 = x0.unsqueeze(0)
    sigma = float(payload["sigma"])
    if not math.isfinite(sigma):
        raise ValueError(f"{path}: sigma is not finite")
    return x_t, x0, sigma


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    generation_root = Path(args.generation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.skip_existing:
        raise FileExistsError(
            f"Output already exists; use --skip-existing only to resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path, plan = load_plan(generation_root)
    selected_splits = list(args.splits)
    selected_prompt_ids = {
        split: set(prompt_ids(plan["splits"][split])) for split in selected_splits
    }
    train_related = selected_prompt_ids.get("train", set())
    eval_related = set().union(
        *(ids for split, ids in selected_prompt_ids.items() if split != "train")
    )

    scored_records: dict[tuple[int, int], dict[str, Any]] = {}
    scored_sources: dict[str, Any] = {}
    if train_related:
        records, manifest, path = load_scored_records(
            Path(args.scored_train_dir).resolve(), train_related
        )
        scored_records.update(records)
        scored_sources["train"] = {
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "quality_profile": manifest["quality_profile"],
        }
    if eval_related:
        records, manifest, path = load_scored_records(
            Path(args.scored_eval_dir).resolve(), eval_related
        )
        overlap = set(scored_records) & set(records)
        if overlap:
            raise ValueError(
                f"Scored train/eval record keys overlap: {sorted(overlap)[:10]}"
            )
        scored_records.update(records)
        scored_sources["eval"] = {
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "quality_profile": manifest["quality_profile"],
        }

    latency_profile_path = Path(args.latency_profile).resolve()
    scored_train_manifest_path = (
        Path(args.scored_train_dir).resolve() / "dataset_manifest.json"
    )
    latency_profile = load_latency_profile(
        latency_profile_path,
        candidate_steps=[int(value) for value in plan["candidate_steps"]],
        scored_train_manifest_sha256=sha256_file(scored_train_manifest_path),
        expected_train_prompts=int(plan["splits"]["train"]["prompt_count"]),
    )

    physical_roots = {
        "train": generation_root / "train",
        "validation": generation_root / "eval",
        "test": generation_root / "eval",
    }
    manifest_maps = {
        "train": discover_sample_manifests(generation_root / "train", train_related)
        if train_related
        else {},
        "eval": discover_sample_manifests(generation_root / "eval", eval_related)
        if eval_related
        else {},
    }

    dummy = torch.zeros(1, 16, 21, 46, 80, dtype=torch.float32)
    _, canonical_names, canonical_groups = extract_step_features(
        dummy, dummy, 1.0, None, None
    )
    del dummy
    split_reports: dict[str, Any] = {}
    total_expected = sum(
        len(selected_prompt_ids[split]) * len(plan["splits"][split]["seeds"])
        for split in selected_splits
    )
    completed = 0
    for split in selected_splits:
        spec = plan["splits"][split]
        physical_key = str(spec["physical_dataset"])
        if physical_key not in {"train", "eval"}:
            raise ValueError(f"Unsupported physical dataset {physical_key!r}")
        sample_manifests = manifest_maps[physical_key]
        t5_dir = physical_roots[split] / "t5_embeddings"
        if not t5_dir.is_dir():
            raise FileNotFoundError(t5_dir)
        feature_dir = output_dir / "features" / split
        feature_dir.mkdir(parents=True, exist_ok=True)
        index_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for prompt_id in sorted(selected_prompt_ids[split]):
            t5_path = (t5_dir / f"prompt_{prompt_id:06d}.npz").resolve()
            if not t5_path.is_file():
                errors.append(f"prompt={prompt_id}: missing T5 embedding {t5_path}")
                continue
            for seed in sorted(expected_actual_seeds(prompt_id, spec)):
                key = (prompt_id, seed)
                record = scored_records.get(key)
                sample_manifest_path = sample_manifests.get(key)
                if record is None or sample_manifest_path is None:
                    errors.append(
                        f"prompt={prompt_id} seed={seed}: record={record is not None} "
                        f"sample_manifest={sample_manifest_path is not None}"
                    )
                    continue
                output_path = feature_dir / f"p{prompt_id:06d}_s{seed}.npz"
                if args.skip_existing and output_path.is_file():
                    try:
                        with np.load(output_path, allow_pickle=False) as existing:
                            if tuple(existing["features"].shape)[0] != len(
                                FORMAL_STEPS
                            ):
                                raise ValueError("candidate count mismatch")
                            if tuple(existing["features"].shape)[1] != len(
                                canonical_names
                            ):
                                raise ValueError("feature count mismatch")
                            if not np.isfinite(existing["features"]).all():
                                raise ValueError("non-finite features")
                    except Exception as exc:
                        errors.append(
                            f"Invalid existing feature file {output_path}: {exc}"
                        )
                        continue
                else:
                    sample_manifest = load_json(sample_manifest_path)
                    try:
                        latent_paths = resolve_latent_paths(
                            sample_manifest_path, sample_manifest
                        )
                        feature_rows = []
                        sigmas = []
                        previous_x0 = None
                        previous_sigma = None
                        for step, latent_path in zip(FORMAL_STEPS, latent_paths):
                            x_t, x0, sigma = load_latent_archive(
                                latent_path, prompt_id, seed, step
                            )
                            features, names, groups = extract_step_features(
                                x_t, x0, sigma, previous_x0, previous_sigma
                            )
                            if names != canonical_names or groups != canonical_groups:
                                raise ValueError(
                                    "feature schema changed across archives"
                                )
                            feature_rows.append(features)
                            sigmas.append(sigma)
                            previous_x0 = x0
                            previous_sigma = sigma
                        candidates = record["candidates"]
                        qualities = np.asarray(
                            [candidate["vbench5"] for candidate in candidates],
                            dtype=np.float32,
                        )
                        latencies = np.asarray(
                            [candidate["latency_seconds"] for candidate in candidates],
                            dtype=np.float32,
                        )
                        dimensions = np.asarray(
                            [
                                [
                                    candidate["dimensions"][name]
                                    for name in QUALITY5_DIMENSIONS
                                ]
                                for candidate in candidates
                            ],
                            dtype=np.float32,
                        )
                        np.savez_compressed(
                            output_path,
                            features=np.stack(feature_rows).astype(np.float32),
                            candidate_steps=np.asarray(FORMAL_STEPS, dtype=np.int64),
                            sigmas=np.asarray(sigmas, dtype=np.float32),
                            qualities=qualities,
                            latencies=latencies,
                            native_latency=np.asarray(
                                record["native_latency_seconds"], dtype=np.float32
                            ),
                            dimensions=dimensions,
                        )
                    except Exception as exc:
                        errors.append(
                            f"prompt={prompt_id} seed={seed} manifest={sample_manifest_path}: {exc}"
                        )
                        continue
                index_rows.append(
                    {
                        "split": split,
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "base_seed": seed - prompt_id,
                        "prompt_text": record["prompt_text"],
                        "feature_file": str(output_path.relative_to(output_dir)),
                        "t5_embedding_path": str(t5_path),
                        "record_path": record["record_path"],
                        "record_sha256": record["record_sha256"],
                        "sample_manifest_path": str(sample_manifest_path),
                    }
                )
                completed += 1
                if completed % args.progress_every == 0 or completed == total_expected:
                    print(
                        f"[{completed}/{total_expected}] extracted {split} p{prompt_id} s{seed}"
                    )
        if errors:
            preview = "\n".join(f"  - {item}" for item in errors[:50])
            suffix = "" if len(errors) <= 50 else f"\n  ... and {len(errors) - 50} more"
            raise RuntimeError(
                f"State feature preparation failed for {split}:\n{preview}{suffix}"
            )
        expected_count = len(selected_prompt_ids[split]) * len(spec["seeds"])
        if len(index_rows) != expected_count:
            raise RuntimeError(
                f"{split}: produced {len(index_rows)} trajectories, expected {expected_count}"
            )
        index_path = output_dir / f"{split}_trajectories.jsonl"
        write_jsonl(index_path, index_rows)
        split_reports[split] = {
            "prompt_offset": int(spec["prompt_offset"]),
            "prompt_count": int(spec["prompt_count"]),
            "base_seeds": [int(seed) for seed in spec["seeds"]],
            "trajectory_count": len(index_rows),
            "physical_dataset": physical_key,
            "index_file": index_path.name,
            "index_sha256": sha256_file(index_path),
        }

    manifest = {
        "schema": OUTPUT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generation_root": str(generation_root),
        "generation_plan": str(plan_path),
        "generation_plan_sha256": sha256_file(plan_path),
        "candidate_steps": FORMAL_STEPS,
        "quality_dimensions": QUALITY5_DIMENSIONS,
        "feature_names": canonical_names,
        "feature_groups": canonical_groups,
        "feature_count": len(canonical_names),
        "lambda_dependent_features": False,
        "selected_splits": selected_splits,
        "test_accessed": "test" in selected_splits,
        "splits": split_reports,
        "scored_sources": scored_sources,
        "latency_profile": {
            "schema": latency_profile["schema"],
            "path": str(latency_profile_path),
            "sha256": sha256_file(latency_profile_path),
            "hardware_label": latency_profile["hardware_label"],
            "source_split": latency_profile["source_split"],
            "aggregation": latency_profile["aggregation_used_for_selection"],
        },
        "is_complete": True,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Variable-lambda state dataset ready: {output_dir}")


if __name__ == "__main__":
    main()
