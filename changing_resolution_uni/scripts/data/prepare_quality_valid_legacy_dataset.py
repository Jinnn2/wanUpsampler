#!/usr/bin/env python3
"""Build an isolated quality-valid development dataset from legacy oracle records."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FORMAL_STEPS = [30, 35, *range(40, 51)]
QUALITY5_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]
QUALITY_PROFILE = "quality_valid_legacy_vbench5_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter legacy records to complete three-seed prompts with genuine "
            "VBench-5 dimensions and estimated branch latency."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-seeds", type=int, nargs="+", default=[42, 100, 2024])
    parser.add_argument("--primary-lambda", type=float, default=0.01)
    parser.add_argument("--quality-mean-tolerance", type=float, default=1e-5)
    parser.add_argument("--equality-tolerance", type=float, default=1e-12)
    parser.add_argument("--latency-monotonic-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def quality(value: Any, field: str) -> float:
    number = finite_float(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def quality_dimensions(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    missing = [name for name in QUALITY5_DIMENSIONS if name not in value]
    if missing:
        raise ValueError(f"{field} missing dimensions {missing}")
    return {
        name: quality(value[name], f"{field}.{name}")
        for name in QUALITY5_DIMENSIONS
    }


def canonicalize_record(
    raw: dict[str, Any],
    *,
    source_path: Path,
    primary_lambda: float,
    quality_mean_tolerance: float,
    equality_tolerance: float,
    latency_monotonic_tolerance: float,
) -> dict[str, Any]:
    prompt_id = int(raw["prompt_id"])
    seed = int(raw["seed"])
    prompt_text = str(raw.get("prompt_text", "")).strip()
    if not prompt_text:
        raise ValueError("empty prompt_text")

    original_native_quality = quality(raw.get("native_vbench5"), "native_vbench5")
    raw_native_dimensions = raw.get("native_dimensions")
    if isinstance(raw_native_dimensions, dict) and raw_native_dimensions:
        native_dimensions = quality_dimensions(
            raw_native_dimensions, "native_dimensions"
        )
        recomputed_native_quality = math.fsum(native_dimensions.values()) / len(
            QUALITY5_DIMENSIONS
        )
        if not math.isclose(
            original_native_quality,
            recomputed_native_quality,
            rel_tol=0.0,
            abs_tol=quality_mean_tolerance,
        ):
            raise ValueError("native scalar does not match dimension mean")
        native_quality_source = "legacy_dimensions_recomputed"
    else:
        native_dimensions = {}
        recomputed_native_quality = original_native_quality
        native_quality_source = "legacy_scalar_without_native_dimensions"
    native_latency = finite_float(
        raw.get("native_latency_seconds"), "native_latency_seconds"
    )
    if native_latency <= 0.0:
        raise ValueError("native latency must be positive")

    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates must be a list")
    by_step: dict[int, dict[str, Any]] = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be a mapping")
        step = int(candidate["step"])
        if step in by_step:
            raise ValueError(f"duplicate step {step}")
        by_step[step] = candidate
    if set(by_step) != set(FORMAL_STEPS):
        raise ValueError("formal step coverage mismatch")

    candidates = []
    original_qualities = []
    latencies = []
    for step in FORMAL_STEPS:
        candidate = by_step[step]
        dimensions = quality_dimensions(candidate.get("dimensions"), f"step {step}")
        original_quality = quality(candidate.get("vbench5"), f"step {step}.vbench5")
        recomputed_quality = math.fsum(dimensions.values()) / len(QUALITY5_DIMENSIONS)
        if not math.isclose(
            original_quality,
            recomputed_quality,
            rel_tol=0.0,
            abs_tol=quality_mean_tolerance,
        ):
            raise ValueError(f"step {step} scalar does not match dimension mean")
        latency = finite_float(
            candidate.get("latency_seconds"), f"step {step}.latency_seconds"
        )
        if latency <= 0.0:
            raise ValueError(f"step {step} latency must be positive")
        utility = recomputed_quality - primary_lambda * latency / native_latency
        candidates.append(
            {
                "step": step,
                "vbench5": recomputed_quality,
                "legacy_original_vbench5": original_quality,
                "dimensions": dimensions,
                "latency_seconds": latency,
                "latency_source": "legacy_branch_estimate",
                "speedup_vs_native": native_latency / latency,
                "utilities": {f"u_{primary_lambda:.2f}": utility},
            }
        )
        original_qualities.append(original_quality)
        latencies.append(latency)

    if all(
        abs(value - original_native_quality) <= equality_tolerance
        for value in original_qualities
    ):
        raise ValueError("native_hr_quality_fallback_signature")
    if not all(
        latencies[index + 1] <= latencies[index] + latency_monotonic_tolerance
        for index in range(len(latencies) - 1)
    ):
        raise ValueError("candidate_latency_not_monotonic_nonincreasing")

    best = max(
        candidates,
        key=lambda candidate: candidate["utilities"][f"u_{primary_lambda:.2f}"],
    )
    return {
        "schema": "quality_valid_legacy_oracle_record_v1",
        "quality_profile": QUALITY_PROFILE,
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_text": prompt_text,
        "native_vbench5": recomputed_native_quality,
        "legacy_original_native_vbench5": original_native_quality,
        "native_dimensions": native_dimensions,
        "native_quality_source": native_quality_source,
        "native_latency_seconds": native_latency,
        "native_latency_source": "legacy_default_or_unprovenanced",
        "candidates": candidates,
        f"optimal_step_lambda_{int(primary_lambda * 100):03d}": best["step"],
        "legacy_source": {
            "record_path": str(source_path),
            "record_sha256": sha256_file(source_path),
            "formal_evidence": False,
        },
    }


def source_record_paths(source_dir: Path) -> list[Path]:
    records_dir = source_dir / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"Source records directory not found: {records_dir}")
    manifest_path = source_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = manifest.get("record_files")
        if isinstance(names, list) and names:
            paths = [(records_dir / str(name)).resolve() for name in names]
            if all(path.parent == records_dir.resolve() for path in paths):
                return paths
    return sorted(records_dir.glob("p*_s*.json"))


def main() -> None:
    args = parse_args()
    if args.primary_lambda < 0 or not math.isfinite(args.primary_lambda):
        raise ValueError("primary-lambda must be finite and non-negative")
    for name in (
        "quality_mean_tolerance",
        "equality_tolerance",
        "latency_monotonic_tolerance",
    ):
        value = float(getattr(args, name))
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to mix runs: {output_dir}"
        )
    if output_dir == source_dir:
        raise ValueError("output-dir must differ from source-dir")

    accepted_by_prompt: dict[int, dict[int, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    rejection_counts: Counter[str] = Counter()
    parse_errors = []
    observed_paths = source_record_paths(source_dir)
    for path in observed_paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            canonical = canonicalize_record(
                raw,
                source_path=path,
                primary_lambda=args.primary_lambda,
                quality_mean_tolerance=args.quality_mean_tolerance,
                equality_tolerance=args.equality_tolerance,
                latency_monotonic_tolerance=args.latency_monotonic_tolerance,
            )
            prompt_id = int(canonical["prompt_id"])
            seed = int(canonical["seed"])
            if seed in accepted_by_prompt[prompt_id]:
                raise ValueError("duplicate prompt/seed")
            accepted_by_prompt[prompt_id][seed] = (path, canonical)
        except Exception as exc:
            reason = str(exc)
            rejection_counts[reason] += 1
            if len(parse_errors) < 100:
                parse_errors.append(f"{path}: {reason}")

    base_seeds = sorted({int(seed) for seed in args.base_seeds})
    selected: list[tuple[Path, dict[str, Any]]] = []
    dropped_prompt_counts: Counter[str] = Counter()
    for prompt_id, by_seed in sorted(accepted_by_prompt.items()):
        expected = {base_seed + prompt_id for base_seed in base_seeds}
        if set(by_seed) != expected:
            dropped_prompt_counts["incomplete_prompt_offset_seed_group"] += 1
            continue
        prompt_texts = {record["prompt_text"] for _, record in by_seed.values()}
        if len(prompt_texts) != 1:
            dropped_prompt_counts["prompt_text_mismatch_across_seeds"] += 1
            continue
        selected.extend(by_seed[seed] for seed in sorted(by_seed))
    if not selected:
        raise RuntimeError("No complete quality-valid prompt groups survived filtering")

    selected_prompt_ids = sorted({int(record["prompt_id"]) for _, record in selected})
    t5_source = source_dir / "t5_embeddings"
    missing_t5 = [
        prompt_id
        for prompt_id in selected_prompt_ids
        if not (t5_source / f"prompt_{prompt_id:06d}.npz").is_file()
    ]
    if missing_t5:
        raise RuntimeError(
            f"Missing T5 embeddings for {len(missing_t5)} selected prompts; "
            f"examples={missing_t5[:20]}"
        )

    staging_dir = output_dir.parent / (
        f".{output_dir.name}.staging-{os.getpid()}-"
        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    staging_records = staging_dir / "records"
    staging_records.mkdir(parents=True, exist_ok=False)
    record_names = []
    record_hashes = {}
    for _, record in sorted(
        selected, key=lambda item: (int(item[1]["prompt_id"]), int(item[1]["seed"]))
    ):
        name = f"p{int(record['prompt_id']):06d}_s{int(record['seed'])}.json"
        destination = staging_records / name
        destination.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        record_names.append(name)
        record_hashes[name] = sha256_file(destination)

    staging_t5 = staging_dir / "t5_embeddings"
    try:
        staging_t5.symlink_to(t5_source, target_is_directory=True)
        t5_storage = "absolute_directory_symlink"
    except OSError:
        staging_t5.mkdir()
        for prompt_id in selected_prompt_ids:
            source_npz = t5_source / f"prompt_{prompt_id:06d}.npz"
            shutil.copy2(source_npz, staging_t5 / source_npz.name)
            source_json = source_npz.with_suffix(".json")
            if source_json.is_file():
                shutil.copy2(source_json, staging_t5 / source_json.name)
        t5_storage = "selected_prompt_files_copied"
    filter_report = {
        "schema": "quality_valid_legacy_filter_report_v1",
        "source_dir": str(source_dir),
        "observed_source_records": len(observed_paths),
        "individually_accepted_records": sum(
            len(by_seed) for by_seed in accepted_by_prompt.values()
        ),
        "selected_prompts": len(selected_prompt_ids),
        "selected_trajectories": len(selected),
        "selected_prompt_id_min": min(selected_prompt_ids),
        "selected_prompt_id_max": max(selected_prompt_ids),
        "rejection_counts": dict(rejection_counts.most_common()),
        "dropped_prompt_counts": dict(dropped_prompt_counts),
        "parse_error_examples": parse_errors,
    }
    (staging_dir / "filter_report.json").write_text(
        json.dumps(filter_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "schema": "prompt_conditioned_router_dataset_v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "quality_profile": QUALITY_PROFILE,
        "quality_dimensions": QUALITY5_DIMENSIONS,
        "quality_aggregation": "arithmetic_mean_legacy_dimensions_float64",
        "latency_profile": "legacy_branch_estimate_with_unprovenanced_native",
        "formal_evidence": False,
        "total_prompts_found": len(selected_prompt_ids),
        "expected_prompts": len(selected_prompt_ids),
        "total_trajectories": len(selected),
        "expected_trajectories": len(selected),
        "expected_base_seeds": base_seeds,
        "seed_policy": "prompt_offset",
        "candidate_steps": FORMAL_STEPS,
        "primary_lambda": args.primary_lambda,
        "record_files": record_names,
        "record_sha256": record_hashes,
        "filter_report": "filter_report.json",
        "t5_storage": t5_storage,
        "is_complete": True,
    }
    (staging_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    staging_dir.rename(output_dir)
    print(json.dumps(filter_report, indent=2, ensure_ascii=False))
    print(f"Quality-valid development dataset: {output_dir}")


if __name__ == "__main__":
    main()
