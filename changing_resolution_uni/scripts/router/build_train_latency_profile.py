#!/usr/bin/env python3
"""Build a locked H100 cost profile from strict scored train trajectories."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    QUALITY5_DIMENSIONS,
    validate_scored_record,
)


SCHEMA = "train_calibrated_latency_profile_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-train-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-label", default="H100")
    parser.add_argument("--expected-prompts", type=int, default=1000)
    parser.add_argument("--expected-base-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    if args.expected_prompts < 1 or args.bootstrap_samples < 1:
        parser.error("expected-prompts and bootstrap-samples must be positive")
    if not str(args.hardware_label).strip():
        parser.error("hardware-label must not be empty")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def vector_summary(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    means = values.mean(axis=0, dtype=np.float64)
    medians = np.median(values, axis=0)
    stds = values.std(axis=0, dtype=np.float64)
    p05 = np.quantile(values, 0.05, axis=0)
    p95 = np.quantile(values, 0.95, axis=0)
    ci = [
        bootstrap_mean_ci(values[:, index], samples=bootstrap_samples, rng=rng)
        for index in range(values.shape[1])
    ]
    return {
        "mean": means.tolist(),
        "median": medians.tolist(),
        "std": stds.tolist(),
        "coefficient_of_variation": np.divide(
            stds,
            means,
            out=np.zeros_like(stds),
            where=means > 0,
        ).tolist(),
        "p05": p05.tolist(),
        "p95": p95.tolist(),
        "bootstrap_ci95_low": [item[0] for item in ci],
        "bootstrap_ci95_high": [item[1] for item in ci],
    }


def main() -> None:
    args = parse_args()
    scored_dir = Path(args.scored_train_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite locked latency profile: {output}")
    manifest_path = scored_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("quality_profile") != "strict_vbench5_v1":
        raise ValueError("Train scoring is not strict_vbench5_v1")
    if manifest.get("quality_dimensions") != QUALITY5_DIMENSIONS:
        raise ValueError("Train scoring dimensions differ from VBench-5")
    if manifest.get("is_complete") is not True:
        raise ValueError("Train scored dataset is incomplete")
    candidate_steps = [int(value) for value in manifest.get("candidate_steps", [])]
    if not candidate_steps:
        raise ValueError("Train scored manifest has no candidate steps")

    record_files = [str(value) for value in manifest.get("record_files", [])]
    record_hashes = manifest.get("record_sha256", {})
    if len(record_files) != args.expected_prompts:
        raise ValueError(
            f"Expected {args.expected_prompts} train records, got {len(record_files)}"
        )
    normalized_records = []
    for name in record_files:
        path = scored_dir / "records" / name
        observed_hash = sha256_file(path)
        if record_hashes.get(name) != observed_hash:
            raise ValueError(f"Train record SHA256 mismatch: {path}")
        record = validate_scored_record(
            json.loads(path.read_text(encoding="utf-8")),
            candidate_steps=candidate_steps,
            require_dimensions=True,
            require_provenance=True,
        )
        if int(record["seed"]) - int(record["prompt_id"]) != args.expected_base_seed:
            raise ValueError(
                "Train record does not follow the expected prompt-offset seed policy: "
                f"{path}"
            )
        normalized_records.append(record)

    prompt_ids = [int(record["prompt_id"]) for record in normalized_records]
    if len(set(prompt_ids)) != args.expected_prompts:
        raise ValueError("Train records do not contain unique prompt IDs")
    order = np.argsort(np.asarray(prompt_ids, dtype=np.int64))
    prompt_ids_array = np.asarray(prompt_ids, dtype=np.int64)[order]
    candidate_seconds = np.asarray(
        [
            [float(candidate["latency_seconds"]) for candidate in record["candidates"]]
            for record in normalized_records
        ],
        dtype=np.float64,
    )[order]
    native_seconds = np.asarray(
        [float(record["native_latency_seconds"]) for record in normalized_records],
        dtype=np.float64,
    )[order]
    normalized_costs = candidate_seconds / native_seconds[:, None]
    if not np.isfinite(normalized_costs).all() or np.any(normalized_costs <= 0):
        raise ValueError("Train normalized costs must be finite and positive")

    rng = np.random.default_rng(args.bootstrap_seed)
    normalized_summary = vector_summary(
        normalized_costs,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    candidate_seconds_summary = vector_summary(
        candidate_seconds,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    native_summary = {
        "mean": float(native_seconds.mean(dtype=np.float64)),
        "median": float(np.median(native_seconds)),
        "std": float(native_seconds.std(dtype=np.float64)),
        "coefficient_of_variation": float(
            native_seconds.std(dtype=np.float64) / native_seconds.mean(dtype=np.float64)
        ),
        "p05": float(np.quantile(native_seconds, 0.05)),
        "p95": float(np.quantile(native_seconds, 0.95)),
    }
    native_ci = bootstrap_mean_ci(
        native_seconds,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    native_summary["bootstrap_ci95_low"] = native_ci[0]
    native_summary["bootstrap_ci95_high"] = native_ci[1]

    selected_profile = np.asarray(normalized_summary["mean"], dtype=np.float64)
    calibrated_native_seconds = float(native_summary["mean"])
    calibrated_candidate_seconds = selected_profile * calibrated_native_seconds
    monotonic_violations = [
        {
            "from_step": candidate_steps[index],
            "to_step": candidate_steps[index + 1],
            "increase": float(selected_profile[index + 1] - selected_profile[index]),
        }
        for index in range(len(candidate_steps) - 1)
        if selected_profile[index + 1] > selected_profile[index]
    ]
    even = normalized_costs[prompt_ids_array % 2 == 0].mean(axis=0)
    odd = normalized_costs[prompt_ids_array % 2 == 1].mean(axis=0)

    candidate_sources = collections.Counter(
        str(candidate["latency_source"])
        for record in normalized_records
        for candidate in record["candidates"]
    )
    native_sources = collections.Counter(
        str(record["native_latency_source"]) for record in normalized_records
    )
    payload = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware_label": str(args.hardware_label),
        "source_split": "train",
        "source_scored_manifest": str(manifest_path),
        "source_scored_manifest_sha256": sha256_file(manifest_path),
        "source_prompt_count": args.expected_prompts,
        "source_trajectory_count": len(normalized_records),
        "expected_base_seed": args.expected_base_seed,
        "candidate_steps": candidate_steps,
        "aggregation_used_for_selection": "mean_of_per_trajectory_normalized_costs",
        "selected_normalized_cost_profile": selected_profile.tolist(),
        "calibrated_native_latency_seconds": calibrated_native_seconds,
        "calibrated_candidate_latency_seconds": calibrated_candidate_seconds.tolist(),
        "normalized_cost_distribution": normalized_summary,
        "candidate_latency_seconds_distribution": candidate_seconds_summary,
        "native_latency_seconds_distribution": native_summary,
        "stability": {
            "even_prompt_mean": even.tolist(),
            "odd_prompt_mean": odd.tolist(),
            "max_even_odd_absolute_difference": float(np.max(np.abs(even - odd))),
        },
        "monotonic_nonincreasing": not monotonic_violations,
        "monotonic_violations": monotonic_violations,
        "latency_source_counts": {
            "candidate": dict(sorted(candidate_sources.items())),
            "native": dict(sorted(native_sources.items())),
        },
        "bootstrap": {
            "unit": "train_prompt_trajectory",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
