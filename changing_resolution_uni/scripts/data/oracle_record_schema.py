#!/usr/bin/env python3
"""Strict schema helpers for scored oracle timestep records."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


FORMAL_STEPS = [30, 35, *range(40, 51)]
QUALITY5_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


class OracleRecordError(ValueError):
    """Raised when an oracle record cannot be used for formal router training."""


def _finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleRecordError(f"{field} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise OracleRecordError(f"{field} must be finite; got {number!r}")
    return number


def _quality(value: Any, field: str) -> float:
    number = _finite_float(value, field)
    if not 0.0 <= number <= 1.0:
        raise OracleRecordError(f"{field} must be in [0, 1]; got {number}")
    return number


def validate_scored_record(
    record: dict[str, Any],
    *,
    candidate_steps: Iterable[int] = FORMAL_STEPS,
    quality_dimensions: Iterable[str] = QUALITY5_DIMENSIONS,
) -> dict[str, Any]:
    """Validate and normalize one fully scored prompt/seed trajectory record."""
    expected_steps = [int(step) for step in candidate_steps]
    expected_dimensions = [str(name) for name in quality_dimensions]

    try:
        prompt_id = int(record["prompt_id"])
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleRecordError("record requires integer prompt_id and seed") from exc

    prompt_text = str(record.get("prompt_text", "")).strip()
    if not prompt_text:
        raise OracleRecordError(f"prompt {prompt_id} seed {seed} has empty prompt_text")

    native_latency = _finite_float(
        record.get("native_latency_seconds"), "native_latency_seconds"
    )
    if native_latency <= 0.0:
        raise OracleRecordError("native_latency_seconds must be positive")
    native_vbench5 = _quality(record.get("native_vbench5"), "native_vbench5")

    native_dimensions = record.get("native_dimensions")
    if not isinstance(native_dimensions, dict):
        raise OracleRecordError("native_dimensions must be a mapping")
    normalized_native_dimensions = {
        name: _quality(native_dimensions.get(name), f"native_dimensions.{name}")
        for name in expected_dimensions
    }

    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise OracleRecordError("candidates must be a list")
    if len(candidates) != len(expected_steps):
        raise OracleRecordError(
            f"expected {len(expected_steps)} candidates; got {len(candidates)}"
        )

    by_step: dict[int, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise OracleRecordError(f"candidates[{index}] must be a mapping")
        try:
            step = int(candidate["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleRecordError(f"candidates[{index}] has invalid step") from exc
        if step in by_step:
            raise OracleRecordError(f"duplicate candidate step {step}")
        by_step[step] = candidate

    observed_steps = sorted(by_step)
    if set(observed_steps) != set(expected_steps):
        missing = sorted(set(expected_steps) - set(observed_steps))
        extra = sorted(set(observed_steps) - set(expected_steps))
        raise OracleRecordError(
            f"candidate step coverage mismatch; missing={missing}, extra={extra}"
        )

    normalized_candidates = []
    for step in expected_steps:
        candidate = by_step[step]
        vbench5 = _quality(candidate.get("vbench5"), f"step {step}.vbench5")
        latency = _finite_float(
            candidate.get("latency_seconds"), f"step {step}.latency_seconds"
        )
        if latency <= 0.0:
            raise OracleRecordError(f"step {step}.latency_seconds must be positive")

        dimensions = candidate.get("dimensions")
        if not isinstance(dimensions, dict):
            raise OracleRecordError(f"step {step}.dimensions must be a mapping")
        normalized_dimensions = {
            name: _quality(dimensions.get(name), f"step {step}.dimensions.{name}")
            for name in expected_dimensions
        }
        normalized_candidates.append(
            {
                "step": step,
                "vbench5": vbench5,
                "latency_seconds": latency,
                "dimensions": normalized_dimensions,
                "latency_source": str(candidate.get("latency_source", "unknown")),
            }
        )

    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_text": prompt_text,
        "native_vbench5": native_vbench5,
        "native_latency_seconds": native_latency,
        "native_dimensions": normalized_native_dimensions,
        "native_latency_source": str(record.get("native_latency_source", "unknown")),
        "candidates": normalized_candidates,
    }


def utility_vector(
    normalized_record: dict[str, Any], primary_lambda: float
) -> np.ndarray:
    """Recompute utility from validated quality and latency instead of stored labels."""
    native_latency = float(normalized_record["native_latency_seconds"])
    return np.asarray(
        [
            float(candidate["vbench5"])
            - float(primary_lambda)
            * (float(candidate["latency_seconds"]) / native_latency)
            for candidate in normalized_record["candidates"]
        ],
        dtype=np.float32,
    )


def aggregate_prompt_records(
    records_by_prompt: dict[int, list[dict[str, Any]]],
    *,
    candidate_steps: list[int],
    primary_lambda: float,
    expected_seeds: Iterable[int] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Build one prompt-level sample by averaging utility across seeds."""
    if not records_by_prompt:
        raise ValueError("No oracle records were provided")

    expected_seed_set = (
        {int(seed) for seed in expected_seeds} if expected_seeds is not None else None
    )
    if expected_seed_set is not None and not expected_seed_set:
        raise ValueError("expected_seeds cannot be empty")
    prompt_samples: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    for prompt_id, raw_records in sorted(records_by_prompt.items()):
        normalized_records = []
        for raw_record in raw_records:
            try:
                normalized_records.append(
                    validate_scored_record(raw_record, candidate_steps=candidate_steps)
                )
            except OracleRecordError as exc:
                errors.append(
                    f"prompt {prompt_id} seed {raw_record.get('seed', '?')}: {exc}"
                )
        if len(normalized_records) != len(raw_records):
            continue

        seeds = [int(record["seed"]) for record in normalized_records]
        seed_set = set(seeds)
        if len(seed_set) != len(seeds):
            errors.append(f"prompt {prompt_id}: duplicate seed records {seeds}")
            continue
        if expected_seed_set is None:
            expected_seed_set = seed_set
        elif seed_set != expected_seed_set:
            errors.append(
                f"prompt {prompt_id}: seed coverage {sorted(seed_set)} does not match "
                f"expected {sorted(expected_seed_set)}"
            )
            continue

        prompt_texts = {record["prompt_text"] for record in normalized_records}
        if len(prompt_texts) != 1:
            errors.append(f"prompt {prompt_id}: prompt_text differs across seeds")
            continue

        utilities = np.stack(
            [utility_vector(record, primary_lambda) for record in normalized_records]
        )
        vbench5 = np.asarray(
            [
                [candidate["vbench5"] for candidate in record["candidates"]]
                for record in normalized_records
            ],
            dtype=np.float32,
        )
        latencies = np.asarray(
            [
                [candidate["latency_seconds"] for candidate in record["candidates"]]
                for record in normalized_records
            ],
            dtype=np.float32,
        )
        native_latencies = np.asarray(
            [record["native_latency_seconds"] for record in normalized_records],
            dtype=np.float32,
        )
        prompt_samples[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_texts.pop(),
            "seed_count": len(normalized_records),
            "seeds": sorted(seeds),
            "utilities": utilities.mean(axis=0),
            "vbench5": vbench5.mean(axis=0),
            "latencies": latencies.mean(axis=0),
            "native_latency_seconds": float(native_latencies.mean()),
            "seed_oracle_utility": float(utilities.max(axis=1).mean()),
        }

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise ValueError(f"Oracle record coverage check failed:\n{preview}{suffix}")
    if not prompt_samples or expected_seed_set is None:
        raise ValueError("No valid prompt-level oracle samples were produced")
    return prompt_samples, sorted(expected_seed_set)
