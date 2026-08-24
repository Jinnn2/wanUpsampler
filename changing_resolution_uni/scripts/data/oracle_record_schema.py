#!/usr/bin/env python3
"""Strict schema helpers for scored oracle timestep records."""

from __future__ import annotations

import math
import re
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
RECORD_PROVENANCE_SCHEMA = "strict_vbench5_record_provenance_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _sha256(value: Any, field: str) -> str:
    normalized = str(value).lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise OracleRecordError(f"{field} must be a lowercase SHA256 digest")
    return normalized


def _diagnostic(value: Any, field: str, name: str) -> float:
    number = _finite_float(value, field)
    lower_bound = -1.0 if name == "overall_consistency" else 0.0
    if not lower_bound <= number <= 1.0:
        raise OracleRecordError(f"{field} must be in [{lower_bound}, 1]; got {number}")
    return number


def validate_scored_record(
    record: dict[str, Any],
    *,
    candidate_steps: Iterable[int] = FORMAL_STEPS,
    quality_dimensions: Iterable[str] = QUALITY5_DIMENSIONS,
    require_dimensions: bool = True,
    require_native_dimensions: bool | None = None,
    require_provenance: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one fully scored prompt/seed trajectory record."""
    expected_steps = [int(step) for step in candidate_steps]
    expected_dimensions = [str(name) for name in quality_dimensions]
    if require_native_dimensions is None:
        require_native_dimensions = require_dimensions

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
    if require_native_dimensions:
        if not isinstance(native_dimensions, dict):
            raise OracleRecordError("native_dimensions must be a mapping")
        normalized_native_dimensions = {
            name: _quality(native_dimensions.get(name), f"native_dimensions.{name}")
            for name in expected_dimensions
        }
        native_mean = math.fsum(normalized_native_dimensions.values()) / len(
            expected_dimensions
        )
        if not math.isclose(native_vbench5, native_mean, rel_tol=0.0, abs_tol=1e-12):
            raise OracleRecordError(
                "native_vbench5 does not equal the float64 mean of native_dimensions; "
                f"scalar={native_vbench5}, recomputed={native_mean}"
            )
    else:
        normalized_native_dimensions = {
            name: _quality(native_dimensions[name], f"native_dimensions.{name}")
            for name in expected_dimensions
            if isinstance(native_dimensions, dict) and name in native_dimensions
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
        if require_dimensions:
            if not isinstance(dimensions, dict):
                raise OracleRecordError(f"step {step}.dimensions must be a mapping")
            normalized_dimensions = {
                name: _quality(dimensions.get(name), f"step {step}.dimensions.{name}")
                for name in expected_dimensions
            }
            dimension_mean = math.fsum(normalized_dimensions.values()) / len(
                expected_dimensions
            )
            if not math.isclose(vbench5, dimension_mean, rel_tol=0.0, abs_tol=1e-12):
                raise OracleRecordError(
                    f"step {step}.vbench5 does not equal the float64 mean of dimensions; "
                    f"scalar={vbench5}, recomputed={dimension_mean}"
                )
        else:
            normalized_dimensions = {
                name: _quality(dimensions[name], f"step {step}.dimensions.{name}")
                for name in expected_dimensions
                if isinstance(dimensions, dict) and name in dimensions
            }
        normalized_candidates.append(
            {
                "step": step,
                "vbench5": vbench5,
                "latency_seconds": latency,
                "dimensions": normalized_dimensions,
                "latency_source": str(candidate.get("latency_source", "unknown")),
                "diagnostics": candidate.get("diagnostics", {}),
            }
        )

    scoring_provenance = record.get("scoring_provenance")
    if require_provenance:
        if not isinstance(scoring_provenance, dict):
            raise OracleRecordError("scoring_provenance must be a mapping")
        if scoring_provenance.get("schema") != RECORD_PROVENANCE_SCHEMA:
            raise OracleRecordError(
                f"scoring_provenance.schema must be {RECORD_PROVENANCE_SCHEMA!r}"
            )
        if scoring_provenance.get("quality_dimensions") != expected_dimensions:
            raise OracleRecordError(
                "scoring_provenance.quality_dimensions does not match formal VBench-5"
            )
        diagnostic_dimensions = scoring_provenance.get("diagnostic_dimensions", [])
        if not isinstance(diagnostic_dimensions, list) or len(
            set(diagnostic_dimensions)
        ) != len(diagnostic_dimensions):
            raise OracleRecordError(
                "scoring_provenance.diagnostic_dimensions must be a unique list"
            )
        if (
            scoring_provenance.get("quality_aggregation")
            != "arithmetic_mean_raw_vbench5_float64"
        ):
            raise OracleRecordError(
                "unsupported scoring_provenance quality aggregation"
            )
        vbench = scoring_provenance.get("vbench")
        if not isinstance(vbench, dict):
            raise OracleRecordError("scoring_provenance.vbench must be a mapping")
        commit = str(vbench.get("git_commit", "")).lower()
        if re.fullmatch(r"^[0-9a-f]{40}$", commit) is None:
            raise OracleRecordError("scoring_provenance.vbench.git_commit is invalid")
        _sha256(
            vbench.get("evaluate_py_sha256"),
            "scoring_provenance.vbench.evaluate_py_sha256",
        )
        if bool(vbench.get("tracked_dirty")):
            raise OracleRecordError(
                "formal scoring provenance must use a clean VBench checkout"
            )

        cases = scoring_provenance.get("cases")
        if not isinstance(cases, dict):
            raise OracleRecordError("scoring_provenance.cases must be a mapping")
        expected_cases = {"native_hr", *(f"step{step}" for step in expected_steps)}
        if set(cases) != expected_cases:
            raise OracleRecordError(
                "scoring_provenance case coverage mismatch; "
                f"missing={sorted(expected_cases - set(cases))}, "
                f"extra={sorted(set(cases) - expected_cases)}"
            )
        for case_name, case in cases.items():
            if not isinstance(case, dict):
                raise OracleRecordError(
                    f"scoring_provenance.cases.{case_name} must be a mapping"
                )
            for field in ("request_sha256", "result_sha256", "full_info_sha256"):
                _sha256(
                    case.get(field),
                    f"scoring_provenance.cases.{case_name}.{field}",
                )
            if not str(case.get("run_manifest_path", "")).strip():
                raise OracleRecordError(
                    f"scoring_provenance.cases.{case_name}.run_manifest_path is empty"
                )

        diagnostic_cases = scoring_provenance.get("diagnostic_cases", {})
        expected_diagnostic_cases = expected_cases if diagnostic_dimensions else set()
        if (
            not isinstance(diagnostic_cases, dict)
            or set(diagnostic_cases) != expected_diagnostic_cases
        ):
            raise OracleRecordError(
                "scoring_provenance diagnostic case coverage mismatch; "
                f"missing={sorted(expected_diagnostic_cases - set(diagnostic_cases or {}))}, "
                f"extra={sorted(set(diagnostic_cases or {}) - expected_diagnostic_cases)}"
            )
        for case_name, case in diagnostic_cases.items():
            if not isinstance(case, dict):
                raise OracleRecordError(
                    f"scoring_provenance.diagnostic_cases.{case_name} must be a mapping"
                )
            for field in ("request_sha256", "result_sha256", "full_info_sha256"):
                _sha256(
                    case.get(field),
                    f"scoring_provenance.diagnostic_cases.{case_name}.{field}",
                )
            if not str(case.get("run_manifest_path", "")).strip():
                raise OracleRecordError(
                    f"scoring_provenance.diagnostic_cases.{case_name}.run_manifest_path "
                    "is empty"
                )

        if str(record.get("native_latency_source")) != "warm_pipeline_seconds":
            raise OracleRecordError(
                "native_latency_source must be warm_pipeline_seconds for formal data"
            )
        allowed_candidate_latency_sources = {
            "warm_pipeline_seconds",
            "estimated_warm_pipeline_seconds",
        }
        for candidate in normalized_candidates:
            if candidate["latency_source"] not in allowed_candidate_latency_sources:
                raise OracleRecordError(
                    f"step {candidate['step']}.latency_source is not traceable: "
                    f"{candidate['latency_source']!r}"
                )
        native_diagnostics = record.get("native_diagnostics", {})
        if not isinstance(native_diagnostics, dict) or set(native_diagnostics) != set(
            diagnostic_dimensions
        ):
            raise OracleRecordError(
                "native_diagnostics does not match provenance diagnostic dimensions"
            )
        normalized_native_diagnostics = {
            name: _diagnostic(
                native_diagnostics[name], f"native_diagnostics.{name}", name
            )
            for name in diagnostic_dimensions
        }
        for candidate in normalized_candidates:
            raw_diagnostics = candidate["diagnostics"]
            if not isinstance(raw_diagnostics, dict) or set(raw_diagnostics) != set(
                diagnostic_dimensions
            ):
                raise OracleRecordError(
                    f"step {candidate['step']}.diagnostics does not match provenance "
                    "diagnostic dimensions"
                )
            candidate["diagnostics"] = {
                name: _diagnostic(
                    raw_diagnostics[name],
                    f"step {candidate['step']}.diagnostics.{name}",
                    name,
                )
                for name in diagnostic_dimensions
            }
    else:
        normalized_native_diagnostics = record.get("native_diagnostics", {})

    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_text": prompt_text,
        "native_vbench5": native_vbench5,
        "native_latency_seconds": native_latency,
        "native_dimensions": normalized_native_dimensions,
        "native_diagnostics": normalized_native_diagnostics,
        "native_latency_source": str(record.get("native_latency_source", "unknown")),
        "candidates": normalized_candidates,
        "scoring_provenance": scoring_provenance,
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
    seed_policy: str = "fixed",
    require_dimensions: bool = False,
    require_native_dimensions: bool | None = None,
    require_provenance: bool = False,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Build one prompt-level sample by averaging utility across seeds."""
    if not records_by_prompt:
        raise ValueError("No oracle records were provided")

    expected_base_seeds = (
        {int(seed) for seed in expected_seeds} if expected_seeds is not None else None
    )
    if expected_base_seeds is not None and not expected_base_seeds:
        raise ValueError("expected_seeds cannot be empty")
    if seed_policy not in {"fixed", "prompt_offset", "count_only"}:
        raise ValueError(f"Unsupported seed_policy: {seed_policy}")
    observed_seed_count: int | None = None
    prompt_samples: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    for prompt_id, raw_records in sorted(records_by_prompt.items()):
        normalized_records = []
        for raw_record in raw_records:
            try:
                normalized_records.append(
                    validate_scored_record(
                        raw_record,
                        candidate_steps=candidate_steps,
                        require_dimensions=require_dimensions,
                        require_native_dimensions=require_native_dimensions,
                        require_provenance=require_provenance,
                    )
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
        if expected_base_seeds is None:
            if observed_seed_count is None:
                observed_seed_count = len(seed_set)
            expected_for_prompt = None
        elif seed_policy == "prompt_offset":
            expected_for_prompt = {
                base_seed + prompt_id for base_seed in expected_base_seeds
            }
        elif seed_policy == "fixed":
            expected_for_prompt = expected_base_seeds
        else:
            expected_for_prompt = None
            observed_seed_count = len(expected_base_seeds)

        required_count = observed_seed_count or (
            len(expected_base_seeds)
            if expected_base_seeds is not None
            else len(seed_set)
        )
        if expected_for_prompt is not None and seed_set != expected_for_prompt:
            errors.append(
                f"prompt {prompt_id}: seed coverage {sorted(seed_set)} does not match "
                f"expected {sorted(expected_for_prompt)} under {seed_policy} policy"
            )
            continue
        if expected_for_prompt is None and len(seed_set) != required_count:
            errors.append(
                f"prompt {prompt_id}: expected {required_count} unique seeds, got {sorted(seed_set)}"
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
        available_dimensions = [
            dimension
            for dimension in QUALITY5_DIMENSIONS
            if all(
                dimension in candidate.get("dimensions", {})
                for record in normalized_records
                for candidate in record["candidates"]
            )
        ]
        dimension_values = {
            dimension: np.asarray(
                [
                    [
                        candidate["dimensions"][dimension]
                        for candidate in record["candidates"]
                    ]
                    for record in normalized_records
                ],
                dtype=np.float32,
            ).mean(axis=0)
            for dimension in available_dimensions
        }
        native_dimension_values = {
            dimension: float(
                np.mean(
                    [
                        record.get("native_dimensions", {}).get(dimension)
                        for record in normalized_records
                    ]
                )
            )
            for dimension in available_dimensions
            if all(
                dimension in record.get("native_dimensions", {})
                for record in normalized_records
            )
        }
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
            "dimensions": dimension_values,
            "native_dimensions": native_dimension_values,
            "latencies": latencies.mean(axis=0),
            "native_latency_seconds": float(native_latencies.mean()),
            "seed_oracle_utility": float(utilities.max(axis=1).mean()),
        }

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise ValueError(f"Oracle record coverage check failed:\n{preview}{suffix}")
    if not prompt_samples:
        raise ValueError("No valid prompt-level oracle samples were produced")
    return prompt_samples, sorted(expected_base_seeds or [])
