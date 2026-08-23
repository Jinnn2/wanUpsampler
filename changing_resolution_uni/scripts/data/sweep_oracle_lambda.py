#!/usr/bin/env python3
"""Sweep oracle utility lambda and report prompt-level timestep distributions."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    OracleRecordError,
    validate_scored_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute prompt-level oracle labels over an inclusive lambda grid. "
            "Stored optimal_step fields are deliberately ignored."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lambda-min", type=str, default="0.001")
    parser.add_argument("--lambda-max", type=str, default="0.100")
    parser.add_argument("--lambda-step", type=str, default="0.001")
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=0.001,
        help="Count labels whose best-vs-second-best mean-utility margin is at most this value.",
    )
    return parser.parse_args()


def inclusive_decimal_grid(start: str, stop: str, step: str) -> list[float]:
    first = Decimal(start)
    last = Decimal(stop)
    stride = Decimal(step)
    if stride <= 0:
        raise ValueError("lambda-step must be positive")
    if first < 0 or last < first:
        raise ValueError("Require 0 <= lambda-min <= lambda-max")
    span = last - first
    if span % stride != 0:
        raise ValueError(
            "lambda range must be exactly divisible by lambda-step so both endpoints are included"
        )
    return [float(first + index * stride) for index in range(int(span / stride) + 1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_seed_set(
    prompt_id: int, base_seeds: Iterable[int], seed_policy: str
) -> set[int]:
    base = {int(seed) for seed in base_seeds}
    if seed_policy == "prompt_offset":
        return {seed + prompt_id for seed in base}
    if seed_policy == "fixed":
        return base
    raise ValueError(f"Unsupported seed_policy: {seed_policy}")


def load_prompt_arrays(dataset_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Strict lambda sweep requires dataset manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("is_complete") is not True:
        raise ValueError(f"Dataset manifest is not complete: {manifest_path}")

    candidate_steps = [int(step) for step in manifest.get("candidate_steps", FORMAL_STEPS)]
    if candidate_steps != FORMAL_STEPS:
        raise ValueError(
            f"Expected formal candidate steps {FORMAL_STEPS}; got {candidate_steps}"
        )
    base_seeds = [
        int(seed)
        for seed in manifest.get(
            "expected_base_seeds", manifest.get("expected_seeds", [])
        )
    ]
    if not base_seeds:
        raise ValueError("Dataset manifest must declare expected_base_seeds or expected_seeds")
    seed_policy = str(manifest.get("seed_policy", "fixed"))

    records_dir = dataset_dir / "records"
    record_names = manifest.get("record_files")
    if not isinstance(record_names, list) or not record_names:
        raise ValueError("Dataset manifest must contain a non-empty record_files list")

    records_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    latency_source_counts: Counter[str] = Counter()
    errors: list[str] = []
    seen_keys: set[tuple[int, int]] = set()
    for name in record_names:
        path = records_dir / str(name)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            normalized = validate_scored_record(
                raw,
                candidate_steps=candidate_steps,
                require_dimensions=False,
            )
            key = (int(normalized["prompt_id"]), int(normalized["seed"]))
            if key in seen_keys:
                raise ValueError(f"duplicate prompt/seed key {key}")
            seen_keys.add(key)
            records_by_prompt[key[0]].append(normalized)
            latency_source_counts.update(
                str(candidate["latency_source"])
                for candidate in normalized["candidates"]
            )
        except (OSError, json.JSONDecodeError, OracleRecordError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise ValueError(f"Failed to load strict oracle records:\n{preview}{suffix}")

    expected_prompts = int(manifest.get("expected_prompts", len(records_by_prompt)))
    if len(records_by_prompt) != expected_prompts:
        raise ValueError(
            f"Prompt coverage mismatch: expected {expected_prompts}, got {len(records_by_prompt)}"
        )

    prompt_arrays: dict[int, dict[str, Any]] = {}
    for prompt_id, records in sorted(records_by_prompt.items()):
        observed_seeds = {int(record["seed"]) for record in records}
        expected_seeds = _expected_seed_set(prompt_id, base_seeds, seed_policy)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"prompt {prompt_id}: seeds {sorted(observed_seeds)} != "
                f"expected {sorted(expected_seeds)} under {seed_policy}"
            )
        prompt_texts = {str(record["prompt_text"]) for record in records}
        if len(prompt_texts) != 1:
            raise ValueError(f"prompt {prompt_id}: prompt text differs across seeds")
        ordered = sorted(records, key=lambda record: int(record["seed"]))
        qualities = np.asarray(
            [
                [float(candidate["vbench5"]) for candidate in record["candidates"]]
                for record in ordered
            ],
            dtype=np.float64,
        )
        latencies = np.asarray(
            [
                [float(candidate["latency_seconds"]) for candidate in record["candidates"]]
                for record in ordered
            ],
            dtype=np.float64,
        )
        native_latencies = np.asarray(
            [float(record["native_latency_seconds"]) for record in ordered],
            dtype=np.float64,
        )
        prompt_arrays[prompt_id] = {
            "qualities": qualities,
            "normalized_latencies": latencies / native_latencies[:, None],
            "seed_count": len(ordered),
        }

    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "record_count": len(seen_keys),
        "prompt_count": len(prompt_arrays),
        "candidate_steps": candidate_steps,
        "base_seeds": sorted(base_seeds),
        "seed_policy": seed_policy,
        "latency_source_counts": dict(sorted(latency_source_counts.items())),
    }
    return prompt_arrays, metadata


def sweep_distributions(
    prompt_arrays: dict[int, dict[str, Any]],
    candidate_steps: list[int],
    lambdas: list[float],
    near_tie_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not prompt_arrays:
        raise ValueError("No prompt arrays to sweep")
    if near_tie_threshold < 0:
        raise ValueError("near-tie-threshold must be non-negative")

    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    previous_labels: dict[int, int] | None = None
    prompt_count = len(prompt_arrays)
    num_steps = len(candidate_steps)

    for lam in lambdas:
        histogram: Counter[int] = Counter()
        margins: list[float] = []
        seed_agreements: list[float] = []
        unanimous: list[bool] = []
        labels: dict[int, int] = {}

        for prompt_id, arrays in prompt_arrays.items():
            seed_utilities = arrays["qualities"] - lam * arrays["normalized_latencies"]
            mean_utility = seed_utilities.mean(axis=0)
            winner_idx = int(np.argmax(mean_utility))
            winner_step = int(candidate_steps[winner_idx])
            labels[prompt_id] = winner_step
            histogram[winner_step] += 1

            sorted_utility = np.sort(mean_utility)
            margin = float(sorted_utility[-1] - sorted_utility[-2])
            margins.append(margin)
            seed_winners = np.argmax(seed_utilities, axis=1)
            seed_agreements.append(float(np.mean(seed_winners == winner_idx)))
            unanimous.append(bool(np.all(seed_winners == winner_idx)))

        counts = np.asarray([histogram[step] for step in candidate_steps], dtype=np.float64)
        probabilities = counts / prompt_count
        nonzero = probabilities[probabilities > 0]
        entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))
        normalized_entropy = entropy_bits / math.log2(num_steps) if num_steps > 1 else 0.0
        endpoint_count = histogram[candidate_steps[0]] + histogram[candidate_steps[-1]]
        changes = (
            0
            if previous_labels is None
            else sum(labels[prompt_id] != previous_labels[prompt_id] for prompt_id in labels)
        )

        summary_rows.append(
            {
                "lambda": lam,
                "prompt_count": prompt_count,
                "step30_count": histogram[candidate_steps[0]],
                "step30_fraction": histogram[candidate_steps[0]] / prompt_count,
                "step50_count": histogram[candidate_steps[-1]],
                "step50_fraction": histogram[candidate_steps[-1]] / prompt_count,
                "endpoint_fraction": endpoint_count / prompt_count,
                "active_step_count": int(np.count_nonzero(counts)),
                "entropy_bits": entropy_bits,
                "normalized_entropy": normalized_entropy,
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "p10_margin": float(np.quantile(margins, 0.10)),
                "near_tie_fraction": float(
                    np.mean(np.asarray(margins) <= near_tie_threshold)
                ),
                "mean_seed_agreement": float(np.mean(seed_agreements)),
                "unanimous_seed_fraction": float(np.mean(unanimous)),
                "label_changes_from_previous_lambda": changes,
            }
        )
        for step in candidate_steps:
            distribution_rows.append(
                {
                    "lambda": lam,
                    "step": step,
                    "count": histogram[step],
                    "fraction": histogram[step] / prompt_count,
                }
            )
        previous_labels = labels

    return summary_rows, distribution_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    lambdas = inclusive_decimal_grid(args.lambda_min, args.lambda_max, args.lambda_step)
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_arrays, metadata = load_prompt_arrays(dataset_dir)
    summary_rows, distribution_rows = sweep_distributions(
        prompt_arrays,
        candidate_steps=metadata["candidate_steps"],
        lambdas=lambdas,
        near_tie_threshold=args.near_tie_threshold,
    )

    summary_csv = out_dir / "lambda_sweep_summary.csv"
    distribution_csv = out_dir / "lambda_step_distribution.csv"
    report_json = out_dir / "lambda_sweep_report.json"
    write_csv(summary_csv, summary_rows)
    write_csv(distribution_csv, distribution_rows)
    report_json.write_text(
        json.dumps(
            {
                "schema": "prompt_oracle_lambda_sweep_v1",
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "dataset": metadata,
                "lambda_grid": {
                    "min": lambdas[0],
                    "max": lambdas[-1],
                    "step": float(Decimal(args.lambda_step)),
                    "count": len(lambdas),
                },
                "near_tie_threshold": args.near_tie_threshold,
                "summary": summary_rows,
                "distribution": distribution_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selected = {0.001, 0.005, 0.01, 0.02, 0.05, 0.1}
    by_lambda: dict[float, dict[int, int]] = defaultdict(dict)
    for row in distribution_rows:
        by_lambda[round(float(row["lambda"]), 6)][int(row["step"])] = int(row["count"])
    summary_by_lambda = {
        round(float(row["lambda"]), 6): row for row in summary_rows
    }
    print(
        f"Validated {metadata['prompt_count']} prompts / {metadata['record_count']} records; "
        f"swept {len(lambdas)} lambda values."
    )
    print(f"Latency sources: {metadata['latency_source_counts']}")
    for lam in sorted(selected & set(summary_by_lambda)):
        row = summary_by_lambda[lam]
        histogram = {step: count for step, count in by_lambda[lam].items() if count}
        print(
            f"lambda={lam:.3f} endpoint={row['endpoint_fraction']:.1%} "
            f"near_tie={row['near_tie_fraction']:.1%} "
            f"seed_agreement={row['mean_seed_agreement']:.1%} hist={histogram}"
        )
    print(f"Summary CSV: {summary_csv}")
    print(f"Distribution CSV: {distribution_csv}")
    print(f"Report JSON: {report_json}")


if __name__ == "__main__":
    main()
