#!/usr/bin/env python3
"""Analyze strict per-dimension oracle scores without changing router labels."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    QUALITY5_DIMENSIONS,
    OracleRecordError,
    validate_scored_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-dimension timestep sensitivity in strict oracle records."
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-6,
        help="Treat best-vs-second margin at or below this value as a tie.",
    )
    parser.add_argument(
        "--flat_tolerance",
        type=float,
        default=1e-3,
        help="Treat a prompt's max-minus-min metric range at or below this as flat.",
    )
    return parser.parse_args()


def expected_seed_set(
    prompt_id: int, base_seeds: list[int], seed_policy: str
) -> set[int]:
    if seed_policy == "prompt_offset":
        return {seed + prompt_id for seed in base_seeds}
    if seed_policy == "fixed":
        return set(base_seeds)
    raise ValueError(f"Unsupported seed_policy: {seed_policy}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_records(
    dataset_dir: Path,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("is_complete") is not True:
        raise ValueError(f"Dataset manifest is incomplete: {manifest_path}")
    if manifest.get("quality_profile") != "strict_vbench5_v1":
        raise ValueError("Dimension analysis requires quality_profile='strict_vbench5_v1'")
    if manifest.get("quality_dimensions") != QUALITY5_DIMENSIONS:
        raise ValueError("Dataset manifest does not declare canonical VBench-5 dimensions")

    diagnostic_dimensions = list(manifest.get("diagnostic_dimensions", []))
    records_dir = (dataset_dir / "records").resolve()
    record_names = manifest.get("record_files")
    if not isinstance(record_names, list) or not record_names:
        raise ValueError("Dataset manifest must contain record_files")
    record_hashes = manifest.get("record_sha256")
    if not isinstance(record_hashes, dict) or set(record_hashes) != {
        str(name) for name in record_names
    }:
        raise ValueError("Dataset manifest record_sha256 does not cover record_files")

    records_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    errors: list[str] = []
    for raw_name in record_names:
        path = (records_dir / str(raw_name)).resolve()
        try:
            if path.parent != records_dir:
                raise ValueError("record path escapes records directory")
            if sha256_file(path) != record_hashes[str(raw_name)]:
                raise ValueError("record SHA256 differs from dataset manifest")
            raw = json.loads(path.read_text(encoding="utf-8"))
            normalized = validate_scored_record(
                raw,
                candidate_steps=FORMAL_STEPS,
                require_dimensions=True,
                require_provenance=True,
            )
            provenance_diagnostics = normalized["scoring_provenance"].get(
                "diagnostic_dimensions", []
            )
            if provenance_diagnostics != diagnostic_dimensions:
                raise ValueError(
                    "record diagnostic dimensions differ from dataset manifest"
                )
            key = (int(normalized["prompt_id"]), int(normalized["seed"]))
            if key in seen:
                raise ValueError(f"duplicate prompt/seed {key}")
            seen.add(key)
            records_by_prompt[key[0]].append(normalized)
        except (OSError, json.JSONDecodeError, OracleRecordError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise ValueError(f"Strict dimension record loading failed:\n{preview}{suffix}")

    expected_prompts = int(manifest["expected_prompts"])
    if len(records_by_prompt) != expected_prompts:
        raise ValueError(
            f"Prompt coverage mismatch: expected {expected_prompts}, "
            f"got {len(records_by_prompt)}"
        )
    base_seeds = [int(seed) for seed in manifest["expected_base_seeds"]]
    seed_policy = str(manifest["seed_policy"])
    for prompt_id, records in records_by_prompt.items():
        observed = {int(record["seed"]) for record in records}
        expected = expected_seed_set(prompt_id, base_seeds, seed_policy)
        if observed != expected:
            raise ValueError(
                f"prompt {prompt_id}: seeds={sorted(observed)}, expected={sorted(expected)}"
            )
    return records_by_prompt, manifest


def metric_arrays(
    records_by_prompt: dict[int, list[dict[str, Any]]],
    metric: str,
    group: str,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_candidates = []
    prompt_native = []
    for records in records_by_prompt.values():
        ordered = sorted(records, key=lambda record: int(record["seed"]))
        if group == "quality":
            candidate = np.asarray(
                [
                    [item["dimensions"][metric] for item in record["candidates"]]
                    for record in ordered
                ],
                dtype=np.float64,
            )
            native = np.asarray(
                [record["native_dimensions"][metric] for record in ordered],
                dtype=np.float64,
            )
        else:
            candidate = np.asarray(
                [
                    [item["diagnostics"][metric] for item in record["candidates"]]
                    for record in ordered
                ],
                dtype=np.float64,
            )
            native = np.asarray(
                [record["native_diagnostics"][metric] for record in ordered],
                dtype=np.float64,
            )
        prompt_candidates.append(candidate.mean(axis=0))
        prompt_native.append(float(native.mean()))
    return np.stack(prompt_candidates), np.asarray(prompt_native, dtype=np.float64)


def entropy_bits(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def analyze(
    records_by_prompt: dict[int, list[dict[str, Any]]],
    manifest: dict[str, Any],
    *,
    tie_tolerance: float,
    flat_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_groups = [
        (name, "quality") for name in QUALITY5_DIMENSIONS
    ] + [
        (name, "diagnostic")
        for name in manifest.get("diagnostic_dimensions", [])
    ]
    step_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    steps = np.asarray(FORMAL_STEPS, dtype=np.float64)

    for metric, group in metric_groups:
        values, native = metric_arrays(records_by_prompt, metric, group)
        sorted_values = np.sort(values, axis=1)
        margins = sorted_values[:, -1] - sorted_values[:, -2]
        ties = margins <= tie_tolerance
        winners = np.argmax(values, axis=1)
        winner_counts = np.bincount(winners[~ties], minlength=len(FORMAL_STEPS))
        prompt_ranges = values.max(axis=1) - values.min(axis=1)
        adjacent_abs = np.abs(np.diff(values, axis=1))
        step_means = values.mean(axis=0)
        correlation = (
            float(np.corrcoef(steps, step_means)[0, 1])
            if float(np.std(step_means)) > 0.0
            else None
        )
        for index, step in enumerate(FORMAL_STEPS):
            step_rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "step": step,
                    "prompt_mean": float(step_means[index]),
                    "prompt_std": float(values[:, index].std()),
                    "mean_minus_native": float((values[:, index] - native).mean()),
                    "unique_winner_count": int(winner_counts[index]),
                    "unique_winner_fraction": float(
                        winner_counts[index] / values.shape[0]
                    ),
                }
            )
        endpoint_count = int(winner_counts[0] + winner_counts[-1])
        metric_rows.append(
            {
                "metric": metric,
                "group": group,
                "prompt_count": int(values.shape[0]),
                "native_mean": float(native.mean()),
                "mean_prompt_range": float(prompt_ranges.mean()),
                "median_prompt_range": float(np.median(prompt_ranges)),
                "flat_prompt_fraction": float(np.mean(prompt_ranges <= flat_tolerance)),
                "tie_fraction": float(ties.mean()),
                "mean_adjacent_abs_delta": float(adjacent_abs.mean()),
                "step50_minus_step30_mean": float(
                    (values[:, -1] - values[:, 0]).mean()
                ),
                "step50_minus_step49_mean": float(
                    (values[:, -1] - values[:, -2]).mean()
                ),
                "endpoint_unique_winner_fraction": float(
                    endpoint_count / values.shape[0]
                ),
                "unique_winner_entropy_bits": entropy_bits(winner_counts),
                "step_mean_correlation": correlation,
            }
        )
    return step_rows, metric_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.tie_tolerance) or args.tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    if not math.isfinite(args.flat_tolerance) or args.flat_tolerance < 0.0:
        raise ValueError("flat_tolerance must be finite and non-negative")
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    records_by_prompt, manifest = load_strict_records(dataset_dir)
    step_rows, metric_rows = analyze(
        records_by_prompt,
        manifest,
        tie_tolerance=args.tie_tolerance,
        flat_tolerance=args.flat_tolerance,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dimension_step_summary.csv", step_rows)
    write_csv(out_dir / "dimension_discriminability.csv", metric_rows)
    report = {
        "schema": "strict_oracle_dimension_analysis_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "quality_profile": manifest["quality_profile"],
        "prompt_count": len(records_by_prompt),
        "trajectory_count": sum(len(records) for records in records_by_prompt.values()),
        "candidate_steps": FORMAL_STEPS,
        "quality_dimensions": QUALITY5_DIMENSIONS,
        "diagnostic_dimensions": manifest.get("diagnostic_dimensions", []),
        "tie_tolerance": args.tie_tolerance,
        "flat_tolerance": args.flat_tolerance,
        "protocol_notes": {
            "overall_consistency": "General prompt-video ViCLIP similarity diagnostic.",
            "dynamic_degree": "Protocol-bound motion diagnostic; not a monotonic quality score.",
            "temporal_flickering": "Protocol-bound static-video diagnostic; interpret moving prompts cautiously.",
        },
        "metrics": metric_rows,
        "steps": step_rows,
    }
    (out_dir / "dimension_analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Strict dimension analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
