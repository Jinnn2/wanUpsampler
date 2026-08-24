#!/usr/bin/env python3
"""Bootstrap the locked confirmation test by prompt and compare it to best fixed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "policy_regret",
    "realized_utility",
    "realized_vbench5",
    "realized_latency_sec",
    "speedup_vs_native",
    "step_abs_error",
    "top1_correct",
    "top3_correct",
    "realized_subject_consistency",
    "realized_background_consistency",
    "realized_motion_smoothness",
    "realized_aesthetic_quality",
    "realized_imaging_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(
    values: list[float], samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "router_benchmark_summary.json"
    predictions_path = run_dir / "router_test_predictions.csv"
    guard_path = run_dir / "test_access_guard.json"
    if not guard_path.is_file():
        raise FileNotFoundError("Missing confirmation test_access_guard.json")
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if not guard.get("completed_at_utc"):
        raise ValueError("Confirmation test-access guard is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("evaluation_stage") != "confirmation":
        raise ValueError(
            "Bootstrap confirmation requires evaluation_stage=confirmation"
        )
    if summary.get("evaluation_split") != "test" or not summary.get("test_accessed"):
        raise ValueError("Confirmation summary does not record one test evaluation")
    meta = summary.get("meta", {})
    if not meta.get("formal_evidence") or not meta.get("measured_latency_only"):
        raise ValueError(
            "Confirmation bootstrap requires formal measured-latency evidence"
        )

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    if not rows or {row["split"] for row in rows} != {"test"}:
        raise ValueError("Test prediction file is empty or contains another split")
    for row in rows:
        row["prompt_id"] = int(row["prompt_id"])
        for metric in METRICS:
            row[metric] = float(row[metric])

    rng = np.random.default_rng(args.bootstrap_seed)
    method_rows: list[dict[str, Any]] = []
    for method in sorted({str(row["Method"]) for row in rows}):
        selected = [row for row in rows if row["Method"] == method]
        if len({row["prompt_id"] for row in selected}) != len(selected):
            raise ValueError(f"Duplicate prompt rows for method {method}")
        for metric in METRICS:
            point, low, high = bootstrap_mean(
                [row[metric] for row in selected], args.bootstrap_samples, rng
            )
            method_rows.append(
                {
                    "Method": method,
                    "method_role": selected[0]["method_role"],
                    "model_type": selected[0]["model_type"],
                    "metric": metric,
                    "mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "prompt_count": len(selected),
                }
            )

    learned = [row for row in rows if row["method_role"] == "learned"]
    baseline = [row for row in rows if row["method_role"] == "best_fixed"]
    if len({row["model_type"] for row in learned}) != 1:
        raise ValueError("Confirmation must contain exactly one learned architecture")
    baseline_by_prompt = {row["prompt_id"]: row for row in baseline}
    directions = {
        "policy_regret": "baseline_minus_learned",
        "realized_utility": "learned_minus_baseline",
        "realized_vbench5": "learned_minus_baseline",
        "realized_latency_sec": "baseline_minus_learned",
        "speedup_vs_native": "learned_minus_baseline",
        "realized_subject_consistency": "learned_minus_baseline",
        "realized_background_consistency": "learned_minus_baseline",
        "realized_motion_smoothness": "learned_minus_baseline",
        "realized_aesthetic_quality": "learned_minus_baseline",
        "realized_imaging_quality": "learned_minus_baseline",
    }
    paired_rows = []
    for metric, direction in directions.items():
        deltas = []
        for row in learned:
            baseline_row = baseline_by_prompt.get(row["prompt_id"])
            if baseline_row is None:
                raise ValueError(
                    f"Missing best-fixed pair for prompt {row['prompt_id']}"
                )
            if direction == "baseline_minus_learned":
                deltas.append(baseline_row[metric] - row[metric])
            else:
                deltas.append(row[metric] - baseline_row[metric])
        point, low, high = bootstrap_mean(deltas, args.bootstrap_samples, rng)
        paired_rows.append(
            {
                "model_type": learned[0]["model_type"],
                "metric": metric,
                "positive_means": "learned_better",
                "mean_delta": point,
                "ci95_low": low,
                "ci95_high": high,
                "prompt_count": len(deltas),
            }
        )

    for filename, output_rows in (
        ("confirmation_test_intervals.csv", method_rows),
        ("confirmation_test_paired_deltas.csv", paired_rows),
    ):
        with (run_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    report = {
        "schema": "router_confirmation_bootstrap_v1",
        "bootstrap_unit": "prompt",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "formal_evidence": True,
        "measured_latency_only": True,
        "test_accessed_by_training_run": True,
        "predictions_sha256": sha256_file(predictions_path),
        "summary_sha256": sha256_file(summary_path),
        "test_access_guard_sha256": sha256_file(guard_path),
        "artifacts": {
            "intervals": "confirmation_test_intervals.csv",
            "paired_deltas": "confirmation_test_paired_deltas.csv",
        },
    }
    (run_dir / "confirmation_bootstrap_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
