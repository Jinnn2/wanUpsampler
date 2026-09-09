#!/usr/bin/env python3
"""Bootstrap validation-only continuous-budget runs across training seeds."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


INTERVAL_METRICS = (
    "policy_regret",
    "realized_utility",
    "realized_vbench5",
    "realized_latency_sec",
    "speedup_vs_native",
    "target_budget",
    "chosen_budget",
    "budget_abs_error",
    "realized_subject_consistency",
    "realized_background_consistency",
    "realized_motion_smoothness",
    "realized_aesthetic_quality",
    "realized_imaging_quality",
)
PAIRED_METRICS = tuple(
    metric
    for metric in INTERVAL_METRICS
    if metric not in {"target_budget", "chosen_budget"}
)
LOWER_IS_BETTER = {"policy_regret", "realized_latency_sec", "budget_abs_error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    return args


def mean_ci(
    values_by_prompt: dict[int, list[float]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(
        [np.mean(values_by_prompt[key]) for key in sorted(values_by_prompt)],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty prompt set")
    draws = values[rng.integers(0, values.size, size=(samples, values.size))].mean(
        axis=1
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_runs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_paths = sorted(
        root.glob("seed_*/continuous_budget_validation_summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if not summary_paths:
        raise FileNotFoundError(f"No continuous-budget seed runs found under {root}")
    rows: list[dict[str, Any]] = []
    run_meta = []
    signature: tuple[Any, ...] | None = None
    expected_prompts: tuple[int, ...] | None = None
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "continuous_prompt_budget_validation_v2":
            raise ValueError(
                f"Run does not use the utility-aware v2 protocol: {summary_path}"
            )
        if summary.get("evaluation_split") != "validation" or summary.get(
            "test_accessed"
        ):
            raise ValueError(f"Run is not validation-only: {summary_path}")
        meta = summary.get("meta", {})
        current_signature = (
            summary.get("primary_lambda"),
            summary.get("target_type"),
            summary.get("loss_type"),
            json.dumps(summary.get("loss"), sort_keys=True),
            summary.get("architecture"),
            tuple(summary.get("candidate_steps", [])),
            tuple(summary.get("budget_grid", [])),
            meta.get("split_seed"),
            meta.get("quality_profile"),
            json.dumps(meta.get("latency_profile"), sort_keys=True),
        )
        if signature is None:
            signature = current_signature
        elif signature != current_signature:
            raise ValueError(f"Seed runs use incompatible protocols: {summary_path}")
        predictions_path = summary_path.parent / summary["artifacts"]["predictions"]
        with predictions_path.open(newline="", encoding="utf-8") as handle:
            current_rows = list(csv.DictReader(handle))
        if not current_rows or {row["split"] for row in current_rows} != {"validation"}:
            raise ValueError(f"Invalid validation predictions: {predictions_path}")
        prompt_ids = tuple(sorted({int(row["prompt_id"]) for row in current_rows}))
        if expected_prompts is None:
            expected_prompts = prompt_ids
        elif expected_prompts != prompt_ids:
            raise ValueError("Validation prompt coverage differs across training seeds")
        run_id = summary_path.parent.name
        for row in current_rows:
            item: dict[str, Any] = dict(row)
            item["run_id"] = run_id
            item["prompt_id"] = int(item["prompt_id"])
            for metric in INTERVAL_METRICS:
                item[metric] = float(item[metric])
            rows.append(item)
        run_meta.append(
            {
                "run_id": run_id,
                "train_seed": meta.get("train_seed"),
                "summary": str(summary_path),
                "predictions": str(predictions_path),
            }
        )
    return rows, run_meta


def metric_delta(candidate: float, reference: float, metric: str) -> float:
    if metric in LOWER_IS_BETTER:
        return reference - candidate
    return candidate - reference


def paired_intervals(
    rows: list[dict[str, Any]],
    *,
    reference_model: str,
    candidate_models: list[str],
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    reference = {
        (str(row["run_id"]), int(row["prompt_id"])): row
        for row in rows
        if row["model_type"] == reference_model
    }
    output = []
    for candidate_model in candidate_models:
        selected = [row for row in rows if row["model_type"] == candidate_model]
        for metric in PAIRED_METRICS:
            by_prompt: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                key = (str(row["run_id"]), int(row["prompt_id"]))
                if key not in reference:
                    raise ValueError(f"Missing paired {reference_model} row for {key}")
                by_prompt[int(row["prompt_id"])].append(
                    metric_delta(
                        float(row[metric]), float(reference[key][metric]), metric
                    )
                )
            point, low, high = mean_ci(by_prompt, samples=samples, rng=rng)
            output.append(
                {
                    "reference_model": reference_model,
                    "candidate_model": candidate_model,
                    "metric": metric,
                    "positive_means": "candidate_better",
                    "mean_delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "prompt_count": len(by_prompt),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "selection"
    summary_path = out_dir / "continuous_budget_prior_selection.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, run_meta = load_runs(root)
    rng = np.random.default_rng(args.bootstrap_seed)
    model_types = sorted({str(row["model_type"]) for row in rows})
    intervals = []
    for model_type in model_types:
        selected = [row for row in rows if row["model_type"] == model_type]
        method = {str(row["Method"]) for row in selected}
        if len(method) != 1:
            raise ValueError(f"Inconsistent method label for {model_type}: {method}")
        for metric in INTERVAL_METRICS:
            by_prompt: dict[int, list[float]] = defaultdict(list)
            by_run: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                by_prompt[int(row["prompt_id"])].append(float(row[metric]))
                by_run[str(row["run_id"])].append(float(row[metric]))
            point, low, high = mean_ci(
                by_prompt, samples=args.bootstrap_samples, rng=rng
            )
            intervals.append(
                {
                    "model_type": model_type,
                    "Method": next(iter(method)),
                    "metric": metric,
                    "mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "train_seed_std": float(
                        np.std([np.mean(values) for values in by_run.values()])
                    ),
                    "run_count": len(by_run),
                    "prompt_count": len(by_prompt),
                }
            )

    adaptive_models = [
        "b4_argmax",
        "b4_projected_nearest",
        "continuous_budget_nearest",
    ]
    vs_fixed = paired_intervals(
        rows,
        reference_model="best_fixed",
        candidate_models=adaptive_models,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    vs_b4 = paired_intervals(
        rows,
        reference_model="b4_argmax",
        candidate_models=["b4_projected_nearest", "continuous_budget_nearest"],
        samples=args.bootstrap_samples,
        rng=rng,
    )
    vs_matched_fixed = paired_intervals(
        rows,
        reference_model="matched_fixed_mixture",
        candidate_models=["continuous_budget_nearest"],
        samples=args.bootstrap_samples,
        rng=rng,
    )
    write_csv(out_dir / "validation_intervals.csv", intervals)
    write_csv(out_dir / "paired_vs_fixed.csv", vs_fixed)
    write_csv(out_dir / "paired_vs_b4.csv", vs_b4)
    write_csv(out_dir / "paired_vs_matched_fixed.csv", vs_matched_fixed)
    summary = {
        "schema": "continuous_prompt_budget_multiseed_selection_v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_scope": "validation_only_diagnostic",
        "test_accessed": False,
        "run_count": len(run_meta),
        "train_seeds": [item["train_seed"] for item in run_meta],
        "bootstrap": {
            "unit": "prompt_after_averaging_training_seeds",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "inputs": run_meta,
        "artifacts": {
            "intervals": "validation_intervals.csv",
            "paired_vs_fixed": "paired_vs_fixed.csv",
            "paired_vs_b4": "paired_vs_b4.csv",
            "paired_vs_matched_fixed": "paired_vs_matched_fixed.csv",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
