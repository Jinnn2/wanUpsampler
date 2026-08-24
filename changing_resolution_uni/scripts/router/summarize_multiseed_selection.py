#!/usr/bin/env python3
"""Aggregate validation-only router runs and lock one architecture without test access."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
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
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-dir", default=None)
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


def mean_ci(
    values_by_prompt: dict[int, list[float]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    prompt_values = np.asarray(
        [np.mean(values_by_prompt[prompt]) for prompt in sorted(values_by_prompt)],
        dtype=np.float64,
    )
    if prompt_values.size == 0:
        raise ValueError("Cannot bootstrap an empty prompt set")
    draw_indices = rng.integers(
        0, prompt_values.size, size=(samples, prompt_values.size)
    )
    draws = prompt_values[draw_indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(prompt_values.mean()), float(low), float(high)


def load_runs(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_paths = sorted(
        runs_root.glob("seed_*/router_validation_predictions.csv"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if not prediction_paths:
        raise FileNotFoundError(
            f"No seed_*/router_validation_predictions.csv found under {runs_root}"
        )
    all_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    expected_signature: tuple[Any, ...] | None = None
    for prediction_path in prediction_paths:
        summary_path = prediction_path.parent / "router_validation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("evaluation_stage") != "selection":
            raise ValueError(f"Not a selection-stage run: {summary_path}")
        if summary.get("evaluation_split") != "validation" or summary.get(
            "test_accessed"
        ):
            raise ValueError(f"Selection run accessed test data: {summary_path}")
        meta = summary.get("meta", {})
        signature = (
            summary.get("primary_lambda"),
            meta.get("split_seed"),
            meta.get("quality_profile"),
            meta.get("latency_profile"),
            tuple(meta.get("candidate_steps", [])),
        )
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(
                f"Selection runs do not share lambda/split/evidence protocol: {summary_path}"
            )
        with prediction_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or {row["split"] for row in rows} != {"validation"}:
            raise ValueError(f"Invalid validation predictions: {prediction_path}")
        run_id = prediction_path.parent.name
        for row in rows:
            item = dict(row)
            item["run_id"] = run_id
            for metric in METRICS:
                item[metric] = float(item[metric])
            item["prompt_id"] = int(item["prompt_id"])
            all_rows.append(item)
        run_meta.append(
            {
                "run_id": run_id,
                "train_seed": meta.get("train_seed"),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "predictions_path": str(prediction_path),
                "predictions_sha256": sha256_file(prediction_path),
            }
        )
    return all_rows, run_meta


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs_root / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, run_meta = load_runs(runs_root)
    rng = np.random.default_rng(args.bootstrap_seed)

    learned_rows = [row for row in rows if row["method_role"] == "learned"]
    model_types = sorted({str(row["model_type"]) for row in learned_rows})
    run_ids = sorted({str(row["run_id"]) for row in rows})
    if not model_types:
        raise ValueError("No learned models found in validation predictions")

    interval_rows: list[dict[str, Any]] = []
    regret_by_model: dict[str, float] = {}
    for model_type in model_types:
        model_rows = [row for row in learned_rows if row["model_type"] == model_type]
        methods = {str(row["Method"]) for row in model_rows}
        if len(methods) != 1:
            raise ValueError(f"Inconsistent method labels for {model_type}: {methods}")
        prompt_coverage = {
            run_id: {row["prompt_id"] for row in model_rows if row["run_id"] == run_id}
            for run_id in run_ids
        }
        if len({tuple(sorted(value)) for value in prompt_coverage.values()}) != 1:
            raise ValueError(f"Prompt coverage differs across seeds for {model_type}")
        for metric in METRICS:
            by_prompt: dict[int, list[float]] = defaultdict(list)
            by_run: dict[str, list[float]] = defaultdict(list)
            for row in model_rows:
                by_prompt[row["prompt_id"]].append(row[metric])
                by_run[row["run_id"]].append(row[metric])
            point, low, high = mean_ci(
                by_prompt,
                samples=args.bootstrap_samples,
                rng=rng,
            )
            run_means = [float(np.mean(by_run[run_id])) for run_id in run_ids]
            interval_rows.append(
                {
                    "model_type": model_type,
                    "Method": next(iter(methods)),
                    "metric": metric,
                    "mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "train_seed_std": float(np.std(run_means)),
                    "run_count": len(run_ids),
                    "prompt_count": len(by_prompt),
                }
            )
            if metric == "policy_regret":
                regret_by_model[model_type] = point

    baseline_rows = [row for row in rows if row["method_role"] == "best_fixed"]
    baseline_by_key = {
        (str(row["run_id"]), int(row["prompt_id"])): row for row in baseline_rows
    }
    paired_rows: list[dict[str, Any]] = []
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
    for model_type in model_types:
        model_rows = [row for row in learned_rows if row["model_type"] == model_type]
        for metric, direction in directions.items():
            by_prompt: dict[int, list[float]] = defaultdict(list)
            for row in model_rows:
                key = (str(row["run_id"]), int(row["prompt_id"]))
                if key not in baseline_by_key:
                    raise ValueError(f"Missing paired best-fixed row for {key}")
                baseline = baseline_by_key[key]
                if direction == "baseline_minus_learned":
                    delta = baseline[metric] - row[metric]
                else:
                    delta = row[metric] - baseline[metric]
                by_prompt[row["prompt_id"]].append(float(delta))
            point, low, high = mean_ci(
                by_prompt,
                samples=args.bootstrap_samples,
                rng=rng,
            )
            paired_rows.append(
                {
                    "model_type": model_type,
                    "metric": metric,
                    "positive_means": "learned_better",
                    "mean_delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "run_count": len(run_ids),
                    "prompt_count": len(by_prompt),
                }
            )

    selected_model = min(regret_by_model, key=regret_by_model.get)
    first_summary = json.loads(
        Path(run_meta[0]["summary_path"]).read_text(encoding="utf-8")
    )
    selection = {
        "schema": "router_architecture_selection_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_rule": "minimum mean validation policy_regret across fixed-split training seeds",
        "selected_model_type": selected_model,
        "selected_validation_policy_regret": regret_by_model[selected_model],
        "primary_lambda": first_summary["primary_lambda"],
        "split_seed": first_summary["meta"]["split_seed"],
        "test_accessed": False,
        "run_count": len(run_ids),
        "train_seeds": [item["train_seed"] for item in run_meta],
        "bootstrap": {
            "unit": "prompt_after_averaging_training_seeds",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "inputs": run_meta,
        "artifacts": {
            "metric_intervals": "multiseed_validation_intervals.csv",
            "paired_best_fixed_deltas": "multiseed_paired_deltas.csv",
        },
    }

    for filename, output_rows in (
        ("multiseed_validation_intervals.csv", interval_rows),
        ("multiseed_paired_deltas.csv", paired_rows),
    ):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    (out_dir / "architecture_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
