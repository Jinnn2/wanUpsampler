#!/usr/bin/env python3
"""Summarize matched B4-residual and soft-margin runs on candidate steps 40-50."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import summarize_variable_lambda_runs as summary
except ImportError:
    import summarize_variable_lambda_runs as summary


REPORT_SCHEMA = "variable_lambda_steps40_50_overall_selection_v1"
EXPECTED_STEPS = tuple(range(40, 51))
EXPECTED_RESIDUAL_MODELS = {
    "b4_offline",
    "b4_residual_prompt",
    "b4_residual_state",
}
EXPECTED_SOFT_MODELS = {
    "b4_offline",
    "soft_margin_control",
    "soft_margin_state",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-runs-root", required=True)
    parser.add_argument("--soft-runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2029)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    return args


def prediction_key(row: dict[str, Any]) -> tuple[str, int, int, float]:
    return (
        str(row["run_id"]),
        int(row["prompt_id"]),
        int(row["seed"]),
        float(row["lambda"]),
    )


def model_rows(rows: list[dict[str, Any]], model_type: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["model_type"]) == model_type]


def validate_suite_metadata(
    metadata: list[dict[str, Any]], expected_models: set[str]
) -> dict[str, Any]:
    normalized = None
    for item in metadata:
        run_summary = json.loads(Path(item["summary_path"]).read_text(encoding="utf-8"))
        observed_models = set(run_summary["model_types"])
        if observed_models != expected_models:
            raise ValueError(
                f"Unexpected models in {item['summary_path']}: {sorted(observed_models)}"
            )
        if tuple(run_summary.get("candidate_steps", [])) != EXPECTED_STEPS:
            raise ValueError(
                f"Run does not use exactly steps 40-50: {item['summary_path']}"
            )
        training = run_summary["training"]
        current = {
            "dataset_manifest_sha256": run_summary["dataset_manifest_sha256"],
            "source_candidate_steps": tuple(run_summary["source_candidate_steps"]),
            "candidate_steps": tuple(run_summary["candidate_steps"]),
            "train_lambdas": tuple(run_summary["train_lambdas"]),
            "eval_lambdas": tuple(run_summary["eval_lambdas"]),
            "primary_lambda": float(run_summary["primary_lambda"]),
            "harm_epsilon": float(run_summary["harm_epsilon"]),
            "feature_groups": tuple(run_summary["feature_groups"]),
            "selected_feature_count": int(run_summary["selected_feature_count"]),
            "train_prompts": int(run_summary["train_prompts"]),
            "validation_prompts": int(run_summary["validation_prompts"]),
            "latency_profile_sha256": run_summary["latency_profile"]["sha256"],
            "epochs": int(training["epochs"]),
            "batch_size": int(
                training.get("batch_size", training.get("batch_size_trajectories"))
            ),
            "lr": float(training["lr"]),
            "weight_decay": float(training["weight_decay"]),
            "dropout": float(training["dropout"]),
            "b4_temperature": float(training["b4_temperature"]),
            "b4_emd_weight": float(training["b4_emd_weight"]),
        }
        if normalized is None:
            normalized = current
        elif normalized != current:
            raise ValueError(f"Suite protocol differs: {item['summary_path']}")
    if normalized is None:
        raise ValueError("Suite metadata is empty")
    return normalized


def validate_frozen_b4_equivalence(
    residual_rows: list[dict[str, Any]], soft_rows: list[dict[str, Any]]
) -> None:
    residual = {
        prediction_key(row): row for row in model_rows(residual_rows, "b4_offline")
    }
    soft = {prediction_key(row): row for row in model_rows(soft_rows, "b4_offline")}
    if residual.keys() != soft.keys():
        raise ValueError("B4 prediction coverage differs across suites")
    exact_fields = ("chosen_step", "oracle_step", "best_fixed_step")
    numeric_fields = (
        "policy_regret",
        "best_fixed_regret",
        "realized_utility",
        "oracle_utility",
        "realized_vbench5",
        "realized_latency_sec",
        "speedup_vs_native",
        "harmful_stop",
    )
    for key in residual:
        left = residual[key]
        right = soft[key]
        if any(str(left[field]) != str(right[field]) for field in exact_fields):
            raise ValueError(f"B4 discrete prediction differs across suites: {key}")
        if any(
            not np.isclose(
                float(left[field]),
                float(right[field]),
                rtol=0.0,
                atol=1e-12,
            )
            for field in numeric_fields
        ):
            raise ValueError(f"B4 metric differs across suites: {key}")


def validate_coverage(rows: list[dict[str, Any]], model_types: list[str]) -> None:
    coverage = {
        model_type: {
            prediction_key(row) for row in rows if str(row["model_type"]) == model_type
        }
        for model_type in model_types
    }
    if not coverage or len({frozenset(keys) for keys in coverage.values()}) != 1:
        raise ValueError("Prediction coverage differs across overall model types")


def interval_rows(
    rows: list[dict[str, Any]],
    model_types: list[str],
    lambdas: list[float],
    run_ids: list[str],
    samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    per_lambda: list[dict[str, Any]] = []
    macro: list[dict[str, Any]] = []
    macro_regret: dict[str, float] = {}
    for model_type in model_types:
        selected = model_rows(rows, model_type)
        for lambda_value in lambdas:
            lambda_rows = [
                row for row in selected if float(row["lambda"]) == lambda_value
            ]
            for metric in summary.METRIC_DIRECTIONS:
                by_prompt: dict[int, list[float]] = defaultdict(list)
                by_run: dict[str, list[float]] = defaultdict(list)
                for row in lambda_rows:
                    by_prompt[int(row["prompt_id"])].append(float(row[metric]))
                    by_run[str(row["run_id"])].append(float(row[metric]))
                point, low, high = summary.bootstrap_mean(by_prompt, samples, rng)
                per_lambda.append(
                    {
                        "model_type": model_type,
                        "lambda": lambda_value,
                        "metric": metric,
                        "mean": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "train_seed_std": float(
                            np.std([np.mean(by_run[run_id]) for run_id in run_ids])
                        ),
                    }
                )
        for metric in summary.METRIC_DIRECTIONS:
            by_prompt: dict[int, list[float]] = defaultdict(list)
            by_run: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                by_prompt[int(row["prompt_id"])].append(float(row[metric]))
                by_run[str(row["run_id"])].append(float(row[metric]))
            point, low, high = summary.bootstrap_mean(by_prompt, samples, rng)
            macro.append(
                {
                    "model_type": model_type,
                    "metric": metric,
                    "mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "train_seed_std": float(
                        np.std([np.mean(by_run[run_id]) for run_id in run_ids])
                    ),
                }
            )
            if metric == "policy_regret":
                macro_regret[model_type] = point
    return per_lambda, macro, macro_regret


def main() -> None:
    args = parse_args()
    residual_root = Path(args.residual_runs_root).resolve()
    soft_root = Path(args.soft_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Overall output already exists: {out_dir}")
    residual_rows, residual_metadata = summary.load_runs(residual_root)
    soft_rows, soft_metadata = summary.load_runs(soft_root)
    residual_signature = validate_suite_metadata(
        residual_metadata, EXPECTED_RESIDUAL_MODELS
    )
    soft_signature = validate_suite_metadata(soft_metadata, EXPECTED_SOFT_MODELS)
    if residual_signature != soft_signature:
        raise ValueError("Residual and soft-margin suite protocols differ")
    residual_run_ids = {str(row["run_id"]) for row in residual_rows}
    soft_run_ids = {str(row["run_id"]) for row in soft_rows}
    if residual_run_ids != soft_run_ids:
        raise ValueError("Training seed coverage differs across suites")
    validate_frozen_b4_equivalence(residual_rows, soft_rows)
    rows = residual_rows + [
        row for row in soft_rows if str(row["model_type"]) != "b4_offline"
    ]
    model_types = sorted({str(row["model_type"]) for row in rows})
    expected_overall = sorted(
        EXPECTED_RESIDUAL_MODELS | (EXPECTED_SOFT_MODELS - {"b4_offline"})
    )
    if model_types != expected_overall:
        raise ValueError(f"Unexpected overall models: {model_types}")
    validate_coverage(rows, model_types)
    lambdas = sorted({float(row["lambda"]) for row in rows})
    run_ids = sorted(residual_run_ids)
    rng = np.random.default_rng(args.bootstrap_seed)
    per_lambda, macro, macro_regret = interval_rows(
        rows,
        model_types,
        lambdas,
        run_ids,
        args.bootstrap_samples,
        rng,
    )
    paired_b4 = summary.paired_rows_against_reference(
        rows,
        model_types,
        lambdas,
        run_ids,
        "b4_offline",
        args.bootstrap_samples,
        rng,
    )
    matched_pairs = []
    for reference, candidate in (
        ("b4_residual_prompt", "b4_residual_state"),
        ("soft_margin_control", "soft_margin_state"),
    ):
        candidates = summary.paired_rows_against_reference(
            rows,
            model_types,
            lambdas,
            run_ids,
            reference,
            args.bootstrap_samples,
            rng,
        )
        matched_pairs.extend(
            row for row in candidates if row["candidate_model"] == candidate
        )
    out_dir.mkdir(parents=True)
    artifacts = {
        "overall_per_lambda_intervals": (
            "overall_per_lambda_intervals.csv",
            per_lambda,
        ),
        "overall_macro_intervals": ("overall_macro_intervals.csv", macro),
        "overall_paired_b4_deltas": (
            "overall_paired_b4_deltas.csv",
            paired_b4,
        ),
        "overall_matched_control_deltas": (
            "overall_matched_control_deltas.csv",
            matched_pairs,
        ),
    }
    for _, (filename, artifact_rows) in artifacts.items():
        summary.write_csv(out_dir / filename, artifact_rows)
    selected_model = min(macro_regret, key=macro_regret.get)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "evaluation_protocol": summary.EVALUATION_PROTOCOL,
        "selection_only": True,
        "formal_evidence": False,
        "test_accessed": False,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": summary.sha256_file(Path(__file__).resolve()),
        },
        "candidate_steps": list(EXPECTED_STEPS),
        "excluded_source_steps": [30, 35],
        "model_types": model_types,
        "b4_predictions_identical_across_suites": True,
        "selected_model_type": selected_model,
        "selected_macro_policy_regret": macro_regret[selected_model],
        "selection_rule": "minimum validation macro policy regret across lambdas and train seeds",
        "protocol": residual_signature,
        "bootstrap": {
            "unit": "prompt_after_averaging_generation_and_training_seeds_and_lambdas",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "inputs": {
            "residual_runs_root": str(residual_root),
            "soft_runs_root": str(soft_root),
            "residual_run_summaries": residual_metadata,
            "soft_run_summaries": soft_metadata,
        },
        "artifacts": {
            name: {
                "path": filename,
                "sha256": summary.sha256_file(out_dir / filename),
            }
            for name, (filename, _) in artifacts.items()
        },
    }
    report_path = out_dir / "overall_selection.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
