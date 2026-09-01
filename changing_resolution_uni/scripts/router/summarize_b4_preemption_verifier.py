#!/usr/bin/env python3
"""Summarize multi-seed B4-3 sparse preemption verifier selection runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import summarize_variable_lambda_runs as common
except ImportError:
    import summarize_variable_lambda_runs as common


REPORT_SCHEMA = "b4_sparse_preemption_verifier_selection_v1"
EXPECTED_RUN_SCHEMA = "b4_sparse_preemption_verifier_run_v1"
RAW_METRICS = (
    "policy_regret",
    "realized_utility",
    "realized_vbench5",
    "realized_latency_sec",
    "speedup_vs_native",
    "harm_vs_b4",
    "decision_changed",
    "step_delta_vs_b4",
)
PAIRED_DIRECTIONS = {
    "policy_regret": "lower",
    "realized_utility": "higher",
    "realized_vbench5": "higher",
    "realized_latency_sec": "lower",
    "speedup_vs_native": "higher",
    "harm_vs_b4": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2037)
    parser.add_argument("--max-harm-rate", type=float, default=0.02)
    parser.add_argument("--minimum-positive-train-seeds", type=int, default=4)
    parser.add_argument("--minimum-nonnegative-lambdas", type=int, default=8)
    parser.add_argument("--minimum-worst-lambda-gain", type=float, default=-0.0002)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.minimum_positive_train_seeds < 1:
        parser.error("bootstrap samples and positive-seed requirement must be positive")
    if args.max_harm_rate < 0 or not 1 <= args.minimum_nonnegative_lambdas <= 10:
        parser.error("invalid harm-rate or lambda gate")
    return args


def protocol_signature(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary["schema"],
        "evaluation_protocol": summary["evaluation_protocol"],
        "evaluation_split": summary["evaluation_split"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "train_lambdas": summary["train_lambdas"],
        "eval_lambdas": summary["eval_lambdas"],
        "primary_lambda": summary["primary_lambda"],
        "harm_epsilon": summary["harm_epsilon"],
        "risk_thresholds": summary["risk_thresholds"],
        "checkpoint_risk_threshold": summary["checkpoint_risk_threshold"],
        "candidate_steps": summary["candidate_steps"],
        "radius": summary["radius"],
        "base_state_feature_names": summary["base_state_feature_names"],
        "sparse_signal_names": summary["sparse_signal_names"],
        "state_normalization": summary["state_normalization"],
        "train_prompts": summary["train_prompts"],
        "validation_prompts": summary["validation_prompts"],
        "training": summary["training"],
        "validation_shuffle": summary["validation_shuffle"],
        "cost_profile": summary["cost_profile"],
        "latency_profile_sha256": summary["latency_profile"]["sha256"],
        "b4_ensemble_size": summary["b4_ensemble_size"],
    }


def load_runs(
    runs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_paths = sorted(
        runs_root.glob("seed_*/run_summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if len(summary_paths) != 5:
        raise ValueError(f"Expected exactly five seed runs, found {len(summary_paths)}")
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    signature = None
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema") != EXPECTED_RUN_SCHEMA
            or summary.get("evaluation_split") != "validation"
            or summary.get("test_accessed")
        ):
            raise ValueError(f"Invalid validation-only verifier run: {summary_path}")
        current = protocol_signature(summary)
        if signature is None:
            signature = current
        elif signature != current:
            raise ValueError(f"Verifier run protocol differs: {summary_path}")
        artifacts = summary["artifacts"]
        predictions_meta = artifacts["predictions"]
        predictions_path = summary_path.parent / predictions_meta["path"]
        if common.sha256_file(predictions_path) != predictions_meta["sha256"]:
            raise ValueError(f"Prediction SHA256 mismatch: {predictions_path}")
        for group in ("checkpoints", "training_histories"):
            for item in artifacts[group].values():
                path = summary_path.parent / item["path"]
                if common.sha256_file(path) != item["sha256"]:
                    raise ValueError(f"Artifact SHA256 mismatch: {path}")
        run_id = summary_path.parent.name
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        for raw in run_rows:
            row = dict(raw)
            row["run_id"] = run_id
            row["prompt_id"] = int(row["prompt_id"])
            row["seed"] = int(row["seed"])
            row["lambda"] = float(row["lambda"])
            if row["risk_threshold"] != "baseline":
                row["risk_threshold"] = float(row["risk_threshold"])
            for metric in RAW_METRICS:
                row[metric] = float(row[metric])
            rows.append(row)
        metadata.append(
            {
                "run_id": run_id,
                "train_seed": int(summary["train_seed"]),
                "summary_path": str(summary_path),
                "summary_sha256": common.sha256_file(summary_path),
                "best_epochs": {
                    item["model_type"]: int(item["best_epoch"])
                    for item in summary["models"]
                },
            }
        )
    if signature is None:
        raise ValueError("No verifier runs found")
    return rows, metadata, signature


def variant_rows(
    rows: list[dict[str, Any]], model_type: str, threshold: str | float
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["model_type"] == model_type and row["risk_threshold"] == threshold
    ]


def validate_coverage(rows: list[dict[str, Any]], thresholds: list[float]) -> None:
    variants = [("b4_offline", "baseline")]
    variants.extend(
        (model_type, threshold)
        for model_type in (
            "preemption_control",
            "preemption_state",
            "preemption_state_shuffled",
        )
        for threshold in thresholds
    )
    coverage = []
    for model_type, threshold in variants:
        selected = variant_rows(rows, model_type, threshold)
        keys = {
            (row["run_id"], row["prompt_id"], row["seed"], row["lambda"])
            for row in selected
        }
        if len(keys) != len(selected):
            raise ValueError(f"Duplicate predictions for {model_type}@{threshold}")
        coverage.append(keys)
    if not coverage or len({frozenset(value) for value in coverage}) != 1:
        raise ValueError("Prediction coverage differs across verifier variants")


def bootstrap_raw(
    rows: list[dict[str, Any]],
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    by_prompt: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(float(row[metric]))
    return common.bootstrap_mean(by_prompt, samples, rng)


def pair_key(row: dict[str, Any]) -> tuple[str, int, int, float]:
    return row["run_id"], row["prompt_id"], row["seed"], row["lambda"]


def paired_interval(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric: str,
    direction: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    reference_by_key = {pair_key(row): row for row in reference}
    by_prompt: dict[int, list[float]] = defaultdict(list)
    for row in candidate:
        key = pair_key(row)
        baseline = reference_by_key.get(key)
        if baseline is None:
            raise ValueError(f"Missing paired reference row: {key}")
        delta = (
            float(row[metric]) - float(baseline[metric])
            if direction == "higher"
            else float(baseline[metric]) - float(row[metric])
        )
        by_prompt[row["prompt_id"]].append(delta)
    return common.bootstrap_mean(by_prompt, samples, rng)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs_root / "selection"
    if out_dir.exists():
        raise FileExistsError(f"Selection output already exists: {out_dir}")
    rows, run_metadata, signature = load_runs(runs_root)
    thresholds = [float(value) for value in signature["risk_thresholds"]]
    validate_coverage(rows, thresholds)
    out_dir.mkdir(parents=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    variants = [("b4_offline", "baseline")]
    variants.extend(
        (model_type, threshold)
        for model_type in (
            "preemption_control",
            "preemption_state",
            "preemption_state_shuffled",
        )
        for threshold in thresholds
    )
    macro_rows = []
    per_lambda_rows = []
    for model_type, threshold in variants:
        selected = variant_rows(rows, model_type, threshold)
        for metric in RAW_METRICS:
            point, low, high = bootstrap_raw(
                selected, metric, args.bootstrap_samples, rng
            )
            by_run = defaultdict(list)
            for row in selected:
                by_run[row["run_id"]].append(float(row[metric]))
            macro_rows.append(
                {
                    "model_type": model_type,
                    "risk_threshold": threshold,
                    "metric": metric,
                    "mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "train_seed_std": float(
                        np.std([np.mean(values) for values in by_run.values()])
                    ),
                }
            )
        for lambda_value in signature["eval_lambdas"]:
            lambda_rows = [row for row in selected if row["lambda"] == lambda_value]
            for metric in RAW_METRICS:
                point, low, high = bootstrap_raw(
                    lambda_rows, metric, args.bootstrap_samples, rng
                )
                per_lambda_rows.append(
                    {
                        "model_type": model_type,
                        "risk_threshold": threshold,
                        "lambda": lambda_value,
                        "metric": metric,
                        "mean": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

    b4_rows = variant_rows(rows, "b4_offline", "baseline")
    paired_rows = []
    comparisons = []
    for threshold in thresholds:
        comparisons.extend(
            [
                (
                    "b4_offline",
                    "baseline",
                    "preemption_control",
                    threshold,
                    "control_vs_b4",
                ),
                (
                    "b4_offline",
                    "baseline",
                    "preemption_state",
                    threshold,
                    "state_vs_b4",
                ),
                (
                    "preemption_control",
                    threshold,
                    "preemption_state",
                    threshold,
                    "state_vs_control",
                ),
                (
                    "preemption_state_shuffled",
                    threshold,
                    "preemption_state",
                    threshold,
                    "state_vs_shuffled",
                ),
            ]
        )
    for ref_model, ref_threshold, cand_model, cand_threshold, comparison in comparisons:
        reference = variant_rows(rows, ref_model, ref_threshold)
        candidate = variant_rows(rows, cand_model, cand_threshold)
        for lambda_value in [*signature["eval_lambdas"], None]:
            ref_subset = (
                reference
                if lambda_value is None
                else [row for row in reference if row["lambda"] == lambda_value]
            )
            cand_subset = (
                candidate
                if lambda_value is None
                else [row for row in candidate if row["lambda"] == lambda_value]
            )
            for metric, direction in PAIRED_DIRECTIONS.items():
                point, low, high = paired_interval(
                    ref_subset,
                    cand_subset,
                    metric,
                    direction,
                    args.bootstrap_samples,
                    rng,
                )
                paired_rows.append(
                    {
                        "comparison": comparison,
                        "reference_model": ref_model,
                        "candidate_model": cand_model,
                        "risk_threshold": cand_threshold,
                        "lambda": "macro" if lambda_value is None else lambda_value,
                        "metric": metric,
                        "positive_means": "candidate_better",
                        "mean_delta": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

    per_seed_rows = []
    for threshold in thresholds:
        state = variant_rows(rows, "preemption_state", threshold)
        b4_by_key = {pair_key(row): row for row in b4_rows}
        for metadata in run_metadata:
            run_rows = [row for row in state if row["run_id"] == metadata["run_id"]]
            gains = [
                row["realized_utility"] - b4_by_key[pair_key(row)]["realized_utility"]
                for row in run_rows
            ]
            per_seed_rows.append(
                {
                    "risk_threshold": threshold,
                    "train_seed": metadata["train_seed"],
                    "mean_utility_gain_vs_b4": float(np.mean(gains)),
                    "best_epoch": metadata["best_epochs"]["preemption_state"],
                }
            )

    def paired_lookup(
        comparison: str, threshold: float, lambda_value: str, metric: str
    ):
        matches = [
            row
            for row in paired_rows
            if row["comparison"] == comparison
            and row["risk_threshold"] == threshold
            and str(row["lambda"]) == str(lambda_value)
            and row["metric"] == metric
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing unique paired result: {comparison}@{threshold}")
        return matches[0]

    gate_rows = []
    for threshold in thresholds:
        state_vs_b4 = paired_lookup(
            "state_vs_b4", threshold, "macro", "realized_utility"
        )
        state_vs_control = paired_lookup(
            "state_vs_control", threshold, "macro", "realized_utility"
        )
        state_vs_shuffle = paired_lookup(
            "state_vs_shuffled", threshold, "macro", "realized_utility"
        )
        state_rows = variant_rows(rows, "preemption_state", threshold)
        harm_rate = float(np.mean([row["harm_vs_b4"] for row in state_rows]))
        seed_rows = [row for row in per_seed_rows if row["risk_threshold"] == threshold]
        positive_seed_count = sum(
            row["mean_utility_gain_vs_b4"] > 0 for row in seed_rows
        )
        positive_epoch_count = sum(row["best_epoch"] > 0 for row in seed_rows)
        lambda_gains = [
            paired_lookup("state_vs_b4", threshold, str(value), "realized_utility")[
                "mean_delta"
            ]
            for value in signature["eval_lambdas"]
        ]
        nonnegative_lambda_count = sum(value >= 0 for value in lambda_gains)
        criteria = {
            "state_vs_b4_ci_positive": state_vs_b4["ci95_low"] > 0,
            "state_vs_control_ci_positive": state_vs_control["ci95_low"] > 0,
            "state_vs_shuffle_ci_positive": state_vs_shuffle["ci95_low"] > 0,
            "harm_rate_within_limit": harm_rate <= args.max_harm_rate,
            "train_seed_consistency": positive_seed_count
            >= args.minimum_positive_train_seeds,
            "nonzero_checkpoint_count": positive_epoch_count
            >= args.minimum_positive_train_seeds,
            "lambda_consistency": nonnegative_lambda_count
            >= args.minimum_nonnegative_lambdas
            and min(lambda_gains) >= args.minimum_worst_lambda_gain,
        }
        gate_rows.append(
            {
                "risk_threshold": threshold,
                "mean_utility_gain_vs_b4": state_vs_b4["mean_delta"],
                "utility_gain_vs_b4_ci95_low": state_vs_b4["ci95_low"],
                "utility_gain_vs_b4_ci95_high": state_vs_b4["ci95_high"],
                "utility_gain_vs_control": state_vs_control["mean_delta"],
                "utility_gain_vs_control_ci95_low": state_vs_control["ci95_low"],
                "utility_gain_vs_shuffled": state_vs_shuffle["mean_delta"],
                "utility_gain_vs_shuffled_ci95_low": state_vs_shuffle["ci95_low"],
                "harm_vs_b4_rate": harm_rate,
                "positive_train_seed_count": positive_seed_count,
                "nonzero_checkpoint_count": positive_epoch_count,
                "nonnegative_lambda_count": nonnegative_lambda_count,
                "worst_lambda_utility_gain": min(lambda_gains),
                **criteria,
                "passes_all_gates": all(criteria.values()),
            }
        )

    eligible = [row for row in gate_rows if row["passes_all_gates"]]
    selected_gate = (
        max(eligible, key=lambda row: row["mean_utility_gain_vs_b4"])
        if eligible
        else None
    )
    selected_model = "preemption_state" if selected_gate else "b4_offline"
    selected_threshold: str | float = (
        selected_gate["risk_threshold"] if selected_gate else "baseline"
    )

    artifacts = {
        "threshold_macro_summary": ("threshold_macro_summary.csv", macro_rows),
        "threshold_per_lambda": ("threshold_per_lambda.csv", per_lambda_rows),
        "paired_deltas": ("paired_deltas.csv", paired_rows),
        "per_train_seed": ("per_train_seed.csv", per_seed_rows),
        "selection_gate": ("selection_gate.csv", gate_rows),
    }
    for _, (filename, artifact_rows) in artifacts.items():
        common.write_csv(out_dir / filename, artifact_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "selection_only": True,
        "formal_evidence": False,
        "test_accessed": False,
        "selected_model_type": selected_model,
        "selected_risk_threshold": selected_threshold,
        "verifier_passed_all_gates": selected_gate is not None,
        "selection_rule": (
            "highest macro utility gain among sparse-state thresholds passing every "
            "pre-registered safety, control, shuffle, seed, epoch, and lambda gate; "
            "otherwise frozen B4"
        ),
        "gate_protocol": {
            "max_harm_rate": args.max_harm_rate,
            "minimum_positive_train_seeds": args.minimum_positive_train_seeds,
            "minimum_nonnegative_lambdas": args.minimum_nonnegative_lambdas,
            "minimum_worst_lambda_gain": args.minimum_worst_lambda_gain,
            "paired_ci_requirement": "ci95_low_strictly_greater_than_zero",
        },
        "protocol": signature,
        "bootstrap": {
            "unit": "prompt_after_averaging_generation_and_training_seeds_and_lambdas",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "inputs": run_metadata,
        "artifacts": {
            name: {
                "path": filename,
                "sha256": common.sha256_file(out_dir / filename),
            }
            for name, (filename, _) in artifacts.items()
        },
    }
    report_path = out_dir / "architecture_selection.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
