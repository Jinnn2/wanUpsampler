#!/usr/bin/env python3
"""Aggregate multi-seed variable-lambda validation runs with prompt bootstrap."""

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


METRIC_DIRECTIONS = {
    "policy_regret": "lower",
    "realized_utility": "higher",
    "realized_vbench5": "higher",
    "realized_latency_sec": "lower",
    "speedup_vs_native": "higher",
    "harmful_stop": "lower",
}
EVALUATION_PROTOCOL = "deterministic_eval_mode_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--reference-model", default="prompt_only")
    parser.add_argument(
        "--secondary-reference-model",
        default=None,
        help="Optional second paired baseline, such as b4_offline.",
    )
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
    values_by_prompt: dict[int, list[float]],
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(
        [np.mean(values_by_prompt[prompt]) for prompt in sorted(values_by_prompt)],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty prompt set")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_runs(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = sorted(
        runs_root.glob("seed_*/run_summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if len(summaries) < 3:
        raise ValueError(f"Need at least three completed seed runs under {runs_root}")
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    signature = None
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("evaluation_split") != "validation" or summary.get(
            "test_accessed"
        ):
            raise ValueError(f"Run is not validation-only: {summary_path}")
        if summary.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            raise ValueError(f"Run does not use {EVALUATION_PROTOCOL}: {summary_path}")
        decision_parameter = str(summary.get("decision_parameter", "risk_threshold"))
        decision_value = float(
            summary.get("risk_margin", summary.get("risk_threshold", 0.5))
        )
        current_signature = (
            str(summary["evaluation_protocol"]),
            tuple(summary["train_lambdas"]),
            tuple(summary["eval_lambdas"]),
            float(summary["primary_lambda"]),
            float(summary["harm_epsilon"]),
            decision_parameter,
            decision_value,
            tuple(summary["feature_groups"]),
            int(summary["selected_feature_count"]),
            str(summary["dataset_manifest_sha256"]),
            json.dumps(summary["training"], sort_keys=True),
            tuple(float(value) for value in summary["cost_profile"]),
            int(summary["train_prompts"]),
            int(summary["validation_prompts"]),
            json.dumps(summary["latency_profile"], sort_keys=True),
        )
        if signature is None:
            signature = current_signature
        elif signature != current_signature:
            raise ValueError(f"Run protocol differs: {summary_path}")
        predictions_path = summary_path.parent / summary["artifacts"]["predictions"]
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        if not run_rows:
            raise ValueError(f"Empty predictions: {predictions_path}")
        checkpoint_metadata = summary.get("artifacts", {}).get("checkpoints", {})
        if checkpoint_metadata:
            run_model_types = {str(row["model_type"]) for row in run_rows}
            if set(checkpoint_metadata) != run_model_types:
                raise ValueError(f"Checkpoint coverage differs: {summary_path}")
            for model_type, metadata_item in checkpoint_metadata.items():
                checkpoint_path = summary_path.parent / metadata_item["path"]
                if sha256_file(checkpoint_path) != metadata_item["sha256"]:
                    raise ValueError(
                        f"Checkpoint SHA256 mismatch for {model_type}: "
                        f"{checkpoint_path}"
                    )
        history_metadata = summary.get("artifacts", {}).get("training_histories", {})
        if set(history_metadata) != {str(row["model_type"]) for row in run_rows}:
            raise ValueError(f"Training-history coverage differs: {summary_path}")
        for model_type, metadata_item in history_metadata.items():
            history_path = summary_path.parent / metadata_item["path"]
            if sha256_file(history_path) != metadata_item["sha256"]:
                raise ValueError(
                    f"Training-history SHA256 mismatch for {model_type}: {history_path}"
                )
        run_id = summary_path.parent.name
        for raw in run_rows:
            row = dict(raw)
            row["run_id"] = run_id
            row["prompt_id"] = int(row["prompt_id"])
            row["seed"] = int(row["seed"])
            row["lambda"] = float(row["lambda"])
            for metric in METRIC_DIRECTIONS:
                row[metric] = float(row[metric])
            row["best_fixed_regret"] = float(row["best_fixed_regret"])
            rows.append(row)
        metadata.append(
            {
                "run_id": run_id,
                "train_seed": int(summary["train_seed"]),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "predictions_path": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "checkpoints": checkpoint_metadata,
                "training_histories": history_metadata,
                "latency_profile": summary["latency_profile"],
            }
        )
    return rows, metadata


def paired_rows_against_reference(
    rows: list[dict[str, Any]],
    model_types: list[str],
    lambdas: list[float],
    run_ids: list[str],
    reference_model: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    reference_by_key = {
        (
            str(row["run_id"]),
            int(row["prompt_id"]),
            int(row["seed"]),
            float(row["lambda"]),
        ): row
        for row in rows
        if row["model_type"] == reference_model
    }
    paired_rows = []
    for candidate_model in model_types:
        if candidate_model == reference_model:
            continue
        candidate_rows = [row for row in rows if row["model_type"] == candidate_model]
        for lambda_value in [*lambdas, None]:
            subset = (
                candidate_rows
                if lambda_value is None
                else [
                    row
                    for row in candidate_rows
                    if float(row["lambda"]) == lambda_value
                ]
            )
            for metric, direction in METRIC_DIRECTIONS.items():
                by_prompt: dict[int, list[float]] = defaultdict(list)
                for row in subset:
                    key = (
                        str(row["run_id"]),
                        int(row["prompt_id"]),
                        int(row["seed"]),
                        float(row["lambda"]),
                    )
                    reference = reference_by_key.get(key)
                    if reference is None:
                        raise ValueError(f"Missing reference prediction for {key}")
                    delta = (
                        float(reference[metric]) - float(row[metric])
                        if direction == "lower"
                        else float(row[metric]) - float(reference[metric])
                    )
                    by_prompt[row["prompt_id"]].append(delta)
                point, low, high = bootstrap_mean(by_prompt, bootstrap_samples, rng)
                paired_rows.append(
                    {
                        "reference_model": reference_model,
                        "candidate_model": candidate_model,
                        "lambda": "macro" if lambda_value is None else lambda_value,
                        "metric": metric,
                        "positive_means": "candidate_better",
                        "mean_delta": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "run_count": len(run_ids),
                        "prompt_count": len(by_prompt),
                    }
                )
    return paired_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs_root / "selection"
    if out_dir.exists():
        raise FileExistsError(f"Selection output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    rows, run_metadata = load_runs(runs_root)
    rng = np.random.default_rng(args.bootstrap_seed)
    model_types = sorted({str(row["model_type"]) for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    run_ids = sorted({str(row["run_id"]) for row in rows})
    if args.reference_model not in model_types:
        raise ValueError(
            f"Reference model {args.reference_model!r} not in {model_types}"
        )
    if (
        args.secondary_reference_model is not None
        and args.secondary_reference_model not in model_types
    ):
        raise ValueError(
            f"Secondary reference model {args.secondary_reference_model!r} "
            f"not in {model_types}"
        )
    coverage_by_model = {
        model_type: {
            (
                str(row["run_id"]),
                int(row["prompt_id"]),
                int(row["seed"]),
                float(row["lambda"]),
            )
            for row in rows
            if row["model_type"] == model_type
        }
        for model_type in model_types
    }
    if len({frozenset(coverage) for coverage in coverage_by_model.values()}) != 1:
        raise ValueError("Prediction coverage differs across model types")

    per_lambda_rows = []
    macro_rows = []
    macro_regret_by_model = {}
    for model_type in model_types:
        model_rows = [row for row in rows if row["model_type"] == model_type]
        expected_keys = {
            (run_id, prompt_id, seed, lambda_value)
            for run_id in run_ids
            for lambda_value in lambdas
            for prompt_id, seed in {
                (int(row["prompt_id"]), int(row["seed"])) for row in model_rows
            }
        }
        observed_keys = {
            (
                str(row["run_id"]),
                int(row["prompt_id"]),
                int(row["seed"]),
                float(row["lambda"]),
            )
            for row in model_rows
        }
        if observed_keys != expected_keys:
            raise ValueError(f"Coverage differs across runs/lambdas for {model_type}")
        for lambda_value in lambdas:
            lambda_rows = [
                row for row in model_rows if float(row["lambda"]) == lambda_value
            ]
            for metric in METRIC_DIRECTIONS:
                by_prompt: dict[int, list[float]] = defaultdict(list)
                by_run: dict[str, list[float]] = defaultdict(list)
                for row in lambda_rows:
                    by_prompt[row["prompt_id"]].append(float(row[metric]))
                    by_run[row["run_id"]].append(float(row[metric]))
                point, low, high = bootstrap_mean(
                    by_prompt, args.bootstrap_samples, rng
                )
                per_lambda_rows.append(
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
                        "run_count": len(run_ids),
                        "prompt_count": len(by_prompt),
                    }
                )
        for metric in METRIC_DIRECTIONS:
            by_prompt: dict[int, list[float]] = defaultdict(list)
            by_run: dict[str, list[float]] = defaultdict(list)
            for row in model_rows:
                by_prompt[row["prompt_id"]].append(float(row[metric]))
                by_run[row["run_id"]].append(float(row[metric]))
            point, low, high = bootstrap_mean(by_prompt, args.bootstrap_samples, rng)
            macro_rows.append(
                {
                    "model_type": model_type,
                    "metric": metric,
                    "mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "train_seed_std": float(
                        np.std([np.mean(by_run[run_id]) for run_id in run_ids])
                    ),
                    "run_count": len(run_ids),
                    "prompt_count": len(by_prompt),
                    "lambda_count": len(lambdas),
                }
            )
            if metric == "policy_regret":
                macro_regret_by_model[model_type] = point

    paired_rows = paired_rows_against_reference(
        rows,
        model_types,
        lambdas,
        run_ids,
        args.reference_model,
        args.bootstrap_samples,
        rng,
    )
    secondary_paired_rows = (
        paired_rows_against_reference(
            rows,
            model_types,
            lambdas,
            run_ids,
            args.secondary_reference_model,
            args.bootstrap_samples,
            rng,
        )
        if args.secondary_reference_model is not None
        else []
    )

    fixed_paired_rows = []
    for model_type in model_types:
        for lambda_value in [*lambdas, None]:
            subset = [row for row in rows if row["model_type"] == model_type]
            if lambda_value is not None:
                subset = [row for row in subset if float(row["lambda"]) == lambda_value]
            by_prompt: dict[int, list[float]] = defaultdict(list)
            for row in subset:
                by_prompt[row["prompt_id"]].append(
                    float(row["best_fixed_regret"]) - float(row["policy_regret"])
                )
            point, low, high = bootstrap_mean(by_prompt, args.bootstrap_samples, rng)
            fixed_paired_rows.append(
                {
                    "model_type": model_type,
                    "lambda": "macro" if lambda_value is None else lambda_value,
                    "metric": "policy_regret",
                    "positive_means": "model_better_than_train_selected_fixed",
                    "mean_delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "run_count": len(run_ids),
                    "prompt_count": len(by_prompt),
                }
            )

    write_csv(out_dir / "per_lambda_intervals.csv", per_lambda_rows)
    write_csv(out_dir / "macro_intervals.csv", macro_rows)
    write_csv(out_dir / "paired_reference_deltas.csv", paired_rows)
    if secondary_paired_rows:
        write_csv(
            out_dir / "paired_secondary_reference_deltas.csv",
            secondary_paired_rows,
        )
        if args.secondary_reference_model == "b4_offline":
            write_csv(out_dir / "paired_b4_deltas.csv", secondary_paired_rows)
    write_csv(out_dir / "paired_fixed_deltas.csv", fixed_paired_rows)
    selected_model = min(macro_regret_by_model, key=macro_regret_by_model.get)
    selection = {
        "schema": "variable_lambda_multiseed_selection_v3",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_rule": "minimum validation macro policy regret across lambdas and train seeds",
        "selected_model_type": selected_model,
        "selected_macro_policy_regret": macro_regret_by_model[selected_model],
        "reference_model": args.reference_model,
        "secondary_reference_model": args.secondary_reference_model,
        "eval_lambdas": lambdas,
        "test_accessed": False,
        "run_count": len(run_ids),
        "train_seeds": [item["train_seed"] for item in run_metadata],
        "latency_profile": run_metadata[0]["latency_profile"],
        "bootstrap": {
            "unit": "prompt_after_averaging_generation_and_training_seeds_and_lambdas",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "inputs": run_metadata,
        "artifacts": {
            "per_lambda_intervals": "per_lambda_intervals.csv",
            "macro_intervals": "macro_intervals.csv",
            "paired_reference_deltas": "paired_reference_deltas.csv",
            **(
                {
                    "paired_secondary_reference_deltas": (
                        "paired_secondary_reference_deltas.csv"
                    )
                }
                if secondary_paired_rows
                else {}
            ),
            **(
                {"paired_b4_deltas": "paired_b4_deltas.csv"}
                if secondary_paired_rows
                and args.secondary_reference_model == "b4_offline"
                else {}
            ),
            "paired_fixed_deltas": "paired_fixed_deltas.csv",
        },
    }
    (out_dir / "architecture_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
