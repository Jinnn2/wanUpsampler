#!/usr/bin/env python3
"""Evaluate frozen train800 control routers on the existing OOD validation split."""

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
import torch

from changing_resolution_uni.scripts.router import train_variable_lambda_router as train


REPORT_SCHEMA = "train800_control_ood_diagnostic_v1"
HIGHER_IS_BETTER = {
    "realized_utility",
    "realized_vbench5",
    "speedup_vs_native",
    *(
        f"realized_{name}"
        for name in (
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
        )
    ),
}
LOWER_IS_BETTER = {
    "policy_regret",
    "realized_latency_sec",
    "harmful_stop",
}
METRICS = [
    "policy_regret",
    "realized_utility",
    "realized_vbench5",
    "realized_latency_sec",
    "speedup_vs_native",
    "harmful_stop",
    *(
        f"realized_{name}"
        for name in (
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
        )
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--ood-dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-runs-root", default=None)
    parser.add_argument("--base-seeds", type=int, nargs="+", default=[42, 100, 2024])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--eval-batch-trajectories", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.base_seeds = sorted(set(args.base_seeds))
    if not args.base_seeds:
        parser.error("base-seeds must not be empty")
    if args.bootstrap_samples < 1 or args.eval_batch_trajectories < 1:
        parser.error("bootstrap-samples and eval-batch-trajectories must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return payload


def bootstrap_values(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    if values.ndim != 1 or not values.size:
        raise ValueError("Bootstrap requires a non-empty vector")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def prompt_means(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[int(row["prompt_id"])].append(float(row[metric]))
    return np.asarray(
        [np.mean(values[prompt_id]) for prompt_id in sorted(values)],
        dtype=np.float64,
    )


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    model_type: str,
    lambda_value: float | None,
    base_seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    result = [row for row in rows if str(row["model_type"]) == model_type]
    if lambda_value is not None:
        result = [row for row in result if float(row["lambda"]) == lambda_value]
    if base_seeds is not None:
        result = [
            row
            for row in result
            if int(row["seed"]) - int(row["prompt_id"]) in base_seeds
        ]
    return result


def interval_rows(
    rows: list[dict[str, Any]],
    base_seeds: list[int],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected_base_seeds = set(base_seeds)
    model_types = sorted({str(row["model_type"]) for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    output = []
    for model_type in model_types:
        for lambda_value in [*lambdas, None]:
            subset = filter_rows(
                rows,
                model_type=model_type,
                lambda_value=lambda_value,
                base_seeds=selected_base_seeds,
            )
            if not subset:
                raise ValueError(
                    f"No OOD rows for model={model_type}, lambda={lambda_value}, "
                    f"base_seeds={base_seeds}"
                )
            for metric in METRICS:
                values = prompt_means(subset, metric)
                mean, low, high = bootstrap_values(values, bootstrap_samples, rng)
                by_run: dict[str, list[float]] = defaultdict(list)
                for row in subset:
                    by_run[str(row["run_id"])].append(float(row[metric]))
                output.append(
                    {
                        "model_type": model_type,
                        "lambda": "macro" if lambda_value is None else lambda_value,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "train_seed_std": float(
                            np.std([np.mean(values) for values in by_run.values()])
                        ),
                        "run_count": len(by_run),
                        "prompt_count": len(values),
                        "generation_base_seeds": " ".join(map(str, base_seeds)),
                        "lambda_count": len(lambdas) if lambda_value is None else 1,
                    }
                )
    return output


def paired_rows(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    candidate_label: str,
    reference_label: str,
    base_seeds: list[int],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected_base_seeds = set(base_seeds)
    model_types = sorted({str(row["model_type"]) for row in candidate_rows})
    lambdas = sorted({float(row["lambda"]) for row in candidate_rows})
    reference_by_key = {
        (
            str(row["run_id"]),
            str(row["model_type"]),
            int(row["prompt_id"]),
            int(row["seed"]),
            float(row["lambda"]),
        ): row
        for row in reference_rows
        if int(row["seed"]) - int(row["prompt_id"]) in selected_base_seeds
    }
    output = []
    for model_type in model_types:
        for lambda_value in [*lambdas, None]:
            subset = filter_rows(
                candidate_rows,
                model_type=model_type,
                lambda_value=lambda_value,
                base_seeds=selected_base_seeds,
            )
            for metric in METRICS:
                differences: dict[int, list[float]] = defaultdict(list)
                for row in subset:
                    key = (
                        str(row["run_id"]),
                        model_type,
                        int(row["prompt_id"]),
                        int(row["seed"]),
                        float(row["lambda"]),
                    )
                    reference = reference_by_key.get(key)
                    if reference is None:
                        raise ValueError(f"Missing paired reference row: {key}")
                    if metric in LOWER_IS_BETTER:
                        delta = float(reference[metric]) - float(row[metric])
                    else:
                        delta = float(row[metric]) - float(reference[metric])
                    differences[int(row["prompt_id"])].append(delta)
                values = np.asarray(
                    [
                        np.mean(differences[prompt_id])
                        for prompt_id in sorted(differences)
                    ],
                    dtype=np.float64,
                )
                mean, low, high = bootstrap_values(values, bootstrap_samples, rng)
                output.append(
                    {
                        "reference": reference_label,
                        "candidate": candidate_label,
                        "model_type": model_type,
                        "lambda": "macro" if lambda_value is None else lambda_value,
                        "metric": metric,
                        "positive_means": "candidate_better",
                        "mean_delta": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "prompt_count": len(values),
                        "generation_base_seeds": " ".join(map(str, base_seeds)),
                    }
                )
    return output


def paired_model_rows(
    rows: list[dict[str, Any]],
    *,
    reference_models: list[str],
    base_seeds: list[int],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected_base_seeds = set(base_seeds)
    model_types = sorted({str(row["model_type"]) for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    output = []
    for reference_model in reference_models:
        if reference_model not in model_types:
            raise ValueError(f"Missing OOD reference model: {reference_model}")
        reference_by_key = {
            (
                str(row["run_id"]),
                int(row["prompt_id"]),
                int(row["seed"]),
                float(row["lambda"]),
            ): row
            for row in rows
            if row["model_type"] == reference_model
            and int(row["seed"]) - int(row["prompt_id"]) in selected_base_seeds
        }
        for candidate_model in model_types:
            if candidate_model == reference_model:
                continue
            for lambda_value in [*lambdas, None]:
                subset = filter_rows(
                    rows,
                    model_type=candidate_model,
                    lambda_value=lambda_value,
                    base_seeds=selected_base_seeds,
                )
                for metric in METRICS:
                    differences: dict[int, list[float]] = defaultdict(list)
                    for row in subset:
                        key = (
                            str(row["run_id"]),
                            int(row["prompt_id"]),
                            int(row["seed"]),
                            float(row["lambda"]),
                        )
                        reference = reference_by_key.get(key)
                        if reference is None:
                            raise ValueError(f"Missing OOD reference prediction: {key}")
                        if metric in LOWER_IS_BETTER:
                            delta = float(reference[metric]) - float(row[metric])
                        else:
                            delta = float(row[metric]) - float(reference[metric])
                        differences[int(row["prompt_id"])].append(delta)
                    values = np.asarray(
                        [
                            np.mean(differences[prompt_id])
                            for prompt_id in sorted(differences)
                        ],
                        dtype=np.float64,
                    )
                    mean, low, high = bootstrap_values(values, bootstrap_samples, rng)
                    output.append(
                        {
                            "reference_model": reference_model,
                            "candidate_model": candidate_model,
                            "lambda": (
                                "macro" if lambda_value is None else lambda_value
                            ),
                            "metric": metric,
                            "positive_means": "candidate_better",
                            "mean_delta": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "prompt_count": len(values),
                            "generation_base_seeds": " ".join(map(str, base_seeds)),
                        }
                    )
    return output


def paired_fixed_rows(
    rows: list[dict[str, Any]],
    *,
    base_seeds: list[int],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected_base_seeds = set(base_seeds)
    model_types = sorted({str(row["model_type"]) for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    output = []
    for model_type in model_types:
        for lambda_value in [*lambdas, None]:
            subset = filter_rows(
                rows,
                model_type=model_type,
                lambda_value=lambda_value,
                base_seeds=selected_base_seeds,
            )
            differences: dict[int, list[float]] = defaultdict(list)
            for row in subset:
                differences[int(row["prompt_id"])].append(
                    float(row["best_fixed_regret"]) - float(row["policy_regret"])
                )
            values = np.asarray(
                [np.mean(differences[prompt_id]) for prompt_id in sorted(differences)],
                dtype=np.float64,
            )
            mean, low, high = bootstrap_values(values, bootstrap_samples, rng)
            output.append(
                {
                    "model_type": model_type,
                    "lambda": "macro" if lambda_value is None else lambda_value,
                    "metric": "policy_regret",
                    "positive_means": "model_better_than_train_selected_fixed",
                    "mean_delta": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "prompt_count": len(values),
                    "generation_base_seeds": " ".join(map(str, base_seeds)),
                }
            )
    return output


def unpaired_domain_rows(
    control_rows: list[dict[str, Any]],
    ood_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    model_types = sorted({str(row["model_type"]) for row in control_rows})
    lambdas = sorted({float(row["lambda"]) for row in control_rows})
    output = []
    for model_type in model_types:
        for lambda_value in [*lambdas, None]:
            control_subset = filter_rows(
                control_rows, model_type=model_type, lambda_value=lambda_value
            )
            ood_subset = filter_rows(
                ood_rows,
                model_type=model_type,
                lambda_value=lambda_value,
                base_seeds={42},
            )
            for metric in METRICS:
                control_values = prompt_means(control_subset, metric)
                ood_values = prompt_means(ood_subset, metric)
                if len(control_values) != 200 or len(ood_values) != 200:
                    raise ValueError(
                        "Domain comparison requires 200 prompts per domain"
                    )
                control_indices = rng.integers(
                    0,
                    len(control_values),
                    size=(bootstrap_samples, len(control_values)),
                )
                ood_indices = rng.integers(
                    0, len(ood_values), size=(bootstrap_samples, len(ood_values))
                )
                draws = control_values[control_indices].mean(axis=1) - ood_values[
                    ood_indices
                ].mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                output.append(
                    {
                        "control_domain": "train_pool_hash_heldout200_base42",
                        "ood_domain": "prompt_1000_1199_base42",
                        "model_type": model_type,
                        "lambda": "macro" if lambda_value is None else lambda_value,
                        "metric": metric,
                        "delta_definition": "control_mean_minus_ood_mean",
                        "mean_delta": float(control_values.mean() - ood_values.mean()),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "control_prompt_count": len(control_values),
                        "ood_prompt_count": len(ood_values),
                    }
                )
    return output


def read_run_predictions(
    runs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summaries = []
    paths = sorted(
        runs_root.glob("seed_*/run_summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if len(paths) < 3:
        raise ValueError(f"Need at least three completed runs under {runs_root}")
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("evaluation_protocol") != train.EVALUATION_PROTOCOL:
            raise ValueError(f"Unexpected evaluation protocol: {path}")
        if summary.get("evaluation_split") != "validation" or summary.get(
            "test_accessed"
        ):
            raise ValueError(f"Run is not validation-only: {path}")
        predictions_path = path.parent / summary["artifacts"]["predictions"]
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        run_id = path.parent.name
        for raw in predictions:
            row = dict(raw)
            row["run_id"] = run_id
            row["train_seed"] = int(summary["train_seed"])
            row["prompt_id"] = int(row["prompt_id"])
            row["seed"] = int(row["seed"])
            row["lambda"] = float(row["lambda"])
            for metric in METRICS:
                row[metric] = float(row[metric])
            rows.append(row)
        summaries.append({"path": path, "payload": summary})
    return rows, summaries


def checkpoint_path(
    summary_path: Path, summary: dict[str, Any], model_type: str
) -> Path:
    metadata = summary["artifacts"]["checkpoints"][model_type]
    path = Path(str(metadata["path"]))
    path = path if path.is_absolute() else summary_path.parent / path
    path = path.resolve()
    if sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    return path


def build_model(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_type = str(checkpoint["model_type"])
    candidate_steps = np.asarray(checkpoint["candidate_steps"], dtype=np.int64)
    state_dim = int(checkpoint["state_dim"])
    dropout = float(checkpoint["dropout"])
    if model_type in {"prompt_only", "prompt_state"}:
        model: torch.nn.Module = train.VariableLambdaRouter(
            state_dim=state_dim,
            use_state=model_type == "prompt_state",
            dropout=dropout,
        )
    elif model_type == "b4_offline":
        model = train.VariableLambdaB4Prior(
            candidate_count=len(candidate_steps), dropout=dropout
        )
    elif model_type == "b4_prompt_state":
        prior = train.VariableLambdaB4Prior(
            candidate_count=len(candidate_steps), dropout=dropout
        )
        model = train.B4PromptStateRouter(
            b4_prior=prior,
            state_dim=state_dim,
            candidate_steps=candidate_steps,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def fixed_indices(
    checkpoint: dict[str, Any], summary: dict[str, Any], candidate_steps: np.ndarray
) -> dict[float, int]:
    values = checkpoint.get("fixed_steps", summary.get("fixed_steps"))
    if not isinstance(values, dict):
        raise ValueError("Checkpoint/run summary has no train-selected fixed steps")
    by_step = {int(step): index for index, step in enumerate(candidate_steps)}
    result = {}
    for raw_lambda, raw_step in values.items():
        step = int(raw_step)
        if step not in by_step:
            raise ValueError(f"Fixed step {step} is not a candidate")
        result[float(raw_lambda)] = by_step[step]
    return result


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    ood_dataset_dir = Path(args.ood_dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite OOD diagnostic: {out_dir}")

    control_rows, run_inputs = read_run_predictions(runs_root)
    if len(run_inputs) != 5 or {
        int(item["payload"]["train_seed"]) for item in run_inputs
    } != {42, 100, 2024, 31415, 27182}:
        raise ValueError(
            "Control diagnostic requires the five pre-registered train seeds"
        )
    control_prompt_ids = {int(row["prompt_id"]) for row in control_rows}
    if len(control_prompt_ids) != 200:
        raise ValueError("Control runs must evaluate exactly 200 held-out prompts")
    first_summary = run_inputs[0]["payload"]
    control_manifest_path = Path(first_summary["dataset_manifest"])
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    if control_manifest.get("derivation", {}).get("schema") != (
        "train800_control200_hash_split_v1"
    ):
        raise ValueError("Runs were not trained on the approved train800 control split")
    if control_manifest.get("test_accessed") is not False:
        raise ValueError("Control dataset accessed test")
    control_manifest_sha256 = sha256_file(control_manifest_path)
    for run_input in run_inputs:
        summary = run_input["payload"]
        if summary.get("dataset_manifest_sha256") != control_manifest_sha256:
            raise ValueError(f"Control run dataset differs: {run_input['path']}")
        if (
            int(summary.get("train_prompts", -1)) != 800
            or int(summary.get("validation_prompts", -1)) != 200
            or int(summary.get("train_trajectories", -1)) != 800
            or int(summary.get("validation_trajectories", -1)) != 200
        ):
            raise ValueError(f"Control run is not 800/200: {run_input['path']}")
        if set(summary.get("model_types", [])) != {
            "prompt_only",
            "prompt_state",
            "b4_offline",
            "b4_prompt_state",
        }:
            raise ValueError(
                f"Control run lacks four-model coverage: {run_input['path']}"
            )

    selection_path = runs_root / "selection" / "architecture_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_accessed") is not False:
        raise ValueError("Control selection accessed test")
    if int(selection.get("run_count", -1)) != 5:
        raise ValueError("Control selection does not contain five runs")

    ood_manifest = train.load_dataset_manifest(ood_dataset_dir)
    split_meta = ood_manifest["splits"]["validation"]
    if (
        int(split_meta["prompt_count"]) != 200
        or int(split_meta["trajectory_count"]) != 600
    ):
        raise ValueError("OOD validation must contain 200 prompts and 600 trajectories")

    first_model_type = first_summary["model_types"][0]
    first_checkpoint_path = checkpoint_path(
        run_inputs[0]["path"], first_summary, first_model_type
    )
    first_checkpoint = safe_torch_load(first_checkpoint_path)
    selected_names = [
        str(value) for value in first_checkpoint["selected_feature_names"]
    ]
    feature_positions = {
        str(name): index for index, name in enumerate(ood_manifest["feature_names"])
    }
    missing_features = [
        name for name in selected_names if name not in feature_positions
    ]
    if missing_features:
        raise ValueError(f"OOD dataset lacks selected features: {missing_features}")
    selected_indices = np.asarray(
        [feature_positions[name] for name in selected_names], dtype=np.int64
    )
    trajectories = train.load_trajectories(
        ood_dataset_dir, ood_manifest, "validation", selected_indices
    )
    ood_prompt_ids = {int(item["prompt_id"]) for item in trajectories}
    if ood_prompt_ids != set(range(1000, 1200)):
        raise ValueError("OOD diagnostic may load only prompt IDs 1000..1199")
    if control_prompt_ids & ood_prompt_ids:
        raise ValueError("Control and OOD validation prompts overlap")
    observed_base_seeds = {
        int(item["seed"]) - int(item["prompt_id"]) for item in trajectories
    }
    if observed_base_seeds != set(args.base_seeds):
        raise ValueError(
            f"OOD base-seed coverage mismatch: {sorted(observed_base_seeds)}"
        )

    device = torch.device(args.device)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    prediction_rows: list[dict[str, Any]] = []
    checkpoint_inputs = []
    signature = None
    for run_input in run_inputs:
        summary_path = run_input["path"]
        summary = run_input["payload"]
        run_id = summary_path.parent.name
        for model_type in summary["model_types"]:
            path = checkpoint_path(summary_path, summary, model_type)
            checkpoint = safe_torch_load(path)
            if checkpoint.get("schema") not in {
                "variable_lambda_router_checkpoint_v3",
                "variable_lambda_router_checkpoint_v4",
            }:
                raise ValueError(f"Unexpected checkpoint schema: {path}")
            if checkpoint.get("evaluation_protocol") != train.EVALUATION_PROTOCOL:
                raise ValueError(f"Checkpoint protocol mismatch: {path}")
            if str(checkpoint["model_type"]) != model_type:
                raise ValueError(f"Checkpoint model type mismatch: {path}")
            current_signature = (
                tuple(int(value) for value in checkpoint["candidate_steps"]),
                tuple(float(value) for value in checkpoint["eval_lambdas"]),
                tuple(str(value) for value in checkpoint["selected_feature_names"]),
                tuple(float(value) for value in checkpoint["cost_profile"]),
                str(checkpoint["latency_profile"]["sha256"]),
                float(checkpoint["harm_epsilon"]),
                float(checkpoint["risk_threshold"]),
                float(checkpoint["regret_scale"]),
            )
            if signature is None:
                signature = current_signature
            elif signature != current_signature:
                raise ValueError(f"Cross-eval checkpoint protocol differs: {path}")
            if (
                checkpoint["latency_profile"]["sha256"]
                != ood_manifest["latency_profile"]["sha256"]
            ):
                raise ValueError("OOD dataset and checkpoint latency profiles differ")

            candidate_steps = np.asarray(checkpoint["candidate_steps"], dtype=np.int64)
            cost_profile = np.asarray(checkpoint["cost_profile"], dtype=np.float32)
            candidate_seconds = np.asarray(
                checkpoint["calibrated_candidate_latency_seconds"], dtype=np.float32
            )
            native_seconds = float(checkpoint["calibrated_native_latency_seconds"])
            train.apply_locked_latency_profile(
                trajectories, cost_profile, candidate_seconds, native_seconds
            )
            fixed_steps = fixed_indices(checkpoint, summary, candidate_steps)
            eval_lambdas = [float(value) for value in checkpoint["eval_lambdas"]]
            if set(fixed_steps) != set(eval_lambdas):
                raise ValueError("Train-selected fixed-step lambda coverage mismatch")
            model = build_model(checkpoint, device)
            metrics, rows = train.evaluate_model(
                model,
                model_type,
                trajectories,
                eval_lambdas,
                candidate_steps,
                np.asarray(checkpoint["state_mean"], dtype=np.float32),
                np.asarray(checkpoint["state_std"], dtype=np.float32),
                cost_profile,
                float(checkpoint["regret_scale"]),
                float(checkpoint["harm_epsilon"]),
                float(checkpoint["risk_threshold"]),
                fixed_steps,
                list(ood_manifest["quality_dimensions"]),
                device,
                args.eval_batch_trajectories,
                emit_rows=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for row in rows:
                row["run_id"] = run_id
                row["train_seed"] = int(summary["train_seed"])
                row["evaluation_domain"] = "ood_validation_prompt_1000_1199"
                row["generation_base_seed"] = int(row["seed"]) - int(row["prompt_id"])
                prediction_rows.append(row)
            checkpoint_inputs.append(
                {
                    "run_id": run_id,
                    "train_seed": int(summary["train_seed"]),
                    "model_type": model_type,
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256_file(path),
                    "macro_policy_regret_all3": metrics["macro_policy_regret"],
                }
            )

    expected_rows = (
        len(run_inputs)
        * len(first_summary["model_types"])
        * len(trajectories)
        * len(first_checkpoint["eval_lambdas"])
    )
    if len(prediction_rows) != expected_rows:
        raise RuntimeError(
            f"OOD prediction coverage mismatch: {len(prediction_rows)} != {expected_rows}"
        )

    out_dir.mkdir(parents=True)
    predictions_path = out_dir / "ood_predictions.csv"
    write_csv(predictions_path, prediction_rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    base42_rows = interval_rows(prediction_rows, [42], args.bootstrap_samples, rng)
    all_seed_rows = interval_rows(
        prediction_rows, args.base_seeds, args.bootstrap_samples, rng
    )
    domain_rows = unpaired_domain_rows(
        control_rows, prediction_rows, args.bootstrap_samples, rng
    )
    base42_model_rows = paired_model_rows(
        prediction_rows,
        reference_models=["prompt_only", "b4_offline"],
        base_seeds=[42],
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    all_seed_model_rows = paired_model_rows(
        prediction_rows,
        reference_models=["prompt_only", "b4_offline"],
        base_seeds=args.base_seeds,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    base42_fixed_rows = paired_fixed_rows(
        prediction_rows,
        base_seeds=[42],
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    all_seed_fixed_rows = paired_fixed_rows(
        prediction_rows,
        base_seeds=args.base_seeds,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    base42_path = out_dir / "ood_base42_intervals.csv"
    all_seed_path = out_dir / "ood_all3_intervals.csv"
    domain_path = out_dir / "control_vs_ood_base42_deltas.csv"
    base42_model_path = out_dir / "ood_base42_paired_model_deltas.csv"
    all_seed_model_path = out_dir / "ood_all3_paired_model_deltas.csv"
    base42_fixed_path = out_dir / "ood_base42_paired_fixed_deltas.csv"
    all_seed_fixed_path = out_dir / "ood_all3_paired_fixed_deltas.csv"
    write_csv(base42_path, base42_rows)
    write_csv(all_seed_path, all_seed_rows)
    write_csv(domain_path, domain_rows)
    write_csv(base42_model_path, base42_model_rows)
    write_csv(all_seed_model_path, all_seed_model_rows)
    write_csv(base42_fixed_path, base42_fixed_rows)
    write_csv(all_seed_fixed_path, all_seed_fixed_rows)

    artifacts = {
        "predictions": predictions_path.name,
        "ood_base42_intervals": base42_path.name,
        "ood_all3_intervals": all_seed_path.name,
        "control_vs_ood_base42_deltas": domain_path.name,
        "ood_base42_paired_model_deltas": base42_model_path.name,
        "ood_all3_paired_model_deltas": all_seed_model_path.name,
        "ood_base42_paired_fixed_deltas": base42_fixed_path.name,
        "ood_all3_paired_fixed_deltas": all_seed_fixed_path.name,
    }
    reference_inputs = []
    if args.reference_runs_root:
        reference_root = Path(args.reference_runs_root).resolve()
        reference_rows, reference_runs = read_run_predictions(reference_root)
        if {
            (
                row["run_id"],
                row["model_type"],
                row["prompt_id"],
                row["seed"],
                row["lambda"],
            )
            for row in reference_rows
        } != {
            (
                row["run_id"],
                row["model_type"],
                row["prompt_id"],
                row["seed"],
                row["lambda"],
            )
            for row in prediction_rows
        }:
            raise ValueError(
                "Reference train1000 runs do not match OOD prediction coverage"
            )
        training_size_rows = paired_rows(
            prediction_rows,
            reference_rows,
            candidate_label="train800_control_selected",
            reference_label="train1000_original_selected",
            base_seeds=args.base_seeds,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
        )
        training_size_path = out_dir / "train800_vs_train1000_ood_paired_deltas.csv"
        write_csv(training_size_path, training_size_rows)
        artifacts["train800_vs_train1000_ood_paired_deltas"] = training_size_path.name
        reference_inputs = [
            {
                "summary": str(item["path"]),
                "summary_sha256": sha256_file(item["path"]),
            }
            for item in reference_runs
        ]

    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "in_distribution_control_vs_existing_ood_validation",
        "diagnostic_only": True,
        "training_size_comparison_role": (
            "descriptive_only_checkpoint_selection_domains_differ"
        ),
        "architecture_selected_on_ood": False,
        "selected_model_from_control_validation": selection["selected_model_type"],
        "test_accessed": False,
        "loaded_prompt_range": [1000, 1199],
        "loaded_generation_base_seeds": args.base_seeds,
        "control_prompt_count": len(control_prompt_ids),
        "ood_prompt_count": len(ood_prompt_ids),
        "ood_trajectory_count": len(trajectories),
        "run_count": len(run_inputs),
        "model_types": first_summary["model_types"],
        "eval_lambdas": first_checkpoint["eval_lambdas"],
        "latency_profile": first_checkpoint["latency_profile"],
        "bootstrap": {
            "unit": "prompt_after_averaging_training_seeds_generation_seeds_and_lambdas",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "control_dataset_manifest": str(control_manifest_path),
        "control_dataset_manifest_sha256": sha256_file(control_manifest_path),
        "ood_dataset_manifest": str(ood_dataset_dir / "dataset_manifest.json"),
        "ood_dataset_manifest_sha256": sha256_file(
            ood_dataset_dir / "dataset_manifest.json"
        ),
        "control_selection": str(selection_path),
        "control_selection_sha256": sha256_file(selection_path),
        "checkpoint_inputs": checkpoint_inputs,
        "reference_train1000_inputs": reference_inputs,
        "artifacts": {
            name: {"path": path, "sha256": sha256_file(out_dir / path)}
            for name, path in artifacts.items()
        },
    }
    report_path = out_dir / "ood_diagnostic_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
