#!/usr/bin/env python3
"""Test whether online state adds conditional signal around a frozen B4 anchor."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import analyze_factor_relevance as factor
    from . import train_soft_margin_router as soft
    from . import train_variable_lambda_router as base
except ImportError:
    import analyze_factor_relevance as factor
    import train_soft_margin_router as soft
    import train_variable_lambda_router as base


REPORT_SCHEMA = "b4_conditional_state_capacity_audit_v1"


@dataclass
class LocalRows:
    control: np.ndarray
    state: np.ndarray
    target: np.ndarray
    prompt_ids: np.ndarray
    trajectory_ids: np.ndarray
    lambda_ids: np.ndarray
    step_ids: np.ndarray
    offsets: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--b4-runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--quality-scale", type=float, default=100.0)
    parser.add_argument("--material-gain", type=float, default=0.001)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0]
    )
    parser.add_argument(
        "--state-groups",
        nargs="+",
        default=["sparse_temporal", "content", "convergence", "all_state"],
    )
    parser.add_argument("--histgb-iterations", type=int, default=100)
    parser.add_argument("--histgb-max-leaves", type=int, nargs="+", default=[7, 15])
    parser.add_argument("--histgb-min-samples-leaf", type=int, default=50)
    parser.add_argument("--top-fractions", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2051)
    parser.add_argument("--shuffle-seed", type=int, default=2053)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.lambdas = sorted(set(float(value) for value in args.lambdas))
    args.ridge_alphas = sorted(set(float(value) for value in args.ridge_alphas))
    args.histgb_max_leaves = sorted(set(int(value) for value in args.histgb_max_leaves))
    args.top_fractions = sorted(set(float(value) for value in args.top_fractions))
    if not args.lambdas or any(
        value < 0 or not math.isfinite(value) for value in args.lambdas
    ):
        parser.error("lambdas must be finite and non-negative")
    if args.radius < 1 or args.cv_folds < 2:
        parser.error("radius must be positive and cv-folds must be at least two")
    if args.quality_scale <= 0 or args.material_gain <= 0:
        parser.error("quality-scale and material-gain must be positive")
    if not args.ridge_alphas or any(value <= 0 for value in args.ridge_alphas):
        parser.error("ridge-alphas must be positive")
    if not args.histgb_max_leaves or any(value < 2 for value in args.histgb_max_leaves):
        parser.error("histgb-max-leaves must be at least two")
    if any(not 0 < value < 1 for value in args.top_fractions):
        parser.error("top-fractions must be in (0, 1)")
    if 0.10 not in args.top_fractions:
        parser.error("top-fractions must include 0.10 for the capacity gate")
    if args.bootstrap_samples < 1 or args.inference_batch_size < 1:
        parser.error("bootstrap-samples and inference-batch-size must be positive")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def ensemble_b4_probabilities(
    models: list[base.VariableLambdaB4Prior],
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    candidate_count = models[0].candidate_count
    result = np.empty(
        (len(trajectories), len(lambdas), candidate_count), dtype=np.float32
    )
    for lambda_index, lambda_value in enumerate(lambdas):
        for start in range(0, len(trajectories), batch_size):
            chunk = trajectories[start : start + batch_size]
            pooled = torch.from_numpy(
                np.stack([trajectory["pooled_t5"] for trajectory in chunk])
            ).to(device)
            lambda_tensor = torch.full(
                (len(chunk),), lambda_value, dtype=torch.float32, device=device
            )
            probabilities = torch.stack(
                [model(pooled, lambda_tensor)["discrete_probs"] for model in models]
            ).mean(dim=0)
            result[start : start + len(chunk), lambda_index] = (
                probabilities.cpu().numpy()
            )
    return result


def b4_context(probabilities: np.ndarray, current: int, radius: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    anchor = int(values.argmax())
    normalized_positions = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    expected = float(np.dot(values, normalized_positions))
    entropy = float(
        -(values * np.log(np.maximum(values, 1e-8))).sum() / math.log(len(values))
    )
    top2 = np.partition(values, -2)[-2:]
    local_log_ratio = math.log(max(float(values[current]), 1e-8)) - math.log(
        max(float(values[current + 1]), 1e-8)
    )
    return np.asarray(
        [
            *values.tolist(),
            expected,
            entropy,
            float(values.max()),
            float(top2.max() - top2.min()),
            anchor / max(len(values) - 1, 1),
            current / max(len(values) - 1, 1),
            (current - anchor) / max(radius, 1),
            float(values[current]),
            float(values[current + 1]),
            local_log_ratio,
        ],
        dtype=np.float32,
    )


def build_local_rows(
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    candidate_steps: np.ndarray,
    cost_profile: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    b4_probabilities: np.ndarray,
    radius: int,
    quality_scale: float,
) -> LocalRows:
    controls: list[np.ndarray] = []
    states: list[np.ndarray] = []
    targets: list[float] = []
    prompt_ids: list[int] = []
    trajectory_ids: list[int] = []
    lambda_ids: list[int] = []
    step_ids: list[int] = []
    offsets: list[int] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        normalized_state = (trajectory["features"] - state_mean) / state_std
        for lambda_index, lambda_value in enumerate(lambdas):
            probabilities = b4_probabilities[trajectory_index, lambda_index]
            anchor = int(probabilities.argmax())
            schedule = base.schedule_features(
                candidate_steps, trajectory["sigmas"], lambda_value, cost_profile
            )
            utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
            first = max(0, anchor - radius)
            last = min(len(candidate_steps) - 2, anchor + radius)
            for current in range(first, last + 1):
                controls.append(
                    np.concatenate(
                        [schedule[current], b4_context(probabilities, current, radius)]
                    ).astype(np.float32)
                )
                states.append(normalized_state[current].astype(np.float32))
                targets.append(
                    float((utility[current] - utility[current + 1]) * quality_scale)
                )
                prompt_ids.append(int(trajectory["prompt_id"]))
                trajectory_ids.append(trajectory_index)
                lambda_ids.append(lambda_index)
                step_ids.append(current)
                offsets.append(current - anchor)
    return LocalRows(
        control=np.stack(controls),
        state=np.stack(states),
        target=np.asarray(targets, dtype=np.float32),
        prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        lambda_ids=np.asarray(lambda_ids, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


def state_groups(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    groups = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in manifest["feature_groups"].items()
    }
    groups["content"] = np.unique(
        np.concatenate(
            [groups["x0_global"], groups["x0_channel"], groups["local_energy"]]
        )
    )
    groups["convergence"] = np.unique(
        np.concatenate(
            [
                groups["residual_global"],
                groups["residual_channel"],
                groups["trajectory_delta"],
            ]
        )
    )
    names = list(manifest["feature_names"])
    sparse_names = (
        "residual.temporal_gradient_abs_mean",
        "residual.temporal_second_abs_mean",
        "x0.temporal_gradient_abs_mean",
        "x0.temporal_second_abs_mean",
    )
    groups["sparse_temporal"] = np.asarray(
        [names.index(name) for name in sparse_names], dtype=np.int64
    )
    groups["all_state"] = np.arange(int(manifest["feature_count"]), dtype=np.int64)
    return groups


def make_regressor(family: str, parameter: float, args: argparse.Namespace) -> Any:
    if family == "ridge":
        return make_pipeline(
            StandardScaler(), Ridge(alpha=parameter, solver="lsqr", tol=1e-4)
        )
    if family == "histgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=args.histgb_iterations,
            max_leaf_nodes=int(parameter),
            min_samples_leaf=args.histgb_min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.bootstrap_seed,
        )
    raise ValueError(f"Unsupported model family: {family}")


def prompt_macro_mse(
    target: np.ndarray, prediction: np.ndarray, prompt_ids: np.ndarray
) -> float:
    return float(
        np.mean(
            [
                np.square(
                    target[prompt_ids == prompt] - prediction[prompt_ids == prompt]
                ).mean()
                for prompt in np.unique(prompt_ids)
            ]
        )
    )


def cross_fitted_regressor(
    features: np.ndarray,
    target: np.ndarray,
    prompt_ids: np.ndarray,
    family: str,
    parameter: float,
    folds: int,
    args: argparse.Namespace,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fit_index, eval_index in splitter.split(features, target, groups=prompt_ids):
        model = make_regressor(family, parameter, args)
        model.fit(features[fit_index], target[fit_index])
        prediction[eval_index] = model.predict(features[eval_index])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Cross-fitted predictions are incomplete")
    return prediction


def select_train_only_spec(
    rows: LocalRows,
    groups: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    specs: list[tuple[str, float]] = [
        *(("ridge", float(alpha)) for alpha in args.ridge_alphas),
        *(("histgb", float(leaves)) for leaves in args.histgb_max_leaves),
    ]
    results: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for family, parameter in specs:
        control_prediction = cross_fitted_regressor(
            rows.control,
            rows.target,
            rows.prompt_ids,
            family,
            parameter,
            args.cv_folds,
            args,
        )
        control_mse = prompt_macro_mse(rows.target, control_prediction, rows.prompt_ids)
        for group_name in args.state_groups:
            indices = groups[group_name]
            state_prediction = cross_fitted_regressor(
                np.concatenate([rows.control, rows.state[:, indices]], axis=1),
                rows.target,
                rows.prompt_ids,
                family,
                parameter,
                args.cv_folds,
                args,
            )
            state_mse = prompt_macro_mse(rows.target, state_prediction, rows.prompt_ids)
            row = {
                "model_family": family,
                "model_parameter": parameter,
                "state_group": group_name,
                "state_feature_count": int(len(indices)),
                "train_oof_control_prompt_macro_mse": control_mse,
                "train_oof_state_prompt_macro_mse": state_mse,
                "train_oof_mse_gain": control_mse - state_mse,
            }
            results.append(row)
            predictions[(family, parameter, group_name)] = (
                control_prediction,
                state_prediction,
            )
    selected = max(
        results,
        key=lambda row: (
            row["train_oof_mse_gain"],
            -row["train_oof_state_prompt_macro_mse"],
            -row["state_feature_count"],
        ),
    )
    key = (
        str(selected["model_family"]),
        float(selected["model_parameter"]),
        str(selected["state_group"]),
    )
    control_prediction, state_prediction = predictions[key]
    return selected, results, control_prediction, state_prediction


def shuffle_rows_within_prompt_key(
    state: np.ndarray,
    prompt_ids: np.ndarray,
    lambda_ids: np.ndarray,
    step_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = state.copy()
    keys = np.stack([prompt_ids, lambda_ids, step_ids], axis=1)
    for key in np.unique(keys, axis=0):
        indices = np.flatnonzero((keys == key).all(axis=1))
        if len(indices) > 1:
            result[indices] = state[rng.permutation(indices)]
    return result


def centered_within_prompt_seed_rows(
    rows: LocalRows, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered_state = np.empty_like(state)
    centered_target = np.empty_like(rows.target)
    keys = np.stack([rows.prompt_ids, rows.lambda_ids, rows.step_ids], axis=1)
    valid = np.zeros(len(rows.target), dtype=bool)
    for key in np.unique(keys, axis=0):
        indices = np.flatnonzero((keys == key).all(axis=1))
        if len(np.unique(rows.trajectory_ids[indices])) < 2:
            continue
        centered_state[indices] = state[indices] - state[indices].mean(
            axis=0, keepdims=True
        )
        centered_target[indices] = rows.target[indices] - rows.target[indices].mean()
        valid[indices] = True
    return centered_state[valid], centered_target[valid], rows.prompt_ids[valid]


def oof_state_only(
    state: np.ndarray,
    target: np.ndarray,
    prompt_ids: np.ndarray,
    family: str,
    parameter: float,
    folds: int,
    args: argparse.Namespace,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fit_index, eval_index in splitter.split(state, target, groups=prompt_ids):
        model = make_regressor(family, parameter, args)
        model.fit(state[fit_index], target[fit_index])
        prediction[eval_index] = model.predict(state[eval_index])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Within-prompt OOF predictions are incomplete")
    return prediction


def safe_auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> float:
    if len(np.unique(labels[weights > 0] if weights is not None else labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def safe_ap(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> float:
    if len(np.unique(labels[weights > 0] if weights is not None else labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores, sample_weight=weights))


def prompt_bootstrap_mean(
    values: np.ndarray,
    prompt_ids: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    prompts = np.unique(prompt_ids)
    means = np.asarray([values[prompt_ids == prompt].mean() for prompt in prompts])
    draws = means[rng.integers(0, len(means), size=(samples, len(means)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(means.mean()), float(low), float(high)


def prompt_bootstrap_score_delta(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    prompt_ids: np.ndarray,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    scorer = safe_auc if metric == "auc" else safe_ap
    point = scorer(labels, candidate) - scorer(labels, reference)
    prompts = np.unique(prompt_ids)
    prompt_position = {int(prompt): index for index, prompt in enumerate(prompts)}
    row_positions = np.asarray([prompt_position[int(value)] for value in prompt_ids])
    draws: list[float] = []
    for _ in range(samples):
        counts = np.bincount(
            rng.integers(0, len(prompts), size=len(prompts)), minlength=len(prompts)
        )
        weights = counts[row_positions].astype(np.float64)
        candidate_score = scorer(labels, candidate, weights)
        reference_score = scorer(labels, reference, weights)
        if math.isfinite(candidate_score) and math.isfinite(reference_score):
            draws.append(candidate_score - reference_score)
    if not draws:
        return float(point), float("nan"), float("nan")
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(point), float(low), float(high)


def evaluate_predictions(
    stage: str,
    target: np.ndarray,
    prompt_ids: np.ndarray,
    predictions: dict[str, np.ndarray],
    material_scaled: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float, float]]]:
    rows: list[dict[str, Any]] = []
    for threshold_name, threshold in (("positive", 0.0), ("material", material_scaled)):
        labels = (target > threshold).astype(np.int64)
        for model_name, prediction in predictions.items():
            rows.append(
                {
                    "stage": stage,
                    "target": threshold_name,
                    "model": model_name,
                    "row_count": len(target),
                    "positive_rate": float(labels.mean()),
                    "auc": safe_auc(labels, prediction),
                    "average_precision": safe_ap(labels, prediction),
                    "prompt_macro_mse": prompt_macro_mse(
                        target, prediction, prompt_ids
                    ),
                }
            )
    comparisons: dict[str, tuple[float, float, float]] = {}
    positive = (target > 0.0).astype(np.int64)
    for reference in ("control", "shuffled"):
        for metric in ("auc", "average_precision"):
            comparisons[f"state_vs_{reference}_{metric}"] = (
                prompt_bootstrap_score_delta(
                    positive,
                    predictions["state"],
                    predictions[reference],
                    prompt_ids,
                    metric,
                    bootstrap_samples,
                    rng,
                )
            )
    control_error = np.square(target - predictions["control"])
    state_error = np.square(target - predictions["state"])
    shuffled_error = np.square(target - predictions["shuffled"])
    comparisons["state_vs_control_mse_gain"] = prompt_bootstrap_mean(
        control_error - state_error, prompt_ids, bootstrap_samples, rng
    )
    comparisons["state_vs_shuffled_mse_gain"] = prompt_bootstrap_mean(
        shuffled_error - state_error, prompt_ids, bootstrap_samples, rng
    )
    return rows, comparisons


def top_fraction_rows(
    stage: str,
    target: np.ndarray,
    prompt_ids: np.ndarray,
    predictions: dict[str, np.ndarray],
    thresholds: dict[tuple[str, float], float],
    fractions: list[float],
    quality_scale: float,
    material_gain: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, prediction in predictions.items():
        for fraction in fractions:
            threshold = thresholds[(model_name, fraction)]
            selected = prediction >= threshold
            selected_values = np.where(selected, target / quality_scale, 0.0)
            point, low, high = prompt_bootstrap_mean(
                selected_values, prompt_ids, bootstrap_samples, rng
            )
            rows.append(
                {
                    "stage": stage,
                    "model": model_name,
                    "top_fraction": fraction,
                    "score_threshold": threshold,
                    "selection_rate": float(selected.mean()),
                    "selected_row_count": int(selected.sum()),
                    "aggregate_adjacent_utility_gain": point,
                    "gain_ci95_low": low,
                    "gain_ci95_high": high,
                    "selected_mean_adjacent_utility_gain": (
                        float((target[selected] / quality_scale).mean())
                        if selected.any()
                        else 0.0
                    ),
                    "selected_positive_rate": (
                        float((target[selected] > 0).mean()) if selected.any() else 0.0
                    ),
                    "selected_material_positive_rate": (
                        float((target[selected] > material_gain * quality_scale).mean())
                        if selected.any()
                        else 0.0
                    ),
                }
            )
    return rows


def comparison_rows(
    stage: str, comparisons: dict[str, tuple[float, float, float]]
) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "comparison": name,
            "mean_delta": values[0],
            "ci95_low": values[1],
            "ci95_high": values[2],
        }
        for name, values in comparisons.items()
    ]


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    b4_runs_root = Path(args.b4_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    manifest = base.load_dataset_manifest(dataset_dir)
    if manifest.get("test_accessed"):
        raise ValueError(
            "Capacity selection refuses a state dataset that accessed test"
        )
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_sha256 = factor.sha256_file(manifest_path)
    indices = np.arange(int(manifest["feature_count"]), dtype=np.int64)
    train_trajectories = base.load_trajectories(dataset_dir, manifest, "train", indices)
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", indices
    )
    train_prompts = {int(item["prompt_id"]) for item in train_trajectories}
    validation_prompts = {int(item["prompt_id"]) for item in validation_trajectories}
    if train_prompts & validation_prompts:
        raise ValueError("Train and validation prompts overlap")
    validation_counts = {
        prompt: sum(
            int(item["prompt_id"]) == prompt for item in validation_trajectories
        )
        for prompt in validation_prompts
    }
    if set(validation_counts.values()) != {3}:
        raise ValueError(
            "Within-prompt audit requires three validation seeds per prompt"
        )
    candidate_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    state_mean, state_std = soft.fit_step_state_normalizer(train_trajectories)
    cost_profile, seconds, native_seconds, latency_provenance = (
        base.load_locked_latency_profile(manifest, candidate_steps, None)
    )
    base.apply_locked_latency_profile(
        train_trajectories, cost_profile, seconds, native_seconds
    )
    base.apply_locked_latency_profile(
        validation_trajectories, cost_profile, seconds, native_seconds
    )
    device = torch.device(args.device)
    models, b4_inputs = factor.load_b4_ensemble(
        b4_runs_root, manifest_sha256, len(candidate_steps), device
    )
    print("Computing frozen B4 ensemble probabilities...")
    train_b4 = ensemble_b4_probabilities(
        models, train_trajectories, args.lambdas, device, args.inference_batch_size
    )
    validation_b4 = ensemble_b4_probabilities(
        models,
        validation_trajectories,
        args.lambdas,
        device,
        args.inference_batch_size,
    )
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_rows = build_local_rows(
        train_trajectories,
        args.lambdas,
        candidate_steps,
        cost_profile,
        state_mean,
        state_std,
        train_b4,
        args.radius,
        args.quality_scale,
    )
    validation_rows = build_local_rows(
        validation_trajectories,
        args.lambdas,
        candidate_steps,
        cost_profile,
        state_mean,
        state_std,
        validation_b4,
        args.radius,
        args.quality_scale,
    )
    groups = state_groups(manifest)
    missing_groups = sorted(set(args.state_groups) - set(groups))
    if missing_groups:
        raise ValueError(f"Unknown state groups: {missing_groups}")
    print("Selecting the probe family and state group with train-only OOF...")
    selected, selection_rows, train_control_oof, train_state_oof = (
        select_train_only_spec(train_rows, groups, args)
    )
    family = str(selected["model_family"])
    parameter = float(selected["model_parameter"])
    group_name = str(selected["state_group"])
    state_indices = groups[group_name]
    control_model = make_regressor(family, parameter, args)
    state_model = make_regressor(family, parameter, args)
    control_model.fit(train_rows.control, train_rows.target)
    state_model.fit(
        np.concatenate(
            [train_rows.control, train_rows.state[:, state_indices]], axis=1
        ),
        train_rows.target,
    )
    validation_control = control_model.predict(validation_rows.control).astype(
        np.float32
    )
    validation_state = state_model.predict(
        np.concatenate(
            [validation_rows.control, validation_rows.state[:, state_indices]], axis=1
        )
    ).astype(np.float32)
    shuffled_validation_state = shuffle_rows_within_prompt_key(
        validation_rows.state[:, state_indices],
        validation_rows.prompt_ids,
        validation_rows.lambda_ids,
        validation_rows.step_ids,
        np.random.default_rng(args.shuffle_seed),
    )
    validation_shuffled = state_model.predict(
        np.concatenate([validation_rows.control, shuffled_validation_state], axis=1)
    ).astype(np.float32)
    rng = np.random.default_rng(args.bootstrap_seed)
    validation_predictions = {
        "control": validation_control,
        "state": validation_state,
        "shuffled": validation_shuffled,
    }
    validation_metric_rows, validation_comparisons = evaluate_predictions(
        "train_to_validation",
        validation_rows.target,
        validation_rows.prompt_ids,
        validation_predictions,
        args.material_gain * args.quality_scale,
        args.bootstrap_samples,
        rng,
    )
    train_predictions = {
        "control": train_control_oof,
        "state": train_state_oof,
        "shuffled": train_state_oof,
    }
    validation_thresholds = {
        (name, fraction): float(np.quantile(train_predictions[name], 1.0 - fraction))
        for name in train_predictions
        for fraction in args.top_fractions
    }
    top_rows = top_fraction_rows(
        "train_to_validation",
        validation_rows.target,
        validation_rows.prompt_ids,
        validation_predictions,
        validation_thresholds,
        args.top_fractions,
        args.quality_scale,
        args.material_gain,
        args.bootstrap_samples,
        rng,
    )
    centered_state, centered_target, centered_prompts = (
        centered_within_prompt_seed_rows(
            validation_rows, validation_rows.state[:, state_indices]
        )
    )
    shuffled_rows = shuffle_rows_within_prompt_key(
        validation_rows.state[:, state_indices],
        validation_rows.prompt_ids,
        validation_rows.lambda_ids,
        validation_rows.step_ids,
        np.random.default_rng(args.shuffle_seed),
    )
    centered_shuffled, shuffled_target, shuffled_prompts = (
        centered_within_prompt_seed_rows(validation_rows, shuffled_rows)
    )
    if not np.array_equal(centered_prompts, shuffled_prompts) or not np.allclose(
        centered_target, shuffled_target
    ):
        raise RuntimeError("Centered state and shuffle rows differ")
    within_state = oof_state_only(
        centered_state,
        centered_target,
        centered_prompts,
        family,
        parameter,
        args.cv_folds,
        args,
    )
    within_shuffled = oof_state_only(
        centered_shuffled,
        centered_target,
        centered_prompts,
        family,
        parameter,
        args.cv_folds,
        args,
    )
    within_predictions = {
        "control": np.zeros_like(centered_target),
        "state": within_state,
        "shuffled": within_shuffled,
    }
    within_metric_rows, within_comparisons = evaluate_predictions(
        "within_prompt_seed_groupkfold",
        centered_target,
        centered_prompts,
        within_predictions,
        args.material_gain * args.quality_scale,
        args.bootstrap_samples,
        rng,
    )
    within_thresholds = {
        (name, fraction): float(np.quantile(prediction, 1.0 - fraction))
        for name, prediction in within_predictions.items()
        for fraction in args.top_fractions
    }
    top_rows.extend(
        top_fraction_rows(
            "within_prompt_seed_groupkfold",
            centered_target,
            centered_prompts,
            within_predictions,
            within_thresholds,
            args.top_fractions,
            args.quality_scale,
            args.material_gain,
            args.bootstrap_samples,
            rng,
        )
    )
    comparison_output = comparison_rows(
        "train_to_validation", validation_comparisons
    ) + comparison_rows("within_prompt_seed_groupkfold", within_comparisons)
    comparison_lookup = {
        (row["stage"], row["comparison"]): row for row in comparison_output
    }
    top_lookup = {
        (row["stage"], row["model"], float(row["top_fraction"])): row
        for row in top_rows
    }
    required_gates = {
        "validation_state_vs_control_auc": comparison_lookup[
            ("train_to_validation", "state_vs_control_auc")
        ]["ci95_low"]
        > 0,
        "validation_state_vs_shuffle_auc": comparison_lookup[
            ("train_to_validation", "state_vs_shuffled_auc")
        ]["ci95_low"]
        > 0,
        "validation_state_vs_control_mse": comparison_lookup[
            ("train_to_validation", "state_vs_control_mse_gain")
        ]["ci95_low"]
        > 0,
        "within_prompt_state_vs_shuffle_auc": comparison_lookup[
            ("within_prompt_seed_groupkfold", "state_vs_shuffled_auc")
        ]["ci95_low"]
        > 0,
        "within_prompt_state_vs_control_mse": comparison_lookup[
            ("within_prompt_seed_groupkfold", "state_vs_control_mse_gain")
        ]["ci95_low"]
        > 0,
        "validation_top10_gain": top_lookup[("train_to_validation", "state", 0.1)][
            "gain_ci95_low"
        ]
        > 0,
    }
    capacity_passed = all(required_gates.values())
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "train_oof_probe_selection.csv", selection_rows)
    write_csv(
        out_dir / "conditional_capacity_metrics.csv",
        validation_metric_rows + within_metric_rows,
    )
    write_csv(out_dir / "paired_capacity_deltas.csv", comparison_output)
    write_csv(out_dir / "top_fraction_gain.csv", top_rows)
    prediction_rows = []
    for index in range(len(validation_rows.target)):
        prediction_rows.append(
            {
                "stage": "train_to_validation",
                "prompt_id": int(validation_rows.prompt_ids[index]),
                "trajectory_id": int(validation_rows.trajectory_ids[index]),
                "lambda": args.lambdas[int(validation_rows.lambda_ids[index])],
                "candidate_step": int(candidate_steps[validation_rows.step_ids[index]]),
                "offset_from_b4": int(validation_rows.offsets[index]),
                "adjacent_utility_delta": float(
                    validation_rows.target[index] / args.quality_scale
                ),
                "control_score": float(validation_control[index]),
                "state_score": float(validation_state[index]),
                "shuffled_score": float(validation_shuffled[index]),
            }
        )
    write_csv(out_dir / "validation_conditional_predictions.csv", prediction_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "incremental_online_state_capacity_conditioned_on_frozen_b4",
        "diagnostic_only": True,
        "formal_evidence": False,
        "test_accessed": False,
        "capacity_passed_all_gates": capacity_passed,
        "gate_results": required_gates,
        "selected_probe": selected,
        "selection_rule": "maximum train OOF prompt-macro MSE gain over matched B4 control",
        "targets": {
            "adjacent_utility_delta": "U(stop_at_t)-U(continue_to_t_plus_1)",
            "quality_scale": args.quality_scale,
            "material_gain": args.material_gain,
        },
        "candidate_steps": candidate_steps.tolist(),
        "radius_around_b4": args.radius,
        "lambdas": args.lambdas,
        "state_feature_count": int(len(state_indices)),
        "state_feature_names": [
            manifest["feature_names"][index] for index in state_indices
        ],
        "train_prompt_count": len(train_prompts),
        "train_trajectory_count": len(train_trajectories),
        "validation_prompt_count": len(validation_prompts),
        "validation_trajectory_count": len(validation_trajectories),
        "validation_seed_count_per_prompt": 3,
        "row_counts": {
            "train_local_rows": len(train_rows.target),
            "validation_local_rows": len(validation_rows.target),
            "within_prompt_centered_rows": len(centered_target),
        },
        "bootstrap": {
            "unit": "prompt_cluster",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "shuffle": {
            "method": "within_prompt_lambda_absolute_step_generation_seed",
            "seed": args.shuffle_seed,
        },
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "latency_profile": latency_provenance,
        "b4_inputs": b4_inputs,
        "artifacts": {
            "train_oof_probe_selection": "train_oof_probe_selection.csv",
            "conditional_capacity_metrics": "conditional_capacity_metrics.csv",
            "paired_capacity_deltas": "paired_capacity_deltas.csv",
            "top_fraction_gain": "top_fraction_gain.csv",
            "validation_conditional_predictions": "validation_conditional_predictions.csv",
        },
        "interpretation": (
            "Pass means the current state statistics add reproducible local utility signal "
            "beyond frozen B4 context and within-prompt shuffled state. It does not select "
            "or confirm a deployment policy."
        ),
    }
    report_path = out_dir / "conditional_state_capacity_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
