#!/usr/bin/env python3
"""Quick model-free audit of state-factor geometry versus 40--50 oracle steps."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    from . import train_variable_lambda_router as base
    from .candidate_step_subset import (
        resolve_candidate_subset,
        subset_trajectory_candidates,
    )
except ImportError:
    import train_variable_lambda_router as base
    from candidate_step_subset import (
        resolve_candidate_subset,
        subset_trajectory_candidates,
    )


REPORT_SCHEMA = "steps40_50_factor_geometry_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate-steps", type=int, nargs="+", default=list(range(40, 51))
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument("--shuffle-repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    args.lambdas = sorted(set(float(value) for value in args.lambdas))
    if not args.lambdas or any(
        value < 0 or not math.isfinite(value) for value in args.lambdas
    ):
        parser.error("lambdas must contain finite non-negative values")
    if args.harm_epsilon < 0 or not math.isfinite(args.harm_epsilon):
        parser.error("harm-epsilon must be finite and non-negative")
    if args.shuffle_repetitions < 1:
        parser.error("shuffle-repetitions must be positive")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x_rank = rankdata(x[mask])
    y_rank = rankdata(y[mask])
    if np.std(x_rank) <= 0 or np.std(y_rank) <= 0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def step_standardize(
    train: np.ndarray, validation: np.ndarray, start_step: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each candidate step from train statistics, preserving NaNs."""
    train_result = np.full_like(train, np.nan, dtype=np.float32)
    validation_result = np.full_like(validation, np.nan, dtype=np.float32)
    for step_index in range(start_step, train.shape[1]):
        mean = np.nanmean(train[:, step_index], axis=0, dtype=np.float64)
        std = np.nanstd(train[:, step_index], axis=0, dtype=np.float64)
        std = np.maximum(std, 1e-6)
        train_result[:, step_index] = (train[:, step_index] - mean) / std
        validation_result[:, step_index] = (validation[:, step_index] - mean) / std
    return train_result, validation_result


def build_signals(
    train_state: np.ndarray, validation_state: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    value_train, value_validation = step_standardize(train_state, validation_state)
    delta_train = np.full_like(train_state, np.nan)
    delta_validation = np.full_like(validation_state, np.nan)
    delta_train[:, 1:] = np.diff(train_state, axis=1)
    delta_validation[:, 1:] = np.diff(validation_state, axis=1)
    delta_train, delta_validation = step_standardize(
        delta_train, delta_validation, start_step=1
    )
    slope_train = np.full_like(train_state, np.nan)
    slope_validation = np.full_like(validation_state, np.nan)
    slope_train[:, 2:] = (train_state[:, 2:] - train_state[:, :-2]) / 2.0
    slope_validation[:, 2:] = (validation_state[:, 2:] - validation_state[:, :-2]) / 2.0
    slope_train, slope_validation = step_standardize(
        slope_train, slope_validation, start_step=2
    )
    return (
        {"value": value_train, "delta": delta_train, "slope2": slope_train},
        {
            "value": value_validation,
            "delta": delta_validation,
            "slope2": slope_validation,
        },
    )


def utility_targets(
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    harm_epsilon: float,
) -> dict[str, np.ndarray]:
    qualities = np.stack([item["qualities"] for item in trajectories])
    costs = np.stack([item["costs"] for item in trajectories])
    utilities = (
        qualities[:, None, :] - np.asarray(lambdas)[None, :, None] * costs[:, None, :]
    )
    oracle_index = utilities.argmax(axis=2)
    oracle_utility = utilities.max(axis=2, keepdims=True)
    regret = np.maximum(oracle_utility - utilities, 0.0)
    acceptable = regret <= harm_epsilon + 1e-7
    earliest = acceptable.argmax(axis=2)
    latest = acceptable.shape[2] - 1 - acceptable[:, :, ::-1].argmax(axis=2)
    return {
        "utilities": utilities.astype(np.float32),
        "oracle_index": oracle_index.astype(np.int64),
        "earliest_acceptable_index": earliest.astype(np.int64),
        "latest_acceptable_index": latest.astype(np.int64),
        "acceptable": acceptable,
        "regret": regret.astype(np.float32),
    }


def fit_balanced_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, str, float]:
    """Fit the exact one-dimensional threshold maximizing balanced accuracy."""
    mask = np.isfinite(x)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.int8)
    if x.size < 2 or np.unique(y).size < 2:
        return 0.0, "ge", 0.5
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    sorted_y = y[order]
    positives = float(sorted_y.sum())
    negatives = float(len(sorted_y) - positives)
    cumulative_positive = np.cumsum(sorted_y)
    cumulative_negative = np.cumsum(1 - sorted_y)
    last_of_value = np.flatnonzero(np.r_[sorted_x[1:] != sorted_x[:-1], True])
    first_of_value = np.r_[0, last_of_value[:-1] + 1]

    tp_le = cumulative_positive[last_of_value]
    tn_le = negatives - cumulative_negative[last_of_value]
    score_le = 0.5 * (tp_le / positives + tn_le / negatives)

    previous_positive = np.where(
        first_of_value > 0, cumulative_positive[first_of_value - 1], 0
    )
    previous_negative = np.where(
        first_of_value > 0, cumulative_negative[first_of_value - 1], 0
    )
    tp_ge = positives - previous_positive
    tn_ge = previous_negative
    score_ge = 0.5 * (tp_ge / positives + tn_ge / negatives)

    le_index = int(np.argmax(score_le))
    ge_index = int(np.argmax(score_ge))
    if score_ge[ge_index] >= score_le[le_index]:
        return (
            float(sorted_x[first_of_value[ge_index]]),
            "ge",
            float(score_ge[ge_index]),
        )
    return (
        float(sorted_x[last_of_value[le_index]]),
        "le",
        float(score_le[le_index]),
    )


def threshold_prediction(x: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "ge":
        return x >= threshold
    if direction == "le":
        return x <= threshold
    raise ValueError(f"Unknown threshold direction: {direction}")


def oriented_auc(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.unique(y).size < 2:
        return float("nan")
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def balanced_accuracy(
    x: np.ndarray, y: np.ndarray, threshold: float, direction: str
) -> float:
    mask = np.isfinite(x)
    if mask.sum() < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    prediction = threshold_prediction(x[mask], threshold, direction)
    return float(balanced_accuracy_score(y[mask], prediction))


def shuffle_within_step(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.empty_like(signal)
    for step_index in range(signal.shape[1]):
        shuffled[:, step_index] = signal[rng.permutation(signal.shape[0]), step_index]
    return shuffled


def trajectory_descriptors(
    value: np.ndarray, delta: np.ndarray, candidate_steps: np.ndarray
) -> dict[str, np.ndarray]:
    centered_steps = candidate_steps - candidate_steps.mean()
    denominator = float(np.square(centered_steps).sum())
    slope = np.nansum(value * centered_steps[None, :, None], axis=1) / denominator
    change_offset = np.nanargmax(np.abs(delta[:, 1:]), axis=1) + 1
    return {
        "mean_value": np.nanmean(value, axis=1),
        "linear_slope": slope,
        "peak_step": candidate_steps[np.nanargmax(value, axis=1)],
        "change_point_step": candidate_steps[change_offset],
    }


def feature_group_by_index(manifest: dict[str, Any]) -> list[str]:
    groups = ["ungrouped"] * int(manifest["feature_count"])
    for group_name, indices in manifest["feature_groups"].items():
        for index in indices:
            groups[int(index)] = str(group_name)
    return groups


def oracle_distribution_rows(
    split: str,
    targets: dict[str, np.ndarray],
    candidate_steps: np.ndarray,
    lambdas: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for lambda_index, lambda_value in enumerate(lambdas):
        oracle = targets["oracle_index"][:, lambda_index]
        earliest = targets["earliest_acceptable_index"][:, lambda_index]
        latest = targets["latest_acceptable_index"][:, lambda_index]
        for step_index, step in enumerate(candidate_steps):
            rows.append(
                {
                    "split": split,
                    "lambda": lambda_value,
                    "step": int(step),
                    "oracle_count": int(np.sum(oracle == step_index)),
                    "oracle_fraction": float(np.mean(oracle == step_index)),
                    "earliest_acceptable_count": int(np.sum(earliest == step_index)),
                    "latest_acceptable_count": int(np.sum(latest == step_index)),
                    "acceptable_at_step_fraction": float(
                        np.mean(targets["acceptable"][:, lambda_index, step_index])
                    ),
                }
            )
    return rows


def aligned_profile_rows(
    signals: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    lambdas: list[float],
    feature_names: list[str],
    feature_groups: list[str],
) -> list[dict[str, Any]]:
    """Average step-detrended factor signals after alignment to oracle step."""
    rows: list[dict[str, Any]] = []
    step_count = next(iter(signals.values())).shape[1]
    step_grid = np.arange(step_count)[None, :]
    for lambda_index, lambda_value in enumerate(lambdas):
        relative = step_grid - targets["oracle_index"][:, lambda_index, None]
        for signal_name, signal in signals.items():
            for offset in range(-(step_count - 1), step_count):
                trajectory_index, step_index = np.nonzero(relative == offset)
                if trajectory_index.size == 0:
                    continue
                values = signal[trajectory_index, step_index]
                for feature_index, feature_name in enumerate(feature_names):
                    feature_values = values[:, feature_index]
                    finite = np.isfinite(feature_values)
                    if not finite.any():
                        continue
                    rows.append(
                        {
                            "lambda": lambda_value,
                            "relative_step_to_oracle": offset,
                            "feature_index": feature_index,
                            "feature_name": feature_name,
                            "feature_group": feature_groups[feature_index],
                            "signal": signal_name,
                            "mean": float(np.mean(feature_values[finite])),
                            "std": float(np.std(feature_values[finite])),
                            "count": int(finite.sum()),
                        }
                    )
    return rows


def trend_rows(
    split: str,
    descriptors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    candidate_steps: np.ndarray,
    lambdas: list[float],
    feature_names: list[str],
    feature_groups: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    oracle_steps = candidate_steps[targets["oracle_index"]]
    earliest_steps = candidate_steps[targets["earliest_acceptable_index"]]
    latest_steps = candidate_steps[targets["latest_acceptable_index"]]
    for descriptor_name, values in descriptors.items():
        for lambda_index, lambda_value in enumerate(lambdas):
            oracle = oracle_steps[:, lambda_index]
            for feature_index, feature_name in enumerate(feature_names):
                feature_values = values[:, feature_index]
                row: dict[str, Any] = {
                    "split": split,
                    "lambda": lambda_value,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_group": feature_groups[feature_index],
                    "descriptor": descriptor_name,
                    "spearman_to_oracle_step": safe_spearman(feature_values, oracle),
                }
                if descriptor_name == "change_point_step":
                    distance = np.maximum(
                        np.maximum(
                            earliest_steps[:, lambda_index] - feature_values,
                            feature_values - latest_steps[:, lambda_index],
                        ),
                        0,
                    )
                    row.update(
                        {
                            "oracle_step_mae": float(
                                np.mean(np.abs(feature_values - oracle))
                            ),
                            "within_one_step": float(
                                np.mean(np.abs(feature_values - oracle) <= 1)
                            ),
                            "acceptable_interval_distance": float(distance.mean()),
                        }
                    )
                else:
                    row.update(
                        {
                            "oracle_step_mae": "",
                            "within_one_step": "",
                            "acceptable_interval_distance": "",
                        }
                    )
                rows.append(row)
    return rows


def boundary_rows(
    train_signals: dict[str, np.ndarray],
    validation_signals: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    validation_targets: dict[str, np.ndarray],
    validation_seeds: np.ndarray,
    lambdas: list[float],
    feature_names: list[str],
    feature_groups: list[str],
    shuffled_validation: dict[str, list[np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step_indices_train = np.broadcast_to(
        np.arange(next(iter(train_signals.values())).shape[1])[None, :],
        next(iter(train_signals.values())).shape[:2],
    )
    step_indices_validation = np.broadcast_to(
        np.arange(next(iter(validation_signals.values())).shape[1])[None, :],
        next(iter(validation_signals.values())).shape[:2],
    )
    for signal_name, train_signal in train_signals.items():
        validation_signal = validation_signals[signal_name]
        for feature_index, feature_name in enumerate(feature_names):
            train_x = train_signal[:, :, feature_index].reshape(-1)
            validation_x = validation_signal[:, :, feature_index].reshape(-1)
            shuffled_x = [
                value[:, :, feature_index].reshape(-1)
                for value in shuffled_validation[signal_name]
            ]
            for lambda_index, lambda_value in enumerate(lambdas):
                train_y = (
                    step_indices_train
                    >= train_targets["oracle_index"][:, lambda_index, None]
                ).reshape(-1)
                validation_y_matrix = (
                    step_indices_validation
                    >= validation_targets["oracle_index"][:, lambda_index, None]
                )
                validation_y = validation_y_matrix.reshape(-1)
                threshold, direction, train_score = fit_balanced_threshold(
                    train_x, train_y
                )
                validation_score = balanced_accuracy(
                    validation_x, validation_y, threshold, direction
                )
                shuffle_scores = [
                    balanced_accuracy(value, validation_y, threshold, direction)
                    for value in shuffled_x
                ]
                seed_scores = []
                for seed in np.unique(validation_seeds):
                    trajectory_mask = validation_seeds == seed
                    seed_x = validation_signal[
                        trajectory_mask, :, feature_index
                    ].reshape(-1)
                    seed_y = validation_y_matrix[trajectory_mask].reshape(-1)
                    seed_scores.append(
                        balanced_accuracy(seed_x, seed_y, threshold, direction)
                    )
                rows.append(
                    {
                        "lambda": lambda_value,
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "feature_group": feature_groups[feature_index],
                        "signal": signal_name,
                        "threshold": threshold,
                        "positive_direction": direction,
                        "train_balanced_accuracy": train_score,
                        "validation_balanced_accuracy": validation_score,
                        "validation_auc_oriented": oriented_auc(
                            validation_x, validation_y
                        ),
                        "validation_seed_min_balanced_accuracy": float(
                            np.nanmin(seed_scores)
                        ),
                        "validation_seed_std_balanced_accuracy": float(
                            np.nanstd(seed_scores)
                        ),
                        "shuffle_balanced_accuracy_mean": float(
                            np.nanmean(shuffle_scores)
                        ),
                        "delta_balanced_accuracy_vs_shuffle": float(
                            validation_score - np.nanmean(shuffle_scores)
                        ),
                        "train_positive_rate": float(train_y.mean()),
                        "validation_positive_rate": float(validation_y.mean()),
                    }
                )
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for key, group in grouped.items():
        summary = dict(zip(keys, key))
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in group if row[metric] != ""],
                dtype=np.float64,
            )
            summary[f"mean_{metric}"] = float(np.nanmean(values))
            summary[f"max_abs_{metric}"] = float(np.nanmax(np.abs(values)))
        result.append(summary)
    return result


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    manifest = base.load_dataset_manifest(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    source_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    subset_indices, candidate_steps = resolve_candidate_subset(
        source_steps, args.candidate_steps
    )
    all_features = np.arange(int(manifest["feature_count"]), dtype=np.int64)
    train_trajectories = base.load_trajectories(
        dataset_dir, manifest, "train", all_features
    )
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", all_features
    )
    train_prompts = {int(item["prompt_id"]) for item in train_trajectories}
    validation_prompts = {int(item["prompt_id"]) for item in validation_trajectories}
    if train_prompts & validation_prompts:
        raise ValueError("Train and validation prompts overlap")
    if len(validation_prompts) != 200 or len(validation_trajectories) != 600:
        raise ValueError("Audit requires 200 validation prompts and 600 trajectories")
    validation_seed_counts = {
        prompt: sum(
            int(item["prompt_id"]) == prompt for item in validation_trajectories
        )
        for prompt in validation_prompts
    }
    if set(validation_seed_counts.values()) != {3}:
        raise ValueError("Every validation prompt must have exactly three seeds")

    cost_profile, seconds, native_seconds, latency_profile = (
        base.load_locked_latency_profile(manifest, source_steps, None)
    )
    base.apply_locked_latency_profile(
        train_trajectories, cost_profile, seconds, native_seconds
    )
    base.apply_locked_latency_profile(
        validation_trajectories, cost_profile, seconds, native_seconds
    )
    subset_trajectory_candidates(train_trajectories, subset_indices)
    subset_trajectory_candidates(validation_trajectories, subset_indices)

    train_state = np.stack([item["features"] for item in train_trajectories])
    validation_state = np.stack([item["features"] for item in validation_trajectories])
    train_signals, validation_signals = build_signals(train_state, validation_state)
    train_targets = utility_targets(train_trajectories, args.lambdas, args.harm_epsilon)
    validation_targets = utility_targets(
        validation_trajectories, args.lambdas, args.harm_epsilon
    )
    train_descriptors = trajectory_descriptors(
        train_signals["value"], train_signals["delta"], candidate_steps
    )
    validation_descriptors = trajectory_descriptors(
        validation_signals["value"],
        validation_signals["delta"],
        candidate_steps,
    )
    feature_names = [str(value) for value in manifest["feature_names"]]
    feature_groups = feature_group_by_index(manifest)

    oracle_distribution = oracle_distribution_rows(
        "train", train_targets, candidate_steps, args.lambdas
    )
    oracle_distribution.extend(
        oracle_distribution_rows(
            "validation", validation_targets, candidate_steps, args.lambdas
        )
    )
    aligned_profiles = aligned_profile_rows(
        validation_signals,
        validation_targets,
        args.lambdas,
        feature_names,
        feature_groups,
    )

    trend = trend_rows(
        "train",
        train_descriptors,
        train_targets,
        candidate_steps,
        args.lambdas,
        feature_names,
        feature_groups,
    )
    trend.extend(
        trend_rows(
            "validation",
            validation_descriptors,
            validation_targets,
            candidate_steps,
            args.lambdas,
            feature_names,
            feature_groups,
        )
    )
    rng = np.random.default_rng(args.seed)
    shuffled_validation = {
        name: [
            shuffle_within_step(signal, rng) for _ in range(args.shuffle_repetitions)
        ]
        for name, signal in validation_signals.items()
    }
    boundary = boundary_rows(
        train_signals,
        validation_signals,
        train_targets,
        validation_targets,
        np.asarray([item["seed"] for item in validation_trajectories]),
        args.lambdas,
        feature_names,
        feature_groups,
        shuffled_validation,
    )
    trend_summary = aggregate_rows(
        [row for row in trend if row["split"] == "validation"],
        ("feature_index", "feature_name", "feature_group", "descriptor"),
        ("spearman_to_oracle_step",),
    )
    boundary_summary = aggregate_rows(
        boundary,
        ("feature_index", "feature_name", "feature_group", "signal"),
        (
            "validation_balanced_accuracy",
            "validation_auc_oriented",
            "validation_seed_min_balanced_accuracy",
            "delta_balanced_accuracy_vs_shuffle",
        ),
    )
    trend_summary.sort(
        key=lambda row: row["max_abs_spearman_to_oracle_step"], reverse=True
    )
    boundary_summary.sort(
        key=lambda row: row["mean_delta_balanced_accuracy_vs_shuffle"],
        reverse=True,
    )

    out_dir.mkdir(parents=True)
    artifacts = {
        "oracle_step_distribution": "oracle_step_distribution.csv",
        "factor_oracle_aligned_profiles": "factor_oracle_aligned_profiles.csv",
        "factor_trend_association": "factor_trend_association.csv",
        "factor_trend_summary": "factor_trend_summary.csv",
        "factor_boundary_judgment": "factor_boundary_judgment.csv",
        "factor_boundary_summary": "factor_boundary_summary.csv",
    }
    write_csv(out_dir / artifacts["oracle_step_distribution"], oracle_distribution)
    write_csv(out_dir / artifacts["factor_oracle_aligned_profiles"], aligned_profiles)
    write_csv(out_dir / artifacts["factor_trend_association"], trend)
    write_csv(out_dir / artifacts["factor_trend_summary"], trend_summary)
    write_csv(out_dir / artifacts["factor_boundary_judgment"], boundary)
    write_csv(out_dir / artifacts["factor_boundary_summary"], boundary_summary)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "model_free_factor_value_and_trend_relation_to_40_50_oracle_step",
        "diagnostic_only": True,
        "evaluation_split": "validation",
        "test_accessed": False,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": base.sha256_file(manifest_path),
        "source_candidate_steps": source_steps.tolist(),
        "candidate_steps": candidate_steps.tolist(),
        "excluded_source_steps": [
            int(step) for step in source_steps if step not in set(candidate_steps)
        ],
        "lambdas": args.lambdas,
        "harm_epsilon": args.harm_epsilon,
        "train_prompt_count": len(train_prompts),
        "train_trajectory_count": len(train_trajectories),
        "validation_prompt_count": len(validation_prompts),
        "validation_trajectory_count": len(validation_trajectories),
        "validation_seed_count_per_prompt": 3,
        "signals": list(train_signals),
        "descriptors": list(train_descriptors),
        "boundary_target": "oracle_step_index_lte_current_step_index",
        "threshold_model": "single_factor_exact_balanced_accuracy_stump",
        "step_detrending": "train_mean_std_per_candidate_step",
        "shuffle_control": {
            "method": "validation_cross_trajectory_within_step",
            "repetitions": args.shuffle_repetitions,
            "seed": args.seed,
        },
        "latency_profile": latency_profile,
        "top_trend_associations": trend_summary[:20],
        "top_boundary_judges": boundary_summary[:20],
        "artifacts": {
            name: {
                "path": filename,
                "sha256": base.sha256_file(out_dir / filename),
            }
            for name, filename in artifacts.items()
        },
        "limitations": [
            "Exact oracle steps remain sensitive to generation-seed noise.",
            "This quick audit tests one factor at a time and does not fit a final router.",
            "Existing step-40 trajectory_delta features may retain causal source-step-35 history.",
            "Threshold metrics pool candidate steps after train-only step detrending.",
        ],
    }
    report_path = out_dir / "factor_geometry_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Factor geometry report: {report_path}")
    print(f"Top boundary judge: {boundary_summary[0]}")


if __name__ == "__main__":
    main()
