#!/usr/bin/env python3
"""Audit which online state factors predict utility-optimal handoff decisions."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import train_soft_margin_router as soft
    from . import train_variable_lambda_router as base
except ImportError:
    import train_soft_margin_router as soft
    import train_variable_lambda_router as base


REPORT_SCHEMA = "variable_lambda_factor_relevance_audit_v1"
MODEL_FAMILIES = ("ridge", "histgb")


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
    parser.add_argument("--margin-temperature", type=float, default=0.02)
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0]
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--histgb-iterations", type=int, default=100)
    parser.add_argument("--histgb-max-leaves", type=int, default=15)
    parser.add_argument("--histgb-min-samples-leaf", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--seed-shuffle-repetitions", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.lambdas = sorted(set(float(value) for value in args.lambdas))
    args.ridge_alphas = sorted(set(float(value) for value in args.ridge_alphas))
    if not args.lambdas or any(
        value < 0 or not math.isfinite(value) for value in args.lambdas
    ):
        parser.error("lambdas must contain finite non-negative values")
    if not args.ridge_alphas or any(value <= 0 for value in args.ridge_alphas):
        parser.error("ridge-alphas must be positive")
    if args.margin_temperature <= 0 or not math.isfinite(args.margin_temperature):
        parser.error("margin-temperature must be finite and positive")
    integer_values = (
        args.cv_folds,
        args.histgb_iterations,
        args.histgb_max_leaves,
        args.histgb_min_samples_leaf,
        args.bootstrap_samples,
        args.seed_shuffle_repetitions,
        args.inference_batch_size,
    )
    if any(value < 1 for value in integer_values):
        parser.error("fold/iteration/bootstrap/shuffle/batch values must be positive")
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def suffix_margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    reverse = torch.flip(logits[:, 1:], dims=[1])
    future_best = torch.flip(torch.cummax(reverse, dim=1).values, dims=[1])
    margin = logits.new_full(logits.shape, 30.0)
    margin[:, :-1] = logits[:, :-1] - future_best
    return margin


def load_b4_ensemble(
    runs_root: Path,
    dataset_manifest_sha256: str,
    candidate_count: int,
    device: torch.device,
) -> tuple[list[base.VariableLambdaB4Prior], list[dict[str, Any]]]:
    summary_paths = sorted(
        runs_root.glob("seed_*/run_summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if len(summary_paths) < 3:
        raise ValueError(f"Need at least three B4 runs under {runs_root}")
    models: list[base.VariableLambdaB4Prior] = []
    inputs: list[dict[str, Any]] = []
    signature = None
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("evaluation_split") != "validation" or summary.get(
            "test_accessed"
        ):
            raise ValueError(f"B4 run is not validation-only: {summary_path}")
        if summary.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError(f"B4 run dataset differs: {summary_path}")
        if "b4_offline" not in summary.get("model_types", []):
            raise ValueError(f"B4 run lacks b4_offline: {summary_path}")
        metadata = summary["artifacts"]["checkpoints"]["b4_offline"]
        checkpoint_path = summary_path.parent / metadata["path"]
        if sha256_file(checkpoint_path) != metadata["sha256"]:
            raise ValueError(f"B4 checkpoint SHA256 mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != "b4_offline":
            raise ValueError(f"Unexpected B4 checkpoint: {checkpoint_path}")
        observed_steps = tuple(int(value) for value in checkpoint["candidate_steps"])
        current_signature = (
            observed_steps,
            tuple(float(value) for value in summary["train_lambdas"]),
            str(summary["latency_profile"]["sha256"]),
        )
        if len(observed_steps) != candidate_count:
            raise ValueError(f"B4 candidate count mismatch: {checkpoint_path}")
        if signature is None:
            signature = current_signature
        elif signature != current_signature:
            raise ValueError(f"B4 ensemble protocol differs: {checkpoint_path}")
        model = base.VariableLambdaB4Prior(candidate_count, dropout=0.0)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        models.append(model)
        inputs.append(
            {
                "run_summary": str(summary_path),
                "run_summary_sha256": sha256_file(summary_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": metadata["sha256"],
                "train_seed": int(summary["train_seed"]),
            }
        )
    return models, inputs


@torch.no_grad()
def ensemble_b4_margins(
    models: list[base.VariableLambdaB4Prior],
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    count = len(trajectories)
    candidate_count = models[0].candidate_count
    result = np.empty((count, len(lambdas), candidate_count), dtype=np.float32)
    for lambda_index, lambda_value in enumerate(lambdas):
        for start in range(0, count, batch_size):
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
            logits = probabilities.clamp_min(1e-8).log()
            margins = suffix_margin_from_logits(logits)
            result[start : start + len(chunk), lambda_index] = margins.cpu().numpy()
    return result


@dataclass
class AuditRows:
    schedule: np.ndarray
    b4_margin: np.ndarray
    state: np.ndarray
    target_scaled_margin: np.ndarray
    target_probability: np.ndarray
    prompt_ids: np.ndarray
    trajectory_ids: np.ndarray
    lambda_ids: np.ndarray
    step_ids: np.ndarray
    utilities: np.ndarray
    trajectory_prompt_ids: np.ndarray


def build_audit_rows(
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    candidate_steps: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cost_profile: np.ndarray,
    b4_margins: np.ndarray,
    margin_temperature: float,
) -> AuditRows:
    trajectory_count = len(trajectories)
    lambda_count = len(lambdas)
    decision_count = len(candidate_steps) - 1
    states = np.stack(
        [
            (trajectory["features"] - state_mean) / state_std
            for trajectory in trajectories
        ]
    ).astype(np.float32)
    utilities = np.empty(
        (trajectory_count, lambda_count, len(candidate_steps)), dtype=np.float32
    )
    schedules = np.empty(
        (
            trajectory_count,
            lambda_count,
            len(candidate_steps),
            len(base.SCHEDULE_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    targets = np.empty(
        (trajectory_count, lambda_count, decision_count), dtype=np.float32
    )
    for trajectory_index, trajectory in enumerate(trajectories):
        for lambda_index, lambda_value in enumerate(lambdas):
            utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
            utilities[trajectory_index, lambda_index] = utility
            targets[trajectory_index, lambda_index] = soft.suffix_best_margin(
                utility, margin_temperature
            )[:-1]
            schedules[trajectory_index, lambda_index] = base.schedule_features(
                candidate_steps,
                trajectory["sigmas"],
                lambda_value,
                cost_profile,
            )
    target_scaled = np.clip(targets, -10.0, 10.0)
    target_probability = 1.0 / (1.0 + np.exp(-target_scaled))
    row_shape = trajectory_count * lambda_count * decision_count
    trajectory_ids = np.repeat(
        np.arange(trajectory_count, dtype=np.int64), lambda_count * decision_count
    )
    lambda_ids = np.tile(
        np.repeat(np.arange(lambda_count, dtype=np.int64), decision_count),
        trajectory_count,
    )
    step_ids = np.tile(
        np.arange(decision_count, dtype=np.int64), trajectory_count * lambda_count
    )
    prompt_by_trajectory = np.asarray(
        [int(trajectory["prompt_id"]) for trajectory in trajectories], dtype=np.int64
    )
    prompt_ids = prompt_by_trajectory[trajectory_ids]
    state_rows = states[trajectory_ids, step_ids]
    if state_rows.shape[0] != row_shape:
        raise RuntimeError("Internal row construction mismatch")
    return AuditRows(
        schedule=schedules[:, :, :decision_count].reshape(row_shape, -1),
        b4_margin=b4_margins[:, :, :decision_count].reshape(row_shape, 1),
        state=state_rows,
        target_scaled_margin=target_scaled.reshape(row_shape),
        target_probability=target_probability.reshape(row_shape),
        prompt_ids=prompt_ids,
        trajectory_ids=trajectory_ids,
        lambda_ids=lambda_ids,
        step_ids=step_ids,
        utilities=utilities,
        trajectory_prompt_ids=prompt_by_trajectory,
    )


def select_global_ridge_alpha(
    rows: AuditRows,
    alphas: list[float],
    folds: int,
) -> tuple[float, list[dict[str, float]]]:
    features = np.concatenate([rows.schedule, rows.b4_margin, rows.state], axis=1)
    unique_prompts = np.unique(rows.prompt_ids)
    if folds > len(unique_prompts):
        raise ValueError("cv-folds exceeds train prompt count")
    splitter = GroupKFold(n_splits=folds)
    results = []
    for alpha in alphas:
        prompt_errors: dict[int, list[float]] = {}
        for train_index, validation_index in splitter.split(
            features, rows.target_scaled_margin, groups=rows.prompt_ids
        ):
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
            )
            model.fit(features[train_index], rows.target_scaled_margin[train_index])
            prediction = model.predict(features[validation_index])
            for prompt_id, error in zip(
                rows.prompt_ids[validation_index],
                np.abs(prediction - rows.target_scaled_margin[validation_index]),
            ):
                prompt_errors.setdefault(int(prompt_id), []).append(float(error))
        score = float(np.mean([np.mean(values) for values in prompt_errors.values()]))
        results.append({"alpha": float(alpha), "prompt_macro_mae": score})
    selected = min(results, key=lambda row: row["prompt_macro_mae"])["alpha"]
    return float(selected), results


def make_regressor(family: str, alpha: float, args: argparse.Namespace) -> Any:
    if family == "ridge":
        return make_pipeline(
            StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
        )
    if family == "histgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=args.histgb_iterations,
            max_leaf_nodes=args.histgb_max_leaves,
            min_samples_leaf=args.histgb_min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.bootstrap_seed,
        )
    raise ValueError(f"Unknown model family: {family}")


def feature_matrix(
    rows: AuditRows, indices: np.ndarray | None, include_b4: bool
) -> np.ndarray:
    parts = [rows.schedule]
    if include_b4:
        parts.append(rows.b4_margin)
    if indices is not None and indices.size:
        parts.append(rows.state[:, indices])
    return np.concatenate(parts, axis=1)


def prompt_bootstrap(
    values: np.ndarray,
    prompt_ids: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    prompts = np.unique(prompt_ids)
    means = np.asarray(
        [values[prompt_ids == prompt].mean() for prompt in prompts], dtype=np.float64
    )
    indices = rng.integers(0, len(means), size=(samples, len(means)))
    draws = means[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(means.mean()), float(low), float(high)


def policy_from_margin(
    predicted_margin: np.ndarray,
    rows: AuditRows,
    candidate_count: int,
    lambda_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory_count = len(rows.trajectory_prompt_ids)
    margins = predicted_margin.reshape(
        trajectory_count, lambda_count, candidate_count - 1
    )
    eligible = margins >= 0.0
    has_stop = eligible.any(axis=2)
    chosen = np.where(has_stop, eligible.argmax(axis=2), candidate_count - 1)
    oracle = rows.utilities.argmax(axis=2)
    realized = np.take_along_axis(rows.utilities, chosen[..., None], axis=2)[..., 0]
    oracle_utility = np.take_along_axis(rows.utilities, oracle[..., None], axis=2)[
        ..., 0
    ]
    regret = np.maximum(oracle_utility - realized, 0.0)
    return chosen.astype(np.int64), regret.astype(np.float32)


def first_nonnegative_margin(margins: np.ndarray) -> np.ndarray:
    eligible = margins[..., :-1] >= 0.0
    has_stop = eligible.any(axis=-1)
    return np.where(has_stop, eligible.argmax(axis=-1), margins.shape[-1] - 1).astype(
        np.int64
    )


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    if denominator <= 0:
        return 0.0
    return 1.0 - float(np.square(target - prediction).sum()) / denominator


def evaluate_prediction(
    name: str,
    family: str,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    rows: AuditRows,
    baseline_regret: np.ndarray,
    b4_chosen: np.ndarray,
    candidate_steps: np.ndarray,
    lambdas: list[float],
    feature_count: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction = np.clip(prediction, -10.0, 10.0)
    baseline_prediction = np.clip(baseline_prediction, -10.0, 10.0)
    predicted_probability = 1.0 / (1.0 + np.exp(-prediction))
    baseline_probability = 1.0 / (1.0 + np.exp(-baseline_prediction))
    absolute_error = np.abs(prediction - rows.target_scaled_margin)
    baseline_absolute_error = np.abs(baseline_prediction - rows.target_scaled_margin)
    brier = np.square(predicted_probability - rows.target_probability)
    baseline_brier = np.square(baseline_probability - rows.target_probability)
    candidate_count = len(candidate_steps)
    chosen, regret = policy_from_margin(prediction, rows, candidate_count, len(lambdas))
    baseline_chosen, _ = policy_from_margin(
        baseline_prediction, rows, candidate_count, len(lambdas)
    )
    oracle_chosen = rows.utilities.argmax(axis=2)
    switch_error = np.abs(candidate_steps[chosen] - candidate_steps[oracle_chosen])
    baseline_switch_error = np.abs(
        candidate_steps[baseline_chosen] - candidate_steps[oracle_chosen]
    )
    row_prompt_ids = rows.prompt_ids
    trajectory_prompt_ids = rows.trajectory_prompt_ids
    mae_delta = prompt_bootstrap(
        baseline_absolute_error - absolute_error,
        row_prompt_ids,
        bootstrap_samples,
        rng,
    )
    brier_delta = prompt_bootstrap(
        baseline_brier - brier, row_prompt_ids, bootstrap_samples, rng
    )
    regret_delta = prompt_bootstrap(
        (baseline_regret - regret).reshape(-1),
        np.repeat(trajectory_prompt_ids, len(lambdas)),
        bootstrap_samples,
        rng,
    )
    switch_delta = prompt_bootstrap(
        (baseline_switch_error - switch_error).reshape(-1),
        np.repeat(trajectory_prompt_ids, len(lambdas)),
        bootstrap_samples,
        rng,
    )
    summary = {
        "model_family": family,
        "factor_group": name,
        "feature_count": feature_count,
        "validation_r2": r2_score(rows.target_scaled_margin, prediction),
        "delta_r2_vs_schedule_b4": r2_score(rows.target_scaled_margin, prediction)
        - r2_score(rows.target_scaled_margin, baseline_prediction),
        "validation_scaled_margin_mae": float(absolute_error.mean()),
        "delta_mae_vs_schedule_b4": mae_delta[0],
        "delta_mae_ci95_low": mae_delta[1],
        "delta_mae_ci95_high": mae_delta[2],
        "validation_brier": float(brier.mean()),
        "delta_brier_vs_schedule_b4": brier_delta[0],
        "delta_brier_ci95_low": brier_delta[1],
        "delta_brier_ci95_high": brier_delta[2],
        "validation_policy_regret": float(regret.mean()),
        "delta_policy_regret_vs_schedule_b4": regret_delta[0],
        "delta_policy_regret_ci95_low": regret_delta[1],
        "delta_policy_regret_ci95_high": regret_delta[2],
        "validation_oracle_switch_step_mae": float(switch_error.mean()),
        "delta_switch_step_mae_vs_schedule_b4": switch_delta[0],
        "delta_switch_step_mae_ci95_low": switch_delta[1],
        "delta_switch_step_mae_ci95_high": switch_delta[2],
        "decision_change_rate_vs_b4": float((chosen != b4_chosen).mean()),
    }
    per_lambda_rows = []
    for lambda_index, lambda_value in enumerate(lambdas):
        delta = baseline_regret[:, lambda_index] - regret[:, lambda_index]
        point, low, high = prompt_bootstrap(
            delta,
            trajectory_prompt_ids,
            bootstrap_samples,
            rng,
        )
        switch_point, switch_low, switch_high = prompt_bootstrap(
            baseline_switch_error[:, lambda_index] - switch_error[:, lambda_index],
            trajectory_prompt_ids,
            bootstrap_samples,
            rng,
        )
        per_lambda_rows.append(
            {
                "model_family": family,
                "factor_group": name,
                "lambda": lambda_value,
                "policy_regret": float(regret[:, lambda_index].mean()),
                "delta_policy_regret_vs_schedule_b4": point,
                "ci95_low": low,
                "ci95_high": high,
                "oracle_switch_step_mae": float(switch_error[:, lambda_index].mean()),
                "delta_switch_step_mae_vs_schedule_b4": switch_point,
                "delta_switch_step_mae_ci95_low": switch_low,
                "delta_switch_step_mae_ci95_high": switch_high,
                "decision_change_rate_vs_b4": float(
                    (chosen[:, lambda_index] != b4_chosen[:, lambda_index]).mean()
                ),
            }
        )
    return summary, per_lambda_rows


def centered_correlations(
    state: np.ndarray,
    target: np.ndarray,
    prompt_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return step/lambda-controlled and within-prompt-seed correlations."""
    trajectory_count, decision_count, feature_count = state.shape
    lambda_count = target.shape[1]
    controlled_num = np.zeros(feature_count, dtype=np.float64)
    controlled_xss = np.zeros(feature_count, dtype=np.float64)
    controlled_yss = 0.0
    within_num = np.zeros(feature_count, dtype=np.float64)
    within_xss = np.zeros(feature_count, dtype=np.float64)
    within_yss = 0.0
    prompt_groups = [
        np.flatnonzero(prompt_ids == prompt) for prompt in np.unique(prompt_ids)
    ]
    for lambda_index in range(lambda_count):
        for step_index in range(decision_count):
            x = state[:, step_index].astype(np.float64)
            y = target[:, lambda_index, step_index].astype(np.float64)
            x_centered = x - x.mean(axis=0, keepdims=True)
            y_centered = y - y.mean()
            controlled_num += (x_centered * y_centered[:, None]).sum(axis=0)
            controlled_xss += np.square(x_centered).sum(axis=0)
            controlled_yss += float(np.square(y_centered).sum())
            for indices in prompt_groups:
                if len(indices) < 2:
                    continue
                prompt_x = x[indices] - x[indices].mean(axis=0, keepdims=True)
                prompt_y = y[indices] - y[indices].mean()
                within_num += (prompt_x * prompt_y[:, None]).sum(axis=0)
                within_xss += np.square(prompt_x).sum(axis=0)
                within_yss += float(np.square(prompt_y).sum())
    controlled = controlled_num / np.sqrt(
        np.maximum(controlled_xss * controlled_yss, 1e-20)
    )
    within = within_num / np.sqrt(np.maximum(within_xss * within_yss, 1e-20))
    return controlled, within


def shuffle_state_within_step(
    state_by_trajectory: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = np.empty_like(state_by_trajectory)
    for step_index in range(state_by_trajectory.shape[1]):
        result[:, step_index] = state_by_trajectory[
            rng.permutation(state_by_trajectory.shape[0]), step_index
        ]
    return result


def shuffle_state_within_prompt(
    state: np.ndarray, prompt_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = state.copy()
    for prompt in np.unique(prompt_ids):
        indices = np.flatnonzero(prompt_ids == prompt)
        if len(indices) > 1:
            for step_index in range(state.shape[1]):
                result[indices, step_index] = state[
                    rng.permutation(indices), step_index
                ]
    return result


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    runs_root = Path(args.b4_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    manifest = base.load_dataset_manifest(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    feature_indices = np.arange(int(manifest["feature_count"]), dtype=np.int64)
    train_trajectories = base.load_trajectories(
        dataset_dir, manifest, "train", feature_indices
    )
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", feature_indices
    )
    train_prompts = {int(item["prompt_id"]) for item in train_trajectories}
    validation_prompts = {int(item["prompt_id"]) for item in validation_trajectories}
    if train_prompts & validation_prompts:
        raise ValueError("Train and validation prompts overlap")
    if len(validation_prompts) != 200 or len(validation_trajectories) != 600:
        raise ValueError(
            "Factor audit requires the formal 200-prompt, three-seed validation"
        )
    validation_seed_counts = {
        prompt: sum(
            int(item["prompt_id"]) == prompt for item in validation_trajectories
        )
        for prompt in validation_prompts
    }
    if set(validation_seed_counts.values()) != {3}:
        raise ValueError("Every validation prompt must have exactly three trajectories")
    candidate_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    state_mean, state_std = soft.fit_step_state_normalizer(train_trajectories)
    (
        cost_profile,
        candidate_seconds,
        native_seconds,
        latency_provenance,
    ) = base.load_locked_latency_profile(manifest, candidate_steps, None)
    base.apply_locked_latency_profile(
        train_trajectories, cost_profile, candidate_seconds, native_seconds
    )
    base.apply_locked_latency_profile(
        validation_trajectories, cost_profile, candidate_seconds, native_seconds
    )
    device = torch.device(args.device)
    models, b4_inputs = load_b4_ensemble(
        runs_root, manifest_sha256, len(candidate_steps), device
    )
    print("Computing B4 ensemble margins...")
    train_b4 = ensemble_b4_margins(
        models, train_trajectories, args.lambdas, device, args.inference_batch_size
    )
    validation_b4 = ensemble_b4_margins(
        models,
        validation_trajectories,
        args.lambdas,
        device,
        args.inference_batch_size,
    )
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_rows = build_audit_rows(
        train_trajectories,
        args.lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        train_b4,
        args.margin_temperature,
    )
    validation_rows = build_audit_rows(
        validation_trajectories,
        args.lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        validation_b4,
        args.margin_temperature,
    )
    print("Selecting one train-only Ridge alpha for all factor groups...")
    selected_alpha, alpha_rows = select_global_ridge_alpha(
        train_rows, args.ridge_alphas, args.cv_folds
    )
    group_indices = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in manifest["feature_groups"].items()
    }
    group_indices["convergence"] = np.unique(
        np.concatenate(
            [
                group_indices["residual_global"],
                group_indices["residual_channel"],
                group_indices["trajectory_delta"],
            ]
        )
    )
    group_indices["content"] = np.unique(
        np.concatenate(
            [
                group_indices["x0_global"],
                group_indices["x0_channel"],
                group_indices["local_energy"],
            ]
        )
    )
    group_indices["all_state"] = np.arange(
        int(manifest["feature_count"]), dtype=np.int64
    )
    model_specs: list[tuple[str, np.ndarray | None, bool]] = [
        ("schedule", None, False),
        ("schedule_b4", None, True),
        *[(name, indices, True) for name, indices in group_indices.items()],
    ]
    rng = np.random.default_rng(args.bootstrap_seed)
    group_rows: list[dict[str, Any]] = []
    per_lambda_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    policy_regrets: dict[tuple[str, str], np.ndarray] = {}
    b4_chosen = first_nonnegative_margin(validation_b4)
    b4_oracle = validation_rows.utilities.argmax(axis=2)
    b4_realized = np.take_along_axis(
        validation_rows.utilities, b4_chosen[..., None], axis=2
    )[..., 0]
    oracle_utility = np.take_along_axis(
        validation_rows.utilities, b4_oracle[..., None], axis=2
    )[..., 0]
    b4_regret = np.maximum(oracle_utility - b4_realized, 0.0)
    for family in MODEL_FAMILIES:
        print(f"Fitting {family} factor models...")
        for name, indices, include_b4 in model_specs:
            train_x = feature_matrix(train_rows, indices, include_b4)
            validation_x = feature_matrix(validation_rows, indices, include_b4)
            model = make_regressor(family, selected_alpha, args)
            model.fit(train_x, train_rows.target_scaled_margin)
            prediction = model.predict(validation_x).astype(np.float32)
            predictions[(family, name)] = prediction
            del model, train_x, validation_x
        baseline_prediction = predictions[(family, "schedule_b4")]
        _, baseline_regret = policy_from_margin(
            baseline_prediction,
            validation_rows,
            len(candidate_steps),
            len(args.lambdas),
        )
        for name, indices, include_b4 in model_specs:
            prediction = predictions[(family, name)]
            summary, lambda_results = evaluate_prediction(
                name,
                family,
                prediction,
                baseline_prediction,
                validation_rows,
                baseline_regret,
                b4_chosen,
                candidate_steps,
                args.lambdas,
                0 if indices is None else int(indices.size),
                args.bootstrap_samples,
                rng,
            )
            group_rows.append(summary)
            per_lambda_rows.extend(lambda_results)
            _, regret = policy_from_margin(
                prediction,
                validation_rows,
                len(candidate_steps),
                len(args.lambdas),
            )
            policy_regrets[(family, name)] = regret
    print("Fitting within-step shuffled Ridge controls...")
    train_state_by_trajectory = np.stack(
        [
            (trajectory["features"] - state_mean) / state_std
            for trajectory in train_trajectories
        ]
    ).astype(np.float32)
    shuffled_state = shuffle_state_within_step(train_state_by_trajectory, rng)
    shuffled_train_rows = train_rows.state.copy()
    shuffled_train_rows[:] = shuffled_state[
        train_rows.trajectory_ids, train_rows.step_ids
    ]
    shuffle_rows = []
    ridge_baseline = predictions[("ridge", "schedule_b4")]
    ridge_baseline_regret = policy_regrets[("ridge", "schedule_b4")]
    for name, indices in group_indices.items():
        train_x = np.concatenate(
            [
                train_rows.schedule,
                train_rows.b4_margin,
                shuffled_train_rows[:, indices],
            ],
            axis=1,
        )
        validation_x = feature_matrix(validation_rows, indices, True)
        model = make_regressor("ridge", selected_alpha, args)
        model.fit(train_x, train_rows.target_scaled_margin)
        prediction = model.predict(validation_x).astype(np.float32)
        summary, _ = evaluate_prediction(
            name,
            "ridge_train_within_step_shuffle",
            prediction,
            ridge_baseline,
            validation_rows,
            ridge_baseline_regret,
            b4_chosen,
            candidate_steps,
            args.lambdas,
            int(indices.size),
            args.bootstrap_samples,
            rng,
        )
        shuffle_rows.append(summary)
    print("Computing individual-factor controlled associations...")
    validation_state = np.stack(
        [
            (trajectory["features"] - state_mean) / state_std
            for trajectory in validation_trajectories
        ]
    ).astype(np.float32)
    validation_target = np.clip(
        np.stack(
            [
                [
                    soft.suffix_best_margin(
                        trajectory["qualities"] - lambda_value * trajectory["costs"],
                        args.margin_temperature,
                    )[:-1]
                    for lambda_value in args.lambdas
                ]
                for trajectory in validation_trajectories
            ]
        ),
        -10.0,
        10.0,
    )
    controlled, within_prompt = centered_correlations(
        validation_state[:, :-1],
        validation_target,
        validation_rows.trajectory_prompt_ids,
    )
    shuffled_correlations = []
    for _ in range(args.seed_shuffle_repetitions):
        seed_shuffled = shuffle_state_within_prompt(
            validation_state[:, :-1], validation_rows.trajectory_prompt_ids, rng
        )
        _, shuffled_within = centered_correlations(
            seed_shuffled,
            validation_target,
            validation_rows.trajectory_prompt_ids,
        )
        shuffled_correlations.append(shuffled_within)
    shuffled_matrix = np.stack(shuffled_correlations)
    feature_to_group = {}
    for group, indices in manifest["feature_groups"].items():
        for index in indices:
            feature_to_group[int(index)] = group
    individual_rows = []
    for index, feature_name in enumerate(manifest["feature_names"]):
        individual_rows.append(
            {
                "feature_index": index,
                "feature_name": feature_name,
                "feature_group": feature_to_group[index],
                "step_lambda_controlled_correlation": float(controlled[index]),
                "within_prompt_seed_correlation": float(within_prompt[index]),
                "seed_shuffle_correlation_mean": float(
                    shuffled_matrix[:, index].mean()
                ),
                "seed_shuffle_correlation_std": float(shuffled_matrix[:, index].std()),
                "absolute_signal_above_shuffle": float(
                    abs(within_prompt[index])
                    - np.mean(np.abs(shuffled_matrix[:, index]))
                ),
            }
        )
    group_association_rows = []
    for name, indices in group_indices.items():
        group_association_rows.append(
            {
                "factor_group": name,
                "feature_count": int(indices.size),
                "mean_abs_step_lambda_controlled_correlation": float(
                    np.mean(np.abs(controlled[indices]))
                ),
                "max_abs_step_lambda_controlled_correlation": float(
                    np.max(np.abs(controlled[indices]))
                ),
                "mean_abs_within_prompt_seed_correlation": float(
                    np.mean(np.abs(within_prompt[indices]))
                ),
                "max_abs_within_prompt_seed_correlation": float(
                    np.max(np.abs(within_prompt[indices]))
                ),
                "mean_absolute_signal_above_shuffle": float(
                    np.mean(
                        np.abs(within_prompt[indices])
                        - np.mean(np.abs(shuffled_matrix[:, indices]), axis=0)
                    )
                ),
            }
        )
    out_dir.mkdir(parents=True)
    artifact_rows = {
        "factor_group_predictive_value": (
            "factor_group_predictive_value.csv",
            group_rows,
        ),
        "per_lambda_factor_results": (
            "per_lambda_factor_results.csv",
            per_lambda_rows,
        ),
        "shuffle_negative_controls": (
            "shuffle_negative_controls.csv",
            shuffle_rows,
        ),
        "individual_factor_association": (
            "individual_factor_association.csv",
            individual_rows,
        ),
        "factor_group_association": (
            "factor_group_association.csv",
            group_association_rows,
        ),
        "ridge_alpha_selection": ("ridge_alpha_selection.csv", alpha_rows),
    }
    for _, (filename, rows) in artifact_rows.items():
        write_csv(out_dir / filename, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "factor_relation_to_variable_lambda_oracle_handoff",
        "diagnostic_only": True,
        "test_accessed": False,
        "primary_target": "signed_suffix_best_utility_margin",
        "secondary_target": "oracle_utility_argmax_step",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "train_prompt_count": len(train_prompts),
        "train_trajectory_count": len(train_trajectories),
        "validation_prompt_count": len(validation_prompts),
        "validation_trajectory_count": len(validation_trajectories),
        "validation_seed_count_per_prompt": 3,
        "candidate_steps": candidate_steps.tolist(),
        "lambdas": args.lambdas,
        "margin_temperature": args.margin_temperature,
        "selected_ridge_alpha": selected_alpha,
        "model_families": list(MODEL_FAMILIES),
        "state_normalization": "train_per_candidate_step_v1",
        "baseline": "schedule_plus_five_seed_b4_probability_ensemble_margin",
        "b4_ensemble_macro_policy_regret": float(b4_regret.mean()),
        "latency_profile": latency_provenance,
        "b4_inputs": b4_inputs,
        "bootstrap": {
            "unit": "prompt_after_averaging_generation_seeds_lambdas_and_steps",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "negative_controls": {
            "train_within_step_shuffle": True,
            "validation_within_prompt_seed_shuffle_repetitions": args.seed_shuffle_repetitions,
        },
        "artifacts": {
            name: {
                "path": filename,
                "sha256": sha256_file(out_dir / filename),
            }
            for name, (filename, _) in artifact_rows.items()
        },
    }
    report_path = out_dir / "factor_relevance_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
