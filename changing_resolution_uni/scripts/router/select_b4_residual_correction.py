#!/usr/bin/env python3
"""Select conservative online state corrections on top of a frozen B4 ensemble."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import analyze_factor_relevance as audit
    from . import train_soft_margin_router as soft
    from . import train_variable_lambda_router as base
except ImportError:
    import analyze_factor_relevance as audit
    import train_soft_margin_router as soft
    import train_variable_lambda_router as base


REPORT_SCHEMA = "variable_lambda_b4_residual_correction_selection_v1"
MODEL_FAMILIES = ("ridge", "histgb")
FACTOR_GROUPS = (
    "schedule_control",
    "trajectory_delta_rms_per_sigma",
    "x0_temporal",
    "combined",
)


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
        "--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0]
    )
    parser.add_argument(
        "--correction-scales", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40]
    )
    parser.add_argument(
        "--gate-thresholds", type=float, nargs="+", default=[0.25, 0.50, 1.0, 2.0]
    )
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--residual-target-clip", type=float, default=10.0)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--histgb-iterations", type=int, default=100)
    parser.add_argument("--histgb-max-leaves", type=int, default=15)
    parser.add_argument("--histgb-min-samples-leaf", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2028)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.lambdas = sorted(set(float(value) for value in args.lambdas))
    args.ridge_alphas = sorted(set(float(value) for value in args.ridge_alphas))
    args.correction_scales = sorted(
        set(float(value) for value in args.correction_scales)
    )
    args.gate_thresholds = sorted(set(float(value) for value in args.gate_thresholds))
    if not args.lambdas or any(
        value < 0 or not math.isfinite(value) for value in args.lambdas
    ):
        parser.error("lambdas must contain finite non-negative values")
    if any(value <= 0 or not math.isfinite(value) for value in args.ridge_alphas):
        parser.error("ridge-alphas must contain finite positive values")
    if (
        not args.correction_scales
        or not args.gate_thresholds
        or any(
            value <= 0 or not math.isfinite(value)
            for value in args.correction_scales + args.gate_thresholds
        )
    ):
        parser.error(
            "correction scales and gate thresholds must be finite and positive"
        )
    positive_floats = (
        args.margin_temperature,
        args.residual_clip,
        args.residual_target_clip,
    )
    if any(value <= 0 or not math.isfinite(value) for value in positive_floats):
        parser.error("temperature and residual clips must be finite and positive")
    positive_integers = (
        args.cv_folds,
        args.histgb_iterations,
        args.histgb_max_leaves,
        args.histgb_min_samples_leaf,
        args.bootstrap_samples,
        args.inference_batch_size,
    )
    if any(value < 1 for value in positive_integers):
        parser.error("fold/iteration/bootstrap/batch values must be positive")
    return args


def select_factor_indices(feature_names: list[str]) -> dict[str, np.ndarray]:
    trajectory = np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if name == "trajectory.delta_rms_per_sigma"
            or name.startswith("trajectory.delta_rms_per_sigma.channel_")
        ],
        dtype=np.int64,
    )
    x0_temporal_names = {
        "x0.temporal_gradient_abs_mean",
        "x0.temporal_second_abs_mean",
    }
    x0_temporal = np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if name in x0_temporal_names
        ],
        dtype=np.int64,
    )
    if trajectory.size != 17:
        raise ValueError(
            "Expected one global and sixteen channel delta_rms_per_sigma features; "
            f"found {trajectory.size}"
        )
    if x0_temporal.size != 2:
        raise ValueError(f"Expected two x0 temporal features; found {x0_temporal.size}")
    return {
        "schedule_control": np.empty(0, dtype=np.int64),
        "trajectory_delta_rms_per_sigma": trajectory,
        "x0_temporal": x0_temporal,
        "combined": np.unique(np.concatenate([trajectory, x0_temporal])),
    }


def correction_features(rows: audit.AuditRows, indices: np.ndarray) -> np.ndarray:
    parts = [rows.schedule, rows.b4_margin]
    if indices.size:
        parts.append(rows.state[:, indices])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def make_regressor(
    family: str, model_parameter: float, args: argparse.Namespace
) -> Any:
    if family == "ridge":
        return make_pipeline(
            StandardScaler(), Ridge(alpha=model_parameter, solver="lsqr", tol=1e-4)
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


def model_parameter_grid(family: str, args: argparse.Namespace) -> list[float]:
    if family == "ridge":
        return args.ridge_alphas
    if family == "histgb":
        return [1.0]
    raise ValueError(f"Unknown model family: {family}")


def fit_oof_residual(
    family: str,
    model_parameter: float,
    features: np.ndarray,
    target: np.ndarray,
    prompt_ids: np.ndarray,
    folds: int,
    args: argparse.Namespace,
) -> np.ndarray:
    unique_prompts = np.unique(prompt_ids)
    if folds > len(unique_prompts):
        raise ValueError("cv-folds exceeds train prompt count")
    prediction = np.full(target.shape, np.nan, dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fit_indices, heldout_indices in splitter.split(
        features, target, groups=prompt_ids
    ):
        model = make_regressor(family, model_parameter, args)
        model.fit(features[fit_indices], target[fit_indices])
        prediction[heldout_indices] = model.predict(features[heldout_indices]).astype(
            np.float32
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("OOF residual prediction is incomplete or non-finite")
    return prediction


def apply_residual_correction(
    b4_margin: np.ndarray,
    predicted_residual: np.ndarray,
    correction_scale: float,
    gate_threshold: float,
    residual_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_margin = np.asarray(b4_margin, dtype=np.float32)
    residual = np.asarray(predicted_residual, dtype=np.float32)
    if base_margin.shape != residual.shape:
        raise ValueError("B4 margin and residual prediction shapes differ")
    if correction_scale == 0.0:
        zeros = np.zeros(base_margin.shape, dtype=bool)
        return base_margin.copy(), zeros, np.zeros_like(base_margin)
    gate = np.abs(base_margin) <= float(gate_threshold)
    applied = np.where(
        gate,
        float(correction_scale) * np.clip(residual, -residual_clip, residual_clip),
        0.0,
    ).astype(np.float32)
    corrected = (base_margin + applied).astype(np.float32)
    if not np.isfinite(corrected).all():
        raise ValueError("Corrected margins contain non-finite values")
    return corrected, gate, applied


def policy_outputs(
    margins: np.ndarray,
    rows: audit.AuditRows,
    candidate_steps: np.ndarray,
    lambdas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen, regret = audit.policy_from_margin(
        margins, rows, len(candidate_steps), len(lambdas)
    )
    realized = np.take_along_axis(rows.utilities, chosen[..., None], axis=2)[..., 0]
    return chosen, regret, realized


def select_oof_configuration(
    family: str,
    factor_group: str,
    features: np.ndarray,
    residual_target: np.ndarray,
    rows: audit.AuditRows,
    candidate_steps: np.ndarray,
    lambdas: list[float],
    b4_chosen: np.ndarray,
    b4_regret: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_margin = rows.b4_margin[:, 0]
    grid_rows: list[dict[str, Any]] = []
    predictions: dict[float, np.ndarray] = {}
    for model_parameter in model_parameter_grid(family, args):
        predictions[model_parameter] = fit_oof_residual(
            family,
            model_parameter,
            features,
            residual_target,
            rows.prompt_ids,
            args.cv_folds,
            args,
        )
        configurations = [(0.0, 0.0)] + [
            (scale, threshold)
            for scale in args.correction_scales
            for threshold in args.gate_thresholds
        ]
        for scale, threshold in configurations:
            corrected, gate, applied = apply_residual_correction(
                base_margin,
                predictions[model_parameter],
                scale,
                threshold,
                args.residual_clip,
            )
            chosen, regret, _ = policy_outputs(
                corrected, rows, candidate_steps, lambdas
            )
            grid_rows.append(
                {
                    "model_family": family,
                    "factor_group": factor_group,
                    "model_parameter": model_parameter,
                    "correction_scale": scale,
                    "gate_threshold": threshold,
                    "train_oof_policy_regret": float(regret.mean()),
                    "delta_policy_regret_vs_frozen_b4": float(
                        b4_regret.mean() - regret.mean()
                    ),
                    "decision_change_rate_vs_frozen_b4": float(
                        (chosen != b4_chosen).mean()
                    ),
                    "row_gate_rate": float(gate.mean()),
                    "mean_abs_applied_correction": float(np.abs(applied).mean()),
                    "selected": False,
                }
            )
    selected = min(
        grid_rows,
        key=lambda row: (
            float(row["train_oof_policy_regret"]),
            float(row["decision_change_rate_vs_frozen_b4"]),
            float(row["correction_scale"]),
            float(row["gate_threshold"]),
            -float(row["model_parameter"]),
        ),
    )
    selected["selected"] = True
    return selected, grid_rows


def bootstrap_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    prompt_ids: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    return audit.prompt_bootstrap(
        (baseline - candidate).reshape(-1),
        prompt_ids,
        samples,
        rng,
    )


def evaluate_candidate(
    family: str,
    factor_group: str,
    selected: dict[str, Any],
    corrected_margin: np.ndarray,
    gate: np.ndarray,
    applied: np.ndarray,
    feature_count: int,
    rows: audit.AuditRows,
    candidate_steps: np.ndarray,
    lambdas: list[float],
    b4_chosen: np.ndarray,
    b4_regret: np.ndarray,
    b4_realized: np.ndarray,
    schedule_control: tuple[np.ndarray, np.ndarray] | None,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    chosen, regret, realized = policy_outputs(
        corrected_margin, rows, candidate_steps, lambdas
    )
    trajectory_prompt_ids = rows.trajectory_prompt_ids
    repeated_prompt_ids = np.repeat(trajectory_prompt_ids, len(lambdas))
    delta_b4 = bootstrap_delta(
        b4_regret, regret, repeated_prompt_ids, bootstrap_samples, rng
    )
    if schedule_control is None:
        delta_control = (0.0, 0.0, 0.0)
    else:
        _, control_regret = schedule_control
        delta_control = bootstrap_delta(
            control_regret, regret, repeated_prompt_ids, bootstrap_samples, rng
        )
    oracle = rows.utilities.argmax(axis=2)
    b4_step_error = np.abs(candidate_steps[b4_chosen] - candidate_steps[oracle])
    step_error = np.abs(candidate_steps[chosen] - candidate_steps[oracle])
    delta_step = bootstrap_delta(
        b4_step_error, step_error, repeated_prompt_ids, bootstrap_samples, rng
    )
    changed = chosen != b4_chosen
    utility_delta = realized - b4_realized
    changed_count = int(changed.sum())
    improved_count = int((changed & (utility_delta > 1e-8)).sum())
    worsened_count = int((changed & (utility_delta < -1e-8)).sum())
    equal_count = changed_count - improved_count - worsened_count
    target = rows.target_scaled_margin
    base_margin = rows.b4_margin[:, 0]
    absolute_error = np.abs(corrected_margin - target)
    base_absolute_error = np.abs(base_margin - target)
    margin_delta = audit.prompt_bootstrap(
        base_absolute_error - absolute_error,
        rows.prompt_ids,
        bootstrap_samples,
        rng,
    )
    summary = {
        "model_family": family,
        "factor_group": factor_group,
        "feature_count": feature_count,
        "model_parameter": float(selected["model_parameter"]),
        "correction_scale": float(selected["correction_scale"]),
        "gate_threshold": float(selected["gate_threshold"]),
        "train_oof_policy_regret": float(selected["train_oof_policy_regret"]),
        "validation_policy_regret": float(regret.mean()),
        "delta_policy_regret_vs_frozen_b4": delta_b4[0],
        "delta_policy_regret_vs_frozen_b4_ci95_low": delta_b4[1],
        "delta_policy_regret_vs_frozen_b4_ci95_high": delta_b4[2],
        "delta_policy_regret_vs_schedule_control": delta_control[0],
        "delta_policy_regret_vs_schedule_control_ci95_low": delta_control[1],
        "delta_policy_regret_vs_schedule_control_ci95_high": delta_control[2],
        "validation_margin_mae": float(absolute_error.mean()),
        "delta_margin_mae_vs_frozen_b4": margin_delta[0],
        "delta_margin_mae_vs_frozen_b4_ci95_low": margin_delta[1],
        "delta_margin_mae_vs_frozen_b4_ci95_high": margin_delta[2],
        "validation_oracle_switch_step_mae": float(step_error.mean()),
        "delta_switch_step_mae_vs_frozen_b4": delta_step[0],
        "delta_switch_step_mae_vs_frozen_b4_ci95_low": delta_step[1],
        "delta_switch_step_mae_vs_frozen_b4_ci95_high": delta_step[2],
        "decision_change_rate_vs_frozen_b4": float(changed.mean()),
        "changed_decision_count": changed_count,
        "changed_better_count": improved_count,
        "changed_worse_count": worsened_count,
        "changed_equal_count": equal_count,
        "row_gate_rate": float(gate.mean()),
        "mean_abs_applied_correction": float(np.abs(applied).mean()),
        "p95_abs_applied_correction": float(np.quantile(np.abs(applied), 0.95)),
    }
    per_lambda_rows: list[dict[str, Any]] = []
    for lambda_index, lambda_value in enumerate(lambdas):
        lambda_delta = bootstrap_delta(
            b4_regret[:, lambda_index],
            regret[:, lambda_index],
            trajectory_prompt_ids,
            bootstrap_samples,
            rng,
        )
        lambda_changed = changed[:, lambda_index]
        lambda_utility_delta = utility_delta[:, lambda_index]
        if schedule_control is None:
            lambda_control_delta = (0.0, 0.0, 0.0)
        else:
            lambda_control_delta = bootstrap_delta(
                schedule_control[1][:, lambda_index],
                regret[:, lambda_index],
                trajectory_prompt_ids,
                bootstrap_samples,
                rng,
            )
        per_lambda_rows.append(
            {
                "model_family": family,
                "factor_group": factor_group,
                "lambda": lambda_value,
                "policy_regret": float(regret[:, lambda_index].mean()),
                "delta_policy_regret_vs_frozen_b4": lambda_delta[0],
                "ci95_low": lambda_delta[1],
                "ci95_high": lambda_delta[2],
                "delta_policy_regret_vs_schedule_control": lambda_control_delta[0],
                "delta_policy_regret_vs_schedule_control_ci95_low": lambda_control_delta[
                    1
                ],
                "delta_policy_regret_vs_schedule_control_ci95_high": lambda_control_delta[
                    2
                ],
                "decision_change_rate_vs_frozen_b4": float(lambda_changed.mean()),
                "changed_better_count": int(
                    (lambda_changed & (lambda_utility_delta > 1e-8)).sum()
                ),
                "changed_worse_count": int(
                    (lambda_changed & (lambda_utility_delta < -1e-8)).sum()
                ),
            }
        )
    return summary, per_lambda_rows, chosen, regret


def validate_formal_split(
    train_trajectories: list[dict[str, Any]],
    validation_trajectories: list[dict[str, Any]],
) -> tuple[set[int], set[int]]:
    train_prompts = {int(item["prompt_id"]) for item in train_trajectories}
    validation_prompts = {int(item["prompt_id"]) for item in validation_trajectories}
    if train_prompts & validation_prompts:
        raise ValueError("Train and validation prompts overlap")
    if len(train_prompts) != 1000 or len(train_trajectories) != 1000:
        raise ValueError("Residual selection requires 1000 train prompts with one seed")
    if len(validation_prompts) != 200 or len(validation_trajectories) != 600:
        raise ValueError(
            "Residual selection requires 200 validation prompts with three seeds"
        )
    seed_counts = {
        prompt: sum(
            int(item["prompt_id"]) == prompt for item in validation_trajectories
        )
        for prompt in validation_prompts
    }
    if set(seed_counts.values()) != {3}:
        raise ValueError("Every validation prompt must have exactly three trajectories")
    return train_prompts, validation_prompts


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    runs_root = Path(args.b4_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    manifest = base.load_dataset_manifest(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_sha256 = audit.sha256_file(manifest_path)
    feature_indices = np.arange(int(manifest["feature_count"]), dtype=np.int64)
    train_trajectories = base.load_trajectories(
        dataset_dir, manifest, "train", feature_indices
    )
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", feature_indices
    )
    train_prompts, validation_prompts = validate_formal_split(
        train_trajectories, validation_trajectories
    )
    candidate_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    factor_indices = select_factor_indices(list(manifest["feature_names"]))
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
    models, b4_inputs = audit.load_b4_ensemble(
        runs_root, manifest_sha256, len(candidate_steps), device
    )
    expected_b4_seeds = {42, 100, 2024, 27182, 31415}
    observed_b4_seeds = {int(item["train_seed"]) for item in b4_inputs}
    if observed_b4_seeds != expected_b4_seeds:
        raise ValueError(
            "Residual selection requires the locked five-seed B4 ensemble; "
            f"observed seeds: {sorted(observed_b4_seeds)}"
        )
    print("Computing frozen B4 ensemble margins...")
    train_b4 = audit.ensemble_b4_margins(
        models, train_trajectories, args.lambdas, device, args.inference_batch_size
    )
    validation_b4 = audit.ensemble_b4_margins(
        models,
        validation_trajectories,
        args.lambdas,
        device,
        args.inference_batch_size,
    )
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_rows = audit.build_audit_rows(
        train_trajectories,
        args.lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        train_b4,
        args.margin_temperature,
    )
    validation_rows = audit.build_audit_rows(
        validation_trajectories,
        args.lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        validation_b4,
        args.margin_temperature,
    )
    train_base_margin = train_rows.b4_margin[:, 0]
    validation_base_margin = validation_rows.b4_margin[:, 0]
    train_residual_target = np.clip(
        train_rows.target_scaled_margin - train_base_margin,
        -args.residual_target_clip,
        args.residual_target_clip,
    ).astype(np.float32)
    train_b4_chosen, train_b4_regret, _ = policy_outputs(
        train_base_margin, train_rows, candidate_steps, args.lambdas
    )
    validation_b4_chosen, validation_b4_regret, validation_b4_realized = policy_outputs(
        validation_base_margin,
        validation_rows,
        candidate_steps,
        args.lambdas,
    )
    zero_margin, _, _ = apply_residual_correction(
        validation_base_margin,
        np.ones_like(validation_base_margin),
        0.0,
        0.0,
        args.residual_clip,
    )
    zero_scale_exact_reproduction = bool(
        np.array_equal(zero_margin, validation_base_margin)
        and np.array_equal(
            policy_outputs(zero_margin, validation_rows, candidate_steps, args.lambdas)[
                0
            ],
            validation_b4_chosen,
        )
    )
    if not zero_scale_exact_reproduction:
        raise RuntimeError("alpha=0 failed to reproduce frozen B4 exactly")

    oof_grid_rows: list[dict[str, Any]] = []
    selected_specs: dict[tuple[str, str], dict[str, Any]] = {}
    validation_outputs: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    out_dir.mkdir(parents=True)
    checkpoint_dir = out_dir / "correction_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_artifacts: dict[str, dict[str, Any]] = {}
    for family in MODEL_FAMILIES:
        print(f"Selecting train-only OOF hyperparameters for {family}...")
        for factor_group in FACTOR_GROUPS:
            indices = factor_indices[factor_group]
            train_x = correction_features(train_rows, indices)
            validation_x = correction_features(validation_rows, indices)
            selected, grid_rows = select_oof_configuration(
                family,
                factor_group,
                train_x,
                train_residual_target,
                train_rows,
                candidate_steps,
                args.lambdas,
                train_b4_chosen,
                train_b4_regret,
                args,
            )
            oof_grid_rows.extend(grid_rows)
            selected_specs[(family, factor_group)] = dict(selected)
            model = make_regressor(family, float(selected["model_parameter"]), args)
            model.fit(train_x, train_residual_target)
            predicted_residual = model.predict(validation_x).astype(np.float32)
            corrected, gate, applied = apply_residual_correction(
                validation_base_margin,
                predicted_residual,
                float(selected["correction_scale"]),
                float(selected["gate_threshold"]),
                args.residual_clip,
            )
            validation_outputs[(family, factor_group)] = (corrected, gate, applied)
            checkpoint_path = (
                checkpoint_dir / f"{family}_{factor_group}_correction.joblib"
            )
            selected_names = [manifest["feature_names"][index] for index in indices]
            joblib.dump(
                {
                    "schema": REPORT_SCHEMA,
                    "model_family": family,
                    "factor_group": factor_group,
                    "feature_indices": indices,
                    "feature_names": selected_names,
                    "candidate_steps": candidate_steps,
                    "lambdas": np.asarray(args.lambdas, dtype=np.float32),
                    "dataset_manifest_sha256": manifest_sha256,
                    "latency_profile_sha256": latency_provenance["sha256"],
                    "b4_checkpoint_sha256s": [
                        item["checkpoint_sha256"] for item in b4_inputs
                    ],
                    "schedule_feature_names": list(base.SCHEDULE_FEATURE_NAMES),
                    "correction_scale": float(selected["correction_scale"]),
                    "gate_threshold": float(selected["gate_threshold"]),
                    "residual_clip": args.residual_clip,
                    "state_normalization": "train_per_candidate_step_v1",
                    "state_mean": state_mean[:, indices],
                    "state_std": state_std[:, indices],
                    "model": model,
                },
                checkpoint_path,
            )
            checkpoint_artifacts[f"{family}_{factor_group}"] = {
                "path": str(checkpoint_path.relative_to(out_dir)),
                "sha256": audit.sha256_file(checkpoint_path),
            }

    rng = np.random.default_rng(args.bootstrap_seed)
    validation_rows_out: list[dict[str, Any]] = []
    per_lambda_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        schedule_key = (family, "schedule_control")
        schedule_margin = validation_outputs[schedule_key][0]
        schedule_policy = policy_outputs(
            schedule_margin, validation_rows, candidate_steps, args.lambdas
        )
        for factor_group in FACTOR_GROUPS:
            key = (family, factor_group)
            corrected, gate, applied = validation_outputs[key]
            control = (
                None
                if factor_group == "schedule_control"
                else (
                    schedule_policy[0],
                    schedule_policy[1],
                )
            )
            summary, lambda_rows, _, _ = evaluate_candidate(
                family,
                factor_group,
                selected_specs[key],
                corrected,
                gate,
                applied,
                int(factor_indices[factor_group].size),
                validation_rows,
                candidate_steps,
                args.lambdas,
                validation_b4_chosen,
                validation_b4_regret,
                validation_b4_realized,
                control,
                args.bootstrap_samples,
                rng,
            )
            validation_rows_out.append(summary)
            per_lambda_rows.extend(lambda_rows)

    validation_oracle = validation_rows.utilities.argmax(axis=2)
    validation_b4_step_error = np.abs(
        candidate_steps[validation_b4_chosen] - candidate_steps[validation_oracle]
    )
    frozen_summary = {key: 0 for key in validation_rows_out[0]}
    frozen_summary.update(
        {
            "model_family": "frozen_b4",
            "factor_group": "frozen_b4",
            "feature_count": 0,
            "train_oof_policy_regret": float(train_b4_regret.mean()),
            "validation_policy_regret": float(validation_b4_regret.mean()),
            "validation_margin_mae": float(
                np.abs(
                    validation_base_margin - validation_rows.target_scaled_margin
                ).mean()
            ),
            "validation_oracle_switch_step_mae": float(validation_b4_step_error.mean()),
        }
    )
    validation_rows_out.insert(0, frozen_summary)
    frozen_lambda_rows: list[dict[str, Any]] = []
    for lambda_index, lambda_value in enumerate(args.lambdas):
        row = {key: 0 for key in per_lambda_rows[0]}
        row.update(
            {
                "model_family": "frozen_b4",
                "factor_group": "frozen_b4",
                "lambda": lambda_value,
                "policy_regret": float(validation_b4_regret[:, lambda_index].mean()),
            }
        )
        frozen_lambda_rows.append(row)
    per_lambda_rows = frozen_lambda_rows + per_lambda_rows

    artifact_rows = {
        "train_oof_hyperparameter_selection": (
            "train_oof_hyperparameter_selection.csv",
            oof_grid_rows,
        ),
        "validation_residual_correction": (
            "validation_residual_correction.csv",
            validation_rows_out,
        ),
        "per_lambda_residual_correction": (
            "per_lambda_residual_correction.csv",
            per_lambda_rows,
        ),
    }
    for _, (filename, rows) in artifact_rows.items():
        audit.write_csv(out_dir / filename, rows)
    selected_summary = {
        f"{family}_{group}": {
            key: value
            for key, value in selected_specs[(family, group)].items()
            if key != "selected"
        }
        for family in MODEL_FAMILIES
        for group in FACTOR_GROUPS
    }
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "bounded_low_confidence_state_residual_on_frozen_b4",
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "evaluation_protocol": "deterministic_eval_mode_v1",
        "selection_only": True,
        "formal_evidence": False,
        "test_accessed": False,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": audit.sha256_file(Path(__file__).resolve()),
        },
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "train_prompt_count": len(train_prompts),
        "train_trajectory_count": len(train_trajectories),
        "validation_prompt_count": len(validation_prompts),
        "validation_trajectory_count": len(validation_trajectories),
        "validation_seed_count_per_prompt": 3,
        "candidate_steps": candidate_steps.tolist(),
        "lambdas": args.lambdas,
        "factor_groups": {
            group: [manifest["feature_names"][index] for index in indices]
            for group, indices in factor_indices.items()
        },
        "frozen_b4_validation_macro_policy_regret": float(validation_b4_regret.mean()),
        "zero_scale_exact_reproduction": zero_scale_exact_reproduction,
        "selection": {
            "unit": "train_prompt_grouped_out_of_fold",
            "folds": args.cv_folds,
            "objective": "macro_policy_regret",
            "tie_break": "fewer_changes_then_smaller_scale_then_narrower_gate",
            "ridge_alphas": args.ridge_alphas,
            "correction_scales": [0.0] + args.correction_scales,
            "gate_thresholds": args.gate_thresholds,
            "residual_clip": args.residual_clip,
            "residual_target_clip": args.residual_target_clip,
            "selected_specs": selected_summary,
        },
        "validation_bootstrap": {
            "unit": "prompt_after_averaging_generation_seeds_and_lambdas",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "latency_profile": latency_provenance,
        "b4_inputs": b4_inputs,
        "artifacts": {
            **{
                name: {
                    "path": filename,
                    "sha256": audit.sha256_file(out_dir / filename),
                }
                for name, (filename, _) in artifact_rows.items()
            },
            "correction_checkpoints": checkpoint_artifacts,
        },
        "limitations": [
            "B4 checkpoints are frozen but their train predictions are in-sample; validation remains prompt-disjoint.",
            "This selection run cannot be used as locked test confirmation.",
        ],
    }
    report_path = out_dir / "residual_correction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote residual correction selection to {out_dir}")


if __name__ == "__main__":
    main()
