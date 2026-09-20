"""B4-style sparse action router with prompt priors and seed residuals.

This development trainer reuses Sparse Action Phase 3 outcomes.  It changes
the learning problem from hard pipeline classification to action-conditioned
soft utility modelling:

* the prompt prior is trained on prompt/action utility averaged across seeds;
* the state branch is trained only on the seed residual around that mean;
* each observed group uses a B4-style soft target over REFERENCE plus its
  sampled actions; and
* regularization is selected by prompt-disjoint OOF policy regret, not row MSE.

The current state source may still be the post-hoc REFERENCE-video proxy.  The
output therefore remains a development capacity audit rather than deployable
early-state evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.phase2_analysis import csv_write  # noqa: E402
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402
from UNIV_adaptor.scripts.router.train_sparse_prompt_state_router import (  # noqa: E402
    AXES,
    build_context,
    load_state_features,
    prepare_rows,
    prompt_cluster_ci,
    prompt_embedding_map,
    read_json,
    shuffled_state_map,
    validate_scored_dir,
)


MODEL_SCHEMA = "univ_sparse_b4_action_router_development_v1"
FAMILIES = (
    "action_main_soft",
    "prompt_prior_soft",
    "action_main_state_residual_soft",
    "prompt_state_residual_soft",
    "prompt_state_residual_shuffled",
)


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def group_index_matrix(group_ids: list[str] | np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        groups.setdefault(str(group), []).append(index)
    sizes = {len(indices) for indices in groups.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) < 1:
        raise ValueError("soft action groups must have one common positive size")
    return np.asarray([groups[key] for key in sorted(groups)], dtype=np.int64)


def soft_policy_loss_and_gradient(
    weights: np.ndarray,
    features: np.ndarray,
    utilities: np.ndarray,
    group_index: np.ndarray,
    *,
    temperature: float,
    alpha: float,
    penalty_start: int,
) -> tuple[float, np.ndarray]:
    """Cross-entropy to soft utility targets with an implicit zero REFERENCE."""

    scores = features @ weights
    grouped_scores = scores[group_index]
    grouped_truth = utilities[group_index]
    zeros = np.zeros((len(group_index), 1), dtype=np.float64)
    target = stable_softmax(
        np.concatenate((zeros, grouped_truth), axis=1) / temperature, axis=1
    )
    predicted = stable_softmax(
        np.concatenate((zeros, grouped_scores), axis=1) / temperature, axis=1
    )
    loss = -float(np.mean(np.sum(target * np.log(predicted + 1e-300), axis=1)))
    row_gradient = np.zeros(len(features), dtype=np.float64)
    row_gradient[group_index.reshape(-1)] = (predicted[:, 1:] - target[:, 1:]).reshape(
        -1
    ) / (len(group_index) * temperature)
    gradient = features.T @ row_gradient
    if penalty_start < len(weights) and alpha > 0:
        penalized = weights[penalty_start:]
        loss += 0.5 * alpha * float(np.dot(penalized, penalized))
        gradient[penalty_start:] += alpha * penalized
    return loss, gradient


def fit_soft_action_model(
    features: np.ndarray,
    utilities: np.ndarray,
    group_ids: list[str] | np.ndarray,
    *,
    temperature: float,
    alpha: float,
    penalty_start: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Fit the convex B4-style linear action scorer with L-BFGS."""

    from scipy.optimize import minimize

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(utilities, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("invalid soft action training matrix")
    if not np.isfinite(y).all() or temperature <= 0 or alpha < 0:
        raise ValueError("invalid soft action target, temperature, or alpha")
    if not 0 <= penalty_start <= x.shape[1]:
        raise ValueError("invalid penalty boundary")
    grouped = group_index_matrix(group_ids)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return soft_policy_loss_and_gradient(
            weights,
            x,
            y,
            grouped,
            temperature=temperature,
            alpha=alpha,
            penalty_start=penalty_start,
        )

    result = minimize(
        objective,
        np.zeros(x.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"non-finite soft action optimization: {result.message}")
    return {
        "weights": result.x,
        "loss": float(result.fun),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "optimizer_message": str(result.message),
    }


def predict_soft_action(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ np.asarray(model["weights"])


def prompt_mean_residual_decomposition(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create prompt-mean prior rows and seed residual rows without hard labels."""

    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["prompt_key"], row["action_id"]), []).append(row)
    mean_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    means: dict[tuple[str, str], float] = {}
    seed_counts = set()
    for key, items in sorted(cells.items()):
        identities = {tuple(np.asarray(item["levels"]).tolist()) for item in items}
        if len(identities) != 1:
            raise ValueError(f"action levels differ across seeds: {key}")
        seed_counts.add(len(items))
        mean = float(np.mean([item["target"] for item in items]))
        means[key] = mean
        representative = dict(items[0])
        representative["group_id"] = f"prompt_mean::{key[0]}"
        representative["observation_id"] = f"prompt_mean::{key[0]}::{key[1]}"
        representative["target"] = mean
        mean_rows.append(representative)
    if len(seed_counts) != 1:
        raise ValueError("prompt/action cells have inconsistent seed coverage")
    for row in rows:
        residual = dict(row)
        residual["target"] = (
            row["target"] - means[(row["prompt_key"], row["action_id"])]
        )
        residual_rows.append(residual)

    mean_values = np.asarray([row["target"] for row in mean_rows])
    residual_values = np.asarray([row["target"] for row in residual_rows])
    prompt_variance = float(np.var(mean_values))
    residual_variance = float(np.var(residual_values))
    denominator = prompt_variance + residual_variance
    per_action = []
    for action in sorted({row["action_id"] for row in rows}):
        action_means = np.asarray(
            [row["target"] for row in mean_rows if row["action_id"] == action]
        )
        action_residuals = np.asarray(
            [row["target"] for row in residual_rows if row["action_id"] == action]
        )
        per_action.append(
            {
                "action_id": action,
                "prompt_cells": int(len(action_means)),
                "seed_rows": int(len(action_residuals)),
                "prompt_mean_variance": float(np.var(action_means)),
                "seed_residual_variance": float(np.var(action_residuals)),
            }
        )
    diagnostics = {
        "prompt_action_cells": len(mean_rows),
        "seed_rows": len(residual_rows),
        "seeds_per_prompt_action": next(iter(seed_counts)),
        "prompt_mean_variance": prompt_variance,
        "seed_residual_variance": residual_variance,
        "prompt_fraction_of_decomposed_variance": prompt_variance / denominator
        if denominator > 0
        else 0.0,
        "per_action": per_action,
    }
    return mean_rows, residual_rows, diagnostics


def prompt_fold_map(
    rows: list[dict[str, Any]], folds: int, seed: int
) -> dict[str, int]:
    prompts = sorted({row["prompt_key"] for row in rows})
    if len(prompts) < folds:
        raise ValueError("not enough prompts for prompt-disjoint folds")
    shuffled = np.random.default_rng(seed).permutation(prompts)
    result: dict[str, int] = {}
    for fold, chunk in enumerate(np.array_split(shuffled, folds)):
        result.update({str(prompt): fold for prompt in chunk})
    return result


def feature_matrix(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    fit_indices: np.ndarray,
    *,
    family: str,
    prompt_mode: str,
    prompt_vectors: dict[str, np.ndarray] | None,
    state: dict[str, np.ndarray],
    shuffled_state: dict[str, np.ndarray],
    max_features: int,
    include_main: bool,
) -> tuple[np.ndarray, int]:
    main = np.stack([rows[int(index)]["main"] for index in indices])
    context = build_context(
        rows,
        indices,
        fit_indices,
        family=family,
        prompt_mode=prompt_mode,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled_state,
        max_features=max_features,
    )
    parts = [main] if include_main else []
    penalty_start = main.shape[1] if include_main else 0
    if context is not None:
        parts.append(context)
    if not parts:
        raise ValueError("empty action feature matrix")
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0], penalty_start


def crossfit_soft_models(
    fit_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    family: str,
    prompt_mode: str,
    prompt_vectors: dict[str, np.ndarray] | None,
    state: dict[str, np.ndarray],
    shuffled_state: dict[str, np.ndarray],
    folds: int,
    seed: int,
    max_features: int,
    alphas: list[float],
    temperature: float,
    max_iterations: int,
    include_main: bool,
) -> dict[float, np.ndarray]:
    assignment = prompt_fold_map(prediction_rows, folds, seed)
    predictions = {alpha: np.zeros(len(prediction_rows)) for alpha in alphas}
    for fold in range(folds):
        fit_ids = np.asarray(
            [
                index
                for index, row in enumerate(fit_rows)
                if assignment[row["prompt_key"]] != fold
            ],
            dtype=np.int64,
        )
        held_ids = np.asarray(
            [
                index
                for index, row in enumerate(prediction_rows)
                if assignment[row["prompt_key"]] == fold
            ],
            dtype=np.int64,
        )
        # Prompt preprocessing must use the actual fold training rows.  When
        # the fit and prediction tables differ, concatenate them so TF-IDF
        # fitting and T5 standardization both use only the fit-side indices.
        if fit_rows is not prediction_rows:
            combined = fit_rows + prediction_rows
            combined_fit = fit_ids
            combined_held = held_ids + len(fit_rows)
            x_fit, penalty_start = feature_matrix(
                combined,
                combined_fit,
                combined_fit,
                family=family,
                prompt_mode=prompt_mode,
                prompt_vectors=prompt_vectors,
                state=state,
                shuffled_state=shuffled_state,
                max_features=max_features,
                include_main=include_main,
            )
            x_held, _ = feature_matrix(
                combined,
                combined_held,
                combined_fit,
                family=family,
                prompt_mode=prompt_mode,
                prompt_vectors=prompt_vectors,
                state=state,
                shuffled_state=shuffled_state,
                max_features=max_features,
                include_main=include_main,
            )
        else:
            x_fit, penalty_start = feature_matrix(
                fit_rows,
                fit_ids,
                fit_ids,
                family=family,
                prompt_mode=prompt_mode,
                prompt_vectors=prompt_vectors,
                state=state,
                shuffled_state=shuffled_state,
                max_features=max_features,
                include_main=include_main,
            )
            x_held, _ = feature_matrix(
                prediction_rows,
                held_ids,
                fit_ids if fit_rows is prediction_rows else np.arange(len(fit_rows)),
                family=family,
                prompt_mode=prompt_mode,
                prompt_vectors=prompt_vectors,
                state=state,
                shuffled_state=shuffled_state,
                max_features=max_features,
                include_main=include_main,
            )
        y_fit = np.asarray([fit_rows[int(index)]["target"] for index in fit_ids])
        groups = [fit_rows[int(index)]["group_id"] for index in fit_ids]
        for alpha in alphas:
            model = fit_soft_action_model(
                x_fit,
                y_fit,
                groups,
                temperature=temperature,
                alpha=alpha,
                penalty_start=penalty_start,
                max_iterations=max_iterations,
            )
            predictions[alpha][held_ids] = predict_soft_action(model, x_held)
    return predictions


def fit_full_predict(
    fit_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    family: str,
    prompt_mode: str,
    prompt_vectors: dict[str, np.ndarray] | None,
    state: dict[str, np.ndarray],
    shuffled_state: dict[str, np.ndarray],
    max_features: int,
    alpha: float,
    temperature: float,
    max_iterations: int,
    include_main: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if fit_rows is not prediction_rows:
        combined = fit_rows + prediction_rows
        fit_ids = np.arange(len(fit_rows))
        prediction_ids = np.arange(len(prediction_rows)) + len(fit_rows)
        x_fit, penalty_start = feature_matrix(
            combined,
            fit_ids,
            fit_ids,
            family=family,
            prompt_mode=prompt_mode,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled_state,
            max_features=max_features,
            include_main=include_main,
        )
        x_prediction, _ = feature_matrix(
            combined,
            prediction_ids,
            fit_ids,
            family=family,
            prompt_mode=prompt_mode,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled_state,
            max_features=max_features,
            include_main=include_main,
        )
    else:
        fit_ids = np.arange(len(fit_rows))
        prediction_ids = np.arange(len(prediction_rows))
        x_fit, penalty_start = feature_matrix(
            fit_rows,
            fit_ids,
            fit_ids,
            family=family,
            prompt_mode=prompt_mode,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled_state,
            max_features=max_features,
            include_main=include_main,
        )
        x_prediction, _ = feature_matrix(
            prediction_rows,
            prediction_ids,
            fit_ids if fit_rows is prediction_rows else np.arange(len(fit_rows)),
            family=family,
            prompt_mode=prompt_mode,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled_state,
            max_features=max_features,
            include_main=include_main,
        )
    model = fit_soft_action_model(
        x_fit,
        np.asarray([row["target"] for row in fit_rows]),
        [row["group_id"] for row in fit_rows],
        temperature=temperature,
        alpha=alpha,
        penalty_start=penalty_start,
        max_iterations=max_iterations,
    )
    return predict_soft_action(model, x_prediction), model


def policy_diagnostics(
    rows: list[dict[str, Any]], predictions: np.ndarray, *, temperature: float
) -> dict[str, float]:
    grouped = group_index_matrix([row["group_id"] for row in rows])
    truth = np.asarray([row["target"] for row in rows])
    regrets = []
    utilities = []
    harms = []
    exact = []
    cross_entropies = []
    for indices in grouped:
        y = truth[indices]
        pred = predictions[indices]
        oracle_slot = int(np.argmax(np.concatenate(([0.0], y))))
        selected_slot = int(np.argmax(np.concatenate(([0.0], pred))))
        realized = 0.0 if selected_slot == 0 else float(y[selected_slot - 1])
        oracle = float(max(0.0, y.max()))
        regrets.append(oracle - realized)
        utilities.append(realized)
        harms.append(realized < -0.001)
        exact.append(selected_slot == oracle_slot)
        target_soft = stable_softmax(np.concatenate(([0.0], y)) / temperature)
        predicted_soft = stable_softmax(np.concatenate(([0.0], pred)) / temperature)
        cross_entropies.append(
            -float(np.sum(target_soft * np.log(predicted_soft + 1e-300)))
        )
    return {
        "policy_regret": float(np.mean(regrets)),
        "realized_delta_utility": float(np.mean(utilities)),
        "material_harm_rate": float(np.mean(harms)),
        "exact_sampled_oracle_rate": float(np.mean(exact)),
        "soft_cross_entropy": float(np.mean(cross_entropies)),
        "row_mse": float(np.mean((predictions - truth) ** 2)),
    }


def select_by_policy(
    rows: list[dict[str, Any]],
    candidates: dict[float | None, np.ndarray],
    *,
    temperature: float,
) -> tuple[float | None, list[dict[str, Any]]]:
    diagnostics = []
    for order, (alpha, prediction) in enumerate(candidates.items()):
        metrics = policy_diagnostics(rows, prediction, temperature=temperature)
        diagnostics.append({"alpha": alpha, "candidate_order": order, **metrics})
    selected = min(
        diagnostics,
        key=lambda row: (
            row["policy_regret"],
            row["soft_cross_entropy"],
            row["row_mse"],
            row["candidate_order"],
        ),
    )
    for row in diagnostics:
        row["selected"] = row is selected
    return selected["alpha"], diagnostics


def evaluate_policies(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    *,
    split: str,
    utility_lambda: float,
    observation_cost_ratio: float,
    harm_epsilon: float,
    bootstrap_repetitions: int,
    seed: int,
    active_state_policies: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_index_matrix([row["group_id"] for row in rows])
    summary = []
    details = []
    realized_by_policy: dict[str, dict[str, float]] = {
        policy: {} for policy in ("reference", *predictions)
    }
    for indices in grouped:
        group_rows = [rows[int(index)] for index in indices]
        group = group_rows[0]["group_id"]
        truth = np.asarray([row["target"] for row in group_rows])
        action_ids = [row["action_id"] for row in group_rows]
        oracle_slot = int(np.argmax(np.concatenate(([0.0], truth))))
        oracle_action = "REFERENCE" if oracle_slot == 0 else action_ids[oracle_slot - 1]
        for policy in ("reference", *predictions):
            if policy == "reference":
                selected_slot = 0
            else:
                selected_slot = int(
                    np.argmax(np.concatenate(([0.0], predictions[policy][indices])))
                )
            selected_action = (
                "REFERENCE" if selected_slot == 0 else action_ids[selected_slot - 1]
            )
            gross = 0.0 if selected_slot == 0 else float(truth[selected_slot - 1])
            penalty = (
                utility_lambda * observation_cost_ratio
                if policy in active_state_policies
                else 0.0
            )
            net = gross - penalty
            oracle = float(max(0.0, truth.max()))
            realized_by_policy[policy][group] = net
            details.append(
                {
                    "split": split,
                    "group_id": group,
                    "prompt_key": group_rows[0]["prompt_key"],
                    "base_seed": group_rows[0]["base_seed"],
                    "policy": policy,
                    "selected_action": selected_action,
                    "oracle_action": oracle_action,
                    "gross_delta_utility": gross,
                    "observation_penalty": penalty,
                    "net_delta_utility": net,
                    "sampled_oracle_delta_utility": oracle,
                    "policy_regret": oracle - net,
                    "material_harm": net < -harm_epsilon,
                }
            )
    baseline = realized_by_policy["action_main_soft"]
    for policy, values in realized_by_policy.items():
        policy_rows = [row for row in details if row["policy"] == policy]
        clustered: dict[str, list[float]] = {}
        delta_baseline: dict[str, list[float]] = {}
        for row in policy_rows:
            clustered.setdefault(row["prompt_key"], []).append(row["net_delta_utility"])
            delta_baseline.setdefault(row["prompt_key"], []).append(
                row["net_delta_utility"] - baseline[row["group_id"]]
            )
        utility_ci = prompt_cluster_ci(
            clustered, seed=seed, repetitions=bootstrap_repetitions
        )
        delta_ci = prompt_cluster_ci(
            delta_baseline, seed=seed + 1, repetitions=bootstrap_repetitions
        )
        summary.append(
            {
                "split": split,
                "policy": policy,
                "prompts": len(clustered),
                "groups": len(values),
                "mean_net_delta_utility": float(np.mean(list(values.values()))),
                "utility_ci_low": utility_ci[0],
                "utility_ci_high": utility_ci[1],
                "gain_vs_action_main_soft": float(
                    np.mean([values[group] - baseline[group] for group in values])
                ),
                "gain_vs_action_main_ci_low": delta_ci[0],
                "gain_vs_action_main_ci_high": delta_ci[1],
                "mean_sampled_regret": float(
                    np.mean([row["policy_regret"] for row in policy_rows])
                ),
                "material_harm_rate": float(
                    np.mean([row["material_harm"] for row in policy_rows])
                ),
                "nonreference_rate": float(
                    np.mean(
                        [row["selected_action"] != "REFERENCE" for row in policy_rows]
                    )
                ),
                "exact_sampled_oracle_rate": float(
                    np.mean(
                        [
                            row["selected_action"] == row["oracle_action"]
                            for row in policy_rows
                        ]
                    )
                ),
            }
        )
    return summary, details


def train(args: argparse.Namespace) -> None:
    scored_dir = Path(args.scored_dir).resolve()
    scored, _, pairs = validate_scored_dir(scored_dir)
    all_rows = prepare_rows(pairs, args.utility_lambda)
    train_rows = [row for row in all_rows if row["cohort"] == "existing_train"]
    fresh_rows = [row for row in all_rows if row["cohort"] == "fresh_development"]
    mean_rows, residual_rows, decomposition = prompt_mean_residual_decomposition(
        train_rows
    )
    expected_groups = {row["group_id"] for row in all_rows}
    state, state_manifest = load_state_features(
        Path(args.state_dir).resolve(),
        expected_input_sha256=scored["input_sha256"],
        expected_groups=expected_groups,
    )
    shuffled = shuffled_state_map(all_rows, state)
    train_prompts = {row["prompt_key"] for row in train_rows}
    prompt_vectors, t5_digest = prompt_embedding_map(
        all_rows,
        mode=args.prompt_features,
        t5_dir=Path(args.t5_dir).resolve() if args.t5_dir else None,
        train_prompts=train_prompts,
    )

    action_oof_by_alpha = crossfit_soft_models(
        mean_rows,
        train_rows,
        family="action_main",
        prompt_mode=args.prompt_features,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled,
        folds=args.folds,
        seed=args.seed,
        max_features=args.max_features,
        alphas=[0.0],
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        include_main=True,
    )
    action_oof = action_oof_by_alpha[0.0]
    prompt_oof_by_alpha = crossfit_soft_models(
        mean_rows,
        train_rows,
        family="prompt_only",
        prompt_mode=args.prompt_features,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled,
        folds=args.folds,
        seed=args.seed,
        max_features=args.max_features,
        alphas=args.alphas,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        include_main=True,
    )
    prompt_candidates: dict[float | None, np.ndarray] = {
        None: action_oof,
        **prompt_oof_by_alpha,
    }
    prompt_alpha, prompt_cv = select_by_policy(
        train_rows, prompt_candidates, temperature=args.temperature
    )
    prompt_oof = prompt_candidates[prompt_alpha]

    state_oof_by_alpha = crossfit_soft_models(
        residual_rows,
        residual_rows,
        family="state_only",
        prompt_mode=args.prompt_features,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled,
        folds=args.folds,
        seed=args.seed,
        max_features=args.max_features,
        alphas=args.alphas,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        include_main=False,
    )
    shuffled_oof_by_alpha = crossfit_soft_models(
        residual_rows,
        residual_rows,
        family="state_only",
        prompt_mode=args.prompt_features,
        prompt_vectors=prompt_vectors,
        state=shuffled,
        shuffled_state=shuffled,
        folds=args.folds,
        seed=args.seed,
        max_features=args.max_features,
        alphas=args.alphas,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        include_main=False,
    )
    state_only_candidates = {
        None: action_oof,
        **{a: action_oof + p for a, p in state_oof_by_alpha.items()},
    }
    fusion_candidates = {
        None: prompt_oof,
        **{a: prompt_oof + p for a, p in state_oof_by_alpha.items()},
    }
    shuffled_candidates = {
        None: prompt_oof,
        **{a: prompt_oof + p for a, p in shuffled_oof_by_alpha.items()},
    }
    state_only_alpha, state_only_cv = select_by_policy(
        train_rows, state_only_candidates, temperature=args.temperature
    )
    fusion_alpha, fusion_cv = select_by_policy(
        train_rows, fusion_candidates, temperature=args.temperature
    )
    shuffled_alpha, shuffled_cv = select_by_policy(
        train_rows, shuffled_candidates, temperature=args.temperature
    )

    action_fresh, action_model = fit_full_predict(
        mean_rows,
        fresh_rows,
        family="action_main",
        prompt_mode=args.prompt_features,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled,
        max_features=args.max_features,
        alpha=0.0,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        include_main=True,
    )
    if prompt_alpha is None:
        prompt_fresh = action_fresh
        prompt_model = None
    else:
        prompt_fresh, prompt_model = fit_full_predict(
            mean_rows,
            fresh_rows,
            family="prompt_only",
            prompt_mode=args.prompt_features,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled,
            max_features=args.max_features,
            alpha=prompt_alpha,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            include_main=True,
        )

    def full_residual(
        alpha: float | None, shuffled_mode: bool
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        if alpha is None:
            return np.zeros(len(fresh_rows)), None
        return fit_full_predict(
            residual_rows,
            fresh_rows,
            family="state_only",
            prompt_mode=args.prompt_features,
            prompt_vectors=prompt_vectors,
            state=shuffled if shuffled_mode else state,
            shuffled_state=shuffled,
            max_features=args.max_features,
            alpha=alpha,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            include_main=False,
        )

    state_only_residual, state_only_model = full_residual(state_only_alpha, False)
    fusion_residual, fusion_model = full_residual(fusion_alpha, False)
    shuffled_residual, shuffled_model = full_residual(shuffled_alpha, True)

    train_predictions = {
        "action_main_soft": action_oof,
        "prompt_prior_soft": prompt_oof,
        "action_main_state_residual_soft": state_only_candidates[state_only_alpha],
        "prompt_state_residual_soft": fusion_candidates[fusion_alpha],
        "prompt_state_residual_shuffled": shuffled_candidates[shuffled_alpha],
    }
    fresh_predictions = {
        "action_main_soft": action_fresh,
        "prompt_prior_soft": prompt_fresh,
        "action_main_state_residual_soft": action_fresh + state_only_residual,
        "prompt_state_residual_soft": prompt_fresh + fusion_residual,
        "prompt_state_residual_shuffled": prompt_fresh + shuffled_residual,
    }
    summary = []
    details = []
    active_state_policies = {
        family
        for family, alpha in (
            ("action_main_state_residual_soft", state_only_alpha),
            ("prompt_state_residual_soft", fusion_alpha),
            ("prompt_state_residual_shuffled", shuffled_alpha),
        )
        if alpha is not None
    }
    for rows, predictions, split in (
        (train_rows, train_predictions, "existing_train_prompt_oof_selection"),
        (fresh_rows, fresh_predictions, "fresh_development_holdout"),
    ):
        split_summary, split_details = evaluate_policies(
            rows,
            predictions,
            split=split,
            utility_lambda=args.utility_lambda,
            observation_cost_ratio=args.observation_cost_ratio,
            harm_epsilon=args.harm_epsilon,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
            active_state_policies=active_state_policies,
        )
        summary.extend(split_summary)
        details.extend(split_details)

    cv_rows = []
    for family, diagnostics in (
        ("prompt_prior_soft", prompt_cv),
        ("action_main_state_residual_soft", state_only_cv),
        ("prompt_state_residual_soft", fusion_cv),
        ("prompt_state_residual_shuffled", shuffled_cv),
    ):
        cv_rows.extend({"family": family, **row} for row in diagnostics)
    prediction_rows = []
    for rows, predictions, split in (
        (train_rows, train_predictions, "existing_train_prompt_oof_selection"),
        (fresh_rows, fresh_predictions, "fresh_development_holdout"),
    ):
        for index, row in enumerate(rows):
            prediction_rows.append(
                {
                    "split": split,
                    "observation_id": row["observation_id"],
                    "group_id": row["group_id"],
                    "prompt_key": row["prompt_key"],
                    "base_seed": row["base_seed"],
                    "action_id": row["action_id"],
                    "target_delta_utility": row["target"],
                    **{
                        f"predicted_{family}": values[index]
                        for family, values in predictions.items()
                    },
                }
            )

    selected = {
        "prompt_prior_soft": prompt_alpha,
        "action_main_state_residual_soft": state_only_alpha,
        "prompt_state_residual_soft": fusion_alpha,
        "prompt_state_residual_shuffled": shuffled_alpha,
    }
    optimizer = {
        "action_main_soft": action_model,
        "prompt_prior_soft": prompt_model,
        "action_main_state_residual_soft": state_only_model,
        "prompt_state_residual_soft": fusion_model,
        "prompt_state_residual_shuffled": shuffled_model,
    }
    optimizer_summary = {
        family: None
        if model is None
        else {
            "loss": model["loss"],
            "iterations": model["iterations"],
            "converged": model["converged"],
            "optimizer_message": model["optimizer_message"],
        }
        for family, model in optimizer.items()
    }
    request = {
        "schema": MODEL_SCHEMA,
        "source_input_sha256": scored["input_sha256"],
        "source_scored_dataset_sha256": scored["dataset_sha256"],
        "state_manifest_sha256": state_manifest["manifest_sha256"],
        "state_role": state_manifest["observation_role"],
        "deployable_early_state": state_manifest["deployable_early_state"],
        "prompt_features": args.prompt_features,
        "t5_manifest_sha256": t5_digest,
        "utility_lambda": args.utility_lambda,
        "utility_definition": "delta_vbench5 - lambda * (time_ratio_to_reference - 1)",
        "supervision": "group_softmax_over_reference_plus_sampled_actions",
        "decomposition": "prompt_action_seed_mean_plus_seed_residual",
        "selection_rule": "minimum_prompt_disjoint_oof_policy_regret_then_soft_ce_then_mse",
        "temperature": args.temperature,
        "alphas": args.alphas,
        "folds": args.folds,
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "max_features": args.max_features,
        "observation_cost_ratio": args.observation_cost_ratio,
        "harm_epsilon": args.harm_epsilon,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "action_axes": list(AXES),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        request_path = out / "training_request.json"
        if request_path.is_file() and read_json(request_path) != request:
            raise RuntimeError("training request changed; use a new --out-dir")
        if not request_path.is_file():
            write_json_atomic(request_path, request)
        csv_write(out / "model_cv.csv", cv_rows)
        csv_write(out / "predictions.csv", prediction_rows)
        csv_write(out / "policy_by_group.csv", details)
        csv_write(out / "policy_summary.csv", summary)
        write_json_atomic(out / "decomposition_diagnostics.json", decomposition)
        body = {
            **request,
            "selected_alpha": selected,
            "optimizer_summary": optimizer_summary,
            "decomposition_diagnostics": decomposition,
            "evaluation_role": "development_existing_videos_with_posthoc_state_proxy",
            "test_accessed": False,
            "candidate_scope": "REFERENCE plus each group's three sampled probes",
            "results": summary,
        }
        write_json_atomic(
            out / "policy_summary.json",
            {**body, "summary_sha256": canonical_sha256(body)},
        )
        fresh = {
            row["policy"]: row
            for row in summary
            if row["split"] == "fresh_development_holdout"
        }
        lines = [
            "# Sparse B4-style prompt prior + state residual audit",
            "",
            f"Lambda: {args.utility_lambda:.6f}; temperature: {args.temperature:.6f}; prompt features: {args.prompt_features}.",
            "",
            "The prompt branch is trained on prompt/action utility averaged across generation seeds. The state branch is trained only on the within-prompt seed residual. Hyperparameters are selected by prompt-disjoint OOF policy regret. The state remains a post-hoc REFERENCE-video proxy unless the bound manifest explicitly says otherwise.",
            "",
            "| Fresh policy | Net dUtility | 95% CI | Regret | Gain vs main | Harm | Nonref | Exact oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in ("reference", *FAMILIES):
            row = fresh[policy]
            lines.append(
                f"| {policy} | {row['mean_net_delta_utility']:.6f} | [{row['utility_ci_low']:.6f}, {row['utility_ci_high']:.6f}] | "
                f"{row['mean_sampled_regret']:.6f} | {row['gain_vs_action_main_soft']:.6f} | {row['material_harm_rate']:.1%} | "
                f"{row['nonreference_rate']:.1%} | {row['exact_sampled_oracle_rate']:.1%} |"
            )
        lines += [
            "",
            "Interpretation gates:",
            "",
            "1. prompt_prior_soft must beat action_main_soft before claiming a prompt prior;",
            "2. prompt_state_residual_soft must beat prompt_prior_soft and its shuffled-state control before collecting/deploying an online correction;",
            "3. Phase 3 lacks complete matched single-axis coverage, so this audit cannot identify every axis causally.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote sparse B4-style audit: {out / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train"))
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--state-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--prompt-features", choices=("tfidf", "t5"), default="tfidf")
    parser.add_argument("--t5-dir")
    parser.add_argument("--utility-lambda", type=float, default=0.05)
    parser.add_argument("--observation-cost-ratio", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    finite_nonnegative = (
        args.utility_lambda,
        args.observation_cost_ratio,
        args.harm_epsilon,
    )
    if (
        args.temperature <= 0
        or not math.isfinite(args.temperature)
        or args.folds < 2
        or args.max_features < 1
        or args.max_iterations < 10
        or args.bootstrap_repetitions < 100
        or any(not math.isfinite(value) or value < 0 for value in finite_nonnegative)
        or any(not math.isfinite(alpha) or alpha <= 0 for alpha in args.alphas)
    ):
        parser.error("invalid soft-target, CV, utility, or optimization arguments")
    scored = Path(args.scored_dir).resolve()
    args.state_dir = args.state_dir or str(scored / "reference_video_proxy")
    args.t5_dir = args.t5_dir or str(scored / "t5_sparse_prompt_state")
    lambda_tag = f"{args.utility_lambda:.6g}".replace(".", "p")
    args.out_dir = args.out_dir or str(
        scored / f"sparse_b4_{args.prompt_features}_lambda_{lambda_tag}"
    )
    return args


def main() -> None:
    args = parse_args()
    scored, references, pairs = validate_scored_dir(Path(args.scored_dir).resolve())
    rows = prepare_rows(pairs, args.utility_lambda)
    train_rows = [row for row in rows if row["cohort"] == "existing_train"]
    mean_rows, residual_rows, diagnostics = prompt_mean_residual_decomposition(
        train_rows
    )
    print(
        f"Verified sparse B4 inputs: {len(references)} groups, {len(rows)} action rows, "
        f"{len(mean_rows)} prompt/action means, {len(residual_rows)} residual rows, "
        f"input={scored['input_sha256'][:12]}",
        flush=True,
    )
    if args.mode == "check":
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
