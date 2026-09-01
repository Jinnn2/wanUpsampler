#!/usr/bin/env python3
"""Audit ranking and calibration of frozen B4-3 preemption verifier scores."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

try:
    from . import analyze_factor_relevance as factor_audit
    from . import train_b4_preemption_verifier as verifier
    from . import train_variable_lambda_router as base
except ImportError:
    import analyze_factor_relevance as factor_audit
    import train_b4_preemption_verifier as verifier
    import train_variable_lambda_router as base


REPORT_SCHEMA = "b4_preemption_score_geometry_audit_v1"
EXPECTED_RUN_SCHEMA = "b4_sparse_preemption_verifier_run_v2"
EXPECTED_CHECKPOINT_SCHEMA = "b4_sparse_preemption_verifier_checkpoint_v2"
EXPECTED_STEPS = tuple(range(40, 51))
TARGETS = {
    "positive_margin": 0.0,
    "material_positive_margin": 0.001,
}
TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15)
NEGATIVE_THRESHOLDS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--b4-runs-root", required=True)
    parser.add_argument("--verifier-runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2041)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--expected-latency-profile-sha256", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.inference_batch_size < 1:
        parser.error("bootstrap samples and inference batch size must be positive")
    return args


def verifier_signature(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "train_lambdas": summary["train_lambdas"],
        "eval_lambdas": summary["eval_lambdas"],
        "harm_epsilon": summary["harm_epsilon"],
        "candidate_steps": summary["candidate_steps"],
        "radius": summary["radius"],
        "base_state_feature_names": summary["base_state_feature_names"],
        "sparse_signal_names": summary["sparse_signal_names"],
        "state_normalization": summary["state_normalization"],
        "training": summary["training"],
        "validation_shuffle": summary["validation_shuffle"],
        "latency_profile_sha256": summary["latency_profile"]["sha256"],
        "b4_ensemble_size": summary["b4_ensemble_size"],
    }


def load_verifier_suite(
    runs_root: Path, device: torch.device
) -> tuple[
    list[dict[str, verifier.SparsePreemptionVerifier]],
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    summary_paths = sorted(
        runs_root.glob("seed_*/run_summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if len(summary_paths) != 5:
        raise ValueError(f"Expected five V2 verifier runs, found {len(summary_paths)}")
    suites = []
    inputs = []
    signature = None
    canonical_mean = None
    canonical_std = None
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema") != EXPECTED_RUN_SCHEMA
            or summary.get("evaluation_split") != "validation"
            or summary.get("test_accessed")
        ):
            raise ValueError(f"Invalid frozen verifier run: {summary_path}")
        current_signature = verifier_signature(summary)
        if signature is None:
            signature = current_signature
        elif signature != current_signature:
            raise ValueError(f"Verifier protocol differs: {summary_path}")
        models = {}
        checkpoint_inputs = {}
        for model_type in ("preemption_control", "preemption_state"):
            metadata = summary["artifacts"]["checkpoints"][model_type]
            checkpoint_path = summary_path.parent / metadata["path"]
            if base.sha256_file(checkpoint_path) != metadata["sha256"]:
                raise ValueError(
                    f"Verifier checkpoint SHA256 mismatch: {checkpoint_path}"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                checkpoint.get("schema") != EXPECTED_CHECKPOINT_SCHEMA
                or checkpoint.get("model_type") != model_type
                or checkpoint.get("checkpoint_role") != "best_validation_soft_margin"
            ):
                raise ValueError(f"Unexpected verifier checkpoint: {checkpoint_path}")
            if tuple(checkpoint["candidate_steps"]) != EXPECTED_STEPS:
                raise ValueError(
                    f"Verifier checkpoint is not steps40-50: {checkpoint_path}"
                )
            signal_mean = np.asarray(checkpoint["signal_mean"], dtype=np.float32)
            signal_std = np.asarray(checkpoint["signal_std"], dtype=np.float32)
            if canonical_mean is None:
                canonical_mean = signal_mean
                canonical_std = signal_std
            elif not (
                np.array_equal(canonical_mean, signal_mean)
                and np.array_equal(canonical_std, signal_std)
            ):
                raise ValueError("Verifier state normalizers differ across seed runs")
            model = verifier.SparsePreemptionVerifier(
                input_dim=int(checkpoint["input_dim"]),
                state_dim=int(checkpoint["state_dim"]),
                hidden_dim=int(summary["training"]["hidden_dim"]),
                dropout=0.0,
                use_state=model_type == "preemption_state",
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            models[model_type] = model
            checkpoint_inputs[model_type] = {
                "path": str(checkpoint_path),
                "sha256": metadata["sha256"],
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_soft_margin_loss": float(
                    checkpoint["best_validation_soft_margin_loss"]
                ),
            }
        suites.append(models)
        inputs.append(
            {
                "train_seed": int(summary["train_seed"]),
                "run_summary": str(summary_path),
                "run_summary_sha256": base.sha256_file(summary_path),
                "checkpoints": checkpoint_inputs,
            }
        )
    if signature is None or canonical_mean is None or canonical_std is None:
        raise ValueError("No verifier suite loaded")
    return suites, inputs, signature, canonical_mean, canonical_std


def build_score_examples(
    split: str,
    trajectories: list[dict[str, Any]],
    normalized_signals: np.ndarray,
    shuffled_signals: np.ndarray,
    probabilities: np.ndarray,
    lambdas: list[float],
    candidate_steps: np.ndarray,
    cost_profile: np.ndarray,
    radius: int,
    temperature: float,
    harm_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    shuffled_inputs = []
    rows = []
    policy_groups = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for lambda_index, lambda_value in enumerate(lambdas):
            prior = probabilities[trajectory_index, lambda_index]
            anchor = int(np.argmax(prior))
            utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
            oracle_utility = float(utility.max())
            policy_groups.append(
                {
                    "trajectory_index": trajectory_index,
                    "lambda_index": lambda_index,
                    "lambda": lambda_value,
                    "prompt_id": int(trajectory["prompt_id"]),
                    "b4_utility": float(utility[anchor]),
                    "oracle_utility": oracle_utility,
                }
            )
            for current in range(max(0, anchor - radius), anchor):
                margin = verifier.restricted_suffix_margin(utility, current, anchor)
                target = 1.0 / (
                    1.0 + math.exp(-float(np.clip(margin / temperature, -8.0, 8.0)))
                )
                schedule = verifier.schedule_at(
                    trajectory, candidate_steps, lambda_value, cost_profile, current
                )
                inputs.append(
                    verifier.verifier_input(
                        normalized_signals[trajectory_index, current],
                        schedule,
                        prior,
                        current,
                    )
                )
                shuffled_inputs.append(
                    verifier.verifier_input(
                        shuffled_signals[trajectory_index, current],
                        schedule,
                        prior,
                        current,
                    )
                )
                rows.append(
                    {
                        "split": split,
                        "trajectory_index": trajectory_index,
                        "lambda_index": lambda_index,
                        "prompt_id": int(trajectory["prompt_id"]),
                        "seed": int(trajectory["seed"]),
                        "base_seed": int(trajectory["seed"])
                        - int(trajectory["prompt_id"]),
                        "lambda": lambda_value,
                        "candidate_index": current,
                        "candidate_step": int(candidate_steps[current]),
                        "b4_index": anchor,
                        "b4_step": int(candidate_steps[anchor]),
                        "offset_from_b4": current - anchor,
                        "raw_margin": margin,
                        "soft_target": target,
                        "positive_margin": int(margin > 0.0),
                        "material_positive_margin": int(margin > harm_epsilon),
                        "current_utility": float(utility[current]),
                        "b4_utility": float(utility[anchor]),
                        "oracle_utility": oracle_utility,
                        "current_quality": float(trajectory["qualities"][current]),
                        "b4_quality": float(trajectory["qualities"][anchor]),
                        "current_latency_sec": float(
                            trajectory["calibrated_latencies"][current]
                        ),
                        "b4_latency_sec": float(
                            trajectory["calibrated_latencies"][anchor]
                        ),
                    }
                )
    if not rows:
        raise ValueError(f"No score examples for {split}")
    return np.stack(inputs), np.stack(shuffled_inputs), rows, policy_groups


def attach_scores(
    inputs: np.ndarray,
    shuffled_inputs: np.ndarray,
    rows: list[dict[str, Any]],
    suites: list[dict[str, verifier.SparsePreemptionVerifier]],
    suite_inputs: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> None:
    control_scores = []
    state_scores = []
    shuffled_scores = []
    for models, metadata in zip(suites, suite_inputs, strict=True):
        train_seed = int(metadata["train_seed"])
        control = verifier.predict_logits(
            models["preemption_control"], inputs, device, batch_size
        )
        state = verifier.predict_logits(
            models["preemption_state"], inputs, device, batch_size
        )
        shuffled = verifier.predict_logits(
            models["preemption_state"], shuffled_inputs, device, batch_size
        )
        control_scores.append(control)
        state_scores.append(state)
        shuffled_scores.append(shuffled)
        for row_index, row in enumerate(rows):
            row[f"control_seed_{train_seed}"] = float(control[row_index])
            row[f"state_seed_{train_seed}"] = float(state[row_index])
            row[f"shuffled_seed_{train_seed}"] = float(shuffled[row_index])
    for name, values in (
        ("control_score", control_scores),
        ("state_score", state_scores),
        ("shuffled_score", shuffled_scores),
    ):
        mean_scores = np.stack(values).mean(axis=0)
        for row, score in zip(rows, mean_scores, strict=True):
            row[name] = float(score)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )


def safe_spearman(margins: np.ndarray, scores: np.ndarray) -> float:
    value = spearmanr(margins, scores).statistic
    return float(value) if np.isfinite(value) else float("nan")


def select_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        raise ValueError("Balanced threshold selection requires both target classes")
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    objective = true_positive - false_positive
    objective[~np.isfinite(thresholds)] = -np.inf
    index = int(np.argmax(objective))
    return float(thresholds[index])


def balanced_accuracy_at(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> float:
    prediction = scores >= threshold
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    sensitivity = prediction[positive].mean()
    specificity = (~prediction[negative]).mean()
    return float(0.5 * (sensitivity + specificity))


def rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = (
        "prompt_id",
        "base_seed",
        "lambda",
        "offset_from_b4",
        "raw_margin",
        "soft_target",
        "positive_margin",
        "material_positive_margin",
        "control_score",
        "state_score",
        "shuffled_score",
    )
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def metric_rows(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    train_arrays = rows_to_arrays(train_rows)
    global_thresholds = {}
    for target in TARGETS:
        labels = train_arrays[target].astype(np.int64)
        for model, score_name in (
            ("control", "control_score"),
            ("state", "state_score"),
            ("shuffled", "shuffled_score"),
        ):
            global_thresholds[(target, model)] = select_balanced_threshold(
                labels, train_arrays[score_name]
            )
    result = []
    for split, source_rows in (("train", train_rows), ("validation", validation_rows)):
        arrays = rows_to_arrays(source_rows)
        slices: list[tuple[str, str, np.ndarray]] = [
            ("macro", "all", np.ones(len(source_rows), dtype=bool))
        ]
        for lambda_value in sorted(set(arrays["lambda"])):
            slices.append(
                ("lambda", str(lambda_value), arrays["lambda"] == lambda_value)
            )
        for offset in (-3, -2, -1):
            slices.append(("offset", str(offset), arrays["offset_from_b4"] == offset))
        if split == "validation":
            for base_seed in sorted(set(arrays["base_seed"])):
                slices.append(
                    ("base_seed", str(base_seed), arrays["base_seed"] == base_seed)
                )
        for slice_type, slice_value, mask in slices:
            for target in TARGETS:
                labels = arrays[target][mask].astype(np.int64)
                margins = arrays["raw_margin"][mask].astype(np.float64)
                for model, score_name in (
                    ("control", "control_score"),
                    ("state", "state_score"),
                    ("shuffled", "shuffled_score"),
                ):
                    scores = arrays[score_name][mask].astype(np.float64)
                    threshold = global_thresholds[(target, model)]
                    result.append(
                        {
                            "split": split,
                            "slice_type": slice_type,
                            "slice_value": slice_value,
                            "target": target,
                            "model": model,
                            "row_count": int(mask.sum()),
                            "positive_rate": float(labels.mean()),
                            "auc": safe_auc(labels, scores),
                            "average_precision": safe_average_precision(labels, scores),
                            "spearman_to_raw_margin": safe_spearman(margins, scores),
                            "train_selected_balanced_threshold": threshold,
                            "balanced_accuracy": balanced_accuracy_at(
                                labels, scores, threshold
                            ),
                        }
                    )
    return result, global_thresholds


def per_checkpoint_metric_rows(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    suite_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        margins = np.asarray([row["raw_margin"] for row in rows], dtype=np.float64)
        for target in TARGETS:
            labels = np.asarray([row[target] for row in rows], dtype=np.int64)
            for metadata in suite_inputs:
                train_seed = int(metadata["train_seed"])
                for model, prefix in (("control", "control"), ("state", "state")):
                    scores = np.asarray(
                        [row[f"{prefix}_seed_{train_seed}"] for row in rows],
                        dtype=np.float64,
                    )
                    result.append(
                        {
                            "split": split,
                            "target": target,
                            "model": model,
                            "train_seed": train_seed,
                            "row_count": len(rows),
                            "positive_rate": float(labels.mean()),
                            "auc": safe_auc(labels, scores),
                            "average_precision": safe_average_precision(labels, scores),
                            "spearman_to_raw_margin": safe_spearman(margins, scores),
                        }
                    )
    return result


def weighted_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    metric: str,
) -> float:
    if metric == "auc":
        return float(roc_auc_score(labels, scores, sample_weight=weights))
    if metric == "average_precision":
        return float(average_precision_score(labels, scores, sample_weight=weights))
    raise ValueError(f"Unsupported weighted metric: {metric}")


def bootstrap_score_deltas(
    validation_rows: list[dict[str, Any]],
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    arrays = rows_to_arrays(validation_rows)
    prompts = np.asarray(sorted(set(arrays["prompt_id"])), dtype=np.int64)
    prompt_position = {prompt: index for index, prompt in enumerate(prompts)}
    row_prompt_position = np.asarray(
        [prompt_position[int(value)] for value in arrays["prompt_id"]], dtype=np.int64
    )
    result = []
    for target in TARGETS:
        labels = arrays[target].astype(np.int64)
        for comparison, reference_name in (
            ("state_vs_control", "control_score"),
            ("state_vs_shuffled", "shuffled_score"),
        ):
            for metric in ("auc", "average_precision"):
                point = weighted_metric(
                    labels,
                    arrays["state_score"],
                    np.ones(len(labels)),
                    metric,
                ) - weighted_metric(
                    labels,
                    arrays[reference_name],
                    np.ones(len(labels)),
                    metric,
                )
                draws = np.empty(samples, dtype=np.float64)
                for draw in range(samples):
                    sampled = rng.integers(0, len(prompts), size=len(prompts))
                    counts = np.bincount(sampled, minlength=len(prompts))
                    weights = counts[row_prompt_position]
                    draws[draw] = weighted_metric(
                        labels, arrays["state_score"], weights, metric
                    ) - weighted_metric(labels, arrays[reference_name], weights, metric)
                low, high = np.quantile(draws, [0.025, 0.975])
                result.append(
                    {
                        "target": target,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_delta": point,
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "bootstrap_samples": samples,
                    }
                )
    return result


def calibration_rows(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        arrays = rows_to_arrays(rows)
        for model, score_name in (
            ("control", "control_score"),
            ("state", "state_score"),
            ("shuffled", "shuffled_score"),
        ):
            scores = arrays[score_name].astype(np.float64)
            order = np.argsort(scores)
            for bin_index, indices in enumerate(np.array_split(order, 10)):
                probabilities = 1.0 / (1.0 + np.exp(-scores[indices]))
                result.append(
                    {
                        "split": split,
                        "model": model,
                        "score_decile": bin_index + 1,
                        "row_count": len(indices),
                        "mean_logit": float(scores[indices].mean()),
                        "mean_predicted_probability": float(probabilities.mean()),
                        "mean_soft_target": float(
                            arrays["soft_target"][indices].mean()
                        ),
                        "positive_margin_rate": float(
                            arrays["positive_margin"][indices].mean()
                        ),
                        "material_positive_margin_rate": float(
                            arrays["material_positive_margin"][indices].mean()
                        ),
                        "mean_raw_margin": float(arrays["raw_margin"][indices].mean()),
                    }
                )
    return result


def summarize_policy(
    rows: list[dict[str, Any]],
    policy_groups: list[dict[str, Any]],
    score_name: str,
    threshold: float,
    harm_epsilon: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["trajectory_index"]), int(row["lambda_index"]))].append(row)
    outcomes = []
    for group in policy_groups:
        key = (int(group["trajectory_index"]), int(group["lambda_index"]))
        candidates = grouped.get(key, [])
        candidates.sort(key=lambda row: int(row["candidate_index"]))
        chosen = next(
            (row for row in candidates if float(row[score_name]) >= threshold), None
        )
        if chosen is None:
            gain = 0.0
            quality_delta = 0.0
            latency_delta = 0.0
            step_delta = 0.0
            changed = 0.0
        else:
            gain = float(chosen["current_utility"] - chosen["b4_utility"])
            quality_delta = float(chosen["current_quality"] - chosen["b4_quality"])
            latency_delta = float(
                chosen["current_latency_sec"] - chosen["b4_latency_sec"]
            )
            step_delta = float(chosen["offset_from_b4"])
            changed = 1.0
        outcomes.append(
            {
                "prompt_id": int(group["prompt_id"]),
                "gain": gain,
                "quality_delta": quality_delta,
                "latency_delta": latency_delta,
                "step_delta": step_delta,
                "changed": changed,
                "harm": float(gain < -harm_epsilon),
                "material_gain": float(gain > harm_epsilon),
                "b4_regret": float(group["oracle_utility"] - group["b4_utility"]),
            }
        )
    by_prompt: dict[int, list[float]] = defaultdict(list)
    for outcome in outcomes:
        by_prompt[outcome["prompt_id"]].append(outcome["gain"])
    prompt_values = np.asarray(
        [np.mean(by_prompt[prompt]) for prompt in sorted(by_prompt)], dtype=np.float64
    )
    indices = rng.integers(
        0, len(prompt_values), size=(bootstrap_samples, len(prompt_values))
    )
    draws = prompt_values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    mean_gain = float(np.mean([row["gain"] for row in outcomes]))
    mean_b4_regret = float(np.mean([row["b4_regret"] for row in outcomes]))
    return {
        "mean_utility_gain_vs_b4": mean_gain,
        "utility_gain_ci95_low": float(low),
        "utility_gain_ci95_high": float(high),
        "harm_vs_b4_rate": float(np.mean([row["harm"] for row in outcomes])),
        "material_gain_rate": float(
            np.mean([row["material_gain"] for row in outcomes])
        ),
        "decision_change_rate": float(np.mean([row["changed"] for row in outcomes])),
        "mean_quality_delta": float(
            np.mean([row["quality_delta"] for row in outcomes])
        ),
        "mean_latency_delta_sec": float(
            np.mean([row["latency_delta"] for row in outcomes])
        ),
        "mean_step_delta": float(np.mean([row["step_delta"] for row in outcomes])),
        "recovered_b4_regret_fraction": (
            mean_gain / mean_b4_regret if mean_b4_regret > 0 else float("nan")
        ),
        "trajectory_lambda_count": len(outcomes),
    }


def policy_frontiers(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    train_policy_groups: list[dict[str, Any]],
    validation_policy_groups: list[dict[str, Any]],
    harm_epsilon: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topk_rows = []
    negative_rows = []
    for model, score_name in (
        ("control", "control_score"),
        ("state", "state_score"),
        ("shuffled", "shuffled_score"),
    ):
        train_scores = np.asarray([row[score_name] for row in train_rows])
        for fraction in TOP_FRACTIONS:
            threshold = float(np.quantile(train_scores, 1.0 - fraction))
            for split, rows, groups in (
                ("train", train_rows, train_policy_groups),
                ("validation", validation_rows, validation_policy_groups),
            ):
                lambda_values = sorted({float(group["lambda"]) for group in groups})
                for lambda_value in [None, *lambda_values]:
                    selected_rows = (
                        rows
                        if lambda_value is None
                        else [row for row in rows if row["lambda"] == lambda_value]
                    )
                    selected_groups = (
                        groups
                        if lambda_value is None
                        else [
                            group for group in groups if group["lambda"] == lambda_value
                        ]
                    )
                    summary = summarize_policy(
                        selected_rows,
                        selected_groups,
                        score_name,
                        threshold,
                        harm_epsilon,
                        bootstrap_samples,
                        rng,
                    )
                    topk_rows.append(
                        {
                            "split": split,
                            "model": model,
                            "lambda": (
                                "macro" if lambda_value is None else lambda_value
                            ),
                            "train_top_fraction": fraction,
                            "train_selected_score_threshold": threshold,
                            **summary,
                        }
                    )
        for threshold in NEGATIVE_THRESHOLDS:
            for split, rows, groups in (
                ("train", train_rows, train_policy_groups),
                ("validation", validation_rows, validation_policy_groups),
            ):
                lambda_values = sorted({float(group["lambda"]) for group in groups})
                for lambda_value in [None, *lambda_values]:
                    selected_rows = (
                        rows
                        if lambda_value is None
                        else [row for row in rows if row["lambda"] == lambda_value]
                    )
                    selected_groups = (
                        groups
                        if lambda_value is None
                        else [
                            group for group in groups if group["lambda"] == lambda_value
                        ]
                    )
                    summary = summarize_policy(
                        selected_rows,
                        selected_groups,
                        score_name,
                        threshold,
                        harm_epsilon,
                        bootstrap_samples,
                        rng,
                    )
                    negative_rows.append(
                        {
                            "split": split,
                            "model": model,
                            "lambda": (
                                "macro" if lambda_value is None else lambda_value
                            ),
                            "score_threshold": threshold,
                            **summary,
                        }
                    )
    return topk_rows, negative_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    b4_runs_root = Path(args.b4_runs_root).resolve()
    verifier_root = Path(args.verifier_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Score audit output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    device = torch.device(args.device)
    suites, verifier_inputs, signature, signal_mean, signal_std = load_verifier_suite(
        verifier_root, device
    )
    manifest = base.load_dataset_manifest(dataset_dir)
    dataset_sha256 = base.sha256_file(dataset_dir / "dataset_manifest.json")
    if dataset_sha256 != signature["dataset_manifest_sha256"]:
        raise ValueError("Verifier and requested state dataset differ")
    if not math.isclose(float(signature["harm_epsilon"]), 0.001):
        raise ValueError("Score audit target definitions require harm_epsilon=0.001")
    feature_indices, feature_names = verifier.select_sparse_feature_indices(manifest)
    if feature_names != signature["base_state_feature_names"]:
        raise ValueError("Verifier sparse feature names differ from dataset")
    train_trajectories = base.load_trajectories(
        dataset_dir, manifest, "train", feature_indices
    )
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", feature_indices
    )
    source_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    candidate_indices, candidate_steps = base.resolve_candidate_subset(
        source_steps, list(EXPECTED_STEPS)
    )
    source_costs, source_seconds, native_seconds, latency_provenance = (
        base.load_locked_latency_profile(
            manifest, source_steps, args.expected_latency_profile_sha256
        )
    )
    if latency_provenance["sha256"] != signature["latency_profile_sha256"]:
        raise ValueError("Verifier and requested locked latency profile differ")
    base.subset_trajectory_candidates(train_trajectories, candidate_indices)
    base.subset_trajectory_candidates(validation_trajectories, candidate_indices)
    cost_profile = source_costs[candidate_indices].copy()
    candidate_seconds = source_seconds[candidate_indices].copy()
    base.apply_locked_latency_profile(
        train_trajectories, cost_profile, candidate_seconds, native_seconds
    )
    base.apply_locked_latency_profile(
        validation_trajectories, cost_profile, candidate_seconds, native_seconds
    )
    train_raw = np.stack([row["features"] for row in train_trajectories])
    validation_raw = np.stack([row["features"] for row in validation_trajectories])
    train_signals = verifier.normalize_signals(
        verifier.build_causal_signals(train_raw), signal_mean, signal_std
    )
    validation_signals = verifier.normalize_signals(
        verifier.build_causal_signals(validation_raw), signal_mean, signal_std
    )
    shuffle_seed = int(signature["validation_shuffle"]["seed"])
    train_shuffled = verifier.shuffled_validation_signals(train_signals, shuffle_seed)
    validation_shuffled = verifier.shuffled_validation_signals(
        validation_signals, shuffle_seed
    )

    b4_models, b4_inputs = factor_audit.load_b4_ensemble(
        b4_runs_root, dataset_sha256, len(candidate_steps), device
    )
    if len(b4_models) != 5:
        raise ValueError(f"Expected five B4 models, found {len(b4_models)}")
    train_lambdas = [float(value) for value in signature["train_lambdas"]]
    eval_lambdas = [float(value) for value in signature["eval_lambdas"]]
    print("Computing frozen B4 ensemble probabilities...", flush=True)
    train_probabilities = verifier.ensemble_b4_probabilities(
        b4_models,
        train_trajectories,
        train_lambdas,
        device,
        args.inference_batch_size,
    )
    validation_probabilities = verifier.ensemble_b4_probabilities(
        b4_models,
        validation_trajectories,
        eval_lambdas,
        device,
        args.inference_batch_size,
    )
    verifier.validate_prompt_only_anchors(
        validation_trajectories, validation_probabilities
    )
    train_inputs, train_shuffle_inputs, train_rows, train_policy_groups = (
        build_score_examples(
            "train",
            train_trajectories,
            train_signals,
            train_shuffled,
            train_probabilities,
            train_lambdas,
            candidate_steps,
            cost_profile,
            int(signature["radius"]),
            float(signature["training"]["margin_temperature"]),
            float(signature["harm_epsilon"]),
        )
    )
    (
        validation_inputs,
        validation_shuffle_inputs,
        validation_rows,
        validation_policy_groups,
    ) = build_score_examples(
        "validation",
        validation_trajectories,
        validation_signals,
        validation_shuffled,
        validation_probabilities,
        eval_lambdas,
        candidate_steps,
        cost_profile,
        int(signature["radius"]),
        float(signature["training"]["margin_temperature"]),
        float(signature["harm_epsilon"]),
    )
    print(
        "Scoring train and validation candidates with frozen checkpoints...", flush=True
    )
    attach_scores(
        train_inputs,
        train_shuffle_inputs,
        train_rows,
        suites,
        verifier_inputs,
        device,
        args.inference_batch_size,
    )
    attach_scores(
        validation_inputs,
        validation_shuffle_inputs,
        validation_rows,
        suites,
        verifier_inputs,
        device,
        args.inference_batch_size,
    )

    scores = train_rows + validation_rows
    metrics, train_thresholds = metric_rows(train_rows, validation_rows)
    per_checkpoint_metrics = per_checkpoint_metric_rows(
        train_rows, validation_rows, verifier_inputs
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_deltas = bootstrap_score_deltas(
        validation_rows, args.bootstrap_samples, rng
    )
    calibration = calibration_rows(train_rows, validation_rows)
    topk, negative = policy_frontiers(
        train_rows,
        validation_rows,
        train_policy_groups,
        validation_policy_groups,
        float(signature["harm_epsilon"]),
        args.bootstrap_samples,
        rng,
    )
    artifacts = {
        "candidate_scores": ("candidate_scores.csv", scores),
        "score_metrics": ("score_metrics.csv", metrics),
        "per_checkpoint_score_metrics": (
            "per_checkpoint_score_metrics.csv",
            per_checkpoint_metrics,
        ),
        "score_bootstrap_deltas": (
            "score_bootstrap_deltas.csv",
            bootstrap_deltas,
        ),
        "calibration_bins": ("calibration_bins.csv", calibration),
        "topk_policy_frontier": ("topk_policy_frontier.csv", topk),
        "negative_threshold_frontier": (
            "negative_threshold_frontier.csv",
            negative,
        ),
    }
    for _, (filename, rows) in artifacts.items():
        write_csv(out_dir / filename, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "frozen_b4_preemption_score_ranking_and_calibration_diagnostic",
        "diagnostic_only": True,
        "formal_evidence": False,
        "evaluation_splits": ["train", "validation"],
        "test_accessed": False,
        "weights_updated": False,
        "candidate_steps": candidate_steps.tolist(),
        "radius": int(signature["radius"]),
        "targets": TARGETS,
        "top_fractions": TOP_FRACTIONS,
        "negative_thresholds": NEGATIVE_THRESHOLDS,
        "train_selected_balanced_thresholds": {
            f"{target}.{model}": threshold
            for (target, model), threshold in train_thresholds.items()
        },
        "bootstrap": {
            "unit": "prompt_cluster_preserving_candidate_generation_seed_lambda_rows",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": dataset_sha256,
        "latency_profile": latency_provenance,
        "verifier_protocol": signature,
        "verifier_inputs": verifier_inputs,
        "b4_inputs": b4_inputs,
        "row_counts": {
            "train_candidates": len(train_rows),
            "validation_candidates": len(validation_rows),
            "train_trajectory_lambdas": len(train_policy_groups),
            "validation_trajectory_lambdas": len(validation_policy_groups),
        },
        "limitations": [
            "The checkpoints were selected on the same validation split, so validation score geometry remains selection-stage diagnostic evidence.",
            "Negative score thresholds are diagnostic only and are not selected deployment policies.",
            "Top-k thresholds are fitted from train score quantiles and do not use validation utility labels.",
        ],
        "artifacts": {
            name: {
                "path": filename,
                "sha256": base.sha256_file(out_dir / filename),
            }
            for name, (filename, _) in artifacts.items()
        },
    }
    report_path = out_dir / "score_geometry_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
