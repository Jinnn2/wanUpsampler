#!/usr/bin/env python3
"""Train a sparse causal verifier that may preempt frozen B4 by up to three steps."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from . import analyze_factor_relevance as factor_audit
    from . import train_variable_lambda_router as base
except ImportError:
    import analyze_factor_relevance as factor_audit
    import train_variable_lambda_router as base


RUN_SCHEMA = "b4_sparse_preemption_verifier_run_v1"
CHECKPOINT_SCHEMA = "b4_sparse_preemption_verifier_checkpoint_v1"
EXPECTED_STEPS = tuple(range(40, 51))
BASE_STATE_FEATURES = (
    "residual.temporal_gradient_abs_mean",
    "residual.temporal_second_abs_mean",
    "x0.temporal_gradient_abs_mean",
    "x0.temporal_second_abs_mean",
)
SIGNAL_NAMES = ("value", "delta", "slope2")
MODEL_LABELS = {
    "b4_offline": "Frozen Five-Seed B4 Ensemble",
    "preemption_control": "B4-3 Schedule/Prior Control",
    "preemption_state": "B4-3 Sparse State Verifier",
    "preemption_state_shuffled": "B4-3 Shuffled-State Diagnostic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--b4-runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate-steps", type=int, nargs="+", default=EXPECTED_STEPS
    )
    parser.add_argument(
        "--train-lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
    )
    parser.add_argument(
        "--eval-lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--primary-lambda", type=float, default=0.08)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--margin-temperature", type=float, default=0.001)
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument(
        "--risk-thresholds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0]
    )
    parser.add_argument("--checkpoint-risk-threshold", type=float, default=1.0)
    parser.add_argument("--max-validation-harm-rate", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-shuffle-seed", type=int, default=2035)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--expected-latency-profile-sha256", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.train_lambdas = sorted(set(float(value) for value in args.train_lambdas))
    args.eval_lambdas = sorted(set(float(value) for value in args.eval_lambdas))
    args.risk_thresholds = sorted(set(float(value) for value in args.risk_thresholds))
    if tuple(args.candidate_steps) != EXPECTED_STEPS:
        parser.error("This experiment is locked to exact candidate steps 40--50")
    if args.radius != 3:
        parser.error("This first verifier experiment is locked to radius=3")
    if args.primary_lambda not in args.eval_lambdas:
        parser.error("primary-lambda must be present in eval-lambdas")
    if args.checkpoint_risk_threshold not in args.risk_thresholds:
        parser.error("checkpoint-risk-threshold must be in risk-thresholds")
    positive = (
        args.margin_temperature,
        args.hidden_dim,
        args.epochs,
        args.batch_size,
        args.lr,
        args.inference_batch_size,
    )
    if any(float(value) <= 0 or not math.isfinite(float(value)) for value in positive):
        parser.error(
            "temperature, dimensions, epochs, batch sizes, and lr must be positive"
        )
    if (
        args.harm_epsilon < 0
        or args.max_validation_harm_rate < 0
        or args.weight_decay < 0
        or not 0 <= args.dropout < 1
        or args.num_workers < 0
        or any(not math.isfinite(value) for value in args.risk_thresholds)
    ):
        parser.error("Invalid harm, threshold, optimizer, dropout, or worker value")
    return args


def select_sparse_feature_indices(
    manifest: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    feature_names = [str(value) for value in manifest["feature_names"]]
    missing = [name for name in BASE_STATE_FEATURES if name not in feature_names]
    if missing:
        raise ValueError(f"Sparse verifier features are missing: {missing}")
    indices = np.asarray([feature_names.index(name) for name in BASE_STATE_FEATURES])
    return indices, list(BASE_STATE_FEATURES)


def build_causal_signals(raw_state: np.ndarray) -> np.ndarray:
    """Return value, one-step delta, and two-step backward slope per factor."""
    values = np.asarray(raw_state, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("raw_state must have [trajectory, step, feature] shape")
    delta = np.zeros_like(values)
    delta[:, 1:] = values[:, 1:] - values[:, :-1]
    slope2 = np.zeros_like(values)
    slope2[:, 2:] = (values[:, 2:] - values[:, :-2]) / 2.0
    return np.stack([values, delta, slope2], axis=-1).reshape(
        values.shape[0], values.shape[1], -1
    )


def sparse_signal_names(base_names: list[str]) -> list[str]:
    return [f"{name}.{signal}" for name in base_names for signal in SIGNAL_NAMES]


def fit_step_normalizer(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = signals.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = signals.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def normalize_signals(
    signals: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    if signals.shape[1:] != mean.shape or mean.shape != std.shape:
        raise ValueError("Sparse signal normalizer shape mismatch")
    return ((signals - mean[None]) / std[None]).astype(np.float32)


def restricted_suffix_margin(
    utility: np.ndarray, current_index: int, anchor_index: int
) -> float:
    if not 0 <= current_index < anchor_index < len(utility):
        raise ValueError("Restricted suffix margin requires current < B4 anchor")
    return float(
        utility[current_index] - utility[current_index + 1 : anchor_index + 1].max()
    )


def prior_context(probabilities: np.ndarray, current_index: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    anchor = int(np.argmax(probs))
    sorted_probs = np.sort(probs)
    entropy = float(
        -(probs * np.log(np.maximum(probs, 1e-8))).sum() / math.log(len(probs))
    )
    extras = np.asarray(
        [
            entropy,
            float(sorted_probs[-1] - sorted_probs[-2]),
            float(probs[current_index]),
            float(probs[anchor]),
            float(np.dot(probs, np.arange(len(probs))) / max(len(probs) - 1, 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([probs, extras])


@torch.no_grad()
def ensemble_b4_probabilities(
    models: list[base.VariableLambdaB4Prior],
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty(
        (len(trajectories), len(lambdas), models[0].candidate_count), dtype=np.float32
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


def validate_prompt_only_anchors(
    trajectories: list[dict[str, Any]], probabilities: np.ndarray
) -> None:
    by_prompt: dict[int, np.ndarray] = {}
    anchors = probabilities.argmax(axis=2)
    for index, trajectory in enumerate(trajectories):
        prompt_id = int(trajectory["prompt_id"])
        if prompt_id in by_prompt and not np.array_equal(
            by_prompt[prompt_id], anchors[index]
        ):
            raise ValueError("Prompt-only B4 ensemble differs across generation seeds")
        by_prompt[prompt_id] = anchors[index].copy()


def schedule_at(
    trajectory: dict[str, Any],
    candidate_steps: np.ndarray,
    lambda_value: float,
    cost_profile: np.ndarray,
    index: int,
) -> np.ndarray:
    return base.schedule_features(
        candidate_steps, trajectory["sigmas"], lambda_value, cost_profile
    )[index]


def verifier_input(
    state: np.ndarray,
    schedule: np.ndarray,
    probabilities: np.ndarray,
    current_index: int,
) -> np.ndarray:
    return np.concatenate(
        [state, schedule, prior_context(probabilities, current_index)]
    ).astype(np.float32, copy=False)


def build_training_examples(
    trajectories: list[dict[str, Any]],
    normalized_signals: np.ndarray,
    b4_probabilities: np.ndarray,
    lambdas: list[float],
    candidate_steps: np.ndarray,
    cost_profile: np.ndarray,
    radius: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for lambda_index, lambda_value in enumerate(lambdas):
            probabilities = b4_probabilities[trajectory_index, lambda_index]
            anchor = int(np.argmax(probabilities))
            utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
            for current in range(max(0, anchor - radius), anchor):
                margin = restricted_suffix_margin(utility, current, anchor)
                target = 1.0 / (
                    1.0 + math.exp(-float(np.clip(margin / temperature, -8, 8)))
                )
                inputs.append(
                    verifier_input(
                        normalized_signals[trajectory_index, current],
                        schedule_at(
                            trajectory,
                            candidate_steps,
                            lambda_value,
                            cost_profile,
                            current,
                        ),
                        probabilities,
                        current,
                    )
                )
                targets.append(target)
    if not inputs:
        raise ValueError("B4 anchors produced no preemption training examples")
    return np.stack(inputs), np.asarray(targets, dtype=np.float32)


class PreemptionDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.inputs[index]), torch.tensor(self.targets[index])


class SparsePreemptionVerifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        dropout: float,
        use_state: bool,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.use_state = bool(use_state)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.SiLU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_state:
            effective = inputs
        else:
            effective = inputs.clone()
            effective[:, : self.state_dim] = 0.0
        return self.network(effective).squeeze(-1)


@torch.no_grad()
def predict_logits(
    model: SparsePreemptionVerifier,
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(inputs), batch_size):
        tensor = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        chunks.append(model(tensor).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def sequential_choice(
    logits: np.ndarray, candidate_indices: list[int], anchor: int, threshold: float
) -> int:
    for logit, candidate in zip(logits, candidate_indices, strict=True):
        if float(logit) >= threshold:
            return int(candidate)
    return int(anchor)


def shuffled_validation_signals(signals: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty_like(signals)
    for step in range(signals.shape[1]):
        result[:, step] = signals[rng.permutation(signals.shape[0]), step]
    return result


def prediction_row(
    trajectory: dict[str, Any],
    lambda_value: float,
    chosen: int,
    anchor: int,
    candidate_steps: np.ndarray,
    quality_dimensions: list[str],
    model_type: str,
    threshold: str | float,
    chosen_logit: str | float,
    harm_epsilon: float,
) -> dict[str, Any]:
    utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
    oracle = int(np.argmax(utility))
    realized = float(utility[chosen])
    b4_utility = float(utility[anchor])
    oracle_utility = float(utility[oracle])
    delta = realized - b4_utility
    row = {
        "split": trajectory["split"],
        "model_type": model_type,
        "Method": MODEL_LABELS[model_type],
        "risk_threshold": threshold,
        "prompt_id": int(trajectory["prompt_id"]),
        "seed": int(trajectory["seed"]),
        "lambda": lambda_value,
        "chosen_step": int(candidate_steps[chosen]),
        "b4_step": int(candidate_steps[anchor]),
        "oracle_step": int(candidate_steps[oracle]),
        "step_delta_vs_b4": int(chosen - anchor),
        "decision_changed": int(chosen != anchor),
        "policy_regret": max(0.0, oracle_utility - realized),
        "realized_utility": realized,
        "b4_utility": b4_utility,
        "utility_gain_vs_b4": delta,
        "harm_vs_b4": int(delta < -harm_epsilon),
        "realized_vbench5": float(trajectory["qualities"][chosen]),
        "realized_latency_sec": float(trajectory["calibrated_latencies"][chosen]),
        "speedup_vs_native": float(
            trajectory["calibrated_native_latency"]
            / trajectory["calibrated_latencies"][chosen]
        ),
        "chosen_preemption_logit": chosen_logit,
    }
    for dimension_index, name in enumerate(quality_dimensions):
        row[f"realized_{name}"] = float(
            trajectory["dimensions"][chosen, dimension_index]
        )
    return row


def evaluate_model(
    model: SparsePreemptionVerifier,
    model_type: str,
    trajectories: list[dict[str, Any]],
    normalized_signals: np.ndarray,
    b4_probabilities: np.ndarray,
    lambdas: list[float],
    candidate_steps: np.ndarray,
    cost_profile: np.ndarray,
    radius: int,
    thresholds: list[float],
    harm_epsilon: float,
    quality_dimensions: list[str],
    device: torch.device,
    batch_size: int,
    emit_rows: bool,
) -> tuple[dict[float, dict[str, float]], list[dict[str, Any]]]:
    flat_inputs: list[np.ndarray] = []
    groups: list[tuple[int, int, int, list[int], int]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for lambda_index, lambda_value in enumerate(lambdas):
            probabilities = b4_probabilities[trajectory_index, lambda_index]
            anchor = int(np.argmax(probabilities))
            candidates = list(range(max(0, anchor - radius), anchor))
            start = len(flat_inputs)
            for current in candidates:
                flat_inputs.append(
                    verifier_input(
                        normalized_signals[trajectory_index, current],
                        schedule_at(
                            trajectory,
                            candidate_steps,
                            lambda_value,
                            cost_profile,
                            current,
                        ),
                        probabilities,
                        current,
                    )
                )
            groups.append((trajectory_index, lambda_index, start, candidates, anchor))
    input_matrix = (
        np.stack(flat_inputs)
        if flat_inputs
        else np.empty((0, model.network[0].in_features), dtype=np.float32)
    )
    logits = predict_logits(model, input_matrix, device, batch_size)
    rows: list[dict[str, Any]] = []
    metrics: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        regrets = []
        gains = []
        harms = []
        changes = []
        for trajectory_index, lambda_index, start, candidates, anchor in groups:
            local_logits = logits[start : start + len(candidates)]
            chosen = sequential_choice(local_logits, candidates, anchor, threshold)
            trajectory = trajectories[trajectory_index]
            lambda_value = lambdas[lambda_index]
            utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
            oracle_utility = float(utility.max())
            realized = float(utility[chosen])
            gain = realized - float(utility[anchor])
            regrets.append(max(0.0, oracle_utility - realized))
            gains.append(gain)
            harms.append(float(gain < -harm_epsilon))
            changes.append(float(chosen != anchor))
            if emit_rows:
                chosen_logit: str | float = ""
                if chosen != anchor:
                    chosen_logit = float(local_logits[candidates.index(chosen)])
                rows.append(
                    prediction_row(
                        trajectory,
                        lambda_value,
                        chosen,
                        anchor,
                        candidate_steps,
                        quality_dimensions,
                        model_type,
                        threshold,
                        chosen_logit,
                        harm_epsilon,
                    )
                )
        metrics[threshold] = {
            "macro_policy_regret": float(np.mean(regrets)),
            "mean_utility_gain_vs_b4": float(np.mean(gains)),
            "harm_vs_b4_rate": float(np.mean(harms)),
            "decision_change_rate": float(np.mean(changes)),
        }
    return metrics, rows


def b4_prediction_rows(
    trajectories: list[dict[str, Any]],
    probabilities: np.ndarray,
    lambdas: list[float],
    candidate_steps: np.ndarray,
    quality_dimensions: list[str],
    harm_epsilon: float,
) -> list[dict[str, Any]]:
    rows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for lambda_index, lambda_value in enumerate(lambdas):
            anchor = int(np.argmax(probabilities[trajectory_index, lambda_index]))
            rows.append(
                prediction_row(
                    trajectory,
                    lambda_value,
                    anchor,
                    anchor,
                    candidate_steps,
                    quality_dimensions,
                    "b4_offline",
                    "baseline",
                    "",
                    harm_epsilon,
                )
            )
    return rows


def train_verifier(
    model_type: str,
    dataset: PreemptionDataset,
    validation_args: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SparsePreemptionVerifier, int, list[dict[str, Any]]]:
    use_state = model_type == "preemption_state"
    model = SparsePreemptionVerifier(
        input_dim=dataset.inputs.shape[1],
        state_dim=len(BASE_STATE_FEATURES) * len(SIGNAL_NAMES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_state=use_state,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    primary = args.checkpoint_risk_threshold
    initial_metrics, _ = evaluate_model(
        model, model_type, thresholds=[primary], emit_rows=False, **validation_args
    )
    best_epoch = 0
    best_regret = initial_metrics[primary]["macro_policy_regret"]
    best_weights = copy.deepcopy(model.state_dict())
    history = [
        {
            "epoch": 0,
            "train_soft_margin_loss": "",
            "train_soft_margin_excess": "",
            "validation_macro_policy_regret": best_regret,
            "validation_mean_utility_gain_vs_b4": initial_metrics[primary][
                "mean_utility_gain_vs_b4"
            ],
            "validation_harm_vs_b4_rate": initial_metrics[primary]["harm_vs_b4_rate"],
            "validation_decision_change_rate": initial_metrics[primary][
                "decision_change_rate"
            ],
            "selected_as_best": True,
        }
    ]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        excesses = []
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = targets.to(device)
            logits = model(inputs.to(device))
            element = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss = element.mean()
            entropy = F.binary_cross_entropy(targets, targets, reduction="none")
            excess = (element - entropy).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {model_type}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            excesses.append(float(excess.detach()))
        scheduler.step()
        metrics, _ = evaluate_model(
            model, model_type, thresholds=[primary], emit_rows=False, **validation_args
        )
        current = metrics[primary]
        eligible = current["harm_vs_b4_rate"] <= args.max_validation_harm_rate
        selected = eligible and current["macro_policy_regret"] < best_regret - 1e-12
        if selected:
            best_epoch = epoch
            best_regret = current["macro_policy_regret"]
            best_weights = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_soft_margin_loss": float(np.mean(losses)),
                "train_soft_margin_excess": float(np.mean(excesses)),
                "validation_macro_policy_regret": current["macro_policy_regret"],
                "validation_mean_utility_gain_vs_b4": current[
                    "mean_utility_gain_vs_b4"
                ],
                "validation_harm_vs_b4_rate": current["harm_vs_b4_rate"],
                "validation_decision_change_rate": current["decision_change_rate"],
                "selected_as_best": selected,
            }
        )
        print(
            f"[{model_type}] epoch={epoch:02d} "
            f"loss={np.mean(losses):.6f} "
            f"val_gain={current['mean_utility_gain_vs_b4']:.6f} "
            f"val_harm={current['harm_vs_b4_rate']:.4f} "
            f"selected={selected}",
            flush=True,
        )
    model.load_state_dict(best_weights)
    model.eval()
    return model, best_epoch, history


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base.seed_everything(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset_dir = Path(args.dataset_dir).resolve()
    b4_runs_root = Path(args.b4_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    print(f"Loading validation-only state dataset: {dataset_dir}", flush=True)
    manifest = base.load_dataset_manifest(dataset_dir)
    feature_indices, base_feature_names = select_sparse_feature_indices(manifest)
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
    source_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    candidate_indices, candidate_steps = base.resolve_candidate_subset(
        source_steps, args.candidate_steps
    )
    source_costs, source_seconds, native_seconds, latency_provenance = (
        base.load_locked_latency_profile(
            manifest, source_steps, args.expected_latency_profile_sha256
        )
    )
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

    train_raw = np.stack([item["features"] for item in train_trajectories])
    validation_raw = np.stack([item["features"] for item in validation_trajectories])
    train_signals = build_causal_signals(train_raw)
    validation_signals = build_causal_signals(validation_raw)
    signal_mean, signal_std = fit_step_normalizer(train_signals)
    train_signals = normalize_signals(train_signals, signal_mean, signal_std)
    validation_signals = normalize_signals(validation_signals, signal_mean, signal_std)

    device = torch.device(args.device)
    dataset_sha256 = base.sha256_file(dataset_dir / "dataset_manifest.json")
    b4_models, b4_inputs = factor_audit.load_b4_ensemble(
        b4_runs_root, dataset_sha256, len(candidate_steps), device
    )
    if len(b4_models) != 5:
        raise ValueError(
            f"Expected exactly five frozen B4 models, found {len(b4_models)}"
        )
    for item in b4_inputs:
        payload = json.loads(Path(item["run_summary"]).read_text(encoding="utf-8"))
        if tuple(payload.get("candidate_steps", [])) != EXPECTED_STEPS:
            raise ValueError(f"B4 input is not exact steps40-50: {item['run_summary']}")

    print("Computing frozen five-seed B4 ensemble probabilities...", flush=True)
    train_probabilities = ensemble_b4_probabilities(
        b4_models,
        train_trajectories,
        args.train_lambdas,
        device,
        args.inference_batch_size,
    )
    validation_probabilities = ensemble_b4_probabilities(
        b4_models,
        validation_trajectories,
        args.eval_lambdas,
        device,
        args.inference_batch_size,
    )
    validate_prompt_only_anchors(validation_trajectories, validation_probabilities)
    training_inputs, training_targets = build_training_examples(
        train_trajectories,
        train_signals,
        train_probabilities,
        args.train_lambdas,
        candidate_steps,
        cost_profile,
        args.radius,
        args.margin_temperature,
    )
    train_dataset = PreemptionDataset(training_inputs, training_targets)
    print(
        f"Prepared {len(train_dataset)} local B4-3 margin examples; "
        f"input_dim={training_inputs.shape[1]}",
        flush=True,
    )
    fixed_steps = base.best_fixed_steps(train_trajectories, args.eval_lambdas)
    validation_args = {
        "trajectories": validation_trajectories,
        "normalized_signals": validation_signals,
        "b4_probabilities": validation_probabilities,
        "lambdas": args.eval_lambdas,
        "candidate_steps": candidate_steps,
        "cost_profile": cost_profile,
        "radius": args.radius,
        "harm_epsilon": args.harm_epsilon,
        "quality_dimensions": manifest["quality_dimensions"],
        "device": device,
        "batch_size": args.inference_batch_size,
    }

    prediction_rows = b4_prediction_rows(
        validation_trajectories,
        validation_probabilities,
        args.eval_lambdas,
        candidate_steps,
        manifest["quality_dimensions"],
        args.harm_epsilon,
    )
    model_summary = []
    checkpoints: dict[str, dict[str, str]] = {}
    histories: dict[str, dict[str, str]] = {}
    trained_models: dict[str, SparsePreemptionVerifier] = {}
    for model_type in ("preemption_control", "preemption_state"):
        base.seed_everything(args.seed)
        model, best_epoch, history = train_verifier(
            model_type, train_dataset, validation_args, args, device
        )
        trained_models[model_type] = model
        metrics, rows = evaluate_model(
            model,
            model_type,
            thresholds=args.risk_thresholds,
            emit_rows=True,
            **validation_args,
        )
        prediction_rows.extend(rows)
        primary_metrics = metrics[args.checkpoint_risk_threshold]
        model_summary.append(
            {
                "model_type": model_type,
                "best_epoch": best_epoch,
                "checkpoint_risk_threshold": args.checkpoint_risk_threshold,
                **primary_metrics,
            }
        )
        history_path = out_dir / f"{model_type}_training_history.csv"
        write_csv(history_path, history)
        histories[model_type] = {
            "path": history_path.name,
            "sha256": base.sha256_file(history_path),
        }
        checkpoint_path = out_dir / f"{model_type}_verifier.pt"
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "evaluation_protocol": base.EVALUATION_PROTOCOL,
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "input_dim": training_inputs.shape[1],
                "state_dim": len(BASE_STATE_FEATURES) * len(SIGNAL_NAMES),
                "base_state_feature_names": base_feature_names,
                "sparse_signal_names": sparse_signal_names(base_feature_names),
                "signal_mean": signal_mean,
                "signal_std": signal_std,
                "candidate_steps": candidate_steps,
                "radius": args.radius,
                "margin_temperature": args.margin_temperature,
                "best_epoch": best_epoch,
                "checkpoint_risk_threshold": args.checkpoint_risk_threshold,
                "b4_ensemble_inputs": b4_inputs,
            },
            checkpoint_path,
        )
        checkpoints[model_type] = {
            "path": checkpoint_path.name,
            "sha256": base.sha256_file(checkpoint_path),
        }

    shuffled_signals = shuffled_validation_signals(
        validation_signals, args.validation_shuffle_seed
    )
    shuffled_args = dict(validation_args)
    shuffled_args["normalized_signals"] = shuffled_signals
    _, shuffled_rows = evaluate_model(
        trained_models["preemption_state"],
        "preemption_state_shuffled",
        thresholds=args.risk_thresholds,
        emit_rows=True,
        **shuffled_args,
    )
    prediction_rows.extend(shuffled_rows)

    predictions_path = out_dir / "selection_predictions.csv"
    summary_path = out_dir / "selection_model_summary.csv"
    write_csv(predictions_path, prediction_rows)
    write_csv(summary_path, model_summary)
    run_summary = {
        "schema": RUN_SCHEMA,
        "evaluation_protocol": base.EVALUATION_PROTOCOL,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_split": "validation",
        "selection_only": True,
        "formal_evidence": False,
        "test_accessed": False,
        "train_seed": args.seed,
        "model_types": list(MODEL_LABELS),
        "train_lambdas": args.train_lambdas,
        "eval_lambdas": args.eval_lambdas,
        "primary_lambda": args.primary_lambda,
        "harm_epsilon": args.harm_epsilon,
        "risk_thresholds": args.risk_thresholds,
        "checkpoint_risk_threshold": args.checkpoint_risk_threshold,
        "max_validation_harm_rate": args.max_validation_harm_rate,
        "source_candidate_steps": source_steps.tolist(),
        "candidate_steps": candidate_steps.tolist(),
        "radius": args.radius,
        "action_space": "sequential_stop_or_continue_over_b4_minus_3_to_b4",
        "target": "soft_restricted_suffix_best_utility_margin",
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": dataset_sha256,
        "base_state_feature_names": base_feature_names,
        "sparse_signal_names": sparse_signal_names(base_feature_names),
        "selected_feature_count": len(base_feature_names) * len(SIGNAL_NAMES),
        "state_normalization": "train_only_per_absolute_candidate_step_v1",
        "train_prompts": len(train_prompts),
        "validation_prompts": len(validation_prompts),
        "train_trajectories": len(train_trajectories),
        "validation_trajectories": len(validation_trajectories),
        "training_example_count": len(train_dataset),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
            "margin_temperature": args.margin_temperature,
            "loss": "equal_weight_soft_target_binary_cross_entropy",
            "output_initialization": "zero_weight_bias_minus_2_fallback_b4",
        },
        "validation_shuffle": {
            "method": "cross_trajectory_independent_per_absolute_step",
            "seed": args.validation_shuffle_seed,
            "selection_role": "diagnostic_only",
        },
        "cost_profile": cost_profile.tolist(),
        "calibrated_candidate_latency_seconds": candidate_seconds.tolist(),
        "calibrated_native_latency_seconds": native_seconds,
        "latency_profile": latency_provenance,
        "b4_ensemble_size": len(b4_models),
        "b4_inputs": b4_inputs,
        "fixed_steps_selected_from_train": {
            str(key): int(candidate_steps[value]) for key, value in fixed_steps.items()
        },
        "models": model_summary,
        "artifacts": {
            "predictions": {
                "path": predictions_path.name,
                "sha256": base.sha256_file(predictions_path),
            },
            "model_summary": {
                "path": summary_path.name,
                "sha256": base.sha256_file(summary_path),
            },
            "checkpoints": checkpoints,
            "training_histories": histories,
        },
    }
    report_path = out_dir / "run_summary.json"
    report_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
