#!/usr/bin/env python3
"""Train a tiny online guard that may move a train-selected fixed step by one."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
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
    from . import train_variable_lambda_router as base
except ImportError:
    import train_variable_lambda_router as base


RUN_SCHEMA = "fixed_guard_router_selection_run_v1"
CHECKPOINT_SCHEMA = "fixed_guard_router_checkpoint_v1"
EVALUATION_PROTOCOL = "deterministic_eval_mode_v1"
MODEL_TYPES = ("fixed_guard_prompt", "fixed_guard_state")
DEFAULT_FEATURES = (
    "residual.temporal_gradient_abs_mean",
    "residual.temporal_second_abs_mean",
    "x0.temporal_gradient_abs_mean",
    "x0.temporal_second_abs_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--eval-lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--primary-lambda", type=float, default=0.08)
    parser.add_argument("--state-features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--split-salt", default="fixed_guard_v1")
    parser.add_argument(
        "--utility-margins",
        type=float,
        nargs="+",
        default=[0.0, 0.00025, 0.0005, 0.001, 0.0015, 0.002, 0.003],
    )
    parser.add_argument("--max-calibration-harm-rate", type=float, default=0.02)
    parser.add_argument("--min-calibration-action-rate", type=float, default=0.02)
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quality-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--expected-latency-profile-sha256", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.eval_lambdas = sorted(set(float(value) for value in args.eval_lambdas))
    args.utility_margins = sorted(set(float(value) for value in args.utility_margins))
    if args.primary_lambda not in args.eval_lambdas:
        parser.error("primary-lambda must be included in eval-lambdas")
    if not 0 < args.calibration_fraction < 0.5:
        parser.error("calibration-fraction must be in (0, 0.5)")
    if not 0 <= args.max_calibration_harm_rate <= 1:
        parser.error("max-calibration-harm-rate must be in [0, 1]")
    if not 0 <= args.min_calibration_action_rate <= 1:
        parser.error("min-calibration-action-rate must be in [0, 1]")
    positive = (
        args.epochs,
        args.batch_size,
        args.lr,
        args.quality_scale,
        args.inference_batch_size,
    )
    if any(float(value) <= 0 or not math.isfinite(float(value)) for value in positive):
        parser.error("epochs, batch sizes, lr, and quality-scale must be positive")
    if any(value < 0 or not math.isfinite(value) for value in args.utility_margins):
        parser.error("utility-margins must be finite and non-negative")
    return args


def prompt_bucket(prompt_id: int, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{prompt_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_fit_calibration(
    trajectories: list[dict[str, Any]], fraction: float, salt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration_ids = {
        int(item["prompt_id"])
        for item in trajectories
        if prompt_bucket(int(item["prompt_id"]), salt) < fraction
    }
    fit = [item for item in trajectories if int(item["prompt_id"]) not in calibration_ids]
    calibration = [
        item for item in trajectories if int(item["prompt_id"]) in calibration_ids
    ]
    if not fit or not calibration:
        raise ValueError("Fit/calibration prompt split is empty")
    return fit, calibration


def build_causal_signals(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    delta = np.zeros_like(values)
    delta[:, 1:] = values[:, 1:] - values[:, :-1]
    slope = np.zeros_like(values)
    slope[:, 2:] = (values[:, 2:] - values[:, :-2]) / 2.0
    return np.stack([values, delta, slope], axis=-1).reshape(
        values.shape[0], values.shape[1], -1
    )


def fit_signal_normalizer(
    trajectories: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    signals = build_causal_signals(
        np.stack([trajectory["features"] for trajectory in trajectories])
    )
    mean = signals.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = signals.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def normalized_signals(
    trajectories: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    raw = np.stack([trajectory["features"] for trajectory in trajectories])
    return ((build_causal_signals(raw) - mean[None]) / std[None]).astype(np.float32)


def pair_schedule(trajectory: dict[str, Any], index: int) -> np.ndarray:
    steps = trajectory["candidate_steps"].astype(np.float64)
    sigmas = np.clip(trajectory["sigmas"].astype(np.float64), 1e-6, 1.0 - 1e-6)
    costs = trajectory["costs"].astype(np.float64)
    current_sigma, next_sigma = sigmas[index : index + 2]
    current_logsnr = 2.0 * np.log1p(-current_sigma) - 2.0 * np.log(current_sigma)
    next_logsnr = 2.0 * np.log1p(-next_sigma) - 2.0 * np.log(next_sigma)
    return np.asarray(
        [
            steps[index] / 50.0,
            steps[index + 1] / 50.0,
            (steps[index + 1] - steps[index]) / 10.0,
            current_sigma,
            next_sigma,
            np.clip(current_logsnr, -20.0, 20.0) / 20.0,
            np.clip(next_logsnr, -20.0, 20.0) / 20.0,
            costs[index],
            costs[index + 1],
            costs[index] - costs[index + 1],
        ],
        dtype=np.float32,
    )


class AdjacentPairDataset(Dataset):
    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        signals: np.ndarray,
        quality_scale: float,
    ):
        self.rows: list[tuple[int, int]] = []
        self.trajectories = trajectories
        self.signals = signals
        self.quality_scale = float(quality_scale)
        for trajectory_index, trajectory in enumerate(trajectories):
            for pair_index in range(len(trajectory["candidate_steps"]) - 1):
                self.rows.append((trajectory_index, pair_index))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory_index, pair_index = self.rows[index]
        trajectory = self.trajectories[trajectory_index]
        quality_delta = (
            float(trajectory["qualities"][pair_index])
            - float(trajectory["qualities"][pair_index + 1])
        )
        return {
            "prompt": torch.from_numpy(trajectory["pooled_t5"]),
            "state": torch.from_numpy(self.signals[trajectory_index, pair_index]),
            "schedule": torch.from_numpy(pair_schedule(trajectory, pair_index)),
            "target": torch.tensor(
                quality_delta * self.quality_scale, dtype=torch.float32
            ),
        }


class TinyFixedGuard(nn.Module):
    def __init__(self, state_dim: int, dropout: float, use_state: bool):
        super().__init__()
        self.state_dim = int(state_dim)
        self.use_state = bool(use_state)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4096, 32), nn.LayerNorm(32), nn.SiLU()
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 16), nn.LayerNorm(16), nn.SiLU()
        )
        self.schedule_encoder = nn.Sequential(
            nn.Linear(10, 16), nn.LayerNorm(16), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self, prompt: torch.Tensor, state: torch.Tensor, schedule: torch.Tensor
    ) -> torch.Tensor:
        state_input = state if self.use_state else torch.zeros_like(state)
        fused = torch.cat(
            [
                self.prompt_encoder(prompt),
                self.state_encoder(state_input),
                self.schedule_encoder(schedule),
            ],
            dim=1,
        )
        return self.head(fused).squeeze(1)


@torch.no_grad()
def predict_pair_deltas(
    model: TinyFixedGuard,
    trajectories: list[dict[str, Any]],
    signals: np.ndarray,
    quality_scale: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    trajectory_count = len(trajectories)
    pair_count = len(trajectories[0]["candidate_steps"]) - 1
    prompts = []
    states = []
    schedules = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for pair_index in range(pair_count):
            prompts.append(trajectory["pooled_t5"])
            states.append(signals[trajectory_index, pair_index])
            schedules.append(pair_schedule(trajectory, pair_index))
    outputs = []
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        outputs.append(
            model(
                torch.from_numpy(np.stack(prompts[start:end])).to(device),
                torch.from_numpy(np.stack(states[start:end])).to(device),
                torch.from_numpy(np.stack(schedules[start:end])).to(device),
            ).cpu().numpy()
        )
    return np.concatenate(outputs).reshape(trajectory_count, pair_count) / quality_scale


def fixed_guard_choice(
    predicted_pair_deltas: np.ndarray,
    costs: np.ndarray,
    fixed_index: int,
    lambda_value: float,
    utility_margin: float,
) -> int:
    if fixed_index > 0:
        early_gain = float(predicted_pair_deltas[fixed_index - 1]) - lambda_value * (
            float(costs[fixed_index - 1]) - float(costs[fixed_index])
        )
        if early_gain > utility_margin:
            return fixed_index - 1
    if fixed_index + 1 < len(costs):
        late_gain = -float(predicted_pair_deltas[fixed_index]) - lambda_value * (
            float(costs[fixed_index + 1]) - float(costs[fixed_index])
        )
        if late_gain > utility_margin:
            return fixed_index + 1
    return fixed_index


def evaluate_policy(
    trajectories: list[dict[str, Any]],
    predictions: np.ndarray,
    lambdas: list[float],
    fixed_steps: dict[float, int],
    utility_margin: float,
    harm_epsilon: float,
    emit_rows: bool,
    model_type: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    regrets = []
    gains = []
    harms = []
    actions = []
    rows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        steps = trajectory["candidate_steps"]
        for lambda_value in lambdas:
            fixed_index = int(fixed_steps[lambda_value])
            chosen = fixed_guard_choice(
                predictions[trajectory_index],
                trajectory["costs"],
                fixed_index,
                lambda_value,
                utility_margin,
            )
            utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
            oracle_index = int(np.argmax(utility))
            oracle_utility = float(utility[oracle_index])
            realized_utility = float(utility[chosen])
            fixed_utility = float(utility[fixed_index])
            regret = max(0.0, oracle_utility - realized_utility)
            fixed_regret = max(0.0, oracle_utility - fixed_utility)
            gain = realized_utility - fixed_utility
            regrets.append(regret)
            gains.append(gain)
            harms.append(float(gain < -harm_epsilon))
            actions.append(float(chosen != fixed_index))
            if emit_rows:
                rows.append(
                    {
                        "split": trajectory["split"],
                        "model_type": model_type,
                        "Method": model_type,
                        "decision_mode": "fixed_adjacent_guard",
                        "prompt_id": trajectory["prompt_id"],
                        "seed": trajectory["seed"],
                        "lambda": lambda_value,
                        "chosen_step": int(steps[chosen]),
                        "oracle_step": int(steps[oracle_index]),
                        "best_fixed_step": int(steps[fixed_index]),
                        "policy_regret": regret,
                        "best_fixed_regret": fixed_regret,
                        "realized_utility": realized_utility,
                        "oracle_utility": oracle_utility,
                        "realized_vbench5": float(trajectory["qualities"][chosen]),
                        "realized_latency_sec": float(
                            trajectory["calibrated_latencies"][chosen]
                        ),
                        "speedup_vs_native": float(
                            trajectory["calibrated_native_latency"]
                            / trajectory["calibrated_latencies"][chosen]
                        ),
                        "normalized_cost": float(trajectory["costs"][chosen]),
                        "raw_manifest_latency_sec_diagnostic": float(
                            trajectory["latencies"][chosen]
                        ),
                        "harmful_stop": int(regret > harm_epsilon),
                        "harm_vs_fixed": int(gain < -harm_epsilon),
                        "action_changed": int(chosen != fixed_index),
                        "utility_gain_vs_fixed": gain,
                        "utility_margin": utility_margin,
                    }
                )
    return {
        "macro_policy_regret": float(np.mean(regrets)),
        "mean_utility_gain_vs_fixed": float(np.mean(gains)),
        "harm_vs_fixed_rate": float(np.mean(harms)),
        "action_rate": float(np.mean(actions)),
    }, rows


def choose_margin(
    trajectories: list[dict[str, Any]],
    predictions: np.ndarray,
    lambdas: list[float],
    fixed_steps: dict[float, int],
    margins: list[float],
    harm_epsilon: float,
    max_harm_rate: float,
    min_action_rate: float,
    model_type: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    eligible = []
    for margin in margins:
        metrics, _ = evaluate_policy(
            trajectories,
            predictions,
            lambdas,
            fixed_steps,
            margin,
            harm_epsilon,
            False,
            model_type,
        )
        row = {"model_type": model_type, "utility_margin": margin, **metrics}
        rows.append(row)
        if (
            metrics["harm_vs_fixed_rate"] <= max_harm_rate
            and metrics["action_rate"] >= min_action_rate
            and metrics["mean_utility_gain_vs_fixed"] > 0
        ):
            eligible.append(row)
    if not eligible:
        return float("inf"), rows
    best = max(
        eligible,
        key=lambda row: (
            row["mean_utility_gain_vs_fixed"],
            -row["harm_vs_fixed_rate"],
            -row["utility_margin"],
        ),
    )
    return float(best["utility_margin"]), rows


def train_model(
    model_type: str,
    fit_dataset: AdjacentPairDataset,
    calibration_dataset: AdjacentPairDataset,
    state_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TinyFixedGuard, int, list[dict[str, Any]]]:
    model = TinyFixedGuard(
        state_dim=state_dim,
        dropout=args.dropout,
        use_state=model_type == "fixed_guard_state",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    loader = DataLoader(
        fit_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["prompt"].to(device),
                batch["state"].to(device),
                batch["schedule"].to(device),
            )
            loss = F.smooth_l1_loss(prediction, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        calibration_losses = []
        with torch.no_grad():
            for batch in calibration_loader:
                prediction = model(
                    batch["prompt"].to(device),
                    batch["state"].to(device),
                    batch["schedule"].to(device),
                )
                calibration_losses.append(
                    float(
                        F.smooth_l1_loss(
                            prediction, batch["target"].to(device)
                        ).detach()
                    )
                )
        calibration_loss = float(np.mean(calibration_losses))
        selected = calibration_loss < best_loss
        if selected:
            best_loss = calibration_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_pairwise_smooth_l1": float(np.mean(train_losses)),
                "calibration_pairwise_smooth_l1": calibration_loss,
                "selected_as_best": selected,
            }
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
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    manifest = base.load_dataset_manifest(dataset_dir)
    all_names = [str(value) for value in manifest["feature_names"]]
    missing = [name for name in args.state_features if name not in all_names]
    if missing:
        raise ValueError(f"State features are missing: {missing}")
    indices = np.asarray([all_names.index(name) for name in args.state_features])
    train_trajectories = base.load_trajectories(dataset_dir, manifest, "train", indices)
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", indices
    )
    source_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    costs, seconds, native_seconds, latency_provenance = base.load_locked_latency_profile(
        manifest, source_steps, args.expected_latency_profile_sha256
    )
    for trajectory in [*train_trajectories, *validation_trajectories]:
        trajectory["candidate_steps"] = source_steps.copy()
    base.apply_locked_latency_profile(
        train_trajectories, costs, seconds, native_seconds
    )
    base.apply_locked_latency_profile(
        validation_trajectories, costs, seconds, native_seconds
    )
    fixed_steps = base.best_fixed_steps(train_trajectories, args.eval_lambdas)
    fit_trajectories, calibration_trajectories = split_fit_calibration(
        train_trajectories, args.calibration_fraction, args.split_salt
    )
    signal_mean, signal_std = fit_signal_normalizer(fit_trajectories)
    fit_signals = normalized_signals(fit_trajectories, signal_mean, signal_std)
    calibration_signals = normalized_signals(
        calibration_trajectories, signal_mean, signal_std
    )
    validation_signals = normalized_signals(
        validation_trajectories, signal_mean, signal_std
    )
    fit_dataset = AdjacentPairDataset(fit_trajectories, fit_signals, args.quality_scale)
    calibration_dataset = AdjacentPairDataset(
        calibration_trajectories, calibration_signals, args.quality_scale
    )
    device = torch.device(args.device)
    prediction_rows = []
    checkpoint_artifacts = {}
    history_artifacts = {}
    calibration_rows = []
    selected_margins = {}
    best_epochs = {}
    for model_type in MODEL_TYPES:
        base.seed_everything(args.seed)
        model, best_epoch, history = train_model(
            model_type,
            fit_dataset,
            calibration_dataset,
            fit_signals.shape[2],
            args,
            device,
        )
        calibration_prediction = predict_pair_deltas(
            model,
            calibration_trajectories,
            calibration_signals,
            args.quality_scale,
            device,
            args.inference_batch_size,
        )
        selected_margin, model_calibration_rows = choose_margin(
            calibration_trajectories,
            calibration_prediction,
            args.eval_lambdas,
            fixed_steps,
            args.utility_margins,
            args.harm_epsilon,
            args.max_calibration_harm_rate,
            args.min_calibration_action_rate,
            model_type,
        )
        calibration_rows.extend(model_calibration_rows)
        selected_margins[model_type] = selected_margin
        best_epochs[model_type] = best_epoch
        validation_prediction = predict_pair_deltas(
            model,
            validation_trajectories,
            validation_signals,
            args.quality_scale,
            device,
            args.inference_batch_size,
        )
        effective_margin = selected_margin if math.isfinite(selected_margin) else 1e9
        _, rows = evaluate_policy(
            validation_trajectories,
            validation_prediction,
            args.eval_lambdas,
            fixed_steps,
            effective_margin,
            args.harm_epsilon,
            True,
            model_type,
        )
        prediction_rows.extend(rows)
        checkpoint_path = out_dir / f"{model_type}_router.pt"
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "state_features": args.state_features,
                "signal_mean": signal_mean,
                "signal_std": signal_std,
                "candidate_steps": source_steps,
                "fixed_steps": fixed_steps,
                "utility_margin": effective_margin,
                "best_epoch": best_epoch,
                "quality_scale": args.quality_scale,
                "latency_profile": latency_provenance,
            },
            checkpoint_path,
        )
        history_path = out_dir / f"{model_type}_training_history.csv"
        write_csv(history_path, history)
        checkpoint_artifacts[model_type] = {
            "path": checkpoint_path.name,
            "sha256": base.sha256_file(checkpoint_path),
        }
        history_artifacts[model_type] = {
            "path": history_path.name,
            "sha256": base.sha256_file(history_path),
        }
    predictions_path = out_dir / "validation_predictions.csv"
    calibration_path = out_dir / "calibration_margin_sweep.csv"
    write_csv(predictions_path, prediction_rows)
    write_csv(calibration_path, calibration_rows)
    manifest_path = dataset_dir / "dataset_manifest.json"
    summary = {
        "schema": RUN_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "test_accessed": False,
        "train_seed": args.seed,
        "train_lambdas": args.eval_lambdas,
        "eval_lambdas": args.eval_lambdas,
        "primary_lambda": args.primary_lambda,
        "harm_epsilon": args.harm_epsilon,
        "decision_parameter": "train_calibrated_model_specific_utility_margin",
        "risk_margin": 0.0,
        "selected_margins": {
            key: value if math.isfinite(value) else "fallback_fixed"
            for key, value in selected_margins.items()
        },
        "feature_groups": ["fixed_guard_sparse_state"],
        "selected_feature_count": len(args.state_features),
        "selected_feature_names": args.state_features,
        "dataset_manifest_sha256": base.sha256_file(manifest_path),
        "source_candidate_steps": source_steps.tolist(),
        "candidate_steps": source_steps.tolist(),
        "model_types": list(MODEL_TYPES),
        "best_epochs": best_epochs,
        "fixed_steps": {str(key): int(value) for key, value in fixed_steps.items()},
        "cost_profile": costs.tolist(),
        "train_prompts": len(fit_trajectories),
        "calibration_prompts": len(calibration_trajectories),
        "validation_prompts": len({item["prompt_id"] for item in validation_trajectories}),
        "training": {
            "objective": "adjacent_pair_quality_delta_smooth_l1",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "quality_scale": args.quality_scale,
            "calibration_split": {
                "fraction": args.calibration_fraction,
                "salt": args.split_salt,
            },
        },
        "latency_profile": latency_provenance,
        "artifacts": {
            "predictions": predictions_path.name,
            "predictions_sha256": base.sha256_file(predictions_path),
            "calibration_margin_sweep": calibration_path.name,
            "calibration_margin_sweep_sha256": base.sha256_file(calibration_path),
            "checkpoints": checkpoint_artifacts,
            "training_histories": history_artifacts,
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
