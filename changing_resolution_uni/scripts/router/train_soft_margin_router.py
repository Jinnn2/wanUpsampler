#!/usr/bin/env python3
"""Train B4-anchored causal soft suffix-margin routers across utility lambdas."""

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
    from . import train_variable_lambda_router as base
except ImportError:
    import train_variable_lambda_router as base


CHECKPOINT_SCHEMA = "variable_lambda_soft_margin_router_checkpoint_v1"
RUN_SCHEMA = "variable_lambda_soft_margin_selection_run_v1"
MODEL_LABELS = {
    "b4_offline": "Variable-Lambda Offline B4 Router",
    "soft_margin_control": "B4 Soft-Margin Schedule Control",
    "soft_margin_state": "B4 Causal Soft-Margin State Router",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model-type",
        choices=[
            "b4_offline",
            "soft_margin_control",
            "soft_margin_state",
            "soft_margin_pair",
        ],
        default="soft_margin_pair",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=[
            "x0_global",
            "residual_global",
            "x0_channel",
            "residual_channel",
            "local_energy",
            "trajectory_delta",
        ],
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
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument(
        "--risk-margin",
        type=float,
        default=0.0,
        help="Stop when the corrected suffix-best margin reaches this value.",
    )
    parser.add_argument("--margin-temperature", type=float, default=0.02)
    parser.add_argument("--b4-temperature", type=float, default=0.02)
    parser.add_argument("--b4-emd-weight", type=float, default=0.5)
    parser.add_argument("--residual-logit-limit", type=float, default=4.0)
    parser.add_argument("--residual-penalty-weight", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-trajectories", type=int, default=64)
    parser.add_argument(
        "--selection-split",
        choices=["validation", "train"],
        default="validation",
        help="Use train only for an explicitly isolated overfit sanity run.",
    )
    parser.add_argument(
        "--max-train-trajectories",
        type=int,
        default=None,
        help="Deterministically truncate train trajectories for a sanity run.",
    )
    parser.add_argument("--expected-latency-profile-sha256", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.train_lambdas = sorted(set(float(value) for value in args.train_lambdas))
    args.eval_lambdas = sorted(set(float(value) for value in args.eval_lambdas))
    for name in ("train_lambdas", "eval_lambdas"):
        values = getattr(args, name)
        if not values or any(value < 0 or not math.isfinite(value) for value in values):
            parser.error(f"{name.replace('_', '-')} must contain finite values >= 0")
    if args.primary_lambda not in args.eval_lambdas:
        parser.error("primary-lambda must be present in eval-lambdas")
    positive_values = {
        "margin-temperature": args.margin_temperature,
        "b4-temperature": args.b4_temperature,
        "residual-logit-limit": args.residual_logit_limit,
        "hidden-dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "eval-batch-trajectories": args.eval_batch_trajectories,
    }
    if any(
        float(value) <= 0 or not math.isfinite(float(value))
        for value in positive_values.values()
    ):
        parser.error(f"These arguments must be finite and positive: {positive_values}")
    if (
        args.harm_epsilon < 0
        or args.b4_emd_weight < 0
        or args.residual_penalty_weight < 0
        or args.weight_decay < 0
        or not 0 <= args.dropout < 1
        or args.num_workers < 0
    ):
        parser.error("Invalid non-negative loss, optimizer, dropout, or worker value")
    if not math.isfinite(args.risk_margin):
        parser.error("risk-margin must be finite")
    if args.max_train_trajectories is not None and args.max_train_trajectories < 1:
        parser.error("max-train-trajectories must be positive")
    if args.selection_split == "train" and args.max_train_trajectories is None:
        parser.error("train selection is reserved for bounded sanity runs")
    return args


def fit_step_state_normalizer(
    trajectories: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([trajectory["features"] for trajectory in trajectories])
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def suffix_best_margin(utilities: np.ndarray, temperature: float) -> np.ndarray:
    """Return (current - strictly-future best) / temperature for each state."""
    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("suffix_best_margin requires a vector with at least two items")
    result = np.full(values.shape, 30.0, dtype=np.float64)
    future_best = np.maximum.accumulate(values[:0:-1])[::-1]
    result[:-1] = (values[:-1] - future_best) / float(temperature)
    return result.astype(np.float32)


def soft_margin_targets(
    qualities: np.ndarray,
    costs: np.ndarray,
    lambda_value: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    utilities = qualities - float(lambda_value) * costs
    target_margin = suffix_best_margin(utilities, temperature)
    target_probability = np.ones_like(target_margin, dtype=np.float32)
    target_probability[:-1] = 1.0 / (
        1.0 + np.exp(-np.clip(target_margin[:-1], -30.0, 30.0))
    )
    reach_weight = np.zeros_like(target_probability, dtype=np.float32)
    survival = 1.0
    for index in range(len(target_probability) - 1):
        reach_weight[index] = survival
        survival *= 1.0 - float(target_probability[index])
    return target_margin, target_probability, reach_weight


class SoftMarginTrajectoryDataset(Dataset):
    """One example is a complete trajectory and one utility lambda."""

    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        lambdas: list[float],
        candidate_steps: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        cost_profile: np.ndarray,
        margin_temperature: float,
    ):
        self.trajectories = trajectories
        self.lambdas = lambdas
        self.candidate_steps = candidate_steps
        self.state_mean = state_mean
        self.state_std = state_std
        self.cost_profile = cost_profile
        self.margin_temperature = margin_temperature

    def __len__(self) -> int:
        return len(self.trajectories) * len(self.lambdas)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        lambda_index = index % len(self.lambdas)
        trajectory_index = index // len(self.lambdas)
        trajectory = self.trajectories[trajectory_index]
        lambda_value = self.lambdas[lambda_index]
        target_margin, target_probability, reach_weight = soft_margin_targets(
            trajectory["qualities"],
            trajectory["costs"],
            lambda_value,
            self.margin_temperature,
        )
        state = (trajectory["features"] - self.state_mean) / self.state_std
        schedule = base.schedule_features(
            self.candidate_steps,
            trajectory["sigmas"],
            lambda_value,
            self.cost_profile,
        )
        return {
            "pooled_t5": torch.from_numpy(trajectory["pooled_t5"]),
            "state": torch.from_numpy(state.astype(np.float32)),
            "schedule": torch.from_numpy(schedule),
            "lambda_value": torch.tensor(lambda_value, dtype=torch.float32),
            "target_margin": torch.from_numpy(target_margin),
            "soft_stop_target": torch.from_numpy(target_probability),
            "reach_weight": torch.from_numpy(reach_weight),
        }


class CausalSoftMarginRouter(nn.Module):
    """Prompt-free causal residual on the frozen B4 suffix-best logit margin."""

    def __init__(
        self,
        b4_prior: base.VariableLambdaB4Prior,
        state_dim: int,
        dropout: float,
        hidden_dim: int,
        residual_logit_limit: float,
        use_state: bool,
    ):
        super().__init__()
        self.b4_prior = copy.deepcopy(b4_prior)
        for parameter in self.b4_prior.parameters():
            parameter.requires_grad_(False)
        self.use_state = bool(use_state)
        self.residual_logit_limit = float(residual_logit_limit)
        self.schedule_encoder = nn.Sequential(
            nn.Linear(len(base.SCHEDULE_FEATURE_NAMES), 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        # Keep the control and state architectures identical. The control feeds
        # zeros through this encoder so paired differences isolate real state.
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        input_dim = 64 + 32 + 2
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def train(self, mode: bool = True) -> "CausalSoftMarginRouter":
        super().train(mode)
        self.b4_prior.eval()
        return self

    @staticmethod
    def offline_suffix_margin(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("B4 logits must have shape [batch, candidate>=2]")
        reverse = torch.flip(logits[:, 1:], dims=[1])
        reverse_best = torch.cummax(reverse, dim=1).values
        future_best = torch.flip(reverse_best, dims=[1])
        margin = logits.new_full(logits.shape, 30.0)
        margin[:, :-1] = logits[:, :-1] - future_best
        return margin

    def forward(
        self,
        pooled_t5: torch.Tensor,
        state: torch.Tensor,
        schedule: torch.Tensor,
        lambda_value: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.b4_prior.eval()
        with torch.no_grad():
            prior = self.b4_prior(pooled_t5, lambda_value)
        offline_margin = self.offline_suffix_margin(prior["logits"])
        probabilities = prior["discrete_probs"].clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=1)
        entropy = entropy / math.log(probabilities.shape[1])
        prior_context = torch.stack(
            [
                offline_margin.clamp(-10.0, 10.0) / 10.0,
                entropy[:, None].expand_as(offline_margin),
            ],
            dim=2,
        )
        state_input = state if self.use_state else torch.zeros_like(state)
        parts = [
            self.state_encoder(state_input),
            self.schedule_encoder(schedule),
            prior_context,
        ]
        hidden, _ = self.gru(torch.cat(parts, dim=2))
        residual = self.residual_logit_limit * torch.tanh(
            self.residual_head(hidden).squeeze(-1)
        )
        residual = residual.clone()
        residual[:, -1] = 0.0
        online_margin = offline_margin + residual
        return {
            "offline_margin": offline_margin,
            "online_margin": online_margin,
            "residual_logit": residual,
            "b4_probabilities": probabilities,
        }


def requested_model_types(request: str) -> list[str]:
    if request == "soft_margin_pair":
        return ["b4_offline", "soft_margin_control", "soft_margin_state"]
    return [request]


def first_margin_stop(margins: np.ndarray, risk_margin: float) -> np.ndarray:
    eligible = margins >= float(risk_margin)
    eligible[:, -1] = True
    return eligible.argmax(axis=1).astype(np.int64)


@torch.no_grad()
def evaluate_soft_margin_model(
    model: CausalSoftMarginRouter,
    model_type: str,
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    candidate_steps: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cost_profile: np.ndarray,
    risk_margin: float,
    harm_epsilon: float,
    fixed_steps: dict[float, int],
    quality_dimensions: list[str],
    device: torch.device,
    eval_batch_trajectories: int,
    emit_rows: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    regrets: list[float] = []
    harms: list[float] = []
    for lambda_value in lambdas:
        fixed_index = fixed_steps[lambda_value]
        for start in range(0, len(trajectories), eval_batch_trajectories):
            chunk = trajectories[start : start + eval_batch_trajectories]
            state = np.stack(
                [
                    (trajectory["features"] - state_mean) / state_std
                    for trajectory in chunk
                ]
            ).astype(np.float32)
            schedule = np.stack(
                [
                    base.schedule_features(
                        candidate_steps,
                        trajectory["sigmas"],
                        lambda_value,
                        cost_profile,
                    )
                    for trajectory in chunk
                ]
            )
            output = model(
                torch.from_numpy(
                    np.stack([trajectory["pooled_t5"] for trajectory in chunk])
                ).to(device),
                torch.from_numpy(state).to(device),
                torch.from_numpy(schedule).to(device),
                torch.full(
                    (len(chunk),), lambda_value, dtype=torch.float32, device=device
                ),
            )
            online_margin = output["online_margin"].cpu().numpy()
            residual = output["residual_logit"].cpu().numpy()
            chosen_indices = first_margin_stop(online_margin, risk_margin)
            for row_index, trajectory in enumerate(chunk):
                chosen = int(chosen_indices[row_index])
                utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
                oracle_index = int(np.argmax(utility))
                oracle_utility = float(utility[oracle_index])
                realized_utility = float(utility[chosen])
                regret = max(0.0, oracle_utility - realized_utility)
                fixed_regret = max(0.0, oracle_utility - float(utility[fixed_index]))
                regrets.append(regret)
                harms.append(float(regret > harm_epsilon))
                if emit_rows:
                    row = {
                        "split": trajectory["split"],
                        "model_type": model_type,
                        "Method": MODEL_LABELS[model_type],
                        "decision_mode": "causal_soft_suffix_margin",
                        "prompt_id": trajectory["prompt_id"],
                        "seed": trajectory["seed"],
                        "lambda": lambda_value,
                        "chosen_step": int(candidate_steps[chosen]),
                        "oracle_step": int(candidate_steps[oracle_index]),
                        "best_fixed_step": int(candidate_steps[fixed_index]),
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
                        "raw_manifest_speedup_diagnostic": float(
                            trajectory["native_latency"]
                            / trajectory["latencies"][chosen]
                        ),
                        "harmful_stop": int(regret > harm_epsilon),
                        "predicted_stop_regret": "",
                        "predicted_harm_probability": float(
                            torch.sigmoid(
                                torch.tensor(online_margin[row_index, chosen])
                            )
                        ),
                        "offline_suffix_margin": float(
                            output["offline_margin"][row_index, chosen].cpu()
                        ),
                        "online_suffix_margin": float(online_margin[row_index, chosen]),
                        "state_residual_logit": float(residual[row_index, chosen]),
                    }
                    for dimension_index, dimension in enumerate(quality_dimensions):
                        row[f"realized_{dimension}"] = float(
                            trajectory["dimensions"][chosen, dimension_index]
                        )
                    rows.append(row)
    return {
        "macro_policy_regret": float(np.mean(regrets)),
        "macro_harmful_stop_rate": float(np.mean(harms)),
    }, rows


def train_soft_margin_model(
    model_type: str,
    b4_prior: base.VariableLambdaB4Prior,
    train_loader: DataLoader,
    selection_trajectories: list[dict[str, Any]],
    eval_lambdas: list[float],
    candidate_steps: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cost_profile: np.ndarray,
    fixed_steps: dict[float, int],
    quality_dimensions: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CausalSoftMarginRouter, int, dict[str, float], list[dict[str, Any]]]:
    model = CausalSoftMarginRouter(
        b4_prior=b4_prior,
        state_dim=state_mean.shape[1],
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        residual_logit_limit=args.residual_logit_limit,
        use_state=model_type == "soft_margin_state",
    ).to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    best_metrics, _ = evaluate_soft_margin_model(
        model,
        model_type,
        selection_trajectories,
        eval_lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        args.risk_margin,
        args.harm_epsilon,
        fixed_steps,
        quality_dimensions,
        device,
        args.eval_batch_trajectories,
        emit_rows=False,
    )
    best_regret = best_metrics["macro_policy_regret"]
    best_epoch = 0
    best_weights = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_total_loss": "",
            "train_soft_margin_loss": "",
            "train_soft_margin_excess": "",
            "train_residual_penalty": "",
            "validation_macro_policy_regret": best_metrics["macro_policy_regret"],
            "validation_macro_harmful_stop_rate": best_metrics[
                "macro_harmful_stop_rate"
            ],
            "selected_as_best": True,
            "checkpoint_role": "exact_b4_argmax_before_online_training",
        }
    ]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        margin_losses: list[float] = []
        margin_excesses: list[float] = []
        penalties: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["pooled_t5"].to(device),
                batch["state"].to(device),
                batch["schedule"].to(device),
                batch["lambda_value"].to(device),
            )
            weights = batch["reach_weight"].to(device)[:, :-1]
            element_loss = F.binary_cross_entropy_with_logits(
                output["online_margin"][:, :-1],
                batch["soft_stop_target"].to(device)[:, :-1],
                reduction="none",
            )
            denominator = weights.sum().clamp_min(1e-8)
            margin_loss = (element_loss * weights).sum() / denominator
            target = batch["soft_stop_target"].to(device)[:, :-1]
            target_entropy = F.binary_cross_entropy(target, target, reduction="none")
            margin_excess = (
                (element_loss - target_entropy) * weights
            ).sum() / denominator
            residual_penalty = (
                output["residual_logit"][:, :-1].square() * weights
            ).sum() / denominator
            loss = margin_loss + args.residual_penalty_weight * residual_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {model_type}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            margin_losses.append(float(margin_loss.detach()))
            margin_excesses.append(float(margin_excess.detach()))
            penalties.append(float(residual_penalty.detach()))
        scheduler.step()
        metrics, _ = evaluate_soft_margin_model(
            model,
            model_type,
            selection_trajectories,
            eval_lambdas,
            candidate_steps,
            state_mean,
            state_std,
            cost_profile,
            args.risk_margin,
            args.harm_epsilon,
            fixed_steps,
            quality_dimensions,
            device,
            args.eval_batch_trajectories,
            emit_rows=False,
        )
        selected_as_best = metrics["macro_policy_regret"] < best_regret
        if selected_as_best:
            best_regret = metrics["macro_policy_regret"]
            best_metrics = metrics
            best_epoch = epoch
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": float(np.mean(losses)),
                "train_soft_margin_loss": float(np.mean(margin_losses)),
                "train_soft_margin_excess": float(np.mean(margin_excesses)),
                "train_residual_penalty": float(np.mean(penalties)),
                "validation_macro_policy_regret": metrics["macro_policy_regret"],
                "validation_macro_harmful_stop_rate": metrics[
                    "macro_harmful_stop_rate"
                ],
                "selected_as_best": selected_as_best,
                "checkpoint_role": "trained_causal_soft_margin_residual",
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"{model_type} epoch={epoch:03d}/{args.epochs} "
                f"loss={np.mean(losses):.6f} margin={np.mean(margin_losses):.6f} "
                f"excess={np.mean(margin_excesses):.6f} "
                f"residual={np.mean(penalties):.6f} "
                f"selection_regret={metrics['macro_policy_regret']:.6f}"
            )
    model.load_state_dict(best_weights)
    final_metrics, _ = evaluate_soft_margin_model(
        model,
        model_type,
        selection_trajectories,
        eval_lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        args.risk_margin,
        args.harm_epsilon,
        fixed_steps,
        quality_dimensions,
        device,
        args.eval_batch_trajectories,
        emit_rows=False,
    )
    if not math.isclose(
        final_metrics["macro_policy_regret"], best_regret, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"Best-checkpoint evaluation changed for {model_type}")
    return model, best_epoch, best_metrics, history


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
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
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    manifest = base.load_dataset_manifest(dataset_dir)
    selected_indices, selected_names = base.select_feature_indices(
        manifest, args.feature_groups
    )
    train_trajectories = base.load_trajectories(
        dataset_dir, manifest, "train", selected_indices
    )
    validation_trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", selected_indices
    )
    if args.max_train_trajectories is not None:
        train_trajectories = train_trajectories[: args.max_train_trajectories]
    train_prompts = {trajectory["prompt_id"] for trajectory in train_trajectories}
    validation_prompts = {
        trajectory["prompt_id"] for trajectory in validation_trajectories
    }
    if train_prompts & validation_prompts:
        raise ValueError("Train and validation prompts overlap")
    candidate_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    state_mean, state_std = fit_step_state_normalizer(train_trajectories)
    (
        cost_profile,
        calibrated_candidate_seconds,
        calibrated_native_seconds,
        latency_profile_provenance,
    ) = base.load_locked_latency_profile(
        manifest, candidate_steps, args.expected_latency_profile_sha256
    )
    base.apply_locked_latency_profile(
        train_trajectories,
        cost_profile,
        calibrated_candidate_seconds,
        calibrated_native_seconds,
    )
    base.apply_locked_latency_profile(
        validation_trajectories,
        cost_profile,
        calibrated_candidate_seconds,
        calibrated_native_seconds,
    )
    selection_trajectories = (
        validation_trajectories
        if args.selection_split == "validation"
        else train_trajectories
    )
    fixed_steps = base.best_fixed_steps(train_trajectories, args.eval_lambdas)
    train_dataset = SoftMarginTrajectoryDataset(
        train_trajectories,
        args.train_lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        args.margin_temperature,
    )
    device = torch.device(args.device)
    b4_dataset = base.B4LambdaDataset(
        train_trajectories, args.train_lambdas, args.b4_temperature
    )
    b4_loader = DataLoader(
        b4_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    b4_prior, b4_best_epoch, b4_metrics, b4_history = base.train_b4_model(
        b4_loader,
        selection_trajectories,
        args.eval_lambdas,
        candidate_steps,
        fixed_steps,
        manifest["quality_dimensions"],
        args,
        device,
    )
    model_types = requested_model_types(args.model_type)
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    checkpoint_artifacts: dict[str, dict[str, str]] = {}
    history_artifacts: dict[str, dict[str, str]] = {}
    for model_type in model_types:
        if model_type == "b4_offline":
            model: nn.Module = b4_prior
            best_epoch = b4_best_epoch
            metrics = b4_metrics
            history = b4_history
            _, rows = base.evaluate_b4_offline_model(
                b4_prior,
                selection_trajectories,
                args.eval_lambdas,
                candidate_steps,
                fixed_steps,
                args.harm_epsilon,
                manifest["quality_dimensions"],
                device,
                args.eval_batch_trajectories,
                emit_rows=True,
            )
            for row in rows:
                row["offline_suffix_margin"] = ""
                row["online_suffix_margin"] = ""
                row["state_residual_logit"] = ""
        else:
            base.seed_everything(args.seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(args.seed),
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            model, best_epoch, metrics, history = train_soft_margin_model(
                model_type,
                b4_prior,
                train_loader,
                selection_trajectories,
                args.eval_lambdas,
                candidate_steps,
                state_mean,
                state_std,
                cost_profile,
                fixed_steps,
                manifest["quality_dimensions"],
                args,
                device,
            )
            _, rows = evaluate_soft_margin_model(
                model,
                model_type,
                selection_trajectories,
                args.eval_lambdas,
                candidate_steps,
                state_mean,
                state_std,
                cost_profile,
                args.risk_margin,
                args.harm_epsilon,
                fixed_steps,
                manifest["quality_dimensions"],
                device,
                args.eval_batch_trajectories,
                emit_rows=True,
            )
        prediction_rows.extend(rows)
        summary_rows.append(
            {
                "model_type": model_type,
                "Method": MODEL_LABELS[model_type],
                "best_epoch": best_epoch,
                "minimum_train_soft_margin_excess": min(
                    (
                        float(row["train_soft_margin_excess"])
                        for row in history
                        if row.get("train_soft_margin_excess") not in {None, ""}
                    ),
                    default="",
                ),
                **metrics,
            }
        )
        history_path = out_dir / f"{model_type}_training_history.csv"
        write_csv(history_path, history)
        history_artifacts[model_type] = {
            "path": history_path.name,
            "sha256": base.sha256_file(history_path),
        }
        checkpoint_path = out_dir / f"{model_type}_router.pt"
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "evaluation_protocol": base.EVALUATION_PROTOCOL,
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "state_dim": state_mean.shape[1],
                "state_normalization": "train_per_candidate_step_v1",
                "state_mean": state_mean,
                "state_std": state_std,
                "selected_feature_names": selected_names,
                "schedule_feature_names": base.SCHEDULE_FEATURE_NAMES,
                "candidate_steps": candidate_steps,
                "cost_profile": cost_profile,
                "risk_margin": args.risk_margin,
                "margin_temperature": args.margin_temperature,
                "residual_logit_limit": args.residual_logit_limit,
                "hidden_dim": args.hidden_dim,
                "b4_prior_frozen": model_type != "b4_offline",
                "best_epoch": best_epoch,
                "selection_metrics": metrics,
            },
            checkpoint_path,
        )
        checkpoint_artifacts[model_type] = {
            "path": checkpoint_path.name,
            "sha256": base.sha256_file(checkpoint_path),
        }
    write_csv(out_dir / "selection_predictions.csv", prediction_rows)
    write_csv(out_dir / "selection_model_summary.csv", summary_rows)
    selected_model = min(summary_rows, key=lambda row: row["macro_policy_regret"])
    evaluation_split = (
        "validation" if args.selection_split == "validation" else "train_sanity"
    )
    run_summary = {
        "schema": RUN_SCHEMA,
        "evaluation_protocol": base.EVALUATION_PROTOCOL,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_split": evaluation_split,
        "test_accessed": False,
        "train_seed": args.seed,
        "model_type_request": args.model_type,
        "model_types": model_types,
        "selected_model_type": selected_model["model_type"],
        "selection_rule": "minimum selection-split macro policy regret",
        "train_lambdas": args.train_lambdas,
        "eval_lambdas": args.eval_lambdas,
        "primary_lambda": args.primary_lambda,
        "harm_epsilon": args.harm_epsilon,
        "decision_parameter": "risk_margin",
        "risk_margin": args.risk_margin,
        "feature_groups": args.feature_groups,
        "selected_feature_count": len(selected_names),
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": base.sha256_file(
            dataset_dir / "dataset_manifest.json"
        ),
        "training": {
            "epochs": args.epochs,
            "batch_size_trajectories": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "margin_temperature": args.margin_temperature,
            "b4_temperature": args.b4_temperature,
            "b4_emd_weight": args.b4_emd_weight,
            "residual_logit_limit": args.residual_logit_limit,
            "residual_penalty_weight": args.residual_penalty_weight,
            "hidden_dim": args.hidden_dim,
            "state_normalization": "train_per_candidate_step_v1",
            "trajectory_training": True,
            "max_train_trajectories": args.max_train_trajectories,
        },
        "train_prompts": len(train_prompts),
        "validation_prompts": len(validation_prompts),
        "train_trajectories": len(train_trajectories),
        "validation_trajectories": len(validation_trajectories),
        "cost_profile": cost_profile.tolist(),
        "calibrated_candidate_latency_seconds": calibrated_candidate_seconds.tolist(),
        "calibrated_native_latency_seconds": calibrated_native_seconds,
        "latency_profile": latency_profile_provenance,
        "models": summary_rows,
        "artifacts": {
            "predictions": "selection_predictions.csv",
            "model_summary": "selection_model_summary.csv",
            "checkpoints": checkpoint_artifacts,
            "training_histories": history_artifacts,
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
