#!/usr/bin/env python3
"""Train a prompt-only continuous budget regressor on the legacy oracle data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (  # noqa: E402
    QUALITY5_DIMENSIONS,
)
from changing_resolution_uni.scripts.router.dataset_router import (  # noqa: E402
    get_dataloaders,
    sha256_file,
)
from changing_resolution_uni.scripts.router.model_router import (  # noqa: E402
    SoftDistillationMLPRouter,
)


LOGGER = logging.getLogger("continuous_budget_prior")
METHODS = (
    ("Prompt Oracle (Upper Bound)", "prompt_oracle", "prompt_oracle"),
    ("Fixed Budget (Train-Selected)", "best_fixed", "best_fixed"),
    ("B4 Argmax (Frozen)", "learned", "b4_argmax"),
    (
        "B4 Expected Budget -> Nearest (Frozen)",
        "learned",
        "b4_projected_nearest",
    ),
    (
        "Continuous Prompt Budget -> Nearest",
        "learned",
        "continuous_budget_nearest",
    ),
)


class ContinuousBudgetRegressor(nn.Module):
    """Map one frozen pooled prompt embedding to a normalized cost in [0, 1]."""

    def __init__(
        self,
        in_dim: int = 4096,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        previous = in_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 1)

    def forward(self, pooled_t5: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.backbone(pooled_t5))).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--b4-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--target-type",
        choices=("soft_expected", "hard_oracle"),
        default="soft_expected",
    )
    parser.add_argument("--primary-lambda", type=float, default=0.08)
    parser.add_argument("--soft-target-tau", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.02)
    parser.add_argument("--allow-estimated-latency", action="store_true")
    parser.add_argument("--require-measured-latency", action="store_true")
    parser.add_argument(
        "--require-b4-temperature-match",
        action="store_true",
        help=(
            "Fail when the frozen B4 checkpoint has a missing or different "
            "soft-target temperature. This is optional because hard_oracle "
            "budget labels do not use that temperature."
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if args.lr <= 0 or args.weight_decay < 0 or args.huber_beta <= 0:
        parser.error(
            "lr and huber-beta must be positive; weight-decay cannot be negative"
        )
    if args.soft_target_tau <= 0:
        parser.error("soft-target-tau must be positive")
    if args.allow_estimated_latency and args.require_measured_latency:
        parser.error(
            "allow-estimated-latency and require-measured-latency are mutually exclusive"
        )
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibrate_budget_grid(
    loader: torch.utils.data.DataLoader,
) -> torch.Tensor:
    """Use train-only median candidate/native latency ratios as budget coordinates."""
    ratios = []
    for batch in loader:
        native = batch["native_latency"].float().unsqueeze(1).clamp(min=1e-6)
        ratios.append(batch["latencies"].float() / native)
    if not ratios:
        raise ValueError("Cannot calibrate a budget grid from an empty loader")
    grid = torch.cat(ratios, dim=0).median(dim=0).values
    if not torch.isfinite(grid).all():
        raise ValueError("Train-calibrated budget grid contains non-finite values")
    if torch.any(grid <= 0) or torch.any(grid > 1):
        raise ValueError(
            f"Normalized candidate costs must lie in (0, 1]; got {grid.tolist()}"
        )
    if torch.unique(grid).numel() != grid.numel():
        raise ValueError(f"Budget coordinates must be unique; got {grid.tolist()}")
    return grid.float()


def budget_targets(
    batch: dict[str, Any],
    budget_grid: torch.Tensor,
    target_type: str,
) -> torch.Tensor:
    grid = budget_grid.to(batch["utilities"].device)
    if target_type == "hard_oracle":
        return grid[batch["target_step_idx"]]
    if target_type == "soft_expected":
        return (batch["soft_utility_target"] * grid.unsqueeze(0)).sum(dim=1)
    raise ValueError(f"Unsupported target type: {target_type}")


def nearest_budget_index(
    predicted_budget: torch.Tensor, budget_grid: torch.Tensor
) -> torch.Tensor:
    if predicted_budget.ndim != 1 or budget_grid.ndim != 1:
        raise ValueError(
            "predicted_budget and budget_grid must both be one-dimensional"
        )
    distances = (predicted_budget.unsqueeze(1) - budget_grid.unsqueeze(0)).abs()
    return distances.argmin(dim=1)


def train_selected_fixed_index(loader: torch.utils.data.DataLoader) -> int:
    utility_sum: torch.Tensor | None = None
    count = 0
    for batch in loader:
        values = batch["utilities"].float().sum(dim=0)
        utility_sum = values if utility_sum is None else utility_sum + values
        count += int(batch["utilities"].shape[0])
    if utility_sum is None or count == 0:
        raise ValueError("Cannot select a fixed budget from an empty train loader")
    return int(torch.argmax(utility_sum / count))


def b4_temperature_compatibility(
    checkpoint_meta: dict[str, Any],
    *,
    requested_tau: float,
    target_type: str,
    require_match: bool,
) -> dict[str, Any]:
    raw_tau = checkpoint_meta.get("soft_target_tau")
    checkpoint_tau = None if raw_tau is None else float(raw_tau)
    matches = (
        checkpoint_tau is not None
        and math.isfinite(checkpoint_tau)
        and math.isclose(
            checkpoint_tau,
            requested_tau,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    result = {
        "checkpoint_soft_target_tau": checkpoint_tau,
        "requested_soft_target_tau": requested_tau,
        "matches": matches,
        "match_required": require_match,
        "continuous_target_uses_temperature": target_type == "soft_expected",
    }
    if require_match and not matches:
        raise ValueError(
            "B4 soft-target temperature differs from this run: "
            f"checkpoint={checkpoint_tau!r}, requested={requested_tau!r}"
        )
    return result


def load_frozen_b4(
    path: Path,
    *,
    candidate_steps: list[int],
    primary_lambda: float,
    soft_target_tau: float,
    split_seed: int,
    train_seed: int,
    target_type: str,
    require_temperature_match: bool,
    dataset_meta: dict[str, Any],
    device: torch.device,
) -> tuple[SoftDistillationMLPRouter, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "mlp_distill":
        raise ValueError(f"Expected an mlp_distill B4 checkpoint: {path}")
    if [int(value) for value in payload.get("candidate_steps", [])] != candidate_steps:
        raise ValueError("B4 candidate steps differ from the continuous-budget dataset")
    if not math.isclose(
        float(payload.get("primary_lambda", float("nan"))),
        primary_lambda,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("B4 primary lambda differs from this run")
    checkpoint_meta = payload.get("meta", {})
    for field in ("split_seed", "quality_profile", "latency_profile"):
        if checkpoint_meta.get(field) != dataset_meta.get(field):
            raise ValueError(f"B4 metadata mismatch for {field}")
    if int(checkpoint_meta.get("train_seed", -1)) != train_seed:
        raise ValueError("B4 training seed differs from the continuous-budget run")
    temperature = b4_temperature_compatibility(
        checkpoint_meta,
        requested_tau=soft_target_tau,
        target_type=target_type,
        require_match=require_temperature_match,
    )
    if not temperature["matches"]:
        log = LOGGER.warning if target_type == "soft_expected" else LOGGER.info
        log(
            "Frozen B4 temperature is not matched (%s); continuing because "
            "strict matching was not requested. target_type=%s",
            temperature,
            target_type,
        )
    model = SoftDistillationMLPRouter(
        in_dim=4096,
        hidden_dims=[256, 128],
        num_classes=len(candidate_steps),
        dropout=0.1,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, temperature


@torch.no_grad()
def continuous_validation_metrics(
    model: ContinuousBudgetRegressor,
    loader: torch.utils.data.DataLoader,
    budget_grid: torch.Tensor,
    target_type: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    regrets = []
    errors = []
    for batch in loader:
        predicted = model(batch["pooled_t5"].to(device)).cpu()
        chosen = nearest_budget_index(predicted, budget_grid)
        row = torch.arange(chosen.shape[0])
        oracle = batch["utilities"].max(dim=1).values
        realized = batch["utilities"][row, chosen]
        target = budget_targets(batch, budget_grid, target_type)
        regrets.append((oracle - realized).clamp(min=0.0))
        errors.append((predicted - target).abs())
    return {
        "policy_regret": float(torch.cat(regrets).mean()),
        "budget_mae": float(torch.cat(errors).mean()),
    }


def train_model(
    model: ContinuousBudgetRegressor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    budget_grid: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float]]]:
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=min(1e-5, args.lr * 0.1),
    )
    best_regret = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            pooled = batch["pooled_t5"].to(device)
            target = budget_targets(batch, budget_grid, args.target_type).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(pooled)
            loss = criterion(predicted, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.detach().item() * pooled.shape[0]
            sample_count += int(pooled.shape[0])
        scheduler.step()
        val = continuous_validation_metrics(
            model, val_loader, budget_grid, args.target_type, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(sample_count, 1),
                "validation_policy_regret": val["policy_regret"],
                "validation_budget_mae": val["budget_mae"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if val["policy_regret"] < best_regret:
            best_regret = val["policy_regret"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            LOGGER.info(
                "epoch=%d train_loss=%.6f val_regret=%.6f val_budget_mae=%.6f",
                epoch,
                history[-1]["train_loss"],
                val["policy_regret"],
                val["budget_mae"],
            )
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    return best_state, best_epoch, history


@torch.no_grad()
def evaluate_all_policies(
    *,
    continuous_model: ContinuousBudgetRegressor,
    b4_model: SoftDistillationMLPRouter,
    loader: torch.utils.data.DataLoader,
    candidate_steps: list[int],
    budget_grid: torch.Tensor,
    fixed_index: int,
    target_type: str,
    device: torch.device,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    continuous_model.eval()
    b4_model.eval()
    rows: list[dict[str, Any]] = []
    steps = torch.tensor(candidate_steps, dtype=torch.long)
    for batch in loader:
        pooled = batch["pooled_t5"].to(device)
        b4_output = b4_model(pooled)
        b4_probs = b4_output["discrete_probs"].cpu()
        b4_argmax = b4_output["pred_step_idx"].cpu()
        projected_budget = (b4_probs * budget_grid.unsqueeze(0)).sum(dim=1)
        continuous_budget = continuous_model(pooled).cpu()
        choices = {
            "prompt_oracle": batch["target_step_idx"],
            "best_fixed": torch.full_like(batch["target_step_idx"], fixed_index),
            "b4_argmax": b4_argmax,
            "b4_projected_nearest": nearest_budget_index(projected_budget, budget_grid),
            "continuous_budget_nearest": nearest_budget_index(
                continuous_budget, budget_grid
            ),
        }
        predicted_budgets = {
            "prompt_oracle": budget_grid[batch["target_step_idx"]],
            "best_fixed": torch.full_like(
                continuous_budget, float(budget_grid[fixed_index])
            ),
            "b4_argmax": budget_grid[b4_argmax],
            "b4_projected_nearest": projected_budget,
            "continuous_budget_nearest": continuous_budget,
        }
        target_budget = budget_targets(batch, budget_grid, target_type)
        batch_rows = torch.arange(batch["utilities"].shape[0])
        oracle_utility = batch["utilities"].max(dim=1).values
        for method, role, model_type in METHODS:
            chosen = choices[model_type]
            realized_utility = batch["utilities"][batch_rows, chosen]
            realized_quality = batch["vbench5"][batch_rows, chosen]
            realized_latency = batch["latencies"][batch_rows, chosen]
            native_latency = batch["native_latency"]
            dimensions = {
                name: values[batch_rows, chosen]
                for name, values in batch["vbench_dimensions"].items()
            }
            for index, prompt_id in enumerate(batch["prompt_id"].tolist()):
                row: dict[str, Any] = {
                    "split": split,
                    "Method": method,
                    "method_role": role,
                    "model_type": model_type,
                    "prompt_id": int(prompt_id),
                    "target_step": int(steps[batch["target_step_idx"][index]]),
                    "chosen_step": int(steps[chosen[index]]),
                    "target_budget": float(target_budget[index]),
                    "predicted_budget": float(predicted_budgets[model_type][index]),
                    "chosen_budget": float(budget_grid[chosen[index]]),
                    "budget_abs_error": float(
                        abs(predicted_budgets[model_type][index] - target_budget[index])
                    ),
                    "policy_regret": float(
                        max(0.0, oracle_utility[index] - realized_utility[index])
                    ),
                    "realized_utility": float(realized_utility[index]),
                    "oracle_utility": float(oracle_utility[index]),
                    "seed_oracle_utility": float(batch["seed_oracle_utility"][index]),
                    "realized_vbench5": float(realized_quality[index]),
                    "realized_latency_sec": float(realized_latency[index]),
                    "native_latency_sec": float(native_latency[index]),
                    "speedup_vs_native": float(
                        native_latency[index] / realized_latency[index].clamp(min=1e-6)
                    ),
                }
                for name, values in dimensions.items():
                    row[f"realized_{name}"] = float(values[index])
                rows.append(row)

    summaries = []
    for method, role, model_type in METHODS:
        selected = [row for row in rows if row["model_type"] == model_type]
        summary: dict[str, Any] = {
            "Method": method,
            "method_role": role,
            "model_type": model_type,
            "prompt_count": len(selected),
        }
        for metric in (
            "policy_regret",
            "realized_utility",
            "oracle_utility",
            "seed_oracle_utility",
            "realized_vbench5",
            "realized_latency_sec",
            "target_budget",
            "chosen_budget",
            "budget_abs_error",
            *[f"realized_{name}" for name in QUALITY5_DIMENSIONS],
        ):
            summary[metric] = float(np.mean([float(row[metric]) for row in selected]))
        summary["speedup_vs_native"] = float(
            np.mean([float(row["native_latency_sec"]) for row in selected])
            / max(summary["realized_latency_sec"], 1e-6)
        )
        summaries.append(summary)
    fixed_utility = next(
        row["realized_utility"]
        for row in summaries
        if row["model_type"] == "best_fixed"
    )
    oracle_utility = next(
        row["realized_utility"]
        for row in summaries
        if row["model_type"] == "prompt_oracle"
    )
    denominator = oracle_utility - fixed_utility
    for summary in summaries:
        summary["oracle_headroom_recovery"] = (
            (summary["realized_utility"] - fixed_utility) / denominator
            if denominator > 1e-12
            else 0.0
        )
    return summaries, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    summary_path = out_dir / "continuous_budget_validation_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {summary_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _test_loader, meta = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.split_seed,
        primary_lambda=args.primary_lambda,
        tau=args.soft_target_tau,
        allow_estimated_latency=args.allow_estimated_latency,
        require_measured_latency=args.require_measured_latency,
    )
    candidate_steps = [int(value) for value in meta["candidate_steps"]]
    budget_grid = calibrate_budget_grid(train_loader)
    fixed_index = train_selected_fixed_index(train_loader)
    b4_model, b4_payload, b4_temperature = load_frozen_b4(
        Path(args.b4_checkpoint).resolve(),
        candidate_steps=candidate_steps,
        primary_lambda=args.primary_lambda,
        soft_target_tau=args.soft_target_tau,
        split_seed=args.split_seed,
        train_seed=args.seed,
        target_type=args.target_type,
        require_temperature_match=args.require_b4_temperature_match,
        dataset_meta=meta,
        device=device,
    )

    continuous_model = ContinuousBudgetRegressor().to(device)
    state, best_epoch, history = train_model(
        continuous_model,
        train_loader,
        val_loader,
        budget_grid,
        args,
        device,
    )
    summaries, predictions = evaluate_all_policies(
        continuous_model=continuous_model,
        b4_model=b4_model,
        loader=val_loader,
        candidate_steps=candidate_steps,
        budget_grid=budget_grid,
        fixed_index=fixed_index,
        target_type=args.target_type,
        device=device,
        split="validation",
    )

    checkpoint = {
        "schema": "continuous_prompt_budget_checkpoint_v1",
        "model_type": "continuous_budget_nearest",
        "state_dict": state,
        "candidate_steps": candidate_steps,
        "budget_grid": budget_grid,
        "budget_definition": "train_median_candidate_latency_over_native_latency",
        "target_type": args.target_type,
        "primary_lambda": args.primary_lambda,
        "soft_target_tau": args.soft_target_tau,
        "b4_temperature_compatibility": b4_temperature,
        "best_epoch": best_epoch,
        "meta": {**meta, "train_seed": args.seed},
    }
    torch.save(checkpoint, out_dir / "continuous_budget_prior.pt")
    write_csv(out_dir / "training_history.csv", history)
    write_csv(out_dir / "continuous_budget_validation_predictions.csv", predictions)
    write_csv(out_dir / "continuous_budget_validation_results.csv", summaries)
    summary = {
        "schema": "continuous_prompt_budget_validation_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "test_accessed": False,
        "primary_lambda": args.primary_lambda,
        "target_type": args.target_type,
        "budget_definition": "train_median_candidate_latency_over_native_latency",
        "candidate_steps": candidate_steps,
        "budget_grid": budget_grid.tolist(),
        "train_selected_fixed_step": candidate_steps[fixed_index],
        "best_epoch": best_epoch,
        "meta": {**meta, "train_seed": args.seed},
        "dataset": {
            "path": str(Path(args.dataset_dir).resolve()),
            "manifest": str(
                (Path(args.dataset_dir).resolve() / "dataset_manifest.json")
            ),
            "manifest_sha256": sha256_file(
                Path(args.dataset_dir).resolve() / "dataset_manifest.json"
            ),
        },
        "b4_checkpoint": {
            "path": str(Path(args.b4_checkpoint).resolve()),
            "sha256": sha256_file(Path(args.b4_checkpoint).resolve()),
            "best_epoch": b4_payload.get("best_epoch"),
            "model_type": b4_payload.get("model_type"),
            "temperature_compatibility": b4_temperature,
        },
        "results": summaries,
        "artifacts": {
            "checkpoint": "continuous_budget_prior.pt",
            "training_history": "training_history.csv",
            "predictions": "continuous_budget_validation_predictions.csv",
            "results": "continuous_budget_validation_results.csv",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"budget_grid": budget_grid.tolist(), "results": summaries}, indent=2
        )
    )
    print(f"Validation summary: {summary_path}")


if __name__ == "__main__":
    main()
