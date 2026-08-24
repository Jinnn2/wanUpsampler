#!/usr/bin/env python3
"""Train and evaluate prompt-conditioned timestep routers without test leakage."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_router")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (  # noqa: E402
    QUALITY5_DIMENSIONS,
)
from changing_resolution_uni.scripts.router.dataset_router import (  # noqa: E402
    get_dataloaders,
)
from changing_resolution_uni.scripts.router.model_router import (  # noqa: E402
    LinearOrdinalRouter,
    LinearProbeRouter,
    OrdinalLoss,
    RelativeQualityCurveMLPRouter,
    SoftDistillationMLPRouter,
    SoftUtilityKLLoss,
    Wasserstein1Loss,
)

MODEL_LABELS = {
    "linear_probe": "Learned: Linear Probe (B1)",
    "linear_ordinal": "Learned: Linear Ordinal Regressor (B3)",
    "mlp_distill": "Learned: Soft Distillation MLP (B4)",
    "mlp_quality_curve": "Learned: Relative Quality Curve MLP (B4-Q)",
    "mlp_quality_aligned": "Learned: Utility-Aligned Quality Curve MLP (B4-QA)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default=str(
            REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"
        ),
    )
    parser.add_argument(
        "--out_dir", default=str(REPO_ROOT / "outputs" / "router_benchmarks_1k")
    )
    parser.add_argument(
        "--model_type",
        default="all",
        choices=[
            "linear_ordinal",
            "linear_probe",
            "mlp_distill",
            "mlp_quality_curve",
            "mlp_quality_aligned",
            "b4_comparison",
            "b4_qa_comparison",
            "all",
        ],
    )
    parser.add_argument(
        "--evaluation_stage",
        default="development",
        choices=["development", "selection", "confirmation"],
        help=(
            "selection evaluates validation only; confirmation evaluates a locked "
            "single architecture on test; development preserves the legacy test run"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--quality_curve_beta",
        type=float,
        default=0.01,
        help="SmoothL1 beta for B4-Q relative VBench-5 curve regression.",
    )
    parser.add_argument(
        "--quality_curve_alpha",
        type=float,
        default=1.0,
        help="Weight on the B4-QA SmoothL1 relative-quality auxiliary loss.",
    )
    parser.add_argument(
        "--soft_target_tau",
        type=float,
        default=0.02,
        help="Temperature shared by soft utility targets and B4-QA utility logits.",
    )
    parser.add_argument("--primary_lambda", type=float, default=0.01)
    parser.add_argument(
        "--seed", type=int, default=42, help="Training initialization/shuffle seed."
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Prompt split seed, intentionally independent of the training seed.",
    )
    parser.add_argument(
        "--allow_estimated_latency",
        action="store_true",
        help="Allow the explicitly non-formal legacy development dataset.",
    )
    parser.add_argument(
        "--require_measured_latency",
        action="store_true",
        help="Require warm_pipeline_seconds for native and every candidate branch.",
    )
    parser.add_argument(
        "--measure_router_overhead",
        action="store_true",
        help="Measure batch-1 router forward latency and include it for learned policies.",
    )
    parser.add_argument("--overhead_warmup", type=int, default=20)
    parser.add_argument("--overhead_repeats", type=int, default=200)
    parser.add_argument(
        "--allow_confirmation_overwrite",
        action="store_true",
        help="Explicit escape hatch; confirmation otherwise refuses to overwrite test results.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    if args.quality_curve_beta <= 0:
        parser.error("quality_curve_beta must be positive")
    if args.quality_curve_alpha < 0:
        parser.error("quality_curve_alpha must be non-negative")
    if args.soft_target_tau <= 0:
        parser.error("soft_target_tau must be positive")
    if args.overhead_warmup < 0 or args.overhead_repeats < 1:
        parser.error(
            "overhead_warmup must be non-negative and repeats must be positive"
        )
    if args.evaluation_stage == "confirmation" and args.model_type in {
        "all",
        "b4_comparison",
        "b4_qa_comparison",
    }:
        parser.error("confirmation requires one validation-selected --model_type")
    if args.evaluation_stage == "confirmation" and args.allow_estimated_latency:
        parser.error("confirmation cannot use --allow_estimated_latency")
    if args.evaluation_stage == "confirmation" and not args.require_measured_latency:
        parser.error("confirmation requires --require_measured_latency")
    if args.evaluation_stage == "confirmation" and not args.measure_router_overhead:
        parser.error("confirmation requires --measure_router_overhead")
    return args


def build_model(model_name: str, candidate_count: int) -> nn.Module:
    if model_name == "linear_ordinal":
        return LinearOrdinalRouter(in_dim=4096, num_classes=candidate_count)
    if model_name == "linear_probe":
        return LinearProbeRouter(in_dim=4096, num_classes=candidate_count)
    if model_name == "mlp_distill":
        return SoftDistillationMLPRouter(
            in_dim=4096,
            hidden_dims=[256, 128],
            num_classes=candidate_count,
            dropout=0.1,
        )
    if model_name in {"mlp_quality_curve", "mlp_quality_aligned"}:
        return RelativeQualityCurveMLPRouter(
            in_dim=4096,
            hidden_dims=[256, 128],
            num_classes=candidate_count,
            dropout=0.1,
        )
    raise ValueError(f"Unknown model_type {model_name}")


def quality_curve_utility(
    quality_deltas: torch.Tensor,
    latencies: torch.Tensor,
    native_latency: torch.Tensor,
    primary_lambda: float,
) -> torch.Tensor:
    """Combine a relative quality curve with normalized candidate cost."""
    latencies = latencies.to(quality_deltas.device)
    native_latency = native_latency.to(quality_deltas.device)
    if quality_deltas.shape != latencies.shape:
        raise ValueError(
            "quality_deltas and latencies must share [batch, candidate] shape; "
            f"got {tuple(quality_deltas.shape)} and {tuple(latencies.shape)}"
        )
    if native_latency.ndim != 1 or native_latency.shape[0] != latencies.shape[0]:
        raise ValueError("native_latency must have shape [batch]")
    return quality_deltas - float(primary_lambda) * (
        latencies / native_latency.unsqueeze(1).clamp(min=1e-6)
    )


def choose_model_step(
    model_output: dict[str, torch.Tensor],
    latencies: torch.Tensor,
    native_latency: torch.Tensor,
    primary_lambda: float,
) -> torch.Tensor:
    """Choose a candidate from either direct probabilities or a quality curve."""
    if "quality_deltas" not in model_output:
        return model_output["pred_step_idx"].detach().cpu()

    quality_deltas = model_output["quality_deltas"].detach().cpu()
    predicted_utility = quality_curve_utility(
        quality_deltas,
        latencies,
        native_latency,
        primary_lambda,
    )
    return torch.argmax(predicted_utility, dim=-1)


@torch.no_grad()
def evaluate_policy_on_loader(
    model: nn.Module | None,
    loader: torch.utils.data.DataLoader,
    candidate_steps: list[int],
    device: torch.device,
    *,
    primary_lambda: float,
    fixed_step: int | None = None,
    is_oracle: bool = False,
    router_overhead_sec: float = 0.0,
    method: str = "",
    method_role: str = "",
    model_type: str = "",
    split_name: str = "",
    prediction_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Evaluate utility and optionally emit one traceable row per prompt."""
    if model is not None:
        model.eval()
    if router_overhead_sec < 0:
        raise ValueError("router_overhead_sec must be non-negative")

    totals = {
        "samples": 0,
        "regret": 0.0,
        "realized_u": 0.0,
        "oracle_u": 0.0,
        "seed_oracle_u": 0.0,
        "vbench": 0.0,
        "latency": 0.0,
        "native_latency": 0.0,
        "step_mae": 0.0,
        "top1": 0,
        "top3": 0,
        "dimensions": {name: 0.0 for name in QUALITY5_DIMENSIONS},
    }
    cand_steps_tensor = torch.tensor(candidate_steps, dtype=torch.long)
    step_to_idx = {step: index for index, step in enumerate(candidate_steps)}

    for batch in loader:
        batch_size = batch["pooled_t5"].shape[0]
        target_idx = batch["target_step_idx"]
        target_step = batch["target_step"]
        utilities = batch["utilities"]
        vbench = batch["vbench5"]
        latencies = batch["latencies"]

        if is_oracle:
            chosen_idx = target_idx
        elif fixed_step is not None:
            chosen_idx = torch.full(
                (batch_size,),
                step_to_idx.get(fixed_step, len(candidate_steps) - 1),
                dtype=torch.long,
            )
        else:
            pooled = batch["pooled_t5"].to(device)
            chosen_idx = choose_model_step(
                model(pooled),
                latencies,
                batch["native_latency"],
                primary_lambda,
            )

        batch_indices = torch.arange(batch_size)
        chosen_step = cand_steps_tensor[chosen_idx]
        native_lat = batch["native_latency"]
        overhead = torch.full_like(native_lat, float(router_overhead_sec))
        realized_u = utilities[batch_indices, chosen_idx]
        if router_overhead_sec:
            realized_u = realized_u - primary_lambda * overhead / native_lat
        oracle_u = utilities[batch_indices, target_idx]
        seed_oracle_u = batch["seed_oracle_utility"]
        realized_vb = vbench[batch_indices, chosen_idx]
        realized_dimensions = {
            name: values[batch_indices, chosen_idx]
            for name, values in batch.get("vbench_dimensions", {}).items()
        }
        realized_lat = latencies[batch_indices, chosen_idx] + overhead
        regret = (oracle_u - realized_u).clamp(min=0.0)
        step_mae = (chosen_step - target_step).abs().float()

        totals["samples"] += batch_size
        totals["regret"] += regret.sum().item()
        totals["realized_u"] += realized_u.sum().item()
        totals["oracle_u"] += oracle_u.sum().item()
        totals["seed_oracle_u"] += seed_oracle_u.sum().item()
        totals["vbench"] += realized_vb.sum().item()
        for name, values in realized_dimensions.items():
            totals["dimensions"][name] += values.sum().item()
        totals["latency"] += realized_lat.sum().item()
        totals["native_latency"] += native_lat.sum().item()
        totals["step_mae"] += step_mae.sum().item()
        totals["top1"] += (chosen_idx == target_idx).sum().item()
        totals["top3"] += ((chosen_idx - target_idx).abs() <= 1).sum().item()

        if prediction_rows is not None:
            prompt_ids = batch["prompt_id"].tolist()
            for index in range(batch_size):
                row = {
                    "split": split_name,
                    "Method": method,
                    "method_role": method_role,
                    "model_type": model_type,
                    "prompt_id": int(prompt_ids[index]),
                    "target_step": int(target_step[index]),
                    "chosen_step": int(chosen_step[index]),
                    "policy_regret": float(regret[index]),
                    "realized_utility": float(realized_u[index]),
                    "oracle_utility": float(oracle_u[index]),
                    "seed_oracle_utility": float(seed_oracle_u[index]),
                    "regret_to_seed_oracle": max(
                        0.0, float(seed_oracle_u[index] - realized_u[index])
                    ),
                    "realized_vbench5": float(realized_vb[index]),
                    "realized_latency_sec": float(realized_lat[index]),
                    "native_latency_sec": float(native_lat[index]),
                    "speedup_vs_native": float(
                        native_lat[index] / realized_lat[index].clamp(min=1e-3)
                    ),
                    "step_abs_error": float(step_mae[index]),
                    "top1_correct": int(chosen_idx[index] == target_idx[index]),
                    "top3_correct": int(
                        abs(int(chosen_idx[index]) - int(target_idx[index])) <= 1
                    ),
                    "router_overhead_sec": float(router_overhead_sec),
                }
                for name, values in realized_dimensions.items():
                    row[f"realized_{name}"] = float(values[index])
                prediction_rows.append(row)

    count = max(int(totals["samples"]), 1)
    mean_latency = float(totals["latency"]) / count
    mean_native_latency = float(totals["native_latency"]) / count
    metrics = {
        "policy_regret": float(totals["regret"]) / count,
        "realized_utility": float(totals["realized_u"]) / count,
        "oracle_utility": float(totals["oracle_u"]) / count,
        "seed_oracle_utility": float(totals["seed_oracle_u"]) / count,
        "regret_to_seed_oracle": max(
            0.0,
            (float(totals["seed_oracle_u"]) - float(totals["realized_u"])) / count,
        ),
        "realized_vbench5": float(totals["vbench"]) / count,
        "realized_latency_sec": mean_latency,
        "speedup_vs_native": mean_native_latency / max(mean_latency, 1e-3),
        "step_mae": float(totals["step_mae"]) / count,
        "top1_acc": float(totals["top1"]) / count * 100.0,
        "top3_acc": float(totals["top3"]) / count * 100.0,
        "router_overhead_ms": router_overhead_sec * 1000.0,
    }
    for name, total in totals["dimensions"].items():
        if total or name in QUALITY5_DIMENSIONS:
            metrics[f"realized_{name}"] = float(total) / count
    return metrics


@torch.no_grad()
def measure_router_overhead(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure isolated batch-1 router forward latency on the evaluation device."""
    model.eval()
    pooled = next(iter(loader))["pooled_t5"][:1].to(device)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        model(pooled)
    if is_cuda:
        torch.cuda.synchronize(device)

    timings_ms: list[float] = []
    for _ in range(repeats):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(pooled)
        if is_cuda:
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(timings_ms)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "schema": "router_overhead_timing_v1",
        "device": str(device),
        "torch_device_name": torch.cuda.get_device_name(device) if is_cuda else "cpu",
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "mean_ms": statistics.fmean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "std_ms": statistics.pstdev(timings_ms),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if is_cuda
        else None,
        "synchronization": "torch.cuda.synchronize" if is_cuda else "not_applicable",
    }


def train_single_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    candidate_steps: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float], int]:
    logger.info("\n%s\n[Training Model]: %s\n%s", "=" * 70, model_name, "=" * 70)
    model = build_model(model_name, len(candidate_steps)).to(device)
    if model_name == "linear_ordinal":
        criterion: nn.Module = OrdinalLoss()
    elif model_name == "linear_probe":
        criterion = nn.CrossEntropyLoss()
    elif model_name in {"mlp_quality_curve", "mlp_quality_aligned"}:
        criterion = nn.SmoothL1Loss(beta=args.quality_curve_beta)
        if model_name == "mlp_quality_aligned":
            criterion_kl = SoftUtilityKLLoss()
            criterion_emd = Wasserstein1Loss()
    else:
        criterion_kl = SoftUtilityKLLoss()
        criterion_emd = Wasserstein1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    best_val_regret = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            pooled = batch["pooled_t5"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(pooled)
            if model_name == "linear_ordinal":
                loss = criterion(
                    out["cumulative_logits"], batch["ordinal_targets"].to(device)
                )
            elif model_name == "linear_probe":
                loss = criterion(out["logits"], batch["target_step_idx"].to(device))
            elif model_name == "mlp_quality_curve":
                loss = criterion(
                    out["quality_deltas"],
                    batch["relative_quality_target"].to(device),
                )
            elif model_name == "mlp_quality_aligned":
                quality_loss = criterion(
                    out["quality_deltas"],
                    batch["relative_quality_target"].to(device),
                )
                predicted_utility = quality_curve_utility(
                    out["quality_deltas"],
                    batch["latencies"],
                    batch["native_latency"],
                    args.primary_lambda,
                )
                utility_logits = predicted_utility / args.soft_target_tau
                utility_probs = torch.softmax(utility_logits, dim=-1)
                soft_target = batch["soft_utility_target"].to(device)
                loss = (
                    criterion_kl(utility_logits, soft_target)
                    + 0.5 * criterion_emd(utility_probs, soft_target)
                    + args.quality_curve_alpha * quality_loss
                )
            else:
                soft_target = batch["soft_utility_target"].to(device)
                loss = criterion_kl(out["logits"], soft_target) + 0.5 * criterion_emd(
                    out["discrete_probs"], soft_target
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        scheduler.step()

        val_metrics = evaluate_policy_on_loader(
            model,
            val_loader,
            candidate_steps,
            device,
            primary_lambda=args.primary_lambda,
        )
        if val_metrics["policy_regret"] < best_val_regret:
            best_val_regret = val_metrics["policy_regret"]
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(
                "Epoch [%02d/%02d] Loss: %.4f | Val Regret: %.6f | Val MAE: %.2f",
                epoch,
                args.epochs,
                running_loss / max(batches, 1),
                val_metrics["policy_regret"],
                val_metrics["step_mae"],
            )

    if best_weights is None:
        raise RuntimeError(f"No validation checkpoint was selected for {model_name}")
    model.load_state_dict(best_weights)
    final_val_metrics = evaluate_policy_on_loader(
        model,
        val_loader,
        candidate_steps,
        device,
        primary_lambda=args.primary_lambda,
    )
    return model, final_val_metrics, best_epoch


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    """Reset model initialization, shuffling, and dropout RNG streams."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    confirmation_guard = out_dir / "test_access_guard.json"
    if args.evaluation_stage == "confirmation":
        existing = out_dir / "router_benchmark_summary.json"
        existing_guard = confirmation_guard.exists()
        if (
            existing.exists() or existing_guard
        ) and not args.allow_confirmation_overwrite:
            raise FileExistsError(
                "Confirmation test was already started or completed; refusing a "
                f"second test read: summary={existing.exists()} guard={existing_guard}"
            )

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        seed=args.split_seed,
        primary_lambda=args.primary_lambda,
        tau=args.soft_target_tau,
        allow_estimated_latency=args.allow_estimated_latency,
        require_measured_latency=args.require_measured_latency,
    )
    meta.update(
        {
            "train_seed": args.seed,
            "split_seed": args.split_seed,
            "evaluation_stage": args.evaluation_stage,
            "quality_curve_target": "candidate_vbench5_minus_final_candidate_vbench5",
            "quality_curve_loss": "smooth_l1",
            "quality_curve_beta": args.quality_curve_beta,
            "quality_curve_alpha": args.quality_curve_alpha,
            "soft_target_tau": args.soft_target_tau,
            "quality_aligned_loss": "kl_utility_plus_0.5_wasserstein_plus_alpha_smooth_l1",
            "paired_model_initialization": args.model_type
            in {"b4_comparison", "b4_qa_comparison"},
        }
    )
    if args.evaluation_stage == "confirmation" and not meta.get("formal_evidence"):
        raise ValueError("Confirmation requires dataset formal_evidence=true")

    candidate_steps = meta["candidate_steps"]
    evaluation_split = "validation" if args.evaluation_stage == "selection" else "test"
    evaluation_loader = val_loader if evaluation_split == "validation" else test_loader
    meta["evaluation_split"] = evaluation_split
    logger.info(
        "Split seed=%d, train seed=%d; train/val/test=%d/%d/%d; evaluating %s only",
        args.split_seed,
        args.seed,
        meta["train_prompts"],
        meta["val_prompts"],
        meta["test_prompts"],
        evaluation_split,
    )
    if not meta.get("formal_evidence", False):
        logger.warning("DEVELOPMENT-ONLY RUN: provenance is not formal evidence")

    train_fixed_regrets = {}
    for step in candidate_steps:
        metrics = evaluate_policy_on_loader(
            None,
            train_loader,
            candidate_steps,
            device,
            primary_lambda=args.primary_lambda,
            fixed_step=step,
        )
        train_fixed_regrets[step] = metrics["policy_regret"]
    best_fixed_step = min(train_fixed_regrets, key=train_fixed_regrets.get)
    fixed_steps = [
        best_fixed_step,
        *[step for step in (47, 45, 50) if step != best_fixed_step],
    ]
    if args.model_type == "all":
        models_to_train = ["linear_probe", "linear_ordinal", "mlp_distill"]
    elif args.model_type == "b4_comparison":
        models_to_train = ["mlp_distill", "mlp_quality_curve"]
    elif args.model_type == "b4_qa_comparison":
        models_to_train = ["mlp_distill", "mlp_quality_aligned"]
    else:
        models_to_train = [args.model_type]
    trained_models: dict[str, tuple[nn.Module, dict[str, float], int]] = {}
    for model_name in models_to_train:
        if args.model_type in {"b4_comparison", "b4_qa_comparison"}:
            seed_everything(args.seed)
        trained_models[model_name] = train_single_model(
            model_name, train_loader, val_loader, candidate_steps, args, device
        )

    if args.evaluation_stage == "confirmation":
        confirmation_guard.write_text(
            json.dumps(
                {
                    "schema": "router_test_access_guard_v1",
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model_type": args.model_type,
                    "primary_lambda": args.primary_lambda,
                    "split_seed": args.split_seed,
                    "train_seed": args.seed,
                    "purpose": "single locked confirmation test",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    oracle_label = "Prompt Oracle (Upper Bound)"
    oracle_metrics = evaluate_policy_on_loader(
        None,
        evaluation_loader,
        candidate_steps,
        device,
        primary_lambda=args.primary_lambda,
        is_oracle=True,
        method=oracle_label,
        method_role="prompt_oracle",
        split_name=evaluation_split,
        prediction_rows=predictions,
    )
    results.append({"Method": oracle_label, **oracle_metrics})
    for index, step in enumerate(fixed_steps):
        if index == 0:
            label = f"Fixed Step {step} (Best Fixed)"
            role = "best_fixed"
        else:
            label = f"Fixed Step {step}" + (" (Pure LR)" if step == 50 else "")
            role = "fixed"
        metrics = evaluate_policy_on_loader(
            None,
            evaluation_loader,
            candidate_steps,
            device,
            primary_lambda=args.primary_lambda,
            fixed_step=step,
            method=label,
            method_role=role,
            split_name=evaluation_split,
            prediction_rows=predictions,
        )
        results.append({"Method": label, **metrics})

    overhead_reports: dict[str, Any] = {}
    for model_name, (model, val_metrics, best_epoch) in trained_models.items():
        overhead_sec = 0.0
        if args.measure_router_overhead:
            overhead = measure_router_overhead(
                model,
                evaluation_loader,
                device,
                warmup=args.overhead_warmup,
                repeats=args.overhead_repeats,
            )
            overhead["model_type"] = model_name
            overhead_reports[model_name] = overhead
            overhead_sec = float(overhead["median_ms"]) / 1000.0

        label = MODEL_LABELS[model_name]
        evaluation_metrics = evaluate_policy_on_loader(
            model,
            evaluation_loader,
            candidate_steps,
            device,
            primary_lambda=args.primary_lambda,
            router_overhead_sec=overhead_sec,
            method=label,
            method_role="learned",
            model_type=model_name,
            split_name=evaluation_split,
            prediction_rows=predictions,
        )
        results.append({"Method": label, **evaluation_metrics})
        checkpoint = {
            "model_type": model_name,
            "state_dict": model.state_dict(),
            "candidate_steps": candidate_steps,
            "primary_lambda": args.primary_lambda,
            "meta": meta,
            "best_epoch": best_epoch,
            "validation_metrics": val_metrics,
            "evaluation_split": evaluation_split,
            "evaluation_metrics": evaluation_metrics,
            "router_overhead": overhead_reports.get(model_name),
        }
        if evaluation_split == "test":
            checkpoint["test_metrics"] = evaluation_metrics
        torch.save(checkpoint, out_dir / f"{model_name}_router.pt")

    if evaluation_split == "validation":
        result_csv = out_dir / "router_validation_results.csv"
        result_json = out_dir / "router_validation_summary.json"
        prediction_csv = out_dir / "router_validation_predictions.csv"
    else:
        result_csv = out_dir / "router_benchmark_results.csv"
        result_json = out_dir / "router_benchmark_summary.json"
        prediction_csv = out_dir / "router_test_predictions.csv"

    write_rows(result_csv, results)
    write_rows(prediction_csv, predictions)
    summary = {
        "schema": "router_evaluation_summary_v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "primary_lambda": args.primary_lambda,
        "evaluation_stage": args.evaluation_stage,
        "evaluation_split": evaluation_split,
        "test_accessed": evaluation_split == "test",
        "meta": meta,
        "router_overhead": overhead_reports,
        "results": results,
        "artifacts": {
            "per_prompt_predictions": prediction_csv.name,
            "test_access_guard": (
                confirmation_guard.name
                if args.evaluation_stage == "confirmation"
                else None
            ),
        },
    }
    result_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.evaluation_stage == "confirmation":
        guard_payload = json.loads(confirmation_guard.read_text(encoding="utf-8"))
        guard_payload.update(
            {
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "summary": result_json.name,
                "predictions": prediction_csv.name,
            }
        )
        confirmation_guard.write_text(
            json.dumps(guard_payload, indent=2), encoding="utf-8"
        )
    if overhead_reports:
        (out_dir / "router_overhead.json").write_text(
            json.dumps(overhead_reports, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("\n" + "=" * 100)
    print(
        f" ROUTER {args.evaluation_stage.upper()} RESULTS "
        f"({evaluation_split}: {len(evaluation_loader.dataset)} prompts)"
    )
    print("=" * 100)
    for row in results:
        print(
            f"{row['Method']:<38} regret={row['policy_regret']:.6f} "
            f"VBench-5={row['realized_vbench5']:.6f} "
            f"latency={row['realized_latency_sec']:.3f}s "
            f"speedup={row['speedup_vs_native']:.2f}x"
        )
    print("=" * 100)
    logger.info("Results: %s; per-prompt predictions: %s", result_json, prediction_csv)


if __name__ == "__main__":
    main()
