#!/usr/bin/env python3
"""
Training & Evaluation Pipeline for Prompt-Conditioned Optimal Switching Routers.
Evaluates Policy Regret, Realized VBench Quality, Latency, Step MAE, and compares
against Fixed Baselines and Oracle Upper Bounds.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_router")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.router.dataset_router import get_dataloaders
from changing_resolution_uni.scripts.router.model_router import (
    LinearOrdinalRouter,
    LinearProbeRouter,
    OrdinalLoss,
    SoftDistillationMLPRouter,
    SoftUtilityKLLoss,
    Wasserstein1Loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate prompt-conditioned switching router.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"),
        help="Path to merged oracle dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "router_benchmarks_1k"),
        help="Directory to save checkpoints, logs, and evaluation metrics.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=["linear_ordinal", "linear_probe", "mlp_distill", "all"],
        help="Model architecture to train (or 'all' to train and benchmark all variants).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument("--primary_lambda", type=float, default=0.01, help="Utility tradeoff lambda.")
    parser.add_argument("--seed", type=int, default=42, help="Random split and init seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate_policy_on_loader(
    model: nn.Module | None,
    loader: torch.utils.data.DataLoader,
    candidate_steps: list[int],
    device: torch.device,
    fixed_step: int | None = None,
    is_oracle: bool = False,
) -> dict[str, float]:
    """
    Evaluates a policy on a dataloader, measuring real Pareto metrics:
      - Policy Regret E[U(k*) - U(k_pred)]
      - Realized Utility E[U(k_pred)]
      - Realized VBench-5 Quality
      - Realized Latency (s) & Speedup
      - Step MAE
      - Top-1 and Top-3 Accuracy
    """
    if model is not None:
        model.eval()

    total_samples = 0
    total_regret = 0.0
    total_realized_u = 0.0
    total_oracle_u = 0.0
    total_seed_oracle_u = 0.0
    total_realized_vbench = 0.0
    total_realized_latency = 0.0
    total_native_latency = 0.0
    total_step_mae = 0.0
    correct_top1 = 0
    correct_top3 = 0

    cand_steps_tensor = torch.tensor(candidate_steps, dtype=torch.long)
    step_to_idx = {s: i for i, s in enumerate(candidate_steps)}

    for batch in loader:
        B = batch["pooled_t5"].shape[0]
        target_idx = batch["target_step_idx"]  # [B]
        target_step = batch["target_step"]  # [B]
        utilities = batch["utilities"]  # [B, K]
        vbench = batch["vbench5"]  # [B, K]
        latencies = batch["latencies"]  # [B, K]

        # Determine chosen step index for each item in batch
        if is_oracle:
            chosen_idx = target_idx
        elif fixed_step is not None:
            f_idx = step_to_idx.get(fixed_step, len(candidate_steps) - 1)
            chosen_idx = torch.full((B,), f_idx, dtype=torch.long)
        else:
            pooled = batch["pooled_t5"].to(device)
            out = model(pooled)
            chosen_idx = out["pred_step_idx"].cpu()

        chosen_step = cand_steps_tensor[chosen_idx]

        # Extract realized metrics for chosen index
        batch_indices = torch.arange(B)
        realized_u = utilities[batch_indices, chosen_idx]
        oracle_u = utilities[batch_indices, target_idx]
        realized_vb = vbench[batch_indices, chosen_idx]
        realized_lat = latencies[batch_indices, chosen_idx]
        native_lat = batch["native_latency"]
        seed_oracle_u = batch["seed_oracle_utility"]

        regret = (oracle_u - realized_u).clamp(min=0.0)
        step_mae = (chosen_step - target_step).abs().float()

        total_samples += B
        total_regret += regret.sum().item()
        total_realized_u += realized_u.sum().item()
        total_oracle_u += oracle_u.sum().item()
        total_seed_oracle_u += seed_oracle_u.sum().item()
        total_realized_vbench += realized_vb.sum().item()
        total_realized_latency += realized_lat.sum().item()
        total_native_latency += native_lat.sum().item()
        total_step_mae += step_mae.sum().item()

        # Accuracy
        correct_top1 += (chosen_idx == target_idx).sum().item()
        # Neighbor accuracy: exact or one adjacent candidate index.
        correct_top3 += ((chosen_idx - target_idx).abs() <= 1).sum().item()

    n = max(total_samples, 1)
    mean_lat = total_realized_latency / n
    native_lat = total_native_latency / n
    speedup = native_lat / max(mean_lat, 1e-3)

    return {
        "policy_regret": total_regret / n,
        "realized_utility": total_realized_u / n,
        "oracle_utility": total_oracle_u / n,
        "seed_oracle_utility": total_seed_oracle_u / n,
        "regret_to_seed_oracle": max(
            0.0, (total_seed_oracle_u - total_realized_u) / n
        ),
        "realized_vbench5": total_realized_vbench / n,
        "realized_latency_sec": mean_lat,
        "speedup_vs_native": speedup,
        "step_mae": total_step_mae / n,
        "top1_acc": (correct_top1 / n) * 100.0,
        "top3_acc": (correct_top3 / n) * 100.0,
    }


def train_single_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    candidate_steps: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    logger.info(f"\n{'='*70}\n[Training Model]: {model_name}\n{'='*70}")
    K = len(candidate_steps)

    if model_name == "linear_ordinal":
        model = LinearOrdinalRouter(in_dim=4096, num_classes=K).to(device)
        criterion = OrdinalLoss()
    elif model_name == "linear_probe":
        model = LinearProbeRouter(in_dim=4096, num_classes=K).to(device)
        criterion = nn.CrossEntropyLoss()
    elif model_name == "mlp_distill":
        model = SoftDistillationMLPRouter(in_dim=4096, hidden_dims=[256, 128], num_classes=K, dropout=0.1).to(device)
        criterion_kl = SoftUtilityKLLoss()
        criterion_emd = Wasserstein1Loss()
    else:
        raise ValueError(f"Unknown model_type {model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_regret = 1e9
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            pooled = batch["pooled_t5"].to(device)
            optimizer.zero_grad()

            if model_name == "linear_ordinal":
                out = model(pooled)
                loss = criterion(out["cumulative_logits"], batch["ordinal_targets"].to(device))
            elif model_name == "linear_probe":
                out = model(pooled)
                loss = criterion(out["logits"], batch["target_step_idx"].to(device))
            elif model_name == "mlp_distill":
                out = model(pooled)
                soft_t = batch["soft_utility_target"].to(device)
                loss_kl = criterion_kl(out["logits"], soft_t)
                loss_emd = criterion_emd(out["discrete_probs"], soft_t)
                loss = loss_kl + 0.5 * loss_emd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(batches, 1)

        # Validation
        val_metrics = evaluate_policy_on_loader(model, val_loader, candidate_steps, device)
        val_regret = val_metrics["policy_regret"]

        if val_regret < best_val_regret:
            best_val_regret = val_regret
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(
                f"Epoch [{epoch:02d}/{args.epochs}] Loss: {epoch_loss:.4f} | "
                f"Val Regret: {val_regret:.6f} | Val MAE: {val_metrics['step_mae']:.2f} steps | "
                f"Val Speedup: {val_metrics['speedup_vs_native']:.2f}x"
            )

    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Final Blind Test Evaluation
    test_metrics = evaluate_policy_on_loader(model, test_loader, candidate_steps, device)
    logger.info(
        f"-> [Test Evaluation ({model_name})]: "
        f"Regret: {test_metrics['policy_regret']:.6f} | "
        f"VBench5: {test_metrics['realized_vbench5']:.4f} | "
        f"Speedup: {test_metrics['speedup_vs_native']:.2f}x | "
        f"MAE: {test_metrics['step_mae']:.2f} steps | "
        f"Top-1: {test_metrics['top1_acc']:.1f}%"
    )

    return model, test_metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    logger.info(f"Loading dataset from: {args.dataset_dir}")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        primary_lambda=args.primary_lambda,
    )
    cand_steps = meta["candidate_steps"]

    logger.info(
        f"Prompt-Disjoint Split: {meta['train_prompts']} Train ({meta['train_trajectories']} trajs), "
        f"{meta['val_prompts']} Val ({meta['val_trajectories']} trajs), "
        f"{meta['test_prompts']} Test ({meta['test_trajectories']} trajs)"
    )
    logger.info(f"Candidate switch steps: {cand_steps}")

    # ── Evaluate Standard Baselines First ─────────────────────────────────────────
    results: list[dict[str, Any]] = []

    # 1. Oracle Upper Bound
    oracle_test = evaluate_policy_on_loader(None, test_loader, cand_steps, device, is_oracle=True)
    results.append({"Method": "Prompt Oracle (Upper Bound)", **oracle_test})

    # Evaluate all fixed steps on training set to find the empirical best fixed policy
    train_fixed_regrets = {}
    for s in cand_steps:
        s_eval = evaluate_policy_on_loader(None, train_loader, cand_steps, device, fixed_step=s)
        train_fixed_regrets[s] = s_eval["policy_regret"]
    best_fixed_step = min(train_fixed_regrets, key=train_fixed_regrets.get)

    # 2. Best Empirical Fixed Step Baseline
    best_fixed_test = evaluate_policy_on_loader(None, test_loader, cand_steps, device, fixed_step=best_fixed_step)
    results.append({"Method": f"Fixed Step {best_fixed_step} (Best Fixed)", **best_fixed_test})

    # 3. Other representative fixed baselines (Step 47, Step 45, Step 50)
    for s in [47, 45, 50]:
        if s != best_fixed_step:
            s_test = evaluate_policy_on_loader(None, test_loader, cand_steps, device, fixed_step=s)
            s_label = f"Fixed Step {s}" + (" (Pure LR)" if s == 50 else "")
            results.append({"Method": s_label, **s_test})

    # ── Train Learned Router Models ──────────────────────────────────────────────
    models_to_train = [args.model_type] if args.model_type != "all" else ["linear_probe", "linear_ordinal", "mlp_distill"]

    for m_name in models_to_train:
        model, test_metrics = train_single_model(
            m_name, train_loader, val_loader, test_loader, cand_steps, args, device
        )
        ckpt_path = out_dir / f"{m_name}_router.pt"
        torch.save({
            "model_type": m_name,
            "state_dict": model.state_dict(),
            "candidate_steps": cand_steps,
            "primary_lambda": args.primary_lambda,
            "meta": meta,
            "test_metrics": test_metrics,
        }, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        label_map = {
            "linear_probe": "Learned: Linear Probe (B1)",
            "linear_ordinal": "Learned: Linear Ordinal Regressor (B3)",
            "mlp_distill": "Learned: Soft Distillation MLP (B4)",
        }
        results.append({"Method": label_map.get(m_name, m_name), **test_metrics})

    # ── Print & Save Master Benchmark Comparison Table ───────────────────────────
    csv_path = out_dir / "router_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = out_dir / "router_benchmark_summary.json"
    json_path.write_text(json.dumps({
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "primary_lambda": args.primary_lambda,
        "meta": meta,
        "results": results,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 95)
    print(f" PROMPT-CONDITIONED ROUTER BENCHMARK RESULTS (Test Set: {meta['test_prompts']} Prompts, {meta['test_trajectories']} Trajectories)")
    print("=" * 95)
    print(f"{'Method':<38} | {'Regret':<9} | {'VBench-5':<9} | {'Latency':<9} | {'Speedup':<8} | {'MAE':<6} | {'Top-1'}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['Method']:<38} | "
            f"{r['policy_regret']:<9.6f} | "
            f"{r['realized_vbench5']:<9.4f} | "
            f"{r['realized_latency_sec']:<7.1f}s | "
            f"{r['speedup_vs_native']:<6.2f}x | "
            f"{r['step_mae']:<4.2f}st | "
            f"{r['top1_acc']:.1f}%"
        )
    print("=" * 95 + "\n")
    logger.info(f"Results saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
