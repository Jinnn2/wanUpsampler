#!/usr/bin/env python3
"""Audit B4-relative one-sided correction headroom on candidate steps 40--50."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import analyze_factor_relevance as factor_audit
    from . import train_variable_lambda_router as base
    from .candidate_step_subset import (
        resolve_candidate_subset,
        subset_trajectory_candidates,
    )
except ImportError:
    import analyze_factor_relevance as factor_audit
    import train_variable_lambda_router as base
    from candidate_step_subset import (
        resolve_candidate_subset,
        subset_trajectory_candidates,
    )


REPORT_SCHEMA = "steps40_50_b4_preemption_headroom_v1"
TIE_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--b4-runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate-steps", type=int, nargs="+", default=list(range(40, 51))
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2033)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.lambdas = sorted(set(float(value) for value in args.lambdas))
    if not args.lambdas or any(
        value < 0 or not math.isfinite(value) for value in args.lambdas
    ):
        parser.error("lambdas must contain finite non-negative values")
    if args.harm_epsilon < 0 or not math.isfinite(args.harm_epsilon):
        parser.error("harm-epsilon must be finite and non-negative")
    if args.bootstrap_samples < 1 or args.inference_batch_size < 1:
        parser.error("bootstrap-samples and inference-batch-size must be positive")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def action_spaces() -> list[dict[str, Any]]:
    return [
        {"name": f"lower_{offset}", "direction": "lower", "max_offset": offset}
        for offset in (1, 2, 3)
    ] + [
        {"name": "lower_all", "direction": "lower", "max_offset": None},
        *[
            {
                "name": f"higher_{offset}",
                "direction": "higher",
                "max_offset": offset,
            }
            for offset in (1, 2, 3)
        ],
        {"name": "higher_all", "direction": "higher", "max_offset": None},
    ]


def build_allowed_mask(
    anchor: np.ndarray,
    candidate_count: int,
    direction: str,
    max_offset: int | None,
) -> np.ndarray:
    candidate = np.arange(candidate_count, dtype=np.int64)[None, None, :]
    anchor_expanded = np.asarray(anchor, dtype=np.int64)[..., None]
    if direction == "lower":
        allowed = candidate <= anchor_expanded
        if max_offset is not None:
            allowed &= candidate >= anchor_expanded - max_offset
    elif direction == "higher":
        allowed = candidate >= anchor_expanded
        if max_offset is not None:
            allowed &= candidate <= anchor_expanded + max_offset
    else:
        raise ValueError(f"Unknown action-space direction: {direction}")
    return allowed


def gather_last(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def choose_best_actions(
    gains: np.ndarray, allowed: np.ndarray, anchor: np.ndarray
) -> np.ndarray:
    """Choose per-seed best actions, preferring B4 whenever gain is a tie."""
    masked = np.where(allowed[:, None, :, :], gains, -np.inf)
    chosen = masked.argmax(axis=-1)
    best_gain = gather_last(gains, chosen)
    return np.where(best_gain > TIE_TOLERANCE, chosen, anchor[:, None, :]).astype(
        np.int64
    )


def choose_common_actions(
    gains: np.ndarray,
    allowed: np.ndarray,
    anchor: np.ndarray,
    harm_epsilon: float,
    minimum_positive_seeds: int,
) -> np.ndarray:
    """Choose one action per prompt/lambda under a seed-consistency constraint."""
    mean_gain = gains.mean(axis=1)
    eligible = allowed.copy()
    if minimum_positive_seeds > 0:
        positive_count = (gains > harm_epsilon).sum(axis=1)
        eligible &= positive_count >= minimum_positive_seeds
    masked = np.where(eligible, mean_gain, -np.inf)
    has_eligible = eligible.any(axis=-1)
    chosen = masked.argmax(axis=-1)
    chosen_gain = gather_last(mean_gain, chosen)
    return np.where(
        has_eligible & (chosen_gain > TIE_TOLERANCE), chosen, anchor
    ).astype(np.int64)


def group_validation_arrays(
    trajectories: list[dict[str, Any]],
    utilities: np.ndarray,
    b4_chosen: np.ndarray,
    expected_base_seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prompt_ids = np.asarray(
        sorted({int(item["prompt_id"]) for item in trajectories}), dtype=np.int64
    )
    base_seeds = np.asarray(sorted(expected_base_seeds), dtype=np.int64)
    index_by_key = {
        (int(item["prompt_id"]), int(item["seed"]) - int(item["prompt_id"])): index
        for index, item in enumerate(trajectories)
    }
    grouped_indices = np.empty((len(prompt_ids), len(base_seeds)), dtype=np.int64)
    for prompt_index, prompt_id in enumerate(prompt_ids):
        for seed_index, base_seed in enumerate(base_seeds):
            key = (int(prompt_id), int(base_seed))
            if key not in index_by_key:
                raise ValueError(f"Missing validation trajectory: {key}")
            grouped_indices[prompt_index, seed_index] = index_by_key[key]
    grouped_utilities = utilities[grouped_indices]
    grouped_b4 = b4_chosen[grouped_indices]
    if np.any(grouped_b4 != grouped_b4[:, :1]):
        raise ValueError("Prompt-only B4 ensemble differs across generation seeds")
    qualities = np.stack([item["qualities"] for item in trajectories])[grouped_indices]
    return (
        prompt_ids,
        base_seeds,
        grouped_utilities,
        grouped_b4[:, 0],
        qualities,
    )


def bootstrap_macro(
    gain: np.ndarray,
    b4_regret: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    prompt_gain = gain.mean(axis=(1, 2), dtype=np.float64)
    prompt_b4_regret = b4_regret.mean(axis=(1, 2), dtype=np.float64)
    indices = rng.integers(0, len(prompt_gain), size=(samples, len(prompt_gain)))
    gain_draws = prompt_gain[indices].mean(axis=1)
    denominator = prompt_b4_regret[indices].mean(axis=1)
    recovery_draws = np.divide(
        gain_draws,
        denominator,
        out=np.zeros_like(gain_draws),
        where=denominator > 0,
    )
    gain_low, gain_high = np.quantile(gain_draws, [0.025, 0.975])
    recovery_low, recovery_high = np.quantile(recovery_draws, [0.025, 0.975])
    recovery = float(gain.mean() / b4_regret.mean()) if b4_regret.mean() > 0 else 0.0
    return {
        "mean_utility_gain": float(gain.mean()),
        "mean_utility_gain_ci95_low": float(gain_low),
        "mean_utility_gain_ci95_high": float(gain_high),
        "recovered_b4_regret_fraction": recovery,
        "recovered_b4_regret_fraction_ci95_low": float(recovery_low),
        "recovered_b4_regret_fraction_ci95_high": float(recovery_high),
    }


def summarize_selection(
    space: dict[str, Any],
    rule: str,
    chosen: np.ndarray,
    gains: np.ndarray,
    anchor: np.ndarray,
    qualities: np.ndarray,
    candidate_steps: np.ndarray,
    candidate_seconds: np.ndarray,
    b4_regret: np.ndarray,
    harm_epsilon: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_count, seed_count, lambda_count = gains.shape[:3]
    anchor_expanded = np.broadcast_to(anchor[:, None, :], chosen.shape)
    realized_gain = gather_last(gains, chosen)
    quality_grid = np.broadcast_to(
        qualities[:, :, None, :],
        (prompt_count, seed_count, lambda_count, qualities.shape[-1]),
    )
    chosen_quality = gather_last(quality_grid, chosen)
    anchor_quality = gather_last(quality_grid, anchor_expanded)
    quality_delta = chosen_quality - anchor_quality
    chosen_seconds = candidate_seconds[chosen]
    anchor_seconds = candidate_seconds[anchor_expanded]
    latency_delta = chosen_seconds - anchor_seconds
    step_delta = candidate_steps[chosen] - candidate_steps[anchor_expanded]
    policy_regret = np.maximum(b4_regret - realized_gain, 0.0)
    common = {
        "action_space": space["name"],
        "direction": space["direction"],
        "max_offset": "" if space["max_offset"] is None else space["max_offset"],
        "selection_rule": rule,
    }
    macro = {
        **common,
        "lambda": "macro",
        **bootstrap_macro(realized_gain, b4_regret, bootstrap_samples, rng),
        "b4_policy_regret": float(b4_regret.mean()),
        "policy_regret": float(policy_regret.mean()),
        "mean_quality_delta": float(quality_delta.mean()),
        "mean_latency_delta_sec": float(latency_delta.mean()),
        "mean_step_delta": float(step_delta.mean()),
        "decision_change_rate": float((chosen != anchor_expanded).mean()),
        "material_gain_rate": float((realized_gain > harm_epsilon).mean()),
        "harm_vs_b4_rate": float((realized_gain < -harm_epsilon).mean()),
    }
    per_lambda = []
    for lambda_index in range(lambda_count):
        lambda_gain = realized_gain[:, :, lambda_index]
        lambda_b4_regret = b4_regret[:, :, lambda_index]
        recovery = (
            float(lambda_gain.mean() / lambda_b4_regret.mean())
            if lambda_b4_regret.mean() > 0
            else 0.0
        )
        per_lambda.append(
            {
                **common,
                "lambda_index": lambda_index,
                "mean_utility_gain": float(lambda_gain.mean()),
                "recovered_b4_regret_fraction": recovery,
                "b4_policy_regret": float(lambda_b4_regret.mean()),
                "policy_regret": float(policy_regret[:, :, lambda_index].mean()),
                "mean_quality_delta": float(quality_delta[:, :, lambda_index].mean()),
                "mean_latency_delta_sec": float(
                    latency_delta[:, :, lambda_index].mean()
                ),
                "mean_step_delta": float(step_delta[:, :, lambda_index].mean()),
                "decision_change_rate": float(
                    (
                        chosen[:, :, lambda_index]
                        != anchor_expanded[:, :, lambda_index]
                    ).mean()
                ),
                "material_gain_rate": float((lambda_gain > harm_epsilon).mean()),
                "harm_vs_b4_rate": float((lambda_gain < -harm_epsilon).mean()),
            }
        )
    return macro, per_lambda


def offset_rows(
    space: dict[str, Any],
    rule: str,
    chosen: np.ndarray,
    anchor: np.ndarray,
    candidate_steps: np.ndarray,
) -> list[dict[str, Any]]:
    anchor_expanded = np.broadcast_to(anchor[:, None, :], chosen.shape)
    offsets = candidate_steps[chosen] - candidate_steps[anchor_expanded]
    rows = []
    for lambda_index in range(offsets.shape[2]):
        values, counts = np.unique(offsets[:, :, lambda_index], return_counts=True)
        for value, count in zip(values, counts):
            rows.append(
                {
                    "action_space": space["name"],
                    "selection_rule": rule,
                    "lambda_index": lambda_index,
                    "step_offset": int(value),
                    "count": int(count),
                    "fraction": float(count / offsets[:, :, lambda_index].size),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    runs_root = Path(args.b4_runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    manifest = base.load_dataset_manifest(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_sha256 = base.sha256_file(manifest_path)
    source_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    subset_indices, candidate_steps = resolve_candidate_subset(
        source_steps, args.candidate_steps
    )
    feature_indices = np.arange(int(manifest["feature_count"]), dtype=np.int64)
    trajectories = base.load_trajectories(
        dataset_dir, manifest, "validation", feature_indices
    )
    validation_meta = manifest["splits"]["validation"]
    expected_base_seeds = [int(value) for value in validation_meta["base_seeds"]]
    if len(expected_base_seeds) != 3:
        raise ValueError("Headroom audit requires exactly three validation base seeds")
    cost_profile, candidate_seconds, native_seconds, latency_profile = (
        base.load_locked_latency_profile(manifest, source_steps, None)
    )
    base.apply_locked_latency_profile(
        trajectories, cost_profile, candidate_seconds, native_seconds
    )
    subset_trajectory_candidates(trajectories, subset_indices)
    candidate_seconds = candidate_seconds[subset_indices]

    device = torch.device(args.device)
    models, b4_inputs = factor_audit.load_b4_ensemble(
        runs_root, manifest_sha256, len(candidate_steps), device
    )
    print("Computing frozen five-seed B4 probability-ensemble decisions...")
    b4_margins = factor_audit.ensemble_b4_margins(
        models,
        trajectories,
        args.lambdas,
        device,
        args.inference_batch_size,
    )
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    b4_chosen = factor_audit.first_nonnegative_margin(b4_margins)
    qualities_flat = np.stack([item["qualities"] for item in trajectories])
    costs_flat = np.stack([item["costs"] for item in trajectories])
    utilities_flat = (
        qualities_flat[:, None, :]
        - np.asarray(args.lambdas)[None, :, None] * costs_flat[:, None, :]
    )
    (
        prompt_ids,
        base_seeds,
        utilities,
        anchor,
        qualities,
    ) = group_validation_arrays(
        trajectories,
        utilities_flat,
        b4_chosen,
        expected_base_seeds,
    )
    utility_grid = utilities
    anchor_expanded = np.broadcast_to(anchor[:, None, :], utilities.shape[:-1])
    anchor_utility = gather_last(utility_grid, anchor_expanded)
    gains = utility_grid - anchor_utility[..., None]
    oracle_utility = utility_grid.max(axis=-1)
    b4_regret = np.maximum(oracle_utility - anchor_utility, 0.0)

    rng = np.random.default_rng(args.bootstrap_seed)
    macro_rows: list[dict[str, Any]] = []
    per_lambda_rows: list[dict[str, Any]] = []
    offset_distribution: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    spaces = action_spaces()
    for space in spaces:
        print(f"Evaluating {space['name']} headroom...")
        allowed = build_allowed_mask(
            anchor,
            len(candidate_steps),
            space["direction"],
            space["max_offset"],
        )
        individual = choose_best_actions(gains, allowed, anchor)
        common_mean = choose_common_actions(
            gains, allowed, anchor, args.harm_epsilon, minimum_positive_seeds=0
        )
        majority = choose_common_actions(
            gains, allowed, anchor, args.harm_epsilon, minimum_positive_seeds=2
        )
        all_three = choose_common_actions(
            gains, allowed, anchor, args.harm_epsilon, minimum_positive_seeds=3
        )
        selections = {
            "per_seed_oracle": individual,
            "mean_seed_common_action": np.broadcast_to(
                common_mean[:, None, :], individual.shape
            ),
            "majority_positive_common_action": np.broadcast_to(
                majority[:, None, :], individual.shape
            ),
            "all3_positive_common_action": np.broadcast_to(
                all_three[:, None, :], individual.shape
            ),
        }
        for rule, chosen in selections.items():
            macro, per_lambda = summarize_selection(
                space,
                rule,
                chosen,
                gains,
                anchor,
                qualities,
                candidate_steps,
                candidate_seconds,
                b4_regret,
                args.harm_epsilon,
                args.bootstrap_samples,
                rng,
            )
            macro_rows.append(macro)
            for row in per_lambda:
                row["lambda"] = args.lambdas[row.pop("lambda_index")]
            per_lambda_rows.extend(per_lambda)
            offsets = offset_rows(space, rule, chosen, anchor, candidate_steps)
            for row in offsets:
                row["lambda"] = args.lambdas[row.pop("lambda_index")]
            offset_distribution.extend(offsets)

        independent_gain = gather_last(gains, individual)
        for prompt_index, prompt_id in enumerate(prompt_ids):
            for lambda_index, lambda_value in enumerate(args.lambdas):
                individual_steps = candidate_steps[
                    individual[prompt_index, :, lambda_index]
                ]
                row: dict[str, Any] = {
                    "action_space": space["name"],
                    "prompt_id": int(prompt_id),
                    "lambda": lambda_value,
                    "b4_step": int(candidate_steps[anchor[prompt_index, lambda_index]]),
                    "independent_best_steps_agree_all3": bool(
                        np.unique(individual_steps).size == 1
                    ),
                    "independent_materially_beneficial_seed_count": int(
                        np.sum(
                            independent_gain[prompt_index, :, lambda_index]
                            > args.harm_epsilon
                        )
                    ),
                    "mean_seed_common_step": int(
                        candidate_steps[common_mean[prompt_index, lambda_index]]
                    ),
                    "majority_positive_common_step": int(
                        candidate_steps[majority[prompt_index, lambda_index]]
                    ),
                    "all3_positive_common_step": int(
                        candidate_steps[all_three[prompt_index, lambda_index]]
                    ),
                }
                for seed_index, base_seed in enumerate(base_seeds):
                    row[f"seed_{base_seed}_independent_best_step"] = int(
                        individual_steps[seed_index]
                    )
                    row[f"seed_{base_seed}_independent_gain"] = float(
                        independent_gain[prompt_index, seed_index, lambda_index]
                    )
                    row[f"seed_{base_seed}_mean_common_gain"] = float(
                        gains[
                            prompt_index,
                            seed_index,
                            lambda_index,
                            common_mean[prompt_index, lambda_index],
                        ]
                    )
                    row[f"seed_{base_seed}_majority_common_gain"] = float(
                        gains[
                            prompt_index,
                            seed_index,
                            lambda_index,
                            majority[prompt_index, lambda_index],
                        ]
                    )
                    row[f"seed_{base_seed}_all3_common_gain"] = float(
                        gains[
                            prompt_index,
                            seed_index,
                            lambda_index,
                            all_three[prompt_index, lambda_index],
                        ]
                    )
                consistency_rows.append(row)

    b4_summary = []
    for lambda_index, lambda_value in enumerate(args.lambdas):
        chosen = anchor[:, lambda_index]
        chosen_seed = np.broadcast_to(
            chosen[:, None], b4_regret[:, :, lambda_index].shape
        )
        quality = gather_last(qualities, chosen_seed)
        latency = candidate_seconds[chosen_seed]
        b4_summary.append(
            {
                "lambda": lambda_value,
                "policy_regret": float(b4_regret[:, :, lambda_index].mean()),
                "realized_vbench5": float(quality.mean()),
                "realized_latency_sec": float(latency.mean()),
                "speedup_vs_native": float((native_seconds / latency).mean()),
                "harmful_stop_rate": float(
                    (b4_regret[:, :, lambda_index] > args.harm_epsilon).mean()
                ),
                "mean_chosen_step": float(candidate_steps[chosen].mean()),
            }
        )

    out_dir.mkdir(parents=True)
    artifacts = {
        "b4_ensemble_per_lambda": "b4_ensemble_per_lambda.csv",
        "headroom_macro_summary": "headroom_macro_summary.csv",
        "headroom_per_lambda": "headroom_per_lambda.csv",
        "offset_distribution": "offset_distribution.csv",
        "prompt_seed_consistency": "prompt_seed_consistency.csv",
    }
    write_csv(out_dir / artifacts["b4_ensemble_per_lambda"], b4_summary)
    write_csv(out_dir / artifacts["headroom_macro_summary"], macro_rows)
    write_csv(out_dir / artifacts["headroom_per_lambda"], per_lambda_rows)
    write_csv(out_dir / artifacts["offset_distribution"], offset_distribution)
    write_csv(out_dir / artifacts["prompt_seed_consistency"], consistency_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "b4_relative_one_sided_correction_headroom_before_verifier_training",
        "diagnostic_only": True,
        "formal_evidence": False,
        "evaluation_split": "validation",
        "test_accessed": False,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "source_candidate_steps": source_steps.tolist(),
        "candidate_steps": candidate_steps.tolist(),
        "lambdas": args.lambdas,
        "harm_epsilon": args.harm_epsilon,
        "prompt_count": len(prompt_ids),
        "validation_base_seeds": base_seeds.tolist(),
        "validation_seed_grouping": "actual_seed_minus_prompt_id",
        "b4_ensemble_size": len(b4_inputs),
        "b4_inputs": b4_inputs,
        "action_spaces": spaces,
        "selection_rules": [
            "per_seed_oracle",
            "mean_seed_common_action",
            "majority_positive_common_action",
            "all3_positive_common_action",
        ],
        "bootstrap": {
            "unit": "prompt_after_averaging_generation_seeds_and_lambdas",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "scope": "macro_rows_only",
        },
        "latency_profile": latency_profile,
        "artifacts": {
            name: {
                "path": filename,
                "sha256": base.sha256_file(out_dir / filename),
            }
            for name, filename in artifacts.items()
        },
        "limitations": [
            "Every headroom rule uses validation utility labels and is an oracle upper bound.",
            "Per-seed oracle may choose different actions for the same prompt and is not deployable.",
            "Common-action rules still use all three validation outcomes and are not learned policies.",
            "No verifier architecture or confidence threshold is selected by this audit.",
        ],
    }
    report_path = out_dir / "b4_preemption_headroom_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"B4 preemption headroom report: {report_path}")


if __name__ == "__main__":
    main()
