#!/usr/bin/env python3
"""
Merge shard parts from multi-GPU workers, verify completion consistency,
and compute scientific statistics (intra-prompt variance, optimal step distribution,
and Prompt-Explainable Regret R_prompt).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge_verify")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and verify oracle trajectory dataset parts.")
    parser.add_argument("--parts_dir", type=str, required=True, help="Directory containing worker parts (_parts/part_00...).")
    parser.add_argument("--out_root", type=str, required=True, help="Final master output root directory.")
    parser.add_argument("--total_prompts", type=int, default=2000, help="Expected total unique prompts.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024], help="Expected seed list per prompt.")
    parser.add_argument("--primary_lambda", type=float, default=0.05, help="Lambda for regret and utility calculation.")
    return parser.parse_args()


def extract_candidates_from_record(
    s_data: dict[str, Any], u_key: str, default_lambda: float = 0.05
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Extract candidate steps, utilities, and vbench5 scores from a trajectory record."""
    if "candidates" in s_data and s_data["candidates"]:
        cands = s_data["candidates"]
        cand_steps = [int(c["step"]) for c in cands]
        utilities = np.array(
            [float(c.get("utilities", {}).get(u_key, c.get("vbench5", 0.0))) for c in cands],
            dtype=np.float64,
        )
        vbench5 = np.array([float(c.get("vbench5", 0.0)) for c in cands], dtype=np.float64)
        return cand_steps, utilities, vbench5

    manifest = s_data.get("manifest", {})
    branches = manifest.get("branches", [])
    if branches:
        cand_steps = [int(b.get("candidate_step", 0)) for b in branches]
        vbench5 = np.array([float(b.get("vbench5", b.get("quality", 0.0))) for b in branches], dtype=np.float64)
        latencies = np.array(
            [float(b.get("latency_seconds", b.get("estimated_warm_pipeline_seconds", 100.0))) for b in branches],
            dtype=np.float64,
        )
        native_lat = (
            float(manifest.get("native_hr", {}).get("estimated_warm_pipeline_seconds", 180.0))
            if isinstance(manifest.get("native_hr"), dict)
            else 180.0
        )
        utilities = vbench5 - default_lambda * (latencies / max(native_lat, 1e-5))
        return cand_steps, utilities, vbench5

    return [], np.array([], dtype=np.float64), np.array([], dtype=np.float64)


def main() -> None:
    args = parse_args()
    parts_dir = Path(args.parts_dir).resolve()
    out_root = Path(args.out_root).resolve()
    final_records_dir = out_root / "records"
    final_records_dir.mkdir(parents=True, exist_ok=True)

    expected_seeds = set(args.seeds)
    expected_total_trajectories = args.total_prompts * len(expected_seeds)

    part_dirs = sorted([d for d in parts_dir.glob("part_*") if d.is_dir()])
    logger.info(f"Merging {len(part_dirs)} part directories from {parts_dir}")

    all_records: dict[str, dict[str, Any]] = {}
    prompts_map: dict[int, dict[str, Any]] = {}

    for p_dir in part_dirs:
        rec_dir = p_dir / "records"
        if not rec_dir.is_dir():
            continue
        for rec_file in rec_dir.glob("*.json"):
            try:
                data = json.loads(rec_file.read_text(encoding="utf-8"))
                p_id = int(data["prompt_id"])
                seed = int(data["seed"])
                sample_key = f"p{p_id:06d}_s{seed}"

                # Copy/move to final records dir
                dest_file = final_records_dir / f"{sample_key}.json"
                if not dest_file.exists():
                    shutil.copy2(rec_file, dest_file)

                all_records[sample_key] = data
                if p_id not in prompts_map:
                    prompts_map[p_id] = {
                        "prompt_id": p_id,
                        "prompt_text": data.get("prompt_text", ""),
                        "t5_embedding_path": data.get("t5_embedding_path"),
                        "seeds": {},
                    }
                prompts_map[p_id]["seeds"][seed] = data
            except Exception as e:
                logger.error(f"Error reading record {rec_file}: {e}")

    num_found = len(all_records)
    logger.info(f"Loaded {num_found} total trajectory records across {len(prompts_map)} prompts.")

    # ── Scientific Metric Computation: R_prompt & Variance ─────────────────────
    u_key = f"u_{args.primary_lambda:.2f}"

    prompt_stats: list[dict[str, Any]] = []
    total_regret_prompt = 0.0
    total_oracle_utilities = 0.0
    total_prompt_utilities = 0.0
    intra_prompt_step_stds: list[float] = []
    all_oracle_optimal_steps: list[int] = []
    evaluated_prompts_count = 0

    for p_id, p_info in sorted(prompts_map.items()):
        seeds_data = p_info["seeds"]
        if not seeds_data:
            continue

        # Get list of candidate steps across seeds
        cand_steps = []
        for s_data in seeds_data.values():
            c_steps, _, _ = extract_candidates_from_record(s_data, u_key, args.primary_lambda)
            if c_steps:
                cand_steps = c_steps
                break

        if not cand_steps:
            prompt_stats.append({
                "prompt_id": p_id,
                "prompt_text": p_info["prompt_text"],
                "seed_count": len(seeds_data),
                "seed_optimal_steps": [],
                "optimal_step_std": 0.0,
                "prompt_optimal_step": None,
                "mean_oracle_utility": 0.0,
                "prompt_policy_utility": 0.0,
                "prompt_explainable_regret": 0.0,
                "status": "pending_vbench_or_manifest_only",
            })
            continue

        K = len(cand_steps)
        mean_utility_per_step = np.zeros(K, dtype=np.float64)
        seed_optimal_steps = []
        seed_max_utilities = []
        valid_seed_count = 0

        for s_idx, (seed_val, s_data) in enumerate(seeds_data.items()):
            _, s_u, _ = extract_candidates_from_record(s_data, u_key, args.primary_lambda)
            if len(s_u) != K:
                continue

            mean_utility_per_step += s_u
            opt_idx = int(np.argmax(s_u))
            seed_optimal_steps.append(cand_steps[opt_idx])
            seed_max_utilities.append(float(s_u[opt_idx]))
            all_oracle_optimal_steps.append(cand_steps[opt_idx])
            valid_seed_count += 1

        if valid_seed_count == 0:
            continue

        mean_utility_per_step /= valid_seed_count

        # Best static choice for this prompt (Prompt-only theory ceiling)
        best_prompt_cand_idx = int(np.argmax(mean_utility_per_step))
        k_prompt = cand_steps[best_prompt_cand_idx]

        # Calculate prompt-explainable regret for this prompt
        prompt_regret_sum = 0.0
        for s_data in seeds_data.values():
            _, s_u, _ = extract_candidates_from_record(s_data, u_key, args.primary_lambda)
            if len(s_u) != K:
                continue
            u_oracle = np.max(s_u)
            u_prompt_choice = s_u[best_prompt_cand_idx]
            regret = u_oracle - u_prompt_choice
            prompt_regret_sum += regret
            total_regret_prompt += regret
            total_oracle_utilities += u_oracle
            total_prompt_utilities += u_prompt_choice

        avg_prompt_regret = prompt_regret_sum / valid_seed_count
        step_std = float(np.std(seed_optimal_steps)) if len(seed_optimal_steps) > 1 else 0.0
        intra_prompt_step_stds.append(step_std)
        evaluated_prompts_count += 1

        prompt_stats.append({
            "prompt_id": p_id,
            "prompt_text": p_info["prompt_text"],
            "seed_count": valid_seed_count,
            "seed_optimal_steps": seed_optimal_steps,
            "optimal_step_std": round(step_std, 3),
            "prompt_optimal_step": k_prompt,
            "mean_oracle_utility": round(float(np.mean(seed_max_utilities)), 5),
            "prompt_policy_utility": round(float(mean_utility_per_step[best_prompt_cand_idx]), 5),
            "prompt_explainable_regret": round(avg_prompt_regret, 6),
            "status": "evaluated",
        })

    # Global summary statistics
    mean_regret_global = total_regret_prompt / max(num_found, 1)
    mean_step_std = float(np.mean(intra_prompt_step_stds)) if intra_prompt_step_stds else 0.0
    step_histogram = Counter(all_oracle_optimal_steps)

    # Write summary CSV
    summary_csv = out_root / "dataset_summary.csv"
    if prompt_stats:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "prompt_id",
                "prompt_text",
                "seed_count",
                "seed_optimal_steps",
                "optimal_step_std",
                "prompt_optimal_step",
                "mean_oracle_utility",
                "prompt_policy_utility",
                "prompt_explainable_regret",
                "status",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prompt_stats)

    # Master manifest
    manifest_path = out_root / "dataset_manifest.json"
    manifest_payload = {
        "schema": "prompt_conditioned_oracle_dataset_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_prompts_found": len(prompts_map),
        "expected_prompts": args.total_prompts,
        "total_trajectories": num_found,
        "expected_trajectories": expected_total_trajectories,
        "is_complete": (num_found == expected_total_trajectories),
        "primary_lambda": args.primary_lambda,
        "scientific_metrics": {
            "prompt_explainable_regret_R_prompt": round(mean_regret_global, 6),
            "mean_intra_prompt_step_std": round(mean_step_std, 3),
            "oracle_optimal_step_distribution": dict(sorted(step_histogram.items())),
            "mean_oracle_utility": round(total_oracle_utilities / max(num_found, 1), 5),
            "mean_prompt_theory_utility": round(total_prompt_utilities / max(num_found, 1), 5),
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f" DATASET MERGE & VERIFICATION REPORT (Lambda = {args.primary_lambda})")
    print("=" * 80)
    print(f" Total Unique Prompts : {len(prompts_map)} / {args.total_prompts}")
    print(f" Total Trajectories    : {num_found} / {expected_total_trajectories}")
    print(f" Completion Status     : {'[COMPLETE]' if num_found == expected_total_trajectories else '[PARTIAL / IN PROGRESS]'}")
    print("-" * 80)
    print(f" Prompt-Explainable Regret (R_prompt) : {mean_regret_global:.6f}")
    print(f" Mean Intra-Prompt Seed Std          : {mean_step_std:.3f} steps")
    print(f" Oracle Optimal Step Distribution    : {dict(sorted(step_histogram.items()))}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
