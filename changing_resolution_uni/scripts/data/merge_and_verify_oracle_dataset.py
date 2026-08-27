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
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="*",
        default=None,
        help="One or more dataset directories to merge (e.g. oracle_dataset_2k oracle_dataset_500_1000).",
    )
    parser.add_argument(
        "--parts_dir",
        type=str,
        default=None,
        help="Legacy option: Directory containing worker parts (_parts/part_00...).",
    )
    parser.add_argument("--out_root", type=str, required=True, help="Final master output root directory.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Expected total unique prompts.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024], help="Expected seed list per prompt.")
    parser.add_argument("--primary_lambda", type=float, default=0.01, help="Lambda for regret and utility calculation.")
    return parser.parse_args()


def extract_candidates_from_record(
    s_data: dict[str, Any], u_key: str, default_lambda: float = 0.01
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Extract candidate steps, utilities, and vbench5 scores from a trajectory record."""
    if "candidates" in s_data and s_data["candidates"]:
        cands = s_data["candidates"]
        cand_steps = [int(c["step"]) for c in cands]
        native_lat = float(s_data.get("native_latency_seconds", 0.0))
        if native_lat > 0.0 and all(
            float(c.get("latency_seconds", 0.0)) > 0.0 for c in cands
        ):
            utilities = np.array(
                [
                    float(c.get("vbench5", 0.0))
                    - default_lambda
                    * (float(c.get("latency_seconds", 0.0)) / native_lat)
                    for c in cands
                ],
                dtype=np.float64,
            )
        elif all(u_key in c.get("utilities", {}) for c in cands):
            utilities = np.array(
                [float(c["utilities"][u_key]) for c in cands], dtype=np.float64
            )
        else:
            utilities = np.array(
                [float(c.get("vbench5", 0.0)) for c in cands], dtype=np.float64
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
    out_root = Path(args.out_root).resolve()
    final_records_dir = out_root / "records"
    final_t5_dir = out_root / "t5_embeddings"
    final_records_dir.mkdir(parents=True, exist_ok=True)
    final_t5_dir.mkdir(parents=True, exist_ok=True)

    expected_seeds = set(args.seeds)
    expected_total_trajectories = args.total_prompts * len(expected_seeds)

    # Collect source json record files from input_dirs or parts_dir
    source_record_files: list[Path] = []
    source_dirs: list[Path] = []
    if args.input_dirs:
        for d in args.input_dirs:
            p = Path(d).resolve()
            if p.is_dir():
                source_dirs.append(p)
    elif args.parts_dir:
        p = Path(args.parts_dir).resolve()
        if p.is_dir():
            source_dirs.append(p)

    logger.info(f"Scanning {len(source_dirs)} source dataset directories: {[str(d) for d in source_dirs]}")

    for s_dir in source_dirs:
        # Check records/
        rec_dir = s_dir / "records"
        if rec_dir.is_dir():
            source_record_files.extend(rec_dir.glob("*.json"))
        # Check _parts/*/records/
        parts_dir = s_dir / "_parts"
        if parts_dir.is_dir():
            for p in parts_dir.glob("part_*"):
                if (p / "records").is_dir():
                    source_record_files.extend((p / "records").glob("*.json"))
        # If s_dir itself is _parts or part_*
        if s_dir.name.startswith("part_") and (s_dir / "records").is_dir():
            source_record_files.extend((s_dir / "records").glob("*.json"))
        for p in s_dir.glob("part_*"):
            if (p / "records").is_dir():
                source_record_files.extend((p / "records").glob("*.json"))

        # Copy T5 embeddings if present
        t5_src = s_dir / "t5_embeddings"
        if t5_src.is_dir():
            for npz in t5_src.glob("*.npz"):
                dest = final_t5_dir / npz.name
                temporary = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
                shutil.copy2(npz, temporary)
                os.replace(temporary, dest)

    logger.info(f"Found {len(source_record_files)} raw record files across sources.")

    all_records: dict[str, dict[str, Any]] = {}
    prompts_map: dict[int, dict[str, Any]] = {}

    for rec_file in source_record_files:
        try:
            data = json.loads(rec_file.read_text(encoding="utf-8"))
            p_id = int(data["prompt_id"])
            seed = int(data["seed"])
            sample_key = f"p{p_id:06d}_s{seed}"

            # Copy to final unified records directory
            dest_file = final_records_dir / f"{sample_key}.json"
            temporary = dest_file.with_name(
                f".{dest_file.name}.tmp.{os.getpid()}"
            )
            shutil.copy2(rec_file, temporary)
            os.replace(temporary, dest_file)

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
    logger.info(f"Successfully indexed {num_found} unique trajectory records across {len(prompts_map)} prompts.")
    seed_coverage_complete = all(
        set(prompt_info["seeds"]) == expected_seeds
        for prompt_info in prompts_map.values()
    )
    is_complete = (
        len(prompts_map) == args.total_prompts
        and num_found == expected_total_trajectories
        and seed_coverage_complete
    )

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
        "expected_seeds": sorted(expected_seeds),
        "seed_coverage_complete": seed_coverage_complete,
        "is_complete": is_complete,
        "primary_lambda": args.primary_lambda,
        "scientific_metrics": {
            "prompt_explainable_regret_R_prompt": round(mean_regret_global, 6),
            "mean_intra_prompt_step_std": round(mean_step_std, 3),
            "oracle_optimal_step_distribution": dict(sorted(step_histogram.items())),
            "mean_oracle_utility": round(total_oracle_utilities / max(num_found, 1), 5),
            "mean_prompt_theory_utility": round(total_prompt_utilities / max(num_found, 1), 5),
        },
    }
    manifest_temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{os.getpid()}"
    )
    manifest_temporary.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(manifest_temporary, manifest_path)

    print("\n" + "=" * 80)
    print(f" DATASET MERGE & VERIFICATION REPORT (Lambda = {args.primary_lambda})")
    print("=" * 80)
    print(f" Total Unique Prompts : {len(prompts_map)} / {args.total_prompts}")
    print(f" Total Trajectories    : {num_found} / {expected_total_trajectories}")
    print(f" Completion Status     : {'[COMPLETE]' if is_complete else '[PARTIAL / IN PROGRESS]'}")
    print("-" * 80)
    print(f" Prompt-Explainable Regret (R_prompt) : {mean_regret_global:.6f}")
    print(f" Mean Intra-Prompt Seed Std          : {mean_step_std:.3f} steps")
    print(f" Oracle Optimal Step Distribution    : {dict(sorted(step_histogram.items()))}")
    print("=" * 80 + "\n")
    if not is_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
