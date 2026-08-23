#!/usr/bin/env python3
"""
Worker pipeline to generate oracle multi-step resolution switching trajectories,
evaluate VBench-5 / timing metrics, and construct lightweight dataset records.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("oracle_dataset")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORMAL_STEPS = [30, 35, *range(40, 51)]
QUALITY5_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build oracle trajectory dataset for prompt/seed slices.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Prompts file path.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory.")
    parser.add_argument("--t5_embed_dir", type=str, default=None, help="Directory containing pre-extracted T5 embeddings.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024], help="Seeds per prompt.")
    parser.add_argument("--prompt_offset", type=int, default=0, help="Offset index into prompts file.")
    parser.add_argument("--limit", type=int, default=None, help="Number of prompts to process.")
    parser.add_argument("--candidate_steps", type=int, nargs="+", default=FORMAL_STEPS, help="Candidate switch steps.")
    parser.add_argument("--infer_steps", type=int, default=50, help="Total diffusion inference steps.")
    parser.add_argument("--lr_h", type=int, default=368, help="Low-res height.")
    parser.add_argument("--lr_w", type=int, default=640, help="Low-res width.")
    parser.add_argument("--hr_h", type=int, default=720, help="High-res height.")
    parser.add_argument("--hr_w", type=int, default=1248, help="High-res width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Video frame count.")
    parser.add_argument("--lambda_list", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20], help="Utility lambdas.")
    parser.add_argument("--primary_lambda", type=float, default=0.01, help="Primary lambda for default decision labeling.")
    parser.add_argument("--lightx2v_repo", type=str, default="/mnt/afs_2/houze/LightX2V")
    parser.add_argument("--model_root", type=str, default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--stage2_checkpoint", type=str, default=None)
    parser.add_argument("--stage2_train_config", type=str, default=None)
    parser.add_argument("--vbench_root", type=str, default=None, help="Path to VBench repo or installation.")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true", help="Simulate video generation and metrics for pipeline smoke testing.")
    parser.add_argument("--clean_videos_after_eval", action="store_true", help="Delete heavy raw MP4s after extracting metrics.")
    return parser.parse_args()


def load_prompt_slice(file_path: Path, offset: int, limit: int | None) -> list[tuple[int, str]]:
    prompts = []
    if file_path.suffix.lower() == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(str(item["prompt"]).strip())
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(line)
    indexed = list(enumerate(prompts))
    selected = indexed[offset:]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def simulate_oracle_evaluation(
    prompt_id: int,
    prompt_text: str,
    seed: int,
    candidate_steps: list[int],
    infer_steps: int,
    lambdas: list[float],
    primary_lambda: float,
) -> dict[str, Any]:
    """Dry-run synthetic trajectory generator for testing sharding and aggregation."""
    rng = np.random.RandomState((prompt_id * 10007 + seed) % (2**31 - 1))
    
    # Base native quality
    native_q5 = float(np.clip(0.81 + rng.normal(0, 0.02), 0.70, 0.95))
    native_latency = 180.0 + rng.normal(0, 5.0)

    candidates = []
    for step in candidate_steps:
        # Quality generally increases with more steps, leveling off around step 45-48
        progress = step / float(infer_steps)
        q_gain = 0.81 * (1.0 - np.exp(-4.5 * progress))
        noise = float(rng.normal(0, 0.003))
        q5 = float(np.clip(q_gain + noise, 0.65, 0.85))
        
        # Branch estimate latency: LR time + HR time
        lr_time = 0.8 * step
        hr_time = 3.6 * (infer_steps - step)
        latency = lr_time + hr_time
        
        vbench_dims = {
            "subject_consistency": float(np.clip(q5 + 0.12 + rng.normal(0, 0.01), 0.7, 1.0)),
            "background_consistency": float(np.clip(q5 + 0.14 + rng.normal(0, 0.01), 0.7, 1.0)),
            "motion_smoothness": float(np.clip(q5 + 0.16 + rng.normal(0, 0.008), 0.8, 1.0)),
            "aesthetic_quality": float(np.clip(q5 - 0.20 + rng.normal(0, 0.02), 0.4, 0.8)),
            "imaging_quality": float(np.clip(q5 - 0.18 + rng.normal(0, 0.02), 0.4, 0.8)),
        }
        
        utilities = {}
        for lam in lambdas:
            u = q5 - lam * (latency / native_latency)
            utilities[f"u_{lam:.2f}"] = float(u)
            
        candidates.append({
            "step": step,
            "vbench5": q5,
            "vbench_details": vbench_dims,
            "latency_seconds": latency,
            "speedup_vs_native": native_latency / latency,
            "utilities": utilities,
        })

    # Find optimal for primary lambda (e.g. 0.05)
    primary_key = f"u_{primary_lambda:.2f}"
    best_primary = -1e9
    optimal_step = candidate_steps[-1]
    for c in candidates:
        if c["utilities"][primary_key] > best_primary:
            best_primary = c["utilities"][primary_key]
            optimal_step = c["step"]

    return {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "seed": seed,
        "native": {
            "vbench5": native_q5,
            "latency_seconds": native_latency,
            "step": infer_steps,
        },
        "candidates": candidates,
        f"optimal_step_lambda_{int(primary_lambda * 100):03d}": optimal_step,
    }


def load_or_compute_batch_metrics(
    args: argparse.Namespace,
    out_root: Path,
    seed: int,
) -> dict[str, Any]:
    """Run VBench/metrics on the batch seed directory if available and load results."""
    seed_dir = out_root / "raw_samples" / f"seed_{seed}"
    metrics_json = seed_dir / "metrics" / "oracle_metrics.json"
    
    if metrics_json.is_file():
        try:
            return json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    # If VBench metrics haven't been computed yet, run batch metric evaluation once for all prompts
    metrics_script = REPO_ROOT / "changing_resolution" / "scripts" / "eval" / "run_clean_360p_stage2_oracle_metrics.py"
    if metrics_script.is_file() and args.vbench_root and (seed_dir / "protocol.json").is_file():
        logger.info(f"Running batch VBench metrics evaluation for seed={seed} (Single VBench model load)...")
        cmd_metrics = [
            args.python_bin,
            str(metrics_script),
            "--oracle-root", str(seed_dir),
            "--action", "all",
            "--vbench-root", args.vbench_root,
            "--latency-lambda", str(args.primary_lambda),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
        res_m = subprocess.run(cmd_metrics, env=env, capture_output=True, text=True)
        if res_m.returncode != 0:
            logger.warning(f"VBench eval note for seed={seed}: {res_m.stderr}")
            
        if metrics_json.is_file():
            return json.loads(metrics_json.read_text(encoding="utf-8"))

    return {}


def parse_sample_manifest(
    args: argparse.Namespace,
    out_root: Path,
    prompt_id: int,
    prompt_text: str,
    seed: int,
    batch_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Extract candidate metrics for a specific prompt/seed from batch outputs."""
    seed_dir = out_root / "raw_samples" / f"seed_{seed}"
    manifest_file = seed_dir / "manifests" / f"{prompt_id:04d}_seed{seed}.json"

    # Check if candidate_per_sample in oracle_metrics.json has our records
    if "candidate_per_sample" in batch_metrics:
        matching_cands = [
            r for r in batch_metrics["candidate_per_sample"]
            if int(r.get("prompt_index", -1)) == prompt_id
        ]
        if matching_cands:
            cands_list = []
            for r in matching_cands:
                cands_list.append({
                    "step": int(r["candidate_step"]),
                    "vbench5": float(r.get("vbench5", 0.0)),
                    "latency_seconds": float(r.get("latency_seconds", 0.0)),
                    "speedup_vs_native": float(r.get("speedup_vs_native", 1.0)),
                    "utilities": {
                        f"u_{lam:.2f}": float(r.get("vbench5", 0.0)) - lam * (float(r.get("latency_seconds", 1.0)) / max(float(r.get("native_latency_seconds", 180.0)), 1e-5))
                        for lam in args.lambda_list
                    },
                })
            opt_step = max(cands_list, key=lambda c: c["utilities"].get(f"u_{args.primary_lambda:.2f}", 0.0))["step"]
            return {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "seed": seed,
                "candidates": cands_list,
                f"optimal_step_lambda_{int(args.primary_lambda * 100):03d}": opt_step,
            }

    # Fallback to reading single manifest if available
    if manifest_file.is_file():
        m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        return {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "seed": seed,
            "manifest": m_data,
            f"optimal_step_lambda_{int(args.primary_lambda * 100):03d}": args.candidate_steps[-1],
        }

    return {"prompt_id": prompt_id, "prompt_text": prompt_text, "seed": seed, "status": "manifest_not_found"}


def main() -> None:
    args = parse_args()
    args.lambda_list = sorted(set([*args.lambda_list, float(args.primary_lambda)]))
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    records_dir = out_root / "records"
    records_dir.mkdir(exist_ok=True)

    prompts = load_prompt_slice(Path(args.prompts_file), offset=args.prompt_offset, limit=args.limit)
    logger.info(f"Worker start: {len(prompts)} prompts (offset={args.prompt_offset}), seeds={args.seeds}, dry_run={args.dry_run}")

    summary_rows: list[dict[str, Any]] = []
    total_trajectories = len(prompts) * len(args.seeds)
    done_count = 0
    t0 = time.perf_counter()

    # Preload batch metrics for each seed if available
    batch_metrics_by_seed: dict[int, dict[str, Any]] = {}
    if not args.dry_run:
        for seed in args.seeds:
            batch_metrics_by_seed[seed] = load_or_compute_batch_metrics(args, out_root, seed)

    for prompt_id, prompt_text in prompts:
        t5_pointer = None
        if args.t5_embed_dir:
            t5_file = Path(args.t5_embed_dir) / f"prompt_{prompt_id:06d}.npz"
            if t5_file.is_file():
                t5_pointer = str(t5_file)

        for seed in args.seeds:
            sample_key = f"p{prompt_id:06d}_s{seed}"
            record_json = records_dir / f"{sample_key}.json"

            if args.skip_existing and record_json.is_file():
                logger.info(f"Skipping existing record: {sample_key}")
                existing_record = json.loads(record_json.read_text(encoding="utf-8"))
                summary_rows.append({
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "prompt_text": prompt_text,
                    "optimal_step": existing_record.get(
                        f"optimal_step_lambda_{int(args.primary_lambda * 100):03d}"
                    ),
                    "record_file": str(record_json),
                })
                done_count += 1
                continue

            if args.dry_run:
                traj_record = simulate_oracle_evaluation(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    seed=seed,
                    candidate_steps=args.candidate_steps,
                    infer_steps=args.infer_steps,
                    lambdas=args.lambda_list,
                    primary_lambda=args.primary_lambda,
                )
            else:
                b_metrics = batch_metrics_by_seed.get(seed, {})
                traj_record = parse_sample_manifest(
                    args=args,
                    out_root=out_root,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    seed=seed,
                    batch_metrics=b_metrics,
                )

            traj_record["t5_embedding_path"] = t5_pointer
            traj_record["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            
            # Save trajectory record
            record_json.write_text(json.dumps(traj_record, ensure_ascii=False, indent=2), encoding="utf-8")

            if args.clean_videos_after_eval and not args.dry_run:
                # Remove raw videos to save disk space
                seed_dir = out_root / "raw_samples" / f"seed_{seed}"
                if seed_dir.exists():
                    for v_dir in seed_dir.glob("videos/step*"):
                        for f in v_dir.glob(f"{prompt_id:04d}_*.mp4"):
                            f.unlink(missing_ok=True)

            summary_rows.append({
                "prompt_id": prompt_id,
                "seed": seed,
                "prompt_text": prompt_text,
                "optimal_step": traj_record.get(
                    f"optimal_step_lambda_{int(args.primary_lambda * 100):03d}"
                ),
                "record_file": str(record_json),
            })
            done_count += 1
            elapsed = time.perf_counter() - t0
            avg_per_traj = elapsed / max(done_count, 1)
            eta_sec = avg_per_traj * (total_trajectories - done_count)
            logger.info(f"[{done_count}/{total_trajectories}] Done {sample_key} | ETA: {eta_sec/60.0:.1f} min")

    # Write worker part summary
    part_summary_csv = out_root / f"summary_offset_{args.prompt_offset:06d}.csv"
    if summary_rows:
        with open(part_summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    logger.info(f"Worker finished! {done_count} trajectory records saved in {records_dir}")


if __name__ == "__main__":
    main()
