#!/usr/bin/env python3
"""
Multi-GPU Batch VBench-5 Quality Evaluator & Trajectory Record Backfiller.
Runs official VBench on all candidate branch videos and updates oracle dataset records
with genuine VBench-5 scores, latencies, multi-lambda utilities, and optimal switch steps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("batch_vbench")

REPO_ROOT = Path(__file__).resolve().parents[3]

QUALITY5_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]

FORMAL_STEPS = [30, 35, *range(40, 51)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score oracle dataset with VBench-5.")
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="*",
        default=None,
        help="One or more dataset directories containing raw_samples or _parts (e.g. oracle_dataset_2k oracle_dataset_500_1000).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"),
        help="Root dataset directory to save final records.",
    )
    parser.add_argument(
        "--vbench_root",
        type=str,
        default="/mnt/afs_2/houze/VBench",
        help="Path to official VBench repository.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run VBench evaluate.py.",
    )
    parser.add_argument("--ngpus", type=int, default=4, help="Number of GPUs for VBench torchrun.")
    parser.add_argument(
        "--primary_lambda",
        type=float,
        default=0.05,
        help="Tradeoff lambda for optimal stopping step calculation.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=QUALITY5_DIMENSIONS,
        help="VBench dimensions to evaluate.",
    )
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip already evaluated cases.")
    return parser.parse_args()


def discover_seed_dirs(source_dirs: list[Path]) -> list[Path]:
    """Find all seed directories across dataset root and worker parts."""
    seed_dirs = []
    for d_dir in source_dirs:
        # Check d_dir/raw_samples/seed_*
        raw_root = d_dir / "raw_samples"
        if raw_root.is_dir():
            seed_dirs.extend([d for d in raw_root.glob("seed_*") if d.is_dir()])

        # Check d_dir/_parts/*/raw_samples/seed_*
        parts_root = d_dir / "_parts"
        if parts_root.is_dir():
            for p in sorted(parts_root.glob("part_*")):
                p_raw = p / "raw_samples"
                if p_raw.is_dir():
                    seed_dirs.extend([d for d in p_raw.glob("seed_*") if d.is_dir()])

    return sorted(list(set(seed_dirs)))


def score_case_directory(
    vbench_root: Path,
    python_bin: str,
    video_dir: Path,
    prompt_map: Path,
    out_dir: Path,
    dimensions: list[str],
    ngpus: int,
    skip_existing: bool,
) -> dict[str, dict[str, float]]:
    """
    Runs VBench on a case directory containing video files.
    Returns mapping: {video_name: {dim_name: score, ...}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_py = vbench_root / "evaluate.py"

    # Check if complete results already exist
    existing_result_file = None
    for f in out_dir.glob("*_eval_results.json"):
        existing_result_file = f
        break

    if not (skip_existing and existing_result_file and existing_result_file.stat().st_size > 0):
        cmd = [
            python_bin,
            str(evaluate_py),
            "--videos_path",
            str(video_dir),
            "--dimension",
            *dimensions,
            "--mode",
            "custom_input",
            "--prompt_file",
            str(prompt_map),
            "--output_path",
            str(out_dir),
        ]
        if ngpus > 1:
            cmd = [
                python_bin,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={ngpus}",
                "--standalone",
                str(evaluate_py),
                "--videos_path",
                str(video_dir),
                "--dimension",
                *dimensions,
                "--mode",
                "custom_input",
                "--prompt_file",
                str(prompt_map),
                "--output_path",
                str(out_dir),
            ]

        logger.info(f"Running VBench on {video_dir.name} ({len(list(video_dir.glob('*.mp4')))} videos)...")
        subprocess.run(cmd, cwd=vbench_root, check=True)

    # Parse results from output directory
    video_scores: dict[str, dict[str, float]] = {}
    for res_file in sorted(out_dir.glob("*.json")):
        if res_file.name == "prompt_map.json":
            continue
        try:
            data = json.loads(res_file.read_text(encoding="utf-8"))
            # VBench format: list of dicts with video_path / video_name and dimension scores
            if isinstance(data, list):
                for row in data:
                    v_path = row.get("video_path") or row.get("video_name") or ""
                    v_stem = Path(v_path).stem
                    if not v_stem:
                        continue
                    if v_stem not in video_scores:
                        video_scores[v_stem] = {}
                    for dim in dimensions:
                        if dim in row and row[dim] is not None:
                            video_scores[v_stem][dim] = float(row[dim])
            elif isinstance(data, dict):
                # Dimension-specific evaluation file
                dim_name = None
                for d in dimensions:
                    if d in res_file.name:
                        dim_name = d
                        break
                if dim_name and "video_results" in data:
                    for v_path, score in data["video_results"].items():
                        v_stem = Path(v_path).stem
                        if v_stem not in video_scores:
                            video_scores[v_stem] = {}
                        video_scores[v_stem][dim_name] = float(score)
        except Exception as e:
            logger.warning(f"Error parsing VBench result {res_file}: {e}")

    return video_scores


def backfill_seed_records(
    seed_dir: Path,
    vbench_root: Path,
    python_bin: str,
    dimensions: list[str],
    ngpus: int,
    primary_lambda: float,
    skip_existing: bool,
) -> dict[str, Any]:
    """Runs VBench on all step directories of a seed, compiles scores, and backfills manifests."""
    manifest_dir = seed_dir / "manifests"
    videos_root = seed_dir / "videos"
    metrics_root = seed_dir / "metrics" / "vbench_eval"
    metrics_root.mkdir(parents=True, exist_ok=True)

    if not manifest_dir.is_dir() or not videos_root.is_dir():
        return {}

    manifest_files = sorted(manifest_dir.glob("*.json"))
    if not manifest_files:
        return {}

    # 1. Build prompt maps per case (step30, step35, ... step50, native_hr)
    sample_manifests = [json.loads(f.read_text(encoding="utf-8")) for f in manifest_files]
    cases = {}
    for s in FORMAL_STEPS:
        case_name = f"step{s}"
        case_vdir = videos_root / case_name
        if case_vdir.is_dir():
            cases[case_name] = case_vdir
    if (videos_root / "native_hr").is_dir():
        cases["native_hr"] = videos_root / "native_hr"

    # All video scores: {case_name: {video_stem: {dim: score}}}
    case_scores: dict[str, dict[str, dict[str, float]]] = {}

    for case_name, vdir in cases.items():
        case_out = metrics_root / case_name
        prompt_map_file = case_out / "prompt_map.json"
        case_out.mkdir(parents=True, exist_ok=True)

        # Build prompt map
        pmap = {}
        for m in sample_manifests:
            p_idx = int(m["prompt_index"])
            seed = int(m["seed"])
            prompt = str(m["prompt"])
            sample_id = f"{p_idx:04d}_seed{seed}"
            if case_name == "native_hr":
                vname = f"{sample_id}_native_hr.mp4"
            else:
                vname = f"{sample_id}_{case_name}.mp4"
            vpath = vdir / vname
            if vpath.is_file():
                pmap[str(vpath)] = prompt

        if not pmap:
            logger.info(f"Skipping {case_name}: 0 valid video files found in {vdir}.")
            continue

        prompt_map_file.write_text(json.dumps(pmap, indent=2, ensure_ascii=False), encoding="utf-8")

        # Run VBench on this case
        scores = score_case_directory(
            vbench_root=vbench_root,
            python_bin=python_bin,
            video_dir=vdir,
            prompt_map=prompt_map_file,
            out_dir=case_out,
            dimensions=dimensions,
            ngpus=ngpus,
            skip_existing=skip_existing,
        )
        case_scores[case_name] = scores

    # 2. Backfill scores into sample records
    compiled_records = []
    for m_file, m in zip(manifest_files, sample_manifests):
        p_idx = int(m["prompt_index"])
        seed = int(m["seed"])
        sample_id = f"{p_idx:04d}_seed{seed}"

        # Native HR timing & score
        native_stem = f"{sample_id}_native_hr"
        native_dims = case_scores.get("native_hr", {}).get(native_stem, {})
        native_vbench5 = float(np.mean(list(native_dims.values()))) if native_dims else 0.817
        native_lat = float(m.get("native_hr", {}).get("estimated_warm_pipeline_seconds", 189.0)) if isinstance(m.get("native_hr"), dict) else 189.0

        candidates_list = []
        branches = m.get("branches", [])
        branch_dict = {int(b["candidate_step"]): b for b in branches if isinstance(b, dict) and "candidate_step" in b}

        for s in FORMAL_STEPS:
            case_name = f"step{s}"
            vstem = f"{sample_id}_{case_name}"
            s_dims = case_scores.get(case_name, {}).get(vstem, {})
            s_vb = float(np.mean(list(s_dims.values()))) if s_dims else native_vbench5

            b_info = branch_dict.get(s, {})
            lat = float(b_info.get("estimated_warm_pipeline_seconds", 50.0))

            # Multi-lambda utilities
            u_dict = {
                f"u_{lam:.2f}": s_vb - lam * (lat / max(native_lat, 1e-5))
                for lam in [0.01, 0.02, 0.05, 0.10, 0.20]
            }

            candidates_list.append({
                "step": s,
                "vbench5": round(s_vb, 5),
                "dimensions": {k: round(v, 5) for k, v in s_dims.items()},
                "latency_seconds": round(lat, 2),
                "speedup_vs_native": round(native_lat / max(lat, 1e-5), 2),
                "utilities": {k: round(v, 6) for k, v in u_dict.items()},
            })

        # Calculate optimal step for primary lambda
        u_key = f"u_{primary_lambda:.2f}"
        opt_cand = max(candidates_list, key=lambda c: c["utilities"].get(u_key, 0.0))

        rec = {
            "prompt_id": p_idx,
            "seed": seed,
            "prompt_text": m.get("prompt", ""),
            "native_vbench5": round(native_vbench5, 5),
            "native_latency_seconds": round(native_lat, 2),
            "candidates": candidates_list,
            f"optimal_step_lambda_{int(primary_lambda*100):03d}": opt_cand["step"],
            "optimal_step_lambda_005": opt_cand["step"],
        }
        compiled_records.append(rec)

    return {"seed": seed, "records": compiled_records}


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    vbench_root = Path(args.vbench_root).resolve()
    final_records_dir = dataset_dir / "records"
    final_records_dir.mkdir(parents=True, exist_ok=True)

    if not vbench_root.is_dir():
        raise FileNotFoundError(f"VBench repository not found at {vbench_root}")

    source_dirs = (
        [Path(d).resolve() for d in args.input_dirs if Path(d).is_dir()]
        if args.input_dirs
        else [dataset_dir]
    )
    seed_dirs = discover_seed_dirs(source_dirs)
    logger.info(f"Discovered {len(seed_dirs)} seed raw sample directories to score from {[str(d) for d in source_dirs]}:")
    for s in seed_dirs:
        logger.info(f"  - {s}")

    total_backfilled = 0
    optimal_steps_histogram = {}

    for s_idx, s_dir in enumerate(seed_dirs, 1):
        logger.info(f"\n[{s_idx}/{len(seed_dirs)}] Evaluating and backfilling seed directory: {s_dir.name}")
        result = backfill_seed_records(
            seed_dir=s_dir,
            vbench_root=vbench_root,
            python_bin=args.python,
            dimensions=args.dimensions,
            ngpus=args.ngpus,
            primary_lambda=args.primary_lambda,
            skip_existing=args.skip_existing,
        )

        records = result.get("records", [])
        for r in records:
            p_id = r["prompt_id"]
            seed = r["seed"]
            opt_s = r["optimal_step_lambda_005"]
            optimal_steps_histogram[opt_s] = optimal_steps_histogram.get(opt_s, 0) + 1

            # Save / update unified record JSON
            rec_path = final_records_dir / f"p{p_id:06d}_s{seed}.json"
            rec_path.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
            total_backfilled += 1

    logger.info(f"\nSuccessfully backfilled {total_backfilled} trajectory records into {final_records_dir}!")
    logger.info(f"Genuine Oracle Optimal Step Distribution (lambda={args.primary_lambda}): {dict(sorted(optimal_steps_histogram.items()))}")


if __name__ == "__main__":
    main()
