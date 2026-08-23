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

import numpy as np

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
UTILITY_LAMBDAS = [0.01, 0.02, 0.05, 0.10, 0.20]


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
        "--expected_prompts",
        type=int,
        default=None,
        help="Require this exact number of prompt IDs before writing scored records.",
    )
    parser.add_argument(
        "--expected_seeds",
        type=int,
        nargs="+",
        default=[42, 100, 2024],
        help="Base seeds used by the generator.",
    )
    parser.add_argument(
        "--seed_policy",
        choices=["fixed", "prompt_offset"],
        default="prompt_offset",
        help="Expected stored seed rule; generation uses base_seed + prompt_id.",
    )
    parser.add_argument(
        "--primary_lambda",
        type=float,
        default=0.01,
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


def warmup_vbench_cache(python_bin: str, vbench_root: Path) -> None:
    """Pre-downloads torch.hub dependencies (DINO) and OpenAI CLIP models in a single process to prevent multi-GPU race condition and network timeout."""
    hub_dir = Path.home() / ".cache" / "torch" / "hub"
    clip_dir = Path.home() / ".cache" / "clip"
    dino_dir = hub_dir / "facebookresearch_dino_main"

    # If DINO is corrupted, clean it
    if dino_dir.is_dir() and not (dino_dir / "vision_transformer.py").is_file():
        logger.warning(f"Cleaning corrupted DINO torch.hub directory at {dino_dir}")
        shutil.rmtree(dino_dir, ignore_errors=True)
    if hub_dir.is_dir():
        for p in hub_dir.glob("facebookresearch-dino-*"):
            shutil.rmtree(p, ignore_errors=True)
        for z in hub_dir.glob("main.zip*"):
            z.unlink(missing_ok=True)

    logger.info("Pre-warming VBench dependencies (DINO & CLIP) in single-process mode...")
    prewarm_script = """
import time
import torch
import clip

# 1. Warm up DINO
for attempt in range(5):
    try:
        print("[Warmup] Loading DINO...")
        torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        print("[Warmup] DINO loaded successfully.")
        break
    except Exception as e:
        print(f"[Warmup] DINO load attempt {attempt+1} failed: {e}. Retrying in 3s...")
        time.sleep(3)

# 2. Warm up CLIP models (ViT-B/32 and ViT-L/14 used by VBench)
for m_name in ['ViT-B/32', 'ViT-L/14']:
    for attempt in range(5):
        try:
            print(f"[Warmup] Pre-downloading CLIP {m_name}...")
            clip.load(m_name, device='cpu')
            print(f"[Warmup] CLIP {m_name} ready.")
            break
        except Exception as e:
            print(f"[Warmup] CLIP {m_name} attempt {attempt+1} failed: {e}. Retrying in 3s...")
            time.sleep(3)
"""
    try:
        subprocess.run(
            [python_bin, "-c", prewarm_script],
            check=False,
            cwd=vbench_root,
        )
    except Exception as e:
        logger.warning(f"Prewarm warning: {e}")


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
    for f in out_dir.glob("*.json"):
        if f.name != "prompt_map.json" and ("eval_results" in f.name or "full_info" in f.name) and f.stat().st_size > 0:
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
            if isinstance(data, dict):
                # Format A: Official VBench output {dimension: [overall_score, [{video_path: ..., video_results: ...}]]}
                for dim in dimensions:
                    if dim in data:
                        payload = data[dim]
                        if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list):
                            for row in payload[1]:
                                if isinstance(row, dict):
                                    v_path = row.get("video_path") or row.get("video_name") or ""
                                    v_stem = Path(v_path).stem
                                    score = row.get("video_results")
                                    if score is not None and v_stem:
                                        val = float(score)
                                        if dim == "imaging_quality" and val > 1.0:
                                            val /= 100.0
                                        video_scores.setdefault(v_stem, {})[dim] = val
                        elif isinstance(payload, dict):
                            for v_path, score in payload.items():
                                v_stem = Path(v_path).stem
                                if score is not None and v_stem:
                                    val = float(score)
                                    if dim == "imaging_quality" and val > 1.0:
                                        val /= 100.0
                                    video_scores.setdefault(v_stem, {})[dim] = val

                # Format B: Dimension-specific evaluation file with video_results dict
                for dim in dimensions:
                    if dim in res_file.name and "video_results" in data:
                        for v_path, score in data["video_results"].items():
                            v_stem = Path(v_path).stem
                            if score is not None and v_stem:
                                val = float(score)
                                if dim == "imaging_quality" and val > 1.0:
                                    val /= 100.0
                                video_scores.setdefault(v_stem, {})[dim] = val

            elif isinstance(data, list):
                # Format C: Flat list of records [{video_path: ..., subject_consistency: ...}]
                for row in data:
                    if isinstance(row, dict):
                        v_path = row.get("video_path") or row.get("video_name") or ""
                        v_stem = Path(v_path).stem
                        if not v_stem:
                            continue
                        for dim in dimensions:
                            if dim in row and row[dim] is not None:
                                val = float(row[dim])
                                if dim == "imaging_quality" and val > 1.0:
                                    val /= 100.0
                                video_scores.setdefault(v_stem, {})[dim] = val
        except Exception as e:
            logger.warning(f"Error parsing VBench result {res_file}: {e}")

    logger.info(f"Successfully parsed VBench-5 metrics for {len(video_scores)} videos in {out_dir.name}")
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
    utility_lambdas = sorted(set([*UTILITY_LAMBDAS, float(primary_lambda)]))
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
    validation_errors: list[str] = []
    for m_file, m in zip(manifest_files, sample_manifests):
        p_idx = int(m["prompt_index"])
        seed = int(m["seed"])
        sample_id = f"{p_idx:04d}_seed{seed}"

        # Native HR timing & score
        native_stem = f"{sample_id}_native_hr"
        native_dims = case_scores.get("native_hr", {}).get(native_stem, {})
        missing_native_dims = [dim for dim in dimensions if dim not in native_dims]
        native_info = m.get("native_hr") if isinstance(m.get("native_hr"), dict) else {}
        native_lat = float(native_info.get("warm_pipeline_seconds", 0.0))
        if missing_native_dims:
            validation_errors.append(
                f"{sample_id}: native_hr missing VBench dimensions {missing_native_dims}"
            )
        if not np.isfinite(native_lat) or native_lat <= 0.0:
            validation_errors.append(
                f"{sample_id}: native_hr missing positive warm_pipeline_seconds"
            )
        if missing_native_dims or not np.isfinite(native_lat) or native_lat <= 0.0:
            continue
        native_vbench5 = float(np.mean([native_dims[dim] for dim in dimensions]))

        candidates_list = []
        branches = m.get("branches", [])
        branch_dict = {int(b["candidate_step"]): b for b in branches if isinstance(b, dict) and "candidate_step" in b}

        for s in FORMAL_STEPS:
            case_name = f"step{s}"
            vstem = f"{sample_id}_{case_name}"
            s_dims = case_scores.get(case_name, {}).get(vstem, {})
            missing_dims = [dim for dim in dimensions if dim not in s_dims]
            if missing_dims:
                validation_errors.append(
                    f"{sample_id}: step {s} missing VBench dimensions {missing_dims}"
                )
                continue
            s_vb = float(np.mean([s_dims[dim] for dim in dimensions]))

            b_info = branch_dict.get(s, {})
            if "warm_pipeline_seconds" in b_info:
                latency_source = "warm_pipeline_seconds"
                lat = float(b_info[latency_source])
            elif "estimated_warm_pipeline_seconds" in b_info:
                latency_source = "estimated_warm_pipeline_seconds"
                lat = float(b_info[latency_source])
            else:
                latency_source = "missing"
                lat = 0.0
            if not np.isfinite(lat) or lat <= 0.0:
                validation_errors.append(
                    f"{sample_id}: step {s} missing positive branch latency"
                )
                continue

            # Multi-lambda utilities
            u_dict = {
                f"u_{lam:.2f}": s_vb - lam * (lat / max(native_lat, 1e-5))
                for lam in utility_lambdas
            }

            candidates_list.append({
                "step": s,
                "vbench5": round(s_vb, 5),
                "dimensions": {k: round(v, 5) for k, v in s_dims.items()},
                "latency_seconds": round(lat, 2),
                "latency_source": latency_source,
                "speedup_vs_native": round(native_lat / max(lat, 1e-5), 2),
                "utilities": {k: round(v, 6) for k, v in u_dict.items()},
            })

        if len(candidates_list) != len(FORMAL_STEPS):
            continue

        # Calculate optimal steps for every stored lambda.
        u_key = f"u_{primary_lambda:.2f}"
        opt_cand = max(candidates_list, key=lambda c: c["utilities"].get(u_key, 0.0))

        rec = {
            "prompt_id": p_idx,
            "seed": seed,
            "prompt_text": m.get("prompt", ""),
            "native_vbench5": round(native_vbench5, 5),
            "native_latency_seconds": round(native_lat, 2),
            "native_dimensions": {k: round(native_dims[k], 5) for k in dimensions},
            "native_latency_source": "warm_pipeline_seconds",
            "candidates": candidates_list,
            f"optimal_step_lambda_{int(primary_lambda*100):03d}": opt_cand["step"],
        }
        for lam in utility_lambdas:
            key = f"u_{lam:.2f}"
            best = max(candidates_list, key=lambda c: c["utilities"][key])
            rec[f"optimal_step_lambda_{int(lam * 100):03d}"] = best["step"]
        compiled_records.append(rec)

    if validation_errors:
        preview = "\n".join(f"  - {item}" for item in validation_errors[:50])
        suffix = (
            ""
            if len(validation_errors) <= 50
            else f"\n  ... and {len(validation_errors) - 50} more"
        )
        raise RuntimeError(
            "Refusing to backfill incomplete oracle scores or timings:\n"
            f"{preview}{suffix}"
        )

    return {"seed": seed, "records": compiled_records}


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    vbench_root = Path(args.vbench_root).resolve()
    final_records_dir = dataset_dir / "records"
    final_records_dir.mkdir(parents=True, exist_ok=True)

    if not vbench_root.is_dir():
        raise FileNotFoundError(f"VBench repository not found at {vbench_root}")

    # Pre-warm dependencies to prevent multi-GPU torch.hub race conditions
    warmup_vbench_cache(args.python, vbench_root)

    source_dirs = (
        [Path(d).resolve() for d in args.input_dirs if Path(d).is_dir()]
        if args.input_dirs
        else [dataset_dir]
    )
    seed_dirs = discover_seed_dirs(source_dirs)
    if not seed_dirs:
        raise FileNotFoundError(
            f"No raw_samples/seed_* directories found under {[str(d) for d in source_dirs]}"
        )
    logger.info(f"Discovered {len(seed_dirs)} seed raw sample directories to score from {[str(d) for d in source_dirs]}:")
    for s in seed_dirs:
        logger.info(f"  - {s}")

    optimal_steps_histogram = {}
    compiled_by_key: dict[tuple[int, int], tuple[dict[str, Any], Path]] = {}

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
            record_key = (int(p_id), int(seed))
            if record_key in compiled_by_key:
                previous_record, previous_source = compiled_by_key[record_key]
                if previous_record != r:
                    raise RuntimeError(
                        f"Conflicting duplicate oracle record {record_key} from "
                        f"{previous_source} and {s_dir}"
                    )
                logger.info(
                    f"Skipping identical duplicate oracle record {record_key} from {s_dir}"
                )
                continue
            compiled_by_key[record_key] = (r, s_dir)
    if not compiled_by_key:
        raise RuntimeError("No oracle records were backfilled; refusing to continue")

    expected_base_seeds = {int(seed) for seed in args.expected_seeds}
    records_by_prompt: dict[int, set[int]] = {}
    for prompt_id, seed in compiled_by_key:
        records_by_prompt.setdefault(prompt_id, set()).add(seed)
    seed_errors = {}
    for prompt_id, seeds in records_by_prompt.items():
        expected_for_prompt = (
            {seed + prompt_id for seed in expected_base_seeds}
            if args.seed_policy == "prompt_offset"
            else expected_base_seeds
        )
        if seeds != expected_for_prompt:
            seed_errors[prompt_id] = {
                "observed": sorted(seeds),
                "expected": sorted(expected_for_prompt),
            }
    if seed_errors:
        preview = list(sorted(seed_errors.items()))[:30]
        raise RuntimeError(
            f"Seed coverage mismatch under {args.seed_policy} policy, "
            f"examples={preview}"
        )
    if args.expected_prompts is not None and len(records_by_prompt) != args.expected_prompts:
        raise RuntimeError(
            f"Prompt coverage mismatch: expected {args.expected_prompts}, "
            f"got {len(records_by_prompt)}"
        )

    record_files = []
    opt_key = f"optimal_step_lambda_{int(args.primary_lambda * 100):03d}"
    for (prompt_id, seed), (record, _) in sorted(compiled_by_key.items()):
        opt_s = record[opt_key]
        optimal_steps_histogram[opt_s] = optimal_steps_histogram.get(opt_s, 0) + 1
        rec_path = final_records_dir / f"p{prompt_id:06d}_s{seed}.json"
        rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        record_files.append(rec_path.name)

    manifest = {
        "schema": "prompt_conditioned_scored_oracle_dataset_v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_prompts_found": len(records_by_prompt),
        "expected_prompts": args.expected_prompts or len(records_by_prompt),
        "total_trajectories": len(compiled_by_key),
        "expected_trajectories": (args.expected_prompts or len(records_by_prompt))
        * len(expected_base_seeds),
        "expected_base_seeds": sorted(expected_base_seeds),
        "seed_policy": args.seed_policy,
        "candidate_steps": FORMAL_STEPS,
        "primary_lambda": args.primary_lambda,
        "record_files": record_files,
        "is_complete": True,
    }
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_backfilled = len(compiled_by_key)
    logger.info(f"\nSuccessfully backfilled {total_backfilled} trajectory records into {final_records_dir}!")
    logger.info(f"Genuine Oracle Optimal Step Distribution (lambda={args.primary_lambda}): {dict(sorted(optimal_steps_histogram.items()))}")


if __name__ == "__main__":
    main()
