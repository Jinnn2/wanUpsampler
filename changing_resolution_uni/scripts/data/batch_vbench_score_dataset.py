#!/usr/bin/env python3
"""
Multi-GPU Batch VBench-5 Quality Evaluator & Trajectory Record Backfiller.
Runs official VBench on all candidate branch videos and updates oracle dataset records
with genuine VBench-5 scores, latencies, multi-lambda utilities, and optimal switch steps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
SAFE_DIAGNOSTIC_DIMENSIONS = [
    "overall_consistency",
    "dynamic_degree",
    "temporal_flickering",
]
CASE_REQUEST_SCHEMA = "strict_vbench_case_request_v1"
CASE_RESULT_SCHEMA = "strict_vbench_case_result_v1"
RECORD_PROVENANCE_SCHEMA = "strict_vbench5_record_provenance_v1"


@dataclass(frozen=True)
class CaseScoreBundle:
    scores: dict[str, dict[str, float]]
    provenance: dict[str, Any]


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
        help="Five quality dimensions used for oracle utility; must match VBench-5.",
    )
    parser.add_argument(
        "--diagnostic_dimensions",
        nargs="*",
        choices=SAFE_DIAGNOSTIC_DIMENSIONS,
        default=[],
        help=(
            "Optional custom-input diagnostics stored separately and never averaged "
            "into oracle utility."
        ),
    )
    parser.add_argument(
        "--force_rescore",
        action="store_true",
        help="Run a new isolated VBench evaluation even when an exactly matching run exists.",
    )
    parser.add_argument(
        "--expected_vbench_commit",
        default=None,
        help="Optional exact Git commit required for the VBench repository.",
    )
    return parser.parse_args()


def warmup_vbench_cache(python_bin: str, vbench_root: Path) -> None:
    """Pre-downloads torch.hub dependencies (DINO) and OpenAI CLIP models in a single process to prevent multi-GPU race condition and network timeout."""
    hub_dir = Path.home() / ".cache" / "torch" / "hub"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_vbench_checkout(
    vbench_root: Path,
    *,
    expected_commit: str | None,
) -> dict[str, Any]:
    evaluate_py = vbench_root / "evaluate.py"
    if not evaluate_py.is_file():
        raise FileNotFoundError(f"VBench evaluate.py not found: {evaluate_py}")

    def git_output(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(vbench_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = git_output("rev-parse", "HEAD")
        dirty_paths = git_output("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Strict scoring requires a Git-backed VBench checkout: {vbench_root}"
        ) from exc
    if expected_commit and commit != expected_commit:
        raise RuntimeError(
            f"VBench commit mismatch: expected {expected_commit}, observed {commit}"
        )
    if dirty_paths:
        raise RuntimeError(
            "VBench checkout has tracked modifications; commit or stash them before "
            "formal scoring"
        )
    return {
        "git_commit": commit,
        "tracked_dirty": bool(dirty_paths),
        "tracked_dirty_paths": dirty_paths.splitlines(),
        "evaluate_py_sha256": sha256_file(evaluate_py),
    }


def build_case_request(
    *,
    video_dir: Path,
    prompt_map: Path,
    dimensions: list[str],
    quality_dimensions: list[str],
    diagnostic_dimensions: list[str],
    python_bin: str,
    vbench_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        prompt_payload = json.loads(prompt_map.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid prompt map {prompt_map}: {exc}") from exc
    if not isinstance(prompt_payload, dict) or not prompt_payload:
        raise RuntimeError(f"Prompt map must be a non-empty mapping: {prompt_map}")

    resolved_video_dir = video_dir.resolve()
    mapped_videos: dict[str, str] = {}
    inventory = []
    for raw_path, raw_prompt in sorted(prompt_payload.items()):
        path = Path(str(raw_path)).resolve()
        if path.parent != resolved_video_dir:
            raise RuntimeError(f"Prompt map video is outside case directory: {path}")
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise RuntimeError(f"Prompt map video is missing or not MP4: {path}")
        if path.stem in mapped_videos:
            raise RuntimeError(f"Duplicate video stem in prompt map: {path.stem}")
        mapped_videos[path.stem] = str(raw_prompt)
        inventory.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    actual_names = {path.name for path in resolved_video_dir.glob("*.mp4")}
    mapped_names = {item["name"] for item in inventory}
    if actual_names != mapped_names:
        raise RuntimeError(
            f"Prompt/video coverage mismatch in {video_dir}: "
            f"missing_from_prompt_map={sorted(actual_names - mapped_names)[:20]}, "
            f"missing_from_directory={sorted(mapped_names - actual_names)[:20]}"
        )

    request = {
        "schema": CASE_REQUEST_SCHEMA,
        "mode": "custom_input",
        "imaging_quality_preprocessing_mode": "longer",
        "quality_dimensions": quality_dimensions,
        "diagnostic_dimensions": diagnostic_dimensions,
        "dimensions": dimensions,
        "prompt_map_sha256": sha256_file(prompt_map),
        "videos": inventory,
        "python_bin": python_bin,
        "python_version": sys.version,
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "vbench": vbench_identity,
    }
    request["request_sha256"] = canonical_json_sha256(request)
    return request, mapped_videos


def parse_vbench_eval_result(
    result_file: Path,
    *,
    dimensions: list[str],
    expected_stems: set[str],
) -> dict[str, dict[str, float]]:
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid VBench result {result_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"VBench result must be a mapping: {result_file}")
    if set(data) != set(dimensions):
        raise RuntimeError(
            f"VBench result dimension mismatch in {result_file}: "
            f"expected={dimensions}, observed={sorted(data)}"
        )

    scores: dict[str, dict[str, float]] = {}
    for dimension in dimensions:
        payload = data[dimension]
        if not (
            isinstance(payload, list)
            and len(payload) >= 2
            and isinstance(payload[1], list)
        ):
            raise RuntimeError(
                f"Unsupported official VBench payload for {dimension} in {result_file}"
            )
        for row in payload[1]:
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Non-mapping per-video result for {dimension} in {result_file}"
                )
            video_path = row.get("video_path") or row.get("video_name")
            if not video_path or "video_results" not in row:
                raise RuntimeError(
                    f"Incomplete per-video result for {dimension} in {result_file}"
                )
            stem = Path(str(video_path)).stem
            if stem not in expected_stems:
                raise RuntimeError(
                    f"Unexpected video stem {stem!r} for {dimension} in {result_file}"
                )
            try:
                value = float(row["video_results"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Non-numeric score for {stem}.{dimension}: {row['video_results']!r}"
                ) from exc
            if dimension == "imaging_quality" and value > 1.0:
                value /= 100.0
            lower_bound = -1.0 if dimension == "overall_consistency" else 0.0
            if not math.isfinite(value) or not lower_bound <= value <= 1.0:
                raise RuntimeError(
                    f"Out-of-range score for {stem}.{dimension}: {value}"
                )
            previous = scores.setdefault(stem, {}).get(dimension)
            if previous is not None:
                raise RuntimeError(
                    f"Duplicate score for {stem}.{dimension} in {result_file}"
                )
            scores[stem][dimension] = value

    if set(scores) != expected_stems:
        raise RuntimeError(
            f"VBench video coverage mismatch in {result_file}: "
            f"missing={sorted(expected_stems - set(scores))[:20]}, "
            f"extra={sorted(set(scores) - expected_stems)[:20]}"
        )
    for stem, by_dimension in scores.items():
        if set(by_dimension) != set(dimensions):
            raise RuntimeError(
                f"VBench dimension coverage mismatch for {stem}: "
                f"observed={sorted(by_dimension)}"
            )
    return scores


def load_matching_cached_run(
    out_dir: Path,
    *,
    request_sha256: str,
    dimensions: list[str],
    expected_stems: set[str],
) -> CaseScoreBundle | None:
    runs_dir = out_dir / "runs"
    if not runs_dir.is_dir():
        return None
    for manifest_path in sorted(
        runs_dir.glob("*/score_run_manifest.json"), reverse=True
    ):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema") != CASE_RESULT_SCHEMA
                or manifest.get("request_sha256") != request_sha256
            ):
                continue
            result_path = manifest_path.parent / str(manifest["result_file"])
            if result_path.parent.resolve() != manifest_path.parent.resolve():
                raise RuntimeError("Cached result path escapes its run directory")
            if sha256_file(result_path) != manifest.get("result_sha256"):
                raise RuntimeError("Cached result SHA256 mismatch")
            scores = parse_vbench_eval_result(
                result_path,
                dimensions=dimensions,
                expected_stems=expected_stems,
            )
            logger.info(f"Reusing verified VBench run: {manifest_path.parent}")
            return CaseScoreBundle(
                scores=scores,
                provenance={
                    **manifest,
                    "run_manifest_path": str(manifest_path.resolve()),
                },
            )
        except Exception as exc:
            logger.warning(f"Ignoring invalid cached VBench run {manifest_path}: {exc}")
    return None


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
    quality_dimensions: list[str],
    diagnostic_dimensions: list[str],
    ngpus: int,
    force_rescore: bool,
    vbench_identity: dict[str, Any],
) -> CaseScoreBundle:
    """
    Run one isolated, content-bound VBench case or reuse an exactly matching run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_py = vbench_root / "evaluate.py"
    request, mapped_videos = build_case_request(
        video_dir=video_dir,
        prompt_map=prompt_map,
        dimensions=dimensions,
        quality_dimensions=quality_dimensions,
        diagnostic_dimensions=diagnostic_dimensions,
        python_bin=python_bin,
        vbench_identity=vbench_identity,
    )
    expected_stems = set(mapped_videos)
    if not force_rescore:
        cached = load_matching_cached_run(
            out_dir,
            request_sha256=str(request["request_sha256"]),
            dimensions=dimensions,
            expected_stems=expected_stems,
        )
        if cached is not None:
            return cached

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        out_dir
        / "runs"
        / f"run_{timestamp}_{str(request['request_sha256'])[:12]}_{os.getpid()}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    run_prompt_map = run_dir / "prompt_map.json"
    shutil.copy2(prompt_map, run_prompt_map)
    (run_dir / "score_request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    command_tail = [
        str(evaluate_py),
        "--videos_path",
        str(video_dir),
        "--dimension",
        *dimensions,
        "--mode",
        "custom_input",
        "--prompt_file",
        str(run_prompt_map),
        "--output_path",
        str(run_dir),
        "--imaging_quality_preprocessing_mode",
        "longer",
    ]
    cmd = [python_bin, *command_tail]
    if ngpus > 1:
        cmd = [
            python_bin,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={ngpus}",
            "--standalone",
            *command_tail,
        ]

    logger.info(
        f"Running strict VBench on {video_dir.name} "
        f"({len(expected_stems)} videos, request={request['request_sha256'][:12]})..."
    )
    subprocess.run(cmd, cwd=vbench_root, check=True)

    result_files = sorted(run_dir.glob("*_eval_results.json"))
    full_info_files = sorted(run_dir.glob("*_full_info.json"))
    if len(result_files) != 1:
        raise RuntimeError(
            f"Strict VBench run must produce exactly one eval result JSON in {run_dir}; "
            f"got results={len(result_files)}"
        )
    result_file = result_files[0]
    result_prefix = result_file.name.removesuffix("_eval_results.json")
    full_info_file = run_dir / f"{result_prefix}_full_info.json"
    if not full_info_file.is_file():
        raise RuntimeError(
            "Strict VBench run is missing the full-info JSON corresponding to its "
            f"unique eval result: {full_info_file}"
        )
    try:
        selected_full_info = json.loads(full_info_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid selected VBench full-info JSON: {full_info_file}") from exc
    if not isinstance(selected_full_info, list):
        raise RuntimeError(f"VBench full-info JSON must contain a list: {full_info_file}")

    def normalized_full_info(value: list[Any]) -> list[str]:
        return sorted(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in value
        )

    selected_normalized = normalized_full_info(selected_full_info)
    equivalent_extra_full_info = []
    for candidate in full_info_files:
        if candidate == full_info_file:
            continue
        try:
            candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid extra VBench full-info JSON: {candidate}") from exc
        if not isinstance(candidate_payload, list) or normalized_full_info(
            candidate_payload
        ) != selected_normalized:
            raise RuntimeError(
                "Multiple VBench full-info JSONs were produced and are not equivalent: "
                f"selected={full_info_file}, conflicting={candidate}"
            )
        equivalent_extra_full_info.append(
            {"file": candidate.name, "sha256": sha256_file(candidate)}
        )
    scores = parse_vbench_eval_result(
        result_file,
        dimensions=dimensions,
        expected_stems=expected_stems,
    )
    manifest = {
        "schema": CASE_RESULT_SCHEMA,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request_sha256": request["request_sha256"],
        "request_file": "score_request.json",
        "result_file": result_file.name,
        "result_sha256": sha256_file(result_file),
        "full_info_file": full_info_file.name,
        "full_info_sha256": sha256_file(full_info_file),
        "equivalent_extra_full_info": equivalent_extra_full_info,
        "video_count": len(expected_stems),
        "dimensions": dimensions,
        "quality_dimensions": quality_dimensions,
        "diagnostic_dimensions": diagnostic_dimensions,
        "vbench": vbench_identity,
    }
    manifest_path = run_dir / "score_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(
        f"Verified {len(scores)} videos x {len(dimensions)} dimensions in {run_dir}"
    )
    return CaseScoreBundle(
        scores=scores,
        provenance={
            **manifest,
            "run_manifest_path": str(manifest_path.resolve()),
        },
    )


def backfill_seed_records(
    seed_dir: Path,
    vbench_root: Path,
    python_bin: str,
    quality_dimensions: list[str],
    diagnostic_dimensions: list[str],
    ngpus: int,
    primary_lambda: float,
    force_rescore: bool,
    vbench_identity: dict[str, Any],
) -> dict[str, Any]:
    """Runs VBench on all step directories of a seed, compiles scores, and backfills manifests."""
    dimensions = [*quality_dimensions, *diagnostic_dimensions]
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
    case_provenance: dict[str, dict[str, Any]] = {}
    diagnostic_case_provenance: dict[str, dict[str, Any]] = {}

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
        quality_bundle = score_case_directory(
            vbench_root=vbench_root,
            python_bin=python_bin,
            video_dir=vdir,
            prompt_map=prompt_map_file,
            out_dir=case_out,
            dimensions=quality_dimensions,
            quality_dimensions=quality_dimensions,
            diagnostic_dimensions=[],
            ngpus=ngpus,
            force_rescore=force_rescore,
            vbench_identity=vbench_identity,
        )
        combined_scores = {
            stem: dict(by_dimension)
            for stem, by_dimension in quality_bundle.scores.items()
        }
        case_provenance[case_name] = quality_bundle.provenance
        if diagnostic_dimensions:
            diagnostic_bundle = score_case_directory(
                vbench_root=vbench_root,
                python_bin=python_bin,
                video_dir=vdir,
                prompt_map=prompt_map_file,
                out_dir=case_out,
                dimensions=diagnostic_dimensions,
                quality_dimensions=[],
                diagnostic_dimensions=diagnostic_dimensions,
                ngpus=ngpus,
                force_rescore=force_rescore,
                vbench_identity=vbench_identity,
            )
            if set(diagnostic_bundle.scores) != set(combined_scores):
                raise RuntimeError(
                    f"Quality/diagnostic video coverage differs for {case_name}"
                )
            for stem, by_dimension in diagnostic_bundle.scores.items():
                overlap = set(combined_scores[stem]) & set(by_dimension)
                if overlap:
                    raise RuntimeError(
                        f"Quality/diagnostic dimensions overlap for {stem}: {sorted(overlap)}"
                    )
                combined_scores[stem].update(by_dimension)
            diagnostic_case_provenance[case_name] = diagnostic_bundle.provenance
        case_scores[case_name] = combined_scores

    # 2. Backfill scores into sample records
    compiled_records = []
    validation_errors: list[str] = []
    for m in sample_manifests:
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
        native_vbench5 = float(
            np.mean([native_dims[dim] for dim in quality_dimensions], dtype=np.float64)
        )
        native_diagnostics = {
            dim: float(native_dims[dim]) for dim in diagnostic_dimensions
        }

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
            s_vb = float(
                np.mean([s_dims[dim] for dim in quality_dimensions], dtype=np.float64)
            )

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
                "vbench5": s_vb,
                "dimensions": {
                    key: float(s_dims[key]) for key in quality_dimensions
                },
                "diagnostics": {
                    key: float(s_dims[key]) for key in diagnostic_dimensions
                },
                "latency_seconds": lat,
                "latency_source": latency_source,
                "speedup_vs_native": native_lat / max(lat, 1e-5),
                "utilities": u_dict,
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
            "native_vbench5": native_vbench5,
            "native_latency_seconds": native_lat,
            "native_dimensions": {
                key: float(native_dims[key]) for key in quality_dimensions
            },
            "native_diagnostics": native_diagnostics,
            "native_latency_source": "warm_pipeline_seconds",
            "candidates": candidates_list,
            "scoring_provenance": {
                "schema": RECORD_PROVENANCE_SCHEMA,
                "quality_dimensions": quality_dimensions,
                "diagnostic_dimensions": diagnostic_dimensions,
                "quality_aggregation": "arithmetic_mean_raw_vbench5_float64",
                "vbench": vbench_identity,
                "cases": {
                    case_name: {
                        "request_sha256": provenance["request_sha256"],
                        "result_sha256": provenance["result_sha256"],
                        "full_info_sha256": provenance["full_info_sha256"],
                        "run_manifest_path": provenance["run_manifest_path"],
                    }
                    for case_name, provenance in sorted(case_provenance.items())
                },
                "diagnostic_cases": {
                    case_name: {
                        "request_sha256": provenance["request_sha256"],
                        "result_sha256": provenance["result_sha256"],
                        "full_info_sha256": provenance["full_info_sha256"],
                        "run_manifest_path": provenance["run_manifest_path"],
                    }
                    for case_name, provenance in sorted(
                        diagnostic_case_provenance.items()
                    )
                },
            },
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
    if list(args.dimensions) != QUALITY5_DIMENSIONS:
        raise ValueError(
            f"Strict oracle utility requires exactly {QUALITY5_DIMENSIONS}; "
            f"got {list(args.dimensions)}"
        )
    diagnostic_dimensions = list(dict.fromkeys(args.diagnostic_dimensions))
    overlap = sorted(set(args.dimensions) & set(diagnostic_dimensions))
    if overlap:
        raise ValueError(f"Quality and diagnostic dimensions overlap: {overlap}")
    vbench_identity = inspect_vbench_checkout(
        vbench_root,
        expected_commit=args.expected_vbench_commit,
    )
    logger.info(
        "VBench identity: commit=%s evaluate.py=%s dirty=%s",
        vbench_identity["git_commit"],
        vbench_identity["evaluate_py_sha256"],
        vbench_identity["tracked_dirty"],
    )

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
            quality_dimensions=list(args.dimensions),
            diagnostic_dimensions=diagnostic_dimensions,
            ngpus=args.ngpus,
            primary_lambda=args.primary_lambda,
            force_rescore=args.force_rescore,
            vbench_identity=vbench_identity,
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
    record_sha256: dict[str, str] = {}
    opt_key = f"optimal_step_lambda_{int(args.primary_lambda * 100):03d}"
    for (prompt_id, seed), (record, _) in sorted(compiled_by_key.items()):
        opt_s = record[opt_key]
        optimal_steps_histogram[opt_s] = optimal_steps_histogram.get(opt_s, 0) + 1
        rec_path = final_records_dir / f"p{prompt_id:06d}_s{seed}.json"
        rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        record_files.append(rec_path.name)
        record_sha256[rec_path.name] = sha256_file(rec_path)

    manifest = {
        "schema": "prompt_conditioned_scored_oracle_dataset_v3",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "quality_profile": "strict_vbench5_v1",
        "quality_dimensions": list(args.dimensions),
        "diagnostic_dimensions": diagnostic_dimensions,
        "quality_aggregation": "arithmetic_mean_raw_vbench5_float64",
        "vbench": vbench_identity,
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
        "record_sha256": record_sha256,
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
