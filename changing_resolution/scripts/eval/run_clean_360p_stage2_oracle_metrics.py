from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


FORMAL_STEPS = [30, 35, *range(40, 51)]
FORMAL_SAMPLE_COUNT = 10
FORMAL_INFER_STEPS = 50
QUALITY5_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]
OVERALL_DIMENSIONS = ["overall_consistency"]
MIN_VALID_VIDEO_BYTES = 1024


def main() -> None:
    args = parse_args()
    root = Path(args.oracle_root).resolve()
    inventory = load_inventory(
        root,
        strict_protocol=args.strict_protocol,
        min_video_bytes=args.min_video_bytes,
    )
    print_inventory(inventory)
    if args.action == "check":
        resolved_timing_source, _ = collect_timings(
            root, inventory, requested=args.timing_source
        )
        print(f"[oracle-metrics] timing_source={resolved_timing_source}")
        return

    prepare_inputs(root, inventory)
    if args.action == "prepare":
        return

    if args.action in {"run", "all"}:
        if not args.vbench_root:
            raise SystemExit("--vbench-root (or VBENCH_ROOT) is required for VBench")
        vbench_root = Path(args.vbench_root).resolve()
        run_vbench_profile(
            root,
            inventory,
            profile="quality5",
            dimensions=QUALITY5_DIMENSIONS,
            vbench_root=vbench_root,
            python=args.python,
            ngpus=args.ngpus,
            selected_cases=args.cases,
            skip_existing=args.skip_existing,
        )
        if args.include_overall:
            run_vbench_profile(
                root,
                inventory,
                profile="overall",
                dimensions=OVERALL_DIMENSIONS,
                vbench_root=vbench_root,
                python=args.python,
                ngpus=args.ngpus,
                selected_cases=args.cases,
                skip_existing=args.skip_existing,
            )
        if args.action == "run":
            return

    outputs = collect_metrics(
        root,
        inventory,
        include_overall=args.include_overall,
        overall_weight=args.overall_weight,
        max_quality_drop=args.max_quality_drop,
        latency_lambda=args.latency_lambda,
        timing_source=args.timing_source,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def load_inventory(
    root: Path,
    *,
    strict_protocol: bool,
    min_video_bytes: int,
) -> dict[str, Any]:
    if not root.is_dir():
        raise SystemExit(f"Oracle output root not found: {root}")
    protocol_path = root / "protocol.json"
    protocol = load_json_object(protocol_path)
    if protocol.get("schema") != "wan_taa_free_oracle_protocol_v1":
        raise SystemExit(f"Invalid or missing branch protocol: {protocol_path}")
    if protocol.get("execution_mode") != "branch":
        raise SystemExit("Oracle metric evaluation requires the branch protocol")
    steps = [int(step) for step in protocol.get("candidate_steps", [])]
    if not steps or steps != sorted(set(steps)):
        raise SystemExit(f"Invalid candidate_steps in {protocol_path}: {steps}")
    if strict_protocol and steps != FORMAL_STEPS:
        raise SystemExit(
            f"Strict metric protocol requires candidates {FORMAL_STEPS}; got {steps}"
        )
    if protocol.get("taa_enabled") is not False:
        raise SystemExit("Metric input protocol does not explicitly disable TAA")
    if int(protocol.get("infer_steps", -1)) != FORMAL_INFER_STEPS:
        raise SystemExit(f"Metric protocol requires infer_steps={FORMAL_INFER_STEPS}")
    if strict_protocol:
        required_protocol = {
            "strict_protocol": True,
            "include_native_hr": True,
            "runtime_lora_allowed": False,
            "feature_caching": "NoCaching",
            "target_video_length": 81,
            "lr_rgb_size": [368, 640],
            "hr_rgb_size": [720, 1248],
            "lr_latent_size": [46, 80],
        }
        mismatches = {
            key: {"expected": expected, "actual": protocol.get(key)}
            for key, expected in required_protocol.items()
            if protocol.get(key) != expected
        }
        if mismatches:
            raise SystemExit(f"Strict metric protocol mismatch: {mismatches}")

    manifest_root = root / "manifests"
    manifest_paths = sorted(manifest_root.glob("*.json"))
    expected_count = int(protocol.get("prompt_count", -1))
    if strict_protocol and expected_count != FORMAL_SAMPLE_COUNT:
        raise SystemExit(
            f"Strict metric protocol requires {FORMAL_SAMPLE_COUNT} prompts; "
            f"got {expected_count}"
        )
    if expected_count <= 0 or len(manifest_paths) != expected_count:
        raise SystemExit(
            f"Expected {expected_count} sample manifests, found {len(manifest_paths)} "
            f"under {manifest_root}"
        )

    samples: list[dict[str, Any]] = []
    seen_prompt_indices: set[int] = set()
    seen_seeds: set[int] = set()
    for manifest_path in manifest_paths:
        manifest = load_json_object(manifest_path)
        if manifest.get("schema") != "wan_taa_free_oracle_v1":
            raise SystemExit(f"Invalid sample manifest schema: {manifest_path}")
        if manifest.get("execution_mode") != "branch":
            raise SystemExit(f"Non-branch sample manifest: {manifest_path}")
        if manifest.get("taa_enabled") is not False:
            raise SystemExit(f"Sample does not explicitly disable TAA: {manifest_path}")
        manifest_steps = [int(step) for step in manifest.get("candidate_steps", [])]
        if manifest_steps != steps:
            raise SystemExit(
                f"Sample candidate_steps mismatch in {manifest_path}: "
                f"expected={steps}, got={manifest_steps}"
            )
        prompt_index = int(manifest["prompt_index"])
        seed = int(manifest["seed"])
        prompt = str(manifest["prompt"])
        sample_id = manifest_path.stem
        expected_id = f"{prompt_index:04d}_seed{seed}"
        if sample_id != expected_id:
            raise SystemExit(
                f"Sample id mismatch in {manifest_path}: {sample_id} != {expected_id}"
            )
        if prompt_index in seen_prompt_indices or seed in seen_seeds:
            raise SystemExit(f"Duplicate prompt index or seed in {manifest_path}")
        seen_prompt_indices.add(prompt_index)
        seen_seeds.add(seed)

        branch_rows = {
            int(row["candidate_step"]): row
            for row in manifest.get("branches", [])
            if isinstance(row, dict) and "candidate_step" in row
        }
        if sorted(branch_rows) != steps:
            raise SystemExit(
                f"Candidate rows mismatch in {manifest_path}: "
                f"expected={steps}, got={sorted(branch_rows)}"
            )
        branch_videos: dict[int, Path] = {}
        for step in steps:
            row = branch_rows[step]
            if int(row.get("lr_evaluations", -1)) != step:
                raise SystemExit(f"Invalid LR NFE for step {step} in {manifest_path}")
            if int(row.get("hr_evaluations", -1)) != 50 - step:
                raise SystemExit(f"Invalid HR NFE for step {step} in {manifest_path}")
            expected_video = (
                root / "videos" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.mp4"
            )
            validate_video(expected_video, min_video_bytes=min_video_bytes)
            branch_videos[step] = expected_video.resolve()

        native = manifest.get("native_hr")
        if not isinstance(native, dict):
            raise SystemExit(f"Native-HR timing/output is missing in {manifest_path}")
        if (
            int(native.get("lr_evaluations", -1)) != 0
            or int(native.get("hr_evaluations", -1)) != FORMAL_INFER_STEPS
        ):
            raise SystemExit(f"Invalid Native-HR NFE in {manifest_path}")
        native_video = root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4"
        validate_video(native_video, min_video_bytes=min_video_bytes)
        samples.append(
            {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "manifest_path": manifest_path.resolve(),
                "branch_rows": branch_rows,
                "branch_videos": branch_videos,
                "native_row": native,
                "native_video": native_video.resolve(),
            }
        )

    samples.sort(key=lambda item: item["prompt_index"])
    cases: dict[str, dict[str, Any]] = {}
    for step in steps:
        name = f"step{step:02d}"
        cases[name] = {
            "name": name,
            "candidate_step": step,
            "video_dir": (root / "videos" / name).resolve(),
            "videos": [sample["branch_videos"][step] for sample in samples],
            "prompts": [sample["prompt"] for sample in samples],
        }
    cases["native_hr"] = {
        "name": "native_hr",
        "candidate_step": None,
        "video_dir": (root / "videos" / "native_hr").resolve(),
        "videos": [sample["native_video"] for sample in samples],
        "prompts": [sample["prompt"] for sample in samples],
    }
    for case_name, case in cases.items():
        expected_videos = {video.resolve() for video in case["videos"]}
        actual_videos = {video.resolve() for video in case["video_dir"].glob("*.mp4")}
        if actual_videos != expected_videos:
            missing = sorted(str(path) for path in expected_videos - actual_videos)
            extra = sorted(str(path) for path in actual_videos - expected_videos)
            raise SystemExit(
                f"Video inventory mismatch for {case_name}: "
                f"missing={missing}, extra={extra}"
            )
    return {
        "root": root,
        "protocol": protocol,
        "protocol_path": protocol_path.resolve(),
        "steps": steps,
        "samples": samples,
        "cases": cases,
        "min_video_bytes": min_video_bytes,
    }


def print_inventory(inventory: dict[str, Any]) -> None:
    print(
        "[oracle-metrics] "
        f"root={inventory['root']} samples={len(inventory['samples'])} "
        f"candidates={inventory['steps']} cases={len(inventory['cases'])}"
    )


def prepare_inputs(root: Path, inventory: dict[str, Any]) -> Path:
    input_root = root / "metrics" / "oracle_vbench_inputs"
    case_records: dict[str, Any] = {}
    for case_name, case in inventory["cases"].items():
        mapping = {
            str(video): prompt for video, prompt in zip(case["videos"], case["prompts"])
        }
        case_root = input_root / case_name
        prompt_map = case_root / "prompt_map.json"
        write_json(prompt_map, mapping)
        case_records[case_name] = {
            "candidate_step": case["candidate_step"],
            "video_dir": str(case["video_dir"]),
            "prompt_map": str(prompt_map.resolve()),
            "input_signature_quality5": case_signature(case, QUALITY5_DIMENSIONS),
            "input_signature_overall": case_signature(case, OVERALL_DIMENSIONS),
        }
    payload = {
        "schema": "wan_taa_free_oracle_vbench_inputs_v1",
        "generated_at_utc": utc_now(),
        "oracle_root": str(root),
        "protocol": str(inventory["protocol_path"]),
        "protocol_sha256": sha256_file(inventory["protocol_path"]),
        "candidate_steps": inventory["steps"],
        "sample_count": len(inventory["samples"]),
        "quality5_dimensions": QUALITY5_DIMENSIONS,
        "overall_dimensions": OVERALL_DIMENSIONS,
        "cases": case_records,
    }
    output = root / "metrics" / "oracle_evaluation_manifest.json"
    write_json(output, payload)
    print(f"Prepared VBench inputs: {input_root}")
    return output


def run_vbench_profile(
    root: Path,
    inventory: dict[str, Any],
    *,
    profile: str,
    dimensions: list[str],
    vbench_root: Path,
    python: str,
    ngpus: int,
    selected_cases: list[str] | None,
    skip_existing: bool,
) -> None:
    evaluate = vbench_root / "evaluate.py"
    if not evaluate.is_file():
        raise SystemExit(f"Official VBench evaluate.py not found: {evaluate}")
    case_names = list(inventory["cases"])
    if selected_cases:
        unknown = sorted(set(selected_cases) - set(case_names))
        if unknown:
            raise SystemExit(f"Unknown oracle VBench cases: {unknown}")
        if len(selected_cases) != len(set(selected_cases)):
            raise SystemExit("--cases contains duplicates")
        case_names = selected_cases
    revision = git_revision(vbench_root)
    raw_root = root / "metrics" / f"oracle_vbench_raw_{profile}"
    input_root = root / "metrics" / "oracle_vbench_inputs"
    for case_name in case_names:
        case = inventory["cases"][case_name]
        output = raw_root / case_name
        output.mkdir(parents=True, exist_ok=True)
        signature = case_signature(case, dimensions)
        record_path = output / "run_record.json"
        if skip_existing and valid_run_record(
            record_path,
            profile=profile,
            signature=signature,
            dimensions=dimensions,
            output_root=output,
            case=case,
        ):
            print(f"[VBench:{profile}:skip] {case_name}")
            continue
        prompt_map = input_root / case_name / "prompt_map.json"
        base = [
            str(evaluate),
            "--videos_path",
            str(case["video_dir"]),
            "--dimension",
            *dimensions,
            "--mode",
            "custom_input",
            "--prompt_file",
            str(prompt_map),
            "--output_path",
            str(output),
        ]
        command = (
            [
                python,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={ngpus}",
                "--standalone",
                *base,
            ]
            if ngpus > 1
            else [python, *base]
        )
        print(f"[VBench:{profile}] {case_name}", flush=True)
        previous_results = result_snapshots(output)
        run_started_ns = time.time_ns()
        subprocess.run(command, cwd=vbench_root, check=True)
        result_path, _, _ = find_complete_result(
            output,
            case,
            dimensions,
            previous_results=previous_results,
        )
        write_json(
            record_path,
            {
                "schema": "wan_taa_free_oracle_vbench_run_v1",
                "completed_at_utc": utc_now(),
                "profile": profile,
                "case": case_name,
                "dimensions": dimensions,
                "input_signature": signature,
                "result_path": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path),
                "run_started_ns": run_started_ns,
                "vbench_root": str(vbench_root),
                "vbench_revision": revision,
                "python": python,
                "ngpus": ngpus,
                "command": command,
            },
        )


def valid_run_record(
    path: Path,
    *,
    profile: str,
    signature: str,
    dimensions: list[str],
    output_root: Path,
    case: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        record = load_json_object(path)
    except SystemExit:
        return False
    if record.get("schema") != "wan_taa_free_oracle_vbench_run_v1":
        return False
    if record.get("profile") != profile or record.get("case") != case["name"]:
        return False
    if record.get("input_signature") != signature:
        return False
    if record.get("dimensions") != dimensions:
        return False
    try:
        load_recorded_result(
            record,
            record_path=path,
            output_root=output_root,
            case=case,
            dimensions=dimensions,
        )
    except SystemExit:
        return False
    return True


def result_snapshots(output_root: Path) -> dict[Path, tuple[int, int]]:
    return {
        path.resolve(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output_root.glob("*_eval_results.json")
    }


def find_complete_result(
    output_root: Path,
    case: dict[str, Any],
    dimensions: list[str],
    *,
    previous_results: dict[Path, tuple[int, int]] | None = None,
) -> tuple[Path, dict[str, float], dict[str, dict[str, float]]]:
    candidates = sorted(
        output_root.glob("*_eval_results.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    failures: list[str] = []
    for path in candidates:
        if previous_results is not None:
            before = previous_results.get(path.resolve())
            after = (path.stat().st_size, path.stat().st_mtime_ns)
            if before == after:
                continue
        try:
            aggregates, per_video = parse_vbench_result(path, case, dimensions)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        return path, aggregates, per_video
    suffix = f" Invalid candidates: {failures}" if failures else ""
    raise SystemExit(
        f"No complete VBench result for {case['name']} under {output_root}.{suffix}"
    )


def load_recorded_result(
    record: dict[str, Any],
    *,
    record_path: Path,
    output_root: Path,
    case: dict[str, Any],
    dimensions: list[str],
) -> tuple[Path, dict[str, float], dict[str, dict[str, float]]]:
    raw_path = record.get("result_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"Missing result_path in {record_path}")
    result_path = Path(raw_path).resolve()
    if result_path.parent != output_root.resolve():
        raise SystemExit(
            f"Recorded VBench result escapes its case directory: {result_path}"
        )
    if not result_path.is_file() or not result_path.name.endswith("_eval_results.json"):
        raise SystemExit(f"Recorded VBench result is missing: {result_path}")
    expected_sha256 = record.get("result_sha256")
    if (
        not isinstance(expected_sha256, str)
        or sha256_file(result_path) != expected_sha256
    ):
        raise SystemExit(f"Recorded VBench result checksum mismatch: {result_path}")
    try:
        aggregates, per_video = parse_vbench_result(result_path, case, dimensions)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid recorded VBench result {result_path}: {exc}"
        ) from exc
    return result_path, aggregates, per_video


def parse_vbench_result(
    path: Path,
    case: dict[str, Any],
    dimensions: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError("result root must be an object")
    expected = {video.name for video in case["videos"]}
    aggregates: dict[str, float] = {}
    per_video: dict[str, dict[str, float]] = {}
    for dimension in dimensions:
        item = payload[dimension]
        if not isinstance(item, list) or len(item) < 2 or not isinstance(item[1], list):
            raise TypeError(f"invalid payload for {dimension}")
        aggregates[dimension] = normalize_vbench_value(dimension, item[0])
        values: dict[str, float] = {}
        for row in item[1]:
            if not isinstance(row, dict):
                raise TypeError(f"invalid per-video row for {dimension}")
            name = Path(str(row["video_path"])).name
            if name in values:
                raise ValueError(f"duplicate video {name} for {dimension}")
            values[name] = normalize_vbench_value(dimension, row["video_results"])
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                f"video coverage mismatch for {dimension}: missing={missing}, extra={extra}"
            )
        per_video[dimension] = values
    return aggregates, per_video


def normalize_vbench_value(dimension: str, raw: Any) -> float:
    if isinstance(raw, bool):
        value = float(raw)
    elif isinstance(raw, (int, float)):
        value = float(raw)
    else:
        raise TypeError(f"non-numeric VBench value for {dimension}: {raw!r}")
    if dimension == "imaging_quality" and value > 1.0:
        value /= 100.0
    if not math.isfinite(value):
        raise ValueError(f"non-finite VBench value for {dimension}: {value}")
    return value


def collect_metrics(
    root: Path,
    inventory: dict[str, Any],
    *,
    include_overall: bool,
    overall_weight: float,
    max_quality_drop: float,
    latency_lambda: float,
    timing_source: str,
) -> dict[str, Path]:
    quality_results = collect_profile_results(
        root,
        inventory,
        profile="quality5",
        dimensions=QUALITY5_DIMENSIONS,
    )
    overall_results = (
        collect_profile_results(
            root,
            inventory,
            profile="overall",
            dimensions=OVERALL_DIMENSIONS,
        )
        if include_overall
        else {}
    )
    resolved_timing_source, timings = collect_timings(
        root, inventory, requested=timing_source
    )

    native_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for sample in inventory["samples"]:
        sample_id = sample["sample_id"]
        native_values = metric_values_for_video(
            quality_results["native_hr"], sample["native_video"].name
        )
        native_quality5 = statistics.fmean(
            native_values[dimension] for dimension in QUALITY5_DIMENSIONS
        )
        native_overall = optional_overall_value(
            overall_results, "native_hr", sample["native_video"].name
        )
        native_selection_quality = composite_quality(
            native_quality5,
            native_overall,
            overall_weight=overall_weight,
        )
        native_latency = timings[sample_id]["native_hr"]
        native_row = {
            "sample_id": sample_id,
            "prompt_index": sample["prompt_index"],
            "seed": sample["seed"],
            "prompt": sample["prompt"],
            **native_values,
            "vbench5": native_quality5,
            "overall_consistency": blank_if_none(native_overall),
            "selection_quality": native_selection_quality,
            "latency_seconds": native_latency,
        }
        native_rows.append(native_row)

        sample_candidates: list[dict[str, Any]] = []
        for step in inventory["steps"]:
            case_name = f"step{step:02d}"
            video = sample["branch_videos"][step]
            values = metric_values_for_video(quality_results[case_name], video.name)
            quality5 = statistics.fmean(
                values[dimension] for dimension in QUALITY5_DIMENSIONS
            )
            overall = optional_overall_value(overall_results, case_name, video.name)
            selection_quality = composite_quality(
                quality5,
                overall,
                overall_weight=overall_weight,
            )
            latency = timings[sample_id][step]
            row = {
                "sample_id": sample_id,
                "prompt_index": sample["prompt_index"],
                "seed": sample["seed"],
                "prompt": sample["prompt"],
                "candidate_step": step,
                "lr_evaluations": step,
                "hr_evaluations": 50 - step,
                "video": str(video),
                **values,
                "vbench5": quality5,
                "overall_consistency": blank_if_none(overall),
                "selection_quality": selection_quality,
                "native_vbench5": native_quality5,
                "native_selection_quality": native_selection_quality,
                "quality_delta_vs_native": selection_quality - native_selection_quality,
                "quality_drop_vs_native": native_selection_quality - selection_quality,
                "latency_seconds": latency,
                "native_latency_seconds": native_latency,
                "speedup_vs_native": native_latency / latency,
            }
            sample_candidates.append(row)
        label = label_sample(
            sample,
            sample_candidates,
            native_row,
            max_quality_drop=max_quality_drop,
            latency_lambda=latency_lambda,
        )
        labels.append(label)
        candidate_rows.extend(sample_candidates)

    summary_rows = summarize_candidates(candidate_rows, inventory["steps"])
    metrics_root = root / "metrics"
    candidate_csv = metrics_root / "oracle_candidate_per_sample.csv"
    native_csv = metrics_root / "oracle_native_per_sample.csv"
    summary_csv = metrics_root / "oracle_candidate_summary.csv"
    labels_csv = metrics_root / "oracle_labels.csv"
    write_csv(candidate_csv, candidate_rows)
    write_csv(native_csv, native_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(labels_csv, labels)
    canonical = metrics_root / "oracle_metrics.json"
    write_json(
        canonical,
        {
            "schema": "wan_taa_free_oracle_metrics_v1",
            "generated_at_utc": utc_now(),
            "oracle_root": str(root),
            "candidate_steps": inventory["steps"],
            "sample_count": len(inventory["samples"]),
            "quality_metric": {
                "name": "VBench-5",
                "definition": "unweighted mean of five custom-input VBench dimensions",
                "dimensions": QUALITY5_DIMENSIONS,
                "dynamic_degree_included": False,
                "overall_consistency_included_in_vbench5": False,
            },
            "selection_quality": {
                "overall_consistency_available": include_overall,
                "overall_weight": overall_weight,
                "formula": (
                    "(1-overall_weight)*VBench5 + overall_weight*overall_consistency"
                ),
            },
            "label_policies": {
                "quality_floor": {
                    "max_absolute_quality_drop_vs_native": max_quality_drop,
                    "rule": (
                        "minimum measured latency among eligible candidates; "
                        "fallback to maximum selection quality"
                    ),
                },
                "weighted_utility": {
                    "latency_lambda": latency_lambda,
                    "formula": (
                        "selection_quality - latency_lambda * "
                        "candidate_latency/native_latency"
                    ),
                },
            },
            "timing_source_requested": timing_source,
            "timing_source_resolved": resolved_timing_source,
            "candidate_summary": summary_rows,
            "native_per_sample": native_rows,
            "candidate_per_sample": candidate_rows,
            "labels": labels,
            "quality_result_sources": {
                case: result["result_path"] for case, result in quality_results.items()
            },
            "overall_result_sources": {
                case: result["result_path"] for case, result in overall_results.items()
            },
        },
    )
    return {
        "canonical_json": canonical,
        "candidate_per_sample_csv": candidate_csv,
        "native_per_sample_csv": native_csv,
        "candidate_summary_csv": summary_csv,
        "oracle_labels_csv": labels_csv,
    }


def collect_profile_results(
    root: Path,
    inventory: dict[str, Any],
    *,
    profile: str,
    dimensions: list[str],
) -> dict[str, dict[str, Any]]:
    raw_root = root / "metrics" / f"oracle_vbench_raw_{profile}"
    results: dict[str, dict[str, Any]] = {}
    for case_name, case in inventory["cases"].items():
        output = raw_root / case_name
        record_path = output / "run_record.json"
        record = load_json_object(record_path)
        signature = case_signature(case, dimensions)
        if record.get("schema") != "wan_taa_free_oracle_vbench_run_v1":
            raise SystemExit(f"Invalid VBench run record schema: {record_path}")
        if record.get("profile") != profile or record.get("case") != case_name:
            raise SystemExit(f"Profile/case mismatch in {record_path}")
        if record.get("input_signature") != signature:
            raise SystemExit(
                f"Missing or stale {profile} run record for {case_name}: {record_path}"
            )
        if record.get("dimensions") != dimensions:
            raise SystemExit(f"Dimension mismatch in {record_path}")
        result_path, aggregates, per_video = load_recorded_result(
            record,
            record_path=record_path,
            output_root=output,
            case=case,
            dimensions=dimensions,
        )
        results[case_name] = {
            "result_path": str(result_path.resolve()),
            "aggregates": aggregates,
            "per_video": per_video,
            "run_record": record,
        }
    return results


def metric_values_for_video(
    result: dict[str, Any], video_name: str
) -> dict[str, float]:
    return {
        dimension: values[video_name]
        for dimension, values in result["per_video"].items()
    }


def optional_overall_value(
    results: dict[str, dict[str, Any]], case_name: str, video_name: str
) -> float | None:
    if not results:
        return None
    return results[case_name]["per_video"]["overall_consistency"][video_name]


def composite_quality(
    vbench5: float,
    overall: float | None,
    *,
    overall_weight: float,
) -> float:
    if overall_weight == 0.0:
        return vbench5
    if overall is None:
        raise SystemExit("overall_weight > 0 requires --include-overall")
    return (1.0 - overall_weight) * vbench5 + overall_weight * overall


def collect_timings(
    root: Path,
    inventory: dict[str, Any],
    *,
    requested: str,
) -> tuple[str, dict[str, dict[int | str, float]]]:
    if requested in {"independent", "prefer-independent"}:
        independent = try_independent_timings(root, inventory)
        if independent is not None:
            return "independent", independent
        if requested == "independent":
            raise SystemExit(
                "Complete independent timings are required but were not found under "
                f"{root / 'independent/manifests'}"
            )
    timings: dict[str, dict[int | str, float]] = {}
    for sample in inventory["samples"]:
        values: dict[int | str, float] = {}
        for step in inventory["steps"]:
            values[step] = positive_float(
                sample["branch_rows"][step].get("estimated_warm_pipeline_seconds"),
                context=f"branch timing {sample['sample_id']} step {step}",
            )
        values["native_hr"] = positive_float(
            sample["native_row"].get("warm_pipeline_seconds"),
            context=f"Native-HR timing {sample['sample_id']}",
        )
        timings[sample["sample_id"]] = values
    return "branch_estimate", timings


def try_independent_timings(
    root: Path, inventory: dict[str, Any]
) -> dict[str, dict[int | str, float]] | None:
    manifest_root = root / "independent" / "manifests"
    timings: dict[str, dict[int | str, float]] = {}
    for sample in inventory["samples"]:
        path = manifest_root / f"{sample['sample_id']}.json"
        if not path.is_file():
            return None
        manifest = load_json_object(path)
        if (
            manifest.get("schema") != "wan_taa_free_oracle_v1"
            or manifest.get("execution_mode") != "independent"
            or manifest.get("taa_enabled") is not False
            or int(manifest.get("prompt_index", -1)) != sample["prompt_index"]
            or int(manifest.get("seed", -1)) != sample["seed"]
            or str(manifest.get("prompt", "")) != sample["prompt"]
        ):
            return None
        manifest_steps = [int(step) for step in manifest.get("candidate_steps", [])]
        if manifest_steps != inventory["steps"]:
            return None
        rows = {
            int(row["candidate_step"]): row
            for row in manifest.get("branches", [])
            if isinstance(row, dict) and "candidate_step" in row
        }
        if sorted(rows) != inventory["steps"] or not isinstance(
            manifest.get("native_hr"), dict
        ):
            return None
        values: dict[int | str, float] = {}
        for step in inventory["steps"]:
            values[step] = positive_float(
                rows[step].get("warm_pipeline_seconds"),
                context=f"independent timing {sample['sample_id']} step {step}",
            )
        values["native_hr"] = positive_float(
            manifest["native_hr"].get("warm_pipeline_seconds"),
            context=f"independent Native-HR timing {sample['sample_id']}",
        )
        timings[sample["sample_id"]] = values
    return timings


def label_sample(
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    native_row: dict[str, Any],
    *,
    max_quality_drop: float,
    latency_lambda: float,
) -> dict[str, Any]:
    native_quality = float(native_row["selection_quality"])
    native_latency = float(native_row["latency_seconds"])
    for row in rows:
        row["quality_floor_eligible"] = (
            native_quality - float(row["selection_quality"]) <= max_quality_drop
        )
        row["utility_score"] = float(row["selection_quality"]) - latency_lambda * (
            float(row["latency_seconds"]) / native_latency
        )
        row["pareto_optimal"] = is_pareto_optimal(row, rows)

    max_quality = max(
        rows,
        key=lambda row: (
            float(row["selection_quality"]),
            -float(row["latency_seconds"]),
            int(row["candidate_step"]),
        ),
    )
    eligible = [row for row in rows if row["quality_floor_eligible"]]
    floor_fallback = not eligible
    floor_pool = eligible or [max_quality]
    quality_floor = min(
        floor_pool,
        key=lambda row: (
            float(row["latency_seconds"]),
            -float(row["selection_quality"]),
            -int(row["candidate_step"]),
        ),
    )
    utility = max(
        rows,
        key=lambda row: (
            float(row["utility_score"]),
            -float(row["latency_seconds"]),
            int(row["candidate_step"]),
        ),
    )
    for row in rows:
        row["selected_by_max_quality"] = row is max_quality
        row["selected_by_quality_floor"] = row is quality_floor
        row["selected_by_weighted_utility"] = row is utility
    return {
        "sample_id": sample["sample_id"],
        "prompt_index": sample["prompt_index"],
        "seed": sample["seed"],
        "prompt": sample["prompt"],
        "native_selection_quality": native_quality,
        "native_latency_seconds": native_latency,
        "max_quality_step": max_quality["candidate_step"],
        "max_quality_value": max_quality["selection_quality"],
        "max_quality_latency_seconds": max_quality["latency_seconds"],
        "quality_floor_step": quality_floor["candidate_step"],
        "quality_floor_value": quality_floor["selection_quality"],
        "quality_floor_latency_seconds": quality_floor["latency_seconds"],
        "quality_floor_speedup_vs_native": quality_floor["speedup_vs_native"],
        "quality_floor_fallback_to_max_quality": floor_fallback,
        "weighted_utility_step": utility["candidate_step"],
        "weighted_utility_value": utility["utility_score"],
        "weighted_utility_quality": utility["selection_quality"],
        "weighted_utility_latency_seconds": utility["latency_seconds"],
        "weighted_utility_speedup_vs_native": utility["speedup_vs_native"],
        "pareto_steps": " ".join(
            str(row["candidate_step"]) for row in rows if row["pareto_optimal"]
        ),
    }


def is_pareto_optimal(target: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    target_quality = float(target["selection_quality"])
    target_latency = float(target["latency_seconds"])
    for other in rows:
        if other is target:
            continue
        other_quality = float(other["selection_quality"])
        other_latency = float(other["latency_seconds"])
        if (
            other_quality >= target_quality
            and other_latency <= target_latency
            and (other_quality > target_quality or other_latency < target_latency)
        ):
            return False
    return True


def summarize_candidates(
    rows: list[dict[str, Any]], steps: list[int]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for step in steps:
        group = [row for row in rows if int(row["candidate_step"]) == step]
        output.append(
            {
                "candidate_step": step,
                "samples": len(group),
                "lr_evaluations": step,
                "hr_evaluations": 50 - step,
                "vbench5_mean": mean_field(group, "vbench5"),
                "vbench5_std": std_field(group, "vbench5"),
                "selection_quality_mean": mean_field(group, "selection_quality"),
                "selection_quality_std": std_field(group, "selection_quality"),
                "quality_delta_vs_native_mean": mean_field(
                    group, "quality_delta_vs_native"
                ),
                "latency_seconds_mean": mean_field(group, "latency_seconds"),
                "latency_seconds_std": std_field(group, "latency_seconds"),
                "speedup_vs_native_mean": mean_field(group, "speedup_vs_native"),
                "quality_floor_eligible_rate": mean_bool_field(
                    group, "quality_floor_eligible"
                ),
                "pareto_rate": mean_bool_field(group, "pareto_optimal"),
                "max_quality_selection_count": sum(
                    bool(row["selected_by_max_quality"]) for row in group
                ),
                "quality_floor_selection_count": sum(
                    bool(row["selected_by_quality_floor"]) for row in group
                ),
                "weighted_utility_selection_count": sum(
                    bool(row["selected_by_weighted_utility"]) for row in group
                ),
            }
        )
    return output


def case_signature(case: dict[str, Any], dimensions: list[str]) -> str:
    records = []
    for video, prompt in zip(case["videos"], case["prompts"]):
        stat = video.stat()
        records.append(
            {
                "path": str(video),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "prompt": prompt,
            }
        )
    payload = {
        "case": case["name"],
        "dimensions": dimensions,
        "records": records,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_video(path: Path, *, min_video_bytes: int) -> None:
    if not path.is_file() or path.stat().st_size < min_video_bytes:
        raise SystemExit(
            f"Missing or undersized oracle video: {path} "
            f"(minimum {min_video_bytes} bytes)"
        )


def positive_float(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid {context}: {value!r}") from exc
    if not math.isfinite(result) or result <= 0:
        raise SystemExit(f"Invalid {context}: {result}")
    return result


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def std_field(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def mean_bool_field(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(bool(row[field])) for row in rows)


def blank_if_none(value: float | None) -> float | str:
    return "" if value is None else value


def git_revision(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON {path}: {exc}") from exc


def load_json_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VBench over the TAA-free oracle branches and compile sample-level "
            "quality-efficiency timestep labels."
        )
    )
    parser.add_argument("action", choices=("check", "prepare", "run", "collect", "all"))
    parser.add_argument("--oracle-root", required=True)
    parser.add_argument("--vbench-root", default=os.environ.get("VBENCH_ROOT"))
    parser.add_argument(
        "--python",
        default=os.environ.get("VBENCH_PYTHON", sys.executable),
        help="Python executable from the isolated VBench environment",
    )
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--include-overall", action="store_true")
    parser.add_argument("--overall-weight", type=float, default=0.0)
    parser.add_argument("--max-quality-drop", type=float, default=0.02)
    parser.add_argument("--latency-lambda", type=float, default=0.05)
    parser.add_argument(
        "--timing-source",
        choices=("branch", "independent", "prefer-independent"),
        default="branch",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--strict-protocol", dest="strict_protocol", action="store_true", default=True
    )
    parser.add_argument(
        "--no-strict-protocol", dest="strict_protocol", action="store_false"
    )
    parser.add_argument("--min-video-bytes", type=int, default=MIN_VALID_VIDEO_BYTES)
    args = parser.parse_args()
    if args.ngpus < 1:
        parser.error("--ngpus must be >= 1")
    if args.min_video_bytes < 1:
        parser.error("--min-video-bytes must be >= 1")
    if args.max_quality_drop < 0:
        parser.error("--max-quality-drop must be >= 0")
    if args.latency_lambda < 0:
        parser.error("--latency-lambda must be >= 0")
    if not 0.0 <= args.overall_weight <= 1.0:
        parser.error("--overall-weight must be in [0, 1]")
    if args.overall_weight > 0 and not args.include_overall:
        parser.error("--overall-weight > 0 requires --include-overall")
    if args.cases and args.action != "run":
        parser.error("--cases is supported only with the run action")
    return args


if __name__ == "__main__":
    main()
