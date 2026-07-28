from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
WAN50_BATCH_RUNNER = (
    REPO_ROOT
    / "changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_batch_infer.py"
)
DISTILL4_BATCH_RUNNER = (
    REPO_ROOT
    / "changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_batch_infer.py"
)
MEASUREMENT = "warm_model_single_video_end_to_end"
DEFAULT_CASE_ORDER = (
    "full_hr50",
    "lightx2v_cr40",
    "talh40",
    "lightx2v_cr45",
    "ralu_quality",
    "talh45",
    "lightx2v_cr48",
    "full_lr50_stage2_0hr",
    "full_lr50_stage2_1hr",
    "full_lr50_stage2_2hr",
    "full_lr50_stage2_5hr",
)
IMPLEMENTATION_FILES = (
    WAN50_BATCH_RUNNER,
    DISTILL4_BATCH_RUNNER,
    REPO_ROOT / "changing_resolution/lightx2v_clean_bridge.py",
    REPO_ROOT / "changing_resolution_distill/lightx2v_distill_bridge.py",
    REPO_ROOT / "changing_resolution_distill/rgb_super_resolution.py",
    REPO_ROOT / "changing_resolution_distill/realesrgan_compat.py",
    REPO_ROOT / "changing_resolution_distill/runtime_weights.py",
    REPO_ROOT / "changing_resolution/ralu_nt_math.py",
    REPO_ROOT / "changing_resolution/ralu_wan_state.py",
    REPO_ROOT / "changing_resolution/ralu_wan_quality.py",
    REPO_ROOT / "changing_resolution/dynamic_lora.py",
)
PAIR_FIELDS = (
    "comparison",
    "left_case",
    "right_case",
    "left_display_name",
    "right_display_name",
    "paired_repeats",
    "pipeline_delta_mean_s",
    "pipeline_delta_ci95_low_s",
    "pipeline_delta_ci95_high_s",
    "pipeline_delta_pct_of_left",
    "denoise_delta_mean_s",
    "denoise_delta_ci95_low_s",
    "denoise_delta_ci95_high_s",
    "denoise_delta_pct_of_left",
)


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else suite_root / "warm_quality_efficiency"
    )
    source_manifest = load_json(suite_root / "run_manifest.json")
    source_spec = load_json(suite_root / "benchmark_spec.json")
    batch_runner = batch_runner_for_manifest(source_manifest)
    cases = select_cases(source_manifest, args.cases)
    spec_by_case = index_cases(source_spec.get("cases", []), "benchmark spec")
    validate_cases(suite_root, cases, spec_by_case)

    settings = source_manifest.get("settings", {})
    model_root = Path(args.model_root or settings.get("model_root", "")).resolve()
    prompts = Path(args.prompts or settings.get("prompts_file", "")).resolve()
    require_path(model_root, "model root")
    require_path(prompts, "prompts file")
    num_frames = args.num_frames or int(settings.get("num_frames", 81))
    validate_prompt_count(prompts, args.prompt_offset, args.warmup + args.repeats)

    protocol = build_protocol(
        suite_root=suite_root,
        cases=cases,
        args=args,
        model_root=model_root,
        prompts=prompts,
        num_frames=num_frames,
        batch_runner=batch_runner,
    )
    signature = protocol_signature(protocol)
    protocol["run_signature"] = signature
    prepare_output_root(output_root, protocol, resume=args.resume)
    copy_case_configs(suite_root, output_root, cases)

    print(f"Physical GPU       : {args.gpu}")
    print(f"Cases              : {len(cases)}")
    print(f"Videos per case    : {args.warmup + args.repeats}")
    print(f"Total timed videos : {len(cases) * (args.warmup + args.repeats)}")
    print(f"Output root        : {output_root}")
    if args.dry_run:
        print_commands(
            cases,
            output_root,
            model_root,
            prompts,
            num_frames,
            args,
            batch_runner,
        )
        return

    for case in cases:
        raw_path = output_root / "raw" / f"{case['name']}.jsonl"
        resource_path = output_root / "resources" / f"{case['name']}.json"
        if args.resume and valid_case_result(
            raw_path,
            resource_path,
            case,
            args.warmup,
            args.repeats,
            args.seed,
            args.prompt_offset,
        ):
            print(f"[resume] {case['name']}: completed case retained", flush=True)
            continue
        command = build_command(
            case,
            output_root,
            model_root,
            prompts,
            num_frames,
            args,
            batch_runner,
        )
        print(
            f"[run] {case['name']}: one initialization + "
            f"{args.warmup} warm-up + {args.repeats} measured",
            flush=True,
        )
        print_command(command)
        if not args.allow_busy_gpu:
            ensure_gpu_idle(args.gpu)
        resource = run_and_monitor(command, args.gpu, inference_environment(args.gpu))
        write_json_atomic(resource_path, resource)
        if not valid_case_result(
            raw_path,
            resource_path,
            case,
            args.warmup,
            args.repeats,
            args.seed,
            args.prompt_offset,
        ):
            raise RuntimeError(
                f"{case['name']}: completed process produced invalid timing rows"
            )

    summary_rows, raw_rows = summarize_all(
        cases,
        spec_by_case,
        output_root,
        args.warmup,
        args.repeats,
    )
    summary_path = output_root / "quality_efficiency_warm.csv"
    raw_path = output_root / "quality_efficiency_warm_raw.csv"
    pairs_path = output_root / "quality_efficiency_warm_pairs.csv"
    write_csv_atomic(summary_path, summary_rows)
    write_csv_atomic(raw_path, raw_rows)
    pair_rows = summarize_pairs(
        source_manifest.get("analysis_pairs", []), raw_rows, summary_rows
    )
    # A selected single-case rerun has no complete registered comparison.
    # Keep a schema-only pair table so the timing artifact can still be
    # validated, resumed, and merged into the full-suite results later.
    write_csv_atomic(pairs_path, pair_rows, fieldnames=PAIR_FIELDS)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": MEASUREMENT,
        "quality_outputs_reused": True,
        "settings": protocol,
        "outputs": {
            "summary": fingerprint(summary_path),
            "raw": fingerprint(raw_path),
            "pairs": fingerprint(pairs_path),
        },
    }
    manifest_path = output_root / "warm_timing_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print_summary(summary_rows)
    print(f"Summary       : {summary_path}")
    print(f"Raw timings   : {raw_path}")
    print(f"Paired deltas : {pairs_path}")
    print(f"Audit manifest: {manifest_path}")


def build_protocol(
    *,
    suite_root: Path,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    model_root: Path,
    prompts: Path,
    num_frames: int,
    batch_runner: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "measurement": MEASUREMENT,
        "timing_boundary": {
            "excluded": "one-time runner construction, model/checkpoint loading, and init_modules",
            "pipeline": "CUDA-synchronized runner.run_pipeline, including all per-video handoff, VAE, and output work",
            "denoise": "CUDA-synchronized sum of runner.run_segment calls",
            "initialization_reported_separately": True,
        },
        "gpu_physical": args.gpu,
        "warmup_videos": args.warmup,
        "measured_videos": args.repeats,
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "num_frames": num_frames,
        "model_root": str(model_root),
        "prompts_file": str(prompts),
        "negative_prompt": args.negative_prompt,
        "batch_runner": str(batch_runner),
        "cases": [case["name"] for case in cases],
        "source_suite": str(suite_root),
        "source_manifest_sha256": sha256_file(suite_root / "run_manifest.json"),
        "source_spec_sha256": sha256_file(suite_root / "benchmark_spec.json"),
        "config_sha256": {
            case["name"]: sha256_file(suite_root / "configs" / f"{case['name']}.json")
            for case in cases
        },
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in IMPLEMENTATION_FILES
        },
    }


def protocol_signature(protocol: dict[str, Any]) -> str:
    payload = dict(protocol)
    payload.pop("run_signature", None)
    return sha256_bytes(canonical_json(payload))


def prepare_output_root(
    output_root: Path,
    protocol: dict[str, Any],
    *,
    resume: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "raw", "resources", "videos"):
        (output_root / name).mkdir(exist_ok=True)
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file():
        existing = load_json(protocol_path)
        if existing.get("run_signature") != protocol.get("run_signature"):
            raise SystemExit(
                f"Existing protocol is incompatible with this run: {protocol_path}\n"
                "Use a new --output-root."
            )
        if not resume and any((output_root / "raw").glob("*.jsonl")):
            raise SystemExit(
                f"Timing rows already exist under {output_root}. "
                "Pass --resume or use a new --output-root."
            )
    write_json_atomic(protocol_path, protocol)


def copy_case_configs(
    suite_root: Path,
    output_root: Path,
    cases: list[dict[str, Any]],
) -> None:
    for case in cases:
        source = suite_root / "configs" / f"{case['name']}.json"
        destination = output_root / "configs" / source.name
        content = source.read_bytes()
        if destination.is_file() and destination.read_bytes() != content:
            raise SystemExit(f"Copied config changed inside output root: {destination}")
        destination.write_bytes(content)


def select_cases(
    manifest: dict[str, Any],
    requested: list[str] | None,
) -> list[dict[str, Any]]:
    cases = list(manifest.get("cases", []))
    if not cases:
        raise SystemExit("run_manifest.json contains no cases")
    by_name = index_cases(cases, "run manifest")
    if requested:
        names = requested
    else:
        names = [name for name in DEFAULT_CASE_ORDER if name in by_name]
        names.extend(name for name in by_name if name not in names)
    unknown = set(names) - set(by_name)
    if unknown:
        raise SystemExit(f"Unknown case(s): {sorted(unknown)}")
    if len(names) != len(set(names)):
        raise SystemExit("--cases contains duplicates")
    return [by_name[name] for name in names]


def index_cases(cases: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        name = str(case.get("name", ""))
        if not name or name in result:
            raise SystemExit(f"{label} has an empty or duplicate case name: {name!r}")
        result[name] = case
    return result


def validate_cases(
    suite_root: Path,
    cases: list[dict[str, Any]],
    spec_by_case: dict[str, dict[str, Any]],
) -> None:
    for case in cases:
        name = case["name"]
        if name not in spec_by_case:
            raise SystemExit(f"benchmark_spec.json is missing case {name}")
        config_path = suite_root / "configs" / f"{name}.json"
        require_path(config_path, f"config for {name}")
        config = load_json(config_path)
        if name in {"talh40", "talh45", "talh3", "taa_interp3"}:
            expected_step = (
                3 if name == "taa_interp3" else int(name.removeprefix("talh"))
            )
            active_steps = [int(value) for value in config.get("lora_active_steps", [])]
            lora = config.get("lora_configs") or []
            if active_steps != [expected_step]:
                raise SystemExit(
                    f"{name}: expected lora_active_steps=[{expected_step}], got {active_steps}"
                )
            if len(lora) != 1 or float(lora[0].get("strength", -1)) != 0.75:
                raise SystemExit(
                    f"{name}: expected exactly one LoRA with strength=0.75"
                )
        if name.startswith("endpoint_") and name.endswith("hr"):
            resizer, refinement_token = name.removeprefix("endpoint_").rsplit("_", 1)
            expected_refinements = int(refinement_token.removesuffix("hr"))
            changing_steps = [
                int(value) for value in config.get("changing_resolution_steps", [])
            ]
            if changing_steps != [4]:
                raise SystemExit(
                    f"{name}: expected changing_resolution_steps=[4], got {changing_steps}"
                )
            if int(config.get("wan_final_refine_steps", -1)) != expected_refinements:
                raise SystemExit(
                    f"{name}: expected wan_final_refine_steps={expected_refinements}"
                )
            if resizer == "stage2" and not config.get("wan_clean_resizer_ckpt"):
                raise SystemExit(f"{name}: missing wan_clean_resizer_ckpt")
            if resizer == "rgb" and config.get("wan_rgb_sr_backend") not in {
                "realesrgan",
                "bicubic",
            }:
                raise SystemExit(f"{name}: unsupported wan_rgb_sr_backend")
            if (
                resizer == "rgb"
                and expected_refinements == 1
                and float(config.get("wan_final_refine_sigma", -1.0)) != 0.12
            ):
                raise SystemExit(
                    f"{name}: expected MRFlow-style wan_final_refine_sigma=0.12"
                )


def validate_prompt_count(path: Path, offset: int, count: int) -> None:
    prompts = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if offset < 0 or len(prompts) < offset + count:
        raise SystemExit(
            f"Need {offset + count} usable prompts for offset + warm-up + repeats, "
            f"found {len(prompts)} in {path}"
        )


def build_command(
    case: dict[str, Any],
    output_root: Path,
    model_root: Path,
    prompts: Path,
    num_frames: int,
    args: argparse.Namespace,
    batch_runner: Path,
) -> list[str]:
    name = case["name"]
    return [
        args.python,
        str(batch_runner),
        "--seed",
        str(args.seed),
        "--increment_seed",
        "--model_cls",
        str(case["model_cls"]),
        "--task",
        "t2v",
        "--model_path",
        str(model_root),
        "--config_json",
        str(output_root / "configs" / f"{name}.json"),
        "--prompts_file",
        str(prompts),
        "--out_dir",
        str(output_root / "videos" / name),
        "--name_prefix",
        name,
        "--prompt-offset",
        str(args.prompt_offset),
        "--limit",
        str(args.warmup + args.repeats),
        "--target_video_length",
        str(num_frames),
        "--negative_prompt",
        args.negative_prompt,
        "--timing-jsonl",
        str(output_root / "raw" / f"{name}.jsonl"),
        "--timing-warmup",
        str(args.warmup),
    ]


def print_commands(
    cases: list[dict[str, Any]],
    output_root: Path,
    model_root: Path,
    prompts: Path,
    num_frames: int,
    args: argparse.Namespace,
    batch_runner: Path,
) -> None:
    for case in cases:
        print_command(
            build_command(
                case,
                output_root,
                model_root,
                prompts,
                num_frames,
                args,
                batch_runner,
            )
        )


def run_and_monitor(
    command: list[str],
    gpu: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    baseline = gpu_memory_mib(gpu)
    peak = baseline
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
    while process.poll() is None:
        peak = max(peak, gpu_memory_mib(gpu))
        time.sleep(0.1)
    wall = time.perf_counter() - started
    peak = max(peak, gpu_memory_mib(gpu))
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return {
        "physical_gpu": gpu,
        "process_wall_s": wall,
        "baseline_memory_mib": baseline,
        "peak_memory_mib": peak,
        "peak_memory_delta_gib": max(0.0, peak - baseline) / 1024.0,
    }


def valid_case_result(
    raw_path: Path,
    resource_path: Path,
    case: dict[str, Any],
    warmup: int,
    repeats: int,
    seed_base: int,
    prompt_offset: int,
) -> bool:
    if not raw_path.is_file() or not resource_path.is_file():
        return False
    try:
        rows = read_jsonl(raw_path)
        load_json(resource_path)
        validate_raw_rows(
            rows,
            case,
            warmup,
            repeats,
            seed_base,
            prompt_offset,
        )
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def validate_raw_rows(
    rows: list[dict[str, Any]],
    case: dict[str, Any],
    warmup: int,
    repeats: int,
    seed_base: int,
    prompt_offset: int,
) -> None:
    initialization = [row for row in rows if row.get("kind") == "initialization"]
    videos = [row for row in rows if row.get("kind") == "video"]
    warmups = [row for row in videos if row.get("phase") == "warmup"]
    measured = [row for row in videos if row.get("phase") == "measured"]
    if len(initialization) != 1 or len(warmups) != warmup or len(measured) != repeats:
        raise RuntimeError(
            f"{case['name']}: expected 1/{warmup}/{repeats} init/warmup/measured rows, "
            f"got {len(initialization)}/{len(warmups)}/{len(measured)}"
        )
    expected_model = str(case["model_cls"])
    if any(str(row.get("model_cls")) != expected_model for row in rows):
        raise RuntimeError(f"{case['name']}: raw timing model_cls mismatch")
    ordered = sorted(videos, key=lambda row: int(row["prompt_index"]))
    expected_prompts = list(range(prompt_offset, prompt_offset + warmup + repeats))
    if [int(row["prompt_index"]) for row in ordered] != expected_prompts:
        raise RuntimeError(f"{case['name']}: raw timing prompt indices mismatch")
    expected_seeds = [seed_base + index for index in expected_prompts]
    if [int(row["seed"]) for row in ordered] != expected_seeds:
        raise RuntimeError(f"{case['name']}: raw timing seeds mismatch")
    measured_repeats = sorted(int(row["repeat"]) for row in measured)
    if measured_repeats != list(range(repeats)):
        raise RuntimeError(f"{case['name']}: measured repeat indices mismatch")


def summarize_all(
    cases: list[dict[str, Any]],
    spec_by_case: dict[str, dict[str, Any]],
    output_root: Path,
    warmup: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    for case in cases:
        name = case["name"]
        rows = read_jsonl(output_root / "raw" / f"{name}.jsonl")
        resource = load_json(output_root / "resources" / f"{name}.json")
        measured = sorted(
            [
                row
                for row in rows
                if row.get("kind") == "video" and row.get("phase") == "measured"
            ],
            key=lambda row: int(row["repeat"]),
        )
        if len(measured) != repeats:
            raise RuntimeError(f"{name}: expected {repeats} measured rows")
        pipeline = [float(row["pipeline_elapsed_s"]) for row in measured]
        denoise = [float(row["denoise_elapsed_s"]) for row in measured]
        non_denoise = [total - core for total, core in zip(pipeline, denoise)]
        initialization = next(
            float(row["elapsed_s"])
            for row in rows
            if row.get("kind") == "initialization"
        )
        spec = spec_by_case[name]
        protocol = spec.get("protocol", {})
        summaries.append(
            {
                "family": spec.get("family", "wan50"),
                "case": name,
                "display_name": display_name(case),
                "method": case.get("method", protocol.get("method", "NA")),
                "measurement": MEASUREMENT,
                "lr_evaluations": case.get("lr_evaluations", "NA"),
                "mixed_evaluations": case.get("mixed_evaluations", 0),
                "hr_evaluations": case.get("hr_evaluations", "NA"),
                "total_evaluations": case.get("total_evaluations", "NA"),
                "handoff_step": none_as_empty(case.get("handoff_step")),
                "refinement_steps": none_as_empty(case.get("refinement_steps")),
                "reschedule_mode": case.get("reschedule_mode", "canonical"),
                "gpu": resource["physical_gpu"],
                "warmup": warmup,
                "repeats": repeats,
                "initialization_s": initialization,
                "pipeline_mean_s": statistics.mean(pipeline),
                "pipeline_std_s": sample_std(pipeline),
                "pipeline_median_s": statistics.median(pipeline),
                "denoise_mean_s": statistics.mean(denoise),
                "denoise_std_s": sample_std(denoise),
                "non_denoise_mean_s": statistics.mean(non_denoise),
                "init_plus_pipeline_mean_s": initialization + statistics.mean(pipeline),
                "peak_memory_gib": resource["peak_memory_delta_gib"],
                "speedup_vs_native": "",
                "latency_reduction_vs_native_pct": "",
                "quality_metric": spec.get("quality_metric", ""),
                "quality_value": spec.get("quality_value", ""),
                "quality_components": json.dumps(
                    spec.get("quality_components", {}), sort_keys=True
                ),
                "vbench_source": spec.get("vbench_source", ""),
            }
        )
        for row in rows:
            if row.get("kind") != "video":
                continue
            all_raw.append(
                {
                    "case": name,
                    "display_name": display_name(case),
                    "method": case.get("method", "NA"),
                    "phase": row["phase"],
                    "repeat": row["repeat"],
                    "prompt_index": row["prompt_index"],
                    "seed": row["seed"],
                    "pipeline_elapsed_s": row["pipeline_elapsed_s"],
                    "denoise_elapsed_s": row["denoise_elapsed_s"],
                    "non_denoise_elapsed_s": float(row["pipeline_elapsed_s"])
                    - float(row["denoise_elapsed_s"]),
                    "segment_count": row["segment_count"],
                    "output": row["output"],
                }
            )
    native = next(
        (row for row in summaries if row["case"] in {"full_hr50", "native_hr4"}),
        None,
    )
    if native is not None:
        native_time = float(native["pipeline_mean_s"])
        for row in summaries:
            elapsed = float(row["pipeline_mean_s"])
            row["speedup_vs_native"] = native_time / elapsed
            row["latency_reduction_vs_native_pct"] = 100.0 * (
                1.0 - elapsed / native_time
            )
    return summaries, all_raw


def summarize_pairs(
    pairs: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_case = {row["case"]: row for row in summary_rows}
    raw_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row["phase"] == "measured":
            raw_by_case.setdefault(row["case"], []).append(row)
    results: list[dict[str, Any]] = []
    for pair in pairs:
        left_name = pair["left_case"]
        right_name = pair["right_case"]
        if left_name not in summary_by_case or right_name not in summary_by_case:
            continue
        left = sorted(raw_by_case[left_name], key=lambda row: int(row["repeat"]))
        right = sorted(raw_by_case[right_name], key=lambda row: int(row["repeat"]))
        if len(left) != len(right):
            raise RuntimeError(f"Pair length mismatch: {left_name} vs {right_name}")
        for left_row, right_row in zip(left, right):
            key_left = (
                left_row["repeat"],
                left_row["prompt_index"],
                left_row["seed"],
            )
            key_right = (
                right_row["repeat"],
                right_row["prompt_index"],
                right_row["seed"],
            )
            if key_left != key_right:
                raise RuntimeError(f"Pairing mismatch: {left_name} vs {right_name}")
        pipeline_delta = [
            float(right_row["pipeline_elapsed_s"])
            - float(left_row["pipeline_elapsed_s"])
            for left_row, right_row in zip(left, right)
        ]
        denoise_delta = [
            float(right_row["denoise_elapsed_s"]) - float(left_row["denoise_elapsed_s"])
            for left_row, right_row in zip(left, right)
        ]
        pipeline_mean, pipeline_low, pipeline_high = mean_ci95(pipeline_delta)
        denoise_mean, denoise_low, denoise_high = mean_ci95(denoise_delta)
        left_pipeline = float(summary_by_case[left_name]["pipeline_mean_s"])
        left_denoise = float(summary_by_case[left_name]["denoise_mean_s"])
        results.append(
            {
                "comparison": pair.get("comparison", f"{left_name}_vs_{right_name}"),
                "left_case": left_name,
                "right_case": right_name,
                "left_display_name": summary_by_case[left_name]["display_name"],
                "right_display_name": summary_by_case[right_name]["display_name"],
                "paired_repeats": len(left),
                "pipeline_delta_mean_s": pipeline_mean,
                "pipeline_delta_ci95_low_s": pipeline_low,
                "pipeline_delta_ci95_high_s": pipeline_high,
                "pipeline_delta_pct_of_left": 100.0 * pipeline_mean / left_pipeline,
                "denoise_delta_mean_s": denoise_mean,
                "denoise_delta_ci95_low_s": denoise_low,
                "denoise_delta_ci95_high_s": denoise_high,
                "denoise_delta_pct_of_left": 100.0 * denoise_mean / left_denoise,
            }
        )
    return results


def display_name(case: dict[str, Any]) -> str:
    name = str(case["name"])
    if name == "full_hr50":
        return "Native-HR"
    if name == "native_hr4":
        return "Native-HR4"
    if name == "talh3":
        return "TrajScale-D4@3"
    if name == "cll3":
        return "CLL-D4@3"
    if name == "taa_interp3":
        return "TAA+Interp-D4@3"
    if name in {"interp2", "interp3"}:
        return f"Interp-D4@{name.removeprefix('interp')}"
    if name.startswith("endpoint_") and name.endswith("hr"):
        _, resizer, steps = name.split("_", 2)
        return f"Endpoint-{resizer.upper()}-{steps.removesuffix('hr')}HR"
    if name.startswith("talh"):
        return f"TrajScale-{name.removeprefix('talh')}"
    if name.startswith("lightx2v_cr"):
        return f"LightX2V-{name.removeprefix('lightx2v_cr')}"
    if name == "ralu_quality":
        return "RALU-Quality"
    if name.startswith("full_lr50_stage2_") and name.endswith("hr"):
        steps = name.removeprefix("full_lr50_stage2_").removesuffix("hr")
        return f"Endpoint-{steps}HR"
    return name


def batch_runner_for_manifest(manifest: dict[str, Any]) -> Path:
    family = str(manifest.get("family", ""))
    if family == "distill4_quality_efficiency":
        return DISTILL4_BATCH_RUNNER
    if family in {"wan50_quality_efficiency", "wan50"}:
        return WAN50_BATCH_RUNNER
    raise SystemExit(f"Unsupported warm-benchmark suite family: {family!r}")


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    t_critical = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    df = len(values) - 1
    critical = t_critical[df - 1] if df <= len(t_critical) else 1.96
    margin = critical * statistics.stdev(values) / (len(values) ** 0.5)
    return mean, mean - margin, mean + margin


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def none_as_empty(value: Any) -> Any:
    return "" if value is None else value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require_path(path, path.name)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict[str, Any]:
    require_path(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def require_path(path: Path, label: str) -> None:
    if not str(path) or not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    resolved_fields = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in resolved_fields:
                resolved_fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory_mib(gpu: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip().splitlines()[0])


def ensure_gpu_idle(gpu: int) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    processes = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "No running processes" not in line
    ]
    if processes:
        raise SystemExit(
            f"GPU {gpu} has active compute processes:\n  "
            + "\n  ".join(processes)
            + "\nWait for it to become idle or pass --allow-busy-gpu explicitly."
        )


def inference_environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("LIGHTX2V_REPO", "/path/to/LightX2V")
    environment.setdefault("DIFFSYNTH_REPO", "/path/to/DiffSynth-Studio")
    roots = [
        environment["LIGHTX2V_REPO"],
        environment["DIFFSYNTH_REPO"],
        str(REPO_ROOT),
    ]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def print_command(command: list[str]) -> None:
    import shlex

    print("  " + shlex.join(command), flush=True)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("Warm-model quality-efficiency timing:")
    for row in rows:
        speedup = row["speedup_vs_native"]
        speedup_text = f"{float(speedup):.3f}x" if speedup != "" else "NA"
        print(
            f"  {row['display_name']}: pipeline="
            f"{float(row['pipeline_mean_s']):.3f}±{float(row['pipeline_std_s']):.3f}s "
            f"denoise={float(row['denoise_mean_s']):.3f}s "
            f"init={float(row['initialization_s']):.3f}s "
            f"speedup={speedup_text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Wan50 or Distill4 Pareto suite with one persistent model "
            "initialization per case and per-video warm timing."
        )
    )
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu", type=int, required=True, help="Physical GPU exposed to each case"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >= 1 and --repeats must be >= 2")
    if args.prompt_offset < 0 or args.num_frames < 0:
        parser.error("--prompt-offset and --num-frames must be non-negative")
    return args


if __name__ == "__main__":
    main()
