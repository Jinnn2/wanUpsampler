from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import UniversalAction
from .schedule import resolve_schedule
from .transition import TRANSITION_BASELINES


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALITY_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]
DIAGNOSTIC_DIMENSIONS = ["dynamic_degree"]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON {path}: {exc}") from exc


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_prompts(path: Path, *, offset: int, limit: int) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = prompts[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"Requested {limit} prompts at offset {offset}, found {len(selected)} in {path}"
        )
    return selected


def resolve_cases(spec: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    if spec.get("schema") != "univ_validation_cases_v1":
        raise ValueError("Unsupported UNIV validation case schema")
    profiles = spec.get("profiles")
    definitions = spec.get("cases")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"Unknown validation profile {profile!r}")
    if not isinstance(definitions, dict):
        raise ValueError("Validation spec must contain object-valued cases")
    names = list(profiles[profile])
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Profile {profile!r} must contain unique case names")

    cases: list[dict[str, Any]] = []
    for name in names:
        if name not in definitions:
            raise ValueError(f"Profile {profile!r} references unknown case {name!r}")
        case = {"name": name, **definitions[name]}
        if case.get("kind") == "native":
            cases.append(case)
            continue
        if case.get("kind") != "univ":
            raise ValueError(f"Case {name!r} has unsupported kind {case.get('kind')!r}")
        if case.get("transition") not in TRANSITION_BASELINES:
            raise ValueError(f"Case {name!r} has invalid transition")
        action = UniversalAction(**case["action"])
        action.validate()
        case["resolved_schedule"] = resolve_schedule(
            action,
            reference_nfe=50,
            target_latent_shape=(16, 21, 90, 156),
        ).as_dict()
        cases.append(case)
    if sum(case["kind"] == "native" for case in cases) != 1:
        raise ValueError("Each profile must contain exactly one native baseline")
    return cases


def base_runtime_config(template: dict[str, Any]) -> dict[str, Any]:
    excluded_prefixes = ("univ_", "wan_rgb_")
    return {
        key: value
        for key, value in template.items()
        if not key.startswith(excluded_prefixes)
    }


def case_runtime_config(
    template: dict[str, Any],
    case: dict[str, Any],
    *,
    realesrgan_checkpoint: Path,
    transition_diagnostics: bool,
) -> dict[str, Any]:
    config = base_runtime_config(template)
    if case["kind"] == "native":
        return config
    config.update(
        {
            "univ_action": case["action"],
            "univ_cache_mode": "residual",
            "univ_transition_baseline": case["transition"],
            "univ_enable_transition_diagnostics": transition_diagnostics,
            "univ_native_hr_state_path": "",
            "univ_native_hr_state_key": "state",
            "wan_rgb_sr_backend": "realesrgan",
            "wan_rgb_sr_checkpoint": str(realesrgan_checkpoint),
            "wan_rgb_sr_tile": 0,
            "wan_rgb_sr_tile_pad": 10,
            "wan_rgb_sr_pre_pad": 0,
            "wan_rgb_sr_half": True,
            "wan_rgb_sr_gpu_id": 0,
        }
    )
    return config


def prepare_suite(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = Path(args.case_spec).resolve()
    template_path = Path(args.template_config).resolve()
    prompts_path = Path(args.prompts).resolve()
    out_root = Path(args.out_root).resolve()
    spec = load_json(spec_path)
    template = load_json(template_path)
    cases = resolve_cases(spec, args.profile)
    prompts = load_prompts(prompts_path, offset=args.prompt_offset, limit=args.limit)
    if args.timing_warmup < 0 or args.timing_warmup >= len(prompts):
        raise ValueError("timing_warmup must be >= 0 and smaller than prompt count")

    config_dir = out_root / "configs"
    manifest_cases = []
    prepared_configs: list[tuple[Path, dict[str, Any]]] = []
    for case in cases:
        config = case_runtime_config(
            template,
            case,
            realesrgan_checkpoint=Path(args.realesrgan_checkpoint).resolve(),
            transition_diagnostics=args.transition_diagnostics,
        )
        config_path = config_dir / f"{case['name']}.json"
        prepared_configs.append((config_path, config))
        manifest_cases.append(
            {
                **case,
                "model_cls": (
                    "wan2.1_univ_native"
                    if case["kind"] == "native"
                    else "wan2.1_univ_pipeline"
                ),
                "config_path": str(config_path),
                "config_payload_sha256": canonical_sha256(config),
            }
        )

    protocol = {
        "profile": args.profile,
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompt_count": args.limit,
        "timing_warmup": args.timing_warmup,
        "transition_diagnostics": args.transition_diagnostics,
        "prompts_file": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "selected_prompts": prompts,
        "model_root": str(Path(args.model_root).resolve()),
        "template_config": str(template_path),
        "case_spec": str(spec_path),
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                REPO_ROOT / "UNIV_adaptor/validation.py",
                REPO_ROOT / "UNIV_adaptor/wan_runner.py",
                REPO_ROOT / "UNIV_adaptor/transition.py",
                REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py",
            )
        },
        "cases": manifest_cases,
        "quality_dimensions": QUALITY_DIMENSIONS,
        "diagnostic_dimensions": DIAGNOSTIC_DIMENSIONS,
    }
    manifest = {
        "schema": "univ_validation_manifest_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": canonical_sha256(protocol),
        **protocol,
    }
    manifest_path = out_root / "run_manifest.json"
    if manifest_path.is_file():
        previous = load_json(manifest_path)
        if previous.get("protocol_sha256") != manifest["protocol_sha256"]:
            raise RuntimeError(
                f"Validation protocol changed for existing output root {out_root}. "
                "Use a new OUT_ROOT to prevent mixed evidence."
            )
    for config_path, config in prepared_configs:
        write_json_atomic(config_path, config)
    write_json_atomic(manifest_path, manifest)
    print(f"Prepared {len(cases)} cases x {len(prompts)} prompts: {out_root}")
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def case_generation_complete(root: Path, manifest: dict[str, Any], case: dict[str, Any]) -> bool:
    timing_path = root / "timings" / f"{case['name']}.jsonl"
    if not timing_path.is_file():
        return False
    try:
        rows = read_jsonl(timing_path)
    except (OSError, json.JSONDecodeError):
        return False
    initialization = [row for row in rows if row.get("kind") == "initialization"]
    videos = [row for row in rows if row.get("kind") == "video"]
    if len(initialization) != 1 or len(videos) != manifest["prompt_count"]:
        return False
    if any(row.get("case") != case["name"] for row in rows):
        return False
    warmups = [row for row in videos if row.get("phase") == "warmup"]
    measured = [row for row in videos if row.get("phase") == "measured"]
    if len(warmups) != manifest["timing_warmup"] or len(measured) != (
        manifest["prompt_count"] - manifest["timing_warmup"]
    ):
        return False
    expected_indices = list(
        range(manifest["prompt_offset"], manifest["prompt_offset"] + manifest["prompt_count"])
    )
    if sorted(int(row["prompt_index"]) for row in videos) != expected_indices:
        return False
    for row in videos:
        index = int(row["prompt_index"])
        expected_seed = manifest["seed_base"] + index
        expected_output = (
            root
            / "videos"
            / case["name"]
            / f"{case['name']}_{index:02d}_seed{expected_seed}.mp4"
        ).resolve()
        if int(row["seed"]) != expected_seed or Path(row["output"]).resolve() != expected_output:
            return False
        output = Path(row["output"])
        if not output.is_file() or output.stat().st_size < 1024:
            return False
        if case["kind"] == "univ":
            sidecar = output.with_suffix(output.suffix + ".univ.json")
            if not sidecar.is_file():
                return False
            try:
                sidecar_payload = load_json(sidecar)
            except RuntimeError:
                return False
            if sidecar_payload.get("transition", {}).get("baseline") != case["transition"]:
                return False
    return True


def selected_manifest_cases(
    manifest: Mapping[str, Any], case_names: Sequence[str] | None
) -> list[dict[str, Any]]:
    cases = list(manifest["cases"])
    requested = list(case_names or [])
    if not requested:
        return cases
    if len(requested) != len(set(requested)):
        raise ValueError("--case-name values must be unique")
    by_name = {case["name"]: case for case in cases}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(
            f"Requested cases are absent from the immutable manifest: {missing}"
        )
    return [by_name[name] for name in requested]


def generate_suite(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    root = Path(args.out_root).resolve()
    batch_runner = REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["LIGHTX2V_REPO"] = str(Path(args.lightx2v_repo).resolve())
    python_roots = [
        environment["LIGHTX2V_REPO"],
        str(Path(args.realesrgan_repo).resolve()),
        str(REPO_ROOT),
    ]
    if environment.get("PYTHONPATH"):
        python_roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_roots)

    if sha256_file(Path(manifest["prompts_file"])) != manifest["prompts_sha256"]:
        raise RuntimeError("Prompts file changed after suite preparation")
    for case in selected_manifest_cases(manifest, args.case_name):
        config_payload = load_json(Path(case["config_path"]))
        if canonical_sha256(config_payload) != case["config_payload_sha256"]:
            raise RuntimeError(f"Case config changed after suite preparation: {case['config_path']}")
        if args.resume and case_generation_complete(root, manifest, case):
            print(f"[resume] generation complete: {case['name']}")
            continue
        command = [
            args.wan_python,
            str(batch_runner),
            "--seed",
            str(manifest["seed_base"]),
            "--model_cls",
            case["model_cls"],
            "--model_path",
            manifest["model_root"],
            "--config_json",
            case["config_path"],
            "--prompts_file",
            manifest["prompts_file"],
            "--out_dir",
            str(root / "videos" / case["name"]),
            "--name_prefix",
            case["name"],
            "--limit",
            str(manifest["prompt_count"]),
            "--prompt-offset",
            str(manifest["prompt_offset"]),
            "--timing-jsonl",
            str(root / "timings" / f"{case['name']}.jsonl"),
            "--timing-warmup",
            str(manifest["timing_warmup"]),
            "--target_video_length",
            "81",
            "--negative_prompt",
            args.negative_prompt,
        ]
        print(f"[generate] {case['name']}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def prompt_map_for_case(root: Path, manifest: dict[str, Any], case: dict[str, Any]) -> Path:
    mapping: dict[str, str] = {}
    for position, prompt in enumerate(manifest["selected_prompts"]):
        index = manifest["prompt_offset"] + position
        seed = manifest["seed_base"] + index
        video = root / "videos" / case["name"] / f"{case['name']}_{index:02d}_seed{seed}.mp4"
        if not video.is_file() or video.stat().st_size < 1024:
            raise RuntimeError(f"Missing or undersized VBench input: {video}")
        mapping[str(video.resolve())] = prompt
    output = root / "metrics" / "vbench_inputs" / case["name"] / "prompt_map.json"
    write_json_atomic(output, mapping)
    return output


def run_vbench(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout,
        score_case_directory,
        warmup_vbench_cache,
    )

    root = Path(args.out_root).resolve()
    vbench_root = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(vbench_root, expected_commit=args.vbench_commit or None)
    if not args.skip_vbench_warmup:
        warmup_vbench_cache(args.vbench_python, vbench_root)

    result = {
        "schema": "univ_vbench_scores_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vbench": identity,
        "quality_dimensions": QUALITY_DIMENSIONS,
        "diagnostic_dimensions": DIAGNOSTIC_DIMENSIONS,
        "prompt_count": manifest["prompt_count"],
        "cases": {},
    }
    output = root / "metrics" / "vbench_scores.json"
    for case in manifest["cases"]:
        name = case["name"]
        prompt_map = prompt_map_for_case(root, manifest, case)
        video_dir = root / "videos" / name
        case_root = root / "metrics" / "vbench_runs" / name
        quality = score_case_directory(
            vbench_root,
            args.vbench_python,
            video_dir,
            prompt_map,
            case_root / "quality5",
            QUALITY_DIMENSIONS,
            QUALITY_DIMENSIONS,
            [],
            args.vbench_ngpus,
            args.force_vbench,
            identity,
        )
        diagnostic = score_case_directory(
            vbench_root,
            args.vbench_python,
            video_dir,
            prompt_map,
            case_root / "diagnostic",
            DIAGNOSTIC_DIMENSIONS,
            [],
            DIAGNOSTIC_DIMENSIONS,
            args.vbench_ngpus,
            args.force_vbench,
            identity,
        )
        scores = {
            stem: {**quality.scores[stem], **diagnostic.scores[stem]}
            for stem in sorted(quality.scores)
        }
        aggregate = {
            dimension: statistics.mean(values[dimension] for values in scores.values())
            for dimension in [*QUALITY_DIMENSIONS, *DIAGNOSTIC_DIMENSIONS]
        }
        result["cases"][name] = {
            "aggregate": aggregate,
            "quality5_mean": statistics.mean(aggregate[key] for key in QUALITY_DIMENSIONS),
            "per_video": scores,
            "quality_provenance": quality.provenance,
            "diagnostic_provenance": diagnostic.provenance,
        }
        write_json_atomic(output, result)
        print(f"[VBench] {name}: quality5={result['cases'][name]['quality5_mean']:.6f}")
    return result


def comparison_groups(manifest: dict[str, Any]) -> dict[str, list[str]]:
    available = {case["name"] for case in manifest["cases"]}
    candidates = {
        "joint_sw060": ["native_hr50", "dvg_joint_sw060", "rgb_joint_sw060"],
        "joint_sw080": ["native_hr50", "dvg_joint_sw080", "rgb_joint_sw080"],
        "joint_sw100": ["native_hr50", "dvg_joint_sw100", "rgb_joint_sw100"],
        "dvg_axis_ablation": [
            "native_hr50",
            "dvg_spatial_sw060",
            "dvg_temporal_sw060",
            "dvg_cache_sw060",
        ],
        "rgb_axis_ablation": [
            "native_hr50",
            "rgb_spatial_sw060",
            "rgb_temporal_sw060",
            "rgb_cache_sw060",
        ],
    }
    return {
        name: cases
        for name, cases in candidates.items()
        if all(case in available for case in cases)
    }


def build_visual_comparisons(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    root = Path(args.out_root).resolve()
    groups = comparison_groups(manifest)
    if not groups:
        raise RuntimeError("Selected profile contains no visual comparison group")
    layout = {
        "schema": "univ_visual_comparison_layout_v1",
        "panel_width": 416,
        "panel_height": 240,
        "groups": groups,
        "videos": [],
    }
    for group, case_names in groups.items():
        output_dir = root / "comparisons" / group
        output_dir.mkdir(parents=True, exist_ok=True)
        for position, prompt in enumerate(manifest["selected_prompts"]):
            index = manifest["prompt_offset"] + position
            seed = manifest["seed_base"] + index
            inputs = [
                root / "videos" / case / f"{case}_{index:02d}_seed{seed}.mp4"
                for case in case_names
            ]
            missing = [str(path) for path in inputs if not path.is_file()]
            if missing:
                raise RuntimeError("Missing visual comparison inputs: " + ", ".join(missing))
            output = output_dir / f"{group}_{index:02d}_seed{seed}.mp4"
            if not (args.resume and output.is_file() and output.stat().st_size >= 1024):
                command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
                for path in inputs:
                    command.extend(["-i", str(path)])
                filters = [
                    f"[{panel}:v]scale=416:240:flags=lanczos,setsar=1[v{panel}]"
                    for panel in range(len(inputs))
                ]
                filters.append(
                    "".join(f"[v{panel}]" for panel in range(len(inputs)))
                    + f"hstack=inputs={len(inputs)}[outv]"
                )
                command.extend(
                    [
                        "-filter_complex",
                        ";".join(filters),
                        "-map",
                        "[outv]",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        str(output),
                    ]
                )
                subprocess.run(command, cwd=REPO_ROOT, check=True)
            layout["videos"].append(
                {
                    "group": group,
                    "prompt_index": index,
                    "seed": seed,
                    "prompt": prompt,
                    "panel_order": case_names,
                    "inputs": [str(path) for path in inputs],
                    "output": str(output),
                }
            )
    write_json_atomic(root / "comparisons" / "layout.json", layout)
    print(f"Visual comparisons: {root / 'comparisons'}")


def mean_or_blank(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def summarize_suite(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(args.out_root).resolve()
    vbench = load_json(root / "metrics" / "vbench_scores.json")
    case_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    timing_by_case: dict[str, list[dict[str, Any]]] = {}
    for case in manifest["cases"]:
        name = case["name"]
        timing_rows = read_jsonl(root / "timings" / f"{name}.jsonl")
        initialization = [row for row in timing_rows if row["kind"] == "initialization"]
        video_rows = [row for row in timing_rows if row["kind"] == "video"]
        warmup_rows = [row for row in video_rows if row["phase"] == "warmup"]
        measured = [row for row in video_rows if row["phase"] == "measured"]
        if len(initialization) != 1 or not measured:
            raise RuntimeError(f"Incomplete timing evidence for {name}")
        timing_by_case[name] = measured
        durations = [float(row["pipeline_elapsed_s"]) for row in measured]
        segment = [float(row["segment_elapsed_s"]) for row in measured]
        score_case = vbench["cases"][name]
        aggregate = score_case["aggregate"]
        stage_names = sorted(
            {
                key
                for row in measured
                for key in row.get("univ_stage_timing_s", {})
            }
        )
        config = load_json(Path(case["config_path"]))
        full_dit_evaluations = (
            50
            if case["kind"] == "native"
            else case["resolved_schedule"]["total_full_dit_evaluations"]
        )
        cfg_multiplier = 2 if config.get("enable_cfg", False) else 1
        initialization_s = float(initialization[0]["elapsed_s"])
        row: dict[str, Any] = {
            "case": name,
            "kind": case["kind"],
            "transition": case.get("transition", "native"),
            "measured_videos": len(measured),
            "initialization_s": initialization_s,
            "warmup_pipeline_mean_s": mean_or_blank(
                [float(item["pipeline_elapsed_s"]) for item in warmup_rows]
            ),
            "cold_start_first_video_s": (
                initialization_s + float(warmup_rows[0]["pipeline_elapsed_s"])
                if warmup_rows
                else ""
            ),
            "pipeline_mean_s": statistics.mean(durations),
            "pipeline_std_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "pipeline_median_s": statistics.median(durations),
            "segment_mean_s": statistics.mean(segment),
            "peak_allocated_gib": max(float(item.get("peak_allocated_gib", 0.0)) for item in measured),
            "full_dit_evaluations": full_dit_evaluations,
            "cfg_pass_multiplier": cfg_multiplier,
            "physical_dit_passes": full_dit_evaluations * cfg_multiplier,
            **{f"vbench_{key}": aggregate[key] for key in QUALITY_DIMENSIONS},
            "vbench_quality5_mean": score_case["quality5_mean"],
            "vbench_dynamic_degree": aggregate["dynamic_degree"],
        }
        for stage in stage_names:
            row[f"stage_{stage}_mean_s"] = mean_or_blank(
                [
                    float(item["univ_stage_timing_s"][stage])
                    for item in measured
                    if stage in item.get("univ_stage_timing_s", {})
                ]
            )
        case_rows.append(row)

        scores = score_case["per_video"]
        for timing in video_rows:
            stem = Path(timing["output"]).stem
            if stem not in scores:
                raise RuntimeError(f"Missing VBench score for timed video {stem}")
            per_video_rows.append(
                {
                    "case": name,
                    "phase": timing["phase"],
                    "prompt_index": timing["prompt_index"],
                    "seed": timing["seed"],
                    "pipeline_elapsed_s": timing["pipeline_elapsed_s"],
                    "segment_elapsed_s": timing["segment_elapsed_s"],
                    "peak_allocated_gib": timing.get("peak_allocated_gib", ""),
                    "video": timing["output"],
                    **{f"vbench_{key}": scores[stem][key] for key in scores[stem]},
                }
            )

    native = next(row for row in case_rows if row["kind"] == "native")
    native_time = float(native["pipeline_mean_s"])
    native_quality = float(native["vbench_quality5_mean"])
    native_timing = {
        int(item["prompt_index"]): float(item["pipeline_elapsed_s"])
        for item in timing_by_case[native["case"]]
    }
    for row in case_rows:
        elapsed = float(row["pipeline_mean_s"])
        quality = float(row["vbench_quality5_mean"])
        candidate_timing = {
            int(item["prompt_index"]): float(item["pipeline_elapsed_s"])
            for item in timing_by_case[row["case"]]
        }
        common = sorted(set(native_timing) & set(candidate_timing))
        if not common:
            raise RuntimeError(f"No paired timing prompts for {row['case']}")
        row["speedup_vs_native"] = native_time / elapsed
        row["paired_speedup_mean"] = statistics.mean(
            native_timing[index] / candidate_timing[index] for index in common
        )
        row["paired_timing_samples"] = len(common)
        row["latency_reduction_vs_native_pct"] = 100.0 * (1.0 - elapsed / native_time)
        row["quality5_delta_vs_native"] = quality - native_quality
        row["quality5_retention_pct"] = 100.0 * quality / native_quality

    reports = root / "reports"
    paired_vbench = paired_vbench_rows(manifest, vbench, native["case"])
    write_csv_atomic(reports / "summary.csv", case_rows)
    write_csv_atomic(reports / "per_video.csv", per_video_rows)
    write_csv_atomic(reports / "paired_vbench_vs_native.csv", paired_vbench)
    write_json_atomic(
        reports / "summary.json",
        {
            "schema": "univ_validation_summary_v1",
            "manifest": str(root / "run_manifest.json"),
            "vbench": str(root / "metrics" / "vbench_scores.json"),
            "timing_measurement": "warm_model_full_pipeline_wall_time",
            "native_case": native["case"],
            "cases": case_rows,
        },
    )
    (reports / "SUMMARY.md").write_text(render_markdown(case_rows), encoding="utf-8")
    print(f"Validation summary: {reports / 'summary.csv'}")
    return case_rows


def paired_vbench_rows(
    manifest: dict[str, Any],
    vbench: dict[str, Any],
    native_case: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    native_scores = vbench["cases"][native_case]["per_video"]
    for case in manifest["cases"]:
        name = case["name"]
        if name == native_case:
            continue
        candidate_scores = vbench["cases"][name]["per_video"]
        for dimension in QUALITY_DIMENSIONS:
            deltas: list[float] = []
            for position in range(manifest["prompt_count"]):
                index = manifest["prompt_offset"] + position
                seed = manifest["seed_base"] + index
                native_stem = f"{native_case}_{index:02d}_seed{seed}"
                candidate_stem = f"{name}_{index:02d}_seed{seed}"
                deltas.append(
                    float(candidate_scores[candidate_stem][dimension])
                    - float(native_scores[native_stem][dimension])
                )
            rows.append(
                {
                    "case": name,
                    "native_case": native_case,
                    "dimension": dimension,
                    "samples": len(deltas),
                    "delta_candidate_minus_native_mean": statistics.mean(deltas),
                    "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "candidate_win_rate": sum(delta > 0.0 for delta in deltas) / len(deltas),
                    "ties": sum(delta == 0.0 for delta in deltas),
                }
            )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# UNIV Validation Summary",
        "",
        "Timing is warm-model, synchronized, full-pipeline wall time. Initialization is reported separately.",
        "",
        "| Case | Transition | Full DiT | Mean s | Speedup | Paired speedup | VBench-5 | Delta vs native | Dynamic degree | Peak GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['transition']} | {row['full_dit_evaluations']} | "
            f"{float(row['pipeline_mean_s']):.3f} | {float(row['speedup_vs_native']):.3f}x | "
            f"{float(row['paired_speedup_mean']):.3f}x | "
            f"{float(row['vbench_quality5_mean']):.6f} | "
            f"{float(row['quality5_delta_vs_native']):+.6f} | "
            f"{float(row['vbench_dynamic_degree']):.6f} | "
            f"{float(row['peak_allocated_gib']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score the UNIV validation suite.")
    parser.add_argument(
        "action",
        choices=["prepare", "generate", "visualize", "vbench", "summarize", "all"],
    )
    parser.add_argument("--profile", choices=["smoke", "core", "full"], default="core")
    parser.add_argument(
        "--case-name",
        action="append",
        default=[],
        help="Generate only this immutable-manifest case; repeat for multiple cases.",
    )
    parser.add_argument(
        "--case-spec",
        default=str(REPO_ROOT / "UNIV_adaptor/configs/univ_validation_cases.json"),
    )
    parser.add_argument(
        "--template-config",
        default=str(REPO_ROOT / "UNIV_adaptor/configs/wan21_t2v_univ_rgb_720p.example.json"),
    )
    parser.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"),
    )
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs/univ_validation_core"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--timing-warmup", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-root", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--lightx2v-repo", default="/mnt/afs_2/houze/LightX2V")
    parser.add_argument("--realesrgan-repo", default="/mnt/afs_2/houze/Real-ESRGAN")
    parser.add_argument(
        "--realesrgan-checkpoint",
        default="/mnt/afs_2/houze/Real-ESRGAN/weights/RealESRGAN_x2plus.pth",
    )
    parser.add_argument("--wan-python", default="/opt/conda/bin/python")
    parser.add_argument("--vbench-root", default="/mnt/afs_2/houze/VBench")
    parser.add_argument("--vbench-python", default="/opt/conda/envs/vbench/bin/python")
    parser.add_argument("--vbench-ngpus", type=int, default=1)
    parser.add_argument("--vbench-commit", default="")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-vbench", action="store_true")
    parser.add_argument("--skip-vbench-warmup", action="store_true")
    parser.add_argument(
        "--transition-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable costly transition FFT/state diagnostics during generation.",
    )
    args = parser.parse_args(argv)
    if args.limit < 2 or args.prompt_offset < 0 or args.gpu < 0 or args.vbench_ngpus < 1:
        parser.error("limit must be >=2; prompt_offset/gpu >=0; vbench_ngpus >=1")
    if args.case_name and args.action != "generate":
        parser.error("--case-name is supported only with the generate action")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action in {"prepare", "all"}:
        manifest = prepare_suite(args)
    else:
        manifest = load_json(Path(args.out_root).resolve() / "run_manifest.json")
    if args.action in {"generate", "all"}:
        generate_suite(args, manifest)
    if args.action in {"visualize", "all"}:
        build_visual_comparisons(args, manifest)
    if args.action in {"vbench", "all"}:
        run_vbench(args, manifest)
    if args.action in {"summarize", "all"}:
        summarize_suite(args, manifest)


if __name__ == "__main__":
    main()
